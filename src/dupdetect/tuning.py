"""Storage-aware auto-tune for workers / decode-workers.

The VERDICT does not depend on this: workers/decode only affect SPEED, never the result ->
§0 is safe (which is why auto-adjusting them is legitimate, §1/§3).

Key (§1): MEASURE actual storage, do not trust the OS label — which lies with Storage Spaces
(reports 'Fixed' over a span of mechanical HDDs), iSCSI (looks local but is network), and
network shares. The HDD
signature is RANDOM READ LATENCY: a mechanical head costs ms; SSD/NVMe < 1 ms. Probing is
done on the ACTUAL files being scanned (the disk where they live).
"""
from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

_BLOCK = 1 << 20            # 1 MiB per probe read
_N_SEEKS = 24              # number of random reads for the median
_MIN_SIZE = 16 << 20       # only files >=16MB: random offsets escape the cache

# Concurrency-scaling probe (detects tiered HDD = mechanical + SSD cache, which a random-seek
# probe misreads as SSD: the small cold seeks hit the cache, but real large sequential decode
# reads hit the mechanical tier and THRASH under concurrency).
_SEQ_BYTES = 16 << 20      # 16 MiB COLD sequential read per stream
_CONC_STREAMS = 4          # concurrent streams: forces head contention on a mechanical disk
_SCALE_MIN = 0.6           # concurrent aggregate throughput must keep >=60% of serial to be 'SSD'


@dataclass
class AutoTune:
    workers: int
    decode_workers: int
    kind: str                 # 'hdd' | 'ssd' | 'moderate' | 'network*' | 'unknown'
    seek_ms: float | None
    note: str


def _sample_files(files: list[str], limit: int = 8) -> list[str]:
    """Returns a sample of real files (>=16MB) under the target, for probing their disk."""
    out: list[str] = []
    for p in files:
        try:
            if os.path.getsize(p) >= _MIN_SIZE:
                out.append(p)
        except OSError:
            continue
        if len(out) >= limit:
            break
    return out


def _seek_latency_ms(files: list[str], n: int = _N_SEEKS) -> float | None:
    """Median latency of ONE random 1MB read over real files. Random offsets in large files
    => mostly COLD reads (doesn't re-read the same offset => no cache bias). buffering=0
    prevents Python read-ahead. None if measurement failed."""
    rnd = random.Random(20260605)              # deterministic (reproducible probe)
    samples: list[float] = []
    for _ in range(n):
        if not files:
            break
        p = rnd.choice(files)
        try:
            sz = os.path.getsize(p)
            off = rnd.randint(0, max(0, sz - _BLOCK))
            with open(p, "rb", buffering=0) as f:
                f.seek(off)
                t0 = time.perf_counter()
                f.read(_BLOCK)
                samples.append((time.perf_counter() - t0) * 1000.0)
        except OSError:
            continue
    if not samples:
        return None
    samples.sort()
    return samples[len(samples) // 2]


def _safe_size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def _seq_read_ms(path: str, off: int) -> float | None:
    """Time ONE cold sequential read of `_SEQ_BYTES` from `off` (buffering=0). None on failure."""
    try:
        with open(path, "rb", buffering=0) as f:
            f.seek(off)
            t0 = time.perf_counter()
            got = 0
            while got < _SEQ_BYTES:
                b = f.read(_SEQ_BYTES - got)
                if not b:
                    break
                got += len(b)
            return (time.perf_counter() - t0) * 1000.0 if got >= _SEQ_BYTES else None
    except OSError:
        return None


def _concurrency_scaling(files: list[str]) -> float | None:
    """aggregate_throughput(concurrent) / throughput(serial) for COLD large sequential reads.
    ~>=1 on SSD/NVMe (concurrency holds or scales); <<1 on a mechanical/TIERED HDD (heads thrash,
    aggregate throughput collapses) — which a small random-seek probe can't see when an SSD cache
    serves the tiny reads. None if not measurable (too few large files). Reads release the GIL, so
    the threads contend on real I/O exactly like the pipeline's decode workers do."""
    from concurrent.futures import ThreadPoolExecutor

    big = [p for p in files if _safe_size(p) >= _SEQ_BYTES * 2]
    if len(big) < 2:
        return None
    rnd = random.Random(20260609)

    def _off(p: str) -> int:
        return rnd.randint(0, _safe_size(p) - _SEQ_BYTES)

    serial_ms = _seq_read_ms(big[0], _off(big[0]))     # one stream, nothing competing
    if not serial_ms or serial_ms <= 0:
        return None
    serial_tput = _SEQ_BYTES / serial_ms               # bytes per ms
    streams = [(p, _off(p)) for p in (big * _CONC_STREAMS)[:_CONC_STREAMS]]   # distinct cold reads
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(streams)) as ex:
        ok = sum(1 for r in ex.map(lambda fo: _seq_read_ms(*fo), streams) if r)
    wall_ms = (time.perf_counter() - t0) * 1000.0
    if ok == 0 or wall_ms <= 0:
        return None
    conc_tput = (_SEQ_BYTES * ok) / wall_ms            # aggregate bytes per ms under concurrency
    return conc_tput / serial_tput


def _is_network(path: str) -> bool:
    """Best-effort heuristic: does the target live on a network share (UNC / NFS / CIFS)? Used only for the note."""
    try:
        p = str(Path(path).resolve())
    except OSError:
        p = str(path)
    if os.name == "nt":
        return p.startswith("\\\\") or p.startswith("//")          # UNC path
    try:                                                            # POSIX: fstype from mount
        dev = os.stat(path).st_dev
        with open("/proc/mounts", encoding="utf-8") as fh:
            best = ("", "")
            for line in fh:
                parts = line.split()
                if len(parts) >= 3:
                    mp, fstype = parts[1], parts[2]
                    if p.startswith(mp) and len(mp) >= len(best[0]):
                        best = (mp, fstype)
        return best[1] in ("nfs", "nfs4", "cifs", "smbfs", "smb3")
    except OSError:
        return False


def _recommend(lat: float, net: bool, cpu: int,
               scaling: float | None) -> tuple[str, int, int, str]:
    """Maps the measured (latency, concurrency-scaling) signals to (kind, workers, decode, why).
    Flat guard sequence; the tiered-HDD case wins first (fast seek but concurrency collapses)."""
    if lat < 6.0 and scaling is not None and scaling < _SCALE_MIN:     # tiered HDD (SSD cache)
        return (("network-hdd" if net else "hdd-tiered"), 2, 1,
                f"random read {lat:.2f}ms looked fast (cache) BUT concurrent reads keep only "
                f"{scaling:.0%} of serial throughput — mechanical/tiered disk thrashes")
    if lat >= 6.0:                              # mechanical disk (or slow network)
        return (("network-hdd" if net else "hdd"), 2, 1,
                f"random read {lat:.1f}ms (mechanical/slow network); disk concurrency thrashes")
    if lat <= 1.5:                              # SSD/NVMe: concurrency scales (parallel decode)
        sc = f", concurrency holds ({scaling:.0%})" if scaling is not None else ""
        return "ssd", min(cpu, 12), 4, f"random read {lat:.2f}ms{sc} (SSD/NVMe); concurrency scales"
    return (("network" if net else "moderate"), min(cpu, 6), 2,    # intermediate / fast network share
            f"random read {lat:.1f}ms (intermediate/network share); moderate concurrency")


def autotune(files: list[str], cpu_count: int | None = None,
             seek_ms: float | None = None, scaling: float | None = None) -> AutoTune:
    """Recommends (workers, decode_workers) by probing the disk where `files` live.

    Two signals (§1, MEASURE — don't trust the OS label): (1) random-read latency, the mechanical
    signature; (2) when latency LOOKS fast, a concurrency-SCALING probe that catches TIERED storage
    (HDD + SSD cache) whose cache serves the tiny seeks (looks SSD) while real large reads thrash on
    the mechanical tier. `seek_ms`/`scaling` allow injecting measurements in tests."""
    cpu = cpu_count or os.cpu_count() or 4
    net = bool(files) and _is_network(files[0])
    lat = seek_ms if seek_ms is not None else _seek_latency_ms(_sample_files(files))

    if lat is None:
        return AutoTune(2, 1, "unknown", None,
                        "Auto-tune: storage could not be probed -> conservative "
                        "(workers=2, decode=1). Override: --workers N --decode-workers M.")
    if lat < 6.0 and scaling is None:           # looks fast -> CONFIRM concurrency before trusting it
        scaling = _concurrency_scaling(_sample_files(files, limit=6))

    kind, w, d, why = _recommend(lat, net, cpu, scaling)
    note = (f"Auto-tune: storage={kind} — {why} -> workers={w}, decode-workers={d}. "
            f"(The verdict does NOT change; this is speed only. Override: --workers N --decode-workers M.)")
    return AutoTune(w, d, kind, lat, note)
