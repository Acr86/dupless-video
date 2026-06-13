"""UI data layer: builds clusters for the tree, sortable, with reclaimable GB and
the KEEP flagged. Pure (no Qt) -> testable. Reads the existing DB, does not re-scan.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import numpy as np

from dupdetect.features.audio_fp import AUDIO_COV_TOL, AUDIO_OK_COVERAGE
from dupdetect.quality.color import GRADE_DIVERGENCE, ColorStats
from dupdetect.store import FingerprintStore

# Release/quality tags to strip to get a searchable title (replacement list).
_REL_TAGS = re.compile(
    r"\b(2160p|1080p|720p|480p|4k|8k|x264|x265|h\.?264|h\.?265|hevc|av1|vp9|web-?dl|webrip|"
    r"web|bluray|blu-ray|bdrip|brrip|hdrip|dvdrip|dvd|hdtv|xvid|divx|aac\d?|ac3|eac3|dts(-hd)?|"
    r"truehd|atmos|hdr10?|dolby ?vision|dv|remux|proper|repack|extended|uncut|unrated|"
    r"multi|dual|sub(s|titulado)?|castellano|latino|esp|eng)\b", re.I)


def clean_title(filename: str) -> str:
    """Filename -> readable/searchable title: no extension, no release tags, separators
    normalised. Preserves the year if present. For the 'replacement list' and web search."""
    name = os.path.splitext(os.path.basename(filename))[0]
    name = name.replace(".", " ").replace("_", " ")
    name = re.sub(r"[\[\](){}]", " ", name)            # strip release brackets/parentheses
    name = _REL_TAGS.sub(" ", name)
    name = re.sub(r"-\s*[A-Za-z0-9]+$", "", name)       # release group suffix ("-GROUP")
    name = re.sub(r"\s+", " ", name).strip(" -·")
    return name

# Verdict strength order (for sorting/representing a cluster).
_VERDICT_RANK = {"CERTAIN": 5, "VERY_HIGH": 4, "HIGH": 3, "NAME_COPY": 2, "PROBABLE": 2,
                 "DIFFERENT_EDITION": 1, "DIFFERENT": 0, "": 0}


@dataclass
class FileRow:
    path: str
    height: int = 0
    width: int = 0
    bitrate_kbps: int = 0
    size: int = 0
    vcodec: str = ""
    lang: str = ""
    is_keep: bool = False
    audio_coverage: float = 1.0           # <1 = missing/truncated audio (quality warning)
    color: ColorStats = field(default_factory=ColorStats)   # clipping + grade signals

    @property
    def name(self) -> str:
        return os.path.basename(self.path)

    @property
    def audio_bad(self) -> bool:
        return self.audio_coverage < AUDIO_OK_COVERAGE

    @property
    def audio_note(self) -> str:
        """Human-readable warning ('' if audio is complete)."""
        if not self.audio_bad:
            return ""
        if self.audio_coverage <= 0.001:
            return "no audio"
        return f"audio incomplete (~{self.audio_coverage * 100:.0f}%)"

    @property
    def res(self) -> str:
        return f"{self.height}p" if self.height else "?"


@dataclass
class ClusterRow:
    cluster_id: int
    members: list[FileRow] = field(default_factory=list)
    verdict: str = ""                       # representative verdict (strongest in the cluster)
    confidence: float = 0.0

    @property
    def keep(self) -> FileRow | None:
        return next((m for m in self.members if m.is_keep), None)

    @property
    def title(self) -> str:
        k = self.keep or (self.members[0] if self.members else None)
        return k.name if k else f"cluster {self.cluster_id}"

    @property
    def n_copies(self) -> int:
        return len(self.members)

    @property
    def reclaimable_bytes(self) -> int:
        """Σ size of non-keep members: bytes freed if discards are deleted."""
        return sum(m.size for m in self.members if not m.is_keep)

    def deletable(self) -> list[FileRow]:
        """Members that can be deleted = all EXCEPT the KEEP (safety lock)."""
        return [m for m in self.members if not m.is_keep]

    @property
    def audio_warning(self) -> bool:
        """A copy has truncated audio AND the copies DIFFER in coverage -> the auto-KEEP could drop
        the better-audio copy, so the user decides (review, no direct delete). If every copy shares
        ~the same coverage (same source, even if truncated), audio is not a differentiator -> no
        warning. Mirrors rank_cluster's guard so visibility and KEEP selection stay consistent."""
        covs = [m.audio_coverage for m in self.members]
        if not covs or not any(c < AUDIO_OK_COVERAGE for c in covs):
            return False
        return (max(covs) - min(covs)) > AUDIO_COV_TOL

    @property
    def color_warning(self) -> bool:
        """Two copies' color GRADE (cast/saturation/contrast) differ enough to flag for manual
        choice (e.g. one is color-corrected). The KEEP is still suggested by least-clipping; this
        only tells the user to verify the look. Does NOT block deletion (unlike audio)."""
        cs = [m.color for m in self.members]
        return any(cs[i].grade_distance(cs[j]) > GRADE_DIVERGENCE
                   for i in range(len(cs)) for j in range(i + 1, len(cs)))


def load_clusters(store: FingerprintStore) -> list[ClusterRow]:
    """Rebuilds clusters for the view from the DB: `clusters` ⋈ `files`, with the
    representative verdict from `matches` (strongest among members)."""
    files = {r["path"]: r for r in store.conn.execute(
        "SELECT path, height, width, bitrate_kbps, size, vcodec, lang_detected, "
        "audio_coverage, color_stats FROM files")}
    grouped: dict[int, ClusterRow] = {}
    for r in store.conn.execute("SELECT cluster_id, path, is_keep FROM clusters"):
        cid = r["cluster_id"]
        cl = grouped.setdefault(cid, ClusterRow(cluster_id=cid))
        f = files.get(r["path"])
        cl.members.append(FileRow(
            path=r["path"],
            height=(f["height"] if f else 0) or 0,
            width=(f["width"] if f else 0) or 0,
            bitrate_kbps=(f["bitrate_kbps"] if f else 0) or 0,
            size=(f["size"] if f else 0) or 0,
            vcodec=(f["vcodec"] if f else "") or "",
            lang=(f["lang_detected"] if f else "") or "",
            is_keep=bool(r["is_keep"]),
            audio_coverage=(f["audio_coverage"] if f and f["audio_coverage"] is not None else 1.0),
            color=ColorStats.from_list(
                np.frombuffer(f["color_stats"], np.float32) if f and f["color_stats"] else None),
        ))
    _annotate_verdicts(store, grouped)
    return list(grouped.values())


def _annotate_verdicts(store: FingerprintStore, grouped: dict[int, ClusterRow]) -> None:
    """Assigns each cluster the strongest verdict/confidence from matches between its members."""
    member_to_cid = {m.path: cid for cid, cl in grouped.items() for m in cl.members}
    for a, b, verdict, conf in store.conn.execute(
            "SELECT a_path, b_path, verdict, confidence FROM matches"):
        cid = member_to_cid.get(a)
        if cid is None or member_to_cid.get(b) != cid:
            continue
        cl = grouped[cid]
        if _VERDICT_RANK.get(verdict, 0) > _VERDICT_RANK.get(cl.verdict, 0):
            cl.verdict, cl.confidence = verdict, conf or 0.0


def sort_clusters(clusters: list[ClusterRow], key: str) -> list[ClusterRow]:
    """Sorts clusters. `key`: 'copies' (most copies), 'space' (most reclaimable GB),
    'confidence' (strongest verdict). Stable tiebreak by title."""
    keyfns = {
        "copies": lambda c: (-c.n_copies, -c.reclaimable_bytes),
        "space": lambda c: (-c.reclaimable_bytes, -c.n_copies),
        "confidence": lambda c: (-_VERDICT_RANK.get(c.verdict, 0), -c.confidence, -c.n_copies),
    }
    fn = keyfns.get(key, keyfns["copies"])
    return sorted(clusters, key=lambda c: (*fn(c), c.title.lower()))


# Verdicts that enable direct deletion in the UI (PROBABLE go to review, not deletion).
# NAME_COPY (name-based copies) are shown as actionable: the user manages them manually
# (KEEP protected), so they appear with the default filter.
ACTIONABLE_VERDICTS = {"CERTAIN", "VERY_HIGH", "HIGH", "NAME_COPY"}


def is_actionable(cluster: ClusterRow) -> bool:
    # An audio warning forces Review: there is NO auto-selected KEEP (user decides), so
    # the cluster must not be offered for direct deletion -> avoids losing the copy with audio.
    return cluster.verdict in ACTIONABLE_VERDICTS and not cluster.audio_warning


def cluster_tooltip(cluster: ClusterRow) -> str:
    """Plain-language hover text explaining the cluster's state — chiefly the ⚠ symbol, so a
    first-time user knows WHY a group needs review instead of guessing. Pure (no Qt) -> testable;
    model.py sets it as the row's tooltip. Lists every active reason (audio / color / low verdict)."""
    if is_actionable(cluster):
        return ("Duplicate group ready to act on. The ★ KEEP is auto-selected (best copy); "
                "tick the others to delete them.")
    lines = ["⚠ Needs manual review before deleting — no KEEP was auto-selected:"]
    if cluster.audio_warning:
        lines.append("• Audio differs between copies (one is truncated or muted). Pick which to "
                     "keep so you don't lose the copy with full audio.")
    if cluster.color_warning:
        lines.append("• Color grade differs between copies. The suggested KEEP is the least-clipped "
                     "one; verify the look before deleting.")
    if cluster.verdict not in ACTIONABLE_VERDICTS:
        lines.append(f"• Match confidence is '{cluster.verdict or 'unknown'}', not high enough to act "
                     "automatically. Confirm these are the same video before deleting.")
    return "\n".join(lines)


def drift_report(store: FingerprintStore) -> dict:
    """Detects DESYNC between `clusters` and `matches`. Clusters are a derived view
    of `matches`; if they come from different scans (e.g. an `exact_scan` rebuilds clusters by hash
    without touching matches), their paths do not appear in matches -> the verdict is empty (all fall to
    'Review only') and feedback does not carry over on recalibration. `drifted` = no shared path at all."""
    match_paths: set[str] = set()
    for a, b, _ in store.all_matches():
        match_paths.add(a); match_paths.add(b)
    cluster_paths = {r["path"] for r in store.conn.execute("SELECT path FROM clusters")}
    orphan = cluster_paths - match_paths
    return {
        "orphan_paths": len(orphan),
        "cluster_paths": len(cluster_paths),
        "drifted": bool(cluster_paths) and not (cluster_paths & match_paths),
    }
