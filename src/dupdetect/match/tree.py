"""The staged decision tree. Pure logic — nearly complete as-is.

Each pair descends the tree; the FIRST tier that fires fixes the verdict and
confidence. Precision comes from requiring agreement across independent signals.
"""
from __future__ import annotations

from dupdetect.config import Thresholds
from dupdetect.models import AlignResult, Record, Result, Verdict

# Reason string for the T0 tier (byte-identity by sampled hash). Exposed so the exact-only
# sweep (pipeline.fullscan.exact_scan) stamps the SAME verdict as the full tree — keeping the
# `clusters` and `matches` tables in sync (see ui.data.drift_report).
T0_REASON = "T0 sampled hash identical (verify byte-exact before deleting)"

# Version of the decision LOGIC (tiers/guards), bumped whenever decide_tree can flip a verdict for the
# SAME stored signals + thresholds — e.g. a new guard/tier. The incremental ledger (pipeline.fullscan.
# _scan_fingerprint) folds this in, so a logic change invalidates cached evaluations and a re-scan
# re-decides every pair. Thresholds already key the ledger via th.raw; this covers code-only changes
# (which th.raw cannot see). Bump on any semantic change here.
#   2 -> added the CONTAINS guard (coverage_long): clip-in-compilation no longer clusters.
DECISION_VERSION = 2


def _cut_density(rec: Record) -> float:
    """Scene cuts per second. A coarse signature (low density — SEEK-sampled giants) is not
    discriminative for the scenes-only T4 tier (see align/scenes.py and the T4 guard)."""
    dur = rec.probe.duration_s or 0.0
    n = len(rec.scene_cuts) if rec.scene_cuts is not None else 0
    return (n / dur) if dur > 0 else 0.0


def _coverage_long(a: Record, b: Record, video: AlignResult) -> float:
    """Fraction of the LONGER file that aligns: `video.coverage` measures only the SHORTER (path/min),
    so scale it by the duration ratio. Low -> the shorter is a SEGMENT inside the longer (a clip in a
    compilation), not a mutual duplicate. Unknown duration -> falls back to coverage (won't fire)."""
    da = a.probe.duration_s or 0.0
    db = b.probe.duration_s or 0.0
    mx = max(da, db)
    return video.coverage * (min(da, db) / mx) if mx > 0 else video.coverage


def _structural_verdict(a: Record, b: Record, video: AlignResult, th: Thresholds, disc: bool):
    """A relationship that is NOT a duplicate: a different EDITION (contiguous superset) or CONTAINS
    (a short clip aligned inside a long compilation). Returns (verdict, confidence, reason) or None.
    Kept out of decide_tree so the tier list stays flat. Neither verdict is in DUPLICATE_VERDICTS, so
    union-find never groups these -> the root fix for compilation chain-fusion."""
    if disc and video.score >= th.theta_v and video.contiguous_superset:
        return (Verdict.DIFFERENT_EDITION, min(0.90, video.score),
                f"different edition: contiguous superset (+{video.extra_ratio:.0%} runtime)")
    if video.score >= th.theta_v and video.coverage >= th.min_coverage:
        cov_long = _coverage_long(a, b, video)
        if cov_long < th.min_coverage_long:
            return (Verdict.CONTAINS, min(0.85, video.score),
                    f"contains: shorter aligns (cov {video.coverage:.2f}) but longer mostly unmatched "
                    f"(cov_long {cov_long:.2f})")
    return None


def _discriminative(a: Record, b: Record, th: Thresholds) -> bool:
    """Is there enough CONTENT to trust a strong 'same video' verdict (edition/T1/T2/T3)? A very short
    clip (a few seconds of a near-static scene) has non-discriminative video — any two such clips align
    at ~1.0 — so a strong tier must NOT fire on it; the pair falls through to T4b (review). Same idea as
    the min_cut_density guard on the scenes-only tier. Measured from the SHORTER runtime (probe, §0)."""
    dur = min(a.probe.duration_s or 0.0, b.probe.duration_s or 0.0)
    return dur >= th.min_strong_duration_s


def decide_tree(
    a: Record,
    b: Record,
    audio: AlignResult,
    video: AlignResult,
    scenes: AlignResult,
    th: Thresholds,
) -> Result:
    """Applies tiers in order. Returns a Result with attached evidence."""

    def make(verdict: Verdict, conf: float, reason: str) -> Result:
        return Result(
            candidate_path=b.path, verdict=verdict, confidence=conf, reason=reason,
            audio=audio, video=video, scenes=scenes,
        )

    # ---- T0: identity by sampled hash ----------------------------------
    # M1: xxhash(head|mid|tail)+size is probabilistically very safe, but NOT
    # byte-identical literally. For a DELETE action, verify byte-exact.
    if a.content_hash == b.content_hash and a.size == b.size:
        return make(Verdict.CERTAIN, 1.00, T0_REASON)

    # Enough content to trust a strong 'same video' verdict? Very short near-static clips are not
    # discriminative (phone burst clips of the same subject all align ~1.0) -> such pairs skip the
    # strong tiers below and fall to T4b (review). T0 above is byte-identity, unaffected.
    disc = _discriminative(a, b, th)

    # ---- GUARD: structural relationship (edition / contains), NOT a duplicate -----
    # A different EDITION (contiguous superset) or CONTAINS (a short clip aligned INSIDE a long
    # compilation). Both must exit BEFORE the duplicate tiers so union-find never fuses a long file's
    # unrelated neighbours through it. See _structural_verdict.
    struct = _structural_verdict(a, b, video, th, disc)
    if struct is not None:
        return make(*struct)

    # ---- T1: confirmed by TWO independent modalities -------------------
    # Coverage required: Smith-Waterman selects the most similar frames, so
    # a high score over a tiny path (cov ~0.04 between different films) is NOT
    # "strong video". (Audio is already auto-gated by min_overlap in align_audio.)
    if (disc and audio.score >= th.theta_a and video.score >= th.theta_v
            and video.coverage >= th.min_coverage):
        return make(
            Verdict.CERTAIN, 0.99,
            f"T1 audio({audio.score:.2f})+video({video.score:.2f}, cov {video.coverage:.2f}) agree",
        )

    # ---- T2: same video, audio does NOT align => different dub ----------
    # Where Plex falls short. Audio MUST NOT align (otherwise it would be T1).
    if (
        disc
        and video.score >= th.theta_v_high
        and video.coverage >= th.min_coverage
        and audio.score < th.theta_a_low
    ):
        return make(
            Verdict.VERY_HIGH, 0.95,
            f"T2 video({video.score:.2f}) identical, audio doesn't align => different dub",
        )

    # ---- T3: partial video corroborated by scene structure -------------
    if (disc and video.score >= th.theta_v and video.coverage >= th.min_coverage
            and scenes.score >= th.theta_s):
        return make(
            Verdict.HIGH, 0.88,
            f"T3 video({video.score:.2f}, cov {video.coverage:.2f}) + scenes({scenes.score:.2f})",
        )

    # ---- T4: structure only => possible cam rips -> REVIEW QUEUE -------
    # Never acts alone to delete. Two cam rips kill audio and degrade video
    # but preserve scene cuts. Recall preserved, precision intact.
    # GUARD: scenes ALONE only count with a discriminative cut signature. A coarse one
    # (few cuts over a long runtime — SEEK-sampled giants, ~1 cut/min) is not reliable:
    # unrelated dense films align spuriously (measured ~0.88). Require minimum cut density
    # on BOTH files. T1/T2/T3 corroborate with video, so they're unaffected.
    if (scenes.score >= th.theta_s_high and video.score < th.theta_v
            and _cut_density(a) >= th.min_cut_density
            and _cut_density(b) >= th.min_cut_density):
        return make(
            Verdict.PROBABLE, 0.65,
            f"T4 scenes only({scenes.score:.2f}) => possible cam rips (review)",
        )

    # ---- T4b: A4 — close the dead zone ---------------------------------
    # Strong VIDEO without corroboration -> review ("when in doubt, to the queue"). Video requires
    # coverage: otherwise a spurious short path (cov 0.04) between different films would send false
    # positives to the queue.
    #
    # LAZY AUDIO (perf, §1): the historical audio-only OR-branch (`or audio.score >= th.theta_a`)
    # was REMOVED. It was the only tier that consulted audio for a video-WEAK pair, which forced
    # extracting the whole-file audio fingerprint (the Pass-2 bottleneck: ~3.6s/file off an HDD) for
    # EVERY candidate file, even the unique movies whose faiss neighbours are weak video matches.
    # With it gone, audio can affect the verdict ONLY when `video.score >= theta_v AND coverage >=
    # min_coverage` (T1/T2), so the matcher computes the fingerprint exactly there and skips it for
    # the rest -> "most unique files never pay for it" (the original on-demand intent), restored at
    # full-library scale. Trade-off (a fork, approved): a pair with matching AUDIO but DIFFERENT
    # video no longer reaches the review queue. Physically near-empty for a VIDEO dedup (re-encodes
    # keep the video alignable; cam rips kill the audio), and it never weakened a strong tier (§0:
    # T1/T2 zero-FP guarantee untouched). See matcher.match / _pass2_pair for the gate.
    if video.score >= th.theta_v and video.coverage >= th.min_coverage:
        return make(
            Verdict.PROBABLE, 0.55,
            f"T4b uncorroborated video (v={video.score:.2f}/cov{video.coverage:.2f}) => review",
        )

    # ---- T5: no alignment ----------------------------------------------
    return make(Verdict.DIFFERENT, 0.0, "T5 no alignment")


# Verdicts that count as "same film" for clustering / action.
# NAME_COPY enters clustering (groups `(N)` copies), but is NOT a content tier:
# decide_tree never emits it; it is added by the name-grouping step (with content veto).
# The T1/T2 (content) zero-FP guarantee remains intact.
DUPLICATE_VERDICTS = frozenset(
    {Verdict.CERTAIN, Verdict.VERY_HIGH, Verdict.HIGH, Verdict.NAME_COPY}
)
REVIEW_VERDICTS = frozenset({Verdict.PROBABLE})
