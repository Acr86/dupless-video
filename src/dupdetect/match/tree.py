"""The staged decision tree. Pure logic — nearly complete as-is.

Each pair descends the tree; the FIRST tier that fires fixes the verdict and
confidence. Precision comes from requiring agreement across independent signals.
"""
from __future__ import annotations

from dupdetect.config import Thresholds
from dupdetect.models import AlignResult, Record, Result, Verdict


def _cut_density(rec: Record) -> float:
    """Scene cuts per second. A coarse signature (low density — SEEK-sampled giants) is not
    discriminative for the scenes-only T4 tier (see align/scenes.py and the T4 guard)."""
    dur = rec.probe.duration_s or 0.0
    n = len(rec.scene_cuts) if rec.scene_cuts is not None else 0
    return (n / dur) if dur > 0 else 0.0


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
        return make(Verdict.CERTAIN, 1.00, "T0 sampled hash identical (verify byte-exact before deleting)")

    # ---- GUARD edition vs duplicate (before declaring dup) -------------
    # If video aligns strongly but 'b' is a contiguous superset of 'a' (or vice versa),
    # it is a different edition (director's cut), NOT a junk duplicate.
    if video.score >= th.theta_v and video.contiguous_superset:
        return make(
            Verdict.DIFFERENT_EDITION,
            min(0.90, video.score),
            f"different edition: contiguous superset (+{video.extra_ratio:.0%} runtime)",
        )

    # ---- T1: confirmed by TWO independent modalities -------------------
    # Coverage required: Smith-Waterman selects the most similar frames, so
    # a high score over a tiny path (cov ~0.04 between different films) is NOT
    # "strong video". (Audio is already auto-gated by min_overlap in align_audio.)
    if audio.score >= th.theta_a and video.score >= th.theta_v and video.coverage >= th.min_coverage:
        return make(
            Verdict.CERTAIN, 0.99,
            f"T1 audio({audio.score:.2f})+video({video.score:.2f}, cov {video.coverage:.2f}) agree",
        )

    # ---- T2: same video, audio does NOT align => different dub ----------
    # Where Plex falls short. Audio MUST NOT align (otherwise it would be T1).
    if (
        video.score >= th.theta_v_high
        and video.coverage >= th.min_coverage
        and audio.score < th.theta_a_low
    ):
        return make(
            Verdict.VERY_HIGH, 0.95,
            f"T2 video({video.score:.2f}) identical, audio doesn't align => different dub",
        )

    # ---- T3: partial video corroborated by scene structure -------------
    if video.score >= th.theta_v and video.coverage >= th.min_coverage and scenes.score >= th.theta_s:
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
    # Strong signal from ONE modality without corroboration -> review ("when in doubt,
    # to the queue"). Video requires coverage: otherwise a spurious short path (cov 0.04)
    # between different films would send false positives to the queue. (Audio auto-gated.)
    if (video.score >= th.theta_v and video.coverage >= th.min_coverage) or audio.score >= th.theta_a:
        return make(
            Verdict.PROBABLE, 0.55,
            f"T4b strong signal uncorroborated (v={video.score:.2f}/cov{video.coverage:.2f},"
            f"a={audio.score:.2f}) => review",
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
