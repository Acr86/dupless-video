"""NAME COPY detection: files that differ only by a copy marker
`(N)` — e.g. `movie.avi`, `movie (1).avi`, `movie(2).avi`. Typical Windows copy-paste /
download-manager pattern: 'very likely' the same file.

Safety (to avoid inventing duplicates, §0):
- The marker is `(\\d{1,3})` AT THE END of the name (1-999). Does NOT match years `(2009)` (4 digits)
  or parts like `CD1`/`Part 2` (no parentheses) -> avoids false groups.
- Grouped ONLY within the SAME folder and with the SAME extension (`(N)` collisions
  happen when copying to the same location). Cross-folder is excluded (riskier).
- This is only a CANDIDACY signal: the final verdict is moderated by content (which can
  VETO the group) and deletion is always manual with KEEP protected.
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

# Copy marker at end of stem: optional space + (1..999). Years (4 digits) do NOT match.
_COPY_MARKER = re.compile(r"\s*\(\d{1,3}\)$")


def base_stem(stem: str) -> str:
    """Strips trailing `(N)` copy markers from the stem, repeatedly (e.g. 'x (1) (1)')."""
    prev = None
    while prev != stem:
        prev = stem
        stem = _COPY_MARKER.sub("", stem).rstrip()
    return stem


def name_key(path: str) -> tuple[str, str, str]:
    """Grouping key: (folder, marker-stripped stem, extension), lowercased.
    Two paths with the same key differ ONLY by the `(N)` marker -> probable copies."""
    p = Path(path)
    return (str(p.parent).lower(), base_stem(p.stem).strip().lower(), p.suffix.lower())


def name_sibling_pairs(paths: list[str]) -> list[tuple[str, str]]:
    """Pairs (preferred, copy) among files that differ only by `(N)` in the same folder.
    The 'preferred' in each group is the SHORTEST PATH (the original 'movie.avi' over
    copies 'movie (N).avi'); each copy is paired with it to merge them into a cluster."""
    groups: dict[tuple, list[str]] = defaultdict(list)
    for p in paths:
        groups[name_key(p)].append(p)
    pairs: list[tuple[str, str]] = []
    for members in groups.values():
        if len(members) < 2:
            continue
        members.sort(key=len)                      # shortest path first = preferred
        base = members[0]
        for other in members[1:]:
            pairs.append((base, other))
    return pairs
