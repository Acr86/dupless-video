"""Content hash for T0 (byte-exact identity)."""
from __future__ import annotations

import xxhash

CHUNK = 8 * 1024 * 1024  # 8 MB per region


def content_hash(path: str) -> str:
    """xxhash of head ‖ mid ‖ tail. Catches byte-identical duplicates instantly
    without reading the full file (key for large files)."""
    h = xxhash.xxh3_64()
    import os
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        # head
        h.update(f.read(CHUNK))
        if size > 2 * CHUNK:
            # mid
            f.seek(size // 2)
            h.update(f.read(CHUNK))
            # tail
            f.seek(max(0, size - CHUNK))
            h.update(f.read(CHUNK))
    return h.hexdigest()
