"""Single-file mode: what the Python watcher calls when a new file is dropped.
Reuses EXACTLY analyze_file() + match() against the already-populated store."""
from __future__ import annotations

from dupdetect.config import Thresholds
from dupdetect.features.embeddings import Embedder
from dupdetect.match.cache import EmbeddingCache
from dupdetect.match.matcher import match
from dupdetect.match.retrieval import CoarseIndex
from dupdetect.models import Result
from dupdetect.pipeline.analyze import analyze_file
from dupdetect.store import FingerprintStore


def analyze_single(path: str, store: FingerprintStore, embedder: Embedder,
                   index: CoarseIndex, th: Thresholds,
                   cache: EmbeddingCache | None = None) -> list[Result]:
    """Processes ONE new file and returns its matches against the library.

    The watcher keeps `index` and `cache` alive across files to avoid rebuilding
    FAISS or reloading embeddings. The store is already populated by a prior full-scan.
    """
    rec = analyze_file(path, store, embedder, th)
    results = match(rec, store, index, th, cache=cache)
    # TODO: if index/cache don't yet include `rec`, add it so future
    # files in the same session can see it.
    return results
