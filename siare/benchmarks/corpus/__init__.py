"""Corpus loading utilities for benchmarks."""

from siare.benchmarks.corpus.loader import CorpusLoader

try:
    from siare.benchmarks.corpus.wikipedia_loader import WikipediaCorpusLoader
except ImportError:
    WikipediaCorpusLoader = None  # type: ignore

try:
    from siare.benchmarks.corpus.index_manager import CorpusIndexManager
except ImportError:
    CorpusIndexManager = None  # type: ignore

__all__ = ["CorpusLoader", "WikipediaCorpusLoader", "CorpusIndexManager"]
