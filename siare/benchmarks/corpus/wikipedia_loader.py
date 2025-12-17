"""Wikipedia corpus loader for FRAMES benchmark.

Fetches Wikipedia articles referenced in FRAMES dataset and prepares
them for vector indexing. Implements caching to avoid redundant fetches.
"""

import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional
from urllib.parse import unquote, urlparse

if TYPE_CHECKING:
    from siare.benchmarks.datasets.frames import FRAMESDataset

logger = logging.getLogger(__name__)

# Check for optional dependencies
_wikipedia_available = False
try:
    import wikipedia
    _wikipedia_available = True
except ImportError:
    logger.info("wikipedia library not installed. Install with: pip install wikipedia")


class WikipediaCorpusLoader:
    """Loads Wikipedia articles for FRAMES benchmark corpus.

    Features:
    - Extracts Wikipedia URLs from FRAMES samples
    - Fetches article content with rate limiting
    - Handles disambiguation pages gracefully
    - Caches fetched articles to avoid re-fetching
    - Chunks articles for vector indexing

    Example:
        >>> loader = WikipediaCorpusLoader(cache_dir="./wiki_cache")
        >>> corpus = loader.build_corpus_from_frames(frames_dataset)
        >>> print(f"Built corpus with {len(corpus)} documents")
    """

    DEFAULT_CACHE_DIR = Path.home() / ".siare" / "wiki_cache"
    RATE_LIMIT_DELAY = 0.5  # seconds between requests

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> None:
        """Initialize Wikipedia corpus loader.

        Args:
            cache_dir: Directory for caching fetched articles
            chunk_size: Words per chunk for vector indexing
            chunk_overlap: Word overlap between chunks
        """
        if not _wikipedia_available:
            raise ImportError(
                "wikipedia library required. Install with: pip install wikipedia"
            )

        self.cache_dir = Path(cache_dir) if cache_dir else self.DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._last_request_time = 0.0

    def extract_wiki_urls(self, samples: list[dict[str, Any]]) -> list[str]:
        """Extract unique Wikipedia URLs from FRAMES samples.

        Args:
            samples: List of FRAMES samples with wiki_links field

        Returns:
            List of unique Wikipedia URLs
        """
        urls = set()
        for sample in samples:
            wiki_links = sample.get("wiki_links", [])
            if isinstance(wiki_links, list):
                urls.update(wiki_links)
        return list(urls)

    def _url_to_title(self, url: str) -> str:
        """Convert Wikipedia URL to article title.

        Args:
            url: Wikipedia URL

        Returns:
            Article title
        """
        parsed = urlparse(url)
        path = parsed.path

        # Extract title from /wiki/Title format
        if "/wiki/" in path:
            title = path.split("/wiki/")[-1]
            # URL decode and replace underscores
            title = unquote(title).replace("_", " ")
            return title

        return url

    def _get_cache_path(self, title: str) -> Path:
        """Get cache file path for article title.

        Args:
            title: Article title

        Returns:
            Path to cache file
        """
        # Use hash to handle special characters in titles
        title_hash = hashlib.md5(title.encode()).hexdigest()[:16]
        safe_title = re.sub(r"[^\w\s-]", "", title)[:50]
        return self.cache_dir / f"{safe_title}_{title_hash}.json"

    def _rate_limit(self) -> None:
        """Enforce rate limiting between Wikipedia API requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.RATE_LIMIT_DELAY:
            time.sleep(self.RATE_LIMIT_DELAY - elapsed)
        self._last_request_time = time.time()

    def fetch_article(self, title: str) -> Optional[str]:
        """Fetch Wikipedia article content.

        Args:
            title: Article title

        Returns:
            Article content or None if not found
        """
        # Check cache first
        cache_path = self._get_cache_path(title)
        if cache_path.exists():
            try:
                with open(cache_path) as f:
                    cached = json.load(f)
                    return cached.get("content")
            except (json.JSONDecodeError, KeyError):
                pass  # Cache corrupted, refetch

        # Rate limit API requests
        self._rate_limit()

        try:
            # Try to get the page directly
            page = wikipedia.page(title, auto_suggest=True)
            content = page.content

            # Cache the result
            with open(cache_path, "w") as f:
                json.dump({
                    "title": page.title,
                    "url": page.url,
                    "content": content,
                    "fetched_at": time.time(),
                }, f)

            logger.debug(f"Fetched article: {title}")
            return content

        except wikipedia.exceptions.DisambiguationError as e:
            # Handle disambiguation - try first option
            if e.options:
                logger.debug(f"Disambiguation for '{title}', trying '{e.options[0]}'")
                return self.fetch_article(e.options[0])
            return None

        except wikipedia.exceptions.PageError:
            logger.warning(f"Wikipedia page not found: {title}")
            return None

        except Exception as e:
            logger.warning(f"Error fetching '{title}': {e}")
            return None

    def _chunk_text(self, text: str, doc_id: str) -> list[dict[str, Any]]:
        """Split text into overlapping chunks.

        Args:
            text: Full document text
            doc_id: Document identifier

        Returns:
            List of chunk dictionaries
        """
        words = text.split()
        chunks = []

        if len(words) <= self.chunk_size:
            return [{"doc_id": doc_id, "text": text, "chunk_index": 0}]

        start = 0
        chunk_index = 0
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            chunks.append({
                "doc_id": f"{doc_id}_chunk_{chunk_index}",
                "text": chunk_text,
                "chunk_index": chunk_index,
                "parent_doc_id": doc_id,
            })

            start = end - self.chunk_overlap
            chunk_index += 1

            if start >= len(words) - self.chunk_overlap:
                break

        return chunks

    def build_corpus_from_frames(
        self,
        dataset: "FRAMESDataset",
        max_articles: Optional[int] = None,
        chunk: bool = True,
    ) -> list[dict[str, Any]]:
        """Build corpus from FRAMES dataset's Wikipedia references.

        Args:
            dataset: FRAMES dataset with wiki_links
            max_articles: Maximum articles to fetch (None for all)
            chunk: Whether to chunk long articles

        Returns:
            List of document dictionaries ready for indexing
        """
        if not dataset._loaded:
            dataset.load()

        # Collect all unique Wikipedia URLs
        all_urls = set()
        for sample in dataset._samples:
            wiki_links = sample.metadata.get("wiki_links", [])
            if not wiki_links:
                # Try ground_truth_context which stores wiki_links
                wiki_links = sample.ground_truth_context or []
            all_urls.update(wiki_links)

        logger.info(f"Found {len(all_urls)} unique Wikipedia articles in FRAMES")

        # Limit if requested
        urls_to_fetch = list(all_urls)
        if max_articles:
            urls_to_fetch = urls_to_fetch[:max_articles]

        # Fetch articles
        corpus = []
        for url in urls_to_fetch:
            title = self._url_to_title(url)
            content = self.fetch_article(title)

            if content:
                doc_id = hashlib.md5(url.encode()).hexdigest()[:12]

                if chunk:
                    chunks = self._chunk_text(content, doc_id)
                    for chunk_doc in chunks:
                        chunk_doc["title"] = title
                        chunk_doc["source_url"] = url
                        corpus.append(chunk_doc)
                else:
                    corpus.append({
                        "doc_id": doc_id,
                        "title": title,
                        "text": content,
                        "source_url": url,
                    })

        logger.info(f"Built corpus with {len(corpus)} documents/chunks")
        return corpus

    def get_corpus_stats(self, corpus: list[dict[str, Any]]) -> dict[str, Any]:
        """Get statistics about the corpus.

        Args:
            corpus: List of corpus documents

        Returns:
            Dictionary with corpus statistics
        """
        total_words = sum(len(doc["text"].split()) for doc in corpus)
        unique_sources = len(set(doc.get("source_url", doc["doc_id"]) for doc in corpus))

        return {
            "total_documents": len(corpus),
            "unique_articles": unique_sources,
            "total_words": total_words,
            "avg_words_per_doc": total_words / len(corpus) if corpus else 0,
        }
