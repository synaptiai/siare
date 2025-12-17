"""Corpus index manager with caching for benchmark evaluation.

Manages vector indices for benchmark corpora with:
- Persistent storage for reuse across runs
- Metadata tracking for cache validation
- Search interface compatible with VectorSearchAdapter
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Check for optional dependencies
_chromadb_available = False
_sentence_transformers_available = False

try:
    import chromadb
    from chromadb.config import Settings
    _chromadb_available = True
except ImportError:
    logger.info("chromadb not installed. Install with: pip install chromadb")

try:
    from sentence_transformers import SentenceTransformer
    _sentence_transformers_available = True
except ImportError:
    logger.info("sentence-transformers not installed.")


class CorpusIndexManager:
    """Manages vector indices for benchmark corpora.

    Features:
    - Creates and persists vector indices
    - Validates cache freshness via corpus hash
    - Provides search interface for retrieval evaluation
    - Returns adapter config for integration with SIARE tools

    Example:
        >>> manager = CorpusIndexManager(persist_dir="./vector_store")
        >>> manager.create_index("frames_corpus", documents)
        >>> results = manager.search("Who wrote the Iliad?", "frames_corpus", top_k=10)
        >>> config = manager.get_adapter_config("frames_corpus")
    """

    DEFAULT_PERSIST_DIR = Path.home() / ".siare" / "vector_indices"
    DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

    def __init__(
        self,
        persist_dir: str | None = None,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        """Initialize index manager.

        Args:
            persist_dir: Directory for persistent storage
            embedding_model: Sentence-transformers model name
        """
        if not _chromadb_available:
            raise ImportError(
                "chromadb required for index management. "
                "Install with: pip install chromadb"
            )
        assert chromadb is not None  # Type narrowing for pyright
        assert Settings is not None

        self.persist_dir = Path(persist_dir) if persist_dir else self.DEFAULT_PERSIST_DIR
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_model_name = embedding_model

        # Initialize ChromaDB client
        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )

        # Lazy-load embedding model
        self._embedding_model: SentenceTransformer | None = None

        # Metadata storage
        self._metadata_dir = self.persist_dir / "metadata"
        self._metadata_dir.mkdir(exist_ok=True)

    def _get_embedding_model(self) -> "SentenceTransformer":
        """Lazy-load the embedding model."""
        if self._embedding_model is None:
            if not _sentence_transformers_available:
                raise ImportError(
                    "sentence-transformers required for embeddings. "
                    "Install with: pip install sentence-transformers"
                )
            assert SentenceTransformer is not None  # Type narrowing
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info(f"Loaded embedding model: {self.embedding_model_name}")
        return self._embedding_model

    def _get_metadata_path(self, index_name: str) -> Path:
        """Get path to metadata file for an index."""
        return self._metadata_dir / f"{index_name}_metadata.json"

    def create_index(
        self,
        index_name: str,
        documents: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Create or replace a vector index.

        Args:
            index_name: Name for the index
            documents: List of {"doc_id": ..., "text": ..., ...}
            metadata: Optional metadata to store with index

        Returns:
            Number of documents indexed
        """
        # Delete existing collection if present
        try:
            self._client.delete_collection(index_name)
        except Exception:
            pass

        collection = self._client.create_collection(
            name=index_name,
            metadata={"hnsw:space": "cosine"},
        )

        model = self._get_embedding_model()

        # Batch embed and add
        batch_size = 100
        total_indexed = 0

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]

            ids = [doc["doc_id"] for doc in batch]
            texts = [doc["text"] for doc in batch]

            # Generate embeddings
            embeddings = model.encode(texts).tolist()

            # Prepare metadata for each document
            metadatas = []
            for doc in batch:
                doc_meta = {k: v for k, v in doc.items() if k not in ["doc_id", "text"]}
                # Ensure all values are serializable
                doc_meta = {k: str(v) if not isinstance(v, (str, int, float, bool)) else v
                           for k, v in doc_meta.items()}
                # ChromaDB requires non-empty metadata or None
                if not doc_meta:
                    doc_meta = {"_placeholder": "true"}
                metadatas.append(doc_meta)

            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
            total_indexed += len(batch)

        # Save index metadata
        index_metadata = {
            "index_name": index_name,
            "document_count": total_indexed,
            "embedding_model": self.embedding_model_name,
            "created_at": time.time(),
            **(metadata or {}),
        }

        with open(self._get_metadata_path(index_name), "w") as f:
            json.dump(index_metadata, f, indent=2)

        logger.info(f"Created index '{index_name}' with {total_indexed} documents")
        return total_indexed

    def index_exists(self, index_name: str) -> bool:
        """Check if an index exists.

        Args:
            index_name: Name of the index

        Returns:
            True if index exists
        """
        try:
            self._client.get_collection(index_name)
            return True
        except Exception:
            return False

    def get_document_count(self, index_name: str) -> int:
        """Get number of documents in an index.

        Args:
            index_name: Name of the index

        Returns:
            Number of documents
        """
        try:
            collection = self._client.get_collection(index_name)
            return collection.count()
        except Exception:
            return 0

    def search(
        self,
        query: str,
        index_name: str,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Search an index for relevant documents.

        Args:
            query: Search query
            index_name: Name of the index
            top_k: Number of results to return

        Returns:
            List of document dictionaries with scores
        """
        collection = self._client.get_collection(index_name)
        model = self._get_embedding_model()

        # Generate query embedding
        query_embedding = model.encode([query]).tolist()

        # Search
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        # Format results
        formatted = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                doc = {
                    "doc_id": doc_id,
                    "text": results["documents"][0][i] if results["documents"] else "",
                    "score": 1 - results["distances"][0][i] if results["distances"] else 0,
                }
                if results["metadatas"] and results["metadatas"][0]:
                    doc.update(results["metadatas"][0][i])
                formatted.append(doc)

        return formatted

    def get_index_metadata(self, index_name: str) -> dict[str, Any]:
        """Get metadata for an index.

        Args:
            index_name: Name of the index

        Returns:
            Metadata dictionary
        """
        metadata_path = self._get_metadata_path(index_name)
        if metadata_path.exists():
            with open(metadata_path) as f:
                return json.load(f)
        return {}

    def is_cache_valid(self, index_name: str, expected_hash: str) -> bool:
        """Check if cached index is still valid.

        Args:
            index_name: Name of the index
            expected_hash: Expected corpus hash

        Returns:
            True if cache is valid
        """
        metadata = self.get_index_metadata(index_name)
        return metadata.get("corpus_hash") == expected_hash

    def get_adapter_config(self, index_name: str) -> dict[str, Any]:
        """Get configuration for VectorSearchAdapter.

        Args:
            index_name: Name of the index

        Returns:
            Configuration dictionary
        """
        return {
            "backend": "chroma",
            "index_name": index_name,
            "embedding_model": self.embedding_model_name,
            "backend_config": {
                "persist_directory": str(self.persist_dir),
            },
        }

    def delete_index(self, index_name: str) -> bool:
        """Delete an index.

        Args:
            index_name: Name of the index

        Returns:
            True if deleted successfully
        """
        try:
            self._client.delete_collection(index_name)
            metadata_path = self._get_metadata_path(index_name)
            if metadata_path.exists():
                metadata_path.unlink()
            return True
        except Exception as e:
            logger.warning(f"Failed to delete index '{index_name}': {e}")
            return False
