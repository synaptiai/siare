"""Corpus loader for benchmark datasets.

Handles loading, chunking, and indexing of document corpora for RAG benchmarks.
Supports BEIR format and custom document collections.
"""

import logging
from pathlib import Path
from typing import Any, ClassVar, Optional


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
    SentenceTransformer = None  # type: ignore[misc, assignment]
    logger.info("sentence-transformers not installed.")


class CorpusLoader:
    """Loads and indexes document corpora for RAG benchmarks.

    Example:
        >>> loader = CorpusLoader(persist_dir="./corpus_data")
        >>> loader.load_beir_corpus(corpus, index_name="nfcorpus")
        >>> adapter_config = loader.get_adapter_config("nfcorpus")
    """

    DEFAULT_CHUNK_SIZE = 512  # tokens
    DEFAULT_CHUNK_OVERLAP = 50  # tokens
    DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSIONS: ClassVar[dict[str, int]] = {
        "all-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
        "all-distilroberta-v1": 768,
    }

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        """Initialize corpus loader.

        Args:
            persist_dir: Directory for persistent storage (None for in-memory)
            embedding_model: Sentence-transformers model name
            chunk_size: Maximum tokens per chunk
            chunk_overlap: Token overlap between chunks
        """
        if not _chromadb_available:
            raise ImportError(
                "chromadb required for corpus loading. "
                "Install with: pip install chromadb"
            )

        self.persist_dir = Path(persist_dir) if persist_dir else None
        self.embedding_model_name = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize ChromaDB client
        if self.persist_dir:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=Settings(anonymized_telemetry=False),
            )
        else:
            self._client = chromadb.Client(
                settings=Settings(anonymized_telemetry=False),
            )

        # Lazy-load embedding model
        self._embedding_model: Optional[SentenceTransformer] = None

    def _get_embedding_model(self) -> "SentenceTransformer":
        """Lazy-load the embedding model."""
        if self._embedding_model is None:
            if not _sentence_transformers_available:
                raise ImportError(
                    "sentence-transformers required for embeddings. "
                    "Install with: pip install sentence-transformers"
                )
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info(f"Loaded embedding model: {self.embedding_model_name}")
        return self._embedding_model

    def _chunk_text(self, text: str, doc_id: str) -> list[dict[str, Any]]:
        """Split text into overlapping chunks.

        Args:
            text: Full document text
            doc_id: Original document ID

        Returns:
            List of chunk dicts with id, text, and metadata
        """
        # Simple word-based chunking (approximate tokens)
        words = text.split()
        chunks = []

        if len(words) <= self.chunk_size:
            return [{"id": doc_id, "text": text, "chunk_index": 0}]

        start = 0
        chunk_index = 0
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            chunks.append({
                "id": f"{doc_id}_chunk_{chunk_index}",
                "text": chunk_text,
                "chunk_index": chunk_index,
                "parent_doc_id": doc_id,
            })

            start = end - self.chunk_overlap
            chunk_index += 1

            # Prevent infinite loop for very small overlap
            if start >= len(words) - self.chunk_overlap:
                break

        return chunks

    def load_documents(
        self,
        documents: list[dict[str, Any]],
        index_name: str,
        chunk: bool = True,
    ) -> int:
        """Load documents into a vector index.

        Args:
            documents: List of {"id": ..., "text": ..., "metadata": ...}
            index_name: Name for the vector index
            chunk: Whether to chunk long documents

        Returns:
            Number of vectors indexed
        """
        collection = self._client.get_or_create_collection(
            name=index_name,
            metadata={"hnsw:space": "cosine"},
        )

        model = self._get_embedding_model()
        total_indexed = 0

        for doc in documents:
            doc_id = doc["id"]
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})

            if chunk:
                chunks = self._chunk_text(text, doc_id)
            else:
                chunks = [{"id": doc_id, "text": text, "chunk_index": 0}]

            for chunk_data in chunks:
                chunk_id = chunk_data["id"]
                chunk_text = chunk_data["text"]

                if not chunk_text.strip():
                    continue

                # Generate embedding
                embedding = model.encode(chunk_text).tolist()

                # Store chunk metadata
                chunk_metadata = {
                    **metadata,
                    "chunk_index": chunk_data.get("chunk_index", 0),
                    "parent_doc_id": chunk_data.get("parent_doc_id", doc_id),
                }

                collection.add(
                    ids=[chunk_id],
                    embeddings=[embedding],
                    documents=[chunk_text],
                    metadatas=[chunk_metadata],
                )
                total_indexed += 1

        logger.info(f"Indexed {total_indexed} vectors to '{index_name}'")
        return total_indexed

    def load_beir_corpus(
        self,
        corpus: dict[str, dict[str, str]],
        index_name: str,
    ) -> int:
        """Load BEIR-format corpus into vector index.

        Args:
            corpus: BEIR format {doc_id: {"title": ..., "text": ...}}
            index_name: Name for the vector index

        Returns:
            Number of vectors indexed
        """
        documents = []
        for doc_id, doc_data in corpus.items():
            title = doc_data.get("title", "")
            text = doc_data.get("text", "")
            full_text = f"{title}\n\n{text}" if title else text

            documents.append({
                "id": doc_id,
                "text": full_text,
                "metadata": {"title": title},
            })

        return self.load_documents(documents, index_name)

    def get_document_count(self, index_name: str) -> int:
        """Get number of documents in an index.

        Args:
            index_name: Name of the vector index

        Returns:
            Document count
        """
        try:
            collection = self._client.get_collection(index_name)
        except ValueError:
            # Collection not found
            return 0
        else:
            return collection.count()

    def get_adapter_config(self, index_name: str) -> dict[str, Any]:
        """Get configuration for VectorSearchAdapter.

        Args:
            index_name: Name of the vector index

        Returns:
            Config dict for VectorSearchAdapter
        """
        dimension = self.EMBEDDING_DIMENSIONS.get(self.embedding_model_name, 384)
        config = {
            "backend": "chroma",
            "index_name": index_name,
            "embedding_model": "sentence-transformers",
            "dimension": dimension,
            "backend_config": {},
        }

        if self.persist_dir:
            config["backend_config"]["persist_directory"] = str(self.persist_dir)

        return config

    def delete_index(self, index_name: str) -> bool:
        """Delete a vector index.

        Args:
            index_name: Name of the index to delete

        Returns:
            True if deleted, False if not found
        """
        try:
            self._client.delete_collection(index_name)
        except ValueError:
            # Collection not found
            return False
        else:
            logger.info(f"Deleted index '{index_name}'")
            return True
