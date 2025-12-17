"""Vector Search Tool Adapter"""

import logging
import os
from typing import Any

import numpy as np
import numpy.typing as npt

from siare.adapters.base import ToolAdapter, register_adapter

logger = logging.getLogger(__name__)


# ============================================================================
# Lazy-loaded Embedding Models
# ============================================================================

# Sentence-transformers model (lazy loaded) - lowercase to avoid constant redefinition errors
_sentence_transformer_model: Any | None = None
_sentence_transformer_name = "all-MiniLM-L6-v2"

# Import flags (use lowercase to avoid constant redefinition errors)
_sentence_transformers_available = False
_openai_available = False

try:
    from sentence_transformers import SentenceTransformer

    _sentence_transformers_available = True
except ImportError:
    SentenceTransformer = None  # type: ignore

try:
    from openai import OpenAI

    _openai_available = True
except ImportError:
    OpenAI = None  # type: ignore


def _get_sentence_transformer_model() -> Any | None:
    """Lazy load sentence-transformer model"""
    global _sentence_transformer_model
    if _sentence_transformer_model is None and _sentence_transformers_available:
        if SentenceTransformer is not None:
            _sentence_transformer_model = SentenceTransformer(_sentence_transformer_name)
            logger.info(f"Loaded embedding model: {_sentence_transformer_name}")
    return _sentence_transformer_model


@register_adapter("vector_search")
class VectorSearchAdapter(ToolAdapter):
    """
    Vector search adapter for semantic similarity search

    Supports multiple backends:
    - In-memory (NumPy) - for MVP/testing
    - Pinecone
    - Weaviate
    - ChromaDB
    - Qdrant
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize vector search adapter

        Config keys:
            - backend: "memory", "pinecone", "weaviate", "chroma", "qdrant"
            - embedding_model: Model for generating embeddings ("auto", "sentence-transformers", "openai")
            - index_name: Name of the index/collection
            - dimension: Vector dimension
            - backend_config: Backend-specific configuration
        """
        super().__init__(config)

        self.backend = config.get("backend", "memory")
        self.embedding_model = config.get("embedding_model", "auto")
        self.index_name = config.get("index_name", "default")
        self.dimension = config.get("dimension", 384)
        self.backend_config = config.get("backend_config", {})
        self.max_memory_vectors = config.get("max_memory_vectors", 10000)  # Prevent memory leaks

        # Auto-detect best available embedding model (NO SILENT FALLBACK)
        if self.embedding_model == "auto":
            if _sentence_transformers_available:
                self.embedding_model = "sentence-transformers"
                logger.info("Auto-detected embedding model: sentence-transformers")
            elif _openai_available and os.getenv("OPENAI_API_KEY"):
                self.embedding_model = "openai"
                logger.info("Auto-detected embedding model: openai")
            else:
                # FAIL LOUDLY - no silent fallback
                raise ImportError(
                    "No embedding model available. Install sentence-transformers "
                    "(pip install sentence-transformers) or set OPENAI_API_KEY."
                )

        # Validate dimension
        if self.dimension <= 0:
            raise ValueError(f"dimension must be positive, got {self.dimension}")

        # Validate max_memory_vectors
        if self.max_memory_vectors <= 0:
            raise ValueError(f"max_memory_vectors must be positive, got {self.max_memory_vectors}")

        self.client: Any | None = None
        self.index: Any | None = None

        # In-memory storage for MVP
        self._memory_vectors: list[npt.NDArray[np.float64]] = []
        self._memory_metadata: list[dict[str, Any]] = []

    def initialize(self) -> None:
        """Initialize the vector search backend"""

        # Validate and auto-correct dimension for real embedding models
        self._validate_embedding_dimension()

        if self.backend == "memory":
            # In-memory backend - no initialization needed
            self.is_initialized = True

        elif self.backend == "pinecone":
            try:
                import pinecone  # type: ignore

                api_key = self.backend_config.get("api_key")
                environment = self.backend_config.get("environment")

                pinecone.init(api_key=api_key, environment=environment)  # type: ignore

                # Create or connect to index
                if self.index_name not in pinecone.list_indexes():  # type: ignore
                    pinecone.create_index(  # type: ignore
                        name=self.index_name,
                        dimension=self.dimension,
                        metric=self.backend_config.get("metric", "cosine"),
                    )

                self.index = pinecone.Index(self.index_name)  # type: ignore
                self.is_initialized = True

            except ImportError:
                raise ImportError("Pinecone not installed: pip install pinecone-client")

        elif self.backend == "chroma":
            try:
                import chromadb  # type: ignore

                persist_directory = self.backend_config.get("persist_directory")
                self.client = (
                    chromadb.Client()  # type: ignore
                    if not persist_directory
                    else chromadb.PersistentClient(path=persist_directory)  # type: ignore
                )

                self.index = self.client.get_or_create_collection(  # type: ignore
                    name=self.index_name, metadata={"dimension": self.dimension}
                )
                self.is_initialized = True

            except ImportError:
                raise ImportError("ChromaDB not installed: pip install chromadb")

        elif self.backend == "qdrant":
            try:
                from qdrant_client import QdrantClient
                from qdrant_client.models import Distance, VectorParams

                url = self.backend_config.get("url", "http://localhost:6333")
                api_key = self.backend_config.get("api_key")

                self.client = QdrantClient(url=url, api_key=api_key)

                # Create collection if doesn't exist
                collections = self.client.get_collections().collections
                if self.index_name not in [c.name for c in collections]:
                    self.client.create_collection(
                        collection_name=self.index_name,
                        vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE),
                    )

                self.is_initialized = True

            except ImportError:
                raise ImportError("Qdrant not installed: pip install qdrant-client")

        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Execute vector search

        Inputs:
            - query: Query text or vector
            - top_k: Number of results to return (default: 10)
            - filter: Optional metadata filter
            - return_vectors: Whether to return vectors (default: False)

        Returns:
            - results: List of {id, score, metadata, text, vector?}
        """
        if not self.is_initialized:
            self.initialize()

        query = inputs.get("query")
        top_k = inputs.get("top_k", 10)
        filter_dict = inputs.get("filter")
        return_vectors = inputs.get("return_vectors", False)

        if not query:
            return {"results": [], "error": "No query provided"}

        # Generate query vector
        query_vector = self._get_embedding(query)

        # Search based on backend
        results: list[dict[str, Any]] = []
        if self.backend == "memory":
            results = self._search_memory(query_vector, top_k, filter_dict)

        elif self.backend == "pinecone":
            results = self._search_pinecone(query_vector, top_k, filter_dict)

        elif self.backend == "chroma":
            results = self._search_chroma(query_vector, top_k, filter_dict)

        elif self.backend == "qdrant":
            results = self._search_qdrant(query_vector, top_k, filter_dict)

        # Post-process results
        if not return_vectors:
            for r in results:
                r.pop("vector", None)

        return {
            "results": results,
            "count": len(results),
            "query": query if isinstance(query, str) else "vector",
        }

    def validate_inputs(self, inputs: dict[str, Any]) -> list[str]:
        """Validate search inputs"""
        errors: list[str] = []

        if "query" not in inputs:
            errors.append("Missing required field: query")

        top_k = inputs.get("top_k", 10)
        if not isinstance(top_k, int) or top_k <= 0:
            errors.append("top_k must be a positive integer")

        return errors

    def add_documents(
        self, documents: list[dict[str, Any]], batch_size: int = 100
    ) -> dict[str, Any]:
        """
        Add documents to the index

        Args:
            documents: List of {id, text, metadata, vector?}
            batch_size: Batch size for uploads

        Returns:
            Status dict
        """
        if not self.is_initialized:
            self.initialize()

        added_count = 0

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]

            for doc in batch:
                doc_id = doc.get("id")
                text = doc.get("text", "")
                metadata = doc.get("metadata", {})
                vector = doc.get("vector")

                # Generate embedding if not provided
                if vector is None:
                    vector = self._get_embedding(text)

                # Store based on backend
                if self.backend == "memory":
                    # Check memory limit to prevent memory leaks
                    if len(self._memory_vectors) >= self.max_memory_vectors:
                        raise RuntimeError(
                            f"Memory vector limit reached ({self.max_memory_vectors}). "
                            f"Consider using a persistent backend or increasing max_memory_vectors."
                        )

                    self._memory_vectors.append(np.array(vector, dtype=np.float64))
                    self._memory_metadata.append({"id": doc_id, "text": text, **metadata})

                elif self.backend == "pinecone":
                    self.index.upsert([(doc_id, vector, metadata)])  # type: ignore

                elif self.backend == "chroma":
                    self.index.add(  # type: ignore
                        ids=[doc_id], embeddings=[vector], documents=[text], metadatas=[metadata]
                    )

                elif self.backend == "qdrant":
                    from qdrant_client.models import PointStruct

                    # Ensure doc_id is not None for Qdrant
                    if doc_id is None:
                        raise ValueError("Document ID cannot be None for Qdrant backend")

                    self.client.upsert(  # type: ignore
                        collection_name=self.index_name,
                        points=[
                            PointStruct(
                                id=doc_id, vector=vector, payload={"text": text, **metadata}
                            )
                        ],
                    )

                added_count += 1

        return {"added": added_count, "total": len(documents), "status": "success"}

    def _get_embedding(self, text: str) -> list[float]:
        """Generate embedding for text using configured model"""

        if self.embedding_model == "sentence-transformers":
            model = _get_sentence_transformer_model()
            if model is None:
                raise RuntimeError(
                    "Sentence-transformers model failed to load. "
                    "Ensure sentence-transformers is installed: pip install sentence-transformers"
                )
            embedding: npt.NDArray[np.float64] = model.encode(text, convert_to_numpy=True)
            return embedding.tolist()

        if self.embedding_model == "openai":
            if not _openai_available:
                raise ValueError("OpenAI not installed: pip install openai")

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")

            if OpenAI is None:
                raise RuntimeError("OpenAI client not available")

            client = OpenAI(api_key=api_key)
            response = client.embeddings.create(model="text-embedding-3-small", input=text)
            return response.data[0].embedding

        raise ValueError(f"Unsupported embedding model: {self.embedding_model}")

    def _validate_embedding_dimension(self) -> None:
        """Validate and auto-correct dimension for embedding models"""
        # Known model dimensions
        SENTENCE_TRANSFORMER_DIM = 384  # all-MiniLM-L6-v2
        OPENAI_MODEL_DIMS = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }

        if self.embedding_model == "sentence-transformers":
            expected_dim = SENTENCE_TRANSFORMER_DIM
            if self.dimension != expected_dim:
                logger.warning(
                    f"Config dimension ({self.dimension}) doesn't match "
                    f"sentence-transformers model dimension ({expected_dim}). "
                    f"Auto-correcting to {expected_dim}."
                )
                self.dimension = expected_dim

        elif self.embedding_model == "openai":
            model_name = "text-embedding-3-small"  # Default model
            expected_dim = OPENAI_MODEL_DIMS.get(model_name, 1536)
            if self.dimension != expected_dim:
                logger.warning(
                    f"Config dimension ({self.dimension}) doesn't match "
                    f"OpenAI model {model_name} dimension ({expected_dim}). "
                    f"Auto-correcting to {expected_dim}."
                )
                self.dimension = expected_dim

    def _search_memory(
        self, query_vector: list[float], top_k: int, filter_dict: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Search in-memory vectors"""

        if not self._memory_vectors:
            return []

        query_vec = np.array(query_vector, dtype=np.float64)
        vectors = np.array(self._memory_vectors, dtype=np.float64)

        # Cosine similarity
        similarities: npt.NDArray[np.float64] = np.dot(vectors, query_vec) / (
            np.linalg.norm(vectors, axis=1) * np.linalg.norm(query_vec)
        )

        # Get top-k indices
        top_indices: npt.NDArray[np.intp] = np.argsort(similarities)[::-1][:top_k]

        results: list[dict[str, Any]] = []
        for idx in top_indices:
            metadata = self._memory_metadata[int(idx)]

            # Apply filter if provided
            if filter_dict:
                if not all(metadata.get(k) == v for k, v in filter_dict.items()):
                    continue

            results.append(
                {
                    "id": metadata.get("id"),
                    "score": float(similarities[int(idx)]),
                    "text": metadata.get("text"),
                    "metadata": {k: v for k, v in metadata.items() if k not in ["id", "text"]},
                    "vector": self._memory_vectors[int(idx)].tolist(),
                }
            )

        return results

    def _search_pinecone(
        self, query_vector: list[float], top_k: int, filter_dict: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Search Pinecone index"""

        query_results = self.index.query(  # type: ignore
            vector=query_vector,
            top_k=top_k,
            filter=filter_dict,
            include_metadata=True,
            include_values=True,
        )

        results: list[dict[str, Any]] = []
        for match in query_results.matches:  # type: ignore
            results.append(
                {
                    "id": match.id,
                    "score": match.score,
                    "metadata": match.metadata,
                    "vector": match.values,
                }
            )

        return results

    def _search_chroma(
        self, query_vector: list[float], top_k: int, filter_dict: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Search ChromaDB collection"""

        query_results = self.index.query(  # type: ignore
            query_embeddings=[query_vector],
            n_results=top_k,
            where=filter_dict,
            include=["embeddings", "metadatas", "documents", "distances"],
        )

        results: list[dict[str, Any]] = []
        for i in range(len(query_results["ids"][0])):  # type: ignore
            results.append(
                {
                    "id": query_results["ids"][0][i],  # type: ignore
                    "score": 1.0
                    - query_results["distances"][0][i],  # type: ignore  # Convert distance to similarity
                    "text": query_results["documents"][0][i],  # type: ignore
                    "metadata": query_results["metadatas"][0][i],  # type: ignore
                    "vector": query_results["embeddings"][0][i]  # type: ignore
                    if query_results["embeddings"]  # type: ignore
                    else None,
                }
            )

        return results

    def _search_qdrant(
        self, query_vector: list[float], top_k: int, filter_dict: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Search Qdrant collection"""

        from qdrant_client.models import FieldCondition, Filter, MatchValue

        # Build filter
        qd_filter: Filter | None = None
        if filter_dict:
            conditions: list[FieldCondition] = [
                FieldCondition(key=k, match=MatchValue(value=v)) for k, v in filter_dict.items()
            ]
            qd_filter = Filter(must=conditions)  # type: ignore

        search_results = self.client.search(  # type: ignore
            collection_name=self.index_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=qd_filter,
            with_payload=True,
            with_vectors=True,
        )

        results: list[dict[str, Any]] = []
        for hit in search_results:  # type: ignore
            payload: dict[str, Any] = hit.payload or {}  # type: ignore
            results.append(
                {
                    "id": hit.id,  # type: ignore
                    "score": hit.score,  # type: ignore
                    "text": payload.get("text"),
                    "metadata": {k: v for k, v in payload.items() if k != "text"},
                    "vector": hit.vector,  # type: ignore
                }
            )

        return results

    def get_schema(self) -> dict[str, Any]:
        """Get tool schema"""
        return {
            "inputs": {
                "query": {"type": "string", "required": True, "description": "Search query text"},
                "top_k": {"type": "integer", "default": 10, "description": "Number of results"},
                "filter": {"type": "object", "description": "Metadata filter"},
                "return_vectors": {"type": "boolean", "default": False},
            },
            "outputs": {
                "results": {"type": "array", "description": "Search results"},
                "count": {"type": "integer"},
                "query": {"type": "string"},
            },
        }
