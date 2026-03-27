"""Knowledge base for domain knowledge and prior evolution run patterns.

Provides queryable storage of domain-specific guidance, RAG best practices,
prompt engineering techniques, and learnings from prior evolution runs.

The knowledge base is queried by the AgenticDirector via the
QueryKnowledgeBaseTool during variation sessions.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from siare.core.models import EvolutionRunSummary, KnowledgeDocument

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """Queryable store of domain knowledge for the variation agent.

    Categories:
        - rag_patterns: Common RAG architectures and failure modes.
        - prompt_engineering: Techniques, templates, proven patterns.
        - prior_runs: Learnings from completed evolution runs.
        - domain: User-provided domain-specific guidance.

    Usage:
        kb = KnowledgeBase()
        kb.load_builtin_knowledge()
        docs = kb.query("how to fix hallucination", category="rag_patterns")
    """

    VALID_CATEGORIES = frozenset(
        {"rag_patterns", "prompt_engineering", "prior_runs", "domain"}
    )

    def __init__(
        self,
        knowledge_dir: str | None = None,
    ) -> None:
        self._documents: dict[str, list[KnowledgeDocument]] = {
            cat: [] for cat in self.VALID_CATEGORIES
        }
        self.knowledge_dir = knowledge_dir

    @property
    def total_documents(self) -> int:
        """Total number of documents across all categories."""
        return sum(len(docs) for docs in self._documents.values())

    def add_document(self, doc: KnowledgeDocument) -> None:
        """Add a single document to the knowledge base.

        Args:
            doc: The document to add.

        Raises:
            ValueError: If the category is invalid.
        """
        if doc.category not in self.VALID_CATEGORIES:
            raise ValueError(
                f"Invalid category '{doc.category}'. "
                f"Valid: {sorted(self.VALID_CATEGORIES)}"
            )
        self._documents[doc.category].append(doc)

    def query(
        self,
        query: str,
        category: str | None = None,
        top_k: int = 5,
    ) -> list[KnowledgeDocument]:
        """Retrieve relevant documents for a query.

        Uses simple keyword matching. Override for vector-based retrieval.

        Args:
            query: Natural language query.
            category: Optional category filter.
            top_k: Maximum results to return.

        Returns:
            List of relevant documents sorted by relevance.
        """
        query_lower = query.lower()
        query_terms = set(query_lower.split())

        candidates: list[KnowledgeDocument] = []
        categories = (
            [category] if category and category in self.VALID_CATEGORIES
            else list(self.VALID_CATEGORIES)
        )

        for cat in categories:
            for doc in self._documents.get(cat, []):
                score = self._score_document(doc, query_terms)
                if score > 0:
                    scored = doc.model_copy(update={"relevanceScore": score})
                    candidates.append(scored)

        candidates.sort(
            key=lambda d: d.relevanceScore or 0.0,
            reverse=True,
        )
        return candidates[:top_k]

    def _score_document(
        self,
        doc: KnowledgeDocument,
        query_terms: set[str],
    ) -> float:
        """Score a document against query terms (simple keyword match)."""
        content_lower = doc.content.lower()
        content_terms = set(content_lower.split())
        matching = query_terms & content_terms
        if not matching:
            return 0.0
        return len(matching) / len(query_terms)

    def add_evolution_summary(
        self,
        summary: EvolutionRunSummary,
    ) -> None:
        """Record learnings from a completed evolution run.

        Converts the summary into queryable knowledge documents
        in the 'prior_runs' category.
        """
        parts: list[str] = [
            f"Evolution run {summary.jobId} ({summary.domain}):",
            f"Ran {summary.totalGenerations} generations.",
            f"Best quality: {summary.bestQuality:.3f}.",
            f"Pareto frontier size: {summary.finalParetoSize}.",
        ]

        if summary.effectiveMutations:
            effective = ", ".join(
                f"{k}: {v}" for k, v in summary.effectiveMutations.items()
            )
            parts.append(f"Effective mutations: {effective}.")

        if summary.breakthroughs:
            for bt in summary.breakthroughs:
                desc = bt.get("description", "unknown")
                gen = bt.get("generation", "?")
                jump = bt.get("qualityJump", 0)
                parts.append(
                    f"Breakthrough at gen {gen}: {desc} (+{jump:.3f})."
                )

        if summary.deadEnds:
            parts.append(f"Dead ends: {'; '.join(summary.deadEnds)}.")

        doc = KnowledgeDocument(
            content=" ".join(parts),
            category="prior_runs",
            metadata={
                "jobId": summary.jobId,
                "domain": summary.domain,
                "bestQuality": summary.bestQuality,
            },
        )
        self.add_document(doc)

    def load_builtin_knowledge(self) -> None:
        """Load curated RAG and prompt engineering best practices."""
        builtins = [
            KnowledgeDocument(
                content=(
                    "Hallucination in RAG systems is often caused by the "
                    "generator ignoring retrieved context. Mitigate by adding "
                    "explicit faithfulness instructions: 'Only answer based "
                    "on the provided documents. If the answer is not in the "
                    "documents, say so.'"
                ),
                category="rag_patterns",
            ),
            KnowledgeDocument(
                content=(
                    "When retrieval returns irrelevant documents, add a "
                    "re-ranking step between retrieval and generation. "
                    "A dedicated re-ranker role can filter out noise and "
                    "improve answer quality."
                ),
                category="rag_patterns",
            ),
            KnowledgeDocument(
                content=(
                    "For multi-hop questions, a single retrieval step is "
                    "often insufficient. Use iterative retrieval where the "
                    "agent retrieves, reasons, and retrieves again to "
                    "gather all needed evidence."
                ),
                category="rag_patterns",
            ),
            KnowledgeDocument(
                content=(
                    "Chain-of-thought prompting improves reasoning quality. "
                    "Add 'Think step by step' or provide worked examples "
                    "to guide the model through complex tasks."
                ),
                category="prompt_engineering",
            ),
            KnowledgeDocument(
                content=(
                    "Few-shot examples are most effective when they match "
                    "the distribution of actual queries. Use representative "
                    "examples that cover common edge cases."
                ),
                category="prompt_engineering",
            ),
            KnowledgeDocument(
                content=(
                    "Role-specific prompts should clearly define "
                    "boundaries: what the agent should do, what it should "
                    "NOT do, and how to handle ambiguous cases. Explicit "
                    "constraints reduce hallucination and scope creep."
                ),
                category="prompt_engineering",
            ),
            KnowledgeDocument(
                content=(
                    "Reducing agent count can improve latency and cost "
                    "without quality loss if the removed agent's function "
                    "can be absorbed by another. Consider topology "
                    "simplification when pipeline latency is a concern."
                ),
                category="rag_patterns",
            ),
        ]
        for doc in builtins:
            self.add_document(doc)
        logger.info("Loaded %d built-in knowledge documents", len(builtins))

    def load_from_directory(self, directory: str | None = None) -> int:
        """Load knowledge documents from a directory.

        Each .json file should contain a KnowledgeDocument-compatible dict.
        Each .txt file is loaded as a 'domain' category document.

        Returns:
            Number of documents loaded.
        """
        dir_path = Path(directory or self.knowledge_dir or "")
        if not dir_path.is_dir():
            logger.warning("Knowledge directory not found: %s", dir_path)
            return 0

        count = 0
        for path in sorted(dir_path.iterdir()):
            if path.suffix == ".json":
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    doc = KnowledgeDocument(**data)
                    self.add_document(doc)
                    count += 1
                except Exception as e:
                    logger.warning("Failed to load %s: %s", path, e)
            elif path.suffix == ".txt":
                try:
                    content = path.read_text(encoding="utf-8").strip()
                    if content:
                        doc = KnowledgeDocument(
                            content=content,
                            category="domain",
                            metadata={"source": str(path)},
                        )
                        self.add_document(doc)
                        count += 1
                except Exception as e:
                    logger.warning("Failed to load %s: %s", path, e)

        logger.info("Loaded %d documents from %s", count, dir_path)
        return count
