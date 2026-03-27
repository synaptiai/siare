"""Tests for KnowledgeBase service."""

import json
from pathlib import Path

import pytest

from siare.core.models import EvolutionRunSummary, KnowledgeDocument
from siare.services.knowledge_base import KnowledgeBase


class TestKnowledgeBaseBasic:
    """Tests for basic KnowledgeBase operations."""

    def test_creation(self):
        kb = KnowledgeBase()
        assert kb.total_documents == 0

    def test_add_document(self):
        kb = KnowledgeBase()
        doc = KnowledgeDocument(
            content="Test content",
            category="rag_patterns",
        )
        kb.add_document(doc)
        assert kb.total_documents == 1

    def test_add_document_invalid_category(self):
        kb = KnowledgeBase()
        doc = KnowledgeDocument(
            content="Test",
            category="invalid_category",
        )
        with pytest.raises(ValueError, match="Invalid category"):
            kb.add_document(doc)


class TestKnowledgeBaseQuery:
    """Tests for knowledge base querying."""

    def setup_method(self):
        self.kb = KnowledgeBase()
        self.kb.add_document(KnowledgeDocument(
            content="Hallucination is caused by ignoring context.",
            category="rag_patterns",
        ))
        self.kb.add_document(KnowledgeDocument(
            content="Chain of thought prompting improves reasoning.",
            category="prompt_engineering",
        ))
        self.kb.add_document(KnowledgeDocument(
            content="Re-ranking improves retrieval quality.",
            category="rag_patterns",
        ))

    def test_query_returns_relevant_docs(self):
        results = self.kb.query("hallucination context")
        assert len(results) > 0
        assert results[0].relevanceScore is not None
        assert results[0].relevanceScore > 0
        assert "Hallucination" in results[0].content

    def test_query_with_category_filter(self):
        results = self.kb.query("improves", category="prompt_engineering")
        assert len(results) == 1
        assert "Chain of thought" in results[0].content

    def test_query_top_k_limit(self):
        results = self.kb.query("improves", top_k=1)
        assert len(results) == 1

    def test_query_no_matches(self):
        results = self.kb.query("quantum computing")
        assert len(results) == 0

    def test_query_results_sorted_by_relevance(self):
        results = self.kb.query("retrieval quality")
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert (
                    (results[i].relevanceScore or 0)
                    >= (results[i + 1].relevanceScore or 0)
                )


class TestKnowledgeBaseEvolutionSummary:
    """Tests for recording evolution run summaries."""

    def test_add_evolution_summary(self):
        kb = KnowledgeBase()
        summary = EvolutionRunSummary(
            jobId="job-1",
            domain="support",
            totalGenerations=30,
            effectiveMutations={"prompt_change": 10, "add_role": 2},
            breakthroughs=[{
                "generation": 12,
                "qualityJump": 0.15,
                "description": "Added fact-checker",
            }],
            deadEnds=["remove_role degraded quality"],
            finalParetoSize=4,
            bestQuality=0.91,
        )
        kb.add_evolution_summary(summary)
        assert kb.total_documents == 1

        results = kb.query("prompt_change effective", category="prior_runs")
        assert len(results) > 0
        assert "job-1" in results[0].metadata.get("jobId", "")

    def test_summary_includes_breakthroughs(self):
        kb = KnowledgeBase()
        summary = EvolutionRunSummary(
            jobId="job-2",
            domain="qa",
            totalGenerations=20,
            breakthroughs=[{
                "generation": 5,
                "qualityJump": 0.2,
                "description": "Added re-ranker",
            }],
            bestQuality=0.88,
        )
        kb.add_evolution_summary(summary)
        results = kb.query("re-ranker breakthrough", category="prior_runs")
        assert len(results) > 0


class TestKnowledgeBaseBuiltin:
    """Tests for built-in knowledge loading."""

    def test_load_builtin_knowledge(self):
        kb = KnowledgeBase()
        kb.load_builtin_knowledge()
        assert kb.total_documents > 0

    def test_builtin_covers_hallucination(self):
        kb = KnowledgeBase()
        kb.load_builtin_knowledge()
        results = kb.query("hallucination faithfulness")
        assert len(results) > 0

    def test_builtin_covers_chain_of_thought(self):
        kb = KnowledgeBase()
        kb.load_builtin_knowledge()
        results = kb.query("chain of thought reasoning")
        assert len(results) > 0


class TestKnowledgeBaseFileLoading:
    """Tests for loading knowledge from directory."""

    def test_load_from_nonexistent_directory(self):
        kb = KnowledgeBase()
        count = kb.load_from_directory("/nonexistent/path")
        assert count == 0

    def test_load_json_documents(self, tmp_path):
        doc_data = {
            "content": "Custom domain knowledge about medical RAG.",
            "category": "domain",
        }
        (tmp_path / "medical.json").write_text(json.dumps(doc_data))

        kb = KnowledgeBase()
        count = kb.load_from_directory(str(tmp_path))
        assert count == 1
        assert kb.total_documents == 1

        results = kb.query("medical domain")
        assert len(results) > 0

    def test_load_txt_documents(self, tmp_path):
        (tmp_path / "notes.txt").write_text(
            "Always validate input before passing to LLM."
        )

        kb = KnowledgeBase()
        count = kb.load_from_directory(str(tmp_path))
        assert count == 1

        results = kb.query("validate input")
        assert len(results) > 0
        assert results[0].category == "domain"

    def test_skip_malformed_json(self, tmp_path):
        (tmp_path / "bad.json").write_text("{not valid json")
        (tmp_path / "good.txt").write_text("Valid document.")

        kb = KnowledgeBase()
        count = kb.load_from_directory(str(tmp_path))
        assert count == 1  # Only the txt loaded

    def test_load_from_configured_dir(self, tmp_path):
        (tmp_path / "data.txt").write_text("Some knowledge.")
        kb = KnowledgeBase(knowledge_dir=str(tmp_path))
        count = kb.load_from_directory()
        assert count == 1
