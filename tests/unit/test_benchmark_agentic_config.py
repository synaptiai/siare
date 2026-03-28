"""Tests for agentic variation config in benchmark runners."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from siare.core.models import AgenticVariationConfig


# ============================================================================
# Config Passthrough Tests
# ============================================================================


class TestSelfImprovementConfigAgentic:
    """Tests for agentic_config in SelfImprovementConfig."""

    def test_default_agentic_config_is_none(self):
        from siare.benchmarks.self_improvement_benchmark import (
            SelfImprovementConfig,
        )

        config = SelfImprovementConfig()
        assert config.agentic_config is None

    def test_agentic_config_accepts_variation_config(self):
        from siare.benchmarks.self_improvement_benchmark import (
            SelfImprovementConfig,
        )

        ac = AgenticVariationConfig(mode="adaptive")
        config = SelfImprovementConfig(agentic_config=ac)
        assert config.agentic_config is ac
        assert config.agentic_config.mode == "adaptive"

    def test_agentic_config_all_modes(self):
        from siare.benchmarks.self_improvement_benchmark import (
            SelfImprovementConfig,
        )

        for mode in ("single_turn", "agentic", "adaptive"):
            ac = AgenticVariationConfig(mode=mode)
            config = SelfImprovementConfig(agentic_config=ac)
            assert config.agentic_config.mode == mode


class TestEvolutionBenchmarkConfigAgentic:
    """Tests for agentic_config in EvolutionBenchmarkConfig."""

    def test_default_agentic_config_is_none(self):
        from siare.benchmarks.evolution_runner import (
            EvolutionBenchmarkConfig,
        )

        config = EvolutionBenchmarkConfig()
        assert config.agentic_config is None

    def test_agentic_config_accepts_variation_config(self):
        from siare.benchmarks.evolution_runner import (
            EvolutionBenchmarkConfig,
        )

        ac = AgenticVariationConfig(mode="agentic")
        config = EvolutionBenchmarkConfig(agentic_config=ac)
        assert config.agentic_config is ac


# ============================================================================
# GenerationSnapshot Agentic Stats Tests
# ============================================================================


class TestGenerationSnapshotAgenticStats:
    """Tests for agentic_stats field in GenerationSnapshot."""

    def test_default_agentic_stats_is_none(self):
        from siare.benchmarks.self_improvement_benchmark import (
            GenerationSnapshot,
        )

        snapshot = GenerationSnapshot(
            generation=0,
            best_quality=0.65,
            avg_quality=0.60,
            metrics={"accuracy": 0.65},
            prompt_changes=[],
        )
        assert snapshot.agentic_stats is None

    def test_agentic_stats_populated(self):
        from siare.benchmarks.self_improvement_benchmark import (
            GenerationSnapshot,
        )

        snapshot = GenerationSnapshot(
            generation=3,
            best_quality=0.72,
            avg_quality=0.68,
            metrics={"accuracy": 0.72},
            prompt_changes=[],
            agentic_stats={
                "used_agentic": True,
                "total_inner_iterations": 12,
                "agentic_offspring_count": 3,
            },
        )
        assert snapshot.agentic_stats is not None
        assert snapshot.agentic_stats["total_inner_iterations"] == 12


# ============================================================================
# SelfImprovementResult Summary Tests
# ============================================================================


class TestSelfImprovementResultAgenticSummary:
    """Tests for agentic data in SelfImprovementResult.summary()."""

    def _make_result(self, agentic_config=None, snapshots=None):
        from siare.benchmarks.self_improvement_benchmark import (
            GenerationSnapshot,
            SelfImprovementConfig,
            SelfImprovementResult,
        )
        from siare.core.models import StatisticalTestResult

        config = SelfImprovementConfig(agentic_config=agentic_config)
        return SelfImprovementResult(
            config=config,
            dataset_name="test",
            initial_prompts={},
            evolved_prompts={},
            prompt_diffs={},
            generation_snapshots=snapshots or [],
            generations_run=5,
            initial_metrics={"accuracy": 0.5},
            evolved_metrics={"accuracy": 0.7},
            significance_tests={
                "accuracy": StatisticalTestResult(
                    testType="wilcoxon",
                    statistic=10.0,
                    pValue=0.01,
                    isSignificant=True,
                    effectSize=0.3,
                    hypothesis="accuracy improved",
                ),
            },
        )

    def test_summary_without_agentic(self):
        result = self._make_result()
        summary = result.summary()
        assert summary["agentic"] is None

    def test_summary_with_agentic(self):
        from siare.benchmarks.self_improvement_benchmark import (
            GenerationSnapshot,
        )

        ac = AgenticVariationConfig(mode="adaptive")
        snapshots = [
            GenerationSnapshot(
                generation=0, best_quality=0.5, avg_quality=0.5,
                metrics={}, prompt_changes=[],
                agentic_stats=None,
            ),
            GenerationSnapshot(
                generation=1, best_quality=0.6, avg_quality=0.55,
                metrics={}, prompt_changes=[],
                agentic_stats={
                    "total_inner_iterations": 8,
                },
            ),
            GenerationSnapshot(
                generation=2, best_quality=0.65, avg_quality=0.6,
                metrics={}, prompt_changes=[],
                agentic_stats={
                    "total_inner_iterations": 5,
                },
            ),
        ]
        result = self._make_result(agentic_config=ac, snapshots=snapshots)
        summary = result.summary()

        assert summary["agentic"] is not None
        assert summary["agentic"]["mode"] == "adaptive"
        assert summary["agentic"]["generations_using_agentic"] == 2
        assert summary["agentic"]["total_inner_iterations"] == 13


# ============================================================================
# Comparison Report Tests
# ============================================================================


class TestAgenticComparisonReport:
    """Tests for AgenticComparisonReport."""

    def _make_results(self):
        return {
            "single_turn": {
                "dataset": "BEIR",
                "model": "gpt-4o-mini",
                "generations": 10,
                "improvements": {
                    "benchmark_accuracy": {
                        "initial": 0.55,
                        "evolved": 0.67,
                        "improvement": 0.12,
                        "improvement_pct": "21.8%",
                        "significant": True,
                        "p_value": 0.003,
                    },
                },
                "converged": True,
                "convergence_generation": 8,
                "total_time_seconds": 120.0,
                "agentic": None,
            },
            "agentic": {
                "dataset": "BEIR",
                "model": "gpt-4o-mini",
                "generations": 10,
                "improvements": {
                    "benchmark_accuracy": {
                        "initial": 0.55,
                        "evolved": 0.72,
                        "improvement": 0.17,
                        "improvement_pct": "30.9%",
                        "significant": True,
                        "p_value": 0.001,
                    },
                },
                "converged": True,
                "convergence_generation": 6,
                "total_time_seconds": 340.0,
                "agentic": {
                    "mode": "agentic",
                    "generations_using_agentic": 10,
                    "total_inner_iterations": 45,
                },
            },
        }

    def test_generate_markdown(self):
        from siare.benchmarks.reports.agentic_comparison_report import (
            AgenticComparisonReport,
        )

        report = AgenticComparisonReport(
            results=self._make_results(),
            mode_configs={
                "single_turn": {"mode": "single_turn"},
                "agentic": {"mode": "agentic"},
            },
        )
        md = report.generate_markdown()
        assert "# Agentic Variation Mode Comparison" in md
        assert "single_turn" in md
        assert "agentic" in md
        assert "accuracy" in md

    def test_generate_json(self):
        import json

        from siare.benchmarks.reports.agentic_comparison_report import (
            AgenticComparisonReport,
        )

        report = AgenticComparisonReport(
            results=self._make_results(),
            mode_configs={},
        )
        data = json.loads(report.generate_json())
        assert "single_turn" in data["results"]
        assert "agentic" in data["results"]

    def test_save_creates_files(self, tmp_path):
        from siare.benchmarks.reports.agentic_comparison_report import (
            AgenticComparisonReport,
        )

        report = AgenticComparisonReport(
            results=self._make_results(),
            mode_configs={},
        )
        report.save(str(tmp_path))

        md_files = list(tmp_path.glob("*.md"))
        json_files = list(tmp_path.glob("*.json"))
        assert len(md_files) == 1
        assert len(json_files) == 1


# ============================================================================
# Report Rendering Tests
# ============================================================================


class TestSelfImprovementReportAgentic:
    """Tests for agentic mode in self-improvement reports."""

    def test_metadata_includes_variation_mode(self):
        from siare.benchmarks.reports.self_improvement_report import (
            SelfImprovementReport,
        )
        from siare.benchmarks.self_improvement_benchmark import (
            SelfImprovementConfig,
            SelfImprovementResult,
        )
        from siare.core.models import StatisticalTestResult

        ac = AgenticVariationConfig(mode="adaptive")
        config = SelfImprovementConfig(agentic_config=ac)
        result = SelfImprovementResult(
            config=config,
            dataset_name="test",
            initial_prompts={"role1": "prompt1"},
            evolved_prompts={"role1": "prompt2"},
            prompt_diffs={"role1": {"added": 1, "removed": 0}},
            generation_snapshots=[],
            generations_run=5,
            initial_metrics={"accuracy": 0.5},
            evolved_metrics={"accuracy": 0.7},
            significance_tests={
                "accuracy": StatisticalTestResult(
                    testType="wilcoxon",
                    statistic=10.0,
                    pValue=0.01,
                    isSignificant=True,
                    effectSize=0.3,
                    hypothesis="accuracy improved",
                ),
            },
        )

        report = SelfImprovementReport(result)
        md = report.generate_markdown()
        assert "adaptive" in md
        assert "Variation Mode" in md

    def test_configuration_includes_agentic_fields(self):
        from siare.benchmarks.reports.self_improvement_report import (
            SelfImprovementReport,
        )
        from siare.benchmarks.self_improvement_benchmark import (
            SelfImprovementConfig,
            SelfImprovementResult,
        )
        from siare.core.models import StatisticalTestResult

        ac = AgenticVariationConfig(
            mode="agentic",
            maxInnerIterations=10,
            agentModel="gpt-4o",
        )
        config = SelfImprovementConfig(agentic_config=ac)
        result = SelfImprovementResult(
            config=config,
            dataset_name="test",
            initial_prompts={},
            evolved_prompts={},
            prompt_diffs={},
            generation_snapshots=[],
            generations_run=5,
            initial_metrics={"accuracy": 0.5},
            evolved_metrics={"accuracy": 0.7},
            significance_tests={
                "accuracy": StatisticalTestResult(
                    testType="wilcoxon",
                    statistic=10.0,
                    pValue=0.01,
                    isSignificant=True,
                    effectSize=0.3,
                    hypothesis="accuracy improved",
                ),
            },
        )

        report = SelfImprovementReport(result)
        md = report.generate_markdown()
        assert "Max Inner Iterations" in md
        assert "10" in md
        assert "gpt-4o" in md
