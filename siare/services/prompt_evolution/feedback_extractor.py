"""Extract structured feedback artifacts from evaluation results."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import ClassVar

from siare.core.models import (
    EvaluationVector,
    FailurePattern,
    FeedbackArtifact,
    MetricSource,
)

logger = logging.getLogger(__name__)


class BaseFeedbackExtractor(ABC):
    """Abstract base class for feedback extraction."""

    @abstractmethod
    def extract(
        self,
        evaluations: list[EvaluationVector],
        role_id: str,
    ) -> list[FeedbackArtifact]:
        """Extract feedback artifacts from evaluations.

        Args:
            evaluations: List of evaluation results.
            role_id: Role to extract feedback for.

        Returns:
            List of FeedbackArtifact objects sorted by severity.
        """


class FeedbackArtifactExtractor(BaseFeedbackExtractor):
    """Extracts structured feedback from evaluation results.

    Sources:
    1. Low metric scores (< threshold) with LLM reasoning
    2. Classified failure modes from EvaluationArtifacts
    3. Tool errors from artifacts
    """

    # Score threshold below which we extract feedback
    LOW_SCORE_THRESHOLD = 0.5

    # Map failure mode patterns to FailurePattern enum
    PATTERN_MAPPING: ClassVar[dict[str, FailurePattern]] = {
        "hallucination": FailurePattern.HALLUCINATION,
        "incomplete": FailurePattern.INCOMPLETE,
        "irrelevant": FailurePattern.IRRELEVANT,
        "timeout": FailurePattern.TIMEOUT,
        "tool": FailurePattern.TOOL_MISUSE,
        "format": FailurePattern.FORMAT_ERROR,
        "reasoning": FailurePattern.REASONING_ERROR,
        "context": FailurePattern.CONTEXT_LOSS,
        "safety": FailurePattern.SAFETY_VIOLATION,
    }

    # Severity mapping for failure patterns
    SEVERITY_MAPPING: ClassVar[dict[FailurePattern, float]] = {
        FailurePattern.SAFETY_VIOLATION: 1.0,
        FailurePattern.HALLUCINATION: 0.9,
        FailurePattern.REASONING_ERROR: 0.8,
        FailurePattern.TOOL_MISUSE: 0.7,
        FailurePattern.CONTEXT_LOSS: 0.6,
        FailurePattern.INCOMPLETE: 0.5,
        FailurePattern.IRRELEVANT: 0.5,
        FailurePattern.FORMAT_ERROR: 0.4,
        FailurePattern.TIMEOUT: 0.3,
    }

    def __init__(self, score_threshold: float = 0.5) -> None:
        """Initialize extractor.

        Args:
            score_threshold: Scores below this trigger feedback extraction.
        """
        self.score_threshold = score_threshold

    def extract(
        self,
        evaluations: list[EvaluationVector],
        role_id: str,
    ) -> list[FeedbackArtifact]:
        """Extract feedback artifacts from evaluations."""
        artifacts: list[FeedbackArtifact] = []

        # 1. Extract from low-score metrics
        artifacts.extend(self._extract_from_metrics(evaluations, role_id))

        # 2. Extract from failure modes
        artifacts.extend(self._extract_from_failure_modes(evaluations, role_id))

        # 3. Extract from tool errors
        artifacts.extend(self._extract_from_tool_errors(evaluations, role_id))

        # 4. Deduplicate and rank
        return self._deduplicate_and_rank(artifacts)

    def _extract_from_metrics(
        self,
        evaluations: list[EvaluationVector],
        role_id: str,
    ) -> list[FeedbackArtifact]:
        """Extract feedback from metrics with low scores."""
        artifacts: list[FeedbackArtifact] = []

        for eval_vec in evaluations:
            for metric in eval_vec.metrics:
                if metric.score < self.score_threshold and metric.reasoning:
                    # Calculate severity based on how low the score is
                    severity = 1.0 - metric.score  # Lower score = higher severity

                    artifact = FeedbackArtifact(
                        source_type=(
                            "llm_judge"
                            if metric.source == MetricSource.LLM
                            else "metric"
                        ),
                        role_id=role_id,
                        metric_id=metric.metricId,
                        critique=metric.reasoning,
                        severity=severity,
                        failure_pattern=self._infer_failure_pattern(metric.reasoning),
                        suggested_fix=self._generate_suggestion(metric.reasoning),
                        trace_ref=eval_vec.runId,
                    )
                    artifacts.append(artifact)

        return artifacts

    def _extract_from_failure_modes(
        self,
        evaluations: list[EvaluationVector],
        role_id: str,
    ) -> list[FeedbackArtifact]:
        """Extract feedback from classified failure modes."""
        artifacts: list[FeedbackArtifact] = []

        for eval_vec in evaluations:
            if eval_vec.artifacts and eval_vec.artifacts.failureModes:
                for fm_str in eval_vec.artifacts.failureModes:
                    pattern = self._map_pattern(fm_str)
                    severity = self.SEVERITY_MAPPING.get(pattern, 0.5) if pattern else 0.5

                    trace_ref = None
                    if eval_vec.artifacts.traceRefs:
                        trace_ref = eval_vec.artifacts.traceRefs[0]

                    artifact = FeedbackArtifact(
                        source_type="failure_mode",
                        role_id=role_id,
                        critique=f"Failure pattern detected: {fm_str}",
                        severity=severity,
                        failure_pattern=pattern,
                        suggested_fix=self._get_fix_for_pattern(pattern),
                        trace_ref=trace_ref,
                    )
                    artifacts.append(artifact)

        return artifacts

    def _extract_from_tool_errors(
        self,
        evaluations: list[EvaluationVector],
        role_id: str,
    ) -> list[FeedbackArtifact]:
        """Extract feedback from tool errors."""
        artifacts: list[FeedbackArtifact] = []

        for eval_vec in evaluations:
            if eval_vec.artifacts and eval_vec.artifacts.toolErrors:
                for error in eval_vec.artifacts.toolErrors:
                    # Tool errors are always TOOL_MISUSE by default
                    # Only override for safety violations
                    pattern = FailurePattern.TOOL_MISUSE
                    error_lower = error.lower()
                    if "safety" in error_lower or "policy" in error_lower:
                        pattern = FailurePattern.SAFETY_VIOLATION

                    artifact = FeedbackArtifact(
                        source_type="tool_error",
                        role_id=role_id,
                        critique=f"Tool error: {error}",
                        severity=0.7,
                        failure_pattern=pattern,
                        trace_ref=eval_vec.runId,
                    )
                    artifacts.append(artifact)

        return artifacts

    def _deduplicate_and_rank(
        self,
        artifacts: list[FeedbackArtifact],
    ) -> list[FeedbackArtifact]:
        """Remove duplicates and sort by severity."""
        # Group by critique content (simple deduplication)
        seen_critiques: dict[str, FeedbackArtifact] = {}

        for artifact in artifacts:
            # Normalize critique for comparison
            key = artifact.critique.lower().strip()[:100]

            if key not in seen_critiques or artifact.severity > seen_critiques[key].severity:
                seen_critiques[key] = artifact

        # Sort by severity descending
        return sorted(seen_critiques.values(), key=lambda a: a.severity, reverse=True)

    def _map_pattern(self, pattern_str: str) -> FailurePattern | None:
        """Map string pattern to FailurePattern enum."""
        pattern_lower = pattern_str.lower()
        for key, value in self.PATTERN_MAPPING.items():
            if key in pattern_lower:
                return value
        return None

    def _infer_failure_pattern(self, reasoning: str) -> FailurePattern | None:
        """Infer failure pattern from LLM reasoning text."""
        reasoning_lower = reasoning.lower()

        if any(
            word in reasoning_lower
            for word in ["incorrect", "wrong", "factual", "hallucin"]
        ):
            return FailurePattern.HALLUCINATION
        if any(word in reasoning_lower for word in ["incomplete", "missing", "lack"]):
            return FailurePattern.INCOMPLETE
        if any(
            word in reasoning_lower for word in ["irrelevant", "off-topic", "unrelated"]
        ):
            return FailurePattern.IRRELEVANT
        if any(
            word in reasoning_lower for word in ["format", "structure", "syntax"]
        ):
            return FailurePattern.FORMAT_ERROR
        if any(
            word in reasoning_lower for word in ["reasoning", "logic", "argument"]
        ):
            return FailurePattern.REASONING_ERROR

        return None

    def _generate_suggestion(self, reasoning: str) -> str | None:
        """Generate improvement suggestion from reasoning."""
        # Simple heuristic - extract actionable keywords
        reasoning_lower = reasoning.lower()

        if "specificity" in reasoning_lower or "specific" in reasoning_lower:
            return "Add more specific details and examples to the instructions."
        if "incomplete" in reasoning_lower:
            return "Ensure all required aspects are addressed in the response."
        if "incorrect" in reasoning_lower or "wrong" in reasoning_lower:
            return "Add verification steps to prevent factual errors."

        return None

    def _get_fix_for_pattern(self, pattern: FailurePattern | None) -> str | None:
        """Get standard fix suggestion for failure pattern."""
        if pattern is None:
            return None

        fixes: dict[FailurePattern, str] = {
            FailurePattern.HALLUCINATION: "Add grounding instructions to verify facts against provided context.",
            FailurePattern.INCOMPLETE: "Add explicit checklist of required response components.",
            FailurePattern.IRRELEVANT: "Strengthen focus instructions and add relevance criteria.",
            FailurePattern.TIMEOUT: "Simplify complex operations or add chunking strategy.",
            FailurePattern.TOOL_MISUSE: "Add clearer tool usage guidelines with examples.",
            FailurePattern.FORMAT_ERROR: "Add explicit format template with examples.",
            FailurePattern.REASONING_ERROR: "Add step-by-step reasoning requirements.",
            FailurePattern.CONTEXT_LOSS: "Add context summarization requirements.",
            FailurePattern.SAFETY_VIOLATION: "Strengthen safety guidelines and add explicit constraints.",
        }

        return fixes.get(pattern)
