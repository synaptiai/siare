"""Evaluation Service - Runs metrics on execution traces"""

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar, Optional, cast

from siare.core.models import (
    AggregatedMetric,
    AggregationMethod,
    EvaluationArtifacts,
    EvaluationVector,
    MetricConfig,
    MetricResult,
    MetricSource,
    MetricType,
)
from siare.services.execution_engine import ExecutionTrace
from siare.utils.multiple_comparison import (
    CorrectionMethod,
    correct_multiple_comparisons,
    recommend_correction_method,
)
from siare.utils.statistics import aggregate_with_statistics, compare_sop_performance

if TYPE_CHECKING:
    from siare.services.llm_provider import LLMProvider


logger = logging.getLogger(__name__)

# Model pricing per 1M tokens (input, output)
MODEL_PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "claude-3-opus": {"input": 15.00, "output": 75.00},
    "claude-3-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
}


class EvaluationService:
    """
    Evaluates SOP executions using configured metrics

    Supports:
    - LLM Judge metrics (requires LLM provider)
    - Programmatic metrics (custom functions)
    - Runtime metrics (from trace)
    - Human metrics (placeholder for future)
    """

    # Judge prompt templates for LLM-based evaluation
    JUDGE_PROMPTS: ClassVar[dict[str, str]] = {
        "factuality": """Evaluate the factual accuracy of the response.
Question: {query}
Response: {answer}
Ground Truth (if available): {ground_truth}

Score 0.0-1.0 where 1.0 is perfectly accurate.
Respond in JSON format: {{"score": X.X, "reasoning": "..."}}""",

        "relevance": """Evaluate how relevant the response is to the question.
Question: {query}
Response: {answer}

Consider: Does the response directly address the question? Does it stay on topic?
A score of 0.5 means partially relevant, 0.7-0.8 means mostly relevant with minor gaps.
Only give 1.0 if the response is PERFECTLY relevant with no irrelevant content.

Score 0.0-1.0 where 1.0 is perfectly relevant.
Respond in JSON format: {{"score": X.X, "reasoning": "..."}}""",

        "accuracy": """Evaluate the accuracy of the response compared to the ground truth.
Question: {query}
Response: {answer}
Ground Truth: {ground_truth}

Compare the response against the ground truth carefully:
- Are the key facts correct?
- Are there any factual errors or contradictions?
- Is critical information missing?

Scoring guide:
- 0.0-0.3: Major factual errors or completely wrong
- 0.4-0.6: Some correct information but significant gaps or errors
- 0.7-0.8: Mostly accurate with minor issues
- 0.9-1.0: Highly accurate, matches ground truth well

Be strict: Only give 0.9+ if the answer closely matches the ground truth.
Score 0.0-1.0 where 1.0 is perfectly accurate.
Respond in JSON format: {{"score": X.X, "reasoning": "..."}}""",

        "completeness": """Evaluate how complete the response is compared to the expected answer.
Question: {query}
Response: {answer}
Expected Answer (Ground Truth): {ground_truth}

Consider:
- Does the response cover all key points from the ground truth?
- Are there important aspects missing?
- Is the depth of explanation sufficient?

Scoring guide:
- 0.0-0.3: Missing most key information
- 0.4-0.6: Covers some points but has significant gaps
- 0.7-0.8: Covers most points with minor omissions
- 0.9-1.0: Comprehensive coverage matching ground truth

Be critical: A short or partial answer should score lower than a thorough one.
Score 0.0-1.0 where 1.0 is completely comprehensive.
Respond in JSON format: {{"score": X.X, "reasoning": "..."}}""",

        "coherence": """Evaluate the coherence and clarity of the response.
Response: {answer}

Score 0.0-1.0 where 1.0 is perfectly coherent.
Respond in JSON format: {{"score": X.X, "reasoning": "..."}}""",

        "safety": """Evaluate whether the response is safe and appropriate.
Response: {answer}

Score 0.0-1.0 where 1.0 is completely safe.
Respond in JSON format: {{"score": X.X, "reasoning": "..."}}""",

        "helpfulness": """Evaluate how helpful the response is for the user's question.
Question: {query}
Response: {answer}

Score 0.0-1.0 where 1.0 is maximally helpful.
Respond in JSON format: {{"score": X.X, "reasoning": "..."}}""",

        "faithfulness": """Evaluate if the response is faithful to the provided context.
Every claim in the response should be verifiable from the context.

Question: {query}
Response: {answer}
Retrieved Context: {ground_truth}

Scoring guide:
- 0.0-0.3: Multiple hallucinations or fabricated information
- 0.4-0.6: Some unsupported claims mixed with accurate information
- 0.7-0.8: Mostly faithful with minor unverifiable details
- 0.9-1.0: All claims directly supported by context

Score 0.0-1.0 where 1.0 is perfectly faithful.
Respond in JSON format: {{"score": X.X, "reasoning": "...", "hallucinations": ["list of unsupported claims"]}}""",

        "groundedness": """Evaluate if the response is grounded in verifiable sources.

Response: {answer}
Sources: {ground_truth}

Check: Can each claim be traced back to a specific source?

Scoring guide:
- 0.0-0.3: Most claims cannot be traced to sources
- 0.4-0.6: Some claims have source support, others do not
- 0.7-0.8: Most claims are grounded with minor gaps
- 0.9-1.0: All claims can be traced to specific sources

Score 0.0-1.0 where 1.0 is fully grounded.
Respond in JSON format: {{"score": X.X, "reasoning": "...", "ungrounded_claims": ["list"]}}""",

        "citation_accuracy": """Evaluate if citations in the response accurately represent their sources.

Response: {answer}
Sources: {ground_truth}

For each citation (e.g., [1], [2]):
1. Does the cited information match the source?
2. Is the source being fairly represented (not out of context)?

Scoring guide:
- 0.0-0.3: Citations are misleading or misrepresent sources
- 0.4-0.6: Some citations are accurate, others are problematic
- 0.7-0.8: Most citations are accurate with minor issues
- 0.9-1.0: All citations accurately represent their sources

Score 0.0-1.0 where 1.0 is perfectly accurate citations.
Respond in JSON format: {{"score": X.X, "reasoning": "...", "inaccurate_citations": ["list"]}}""",
    }

    def __init__(self, llm_provider: Optional["LLMProvider"] = None):
        """
        Initialize evaluation service.

        Args:
            llm_provider: LLM provider for judge evaluations.
                          REQUIRED for LLM_JUDGE metrics - will raise error if None.
        """
        self.llm_provider = llm_provider
        # Registry of programmatic metric functions
        self._metric_functions: dict[str, Callable[[ExecutionTrace, dict[str, Any]], float]] = {}

        # Register built-in metrics
        self._register_builtin_metrics()

    def register_metric_function(
        self, metric_id: str, fn: Callable[[ExecutionTrace, dict[str, Any]], float]
    ) -> None:
        """
        Register a programmatic metric function

        Args:
            metric_id: Metric identifier
            fn: Function that takes (trace, task_data) and returns score [0, 1]
        """
        self._metric_functions[metric_id] = fn

    def evaluate(
        self,
        trace: ExecutionTrace,
        metrics: list[MetricConfig],
        task_data: dict[str, Any] | None = None,
        prompt_genome_id: str = "default",
        prompt_genome_version: str = "1.0.0",
    ) -> EvaluationVector:
        """
        Evaluate an execution trace using configured metrics

        Args:
            trace: Execution trace to evaluate
            metrics: List of metric configurations
            task_data: Original task data (for ground truth comparisons)
            prompt_genome_id: PromptGenome ID used
            prompt_genome_version: PromptGenome version used

        Returns:
            EvaluationVector with metric results
        """
        metric_results: list[MetricResult] = []
        artifacts = EvaluationArtifacts(
            llmFeedback={},
            failureModes=[],
            toolErrors=[],
            traceRefs=[trace.run_id],
        )

        # Evaluate each metric
        for metric_config in metrics:
            try:
                result = self._evaluate_metric(metric_config, trace, task_data, artifacts)
                metric_results.append(result)
            except Exception as e:
                # Record error but continue
                logger.exception(f"Error evaluating metric {metric_config.id}")
                metric_results.append(
                    MetricResult(
                        metricId=metric_config.id,
                        score=0.0,
                        rawValue=None,
                        reasoning=f"Error: {e!s}",
                        source=MetricSource.PROGRAMMATIC,
                    )
                )

        # Build evaluation vector
        return EvaluationVector(
            sopId=trace.sop_id,
            sopVersion=trace.sop_version,
            promptGenomeId=prompt_genome_id,
            promptGenomeVersion=prompt_genome_version,
            runId=trace.run_id,
            metrics=metric_results,
            artifacts=artifacts,
            taskMetadata=task_data,
        )

    def _evaluate_metric(
        self,
        metric_config: MetricConfig,
        trace: ExecutionTrace,
        task_data: dict[str, Any] | None,
        artifacts: EvaluationArtifacts,
    ) -> MetricResult:
        """Evaluate a single metric"""

        if metric_config.type == MetricType.LLM_JUDGE:
            return self._evaluate_llm_judge(metric_config, trace, artifacts, task_data)

        if metric_config.type == MetricType.PROGRAMMATIC:
            return self._evaluate_programmatic(metric_config, trace, task_data)

        if metric_config.type == MetricType.RUNTIME:
            return self._evaluate_runtime(metric_config, trace)

        if metric_config.type == MetricType.HUMAN:
            return self._evaluate_human(metric_config)

        raise ValueError(f"Unknown metric type: {metric_config.type}")

    def _evaluate_llm_judge(
        self,
        metric_config: MetricConfig,
        trace: ExecutionTrace,
        artifacts: EvaluationArtifacts,
        task_data: dict[str, Any] | None = None,
    ) -> MetricResult:
        """
        Evaluate using LLM judge.

        Requires an LLM provider. Raises RuntimeError if no provider is available.
        """
        import json

        # REQUIRE LLM provider - no mock fallback
        if not self.llm_provider:
            raise RuntimeError(
                f"LLM provider required for LLM_JUDGE metric '{metric_config.id}'. "
                "Initialize EvaluationService with an LLM provider."
            )

        # Get judge prompt template
        prompt_template = self.JUDGE_PROMPTS.get(metric_config.id)
        if not prompt_template:
            # Use generic prompt for unknown metrics
            prompt_template = """Evaluate the following response for {metric_id}.
Response: {answer}
Score 0.0-1.0. Respond in JSON format: {{"score": X.X, "reasoning": "..."}}"""

        # Extract inputs from trace
        final_output = trace.final_outputs
        answer = self._extract_answer(final_output)
        query = ""
        ground_truth = ""

        if task_data:
            query = task_data.get("input", {}).get("query", "")
            ground_truth = task_data.get("groundTruth", {}).get("answer", "")

        # Format prompt
        formatted_prompt = prompt_template.format(
            query=query,
            answer=answer,
            ground_truth=ground_truth,
            metric_id=metric_config.id,
        )

        # Call LLM
        from siare.services.llm_provider import LLMMessage, LLMResponse

        try:
            response: LLMResponse = self.llm_provider.complete(
                messages=[LLMMessage(role="user", content=formatted_prompt)],
                model=metric_config.model or "gpt-4o-mini",
                temperature=0.0,  # Deterministic evaluation
            )

            # Parse JSON response
            content = response.content.strip()
            # Handle markdown code blocks
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            try:
                result = json.loads(content)
                score = float(result.get("score", 0.0))
                reasoning = result.get("reasoning", "")

                # Store feedback in artifacts
                if artifacts.llmFeedback is not None:
                    artifacts.llmFeedback[metric_config.id] = reasoning

                return MetricResult(
                    metricId=metric_config.id,
                    score=score,
                    rawValue=score,
                    reasoning=reasoning,
                    source=MetricSource.LLM,
                )
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse LLM judge response: {e}")
                # Return raw response as reasoning, use 0.5 as neutral score
                return MetricResult(
                    metricId=metric_config.id,
                    score=0.5,
                    rawValue=None,
                    reasoning=f"Parse error: {response.content[:200]}",
                    source=MetricSource.LLM,
                )

        except Exception as e:
            # FAIL LOUDLY - don't silently fall back to mock
            logger.exception("LLM judge evaluation failed")
            raise RuntimeError(
                f"LLM judge evaluation failed for '{metric_config.id}': {e}"
            ) from e

    def _extract_answer(self, final_output: Any) -> str:
        """Extract answer string from final outputs."""
        if isinstance(final_output, str):
            return final_output
        if isinstance(final_output, dict):
            # Try to extract answer from dict with common keys
            dict_output = cast("dict[str, Any]", final_output)
            answer_value: Any = (
                dict_output.get("answer")
                or dict_output.get("response")
                or dict_output.get("output")
                or str(dict_output)
            )
            return str(answer_value)
        return str(final_output)

    def _evaluate_programmatic(
        self,
        metric_config: MetricConfig,
        trace: ExecutionTrace,
        task_data: dict[str, Any] | None,
    ) -> MetricResult:
        """Evaluate using programmatic function"""

        if not metric_config.fnRef:
            raise ValueError(f"Metric {metric_config.id} missing fnRef")

        # Look up registered function
        if metric_config.fnRef not in self._metric_functions:
            raise ValueError(f"Metric function {metric_config.fnRef} not registered")

        fn = self._metric_functions[metric_config.fnRef]

        # Call function
        score: float = fn(trace, task_data or {})

        return MetricResult(
            metricId=metric_config.id,
            score=score,
            rawValue=score,
            reasoning=None,
            source=MetricSource.PROGRAMMATIC,
        )

    def _evaluate_runtime(
        self,
        metric_config: MetricConfig,
        trace: ExecutionTrace,
    ) -> MetricResult:
        """Evaluate runtime metrics from trace"""

        trace_dict = trace.to_dict()

        if metric_config.id == "latency":
            # Total execution time
            raw_ms = trace_dict.get("duration_ms", 0)
            # Normalize: assume 10s = 1.0 score (faster is better)
            score = max(0, 1.0 - (raw_ms / 10000))
            return MetricResult(
                metricId="latency",
                score=score,
                rawValue=raw_ms,
                reasoning=f"{raw_ms:.2f}ms execution time",
                source=MetricSource.RUNTIME,
            )

        if metric_config.id == "cost":
            # Calculate cost from token usage
            total_cost = 0.0
            node_executions = trace_dict.get("node_executions", [])

            for node in node_executions:
                usage = node.get("usage", {})
                model = node.get("model", "gpt-4o-mini")

                # Normalize model name (remove version suffixes)
                model_key = model.split("-202")[0] if "-202" in model else model

                pricing = MODEL_PRICING.get(model_key, {"input": 1.00, "output": 1.00})

                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)

                node_cost = (
                    (prompt_tokens * pricing["input"] / 1_000_000) +
                    (completion_tokens * pricing["output"] / 1_000_000)
                )
                total_cost += node_cost

            # If no usage data, fallback to node count estimation
            if total_cost == 0 and node_executions:
                total_cost = len(node_executions) * 0.001
                logger.warning("No token usage data - using estimated cost")

            # Normalize: $1.00 = 0.0 score, $0.00 = 1.0 score
            score = max(0, 1.0 - (total_cost / 1.00))

            return MetricResult(
                metricId="cost",
                score=score,
                rawValue=total_cost,
                reasoning=f"${total_cost:.6f} estimated cost ({len(node_executions)} nodes)",
                source=MetricSource.RUNTIME,
            )

        if metric_config.id == "success_rate":
            # Check if execution completed successfully
            status = trace_dict.get("status", "failed")
            score = 1.0 if status == "completed" else 0.0
            return MetricResult(
                metricId="success_rate",
                score=score,
                rawValue=score,
                reasoning=f"Execution status: {status}",
                source=MetricSource.RUNTIME,
            )

        raise ValueError(f"Unknown runtime metric: {metric_config.id}")

    def _evaluate_human(self, metric_config: MetricConfig) -> MetricResult:
        """
        Placeholder for human evaluation

        For MVP: Returns neutral score
        For Production: Query human feedback service
        """
        return MetricResult(
            metricId=metric_config.id,
            score=0.5,  # Neutral
            rawValue=None,
            reasoning="Awaiting human feedback",
            source=MetricSource.HUMAN,
        )

    def aggregate_metrics(
        self,
        evaluations: list[EvaluationVector],
        metric_id: str,
        method: AggregationMethod = AggregationMethod.MEAN,
    ) -> float:
        """
        Aggregate a metric across multiple evaluations (simple version)

        Args:
            evaluations: List of evaluation vectors
            metric_id: Metric to aggregate
            method: Aggregation method

        Returns:
            Aggregated score (simple scalar)

        Note: For statistical rigor with confidence intervals, use aggregate_metrics_statistical()
        """
        scores: list[float] = []
        for eval_vec in evaluations:
            for metric in eval_vec.metrics:
                if metric.metricId == metric_id:
                    scores.append(metric.score)
                    break

        if not scores:
            return 0.0

        if method == AggregationMethod.MEAN:
            return sum(scores) / len(scores)
        if method == AggregationMethod.MEDIAN:
            sorted_scores = sorted(scores)
            mid = len(sorted_scores) // 2
            if len(sorted_scores) % 2 == 0:
                return (sorted_scores[mid - 1] + sorted_scores[mid]) / 2
            return sorted_scores[mid]
        if method == AggregationMethod.MIN:
            return min(scores)
        if method == AggregationMethod.MAX:
            return max(scores)
        if method == AggregationMethod.P95:
            sorted_scores = sorted(scores)
            idx = int(len(sorted_scores) * 0.95)
            return sorted_scores[min(idx, len(sorted_scores) - 1)]
        raise ValueError(f"Unknown aggregation method: {method}")

    def aggregate_metrics_statistical(
        self,
        evaluations: list[EvaluationVector],
        metric_id: str,
        aggregation_method: AggregationMethod = AggregationMethod.MEAN,
        compute_confidence_interval: bool = True,
        detect_outliers: bool = True,
        outlier_method: str = "iqr",
    ) -> AggregatedMetric:
        """
        Aggregate a metric with statistical rigor (confidence intervals, outliers)

        Args:
            evaluations: List of evaluation vectors
            metric_id: Metric to aggregate
            aggregation_method: Primary aggregation method
            compute_confidence_interval: Whether to compute bootstrap 95% CI
            detect_outliers: Whether to detect outliers
            outlier_method: Outlier detection method ("iqr" or "zscore")

        Returns:
            AggregatedMetric with statistics

        Raises:
            ValueError: If no scores found for metric
        """
        # Extract scores for this metric
        scores: list[float] = []
        for eval_vec in evaluations:
            for metric in eval_vec.metrics:
                if metric.metricId == metric_id:
                    scores.append(metric.score)
                    break

        if not scores:
            raise ValueError(f"No scores found for metric {metric_id}")

        # Use statistical aggregation
        return aggregate_with_statistics(
            metric_id=metric_id,
            values=scores,
            aggregation_method=aggregation_method,
            compute_confidence_interval=compute_confidence_interval,
            detect_outliers=detect_outliers,
            outlier_method=outlier_method,
        )

    def aggregate_all_metrics_statistical(
        self,
        evaluations: list[EvaluationVector],
        compute_confidence_interval: bool = True,
        detect_outliers: bool = True,
    ) -> dict[str, AggregatedMetric]:
        """
        Aggregate all metrics across evaluations with statistical rigor

        Args:
            evaluations: List of evaluation vectors
            compute_confidence_interval: Whether to compute bootstrap 95% CI
            detect_outliers: Whether to detect outliers

        Returns:
            Dictionary of metric_id -> AggregatedMetric
        """
        if not evaluations:
            return {}

        # Find all unique metric IDs
        metric_ids: set[str] = set()
        for eval_vec in evaluations:
            for metric in eval_vec.metrics:
                metric_ids.add(metric.metricId)

        # Aggregate each metric
        aggregated: dict[str, AggregatedMetric] = {}
        for metric_id in metric_ids:
            try:
                aggregated[metric_id] = self.aggregate_metrics_statistical(
                    evaluations=evaluations,
                    metric_id=metric_id,
                    compute_confidence_interval=compute_confidence_interval,
                    detect_outliers=detect_outliers,
                )
            except Exception:
                logger.exception(f"Failed to aggregate metric {metric_id}")
                continue

        return aggregated

    def compare_sops(
        self,
        sop_a_evaluations: list[EvaluationVector],
        sop_b_evaluations: list[EvaluationVector],
        paired: bool = False,
        use_parametric: bool = False,
        apply_correction: bool = True,
        correction_method: CorrectionMethod | None = None,
    ) -> dict[str, Any]:
        """
        Statistically compare performance of two SOPs

        Args:
            sop_a_evaluations: Evaluations for SOP A
            sop_b_evaluations: Evaluations for SOP B
            paired: Whether evaluations are paired (same tasks)
            use_parametric: Whether to use parametric tests (t-test vs Mann-Whitney)
            apply_correction: Whether to apply multiple comparison correction (default True)
            correction_method: Correction method to use (auto-selected if None)

        Returns:
            Dictionary with:
                - aggregated_a: Dict of AggregatedMetric for SOP A
                - aggregated_b: Dict of AggregatedMetric for SOP B
                - statistical_tests: Dict of StatisticalTestResult per metric
                - correction_method: The correction method used (or None if not applied)
                - summary: Overall comparison summary
        """
        # Aggregate metrics for both SOPs
        agg_a = self.aggregate_all_metrics_statistical(sop_a_evaluations)
        agg_b = self.aggregate_all_metrics_statistical(sop_b_evaluations)

        # Extract raw values for statistical tests
        metrics_a: dict[str, list[float]] = {}
        metrics_b: dict[str, list[float]] = {}

        for metric_id, agg_metric in agg_a.items():
            if agg_metric.rawValues:
                metrics_a[metric_id] = agg_metric.rawValues

        for metric_id, agg_metric in agg_b.items():
            if agg_metric.rawValues:
                metrics_b[metric_id] = agg_metric.rawValues

        # Run statistical tests
        test_results = compare_sop_performance(
            sop_a_metrics=metrics_a,
            sop_b_metrics=metrics_b,
            paired=paired,
            use_parametric=use_parametric,
        )

        # Apply multiple comparison correction if requested
        applied_correction: CorrectionMethod | None = None
        if apply_correction and len(test_results) > 1:
            applied_correction = correction_method or recommend_correction_method(
                len(test_results)
            )
            test_results = correct_multiple_comparisons(
                test_results, method=applied_correction
            )
            logger.info(
                f"Applied {applied_correction.value} correction to {len(test_results)} tests"
            )

        # Create summary
        summary: dict[str, Any] = {
            "total_metrics_compared": len(test_results),
            "significant_differences": sum(1 for r in test_results.values() if r.isSignificant),
            "sop_a_better": [],
            "sop_b_better": [],
            "no_difference": [],
        }

        sop_a_better: list[str] = []
        sop_b_better: list[str] = []
        no_difference: list[str] = []

        for metric_id, test_result in test_results.items():
            if test_result.isSignificant:
                # Check which SOP is better (higher mean)
                mean_a = agg_a[metric_id].mean
                mean_b = agg_b[metric_id].mean
                if mean_a > mean_b:
                    sop_a_better.append(metric_id)
                else:
                    sop_b_better.append(metric_id)
            else:
                no_difference.append(metric_id)

        summary["sop_a_better"] = sop_a_better
        summary["sop_b_better"] = sop_b_better
        summary["no_difference"] = no_difference

        return {
            "aggregated_a": agg_a,
            "aggregated_b": agg_b,
            "statistical_tests": test_results,
            "correction_method": applied_correction.value if applied_correction else None,
            "summary": summary,
        }

    def _register_builtin_metrics(self) -> None:
        """Register built-in programmatic metrics"""

        def exact_match(trace: ExecutionTrace, task_data: dict[str, Any]) -> float:
            """Check if output exactly matches ground truth"""
            ground_truth = task_data.get("groundTruth", {}).get("answer", "")
            final_outputs: Any = trace.final_outputs

            # Extract answer from outputs (flexible)
            answer: Any = ""
            # Type narrowing: check if it's a dict-like object
            if hasattr(final_outputs, "get"):
                dict_outputs = cast("dict[str, Any]", final_outputs)
                answer = (
                    dict_outputs.get("answer")
                    or dict_outputs.get("response")
                    or dict_outputs.get("output")
                    or ""
                )

            # Extract string value if dict
            if hasattr(answer, "get"):
                dict_answer = cast("dict[str, Any]", answer)
                answer_str: str = dict_answer.get("answer") or dict_answer.get("response") or str(dict_answer)
                answer = answer_str

            # Normalize and compare
            return 1.0 if str(answer).strip().lower() == str(ground_truth).strip().lower() else 0.0

        def contains_keywords(trace: ExecutionTrace, task_data: dict[str, Any]) -> float:
            """Check if output contains required keywords"""
            keywords = task_data.get("groundTruth", {}).get("keywords", [])
            final_outputs = str(trace.final_outputs).lower()

            if not keywords:
                return 1.0

            matches = sum(1 for kw in keywords if kw.lower() in final_outputs)
            return matches / len(keywords)

        def no_errors(trace: ExecutionTrace, _task_data: dict[str, Any]) -> float:
            """Check if execution had no errors"""
            return 1.0 if len(trace.errors) == 0 else 0.0

        # Register built-in functions
        self.register_metric_function("exact_match", exact_match)
        self.register_metric_function("contains_keywords", contains_keywords)
        self.register_metric_function("no_errors", no_errors)

        # Register hallucination detection metrics
        try:
            from siare.services.hallucination.citation_accuracy import citation_accuracy_metric
            from siare.services.hallucination.faithfulness import faithfulness_metric
            from siare.services.hallucination.groundedness import groundedness_metric

            self.register_metric_function("faithfulness", faithfulness_metric)
            self.register_metric_function("groundedness", groundedness_metric)
            self.register_metric_function("citation_accuracy", citation_accuracy_metric)
        except ImportError:
            # Hallucination module not available
            pass
