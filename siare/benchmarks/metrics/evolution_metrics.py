"""Benchmark-specific metrics for evolution integration.

These metrics are designed to work with SIARE's EvaluationService and
evolution loop, providing accuracy and quality metrics for RAG benchmarks.
"""
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from siare.core.models import ProcessConfig, PromptGenome, SOPGene
    from siare.services.evaluation_service import EvaluationService
    from siare.services.execution_engine import ExecutionTrace
    from siare.services.gene_pool import GenePool
    from siare.services.llm_provider import LLMProvider


def normalize_text(text: str) -> str:
    """Normalize text for comparison.

    Args:
        text: Raw text to normalize

    Returns:
        Normalized lowercase text with extra whitespace removed
    """
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    # Remove punctuation for more lenient matching
    return re.sub(r"[^\w\s]", "", text)


def extract_generated_answer(trace: "ExecutionTrace") -> str:
    """Extract generated answer from execution trace.

    Args:
        trace: ExecutionTrace with final outputs

    Returns:
        Generated answer string
    """
    # Look through role outputs
    for role_outputs in trace.final_outputs.values():
        if isinstance(role_outputs, dict):
            answer = role_outputs.get("answer", role_outputs.get("response", ""))
            if answer:
                return str(answer)
    # Fallback to top-level
    return str(
        trace.final_outputs.get("answer", trace.final_outputs.get("response", ""))
    )


def benchmark_accuracy(trace: "ExecutionTrace", task_data: dict[str, Any]) -> float:
    """Accuracy metric for benchmarks with partial match support.

    Compares normalized generated answer with normalized ground truth.
    Supports both exact match and partial match (ground truth in generated answer).
    This aligns with rag_runner.py accuracy calculation.

    Args:
        trace: ExecutionTrace from running the SOP
        task_data: Task data containing groundTruth

    Returns:
        1.0 if exact match or ground truth found in generated answer, 0.0 otherwise
    """
    ground_truth = task_data.get("groundTruth", {}).get("answer", "")
    generated = extract_generated_answer(trace)

    if not ground_truth or not generated:
        return 0.0

    normalized_truth = normalize_text(ground_truth)
    normalized_generated = normalize_text(generated)

    # Exact match
    if normalized_truth == normalized_generated:
        return 1.0
    # Partial match: ground truth is contained in the generated answer
    if normalized_truth in normalized_generated:
        return 1.0
    return 0.0


def benchmark_partial_match(
    trace: "ExecutionTrace", task_data: dict[str, Any]
) -> float:
    """Partial match accuracy for benchmarks.

    Returns 1.0 if ground truth is contained in generated answer,
    or if generated answer is contained in ground truth.

    Args:
        trace: ExecutionTrace from running the SOP
        task_data: Task data containing groundTruth

    Returns:
        1.0 if partial match found, 0.0 otherwise
    """
    ground_truth = task_data.get("groundTruth", {}).get("answer", "")
    generated = extract_generated_answer(trace)

    if not ground_truth or not generated:
        return 0.0

    normalized_truth = normalize_text(ground_truth)
    normalized_generated = normalize_text(generated)

    # Check both directions for containment
    if normalized_truth in normalized_generated:
        return 1.0
    if normalized_generated in normalized_truth:
        return 1.0

    return 0.0


def _tokenize(text: str) -> set[str]:
    """Tokenize text into word set.

    Args:
        text: Text to tokenize

    Returns:
        Set of normalized word tokens
    """
    normalized = normalize_text(text)
    return set(normalized.split())


def benchmark_f1(trace: "ExecutionTrace", task_data: dict[str, Any]) -> float:
    """Token-level F1 score for benchmarks.

    Computes precision and recall based on word overlap between
    generated answer and ground truth.

    Args:
        trace: ExecutionTrace from running the SOP
        task_data: Task data containing groundTruth

    Returns:
        F1 score between 0.0 and 1.0
    """
    ground_truth = task_data.get("groundTruth", {}).get("answer", "")
    generated = extract_generated_answer(trace)

    if not ground_truth or not generated:
        return 0.0

    truth_tokens = _tokenize(ground_truth)
    generated_tokens = _tokenize(generated)

    if not truth_tokens or not generated_tokens:
        return 0.0

    # Calculate overlap
    common_tokens = truth_tokens & generated_tokens

    if not common_tokens:
        return 0.0

    precision = len(common_tokens) / len(generated_tokens)
    recall = len(common_tokens) / len(truth_tokens)

    # F1 score
    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)


def register_benchmark_metrics(evaluation_service: "EvaluationService") -> None:
    """Register all benchmark metrics with EvaluationService.

    Args:
        evaluation_service: EvaluationService instance to register metrics with
    """
    evaluation_service.register_metric_function("benchmark_accuracy", benchmark_accuracy)
    evaluation_service.register_metric_function(
        "benchmark_partial_match", benchmark_partial_match
    )
    evaluation_service.register_metric_function("benchmark_f1", benchmark_f1)


class EvolutionMetrics:
    """Metrics for evaluating evolutionary optimization processes.

    Provides methods to measure optimization efficiency, sample efficiency,
    diversity maintenance, and stability across runs. These metrics are
    essential for publication-grade benchmarks that compare evolved SOPs
    against baselines.

    Example:
        >>> # Measure how many generations to reach 80% accuracy
        >>> gens_to_target = EvolutionMetrics.optimization_efficiency(
        ...     evolution_history, target_performance=0.8
        ... )
        >>> # Measure diversity in the current gene pool
        >>> diversity = EvolutionMetrics.diversity_maintenance(gene_pool)
    """

    @staticmethod
    def optimization_efficiency(
        evolution_history: list["SOPGene"],
        target_performance: float,
        metric_name: str = "benchmark_accuracy",
    ) -> int:
        """Count generations needed to reach target performance.

        Iterates through evolution history to find the first generation
        where performance meets or exceeds the target threshold.

        Args:
            evolution_history: List of SOPGene objects representing evolution,
                sorted by generation (oldest first).
            target_performance: Target metric value to achieve (0-1 range typical).
            metric_name: Metric ID to track (default: benchmark_accuracy).

        Returns:
            Number of generations to reach target (1-indexed), or -1 if never reached.

        Example:
            >>> history = [gene_gen0, gene_gen1, gene_gen2, gene_gen3]
            >>> gens = EvolutionMetrics.optimization_efficiency(history, 0.85)
            >>> print(f"Reached 85% accuracy at generation {gens}")
        """
        if not evolution_history:
            return -1

        for gene in evolution_history:
            metric_value = gene.get_metric_mean(metric_name)
            if metric_value >= target_performance:
                # Return generation number (1-indexed for human readability)
                generation = gene.generation if gene.generation is not None else 0
                return generation + 1

        return -1

    @staticmethod
    def sample_efficiency(
        evolution_history: list["SOPGene"],
        metric_name: str = "benchmark_accuracy",
    ) -> dict[str, int]:
        """Calculate sample efficiency milestones for evolution.

        Tracks how many generations are needed to reach different performance
        percentiles relative to the best achieved performance.

        Args:
            evolution_history: List of SOPGene objects representing evolution,
                sorted by generation (oldest first).
            metric_name: Metric ID to track (default: benchmark_accuracy).

        Returns:
            Dictionary with generation counts:
            - to_50th_percentile: Generations to 50% of best performance
            - to_90th_percentile: Generations to 90% of best performance
            - to_best: Generations to reach best performance
            Values are -1 if threshold never reached.

        Example:
            >>> efficiency = EvolutionMetrics.sample_efficiency(history)
            >>> print(f"50% reached at gen {efficiency['to_50th_percentile']}")
        """
        result: dict[str, int] = {
            "to_50th_percentile": -1,
            "to_90th_percentile": -1,
            "to_best": -1,
        }

        if not evolution_history:
            return result

        # Find best performance achieved
        best_performance = max(
            gene.get_metric_mean(metric_name) for gene in evolution_history
        )

        if best_performance <= 0:
            return result

        # Calculate thresholds
        threshold_50 = best_performance * 0.5
        threshold_90 = best_performance * 0.9

        # Track when each threshold is first crossed
        for gene in evolution_history:
            metric_value = gene.get_metric_mean(metric_name)
            generation = (gene.generation if gene.generation is not None else 0) + 1

            if result["to_50th_percentile"] == -1 and metric_value >= threshold_50:
                result["to_50th_percentile"] = generation

            if result["to_90th_percentile"] == -1 and metric_value >= threshold_90:
                result["to_90th_percentile"] = generation

            if result["to_best"] == -1 and metric_value >= best_performance:
                result["to_best"] = generation

        return result

    @staticmethod
    def diversity_maintenance(gene_pool: "GenePool") -> dict[str, Any]:
        """Calculate diversity metrics for a gene pool.

        Measures how well the evolutionary process maintains diversity through:
        - Number of unique SOPs (distinct configurations)
        - Pareto frontier size (non-dominated solutions)
        - Average edit distance between SOPs (structural diversity)

        Args:
            gene_pool: GenePool instance containing evolved SOPs.

        Returns:
            Dictionary with diversity metrics:
            - unique_sops: Count of unique SOP configurations
            - pareto_frontier_size: Count of Pareto-optimal SOPs
            - avg_edit_distance: Average pairwise edit distance (0 if <2 SOPs)

        Example:
            >>> diversity = EvolutionMetrics.diversity_maintenance(gene_pool)
            >>> print(f"Pool has {diversity['unique_sops']} unique SOPs")
        """
        all_genes = gene_pool.list_sop_genes()

        # Count unique SOPs (by ID)
        unique_sop_ids = {gene.sopId for gene in all_genes}
        unique_count = len(unique_sop_ids)

        # Count Pareto-optimal SOPs
        pareto_count = sum(
            1 for gene in all_genes
            if gene.frontierFlags and gene.frontierFlags.isParetoOptimal
        )

        # Calculate average edit distance between SOP configs
        avg_distance = 0.0
        min_genes_for_comparison = 2
        if len(all_genes) >= min_genes_for_comparison:
            total_distance = 0
            pair_count = 0

            # Sample pairs if too many genes (avoid O(n^2) explosion)
            genes_to_compare = all_genes[:100]  # Cap at 100 for performance

            for i, gene_a in enumerate(genes_to_compare):
                for gene_b in genes_to_compare[i + 1:]:
                    distance = EvolutionMetrics._sop_edit_distance(
                        gene_a.configSnapshot, gene_b.configSnapshot
                    )
                    total_distance += distance
                    pair_count += 1

            if pair_count > 0:
                avg_distance = total_distance / pair_count

        return {
            "unique_sops": unique_count,
            "pareto_frontier_size": pareto_count,
            "avg_edit_distance": avg_distance,
        }

    @staticmethod
    def stability_across_runs(
        sop: "ProcessConfig",
        genome: "PromptGenome",
        queries: list[dict[str, Any]],
        llm_provider: "LLMProvider",
        n_runs: int = 10,
        tool_adapters: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        """Measure stability by running same queries multiple times.

        Executes the SOP multiple times on the same queries to measure
        variance in outputs. High variance indicates instability.

        Args:
            sop: ProcessConfig to evaluate.
            genome: PromptGenome with prompts for roles.
            queries: List of query dictionaries with 'input' and optional 'groundTruth'.
            llm_provider: LLM provider for execution.
            n_runs: Number of evaluation runs (default: 10).
            tool_adapters: Optional tool adapters for execution.

        Returns:
            Dictionary with stability statistics:
            - mean: Average accuracy across runs
            - std: Standard deviation of accuracy
            - cv: Coefficient of variation (std/mean, 0 if mean is 0)
            - min: Minimum accuracy observed
            - max: Maximum accuracy observed

        Example:
            >>> stability = EvolutionMetrics.stability_across_runs(
            ...     sop, genome, queries, provider, n_runs=30
            ... )
            >>> print(f"CV: {stability['cv']:.3f}")  # Lower is more stable
        """
        from siare.services.execution_engine import ExecutionEngine

        run_accuracies: list[float] = []

        for _ in range(n_runs):
            run_correct = 0
            run_total = 0

            for query in queries:
                engine = ExecutionEngine(
                    llm_provider=llm_provider,
                    tool_adapters=tool_adapters or {},
                )

                task_input = query.get("input", query)

                try:
                    trace = engine.execute(
                        sop=sop,
                        prompt_genome=genome,
                        task_input=task_input,
                    )

                    # Calculate accuracy for this sample
                    ground_truth = query.get("groundTruth", {}).get("answer", "")
                    generated = extract_generated_answer(trace)

                    if ground_truth and generated:
                        normalized_truth = normalize_text(ground_truth)
                        normalized_generated = normalize_text(generated)

                        if (normalized_truth == normalized_generated or
                                normalized_truth in normalized_generated):
                            run_correct += 1

                    run_total += 1

                except (ValueError, RuntimeError, KeyError, TimeoutError, OSError):
                    # Count execution failures as incorrect answers
                    run_total += 1

            # Calculate accuracy for this run
            run_accuracy = run_correct / run_total if run_total > 0 else 0.0
            run_accuracies.append(run_accuracy)

        # Calculate statistics
        if not run_accuracies:
            return {"mean": 0.0, "std": 0.0, "cv": 0.0, "min": 0.0, "max": 0.0}

        mean_acc = sum(run_accuracies) / len(run_accuracies)

        # Standard deviation
        variance = sum((x - mean_acc) ** 2 for x in run_accuracies) / len(run_accuracies)
        std_acc = variance ** 0.5

        # Coefficient of variation (relative stability measure)
        cv = std_acc / mean_acc if mean_acc > 0 else 0.0

        return {
            "mean": mean_acc,
            "std": std_acc,
            "cv": cv,
            "min": min(run_accuracies),
            "max": max(run_accuracies),
        }

    @staticmethod
    def _sop_edit_distance(
        sop1: "ProcessConfig",
        sop2: "ProcessConfig",
    ) -> int:
        """Calculate edit distance between two SOP configurations.

        Edit distance measures structural difference as the sum of:
        - Role differences (roles in one but not the other)
        - Prompt differences (different prompt references for same roles)
        - Graph edge differences (edges in one but not the other)

        Args:
            sop1: First ProcessConfig.
            sop2: Second ProcessConfig.

        Returns:
            Edit distance (minimum edits to transform sop1 to sop2).
            Higher values indicate more different configurations.

        Example:
            >>> dist = EvolutionMetrics._sop_edit_distance(sop_a, sop_b)
            >>> print(f"SOPs differ by {dist} edits")
        """
        distance = 0

        # Compare roles
        roles1 = {role.id: role for role in sop1.roles}
        roles2 = {role.id: role for role in sop2.roles}

        all_role_ids = set(roles1.keys()) | set(roles2.keys())

        for role_id in all_role_ids:
            if role_id not in roles1:
                # Role added in sop2
                distance += 1
            elif role_id not in roles2:
                # Role removed in sop2
                distance += 1
            else:
                # Role exists in both - check for prompt differences
                role1 = roles1[role_id]
                role2 = roles2[role_id]

                if role1.promptRef != role2.promptRef:
                    distance += 1

                # Check model differences
                if role1.model != role2.model:
                    distance += 1

                # Check tool differences
                tools1 = set(role1.tools or [])
                tools2 = set(role2.tools or [])
                distance += len(tools1.symmetric_difference(tools2))

        # Compare graph edges
        def edge_key(edge: Any) -> tuple[str, str, str | None]:
            """Create comparable key for graph edge."""
            from_val = edge.from_ if isinstance(edge.from_, str) else tuple(edge.from_)
            return (str(from_val), edge.to, edge.condition)

        edges1 = {edge_key(e) for e in sop1.graph}
        edges2 = {edge_key(e) for e in sop2.graph}

        distance += len(edges1.symmetric_difference(edges2))

        return distance
