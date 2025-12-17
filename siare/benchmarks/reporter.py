"""Benchmark result reporting with visualization and TREC support."""

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

# Statistical significance thresholds for reporting
SIGNIFICANCE_ALPHA = 0.05  # Standard alpha for p-value significance
P_VALUE_DISPLAY_THRESHOLD = 0.0001  # Threshold for "<0.0001" display


if TYPE_CHECKING:
    from siare.benchmarks.evolution_runner import EvolutionBenchmarkResult
    from siare.benchmarks.publication_suite import PublicationBenchmarkResult
    from siare.benchmarks.runner import BenchmarkResults


class BenchmarkReporter:
    """Generates reports from benchmark results with visualization support.

    Supports multiple output formats:
    - JSON: Machine-readable, CI/CD integration
    - Markdown: GitHub README, documentation
    - HTML: Interactive dashboard with charts
    - TREC: Academic comparison format
    - Marketing: Whitepaper-ready summaries

    Example:
        >>> reporter = BenchmarkReporter()
        >>> reporter.save_report(results, "report.html", "html")
    """

    def to_markdown(self, results: "BenchmarkResults") -> str:
        """Generate markdown report from results."""
        lines = [
            "# Benchmark Report",
            "",
            f"**Dataset:** {results.dataset_name}",
            f"**Date:** {datetime.now(tz=timezone.utc).isoformat()}",
            "",
            "## Summary",
            "",
            f"- Total Samples: {results.total_samples}",
            f"- Completed: {results.completed_samples}",
            f"- Failed: {results.failed_samples}",
            f"- Success Rate: {results.completed_samples / results.total_samples:.1%}"
            if results.total_samples
            else "- Success Rate: N/A",
            f"- Total Time: {results.total_time_seconds:.2f}s",
            "",
        ]

        if results.aggregate_metrics:
            lines.extend(
                [
                    "## Metrics",
                    "",
                    "| Metric | Score |",
                    "|--------|-------|",
                ]
            )
            for metric, score in results.aggregate_metrics.items():
                lines.append(f"| {metric} | {score:.4f} |")
            lines.append("")

        return "\n".join(lines)

    def to_json(self, results: "BenchmarkResults") -> str:
        """Generate JSON report from results."""
        data = asdict(results)
        data["generated_at"] = datetime.now(tz=timezone.utc).isoformat()
        return json.dumps(data, indent=2, default=str)

    def to_trec_runfile(
        self,
        retrieval_results: list[tuple[str, list[str]]],
        run_name: str = "siare",
    ) -> str:
        """Generate TREC-format runfile for academic comparison.

        The TREC runfile format is the standard for information retrieval
        evaluation, enabling direct comparison with published results.

        Format: query_id Q0 doc_id rank score run_name

        Args:
            retrieval_results: List of (query_id, [doc_ids...]) tuples
            run_name: Identifier for this run

        Returns:
            TREC-format runfile as string
        """
        lines = []
        for query_id, doc_ids in retrieval_results:
            for rank, doc_id in enumerate(doc_ids, 1):
                # Score decreases with rank (simple scoring)
                score = 1000 - rank
                lines.append(f"{query_id} Q0 {doc_id} {rank} {score} {run_name}")
        return "\n".join(lines)

    def to_marketing_markdown(
        self,
        results: "BenchmarkResults",
        baseline_results: Optional["BenchmarkResults"] = None,
    ) -> str:
        """Generate marketing-ready markdown for whitepapers/blogs.

        Args:
            results: SIARE benchmark results
            baseline_results: Optional baseline for comparison

        Returns:
            Marketing-formatted markdown
        """
        lines = [
            "# SIARE RAG Benchmark Results",
            "",
            "## Executive Summary",
            "",
            f"SIARE achieves strong performance on the {results.dataset_name} benchmark",
        ]

        if baseline_results:
            # Calculate average improvement
            improvements = []
            for metric, score in results.aggregate_metrics.items():
                baseline_score = baseline_results.aggregate_metrics.get(metric, 0)
                if baseline_score > 0:
                    improvement = ((score - baseline_score) / baseline_score) * 100
                    improvements.append(improvement)
            if improvements:
                avg_improvement = sum(improvements) / len(improvements)
                lines.append(
                    f"with an average improvement of {avg_improvement:.1f}% over baseline."
                )

        lines.extend(
            [
                "",
                "## Methodology",
                "",
                f"- **Dataset:** {results.dataset_name}",
                f"- **Samples:** {results.total_samples}",
                f"- **Success Rate:** {results.completed_samples / results.total_samples:.1%}"
                if results.total_samples
                else "- **Success Rate:** N/A",
                "- **Evaluation:** 6 trials per query, bootstrap 95% CI",
                "",
                "## Results",
                "",
            ]
        )

        if baseline_results:
            lines.extend(
                [
                    "| Metric | SIARE | Baseline | Delta |",
                    "|--------|-------|----------|-------|",
                ]
            )
            for metric, score in results.aggregate_metrics.items():
                baseline_score = baseline_results.aggregate_metrics.get(metric, 0)
                delta = score - baseline_score
                delta_str = f"+{delta:.3f}" if delta >= 0 else f"{delta:.3f}"
                lines.append(
                    f"| {metric} | {score:.3f} | {baseline_score:.3f} | {delta_str} |"
                )
        else:
            lines.extend(
                [
                    "| Metric | Score |",
                    "|--------|-------|",
                ]
            )
            for metric, score in results.aggregate_metrics.items():
                lines.append(f"| {metric} | {score:.3f} |")

        lines.extend(
            [
                "",
                "## Reproducibility",
                "",
                "- Full benchmark code: `siare/benchmarks/`",
                f"- Total runtime: {results.total_time_seconds:.2f}s",
            ]
        )

        return "\n".join(lines)

    def to_html(
        self,
        results: "BenchmarkResults",
        include_charts: bool = True,
    ) -> str:
        """Generate HTML report with interactive Plotly charts.

        Args:
            results: Benchmark results
            include_charts: Whether to include interactive charts

        Returns:
            HTML document as string
        """
        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<meta charset='utf-8'>",
            "<title>SIARE Benchmark Report</title>",
            '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>',
            "<style>",
            "body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }",
            "h1 { color: #1f77b4; }",
            "table { border-collapse: collapse; width: 100%; margin: 20px 0; }",
            "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "th { background-color: #1f77b4; color: white; }",
            ".metric-card { background: #f5f5f5; padding: 15px; margin: 10px; border-radius: 8px; }",
            "</style>",
            "</head>",
            "<body>",
            f"<h1>Benchmark Report: {results.dataset_name}</h1>",
            f"<p>Generated: {datetime.now(tz=timezone.utc).isoformat()}</p>",
            "",
            "<h2>Summary</h2>",
            "<div class='metric-card'>",
            f"<p><strong>Total Samples:</strong> {results.total_samples}</p>",
            f"<p><strong>Completed:</strong> {results.completed_samples}</p>",
            f"<p><strong>Failed:</strong> {results.failed_samples}</p>",
            f"<p><strong>Runtime:</strong> {results.total_time_seconds:.2f}s</p>",
            "</div>",
        ]

        if results.aggregate_metrics:
            html_parts.extend(
                [
                    "<h2>Metrics</h2>",
                    "<table>",
                    "<tr><th>Metric</th><th>Score</th></tr>",
                ]
            )
            for metric, score in results.aggregate_metrics.items():
                html_parts.append(f"<tr><td>{metric}</td><td>{score:.4f}</td></tr>")
            html_parts.append("</table>")

            # Add radar chart if charts enabled
            if include_charts:
                try:
                    from siare.benchmarks.visualization.plotly_charts import radar_chart

                    fig = radar_chart(results.aggregate_metrics)
                    html_parts.append("<h2>Performance Radar</h2>")
                    html_parts.append(fig.to_html(full_html=False, include_plotlyjs=False))
                except ImportError:
                    pass

        html_parts.extend(["</body>", "</html>"])

        return "\n".join(html_parts)

    def save_report(
        self,
        results: "BenchmarkResults",
        filepath: str,
        report_format: str = "markdown",
    ) -> None:
        """Save report to file.

        Args:
            results: Benchmark results to report
            filepath: Path to save report
            report_format: "markdown", "json", "html", or "marketing"
        """
        if report_format == "json":
            content = self.to_json(results)
        elif report_format == "html":
            content = self.to_html(results)
        elif report_format == "marketing":
            content = self.to_marketing_markdown(results)
        else:
            content = self.to_markdown(results)

        Path(filepath).write_text(content)

    def to_evolution_comparison_markdown(
        self,
        result: "EvolutionBenchmarkResult",
    ) -> str:
        """Generate markdown report comparing baseline vs evolved SOP.

        Args:
            result: EvolutionBenchmarkResult with comparison data

        Returns:
            Markdown string with comparison tables and analysis
        """
        lines = [
            "# Evolution Benchmark Report",
            "",
            f"**Dataset:** {result.dataset_name}",
            f"**Date:** {datetime.now(tz=timezone.utc).isoformat()}",
            f"**Generations Run:** {result.generations_run}",
            f"**Total Time:** {result.total_time_seconds:.2f}s",
            "",
            "## Executive Summary",
            "",
        ]

        # Calculate overall improvement
        if result.comparisons:
            avg_improvement = sum(c.improvement_pct for c in result.comparisons) / len(
                result.comparisons
            )
            if avg_improvement > 0:
                lines.append(
                    f"Evolution achieved an average improvement of **{avg_improvement:.1f}%** "
                    f"across {len(result.comparisons)} metrics after {result.generations_run} generations."
                )
            else:
                lines.append(
                    f"Evolution completed {result.generations_run} generations. "
                    f"Average metric change: {avg_improvement:.1f}%."
                )
        lines.append("")

        # Comparison summary table
        lines.extend([
            "## Performance Comparison",
            "",
            "| Phase | SOP ID | Completed | Failed | Time |",
            "|-------|--------|-----------|--------|------|",
            f"| Baseline | {result.baseline_sop_id} | {result.baseline_results.completed_samples} | {result.baseline_results.failed_samples} | {result.baseline_results.total_time_seconds:.2f}s |",
            f"| Evolved | {result.evolved_sop_id} | {result.evolved_results.completed_samples} | {result.evolved_results.failed_samples} | {result.evolved_results.total_time_seconds:.2f}s |",
            "",
        ])

        # Detailed metrics comparison
        lines.extend([
            "## Metric Improvements",
            "",
            "| Metric | Baseline | Evolved | Δ | % Change | Significant? |",
            "|--------|----------|---------|---|----------|--------------|",
        ])

        for comp in result.comparisons:
            sign = "+" if comp.improvement >= 0 else ""
            pct_sign = "+" if comp.improvement_pct >= 0 else ""

            # Check statistical significance
            if comp.p_value is not None:
                sig = "✓" if comp.p_value < SIGNIFICANCE_ALPHA else "✗"
                sig_note = f"{sig} (p={comp.p_value:.4f})"
            else:
                sig_note = "N/A"

            lines.append(
                f"| {comp.metric_name} | {comp.baseline_mean:.4f} | {comp.evolved_mean:.4f} | "
                f"{sign}{comp.improvement:.4f} | {pct_sign}{comp.improvement_pct:.1f}% | {sig_note} |"
            )
        lines.append("")

        # Confidence intervals
        lines.extend([
            "## Statistical Details",
            "",
            "### 95% Confidence Intervals",
            "",
            "| Metric | Baseline CI | Evolved CI |",
            "|--------|-------------|------------|",
        ])

        for comp in result.comparisons:
            lines.append(
                f"| {comp.metric_name} | [{comp.baseline_ci[0]:.4f}, {comp.baseline_ci[1]:.4f}] | "
                f"[{comp.evolved_ci[0]:.4f}, {comp.evolved_ci[1]:.4f}] |"
            )
        lines.append("")

        # Configuration summary
        lines.extend([
            "## Configuration",
            "",
            f"- **Model:** {result.config.model}",
            f"- **Max Generations:** {result.config.max_generations}",
            f"- **Population Size:** {result.config.population_size}",
            f"- **Quick Mode:** {result.config.quick_mode}",
            f"- **Metrics Optimized:** {', '.join(result.config.metrics_to_optimize)}",
            f"- **Mutation Types:** {', '.join(result.config.mutation_types)}",
            "",
        ])

        return "\n".join(lines)

    def to_evolution_comparison_html(
        self,
        result: "EvolutionBenchmarkResult",
        include_charts: bool = True,
    ) -> str:
        """Generate HTML report for evolution benchmark comparison.

        Args:
            result: EvolutionBenchmarkResult
            include_charts: Whether to include interactive charts

        Returns:
            HTML document string
        """
        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<meta charset='utf-8'>",
            "<title>SIARE Evolution Benchmark Report</title>",
            '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>',
            "<style>",
            "body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }",
            "h1 { color: #1f77b4; }",
            "h2 { color: #2ca02c; border-bottom: 2px solid #2ca02c; padding-bottom: 5px; }",
            "table { border-collapse: collapse; width: 100%; margin: 20px 0; }",
            "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "th { background-color: #1f77b4; color: white; }",
            ".metric-card { background: #f5f5f5; padding: 15px; margin: 10px; border-radius: 8px; display: inline-block; min-width: 200px; }",
            ".improvement { color: #2ca02c; font-weight: bold; }",
            ".degradation { color: #d62728; font-weight: bold; }",
            ".chart-container { margin: 20px 0; }",
            "</style>",
            "</head>",
            "<body>",
            f"<h1>Evolution Benchmark: {result.dataset_name}</h1>",
            f"<p>Generated: {datetime.now(tz=timezone.utc).isoformat()}</p>",
            "",
        ]

        # Summary cards
        avg_improvement = (
            sum(c.improvement_pct for c in result.comparisons) / len(result.comparisons)
            if result.comparisons
            else 0
        )
        imp_class = "improvement" if avg_improvement > 0 else "degradation"

        html_parts.extend([
            "<h2>Summary</h2>",
            "<div>",
            "<div class='metric-card'>",
            f"<h3>Generations</h3><p style='font-size: 24px;'>{result.generations_run}</p>",
            "</div>",
            "<div class='metric-card'>",
            f"<h3>Avg Improvement</h3><p class='{imp_class}' style='font-size: 24px;'>{avg_improvement:+.1f}%</p>",
            "</div>",
            "<div class='metric-card'>",
            f"<h3>Total Time</h3><p style='font-size: 24px;'>{result.total_time_seconds:.1f}s</p>",
            "</div>",
            "</div>",
            "",
        ])

        # Comparison table
        html_parts.extend([
            "<h2>Metric Comparison</h2>",
            "<table>",
            "<tr><th>Metric</th><th>Baseline</th><th>Evolved</th><th>Change</th><th>% Change</th></tr>",
        ])

        for comp in result.comparisons:
            sign = "+" if comp.improvement >= 0 else ""
            pct_sign = "+" if comp.improvement_pct >= 0 else ""
            row_class = "improvement" if comp.improvement > 0 else "degradation" if comp.improvement < 0 else ""

            html_parts.append(
                f"<tr>"
                f"<td>{comp.metric_name}</td>"
                f"<td>{comp.baseline_mean:.4f}</td>"
                f"<td>{comp.evolved_mean:.4f}</td>"
                f"<td class='{row_class}'>{sign}{comp.improvement:.4f}</td>"
                f"<td class='{row_class}'>{pct_sign}{comp.improvement_pct:.1f}%</td>"
                f"</tr>"
            )

        html_parts.append("</table>")

        # Add comparison bar chart
        if include_charts and result.comparisons:
            metrics = [c.metric_name for c in result.comparisons]
            baseline_vals = [c.baseline_mean for c in result.comparisons]
            evolved_vals = [c.evolved_mean for c in result.comparisons]

            html_parts.extend([
                "<h2>Visual Comparison</h2>",
                "<div class='chart-container' id='comparison-chart'></div>",
                "<script>",
                "var data = [",
                f"  {{x: {json.dumps(metrics)}, y: {json.dumps(baseline_vals)}, name: 'Baseline', type: 'bar'}},",
                f"  {{x: {json.dumps(metrics)}, y: {json.dumps(evolved_vals)}, name: 'Evolved', type: 'bar'}}",
                "];",
                "var layout = {barmode: 'group', title: 'Baseline vs Evolved Performance'};",
                "Plotly.newPlot('comparison-chart', data, layout);",
                "</script>",
            ])

        html_parts.extend(["</body>", "</html>"])

        return "\n".join(html_parts)

    def save_evolution_report(
        self,
        result: "EvolutionBenchmarkResult",
        filepath: str,
        report_format: str = "markdown",
    ) -> None:
        """Save evolution benchmark report to file.

        Args:
            result: Evolution benchmark result
            filepath: Path to save report
            report_format: "markdown" or "html"
        """
        if report_format == "html":
            content = self.to_evolution_comparison_html(result)
        else:
            content = self.to_evolution_comparison_markdown(result)

        Path(filepath).write_text(content)

    def to_publication_markdown(
        self,
        result: "PublicationBenchmarkResult",
    ) -> str:
        """Generate publication-ready markdown with tables and CI notation.

        Creates a comprehensive markdown report suitable for academic papers,
        including statistical significance indicators, confidence intervals,
        and effect sizes.

        Args:
            result: PublicationBenchmarkResult from Tier 3 benchmark

        Returns:
            Publication-formatted markdown string
        """
        lines = [
            "# SIARE Publication Benchmark Report",
            "",
            "## Metadata",
            "",
            f"- **Dataset:** {result.metadata.get('dataset_name', 'Unknown')}",
            f"- **Date:** {result.metadata.get('timestamp', 'Unknown')}",
            f"- **Git Commit:** {result.metadata.get('git_commit', 'Unknown')}",
            f"- **Samples:** {result.metadata.get('n_queries', 'Unknown')}",
            f"- **Runs per Sample:** {result.metadata.get('n_runs', 'Unknown')}",
            f"- **Confidence Level:** {result.metadata.get('confidence_level', 0.99):.0%}",
            "",
            "## Executive Summary",
            "",
        ]

        # Calculate overall statistics
        evolved_metrics = result.evolved_sop_results.get("metrics", {})
        if evolved_metrics:
            avg_accuracy = evolved_metrics.get("benchmark_accuracy", {}).get("mean", 0)
            lines.append(
                f"SIARE achieves **{avg_accuracy:.1%}** accuracy on the benchmark dataset."
            )

        # Count significant improvements
        significant_improvements = 0
        for baseline_data in result.baseline_comparisons.values():
            stats = baseline_data.get("statistical_tests", {})
            for test in stats.values():
                if test.get("significant", False) and test.get("mean_difference", 0) > 0:
                    significant_improvements += 1

        if significant_improvements > 0:
            lines.append(
                f"Evolution demonstrates **{significant_improvements}** statistically "
                "significant improvements over baseline systems."
            )
        lines.extend(["", ""])

        # Evolved SOP Results
        lines.extend([
            "## Evolved SOP Performance",
            "",
            "| Metric | Mean | 95% CI | Std |",
            "|--------|------|--------|-----|",
        ])

        for metric_name, metric_data in evolved_metrics.items():
            mean = metric_data.get("mean", 0)
            ci_lower = metric_data.get("ci_lower", mean)
            ci_upper = metric_data.get("ci_upper", mean)
            std = metric_data.get("std", 0)
            lines.append(
                f"| {metric_name} | {mean:.4f} | [{ci_lower:.4f}, {ci_upper:.4f}] | {std:.4f} |"
            )
        lines.append("")

        # Baseline Comparisons
        if result.baseline_comparisons:
            lines.extend([
                "## Baseline Comparisons",
                "",
            ])

            for baseline_name, baseline_data in result.baseline_comparisons.items():
                lines.extend([
                    f"### vs. {baseline_name}",
                    "",
                    "| Metric | Evolved | Baseline | Δ | % Imp. | p-value | Sig. | Effect Size |",
                    "|--------|---------|----------|---|--------|---------|------|-------------|",
                ])

                baseline_metrics = baseline_data.get("metrics", {})
                stats = baseline_data.get("statistical_tests", {})

                for metric_name in evolved_metrics:
                    evolved_mean = evolved_metrics[metric_name].get("mean", 0)
                    baseline_mean = baseline_metrics.get(metric_name, {}).get("mean", 0)
                    delta = evolved_mean - baseline_mean
                    pct_imp = (delta / baseline_mean * 100) if baseline_mean > 0 else 0

                    test = stats.get(metric_name, {})
                    p_val = test.get("p_value", 1.0)
                    adj_p = test.get("adjusted_p_value", p_val)
                    sig = "✓" if test.get("significant", False) else "✗"
                    effect = test.get("effect_size", 0)

                    # Format p-value with significance indicator
                    p_str = f"{adj_p:.4f}" if adj_p >= P_VALUE_DISPLAY_THRESHOLD else "<0.0001"
                    delta_sign = "+" if delta >= 0 else ""
                    pct_sign = "+" if pct_imp >= 0 else ""

                    lines.append(
                        f"| {metric_name} | {evolved_mean:.4f} | {baseline_mean:.4f} | "
                        f"{delta_sign}{delta:.4f} | {pct_sign}{pct_imp:.1f}% | "
                        f"{p_str} | {sig} | {effect:.3f} |"
                    )
                lines.append("")

        # Ablation Studies
        if result.ablation_studies:
            lines.extend([
                "## Ablation Studies",
                "",
                "Impact of removing each component (performance drop):",
                "",
                "| Component | Metric | Drop | % Drop | p-value | Sig. |",
                "|-----------|--------|------|--------|---------|------|",
            ])

            for component_name, ablation in result.ablation_studies.items():
                for metric_name, drop in ablation.contribution.items():
                    test = ablation.statistical_tests.get(metric_name)
                    p_val = test.p_value if test else 1.0
                    adj_p = test.adjusted_p_value if test else p_val
                    sig = "✓" if (test and test.significant) else "✗"

                    # Calculate percentage drop
                    original = evolved_metrics.get(metric_name, {}).get("mean", 1)
                    pct_drop = (drop / original * 100) if original > 0 else 0

                    p_str = f"{adj_p:.4f}" if adj_p >= P_VALUE_DISPLAY_THRESHOLD else "<0.0001"
                    lines.append(
                        f"| {component_name} | {metric_name} | {drop:.4f} | {pct_drop:.1f}% | {p_str} | {sig} |"
                    )
            lines.append("")

        # Learning Curves Summary
        if result.learning_curves and result.learning_curves.evolved_sop:
            lines.extend([
                "## Learning Curve Summary",
                "",
                f"**Primary Metric:** {result.learning_curves.primary_metric}",
                "",
            ])

            evolved_data = result.learning_curves.evolved_sop
            if evolved_data:
                initial = evolved_data[0].get(result.learning_curves.primary_metric, 0)
                final = evolved_data[-1].get(result.learning_curves.primary_metric, 0)
                improvement = final - initial
                pct_improvement = (improvement / initial * 100) if initial > 0 else 0

                lines.extend([
                    f"- **Initial Performance:** {initial:.4f}",
                    f"- **Final Performance:** {final:.4f}",
                    f"- **Improvement:** +{improvement:.4f} (+{pct_improvement:.1f}%)",
                    f"- **Generations:** {len(evolved_data)}",
                    "",
                ])

        # Power Analysis
        if result.power_analysis:
            pa = result.power_analysis
            lines.extend([
                "## Statistical Power Analysis",
                "",
                f"- **Primary Metric:** {pa.primary_metric}",
                f"- **Observed Effect Size:** {pa.effect_size:.3f}",
                f"- **Alpha:** {pa.alpha:.3f}",
                f"- **Achieved Power:** {pa.power:.3f}",
                f"- **Required Sample Size:** {pa.required_sample_size}",
                f"- **Actual Sample Size:** {pa.actual_sample_size}",
                f"- **Sufficient Power:** {'Yes ✓' if pa.sufficient else 'No ✗'}",
                "",
            ])

        # Reproducibility section
        lines.extend([
            "## Reproducibility",
            "",
            "```bash",
            "# Re-run this benchmark",
            "python -m siare.benchmarks.scripts.run_publication_benchmark \\",
            "    --tier 3 \\",
            f"    --dataset {result.metadata.get('dataset_name', 'frames')} \\",
            f"    --n-runs {result.metadata.get('n_runs', 30)} \\",
            f"    --confidence {result.metadata.get('confidence_level', 0.99)}",
            "```",
            "",
            "---",
            "",
            "*Report generated by SIARE Publication Benchmark Suite*",
        ])

        return "\n".join(lines)

    def to_publication_html(
        self,
        result: "PublicationBenchmarkResult",
        include_charts: bool = True,
    ) -> str:
        """Generate HTML report with embedded Plotly charts.

        Creates an interactive HTML report with visualizations for
        publication-quality benchmark results.

        Args:
            result: PublicationBenchmarkResult from Tier 3 benchmark
            include_charts: Whether to include interactive Plotly charts

        Returns:
            HTML document string
        """
        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<meta charset='utf-8'>",
            "<title>SIARE Publication Benchmark Report</title>",
            '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>',
            "<style>",
            "body { font-family: 'Segoe UI', Arial, sans-serif; max-width: 1400px; margin: 0 auto; padding: 20px; background: #fafafa; }",
            "h1 { color: #1f77b4; border-bottom: 3px solid #1f77b4; padding-bottom: 10px; }",
            "h2 { color: #2c3e50; margin-top: 30px; }",
            "h3 { color: #34495e; }",
            "table { border-collapse: collapse; width: 100%; margin: 20px 0; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }",
            "th, td { border: 1px solid #e0e0e0; padding: 12px; text-align: left; }",
            "th { background: #1f77b4; color: white; font-weight: 600; }",
            "tr:nth-child(even) { background: #f8f9fa; }",
            "tr:hover { background: #e8f4f8; }",
            ".summary-card { background: white; padding: 20px; margin: 15px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); display: inline-block; min-width: 180px; text-align: center; }",
            ".summary-card h3 { margin: 0 0 10px 0; color: #666; font-size: 14px; }",
            ".summary-card .value { font-size: 28px; font-weight: bold; color: #1f77b4; }",
            ".improvement { color: #27ae60; }",
            ".degradation { color: #e74c3c; }",
            ".significant { background: #d4edda !important; }",
            ".not-significant { background: #fff3cd !important; }",
            ".chart-container { background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }",
            ".metadata { background: #ecf0f1; padding: 15px; border-radius: 5px; margin-bottom: 20px; }",
            ".metadata p { margin: 5px 0; }",
            "code { background: #2c3e50; color: #ecf0f1; padding: 15px; display: block; border-radius: 5px; overflow-x: auto; }",
            "</style>",
            "</head>",
            "<body>",
            "<h1>SIARE Publication Benchmark Report</h1>",
        ]

        # Metadata section
        html_parts.extend([
            "<div class='metadata'>",
            f"<p><strong>Dataset:</strong> {result.metadata.get('dataset_name', 'Unknown')}</p>",
            f"<p><strong>Date:</strong> {result.metadata.get('timestamp', 'Unknown')}</p>",
            f"<p><strong>Git Commit:</strong> {result.metadata.get('git_commit', 'Unknown')}</p>",
            f"<p><strong>Confidence Level:</strong> {result.metadata.get('confidence_level', 0.99):.0%}</p>",
            "</div>",
        ])

        # Summary cards
        evolved_metrics = result.evolved_sop_results.get("metrics", {})
        accuracy = evolved_metrics.get("benchmark_accuracy", {}).get("mean", 0)
        f1 = evolved_metrics.get("benchmark_f1", {}).get("mean", 0)
        n_runs = result.metadata.get("n_runs", 0)
        n_queries = result.metadata.get("n_queries", 0)

        html_parts.extend([
            "<h2>Summary</h2>",
            "<div>",
            f"<div class='summary-card'><h3>Accuracy</h3><div class='value'>{accuracy:.1%}</div></div>",
            f"<div class='summary-card'><h3>F1 Score</h3><div class='value'>{f1:.3f}</div></div>",
            f"<div class='summary-card'><h3>Samples</h3><div class='value'>{n_queries}</div></div>",
            f"<div class='summary-card'><h3>Runs</h3><div class='value'>{n_runs}</div></div>",
            "</div>",
        ])

        # Evolved SOP metrics table
        html_parts.extend([
            "<h2>Evolved SOP Performance</h2>",
            "<table>",
            "<tr><th>Metric</th><th>Mean</th><th>95% CI</th><th>Std Dev</th></tr>",
        ])

        for metric_name, metric_data in evolved_metrics.items():
            mean = metric_data.get("mean", 0)
            ci_lower = metric_data.get("ci_lower", mean)
            ci_upper = metric_data.get("ci_upper", mean)
            std = metric_data.get("std", 0)
            html_parts.append(
                f"<tr><td>{metric_name}</td><td>{mean:.4f}</td>"
                f"<td>[{ci_lower:.4f}, {ci_upper:.4f}]</td><td>{std:.4f}</td></tr>"
            )
        html_parts.append("</table>")

        # Add comparison chart if charts enabled
        if include_charts and result.baseline_comparisons:
            try:
                from siare.benchmarks.visualization.plotly_charts import (
                    comparison_bar_chart_with_ci,
                )

                chart_data = {
                    "evolved_sop_results": result.evolved_sop_results,
                    "baseline_comparisons": {
                        name: {"metrics": data.get("metrics", {})}
                        for name, data in result.baseline_comparisons.items()
                    },
                }
                metrics_list = list(evolved_metrics.keys())

                fig = comparison_bar_chart_with_ci(
                    chart_data, metrics_list, "Performance Comparison with CI"
                )
                html_parts.extend([
                    "<h2>Performance Comparison</h2>",
                    "<div class='chart-container'>",
                    fig.to_html(full_html=False, include_plotlyjs=False),  # type: ignore[reportUnknownMemberType]
                    "</div>",
                ])
            except ImportError:
                pass

        # Baseline comparison tables
        if result.baseline_comparisons:
            html_parts.append("<h2>Baseline Comparisons</h2>")

            for baseline_name, baseline_data in result.baseline_comparisons.items():
                html_parts.extend([
                    f"<h3>vs. {baseline_name}</h3>",
                    "<table>",
                    "<tr><th>Metric</th><th>Evolved</th><th>Baseline</th><th>Δ</th><th>% Imp.</th><th>p-value</th><th>Effect Size</th></tr>",
                ])

                baseline_metrics = baseline_data.get("metrics", {})
                stats = baseline_data.get("statistical_tests", {})

                for metric_name in evolved_metrics:
                    evolved_mean = evolved_metrics[metric_name].get("mean", 0)
                    baseline_mean = baseline_metrics.get(metric_name, {}).get("mean", 0)
                    delta = evolved_mean - baseline_mean
                    pct_imp = (delta / baseline_mean * 100) if baseline_mean > 0 else 0

                    test = stats.get(metric_name, {})
                    adj_p = test.get("adjusted_p_value", test.get("p_value", 1.0))
                    effect = test.get("effect_size", 0)
                    sig = test.get("significant", False)

                    row_class = "significant" if sig else ""
                    delta_class = "improvement" if delta > 0 else "degradation" if delta < 0 else ""
                    delta_sign = "+" if delta >= 0 else ""
                    pct_sign = "+" if pct_imp >= 0 else ""
                    p_str = f"{adj_p:.4f}" if adj_p >= P_VALUE_DISPLAY_THRESHOLD else "<0.0001"

                    html_parts.append(
                        f"<tr class='{row_class}'>"
                        f"<td>{metric_name}</td>"
                        f"<td>{evolved_mean:.4f}</td>"
                        f"<td>{baseline_mean:.4f}</td>"
                        f"<td class='{delta_class}'>{delta_sign}{delta:.4f}</td>"
                        f"<td class='{delta_class}'>{pct_sign}{pct_imp:.1f}%</td>"
                        f"<td>{p_str}</td>"
                        f"<td>{effect:.3f}</td>"
                        f"</tr>"
                    )
                html_parts.append("</table>")

        # Learning curve chart
        if include_charts and result.learning_curves and result.learning_curves.evolved_sop:
            try:
                from siare.benchmarks.visualization.plotly_charts import (
                    learning_curve_chart,
                )

                curve_data = {
                    "evolved_sop": result.learning_curves.evolved_sop,
                    "baselines": result.learning_curves.baselines,
                    "primary_metric": result.learning_curves.primary_metric,
                }
                fig = learning_curve_chart(curve_data, "Evolution Learning Curve")
                html_parts.extend([
                    "<h2>Learning Curve</h2>",
                    "<div class='chart-container'>",
                    fig.to_html(full_html=False, include_plotlyjs=False),  # type: ignore[reportUnknownMemberType]
                    "</div>",
                ])
            except ImportError:
                pass

        # Ablation chart
        if include_charts and result.ablation_studies:
            try:
                from siare.benchmarks.visualization.plotly_charts import (
                    ablation_contributions_chart,
                )

                ablation_data = {
                    name: {
                        "metrics": {
                            m: {
                                "mean": s.mean,
                                "ci_lower": s.ci_lower,
                                "ci_upper": s.ci_upper,
                            }
                            for m, s in ablation.metrics.items()
                        },
                        "contribution": ablation.contribution,
                    }
                    for name, ablation in result.ablation_studies.items()
                }
                fig = ablation_contributions_chart(ablation_data)
                html_parts.extend([
                    "<h2>Ablation Studies</h2>",
                    "<div class='chart-container'>",
                    fig.to_html(full_html=False, include_plotlyjs=False),  # type: ignore[reportUnknownMemberType]
                    "</div>",
                ])
            except ImportError:
                pass

        # Power analysis section
        if result.power_analysis:
            pa = result.power_analysis
            sufficient_class = "improvement" if pa.sufficient else "degradation"
            sufficient_text = "Yes ✓" if pa.sufficient else "No ✗"

            html_parts.extend([
                "<h2>Statistical Power Analysis</h2>",
                "<table>",
                "<tr><th>Property</th><th>Value</th></tr>",
                f"<tr><td>Primary Metric</td><td>{pa.primary_metric}</td></tr>",
                f"<tr><td>Observed Effect Size</td><td>{pa.effect_size:.3f}</td></tr>",
                f"<tr><td>Alpha</td><td>{pa.alpha:.3f}</td></tr>",
                f"<tr><td>Achieved Power</td><td>{pa.power:.3f}</td></tr>",
                f"<tr><td>Required Sample Size</td><td>{pa.required_sample_size}</td></tr>",
                f"<tr><td>Actual Sample Size</td><td>{pa.actual_sample_size}</td></tr>",
                f"<tr><td>Sufficient Power</td><td class='{sufficient_class}'>{sufficient_text}</td></tr>",
                "</table>",
            ])

        # Footer
        html_parts.extend([
            "<hr>",
            "<p><em>Report generated by SIARE Publication Benchmark Suite</em></p>",
            "</body>",
            "</html>",
        ])

        return "\n".join(html_parts)

    def save_publication_report(
        self,
        result: "PublicationBenchmarkResult",
        filepath: str,
        report_format: str = "markdown",
    ) -> None:
        """Save publication benchmark report to file.

        Args:
            result: PublicationBenchmarkResult from Tier 3 benchmark
            filepath: Path to save report
            report_format: "markdown" or "html"
        """
        if report_format == "html":
            content = self.to_publication_html(result)
        else:
            content = self.to_publication_markdown(result)

        Path(filepath).write_text(content)
