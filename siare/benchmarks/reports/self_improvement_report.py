"""Self-improvement report generator.

Generates publication-ready reports demonstrating SIARE's self-improvement
capability through prompt evolution.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from html import escape as html_escape
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from siare.benchmarks.self_improvement_benchmark import SelfImprovementResult


class SelfImprovementReport:
    """Generates reports for self-improvement benchmark results.

    Produces both Markdown and HTML reports with:
    - Executive summary showing key improvement metrics
    - Learning curve visualization
    - Prompt diff sections
    - Statistical analysis tables
    """

    def __init__(self, result: SelfImprovementResult) -> None:
        """Initialize the report generator.

        Args:
            result: SelfImprovementResult from benchmark run
        """
        self._result = result

    def generate_markdown(self) -> str:
        """Generate Markdown report.

        Returns:
            Markdown-formatted report string
        """
        sections = [
            self._title_section(),
            self._metadata_section(),
            self._executive_summary(),
            self._performance_comparison_section(),
            self._statistical_analysis_section(),
            self._learning_curve_section(),
            self._prompt_diff_section(),
            self._configuration_section(),
            self._footer_section(),
        ]
        return "\n\n".join(sections)

    def generate_html(self) -> str:
        """Generate HTML report with embedded charts.

        Returns:
            HTML-formatted report string
        """
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>SIARE Self-Improvement Report</title>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <style>
        {self._css_styles()}
    </style>
</head>
<body>
    <div class="container">
        {self._html_header()}
        {self._html_executive_summary()}
        {self._html_performance_comparison()}
        {self._html_statistical_analysis()}
        {self._html_learning_curve()}
        {self._html_prompt_diffs()}
        {self._html_configuration()}
        {self._html_footer()}
    </div>
    <script>
        {self._plotly_scripts()}
    </script>
</body>
</html>
"""

    def save_markdown(self, path: str | Path) -> None:
        """Save Markdown report to file.

        Args:
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.generate_markdown())

    def save_html(self, path: str | Path) -> None:
        """Save HTML report to file.

        Args:
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.generate_html())

    def save_json(self, path: str | Path) -> None:
        """Save result data as JSON.

        Args:
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "metadata": {
                "dataset": self._result.dataset_name,
                "model": self._result.config.model,
                "reasoning_model": self._result.config.reasoning_model,
                "generations": self._result.generations_run,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            "initial_metrics": self._result.initial_metrics,
            "evolved_metrics": self._result.evolved_metrics,
            "improvements": self._compute_improvements(),
            "significance_tests": {
                metric: {
                    "test_type": test.testType,
                    "p_value": test.pValue,
                    "significant": test.isSignificant,
                    "effect_size": test.effectSize,
                }
                for metric, test in self._result.significance_tests.items()
            },
            "learning_curve": self._result.learning_curve_data,
            "converged": self._result.converged,
            "convergence_generation": self._result.convergence_generation,
        }

        path.write_text(json.dumps(data, indent=2))

    # =========================================================================
    # Markdown Section Generators
    # =========================================================================

    def _title_section(self) -> str:
        """Generate title section."""
        return "# SIARE Self-Improvement Benchmark Report"

    def _metadata_section(self) -> str:
        """Generate metadata section."""
        r = self._result
        timestamp = datetime.now(timezone.utc).isoformat()

        return f"""## Metadata

- **Dataset:** {r.dataset_name}
- **Model:** {r.config.model}
- **Reasoning Model:** {r.config.reasoning_model}
- **Date:** {timestamp}
- **Generations:** {r.generations_run}
- **Samples:** {r.config.max_samples}
- **Confidence Level:** {r.config.confidence_level * 100:.0f}%"""

    def _executive_summary(self) -> str:
        """Generate executive summary."""
        r = self._result
        improvements = self._compute_improvements()

        # Find best improving metric
        best_metric = max(improvements.items(), key=lambda x: x[1]["pct"])
        best_name = best_metric[0]
        best_pct = best_metric[1]["pct"]

        # Count significant improvements
        sig_count = sum(
            1 for test in r.significance_tests.values() if test.isSignificant
        )
        total_metrics = len(r.significance_tests)

        return f"""## Executive Summary

**SIARE's prompt evolution achieved {best_pct:.1f}% improvement on {best_name}** while keeping the model ({r.config.model}) and architecture constant.

| Key Finding | Value |
|-------------|-------|
| Generations evolved | {r.generations_run} |
| Statistically significant improvements | {sig_count}/{total_metrics} metrics |
| Best improvement | {best_name}: {best_pct:.1f}% |
| Convergence | {f'Yes (gen {r.convergence_generation})' if r.converged else 'No'} |
| Total time | {r.total_time_seconds:.1f}s |"""

    def _performance_comparison_section(self) -> str:
        """Generate performance comparison table."""
        r = self._result
        improvements = self._compute_improvements()

        rows = []
        for metric in r.config.metrics_to_optimize:
            initial = r.initial_metrics.get(metric, 0.0)
            evolved = r.evolved_metrics.get(metric, 0.0)
            imp = improvements.get(metric, {"abs": 0, "pct": 0})
            test = r.significance_tests.get(metric)

            sig = "✓" if test and test.isSignificant else "✗"
            p_val = f"{test.pValue:.4f}" if test else "N/A"

            rows.append(
                f"| {metric} | {initial:.4f} | {evolved:.4f} | "
                f"+{imp['abs']:.4f} | +{imp['pct']:.1f}% | {p_val} | {sig} |"
            )

        return f"""## Performance Comparison

| Metric | Initial | Evolved | Δ | % Imp. | p-value | Sig. |
|--------|---------|---------|---|--------|---------|------|
{chr(10).join(rows)}"""

    def _statistical_analysis_section(self) -> str:
        """Generate statistical analysis section."""
        r = self._result

        rows = []
        for metric, test in r.significance_tests.items():
            effect = f"{test.effectSize:.3f}" if test.effectSize else "N/A"
            rows.append(
                f"| {metric} | {test.testType} | {test.statistic:.4f} | "
                f"{test.pValue:.4f} | {'Yes' if test.isSignificant else 'No'} | "
                f"{effect} |"
            )

        return f"""## Statistical Analysis

All comparisons use Wilcoxon signed-rank test (paired samples on same queries).

| Metric | Test | Statistic | p-value | Significant | Effect Size |
|--------|------|-----------|---------|-------------|-------------|
{chr(10).join(rows)}

*Significance threshold: α = {1 - r.config.confidence_level:.2f}*"""

    def _learning_curve_section(self) -> str:
        """Generate learning curve section.

        Shows both the weighted_aggregate (used for selection) and individual
        metrics (accuracy, F1) for transparency.
        """
        r = self._result
        curve = r.learning_curve_data

        if not curve.get("generations"):
            return """## Learning Curve

*No learning curve data available.*"""

        generations = curve["generations"]
        best_quality = curve["best_quality"]
        metrics_by_gen = curve.get("metrics", [])

        # Create ASCII representation
        max_quality = max(best_quality) if best_quality else 1.0

        lines = ["## Learning Curve", ""]

        # Show weighted aggregate (selection metric)
        lines.append("### Weighted Aggregate (Selection Metric)")
        lines.append("```")
        lines.append("-" * 45)

        for gen, best in zip(generations, best_quality, strict=False):
            bar_len = int((best / max_quality) * 30) if max_quality > 0 else 0
            bar = "█" * bar_len
            lines.append(f"Gen {gen:2d} | {bar} {best:.3f}")

        lines.append("```")

        # Show individual metrics if available
        if metrics_by_gen:
            lines.append("")
            lines.append("### Individual Metrics by Generation")
            lines.append("")

            # Get all metric names from first generation
            first_metrics = metrics_by_gen[0] if metrics_by_gen else {}
            metric_names = [k for k in first_metrics if k != "weighted_aggregate"]

            if metric_names:
                # Build table header
                header = "| Gen |"
                sep = "|-----|"
                for name in metric_names:
                    header += f" {name} |"
                    sep += "--------|"

                lines.append(header)
                lines.append(sep)

                # Build table rows
                for gen, metrics in zip(generations, metrics_by_gen, strict=False):
                    row = f"| {gen:3d} |"
                    for name in metric_names:
                        value = metrics.get(name, 0.0)
                        row += f" {value:.4f} |"
                    lines.append(row)

        return "\n".join(lines)

    def _prompt_diff_section(self) -> str:
        """Generate prompt diff section."""
        r = self._result

        if not r.prompt_diffs:
            return """## Prompt Changes

*No prompt changes recorded.*"""

        sections = ["## Prompt Changes", ""]
        changes_found = False

        for role_id, diff_data in r.prompt_diffs.items():
            if diff_data and diff_data.get("has_changes"):
                changes_found = True
                sections.append(f"### {role_id}")
                sections.append(f"*+{diff_data['lines_added']}/-{diff_data['lines_removed']} lines*")
                sections.append("")
                if diff_data.get("markdown"):
                    sections.append(diff_data["markdown"])
                sections.append("")

        if not changes_found:
            sections.append("*No significant prompt changes detected.*")

        return "\n".join(sections)

    def _configuration_section(self) -> str:
        """Generate configuration section."""
        r = self._result
        c = r.config

        return f"""## Configuration

| Parameter | Value |
|-----------|-------|
| Model | {c.model} |
| Reasoning Model | {c.reasoning_model} |
| Max Generations | {c.max_generations} |
| Population Size | {c.population_size} |
| Dataset Tier | {c.dataset_tier} |
| Max Samples | {c.max_samples} |
| Prompt Strategy | {c.prompt_strategy} |
| Metrics | {', '.join(c.metrics_to_optimize)} |"""

    def _footer_section(self) -> str:
        """Generate footer section."""
        return """---

*Report generated by SIARE Self-Improvement Benchmark Suite*"""

    # =========================================================================
    # HTML Section Generators
    # =========================================================================

    def _css_styles(self) -> str:
        """Return CSS styles for HTML report."""
        return """
        :root {
            --primary: #2563eb;
            --success: #16a34a;
            --warning: #d97706;
            --danger: #dc2626;
            --gray-100: #f3f4f6;
            --gray-200: #e5e7eb;
            --gray-700: #374151;
            --gray-900: #111827;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: var(--gray-700);
            background: var(--gray-100);
            margin: 0;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            padding: 40px;
        }
        h1, h2, h3 { color: var(--gray-900); }
        h1 { border-bottom: 3px solid var(--primary); padding-bottom: 10px; }
        h2 { margin-top: 40px; border-bottom: 1px solid var(--gray-200); padding-bottom: 8px; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid var(--gray-200); }
        th { background: var(--primary); color: white; }
        tr:hover { background: var(--gray-100); }
        .metric-card {
            display: inline-block;
            background: var(--gray-100);
            border-radius: 8px;
            padding: 20px;
            margin: 10px;
            min-width: 200px;
        }
        .metric-value { font-size: 2em; font-weight: bold; color: var(--primary); }
        .metric-label { color: var(--gray-700); }
        .improvement-positive { color: var(--success); }
        .improvement-negative { color: var(--danger); }
        .significant { background: #dcfce7; }
        .not-significant { background: #fef3c7; }
        .chart-container { width: 100%; height: 400px; margin: 20px 0; }
        .diff-add { background: #dcfce7; color: #166534; }
        .diff-remove { background: #fee2e2; color: #991b1b; }
        .diff-context { color: var(--gray-700); }
        .diff-block { font-family: monospace; padding: 10px; border-radius: 4px; background: var(--gray-100); }
        """

    def _html_header(self) -> str:
        """Generate HTML header section."""
        r = self._result
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        return f"""
        <h1>SIARE Self-Improvement Benchmark Report</h1>
        <p>
            <strong>Dataset:</strong> {html_escape(r.dataset_name)} |
            <strong>Model:</strong> {html_escape(r.config.model)} |
            <strong>Generated:</strong> {timestamp}
        </p>
        """

    def _html_executive_summary(self) -> str:
        """Generate HTML executive summary."""
        r = self._result
        improvements = self._compute_improvements()

        best_metric = max(improvements.items(), key=lambda x: x[1]["pct"])
        best_name = best_metric[0]
        best_pct = best_metric[1]["pct"]

        sig_count = sum(1 for t in r.significance_tests.values() if t.isSignificant)

        return f"""
        <h2>Executive Summary</h2>
        <div class="metric-cards">
            <div class="metric-card">
                <div class="metric-value">{best_pct:.1f}%</div>
                <div class="metric-label">Best Improvement ({best_name})</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{r.generations_run}</div>
                <div class="metric-label">Generations Evolved</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{sig_count}/{len(r.significance_tests)}</div>
                <div class="metric-label">Significant Improvements</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{r.total_time_seconds:.0f}s</div>
                <div class="metric-label">Total Time</div>
            </div>
        </div>
        <p>
            <strong>Key Finding:</strong> SIARE's prompt evolution achieved <strong>{best_pct:.1f}%</strong>
            improvement on <strong>{html_escape(best_name)}</strong> while keeping the model
            (<code>{html_escape(r.config.model)}</code>) and architecture constant.
        </p>
        """

    def _html_performance_comparison(self) -> str:
        """Generate HTML performance comparison table."""
        r = self._result
        improvements = self._compute_improvements()

        rows = []
        for metric in r.config.metrics_to_optimize:
            initial = r.initial_metrics.get(metric, 0.0)
            evolved = r.evolved_metrics.get(metric, 0.0)
            imp = improvements.get(metric, {"abs": 0, "pct": 0})
            test = r.significance_tests.get(metric)

            sig_class = "significant" if test and test.isSignificant else "not-significant"
            imp_class = "improvement-positive" if imp["pct"] > 0 else "improvement-negative"
            p_value = f"{test.pValue:.4f}" if test else "N/A"

            rows.append(f"""
                <tr class="{sig_class}">
                    <td>{html_escape(metric)}</td>
                    <td>{initial:.4f}</td>
                    <td>{evolved:.4f}</td>
                    <td class="{imp_class}">+{imp['abs']:.4f}</td>
                    <td class="{imp_class}">+{imp['pct']:.1f}%</td>
                    <td>{p_value}</td>
                    <td>{'✓' if test and test.isSignificant else '✗'}</td>
                </tr>
            """)

        return f"""
        <h2>Performance Comparison</h2>
        <table>
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Initial</th>
                    <th>Evolved</th>
                    <th>Δ</th>
                    <th>% Improvement</th>
                    <th>p-value</th>
                    <th>Significant</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
        """

    def _html_statistical_analysis(self) -> str:
        """Generate HTML statistical analysis section."""
        r = self._result

        rows = []
        for metric, test in r.significance_tests.items():
            effect = f"{test.effectSize:.3f}" if test.effectSize else "N/A"
            rows.append(f"""
                <tr>
                    <td>{html_escape(metric)}</td>
                    <td>{html_escape(test.testType)}</td>
                    <td>{test.statistic:.4f}</td>
                    <td>{test.pValue:.4f}</td>
                    <td>{'Yes' if test.isSignificant else 'No'}</td>
                    <td>{effect}</td>
                </tr>
            """)

        return f"""
        <h2>Statistical Analysis</h2>
        <p>All comparisons use Wilcoxon signed-rank test (paired samples on same queries).</p>
        <table>
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Test</th>
                    <th>Statistic</th>
                    <th>p-value</th>
                    <th>Significant</th>
                    <th>Effect Size</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
        <p><em>Significance threshold: α = {1 - r.config.confidence_level:.2f}</em></p>
        """

    def _html_learning_curve(self) -> str:
        """Generate HTML learning curve section."""
        return """
        <h2>Learning Curve</h2>
        <div id="learning-curve-chart" class="chart-container"></div>
        """

    def _html_prompt_diffs(self) -> str:
        """Generate HTML prompt diff section."""
        r = self._result

        if not r.prompt_diffs:
            return """
            <h2>Prompt Changes</h2>
            <p><em>No prompt changes recorded.</em></p>
            """

        sections = ["<h2>Prompt Changes</h2>"]
        changes_found = False

        for role_id, diff_data in r.prompt_diffs.items():
            if diff_data and diff_data.get("has_changes"):
                changes_found = True
                sections.append(f"""
                    <h3>{html_escape(role_id)}</h3>
                    <p><em>+{diff_data['lines_added']}/-{diff_data['lines_removed']} lines</em></p>
                    <div class="diff-block">
                        <pre>{html_escape(diff_data.get('markdown', ''))}</pre>
                    </div>
                """)

        if not changes_found:
            sections.append("<p><em>No significant prompt changes detected.</em></p>")

        return "\n".join(sections)

    def _html_configuration(self) -> str:
        """Generate HTML configuration section."""
        c = self._result.config

        return f"""
        <h2>Configuration</h2>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
            <tr><td>Model</td><td>{html_escape(c.model)}</td></tr>
            <tr><td>Reasoning Model</td><td>{html_escape(c.reasoning_model)}</td></tr>
            <tr><td>Max Generations</td><td>{c.max_generations}</td></tr>
            <tr><td>Population Size</td><td>{c.population_size}</td></tr>
            <tr><td>Dataset Tier</td><td>{c.dataset_tier}</td></tr>
            <tr><td>Max Samples</td><td>{c.max_samples}</td></tr>
            <tr><td>Prompt Strategy</td><td>{html_escape(c.prompt_strategy)}</td></tr>
            <tr><td>Metrics</td><td>{html_escape(', '.join(c.metrics_to_optimize))}</td></tr>
        </table>
        """

    def _html_footer(self) -> str:
        """Generate HTML footer."""
        return """
        <hr>
        <p><em>Report generated by SIARE Self-Improvement Benchmark Suite</em></p>
        """

    def _plotly_scripts(self) -> str:
        """Generate Plotly JavaScript for charts.

        Shows both weighted_aggregate (selection metric) and individual
        metrics (accuracy, F1) for transparency.
        """
        r = self._result
        curve = r.learning_curve_data

        if not curve.get("generations"):
            return "// No learning curve data"

        generations = json.dumps(curve.get("generations", []))
        best_quality = json.dumps(curve.get("best_quality", []))
        avg_quality = json.dumps(curve.get("avg_quality", []))
        ci_lower = json.dumps(curve.get("ci_lower", []))
        ci_upper = json.dumps(curve.get("ci_upper", []))
        metrics_by_gen = curve.get("metrics", [])

        # Extract individual metrics across generations
        individual_metrics_data = self._extract_individual_metrics_for_plotly(
            metrics_by_gen
        )

        return f"""
        // Learning Curve Chart
        var generations = {generations};
        var bestQuality = {best_quality};
        var avgQuality = {avg_quality};
        var ciLower = {ci_lower};
        var ciUpper = {ci_upper};
        var individualMetrics = {json.dumps(individual_metrics_data)};

        var trace1 = {{
            x: generations,
            y: bestQuality,
            mode: 'lines+markers',
            name: 'Weighted Aggregate',
            line: {{ color: '#2563eb', width: 2 }},
            marker: {{ size: 8 }}
        }};

        var trace2 = {{
            x: generations,
            y: avgQuality,
            mode: 'lines+markers',
            name: 'Average Quality',
            line: {{ color: '#9ca3af', width: 2, dash: 'dash' }},
            marker: {{ size: 6 }},
            visible: 'legendonly'
        }};

        var trace3 = {{
            x: generations.concat(generations.slice().reverse()),
            y: ciUpper.concat(ciLower.slice().reverse()),
            fill: 'toself',
            fillcolor: 'rgba(37, 99, 235, 0.1)',
            line: {{ color: 'transparent' }},
            name: '95% CI',
            showlegend: true,
            type: 'scatter',
            visible: 'legendonly'
        }};

        // Add individual metric traces
        var metricColors = ['#16a34a', '#d97706', '#dc2626', '#7c3aed'];
        var traces = [trace3, trace1, trace2];
        var colorIdx = 0;
        for (var metricName in individualMetrics) {{
            traces.push({{
                x: generations,
                y: individualMetrics[metricName],
                mode: 'lines+markers',
                name: metricName,
                line: {{ color: metricColors[colorIdx % metricColors.length], width: 2 }},
                marker: {{ size: 6 }}
            }});
            colorIdx++;
        }}

        var layout = {{
            title: 'Learning Curve: Quality vs Generation',
            xaxis: {{ title: 'Generation' }},
            yaxis: {{ title: 'Quality Score', rangemode: 'tozero' }},
            hovermode: 'x unified',
            legend: {{ orientation: 'h', y: -0.15 }}
        }};

        Plotly.newPlot('learning-curve-chart', traces, layout, {{responsive: true}});
        """

    def _extract_individual_metrics_for_plotly(
        self, metrics_by_gen: list[dict[str, float]]
    ) -> dict[str, list[float]]:
        """Extract individual metrics into a format suitable for Plotly.

        Args:
            metrics_by_gen: List of metric dicts per generation

        Returns:
            Dict mapping metric_name to list of values across generations
        """
        if not metrics_by_gen:
            return {}

        # Get all metric names (excluding weighted_aggregate)
        first_metrics = metrics_by_gen[0] if metrics_by_gen else {}
        metric_names = [k for k in first_metrics if k != "weighted_aggregate"]

        # Build lists of values per metric
        result: dict[str, list[float]] = {name: [] for name in metric_names}
        for gen_metrics in metrics_by_gen:
            for name in metric_names:
                result[name].append(gen_metrics.get(name, 0.0))

        return result

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _compute_improvements(self) -> dict[str, dict[str, float]]:
        """Compute improvement metrics."""
        r = self._result
        improvements: dict[str, dict[str, float]] = {}

        for metric in r.config.metrics_to_optimize:
            initial = r.initial_metrics.get(metric, 0.0)
            evolved = r.evolved_metrics.get(metric, 0.0)

            abs_imp = evolved - initial
            pct_imp = (abs_imp / initial * 100) if initial > 0 else 0.0

            improvements[metric] = {
                "abs": abs_imp,
                "pct": pct_imp,
            }

        return improvements
