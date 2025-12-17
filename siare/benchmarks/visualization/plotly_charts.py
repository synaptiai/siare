"""Interactive Plotly charts for HTML reports and blog posts."""

from typing import Any, Optional

import plotly.graph_objects as go


def radar_chart(
    metrics: dict[str, float],
    title: str = "SIARE Performance",
    baseline_metrics: Optional[dict[str, float]] = None,
) -> go.Figure:
    """Generate radar chart comparing metrics against baseline.

    Args:
        metrics: Dictionary of metric_name -> score (0-1)
        title: Chart title
        baseline_metrics: Optional baseline metrics for comparison

    Returns:
        Plotly Figure object
    """
    categories = list(metrics.keys())

    fig = go.Figure()

    # SIARE metrics
    values = list(metrics.values())
    values.append(values[0])  # Close the polygon
    categories_closed = [*categories, categories[0]]

    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=categories_closed,
            fill="toself",
            name="SIARE",
            line={"color": "rgb(31, 119, 180)"},
        )
    )

    # Baseline comparison
    if baseline_metrics:
        baseline_values = [baseline_metrics.get(k, 0) for k in categories]
        baseline_values.append(baseline_values[0])

        fig.add_trace(
            go.Scatterpolar(
                r=baseline_values,
                theta=categories_closed,
                fill="toself",
                name="Baseline",
                line={"color": "rgb(255, 127, 14)", "dash": "dash"},
            )
        )

    fig.update_layout(
        polar={"radialaxis": {"visible": True, "range": [0, 1]}},
        title=title,
        showlegend=True,
    )

    return fig


def beir_comparison_bar(
    results: dict[str, dict[str, float]],
    metric: str = "ndcg@10",
    title: Optional[str] = None,
) -> go.Figure:
    """Generate bar chart comparing metrics across BEIR datasets.

    Args:
        results: Dictionary of dataset_name -> {metric: score}
        metric: Metric to display (default: ndcg@10)
        title: Chart title (auto-generated if None)

    Returns:
        Plotly Figure object
    """
    datasets = list(results.keys())
    scores = [results[d].get(metric, 0) for d in datasets]

    fig = go.Figure(
        data=[
            go.Bar(
                x=datasets,
                y=scores,
                text=[f"{s:.3f}" for s in scores],
                textposition="outside",
                marker_color="rgb(31, 119, 180)",
            )
        ]
    )

    fig.update_layout(
        title=title or f"BEIR {metric.upper()} by Dataset",
        xaxis_title="Dataset",
        yaxis_title=metric.upper(),
        yaxis={"range": [0, 1]},
        xaxis_tickangle=-45,
    )

    return fig


def evolution_progress_chart(
    generations: list[dict[str, float]],
    metric: str = "accuracy",
    title: Optional[str] = None,
) -> go.Figure:
    """Generate evolution progress chart showing improvement over generations.

    This is a SIARE-specific visualization showing how metrics improve
    through evolutionary optimization.

    Args:
        generations: List of generation results [{metric: score}, ...]
        metric: Metric to track (default: accuracy)
        title: Chart title (auto-generated if None)

    Returns:
        Plotly Figure object
    """
    gen_numbers = list(range(len(generations)))
    scores = [g.get(metric, 0) for g in generations]

    fig = go.Figure(
        data=[
            go.Scatter(
                x=gen_numbers,
                y=scores,
                mode="lines+markers",
                name=metric,
                line={"color": "rgb(31, 119, 180)", "width": 2},
                marker={"size": 8},
            )
        ]
    )

    fig.update_layout(
        title=title or f"SIARE Evolution Progress: {metric.title()}",
        xaxis_title="Generation",
        yaxis_title=metric.title(),
        yaxis={"range": [0, 1]},
    )

    return fig


def comparison_table(
    siare_results: dict[str, float],
    baseline_results: dict[str, float],
    baseline_name: str = "Baseline",
) -> go.Figure:
    """Generate comparison table as Plotly table.

    Args:
        siare_results: SIARE benchmark results
        baseline_results: Baseline results for comparison
        baseline_name: Name of baseline system

    Returns:
        Plotly Figure with table
    """
    metrics = list(siare_results.keys())
    siare_scores = [f"{siare_results[m]:.3f}" for m in metrics]
    baseline_scores = [f"{baseline_results.get(m, 0):.3f}" for m in metrics]
    deltas = []

    for m in metrics:
        siare = siare_results[m]
        baseline = baseline_results.get(m, 0)
        delta = siare - baseline
        deltas.append(f"+{delta:.3f}" if delta > 0 else f"{delta:.3f}")

    fig = go.Figure(
        data=[
            go.Table(
                header={
                    "values": ["Metric", "SIARE", baseline_name, "Delta"],
                    "fill_color": "rgb(31, 119, 180)",
                    "font": {"color": "white", "size": 12},
                    "align": "left",
                },
                cells={
                    "values": [metrics, siare_scores, baseline_scores, deltas],
                    "fill_color": "white",
                    "align": "left",
                },
            )
        ]
    )

    fig.update_layout(title="Benchmark Comparison")

    return fig


# Color palette for consistent styling across charts
_EVOLVED_COLOR = "rgb(31, 119, 180)"  # Blue
_BASELINE_COLORS = [
    "rgb(255, 127, 14)",  # Orange
    "rgb(44, 160, 44)",   # Green
    "rgb(214, 39, 40)",   # Red
    "rgb(148, 103, 189)", # Purple
    "rgb(140, 86, 75)",   # Brown
]


def comparison_bar_chart_with_ci(
    results: dict[str, Any],
    metrics: list[str],
    title: str = "Performance Comparison",
) -> go.Figure:
    """Generate bar chart with confidence interval error bars.

    Creates a grouped bar chart comparing evolved SOP against baselines,
    with error bars representing confidence intervals.

    Args:
        results: Dictionary with structure:
            {
                "evolved_sop_results": {
                    "metrics": {
                        "<metric_name>": {"mean": float, "ci_lower": float, "ci_upper": float}
                    }
                },
                "baseline_comparisons": {
                    "<baseline_name>": {
                        "metrics": {
                            "<metric_name>": {"mean": float, "ci_lower": float, "ci_upper": float}
                        }
                    }
                }
            }
        metrics: List of metric names to display
        title: Chart title

    Returns:
        Plotly Figure with grouped bars and CI error bars
    """
    fig = go.Figure()

    # Extract evolved SOP metrics
    evolved_metrics = results.get("evolved_sop_results", {}).get("metrics", {})
    evolved_means = []
    evolved_errors_lower = []
    evolved_errors_upper = []

    for metric in metrics:
        metric_data = evolved_metrics.get(metric, {})
        mean = metric_data.get("mean", 0.0)
        ci_lower = metric_data.get("ci_lower", mean)
        ci_upper = metric_data.get("ci_upper", mean)
        evolved_means.append(mean)
        evolved_errors_lower.append(mean - ci_lower)
        evolved_errors_upper.append(ci_upper - mean)

    # Add evolved SOP bars
    fig.add_trace(
        go.Bar(
            name="Evolved SOP",
            x=metrics,
            y=evolved_means,
            error_y={
                "type": "data",
                "symmetric": False,
                "array": evolved_errors_upper,
                "arrayminus": evolved_errors_lower,
            },
            marker_color=_EVOLVED_COLOR,
        )
    )

    # Add baseline comparison bars
    baseline_comparisons = results.get("baseline_comparisons", {})
    for idx, (baseline_name, baseline_data) in enumerate(baseline_comparisons.items()):
        baseline_metrics = baseline_data.get("metrics", {})
        baseline_means = []
        baseline_errors_lower = []
        baseline_errors_upper = []

        for metric in metrics:
            metric_data = baseline_metrics.get(metric, {})
            mean = metric_data.get("mean", 0.0)
            ci_lower = metric_data.get("ci_lower", mean)
            ci_upper = metric_data.get("ci_upper", mean)
            baseline_means.append(mean)
            baseline_errors_lower.append(mean - ci_lower)
            baseline_errors_upper.append(ci_upper - mean)

        color_idx = idx % len(_BASELINE_COLORS)
        fig.add_trace(
            go.Bar(
                name=baseline_name,
                x=metrics,
                y=baseline_means,
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": baseline_errors_upper,
                    "arrayminus": baseline_errors_lower,
                },
                marker_color=_BASELINE_COLORS[color_idx],
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Metric",
        yaxis_title="Score",
        yaxis={"range": [0, 1.1]},  # Allow room for error bars
        barmode="group",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )

    return fig


def learning_curve_chart(
    learning_curves: dict[str, Any],
    title: str = "Evolution Learning Curve",
) -> go.Figure:
    """Generate learning curve showing metric improvement over generations.

    Creates a line chart tracking performance across evolution generations,
    optionally comparing against static baselines.

    Args:
        learning_curves: Dictionary with structure:
            {
                "evolved_sop": [
                    {"generation": int, "<metric_name>": float, ...},
                    ...
                ],
                "baselines": {
                    "<baseline_name>": [
                        {"generation": int, "<metric_name>": float, ...},
                        ...
                    ]
                },
                "primary_metric": str  # Metric to plot on y-axis
            }
        title: Chart title

    Returns:
        Plotly Figure with learning curves
    """
    fig = go.Figure()

    primary_metric = learning_curves.get("primary_metric", "benchmark_accuracy")

    # Plot evolved SOP learning curve
    evolved_data = learning_curves.get("evolved_sop", [])
    if evolved_data:
        generations = [d.get("generation", i) for i, d in enumerate(evolved_data)]
        metric_values = [d.get(primary_metric, 0.0) for d in evolved_data]

        fig.add_trace(
            go.Scatter(
                x=generations,
                y=metric_values,
                mode="lines+markers",
                name="Evolved SOP",
                line={"color": _EVOLVED_COLOR, "width": 3},
                marker={"size": 10},
            )
        )

    # Plot baseline comparison lines (static performance)
    baselines = learning_curves.get("baselines", {})
    for idx, (baseline_name, baseline_data) in enumerate(baselines.items()):
        if baseline_data:
            generations = [d.get("generation", i) for i, d in enumerate(baseline_data)]
            metric_values = [d.get(primary_metric, 0.0) for d in baseline_data]

            color_idx = idx % len(_BASELINE_COLORS)
            fig.add_trace(
                go.Scatter(
                    x=generations,
                    y=metric_values,
                    mode="lines+markers",
                    name=baseline_name,
                    line={"color": _BASELINE_COLORS[color_idx], "width": 2, "dash": "dash"},
                    marker={"size": 6},
                )
            )

    # Determine x-axis range from data
    all_generations: list[int] = []
    if evolved_data:
        all_generations.extend(d.get("generation", i) for i, d in enumerate(evolved_data))
    for baseline_data in baselines.values():
        if baseline_data:
            all_generations.extend(d.get("generation", i) for i, d in enumerate(baseline_data))

    max_gen = max(all_generations) if all_generations else 10

    fig.update_layout(
        title=title,
        xaxis_title="Generation",
        yaxis_title=primary_metric.replace("_", " ").title(),
        yaxis={"range": [0, 1.05]},
        xaxis={"dtick": 1, "range": [-0.5, max_gen + 0.5]},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )

    return fig


def ablation_contributions_chart(
    ablation_studies: dict[str, Any],
    title: str = "Component Contributions (Ablation Study)",
) -> go.Figure:
    """Generate ablation study chart showing component contributions.

    Creates a grouped bar chart showing the performance impact of removing
    each component, helping identify which parts contribute most to success.

    Args:
        ablation_studies: Dictionary with structure:
            {
                "<component_name>": {
                    "metrics": {
                        "<metric_name>": {"mean": float, "ci_lower": float, "ci_upper": float}
                    },
                    "contribution": {
                        "<metric_name>": float  # Performance drop when component removed
                    }
                }
            }
        title: Chart title

    Returns:
        Plotly Figure with ablation contribution bars
    """
    fig = go.Figure()

    if not ablation_studies:
        # Return empty figure with message
        fig.add_annotation(
            text="No ablation data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 16},
        )
        fig.update_layout(title=title)
        return fig

    # Extract component names and their contributions
    components = list(ablation_studies.keys())

    # Collect all metrics across ablations
    all_metrics: set[str] = set()
    for ablation_data in ablation_studies.values():
        contribution = ablation_data.get("contribution", {})
        all_metrics.update(contribution.keys())

    metrics_list = sorted(all_metrics)

    if not metrics_list:
        fig.add_annotation(
            text="No contribution metrics available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 16},
        )
        fig.update_layout(title=title)
        return fig

    # Create grouped bars for each metric
    colors = [_EVOLVED_COLOR, *_BASELINE_COLORS]
    for metric_idx, metric_name in enumerate(metrics_list):
        contributions = []
        for component in components:
            ablation_data = ablation_studies[component]
            contribution = ablation_data.get("contribution", {})
            # Contribution is typically positive (performance drop when removed)
            contributions.append(contribution.get(metric_name, 0.0))

        color_idx = metric_idx % len(colors)
        fig.add_trace(
            go.Bar(
                name=metric_name.replace("_", " ").title(),
                x=components,
                y=contributions,
                marker_color=colors[color_idx],
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Component Removed",
        yaxis_title="Performance Drop",
        barmode="group",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        xaxis_tickangle=-45,
    )

    return fig
