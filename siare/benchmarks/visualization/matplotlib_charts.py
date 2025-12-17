"""Static Matplotlib charts for papers and publications."""


import matplotlib.pyplot as plt
import numpy as np


def radar_chart_static(
    metrics: dict[str, float],
    output_path: str,
    baseline_metrics: dict[str, float] | None = None,
    title: str = "SIARE Performance",
    dpi: int = 300,
) -> None:
    """Generate static radar chart for papers.

    Args:
        metrics: Dictionary of metric_name -> score (0-1)
        output_path: Path to save the image
        baseline_metrics: Optional baseline metrics for comparison
        title: Chart title
        dpi: Image resolution (default: 300 for publication quality)
    """
    categories = list(metrics.keys())
    N = len(categories)

    # Calculate angles for each category
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})

    # SIARE metrics
    values = list(metrics.values())
    values += values[:1]  # Close the polygon
    ax.plot(angles, values, "o-", linewidth=2, label="SIARE", color="#1f77b4")
    ax.fill(angles, values, alpha=0.25, color="#1f77b4")

    # Baseline comparison
    if baseline_metrics:
        baseline_values = [baseline_metrics.get(k, 0) for k in categories]
        baseline_values += baseline_values[:1]
        ax.plot(
            angles,
            baseline_values,
            "o--",
            linewidth=2,
            label="Baseline",
            color="#ff7f0e",
        )
        ax.fill(angles, baseline_values, alpha=0.1, color="#ff7f0e")

    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10)

    # Set radial limits
    ax.set_ylim(0, 1)

    ax.set_title(title, size=14, y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def beir_comparison_bar_static(
    results: dict[str, dict[str, float]],
    output_path: str,
    metric: str = "ndcg@10",
    title: str | None = None,
    dpi: int = 300,
) -> None:
    """Generate static bar chart for papers.

    Args:
        results: Dictionary of dataset_name -> {metric: score}
        output_path: Path to save the image
        metric: Metric to display
        title: Chart title
        dpi: Image resolution
    """
    datasets = list(results.keys())
    scores = [results[d].get(metric, 0) for d in datasets]

    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(datasets, scores, color="#1f77b4", edgecolor="black")

    # Add value labels on bars
    for bar, score in zip(bars, scores, strict=False):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.02,
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xlabel("Dataset", fontsize=12)
    ax.set_ylabel(metric.upper(), fontsize=12)
    ax.set_title(title or f"BEIR {metric.upper()} by Dataset", fontsize=14)
    ax.set_ylim(0, 1)

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def evolution_progress_static(
    generations: list[dict[str, float]],
    output_path: str,
    metric: str = "accuracy",
    title: str | None = None,
    dpi: int = 300,
) -> None:
    """Generate static evolution progress chart for papers.

    Args:
        generations: List of generation results
        output_path: Path to save the image
        metric: Metric to track
        title: Chart title
        dpi: Image resolution
    """
    gen_numbers = list(range(len(generations)))
    scores = [g.get(metric, 0) for g in generations]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(gen_numbers, scores, "o-", linewidth=2, markersize=8, color="#1f77b4")

    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel(metric.title(), fontsize=12)
    ax.set_title(title or f"SIARE Evolution Progress: {metric.title()}", fontsize=14)
    ax.set_ylim(0, 1)

    ax.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
