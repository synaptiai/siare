"""Generate comparison reports for agentic variation mode benchmarks."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class AgenticComparisonReport:
    """Generates comparison reports across variation modes.

    Compares single_turn vs agentic vs adaptive evolution runs
    with side-by-side metrics, learning curves, and cost analysis.

    Example:
        >>> report = AgenticComparisonReport(results, mode_configs)
        >>> report.save("output/")
        >>> md = report.generate_markdown()
    """

    def __init__(
        self,
        results: dict[str, Any],  # {mode_name: SelfImprovementResult.summary()}
        mode_configs: dict[str, Any],  # {mode_name: config dict}
    ) -> None:
        """Initialize the comparison report generator.

        Args:
            results: Mapping of mode name to SelfImprovementResult.summary() dict.
                     Each value contains keys like 'dataset', 'model', 'generations',
                     'improvements', 'converged', 'convergence_generation',
                     'total_time_seconds', and optionally 'agentic'.
            mode_configs: Mapping of mode name to its configuration dict.
        """
        self._results = results
        self._mode_configs = mode_configs
        self._modes = sorted(results.keys())

    # =========================================================================
    # Public API
    # =========================================================================

    def generate_markdown(self) -> str:
        """Generate full Markdown comparison report.

        Returns:
            Markdown-formatted report string
        """
        sections = [
            self._title_section(),
            self._metadata_section(),
            self._mode_comparison_section(),
            self._winner_analysis_section(),
            self._agentic_metrics_section(),
            self._recommendations_section(),
            self._footer_section(),
        ]
        return "\n\n".join(sections)

    def generate_json(self) -> str:
        """Return the results dict as formatted JSON.

        Returns:
            JSON string with 2-space indentation
        """
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "modes": self._modes,
            "results": self._results,
            "mode_configs": self._mode_configs,
        }
        return json.dumps(payload, indent=2, default=str)

    def save(self, output_dir: str) -> None:
        """Save markdown and JSON files with timestamp.

        Creates both ``agentic_comparison_<timestamp>.md`` and
        ``agentic_comparison_<timestamp>.json`` in *output_dir*.

        Args:
            output_dir: Directory to write report files into.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out.joinpath(f"agentic_comparison_{ts}.md").write_text(
            self.generate_markdown()
        )
        out.joinpath(f"agentic_comparison_{ts}.json").write_text(
            self.generate_json()
        )

    # =========================================================================
    # Markdown Section Generators
    # =========================================================================

    def _title_section(self) -> str:
        """Generate title section."""
        return "# Agentic Variation Mode Comparison"

    def _metadata_section(self) -> str:
        """Generate metadata section with dataset, model, and run info."""
        first = next(iter(self._results.values()), {})
        timestamp = datetime.now(timezone.utc).isoformat()
        dataset = first.get("dataset", "unknown")
        model = first.get("model", "unknown")
        generations = first.get("generations", "N/A")
        samples = self._mode_configs.get(
            self._modes[0], {}
        ).get("max_samples", "N/A") if self._modes else "N/A"

        return f"""## Metadata

- **Dataset:** {dataset}
- **Model:** {model}
- **Date:** {timestamp}
- **Generations per mode:** {generations}
- **Samples:** {samples}
- **Modes compared:** {', '.join(self._modes)}"""

    def _mode_comparison_section(self) -> str:
        """Generate the main side-by-side comparison table."""
        rows = self._build_comparison_rows()

        # Build dynamic header
        header = "| Metric |"
        sep = "|--------|"
        for mode in self._modes:
            header += f" {mode} |"
            sep += "---------|"

        lines = ["## Mode Comparison", "", header, sep]
        for row in rows:
            lines.append(row)

        return "\n".join(lines)

    def _winner_analysis_section(self) -> str:
        """Determine and report which mode won on quality, efficiency, speed."""
        quality_winner = self._find_winner("quality")
        speed_winner = self._find_winner("speed")
        efficiency_winner = self._find_winner("efficiency")

        return f"""## Winner Analysis

| Dimension | Winner | Rationale |
|-----------|--------|-----------|
| Quality (best final accuracy) | {quality_winner[0]} | {quality_winner[1]} |
| Speed (lowest total time) | {speed_winner[0]} | {speed_winner[1]} |
| Efficiency (best accuracy/time) | {efficiency_winner[0]} | {efficiency_winner[1]} |"""

    def _agentic_metrics_section(self) -> str:
        """Generate agentic-specific metrics for modes that used agentic variation."""
        agentic_modes = {
            mode: data
            for mode, data in self._results.items()
            if data.get("agentic") is not None
        }

        if not agentic_modes:
            return """## Agentic-Specific Metrics

*No agentic variation modes present in this comparison.*"""

        # Build header
        header = "| Metric |"
        sep = "|--------|"
        mode_names = sorted(agentic_modes.keys())
        for mode in mode_names:
            header += f" {mode} |"
            sep += "---------|"

        rows: list[str] = []

        # Generations using agentic
        row = "| Generations using agentic |"
        for mode in mode_names:
            ag = agentic_modes[mode].get("agentic", {})
            val = ag.get("generations_using_agentic", "N/A")
            row += f" {val} |"
        rows.append(row)

        # Total inner iterations
        row = "| Total inner iterations |"
        for mode in mode_names:
            ag = agentic_modes[mode].get("agentic", {})
            val = ag.get("total_inner_iterations", "N/A")
            row += f" {val} |"
        rows.append(row)

        # Mode label
        row = "| Variation mode |"
        for mode in mode_names:
            ag = agentic_modes[mode].get("agentic", {})
            val = ag.get("mode", "N/A")
            row += f" {val} |"
        rows.append(row)

        return "\n".join([
            "## Agentic-Specific Metrics",
            "",
            header,
            sep,
            *rows,
        ])

    def _recommendations_section(self) -> str:
        """Generate data-driven recommendations for when to use each mode."""
        recs: list[str] = []

        for mode in self._modes:
            data = self._results[mode]
            rec = self._recommend_for_mode(mode, data)
            recs.append(f"- **{mode}:** {rec}")

        return "\n".join([
            "## Recommendations",
            "",
            *recs,
        ])

    def _footer_section(self) -> str:
        """Generate footer section."""
        return """---

*Report generated by SIARE Agentic Comparison Benchmark Suite*"""

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _build_comparison_rows(self) -> list[str]:
        """Build individual rows for the mode comparison table.

        Returns:
            List of markdown table row strings.
        """
        rows: list[str] = []

        # Final accuracy
        row = "| Final accuracy |"
        for mode in self._modes:
            val = self._best_final_accuracy(mode)
            row += f" {val:.3f} |" if val is not None else " N/A |"
        rows.append(row)

        # Improvement %
        row = "| Improvement % |"
        for mode in self._modes:
            pct = self._best_improvement_pct(mode)
            row += f" +{pct:.1f}% |" if pct is not None else " N/A |"
        rows.append(row)

        # Convergence generation
        row = "| Converged at gen |"
        for mode in self._modes:
            data = self._results[mode]
            if data.get("converged"):
                row += f" {data.get('convergence_generation', 'N/A')} |"
            else:
                row += " -- |"
        rows.append(row)

        # Total time
        row = "| Total time (s) |"
        for mode in self._modes:
            t = self._results[mode].get("total_time_seconds")
            row += f" {t:.1f} |" if t is not None else " N/A |"
        rows.append(row)

        # p-value for best metric
        row = "| p-value (accuracy) |"
        for mode in self._modes:
            p = self._best_p_value(mode)
            row += f" {p:.4f} |" if p is not None else " N/A |"
        rows.append(row)

        return rows

    def _best_final_accuracy(self, mode: str) -> float | None:
        """Extract the best final accuracy from a mode's improvements dict."""
        improvements = self._results[mode].get("improvements", {})
        if not improvements:
            return None
        # Pick the metric with the highest final value
        best: float | None = None
        for metric_data in improvements.values():
            final = metric_data.get("final")
            if final is not None and (best is None or final > best):
                best = final
        return best

    def _best_improvement_pct(self, mode: str) -> float | None:
        """Extract the best improvement percentage from a mode's results."""
        improvements = self._results[mode].get("improvements", {})
        if not improvements:
            return None
        best: float | None = None
        for metric_data in improvements.values():
            pct = metric_data.get("improvement_pct")
            if pct is not None and (best is None or pct > best):
                best = pct
        return best

    def _best_p_value(self, mode: str) -> float | None:
        """Extract the best (lowest) p-value from a mode's improvements."""
        improvements = self._results[mode].get("improvements", {})
        if not improvements:
            return None
        best: float | None = None
        for metric_data in improvements.values():
            p = metric_data.get("p_value")
            if p is not None and (best is None or p < best):
                best = p
        return best

    def _find_winner(
        self, dimension: str
    ) -> tuple[str, str]:
        """Find the winning mode for a given dimension.

        Args:
            dimension: One of 'quality', 'speed', or 'efficiency'.

        Returns:
            Tuple of (winner_mode_name, rationale_string).
        """
        if dimension == "quality":
            scores = {
                m: self._best_final_accuracy(m) for m in self._modes
            }
            scores_clean = {m: v for m, v in scores.items() if v is not None}
            if not scores_clean:
                return ("N/A", "No accuracy data available")
            winner = max(scores_clean, key=lambda m: scores_clean[m])
            return (winner, f"Highest final accuracy: {scores_clean[winner]:.3f}")

        if dimension == "speed":
            times = {
                m: self._results[m].get("total_time_seconds")
                for m in self._modes
            }
            times_clean = {m: v for m, v in times.items() if v is not None}
            if not times_clean:
                return ("N/A", "No timing data available")
            winner = min(times_clean, key=lambda m: times_clean[m])
            return (winner, f"Lowest total time: {times_clean[winner]:.1f}s")

        # efficiency = accuracy / time
        eff: dict[str, float] = {}
        for m in self._modes:
            acc = self._best_final_accuracy(m)
            t = self._results[m].get("total_time_seconds")
            if acc is not None and t is not None and t > 0:
                eff[m] = acc / t
        if not eff:
            return ("N/A", "Insufficient data for efficiency calculation")
        winner = max(eff, key=lambda m: eff[m])
        return (winner, f"Best accuracy/time ratio: {eff[winner]:.4f}")

    def _recommend_for_mode(self, mode: str, data: dict[str, Any]) -> str:
        """Generate a one-line recommendation for a given mode.

        Args:
            mode: The variation mode name.
            data: The summary dict for that mode.

        Returns:
            Human-readable recommendation string.
        """
        agentic_info = data.get("agentic")
        time_s = data.get("total_time_seconds", 0)
        converged = data.get("converged", False)
        conv_gen = data.get("convergence_generation")

        if agentic_info is None:
            # single_turn / non-agentic
            if converged and conv_gen is not None:
                return (
                    f"Best for fast iteration and tight budgets. "
                    f"Converged at generation {conv_gen} in {time_s:.0f}s."
                )
            return "Lightweight baseline; use when compute budget is limited."

        agentic_mode = agentic_info.get("mode", mode)
        inner_iters = agentic_info.get("total_inner_iterations", 0)
        agentic_gens = agentic_info.get("generations_using_agentic", 0)

        if agentic_mode == "adaptive":
            return (
                f"Good balance of quality and cost. Used agentic variation in "
                f"{agentic_gens} generations ({inner_iters} inner iterations). "
                f"Consider when you want quality gains without full agentic overhead."
            )

        # fully agentic
        return (
            f"Highest quality potential at higher compute cost "
            f"({inner_iters} inner iterations across {agentic_gens} generations). "
            f"Use when maximizing performance is the top priority."
        )
