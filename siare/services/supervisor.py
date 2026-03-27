"""Supervisor agent for stagnation analysis and exploration redirection.

When the evolution loop stagnates (no quality improvement over a window
of generations), the supervisor reviews the evolutionary trajectory
and produces a directive that steers the next generation's variation
toward unexplored or promising territory.

Inspired by AVO's self-supervision mechanism.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from siare.core.models import (
    MutationType,
    SupervisorContext,
    SupervisorDirective,
)
from siare.services.circuit_breaker import CircuitBreaker
from siare.services.llm_provider import LLMMessage
from siare.services.retry_handler import RetryHandler

if TYPE_CHECKING:
    from siare.services.gene_pool import GenePool
    from siare.services.llm_provider import LLMProvider
    from siare.services.qd_grid import QDGridManager

logger = logging.getLogger(__name__)

SUPERVISOR_SYSTEM_PROMPT = """\
You are an evolution supervisor for a multi-agent RAG optimization system.

The evolution has stagnated — quality has not improved for several \
generations. Your job is to analyze WHY and propose a new exploration \
direction.

Consider:
1. QD grid coverage: Are there unexplored regions of the feature space?
2. Mutation type patterns: Which types have been tried? Which succeeded?
3. Pareto frontier gaps: Where are there missing tradeoffs?
4. Common failure modes: What keeps going wrong?

You MUST output a directive in this exact format:

STRATEGY: <what general approach to take next>
FOCUS_AREA: <topology | prompts | parameters>
MUTATION_TYPES: <comma-separated list from: prompt_change, param_tweak, \
add_role, remove_role, rewire_graph, crossover>
EXPLORATION_TARGET: <specific QD grid region or Pareto gap to target>
RATIONALE: <why this direction is promising>
"""


class SupervisorAgent:
    """Analyzes evolutionary trajectory and redirects exploration.

    Usage:
        supervisor = SupervisorAgent(
            llm_provider=provider,
            gene_pool=gene_pool,
            qd_grid=qd_grid,
        )
        directive = supervisor.analyze_and_redirect(
            quality_history=[0.5, 0.6, 0.65, 0.65, 0.65],
        )
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        gene_pool: GenePool,
        qd_grid: QDGridManager,
        model: str = "gpt-5",
        retry_handler: RetryHandler | None = None,
        circuit_breaker: CircuitBreaker | None = None,
    ) -> None:
        self.llm_provider = llm_provider
        self.gene_pool = gene_pool
        self.qd_grid = qd_grid
        self.model = model
        self.retry_handler = retry_handler or RetryHandler()
        self.circuit_breaker = circuit_breaker or CircuitBreaker(
            name="supervisor_llm",
            config=CircuitBreaker.LLM_CIRCUIT_CONFIG,
        )

    def analyze_and_redirect(
        self,
        quality_history: list[float],
        recent_generations: int = 10,
        job_constraints: dict[str, Any] | None = None,
    ) -> SupervisorDirective:
        """Analyze stagnation and produce a redirection directive.

        Args:
            quality_history: Best quality per generation.
            recent_generations: How many recent generations to examine.
            job_constraints: Evolution constraints for context.

        Returns:
            SupervisorDirective guiding the next variation.
        """
        context = self._gather_context(quality_history, recent_generations)
        prompt = self._build_analysis_prompt(context, job_constraints)

        messages = [
            LLMMessage(role="system", content=SUPERVISOR_SYSTEM_PROMPT),
            LLMMessage(role="user", content=prompt),
        ]

        from siare.services.circuit_breaker import CircuitBreakerOpenError
        from siare.services.retry_handler import RetryExhausted

        try:
            response = self.circuit_breaker.call(
                lambda: self.retry_handler.execute_with_retry(
                    self.llm_provider.complete,
                    messages=messages,
                    model=self.model,
                    temperature=0.7,
                    retry_config=RetryHandler.LLM_RETRY_CONFIG,
                    component="Supervisor",
                    operation="analyze_stagnation",
                )
            )
        except (CircuitBreakerOpenError, RetryExhausted) as e:
            logger.warning("Supervisor LLM call failed: %s", e)
            return self._fallback_directive(context)

        return self._parse_directive(response.content, context)

    def _gather_context(
        self,
        quality_history: list[float],
        recent_generations: int,
    ) -> SupervisorContext:
        """Gather evidence from the gene pool and QD grid."""
        qd_coverage: dict[str, Any] = {}
        pareto_frontier: list[dict[str, Any]] = []
        recent_genes: list[dict[str, Any]] = []
        diversity_stats: dict[str, Any] = {}

        try:
            qd_coverage = self.qd_grid.get_visit_stats()
        except Exception as e:
            logger.warning("Failed to get QD visit stats: %s", e)

        try:
            frontier = self.gene_pool.get_pareto_frontier()
            pareto_frontier = [
                {
                    "sopId": g.sopId,
                    "version": g.version,
                    "quality": g.get_metric_mean("weighted_aggregate"),
                }
                for g in frontier
            ]
        except Exception as e:
            logger.warning("Failed to get Pareto frontier: %s", e)

        try:
            genes = self.gene_pool.get_genes_from_recent_generations(
                lookback=recent_generations,
            )
            recent_genes = [
                {
                    "sopId": g.sopId,
                    "version": g.version,
                    "generation": g.generation,
                    "quality": g.get_metric_mean("weighted_aggregate"),
                }
                for g in genes
            ]
        except Exception as e:
            logger.warning("Failed to get recent genes: %s", e)

        try:
            diversity_stats = self.gene_pool.get_diversity_stats()
        except Exception as e:
            logger.warning("Failed to get diversity stats: %s", e)

        mutation_rates = self._compute_mutation_success_rates(recent_genes)

        return SupervisorContext(
            qdCoverage=qd_coverage,
            paretoFrontier=pareto_frontier,
            recentGenes=recent_genes,
            diversityStats=diversity_stats,
            mutationSuccessRates=mutation_rates,
            qualityHistory=quality_history,
        )

    def _compute_mutation_success_rates(
        self,
        recent_genes: list[dict[str, Any]],
    ) -> dict[str, float]:
        """Compute success rates per mutation type from recent genes.

        A placeholder — full implementation would track which mutations
        led to quality improvements. Returns empty dict for now.
        """
        return {}

    def _build_analysis_prompt(
        self,
        context: SupervisorContext,
        constraints: dict[str, Any] | None,
    ) -> str:
        """Build the user prompt with trajectory evidence."""
        parts: list[str] = []

        parts.append("## Quality History")
        if context.qualityHistory:
            recent = context.qualityHistory[-20:]
            parts.append(
                f"Last {len(recent)} generations: "
                f"{[f'{q:.3f}' for q in recent]}"
            )
            if len(context.qualityHistory) > 1:
                best = max(context.qualityHistory)
                parts.append(f"Best quality ever: {best:.3f}")
        else:
            parts.append("No quality history available.")

        parts.append("\n## QD Grid Coverage")
        if context.qdCoverage:
            parts.append(json.dumps(context.qdCoverage, indent=2))
        else:
            parts.append("No QD grid data available.")

        parts.append("\n## Pareto Frontier")
        if context.paretoFrontier:
            for p in context.paretoFrontier[:10]:
                parts.append(
                    f"- {p.get('sopId', '?')} v{p.get('version', '?')}: "
                    f"quality={p.get('quality', 0):.3f}"
                )
        else:
            parts.append("No Pareto frontier data.")

        parts.append("\n## Population Diversity")
        if context.diversityStats:
            parts.append(json.dumps(context.diversityStats, indent=2))

        parts.append("\n## Mutation Success Rates")
        if context.mutationSuccessRates:
            for mt, rate in context.mutationSuccessRates.items():
                parts.append(f"- {mt}: {rate:.1%}")
        else:
            parts.append("No mutation success rate data available.")

        if constraints:
            parts.append("\n## Constraints")
            parts.append(json.dumps(constraints, indent=2, default=str))

        parts.append(
            "\nAnalyze the stagnation and propose a new exploration "
            "direction."
        )
        return "\n".join(parts)

    def _parse_directive(
        self,
        content: str,
        context: SupervisorContext,
    ) -> SupervisorDirective:
        """Parse the supervisor's LLM response into a directive."""
        strategy = ""
        focus_area = "prompts"
        mutation_types_raw = ""
        exploration_target = ""
        rationale = ""

        for line in content.split("\n"):
            stripped = line.strip()
            if stripped.startswith("STRATEGY:"):
                strategy = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("FOCUS_AREA:"):
                focus_area = stripped.split(":", 1)[1].strip().lower()
            elif stripped.startswith("MUTATION_TYPES:"):
                mutation_types_raw = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("EXPLORATION_TARGET:"):
                exploration_target = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("RATIONALE:"):
                rationale = stripped.split(":", 1)[1].strip()

        mutation_types = self._parse_mutation_types(mutation_types_raw)
        if not mutation_types:
            mutation_types = [MutationType.PROMPT_CHANGE]

        if focus_area not in ("topology", "prompts", "parameters"):
            focus_area = "prompts"

        return SupervisorDirective(
            strategy=strategy or "explore alternative approaches",
            focusArea=focus_area,
            mutationTypes=mutation_types,
            explorationTarget=exploration_target or "underexplored regions",
            rationale=rationale or "stagnation detected",
        )

    def _parse_mutation_types(
        self, raw: str
    ) -> list[MutationType]:
        """Parse comma-separated mutation type strings."""
        types: list[MutationType] = []
        for part in raw.split(","):
            cleaned = part.strip().lower()
            for mt in MutationType:
                if mt.value == cleaned:
                    types.append(mt)
                    break
        return types

    def _fallback_directive(
        self,
        context: SupervisorContext,
    ) -> SupervisorDirective:
        """Generate a reasonable directive when LLM call fails."""
        if context.qdCoverage.get("cells_unvisited", 0) > 0:
            return SupervisorDirective(
                strategy="explore unvisited QD cells",
                focusArea="topology",
                mutationTypes=[
                    MutationType.ADD_ROLE,
                    MutationType.REWIRE_GRAPH,
                ],
                explorationTarget="unvisited QD grid cells",
                rationale="LLM unavailable; defaulting to QD exploration",
            )
        return SupervisorDirective(
            strategy="diversify prompts",
            focusArea="prompts",
            mutationTypes=[MutationType.PROMPT_CHANGE],
            explorationTarget="prompt diversity",
            rationale="LLM unavailable; defaulting to prompt variation",
        )
