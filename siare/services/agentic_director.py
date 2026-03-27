"""Multi-turn agentic variation operator for hybrid evolution.

Replaces the single-turn Diagnostician + Architect pattern with an
agent session that can iterate: diagnose → propose → dry-run → evaluate
→ revise, up to a configurable iteration budget.

Inspired by AVO's agentic variation operators
(https://arxiv.org/html/2603.24517v1).

The AgenticDirector is a drop-in alternative to DirectorService —
the scheduler chooses which to use based on AgenticVariationConfig.mode.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import TYPE_CHECKING, Any

from siare.core.hooks import HookContext, fire_agentic_evolution_hook
from siare.core.models import (
    AgenticVariationConfig,
    MutationType,
    ProcessConfig,
    PromptGenome,
    SOPGene,
    SOPMutation,
    SupervisorDirective,
    VariationResult,
)
from siare.services.agent_session import AgentSession
from siare.services.circuit_breaker import CircuitBreaker
from siare.services.retry_handler import RetryHandler
from siare.services.variation_tools import VariationToolRegistry

if TYPE_CHECKING:
    from siare.services.llm_provider import LLMProvider

logger = logging.getLogger(__name__)


AGENTIC_DIRECTOR_SYSTEM_PROMPT = """\
You are an expert SOP (Standard Operating Procedure) mutation agent for \
multi-agent RAG systems.

Your task: analyze a parent SOP's performance, diagnose weaknesses, and \
propose a targeted mutation that improves quality.

You have access to tools:
{tools_description}

## Workflow

1. **Diagnose**: Examine the parent SOP's evaluation metrics and traces.
   Use inspect_trace and query_gene_pool to understand what went wrong.
2. **Research**: Use query_knowledge_base to find relevant patterns.
   Use compare_sops to see what worked in similar configurations.
3. **Propose**: Output a mutation in the required format (see below).
4. **Validate**: Use validate_mutation to check constraints before final \
output.

## Mutation Output Format

When you are ready to propose a mutation, output it in this exact format:

MUTATION_TYPE: <prompt_change|param_tweak|add_role|remove_role|rewire_graph>
TARGET_ROLE: <role_id if applicable, otherwise "none">
CHANGES: <detailed description of what to change>
NEW_CONTENT: <the actual new prompt text or parameter values>
RATIONALE: <why this will improve performance>

## Constraints
{constraints_description}

{directive_description}\
"""


class AgenticDirector:
    """Multi-turn agentic variation operator.

    Usage:
        director = AgenticDirector(
            llm_provider=provider,
            config=AgenticVariationConfig(),
        )
        result = director.vary(
            parent_gene=gene,
            parent_config=config,
            parent_genome=genome,
            metrics_to_optimize=["factuality"],
            mutation_types=[MutationType.PROMPT_CHANGE],
        )
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        config: AgenticVariationConfig | None = None,
        tool_registry: VariationToolRegistry | None = None,
        model: str | None = None,
        retry_handler: RetryHandler | None = None,
        circuit_breaker: CircuitBreaker | None = None,
    ) -> None:
        self.llm_provider = llm_provider
        self.config = config or AgenticVariationConfig()
        self.tool_registry = tool_registry or VariationToolRegistry()
        self.model = model or self.config.supervisorModel
        self.retry_handler = retry_handler or RetryHandler()
        self.circuit_breaker = circuit_breaker or CircuitBreaker(
            name="agentic_director_llm",
            config=CircuitBreaker.LLM_CIRCUIT_CONFIG,
        )

    def vary(
        self,
        parent_gene: SOPGene,
        parent_config: ProcessConfig,
        parent_genome: PromptGenome,
        metrics_to_optimize: list[str],
        mutation_types: list[MutationType],
        constraints: dict[str, Any] | None = None,
        sample_tasks: list[dict[str, Any]] | None = None,
        directive: SupervisorDirective | None = None,
    ) -> VariationResult:
        """Run an agentic variation session.

        Iterates: diagnose → propose → validate → (optionally dry-run)
        → revise, up to max_inner_iterations.

        Args:
            parent_gene: The parent SOPGene being varied.
            parent_config: The parent's ProcessConfig.
            parent_genome: The parent's PromptGenome.
            metrics_to_optimize: Metrics the evolution is optimizing.
            mutation_types: Allowed mutation types for this phase.
            constraints: Evolution constraints dict.
            sample_tasks: Tasks for dry-run evaluation.
            directive: Optional supervisor directive for guidance.

        Returns:
            VariationResult with the best mutation found, or empty.
        """
        budget = self.config.innerBudget.model_copy()
        hook_ctx = HookContext(
            correlation_id=str(uuid.uuid4()),
            metadata={
                "parent_sop_id": parent_config.id,
                "parent_version": parent_config.version,
            },
        )

        self._fire_hook(
            "on_variation_start", hook_ctx, parent_config, directive
        )

        system_prompt = self._build_system_prompt(
            mutation_types, constraints, directive
        )
        session = AgentSession(
            llm_provider=self.llm_provider,
            model=self.model,
            system_prompt=system_prompt,
            tools=self.tool_registry.get_tool_specs(),
            tool_executor=self.tool_registry,
            budget=budget,
            retry_handler=self.retry_handler,
            circuit_breaker=self.circuit_breaker,
            max_tool_rounds=5,
            temperature=0.5,
        )

        initial_prompt = self._build_initial_prompt(
            parent_gene, parent_config, parent_genome,
            metrics_to_optimize,
        )

        best_result: VariationResult | None = None

        for iteration in range(self.config.maxInnerIterations):
            if budget.exhausted():
                logger.info("Inner budget exhausted at iteration %d", iteration)
                break

            try:
                if iteration == 0:
                    response = session.turn(initial_prompt)
                else:
                    response = session.turn(
                        self._build_revision_prompt(best_result, iteration)
                    )
            except RuntimeError as e:
                logger.warning("Agent session error: %s", e)
                break

            mutation = self._parse_mutation_from_response(
                response, parent_config, parent_genome, mutation_types
            )

            self._fire_hook(
                "on_variation_iteration", hook_ctx,
                iteration, None, mutation is not None,
            )

            if mutation is None:
                continue  # No valid mutation parsed, retry

            # Accept first valid mutation. Future: compare against
            # dry-run baseline when sample_tasks are provided.
            best_result = VariationResult(
                mutation=mutation,
                quality=None,
                iterationsUsed=iteration + 1,
                innerBudgetUsed=budget.usage_summary(),
                reason="improvement",
            )
            break

        if best_result is None:
            best_result = VariationResult(
                reason="no_improvement",
                iterationsUsed=self.config.maxInnerIterations,
                innerBudgetUsed=budget.usage_summary(),
            )

        self._fire_hook("on_variation_complete", hook_ctx, best_result)
        return best_result

    def _build_system_prompt(
        self,
        mutation_types: list[MutationType],
        constraints: dict[str, Any] | None,
        directive: SupervisorDirective | None,
    ) -> str:
        """Build the system prompt with tools, constraints, and directive."""
        tools_desc = ""
        for spec in self.tool_registry.get_tool_specs():
            tools_desc += f"- {spec.name}: {spec.description}\n"

        constraints_desc = "No specific constraints."
        if constraints:
            parts: list[str] = []
            if "maxRoles" in constraints:
                parts.append(f"Max roles: {constraints['maxRoles']}")
            if "mandatoryRoles" in constraints:
                parts.append(
                    f"Mandatory roles: {constraints['mandatoryRoles']}"
                )
            if "disallowedMutationTypes" in constraints:
                parts.append(
                    "Disallowed mutations: "
                    f"{constraints['disallowedMutationTypes']}"
                )
            allowed = [mt.value for mt in mutation_types]
            parts.append(f"Allowed mutation types: {allowed}")
            constraints_desc = "\n".join(parts)

        directive_desc = ""
        if directive:
            directive_desc = (
                f"\n## Supervisor Directive\n"
                f"Strategy: {directive.strategy}\n"
                f"Focus: {directive.focusArea}\n"
                f"Target: {directive.explorationTarget}\n"
                f"Rationale: {directive.rationale}\n"
                f"Follow this directive when proposing mutations."
            )

        return AGENTIC_DIRECTOR_SYSTEM_PROMPT.format(
            tools_description=tools_desc or "No tools available.",
            constraints_description=constraints_desc,
            directive_description=directive_desc,
        )

    def _build_initial_prompt(
        self,
        parent_gene: SOPGene,
        parent_config: ProcessConfig,
        parent_genome: PromptGenome,
        metrics_to_optimize: list[str],
    ) -> str:
        """Build the first user prompt with parent SOP context."""
        parts: list[str] = []

        parts.append("## Parent SOP")
        parts.append(f"ID: {parent_config.id} v{parent_config.version}")
        parts.append(f"Roles: {[r.id for r in parent_config.roles]}")
        parts.append(
            f"Graph: {[(e.from_, e.to) for e in parent_config.graph]}"
        )

        parts.append("\n## Role Prompts")
        for role in parent_config.roles:
            prompt = parent_genome.rolePrompts.get(role.promptRef)
            if prompt:
                content = prompt.content[:300]
                suffix = "..." if len(prompt.content) > 300 else ""
                parts.append(f"- {role.id}: {content}{suffix}")

        parts.append("\n## Performance Metrics")
        for metric_id in metrics_to_optimize:
            if metric_id in parent_gene.aggregatedMetrics:
                mean = parent_gene.get_metric_mean(metric_id)
                parts.append(f"- {metric_id}: {mean:.3f}")
            else:
                parts.append(f"- {metric_id}: (not evaluated)")

        quality = parent_gene.get_metric_mean("weighted_aggregate")
        parts.append(f"\nOverall quality: {quality:.3f}")

        parts.append(
            "\nDiagnose the weaknesses and propose a mutation that "
            "improves the metrics above."
        )

        return "\n".join(parts)

    def _build_revision_prompt(
        self,
        previous_result: VariationResult | None,
        iteration: int,
    ) -> str:
        """Build a revision prompt when the previous attempt failed to parse."""
        return (
            f"Iteration {iteration}: Your previous response did not "
            f"contain a valid mutation. Please output a mutation in the "
            f"required format:\n\n"
            f"MUTATION_TYPE: <type>\n"
            f"TARGET_ROLE: <role_id>\n"
            f"CHANGES: <description>\n"
            f"NEW_CONTENT: <content>\n"
            f"RATIONALE: <why>"
        )

    def _parse_mutation_from_response(
        self,
        response: str,
        parent_config: ProcessConfig,
        parent_genome: PromptGenome,
        allowed_types: list[MutationType],
    ) -> SOPMutation | None:
        """Parse a mutation proposal from the LLM response."""
        mutation_type: MutationType | None = None
        target_role = ""
        changes = ""
        new_content = ""
        rationale = ""

        current_section = ""
        for line in response.split("\n"):
            stripped = line.strip()
            if stripped.startswith("MUTATION_TYPE:"):
                type_str = stripped.split(":", 1)[1].strip().lower()
                for mt in MutationType:
                    if mt.value == type_str:
                        mutation_type = mt
                        break
                current_section = "mutation_type"
            elif stripped.startswith("TARGET_ROLE:"):
                target_role = stripped.split(":", 1)[1].strip()
                current_section = "target_role"
            elif stripped.startswith("CHANGES:"):
                changes = stripped.split(":", 1)[1].strip()
                current_section = "changes"
            elif stripped.startswith("NEW_CONTENT:"):
                new_content = stripped.split(":", 1)[1].strip()
                current_section = "new_content"
            elif stripped.startswith("RATIONALE:"):
                rationale = stripped.split(":", 1)[1].strip()
                current_section = "rationale"
            elif current_section == "new_content" and stripped:
                new_content += "\n" + stripped
            elif current_section == "rationale" and stripped:
                rationale += " " + stripped

        if mutation_type is None:
            logger.debug("No valid MUTATION_TYPE found in response")
            return None

        if mutation_type not in allowed_types:
            logger.debug(
                "Mutation type %s not in allowed types %s",
                mutation_type, allowed_types,
            )
            return None

        new_config = parent_config.model_copy(deep=True)
        new_genome = parent_genome.model_copy(deep=True)

        if mutation_type == MutationType.PROMPT_CHANGE and new_content:
            if target_role and target_role != "none":
                for role in new_config.roles:
                    if role.id == target_role:
                        prompt = new_genome.rolePrompts.get(role.promptRef)
                        if prompt:
                            prompt.content = new_content.strip()
                        break

        new_config.version = self._bump_version(
            parent_config.version, mutation_type
        )

        return SOPMutation(
            parentSopId=parent_config.id,
            parentVersion=parent_config.version,
            newConfig=new_config,
            newPromptGenome=new_genome,
            rationale=rationale or changes or "Agent-proposed mutation",
            mutationType=mutation_type,
        )

    def _bump_version(
        self, version: str, mutation_type: MutationType
    ) -> str:
        """Bump version based on mutation type."""
        parts = version.split(".")
        if len(parts) != 3:
            return f"{version}-mut"

        try:
            major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            return f"{version}-mut"

        suffix = uuid.uuid4().hex[:6]
        topology_types = {
            MutationType.ADD_ROLE,
            MutationType.REMOVE_ROLE,
            MutationType.REWIRE_GRAPH,
        }
        if mutation_type in topology_types:
            return f"{major + 1}.0.0-{suffix}"
        elif mutation_type == MutationType.PROMPT_CHANGE:
            return f"{major}.{minor + 1}.0-{suffix}"
        else:
            return f"{major}.{minor}.{patch + 1}-{suffix}"

    def _fire_hook(
        self, hook_name: str, ctx: HookContext, *args: Any, **kwargs: Any
    ) -> Any:
        """Fire an agentic evolution hook safely from sync context."""
        from siare.core.hooks import HookRegistry

        if HookRegistry.get_agentic_evolution_hooks() is None:
            return None
        try:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(
                    fire_agentic_evolution_hook(
                        hook_name, ctx, *args, **kwargs
                    )
                )
                return None
            except RuntimeError:
                return asyncio.run(
                    fire_agentic_evolution_hook(
                        hook_name, ctx, *args, **kwargs
                    )
                )
        except Exception as e:
            logger.warning("Failed to fire hook %s: %s", hook_name, e)
            return None
