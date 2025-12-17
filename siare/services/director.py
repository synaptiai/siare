"""Director Service - AI-driven mutation proposals (Diagnostician + Architect)"""

import asyncio
import json
import logging
import re
import uuid
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from siare.core.models import PromptEvolutionResult
    from siare.services.execution_engine import ExecutionTrace
    from siare.services.prompt_evolution.orchestrator import PromptEvolutionOrchestrator

from siare.core.hooks import HookContext, HookRegistry, fire_evolution_hook
from siare.core.models import (
    Diagnosis,
    EvaluationVector,
    GraphEdge,
    MetaConfig,
    MutationType,
    ProcessConfig,
    PromptConstraints,
    PromptGenome,
    RoleConfig,
    RolePrompt,
    SOPGene,
    SOPMutation,
)
from siare.services.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitBreakerRegistry,
)
from siare.services.execution_engine import ExecutionEngine
from siare.services.llm_provider import LLMMessage, LLMProvider
from siare.services.retry_handler import RetryExhausted, RetryHandler

logger = logging.getLogger(__name__)

# Separator for multi-part rationale strings
RATIONALE_SEPARATOR = " | "


class Diagnostician:
    """
    Analyzes SOP performance and identifies weaknesses

    Responsibilities:
    - Analyze evaluation artifacts and metrics
    - Identify failure modes and bottlenecks
    - Perform root cause analysis
    - Generate actionable recommendations
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        meta_config: MetaConfig | None = None,
        model: str = "gpt-5",
        retry_handler: RetryHandler | None = None,
        circuit_breaker: CircuitBreaker | None = None,
    ):
        """
        Initialize Diagnostician

        Args:
            llm_provider: LLM provider for analysis
            meta_config: MetaConfig with diagnostician prompt (uses default if None)
            model: Model to use for diagnosis (default: "gpt-5")
            retry_handler: Retry handler for transient failures (creates default if None)
            circuit_breaker: Circuit breaker for fault tolerance (creates default if None)
        """
        self.llm_provider = llm_provider
        self.meta_config = meta_config
        self.model = model

        # Initialize error handling
        self.retry_handler = retry_handler or RetryHandler()
        self.circuit_breaker = circuit_breaker or CircuitBreaker(
            name="diagnostician_llm",
            config=CircuitBreaker.LLM_CIRCUIT_CONFIG,
        )

    def diagnose(
        self,
        sop_gene: SOPGene,
        sop_config: ProcessConfig,
        prompt_genome: PromptGenome,
        metrics_to_optimize: list[str],
    ) -> Diagnosis:
        """
        Diagnose SOP performance issues

        Args:
            sop_gene: SOPGene with evaluation data
            sop_config: ProcessConfig
            prompt_genome: PromptGenome
            metrics_to_optimize: Metrics to focus on

        Returns:
            Diagnosis with weaknesses and recommendations
        """

        # Build diagnostic prompt
        diagnostic_prompt = self._build_diagnostic_prompt(
            sop_gene, sop_config, prompt_genome, metrics_to_optimize
        )

        # Get director prompt
        system_prompt = self._get_director_prompt()

        # Call LLM with error handling (retry + circuit breaker)
        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=diagnostic_prompt),
        ]

        try:
            # Wrap with circuit breaker first, then retry handler
            response = self.circuit_breaker.call(
                lambda: self.retry_handler.execute_with_retry(
                    self.llm_provider.complete,
                    messages=messages,
                    model=self.model,
                    temperature=0.3,
                    retry_config=RetryHandler.LLM_RETRY_CONFIG,
                    component="Diagnostician",
                    operation="llm_diagnose",
                )
            )
        except CircuitBreakerOpenError as e:
            logger.exception("Circuit breaker open for diagnosis LLM call")
            raise RuntimeError("Diagnostician LLM circuit breaker is open - service temporarily unavailable") from e
        except RetryExhausted as e:
            logger.exception("All retry attempts exhausted for diagnosis")
            raise RuntimeError("Failed to complete diagnosis after all retry attempts") from e

        # Parse diagnosis from response
        return self._parse_diagnosis(response.content)

    def _build_diagnostic_prompt(
        self,
        sop_gene: SOPGene,
        sop_config: ProcessConfig,
        prompt_genome: PromptGenome,
        metrics_to_optimize: list[str],
    ) -> str:
        """Build prompt for diagnosis"""

        # Collect metrics summary
        metrics_summary: list[str] = []
        for metric_id in metrics_to_optimize:
            score = sop_gene.get_metric_mean(metric_id)
            metrics_summary.append(f"- {metric_id}: {score:.3f}")

        # Collect evaluation artifacts
        artifacts_summary: list[str] = []
        for eval_vec in sop_gene.evaluations:
            if eval_vec.artifacts:
                if eval_vec.artifacts.llmFeedback:
                    for metric_id, feedback in eval_vec.artifacts.llmFeedback.items():
                        artifacts_summary.append(f"[{metric_id}] {feedback}")

                if eval_vec.artifacts.failureModes:
                    artifacts_summary.extend(
                        f"Failure: {mode}" for mode in eval_vec.artifacts.failureModes
                    )

                if eval_vec.artifacts.toolErrors:
                    artifacts_summary.extend(
                        f"Tool Error: {error}" for error in eval_vec.artifacts.toolErrors
                    )

        # Build SOP structure summary
        structure_summary = f"""
SOP Structure:
- ID: {sop_config.id}
- Version: {sop_config.version}
- Roles: {len(sop_config.roles)} ({", ".join(r.id for r in sop_config.roles)})
- Graph Edges: {len(sop_config.graph)}
- Tools: {", ".join(sop_config.tools) if sop_config.tools else "None"}

Role Details:
"""
        for role in sop_config.roles:
            prompt_content = prompt_genome.rolePrompts.get(role.promptRef, None)
            prompt_preview = prompt_content.content[:200] if prompt_content else "N/A"
            structure_summary += f"""
- {role.id}:
  - Model: {role.model}
  - Tools: {", ".join(role.tools) if role.tools else "None"}
  - Prompt: {prompt_preview}...
"""

        return f"""
Analyze the following SOP and identify performance weaknesses.

{structure_summary}

Performance Metrics:
{chr(10).join(metrics_summary)}

Evaluation Artifacts:
{chr(10).join(artifacts_summary[:10])}  # Limit to first 10

Your task:
1. Identify the PRIMARY weakness impacting the metrics most
2. List 2-3 secondary weaknesses
3. Identify any strengths to preserve
4. Perform root cause analysis
5. Provide specific, actionable recommendations

Format your response as:
PRIMARY_WEAKNESS: <description>
SECONDARY_WEAKNESSES: <list>
STRENGTHS: <list>
ROOT_CAUSE: <analysis>
RECOMMENDATIONS: <numbered list>
"""

    def _get_director_prompt(self) -> str:
        """Get director system prompt from MetaConfig"""

        if self.meta_config and self.meta_config.directorPrompt:
            return self.meta_config.directorPrompt.content

        # Default diagnostician prompt
        return """
You are an expert SOP (Standard Operating Procedure) diagnostician for AI agent systems.

Your role is to:
1. Analyze agent performance metrics and execution traces
2. Identify failure modes and bottlenecks
3. Perform root cause analysis
4. Recommend specific improvements

Focus on:
- Prompt quality and clarity
- Role responsibilities and boundaries
- Information flow between agents
- Tool usage effectiveness
- Graph structure efficiency

Be specific and actionable in your recommendations.
"""

    def _parse_diagnosis(self, llm_response: str) -> Diagnosis:
        """Parse LLM response into Diagnosis object"""

        lines = llm_response.split("\n")

        primary_weakness = ""
        secondary_weaknesses: list[str] = []
        strengths: list[str] = []
        root_cause = ""
        recommendations: list[str] = []

        current_section: str | None = None

        for line in lines:
            line = line.strip()

            if line.startswith("PRIMARY_WEAKNESS:"):
                primary_weakness = line.split(":", 1)[1].strip()
                current_section = "primary"

            elif line.startswith("SECONDARY_WEAKNESSES:"):
                current_section = "secondary"

            elif line.startswith("STRENGTHS:"):
                current_section = "strengths"

            elif line.startswith("ROOT_CAUSE:"):
                current_section = "root_cause"

            elif line.startswith("RECOMMENDATIONS:"):
                current_section = "recommendations"

            elif line and current_section:
                # Parse list items
                if current_section == "secondary":
                    if line.startswith(("-", "•")):
                        secondary_weaknesses.append(line[1:].strip())
                    elif line and not line.startswith(
                        ("PRIMARY", "SECONDARY", "STRENGTHS", "ROOT", "RECOMMENDATIONS")
                    ):
                        secondary_weaknesses.append(line)

                elif current_section == "strengths":
                    if line.startswith(("-", "•")):
                        strengths.append(line[1:].strip())
                    elif line and not line.startswith(
                        ("PRIMARY", "SECONDARY", "STRENGTHS", "ROOT", "RECOMMENDATIONS")
                    ):
                        strengths.append(line)

                elif current_section == "root_cause":
                    if not line.startswith(
                        ("PRIMARY", "SECONDARY", "STRENGTHS", "ROOT", "RECOMMENDATIONS")
                    ):
                        root_cause += " " + line

                elif current_section == "recommendations":
                    # Extract numbered or bulleted items
                    if any(
                        line.startswith(prefix) for prefix in ["-", "•", "1", "2", "3", "4", "5"]
                    ):
                        # Remove leading markers
                        clean_line = line.lstrip("-•0123456789. ")
                        if clean_line:
                            recommendations.append(clean_line)

        # Fallback if parsing failed
        if not primary_weakness:
            primary_weakness = "Performance issues detected (see full analysis)"
            recommendations = ["Review SOP structure", "Analyze prompts", "Check tool usage"]

        return Diagnosis(
            primaryWeakness=primary_weakness.strip(),
            secondaryWeaknesses=secondary_weaknesses if secondary_weaknesses else None,
            strengths=strengths if strengths else None,
            rootCauseAnalysis=root_cause.strip() or "See primary weakness",
            recommendations=recommendations or ["Refine prompts", "Optimize graph structure"],
        )


class Architect:
    """
    Proposes mutations to SOPs based on diagnosis

    Responsibilities:
    - Generate mutation proposals (prompt changes, graph rewiring, etc.)
    - Ensure mutations are valid and safe
    - Create new SOP versions with changes
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        meta_config: MetaConfig | None = None,
        model: str = "gpt-5",
        retry_handler: RetryHandler | None = None,
        circuit_breaker: CircuitBreaker | None = None,
    ):
        """
        Initialize Architect

        Args:
            llm_provider: LLM provider for generation
            meta_config: MetaConfig with architect prompt
            model: Model to use for mutation proposals (default: "gpt-5")
            retry_handler: Retry handler for transient failures (creates default if None)
            circuit_breaker: Circuit breaker for fault tolerance (creates default if None)
        """
        self.llm_provider = llm_provider
        self.meta_config = meta_config
        self.model = model

        # Initialize error handling
        self.retry_handler = retry_handler or RetryHandler()
        self.circuit_breaker = circuit_breaker or CircuitBreaker(
            name="architect_llm",
            config=CircuitBreaker.LLM_CIRCUIT_CONFIG,
        )

    def propose_mutation(
        self,
        diagnosis: Diagnosis,
        sop_config: ProcessConfig,
        prompt_genome: PromptGenome,
        mutation_types: list[MutationType],
        constraints: dict[str, Any] | None = None,
        prompt_orchestrator: Optional["PromptEvolutionOrchestrator"] = None,
        traces: list["ExecutionTrace"] | None = None,
        evaluations: list["EvaluationVector"] | None = None,
    ) -> SOPMutation:
        """
        Propose a mutation based on diagnosis

        Args:
            diagnosis: Diagnosis from Diagnostician
            sop_config: Current ProcessConfig
            prompt_genome: Current PromptGenome
            mutation_types: Allowed mutation types
            constraints: Optional constraints (max_roles, mandatory_roles, etc.)
            prompt_orchestrator: Optional PromptEvolutionOrchestrator for PROMPT_CHANGE mutations
            traces: Optional execution traces for prompt evolution
            evaluations: Optional evaluation vectors for prompt evolution

        Returns:
            SOPMutation with proposed changes
        """

        # Check if we should use orchestrator for PROMPT_CHANGE mutations
        if (
            MutationType.PROMPT_CHANGE in mutation_types
            and prompt_orchestrator is not None
        ):
            logger.info("Using PromptEvolutionOrchestrator for PROMPT_CHANGE mutation")

            evolution_result = prompt_orchestrator.evolve(
                sop_config=sop_config,
                prompt_genome=prompt_genome,
                diagnosis=diagnosis,
                traces=traces or [],
                evaluations=evaluations or [],
                constraints=constraints,
            )

            if evolution_result.changes_made:
                return self._create_mutation_from_evolution(
                    parent_sop=sop_config,
                    parent_genome=prompt_genome,
                    evolution_result=evolution_result,
                    diagnosis=diagnosis,
                )

        # Build architect prompt
        architect_prompt = self._build_architect_prompt(
            diagnosis, sop_config, prompt_genome, mutation_types, constraints
        )

        # Get architect system prompt
        system_prompt = self._get_architect_prompt()

        # Call LLM with error handling (retry + circuit breaker)
        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=architect_prompt),
        ]

        try:
            # Wrap with circuit breaker first, then retry handler
            response = self.circuit_breaker.call(
                lambda: self.retry_handler.execute_with_retry(
                    self.llm_provider.complete,
                    messages=messages,
                    model=self.model,
                    temperature=0.5,
                    retry_config=RetryHandler.LLM_RETRY_CONFIG,
                    component="Architect",
                    operation="llm_propose_mutation",
                )
            )
        except CircuitBreakerOpenError as e:
            logger.exception("Circuit breaker open for mutation proposal LLM call")
            raise RuntimeError("Architect LLM circuit breaker is open - service temporarily unavailable") from e
        except RetryExhausted as e:
            logger.exception("All retry attempts exhausted for mutation proposal")
            raise RuntimeError("Failed to propose mutation after all retry attempts") from e

        # Parse mutation from response (with constraints)
        return self._parse_mutation(response.content, sop_config, prompt_genome, constraints)

    def _build_architect_prompt(
        self,
        diagnosis: Diagnosis,
        sop_config: ProcessConfig,
        prompt_genome: PromptGenome,
        mutation_types: list[MutationType],
        constraints: dict[str, Any] | None,
    ) -> str:
        """Build prompt for mutation proposal"""

        # Convert mutation types to readable names
        mutation_type_names = [mt.value for mt in mutation_types]

        constraints_text = ""
        if constraints:
            constraints_text = f"""
Constraints:
- Max roles: {constraints.get("max_roles", "no limit")}
- Mandatory roles: {", ".join(constraints.get("mandatory_roles", [])) or "none"}
- Allowed tools: {", ".join(constraints.get("allowed_tools", [])) or "all"}
- Disallowed mutations: {", ".join(constraints.get("disallowed_mutation_types", [])) or "none"}
"""

        return f"""
Based on the following diagnosis, propose a mutation to improve the SOP.

DIAGNOSIS:
Primary Weakness: {diagnosis.primaryWeakness}
Root Cause: {diagnosis.rootCauseAnalysis}

Recommendations:
{chr(10).join(f"{i + 1}. {rec}" for i, rec in enumerate(diagnosis.recommendations))}

CURRENT SOP:
ID: {sop_config.id}
Version: {sop_config.version}
Roles: {", ".join(r.id for r in sop_config.roles)}

{constraints_text}

ALLOWED MUTATION TYPES:
{", ".join(mutation_type_names)}

Your task:
1. Choose the MOST IMPACTFUL mutation type to address the primary weakness
2. Specify exactly what to change
3. Provide the new/modified content
4. Explain the rationale

Format your response as:
MUTATION_TYPE: <type>
TARGET_ROLE: <role_id (if applicable)>
CHANGES: <detailed description>
NEW_CONTENT: <the actual new prompt/config>
RATIONALE: <why this will improve performance>
"""

    def _get_architect_prompt(self) -> str:
        """Get architect system prompt"""

        if self.meta_config and self.meta_config.directorPrompt:
            # Use same director prompt for now
            return self.meta_config.directorPrompt.content

        return """
You are an expert SOP architect for AI agent systems.

Your role is to propose targeted mutations to improve SOP performance.

Mutation types you can propose:
- prompt_change: Modify or enhance a role's prompt
- param_tweak: Adjust hyperparameters
- add_role: Introduce a new role to the workflow
- remove_role: Remove a redundant role
- rewire_graph: Change dependencies between roles
- crossover: Combine elements from multiple SOPs

Guidelines:
- Make targeted changes that address root causes
- Preserve what works well
- Keep mutations simple and testable
- Ensure changes are valid and safe
"""

    def _parse_mutation(
        self,
        llm_response: str,
        parent_sop: ProcessConfig,
        parent_genome: PromptGenome,
        constraints: dict[str, Any] | None = None,
    ) -> SOPMutation:
        """Parse LLM response into SOPMutation"""

        lines = llm_response.split("\n")

        mutation_type = MutationType.PROMPT_CHANGE  # Default
        target_role: str | None = None
        new_content_parts: list[str] = []
        rationale_parts: list[str] = []

        current_section: str | None = None

        for line in lines:
            line_stripped = line.strip()

            if line_stripped.startswith("MUTATION_TYPE:"):
                type_str = line_stripped.split(":", 1)[1].strip().lower()
                # Map to MutationType
                for mt in MutationType:
                    if mt.value in type_str:
                        # CROSSOVER requires two parents and must use propose_crossover()
                        # Fall back to PROMPT_CHANGE if CROSSOVER is requested
                        if mt == MutationType.CROSSOVER:
                            logger.warning(
                                "CROSSOVER mutation requested but requires two parents. "
                                "Falling back to PROMPT_CHANGE."
                            )
                            mutation_type = MutationType.PROMPT_CHANGE
                        else:
                            mutation_type = mt
                        break

            elif line_stripped.startswith("TARGET_ROLE:"):
                target_role = line_stripped.split(":", 1)[1].strip()

            elif line_stripped.startswith("CHANGES:"):
                current_section = "changes"

            elif line_stripped.startswith("NEW_CONTENT:"):
                current_section = "new_content"

            elif line_stripped.startswith("RATIONALE:"):
                current_section = "rationale"

            elif current_section == "new_content" and not line_stripped.startswith(
                ("MUTATION", "TARGET", "CHANGES", "RATIONALE")
            ):
                new_content_parts.append(line)

            elif current_section == "rationale" and not line_stripped.startswith(
                ("MUTATION", "TARGET", "CHANGES", "NEW")
            ):
                rationale_parts.append(line_stripped)

        # Join the accumulated parts
        new_content = "\n".join(new_content_parts)
        rationale = " ".join(rationale_parts)

        # Apply mutation to create new config (with constraint validation)
        new_config, new_genome = self._apply_mutation(
            mutation_type, target_role, new_content.strip(), parent_sop, parent_genome, constraints
        )

        return SOPMutation(
            parentSopId=parent_sop.id,
            parentVersion=parent_sop.version,
            newConfig=new_config,
            newPromptGenome=new_genome,
            rationale=rationale.strip() or f"Mutation: {mutation_type.value}",
            mutationType=mutation_type,
        )

    def _validate_prompt_constraints(
        self,
        old_prompt: str,
        new_prompt: str,
        constraints: PromptConstraints | None,
    ) -> None:
        """
        Validate prompt changes against constraints

        Args:
            old_prompt: Original prompt content
            new_prompt: New prompt content
            constraints: PromptConstraints to validate against

        Raises:
            ValueError: If constraints are violated
        """
        if not constraints:
            return

        # Check mustNotChange constraints
        if constraints.mustNotChange:
            for must_not_change in constraints.mustNotChange:
                # Check if the string was in old prompt but is missing in new prompt
                if must_not_change in old_prompt and must_not_change not in new_prompt:
                    raise ValueError(
                        f"Constraint violation: Required text '{must_not_change}' must remain in prompt. "
                        f"This text cannot be removed or modified."
                    )

    def _validate_evolution_constraints(
        self,
        mutation_type: MutationType,
        target_role: str | None,
        parent_sop: ProcessConfig,
        constraints: dict[str, Any] | None,
    ) -> None:
        """
        Validate mutation against evolution job constraints

        Args:
            mutation_type: Type of mutation being proposed
            target_role: Target role ID (if applicable)
            parent_sop: Current ProcessConfig
            constraints: Evolution constraints from job

        Raises:
            ValueError: If constraints are violated
        """
        if not constraints:
            return

        # Check disallowed mutation types
        disallowed_types = constraints.get("disallowed_mutation_types", [])
        if mutation_type in disallowed_types:
            raise ValueError(
                f"Constraint violation: Mutation type '{mutation_type.value}' is not allowed. "
                f"Disallowed types: {[t.value for t in disallowed_types]}"
            )

        # Check max_roles for ADD_ROLE
        if mutation_type == MutationType.ADD_ROLE:
            max_roles = constraints.get("max_roles")
            if max_roles is not None:
                current_role_count = len(parent_sop.roles)
                if current_role_count >= max_roles:
                    raise ValueError(
                        f"Constraint violation: Cannot add role. "
                        f"Maximum roles ({max_roles}) already reached (current: {current_role_count})"
                    )

        # Check mandatory_roles for REMOVE_ROLE
        if mutation_type == MutationType.REMOVE_ROLE and target_role:
            mandatory_roles = constraints.get("mandatory_roles", [])
            if target_role in mandatory_roles:
                raise ValueError(
                    f"Constraint violation: Cannot remove role '{target_role}'. "
                    f"This is a mandatory role. Mandatory roles: {mandatory_roles}"
                )

        # Check allowed_tools
        allowed_tools = constraints.get("allowed_tools")
        if allowed_tools is not None:
            # This would need to be checked when adding/modifying roles with tools
            # For now, we validate during role addition
            pass

    def _apply_mutation(
        self,
        mutation_type: MutationType,
        target_role: str | None,
        new_content: str,
        parent_sop: ProcessConfig,
        parent_genome: PromptGenome,
        constraints: dict[str, Any] | None = None,
    ) -> tuple[ProcessConfig, PromptGenome]:
        """
        Apply mutation to create new SOP and PromptGenome

        Args:
            mutation_type: Type of mutation to apply
            target_role: Target role ID (if applicable)
            new_content: New content for the mutation
            parent_sop: Current ProcessConfig
            parent_genome: Current PromptGenome
            constraints: Optional evolution constraints to validate

        Returns:
            Tuple of (new ProcessConfig, new PromptGenome)

        Raises:
            ValueError: If mutation violates constraints
        """

        # Validate evolution constraints FIRST
        self._validate_evolution_constraints(mutation_type, target_role, parent_sop, constraints)

        # Create new version numbers
        version_parts = parent_sop.version.split(".")
        major, minor, patch = (
            int(version_parts[0]),
            int(version_parts[1]),
            int(version_parts[2].split("-")[0]),
        )

        # Generate unique suffix to prevent version collisions in gene pool
        # This ensures each mutation produces a unique version even if the same
        # parent is selected multiple times across generations
        unique_suffix = uuid.uuid4().hex[:6]

        if mutation_type in [
            MutationType.ADD_ROLE,
            MutationType.REMOVE_ROLE,
            MutationType.REWIRE_GRAPH,
        ]:
            # Major change
            new_version = f"{major + 1}.0.0-{unique_suffix}"
        elif mutation_type == MutationType.PROMPT_CHANGE:
            # Minor change
            new_version = f"{major}.{minor + 1}.0-{unique_suffix}"
        else:
            # Patch
            new_version = f"{major}.{minor}.{patch + 1}-{unique_suffix}"

        # Clone configs
        new_sop_dict = parent_sop.model_dump()
        new_sop_dict["version"] = new_version

        new_genome_dict = parent_genome.model_dump()
        new_genome_dict["version"] = new_version

        # Apply mutation with validation
        if mutation_type == MutationType.PROMPT_CHANGE and target_role:
            # Find the role's prompt ref
            for role in new_sop_dict["roles"]:
                if role["id"] == target_role:
                    prompt_ref = role["promptRef"]

                    # Get old prompt and validate constraints
                    if prompt_ref in new_genome_dict["rolePrompts"]:
                        old_prompt_data = new_genome_dict["rolePrompts"][prompt_ref]
                        old_prompt = old_prompt_data["content"]

                        # Get constraints from the prompt
                        prompt_constraints = None
                        if old_prompt_data.get("constraints"):
                            prompt_constraints = PromptConstraints(**old_prompt_data["constraints"])

                        # Validate prompt constraints before applying change
                        self._validate_prompt_constraints(old_prompt, new_content, prompt_constraints)

                        # Update prompt content (only after validation passes)
                        new_genome_dict["rolePrompts"][prompt_ref]["content"] = new_content

                    break

        elif mutation_type == MutationType.ADD_ROLE:
            # Parse new role from LLM response
            new_role = self._parse_new_role_from_llm(new_content, parent_sop)
            new_sop_dict["roles"].append(new_role["role"].model_dump())

            # Add edges for new role
            for edge in new_role["edges"]:
                new_sop_dict["graph"].append(edge.model_dump())

            # Add prompt to genome
            new_genome_dict["rolePrompts"][new_role["role"].promptRef] = new_role["prompt"].model_dump()

        elif mutation_type == MutationType.REMOVE_ROLE:
            # Identify role to remove
            role_to_remove = target_role or self._parse_role_to_remove(new_content)

            # Get tools used by the role being removed
            removed_role_tools: set[str] = set()
            for role in new_sop_dict["roles"]:
                if role["id"] == role_to_remove and role.get("tools"):
                    removed_role_tools.update(role["tools"])

            # Remove role from roles list
            new_sop_dict["roles"] = [r for r in new_sop_dict["roles"] if r["id"] != role_to_remove]

            # Remove all edges involving this role
            new_sop_dict["graph"] = [
                e
                for e in new_sop_dict["graph"]
                if not self._edge_involves_role(e, role_to_remove)
            ]

            # Remove unused tools from SOP config
            if removed_role_tools:
                # Check which tools are still used by remaining roles
                remaining_tools: set[str] = set()
                for role in new_sop_dict["roles"]:
                    if role.get("tools"):
                        remaining_tools.update(role["tools"])

                # Remove tools that are no longer used
                tools_to_remove = removed_role_tools - remaining_tools
                if tools_to_remove:
                    new_sop_dict["tools"] = [t for t in new_sop_dict["tools"] if t not in tools_to_remove]

            # Note: We keep the prompt in genome for historical reference

        elif mutation_type == MutationType.REWIRE_GRAPH:
            # Parse graph changes from LLM response
            edges_to_add, edges_to_remove = self._parse_graph_changes(new_content)

            # Remove specified edges
            for edge_to_remove in edges_to_remove:
                new_sop_dict["graph"] = [
                    e
                    for e in new_sop_dict["graph"]
                    if not self._edges_match(e, edge_to_remove)
                ]

            # Add new edges
            for edge_to_add in edges_to_add:
                new_sop_dict["graph"].append(edge_to_add.model_dump())

        elif mutation_type == MutationType.CROSSOVER:
            # CROSSOVER mutations require two parents and must use propose_crossover()
            raise ValueError(
                "CROSSOVER mutations must use propose_crossover() method, "
                "not _apply_mutation(). CROSSOVER requires two parent SOPs."
            )

        # Create new ProcessConfig and PromptGenome
        new_sop = ProcessConfig(**new_sop_dict)
        new_genome = PromptGenome(**new_genome_dict)

        # Validate the new SOP
        engine = ExecutionEngine()
        errors = engine.validate_sop(new_sop)
        if errors:
            raise ValueError(f"Invalid SOP after mutation: {', '.join(errors)}")

        return new_sop, new_genome

    def _parse_new_role_from_llm(
        self, llm_content: str, parent_sop: ProcessConfig
    ) -> dict[str, Any]:
        """
        Parse new role specification from LLM response

        Expected format:
        ROLE_ID: <id>
        MODEL: <model>
        TOOLS: <tool1, tool2, ...>
        PROMPT: <prompt content>
        EDGES_FROM: <from1, from2, ...>
        EDGES_TO: <to1, to2, ...>

        Returns:
            Dict with keys: role (RoleConfig), prompt (RolePrompt), edges (list[GraphEdge])
        """
        lines = llm_content.split("\n")

        role_id: str | None = None
        model = self.model  # Use configured model as default
        tools: list[str] = []
        prompt_content: list[str] = []
        edges_from: list[str] = []
        edges_to: list[str] = []

        current_section: str | None = None

        for line in lines:
            line_stripped = line.strip()

            if line_stripped.startswith("ROLE_ID:"):
                role_id = line_stripped.split(":", 1)[1].strip()

            elif line_stripped.startswith("MODEL:"):
                model = line_stripped.split(":", 1)[1].strip()

            elif line_stripped.startswith("TOOLS:"):
                tools_str = line_stripped.split(":", 1)[1].strip()
                if tools_str and tools_str.lower() not in ["none", "null", ""]:
                    tools = [t.strip() for t in tools_str.split(",")]

            elif line_stripped.startswith("PROMPT:"):
                current_section = "prompt"
                # Capture prompt content after colon
                content_after = line_stripped.split(":", 1)[1].strip()
                if content_after:
                    prompt_content.append(content_after)

            elif line_stripped.startswith("EDGES_FROM:"):
                edges_from_str = line_stripped.split(":", 1)[1].strip()
                if edges_from_str:
                    edges_from = [e.strip() for e in edges_from_str.split(",") if e.strip()]

            elif line_stripped.startswith("EDGES_TO:"):
                edges_to_str = line_stripped.split(":", 1)[1].strip()
                if edges_to_str:
                    edges_to = [e.strip() for e in edges_to_str.split(",") if e.strip()]

            elif current_section == "prompt" and not line_stripped.startswith(
                ("ROLE_ID", "MODEL", "TOOLS", "EDGES")
            ):
                prompt_content.append(line)

        # Generate role ID if not provided
        if not role_id:
            role_id = f"role_{uuid.uuid4().hex[:8]}"

        # Sanitize role ID
        role_id = re.sub(r"[^a-zA-Z0-9_-]", "_", role_id)

        # Create prompt reference
        prompt_ref = f"prompt_{role_id}"

        # Join prompt content
        prompt_text = "\n".join(prompt_content).strip()
        if not prompt_text:
            prompt_text = f"You are {role_id}. Process the inputs and provide outputs."

        # Create RoleConfig
        role_config = RoleConfig(
            id=role_id, model=model, tools=tools if tools else None, promptRef=prompt_ref
        )

        # Create RolePrompt
        role_prompt = RolePrompt(id=prompt_ref, content=prompt_text)

        # Create GraphEdges
        edges: list[GraphEdge] = []
        if edges_from and role_id:
            # Edges TO this new role
            for from_node in edges_from:
                edges.append(GraphEdge(from_=from_node, to=role_id))
        if edges_to and role_id:
            # Edges FROM this new role
            for to_node in edges_to:
                edges.append(GraphEdge(from_=role_id, to=to_node))
        # If no edges specified, connect to user_input by default
        if not edges:
            edges.append(GraphEdge(from_="user_input", to=role_id))
        return {"role": role_config, "prompt": role_prompt, "edges": edges}

    def _parse_role_to_remove(self, llm_content: str) -> str:
        """
        Parse role ID to remove from LLM response

        Expected format:
        REMOVE_ROLE: <role_id>
        or
        ROLE_TO_REMOVE: <role_id>
        """
        lines = llm_content.split("\n")

        for line in lines:
            line_stripped = line.strip()

            if line_stripped.startswith("REMOVE_ROLE:"):
                return line_stripped.split(":", 1)[1].strip()

            if line_stripped.startswith("ROLE_TO_REMOVE:"):
                return line_stripped.split(":", 1)[1].strip()

            if line_stripped.startswith("ROLE_ID:"):
                return line_stripped.split(":", 1)[1].strip()

        # Fallback: try to extract from content
        match = re.search(r"remove.*role[:\s]+([a-zA-Z0-9_-]+)", llm_content, re.IGNORECASE)
        if match:
            return match.group(1)

        raise ValueError("Could not parse role to remove from LLM response")

    def _parse_graph_changes(self, llm_content: str) -> tuple[list[GraphEdge], list[dict[str, Any]]]:
        """
        Parse graph changes from LLM response

        Expected format:
        ADD_EDGES:
        - from: role1, to: role2
        - from: role3, to: role4

        REMOVE_EDGES:
        - from: role5, to: role6
        - from: role7, to: role8

        Or JSON format:
        ADD_EDGES: [{"from": "role1", "to": "role2"}]
        REMOVE_EDGES: [{"from": "role5", "to": "role6"}]

        Returns:
            (edges_to_add, edges_to_remove) tuple
        """
        edges_to_add: list[GraphEdge] = []
        edges_to_remove: list[dict[str, Any]] = []

        # Try JSON format first
        try:
            add_match = re.search(r"ADD_EDGES:\s*(\[.*?\])", llm_content, re.DOTALL)
            if add_match:
                add_list = json.loads(add_match.group(1))
                for edge_dict in add_list:
                    edges_to_add.append(GraphEdge(**edge_dict))

            remove_match = re.search(r"REMOVE_EDGES:\s*(\[.*?\])", llm_content, re.DOTALL)
            if remove_match:
                remove_list = json.loads(remove_match.group(1))
                edges_to_remove = remove_list

            if edges_to_add or edges_to_remove:
                return edges_to_add, edges_to_remove
        except json.JSONDecodeError:
            # JSON parsing failed, fall through to line-by-line parsing
            pass
        except (KeyError, TypeError, ValueError):
            # Invalid edge format, fall through to line-by-line parsing
            pass

        # Parse line-by-line format
        lines = llm_content.split("\n")
        current_section: str | None = None

        for line in lines:
            line_stripped = line.strip()

            if line_stripped.startswith("ADD_EDGES:"):
                current_section = "add"
                continue

            if line_stripped.startswith("REMOVE_EDGES:"):
                current_section = "remove"
                continue

            if current_section == "add" and line_stripped.startswith("-"):
                # Parse edge specification
                edge = self._parse_edge_line(line_stripped)
                if edge:
                    edges_to_add.append(edge)

            elif current_section == "remove" and line_stripped.startswith("-"):
                # Parse edge specification
                edge_dict = self._parse_edge_line_dict(line_stripped)
                if edge_dict:
                    edges_to_remove.append(edge_dict)

        return edges_to_add, edges_to_remove

    def _parse_edge_line(self, line: str) -> GraphEdge | None:
        """Parse a single edge line like '- from: role1, to: role2'"""
        # Remove leading dash
        line = line.lstrip("- ").strip()

        # Try to parse as key-value pairs
        from_node: str | list[str] | None = None
        to_node: str | None = None

        # Pattern: from: X, to: Y
        match = re.search(r"from:\s*([a-zA-Z0-9_,-]+).*to:\s*([a-zA-Z0-9_-]+)", line, re.IGNORECASE)
        if match:
            from_node = match.group(1).strip()
            to_node = match.group(2).strip()

        if from_node and to_node:
            # Handle comma-separated from nodes
            if isinstance(from_node, str) and "," in from_node:
                from_node_list = [n.strip() for n in from_node.split(",") if n.strip()]
                # If only one node after filtering, use string instead of list
                if len(from_node_list) == 1:
                    from_node = from_node_list[0]
                else:
                    from_node = from_node_list
            return GraphEdge(from_=from_node, to=to_node)
        return None

    def _parse_edge_line_dict(self, line: str) -> dict[str, Any] | None:
        """Parse edge line into dict for comparison"""
        edge = self._parse_edge_line(line)
        if edge:
            return {"from": edge.from_, "to": edge.to}
        return None

    def _edge_involves_role(self, edge: dict[str, Any], role_id: str) -> bool:
        """Check if edge involves a specific role"""
        # Handle both 'from' and 'from_' keys (dict vs pydantic model dump)
        from_key = "from" if "from" in edge else "from_"
        to_key = "to"

        from_nodes = edge[from_key] if isinstance(edge[from_key], list) else [edge[from_key]]
        return role_id in from_nodes or edge[to_key] == role_id

    def _edges_match(self, edge1: dict[str, Any], edge2: dict[str, Any]) -> bool:
        """Check if two edges match (for removal)"""
        # Handle both 'from' and 'from_' keys
        from1_key = "from" if "from" in edge1 else "from_"
        from2_key = "from" if "from" in edge2 else "from_"

        from1: str | list[str] = edge1[from1_key]
        from2: str | list[str] = edge2[from2_key]
        to1: str = edge1["to"]
        to2: str = edge2["to"]

        # Normalize to lists for comparison
        from1_list: list[str] = from1 if isinstance(from1, list) else [from1]
        from2_list: list[str] = from2 if isinstance(from2, list) else [from2]

        return set(from1_list) == set(from2_list) and to1 == to2

    # === Crossover Helper Methods ===
    # These methods extract shared logic from _role_crossover, _prompt_crossover, and _graph_crossover

    def _get_role_type(self, role_id: str) -> str:
        """Extract role type from ID (e.g., 'planner_v1' -> 'planner')."""
        return role_id.split("_")[0] if "_" in role_id else role_id

    def _group_roles_by_type(
        self,
        config: ProcessConfig,
    ) -> tuple[dict[str, RoleConfig], dict[str, str]]:
        """Group roles by type prefix.

        Args:
            config: ProcessConfig to extract roles from

        Returns:
            Tuple of (role_id -> RoleConfig, role_type -> role_id)
        """
        roles = {role.id: role for role in config.roles}
        role_types = {self._get_role_type(rid): rid for rid in roles}
        return roles, role_types

    def _select_role_source(
        self,
        role_type: str,
        role_types_a: dict[str, str],
        role_types_b: dict[str, str],
        roles_a: dict[str, RoleConfig],
        roles_b: dict[str, RoleConfig],
        genome_a: PromptGenome,
        genome_b: PromptGenome,
    ) -> tuple[RoleConfig, PromptGenome]:
        """Select role from one of two parents using 50/50 random choice.

        Args:
            role_type: Type of role to select
            role_types_a: Parent A's role_type -> role_id mapping
            role_types_b: Parent B's role_type -> role_id mapping
            roles_a: Parent A's role_id -> RoleConfig mapping
            roles_b: Parent B's role_id -> RoleConfig mapping
            genome_a: Parent A's PromptGenome
            genome_b: Parent B's PromptGenome

        Returns:
            Tuple of (selected_role, source_genome)
        """
        import random

        COIN_FLIP_THRESHOLD = 0.5

        if role_type in role_types_a and role_type in role_types_b:
            if random.random() < COIN_FLIP_THRESHOLD:
                role_id = role_types_a[role_type]
                return roles_a[role_id], genome_a
            role_id = role_types_b[role_type]
            return roles_b[role_id], genome_b
        if role_type in role_types_a:
            role_id = role_types_a[role_type]
            return roles_a[role_id], genome_a
        # role_type must be in role_types_b
        role_id = role_types_b[role_type]
        return roles_b[role_id], genome_b

    def _map_single_node(
        self,
        node: str,
        valid_ids: set[str],
        id_map: dict[str, str],
    ) -> str | None:
        """Map a single node ID to new ID.

        Args:
            node: Node ID to map
            valid_ids: Set of valid role IDs in new config
            id_map: Optional mapping of old_id -> new_id

        Returns:
            Mapped node ID or None if not mappable
        """
        if node == "user_input":
            return node
        if node in id_map:
            return id_map[node]
        if node in valid_ids:
            return node
        return None

    def _map_edge_nodes(
        self,
        from_: str | list[str],
        valid_ids: set[str],
        id_map: dict[str, str],
    ) -> str | list[str] | None:
        """Map edge source nodes to new IDs.

        Args:
            from_: Source node(s) of edge
            valid_ids: Set of valid role IDs in new config
            id_map: Optional mapping of old_id -> new_id

        Returns:
            Mapped from_ value or None if not mappable
        """
        if isinstance(from_, list):
            mapped: list[str] = []
            for f in from_:
                result = self._map_single_node(f, valid_ids, id_map)
                if result:
                    mapped.append(result)
            return mapped if mapped else None
        return self._map_single_node(from_, valid_ids, id_map)

    def _reconstruct_graph_edges(
        self,
        source_config: ProcessConfig,
        new_role_ids: set[str],
        role_id_map: dict[str, str] | None = None,
    ) -> list[GraphEdge]:
        """Reconstruct graph edges for new role set.

        Args:
            source_config: Source config to copy edges from
            new_role_ids: Set of role IDs in new config
            role_id_map: Optional mapping of old_id -> new_id

        Returns:
            List of valid GraphEdges for new config
        """
        new_edges: list[GraphEdge] = []
        id_map = role_id_map or {}

        for edge in source_config.graph:
            # Map from_ nodes
            new_from = self._map_edge_nodes(edge.from_, new_role_ids, id_map)
            if new_from is None:
                continue

            # Map to node
            new_to = self._map_single_node(edge.to, new_role_ids, id_map)
            if new_to is None:
                continue

            # Validate endpoints
            from_nodes = new_from if isinstance(new_from, list) else [new_from]
            if all(f == "user_input" or f in new_role_ids for f in from_nodes) and new_to in new_role_ids:
                new_edges.append(GraphEdge(from_=new_from, to=new_to, condition=edge.condition))

        return new_edges

    def _build_crossover_config(
        self,
        parent_config: ProcessConfig,
        new_roles: list[RoleConfig],
        new_edges: list[GraphEdge],
        used_tools: set[str],
    ) -> dict[str, Any]:
        """Build new config dict from crossover results.

        Args:
            parent_config: Parent config to base new config on
            new_roles: List of roles for new config
            new_edges: List of edges for new config
            used_tools: Set of tools used by roles

        Returns:
            Config dict ready for ProcessConfig creation
        """
        new_config_dict = parent_config.model_dump()
        new_config_dict["roles"] = [r.model_dump() for r in new_roles]
        new_config_dict["graph"] = [e.model_dump(by_alias=True) for e in new_edges]
        new_config_dict["version"] = self._increment_version(parent_config.version, major=True)
        new_config_dict["tools"] = list(used_tools) if used_tools else []
        return new_config_dict

    def _build_crossover_genome(
        self,
        parent_genome: PromptGenome,
        new_roles: list[RoleConfig],
        source_genome_map: dict[str, PromptGenome],
        version: str,
    ) -> dict[str, Any]:
        """Build new genome dict from crossover results.

        Args:
            parent_genome: Parent genome to base new genome on
            new_roles: List of roles for new genome
            source_genome_map: Mapping of role_id -> source PromptGenome
            version: Version string for new genome

        Returns:
            Genome dict ready for PromptGenome creation
        """
        new_genome_dict = parent_genome.model_dump()
        new_genome_dict["version"] = version
        new_genome_dict["rolePrompts"] = {}

        for role in new_roles:
            source_genome = source_genome_map.get(role.id)
            if source_genome and role.promptRef in source_genome.rolePrompts:
                new_genome_dict["rolePrompts"][role.promptRef] = source_genome.rolePrompts[
                    role.promptRef
                ].model_dump()

        return new_genome_dict

    def _validate_crossover_result(
        self,
        config: ProcessConfig,
    ) -> None:
        """Validate crossover result.

        Args:
            config: ProcessConfig to validate

        Raises:
            ValueError: If result is invalid
        """
        engine = ExecutionEngine()
        errors = engine.validate_sop(config)
        if errors:
            raise ValueError(f"Invalid crossover result: {errors}")

    # === End Crossover Helper Methods ===

    def propose_crossover(
        self,
        parent_a_config: ProcessConfig,
        parent_a_genome: PromptGenome,
        parent_b_config: ProcessConfig,
        parent_b_genome: PromptGenome,
        strategy: str = "role_crossover",
    ) -> tuple[ProcessConfig, PromptGenome]:
        """
        Perform crossover between two parent SOPs

        Args:
            parent_a_config: First parent ProcessConfig
            parent_a_genome: First parent PromptGenome
            parent_b_config: Second parent ProcessConfig
            parent_b_genome: Second parent PromptGenome
            strategy: Crossover strategy (role_crossover, prompt_crossover, graph_crossover)

        Returns:
            New (ProcessConfig, PromptGenome) tuple

        Raises:
            ValueError: If strategy is unknown or crossover produces invalid SOP
        """
        if strategy == "role_crossover":
            return self._role_crossover(
                parent_a_config, parent_a_genome, parent_b_config, parent_b_genome
            )
        if strategy == "prompt_crossover":
            return self._prompt_crossover(
                parent_a_config, parent_a_genome, parent_b_config, parent_b_genome
            )
        if strategy == "graph_crossover":
            return self._graph_crossover(
                parent_a_config, parent_a_genome, parent_b_config, parent_b_genome
            )
        raise ValueError(f"Unknown crossover strategy: {strategy}")

    def _role_crossover(
        self,
        parent_a_config: ProcessConfig,
        parent_a_genome: PromptGenome,
        parent_b_config: ProcessConfig,
        parent_b_genome: PromptGenome,
    ) -> tuple[ProcessConfig, PromptGenome]:
        """
        Crossover at role level: mix roles from two parents.

        Strategy:
        1. Identify common role types (planner, retriever, etc.)
        2. For each type, randomly choose from parent A or B
        3. Reconstruct graph edges
        4. Validate DAG

        Args:
            parent_a_config: First parent ProcessConfig
            parent_a_genome: First parent PromptGenome
            parent_b_config: Second parent ProcessConfig
            parent_b_genome: Second parent PromptGenome

        Returns:
            New (ProcessConfig, PromptGenome) tuple

        Raises:
            ValueError: If crossover produces invalid SOP
        """
        # Group roles by type using helper
        roles_a, role_types_a = self._group_roles_by_type(parent_a_config)
        roles_b, role_types_b = self._group_roles_by_type(parent_b_config)

        # Mix roles using helper
        new_roles: list[RoleConfig] = []
        new_role_map: dict[str, str] = {}
        source_genome_map: dict[str, PromptGenome] = {}
        used_tools: set[str] = set()

        for role_type in set(role_types_a.keys()) | set(role_types_b.keys()):
            source_role, source_genome = self._select_role_source(
                role_type, role_types_a, role_types_b,
                roles_a, roles_b, parent_a_genome, parent_b_genome
            )
            new_role = source_role.model_copy(deep=True)
            new_roles.append(new_role)
            new_role_map[source_role.id] = new_role.id
            source_genome_map[new_role.id] = source_genome
            if source_role.tools:
                used_tools.update(source_role.tools)

        # Reconstruct graph edges using helper
        new_role_ids = {r.id for r in new_roles}
        new_edges = self._reconstruct_graph_edges(parent_a_config, new_role_ids, new_role_map)

        # Fallback: create linear chain if no edges
        if not new_edges and new_roles:
            new_edges.append(GraphEdge(from_="user_input", to=new_roles[0].id))

        # Build configs using helpers
        new_config_dict = self._build_crossover_config(
            parent_a_config, new_roles, new_edges, used_tools
        )
        new_genome_dict = self._build_crossover_genome(
            parent_a_genome, new_roles, source_genome_map, new_config_dict["version"]
        )

        # Create and validate
        new_config = ProcessConfig(**new_config_dict)
        new_genome = PromptGenome(**new_genome_dict)
        self._validate_crossover_result(new_config)

        return new_config, new_genome

    def _prompt_crossover(
        self,
        parent_a_config: ProcessConfig,
        parent_a_genome: PromptGenome,
        parent_b_config: ProcessConfig,
        parent_b_genome: PromptGenome,
    ) -> tuple[ProcessConfig, PromptGenome]:
        """
        Crossover at prompt section level: mix prompt sections from two parents.

        Strategy:
        1. First perform role-level crossover to get base config
        2. For roles that exist in both parents, crossover prompts at section level:
           - Parse both prompts into sections
           - For each section type, randomly choose from parent A or B
           - Immutable sections (CONSTRAINTS) always come from parent A (safety)
           - Reconstruct prompt from selected sections
        3. For unique roles, keep their prompts unchanged

        Args:
            parent_a_config: First parent ProcessConfig
            parent_a_genome: First parent PromptGenome
            parent_b_config: Second parent ProcessConfig
            parent_b_genome: Second parent PromptGenome

        Returns:
            New (ProcessConfig, PromptGenome) tuple

        Raises:
            ValueError: If crossover produces invalid SOP
        """
        import random

        from siare.core.models import PromptSectionType
        from siare.services.prompt_evolution.parser import MarkdownSectionParser

        parser = MarkdownSectionParser()

        # Group roles by type using helper
        roles_a, role_types_a = self._group_roles_by_type(parent_a_config)
        roles_b, role_types_b = self._group_roles_by_type(parent_b_config)

        # Build child roles and prompts
        new_roles: list[RoleConfig] = []
        new_role_prompts: dict[str, RolePrompt] = {}
        used_tools: set[str] = set()

        COIN_FLIP_THRESHOLD = 0.5

        for role_type in set(role_types_a.keys()) | set(role_types_b.keys()):
            if role_type in role_types_a and role_type in role_types_b:
                # Both parents have this role type - crossover prompts at section level
                role_a_id = role_types_a[role_type]
                role_b_id = role_types_b[role_type]
                role_a = roles_a[role_a_id]
                role_b = roles_b[role_b_id]

                prompt_a = parent_a_genome.rolePrompts.get(role_a.promptRef)
                prompt_b = parent_b_genome.rolePrompts.get(role_b.promptRef)

                new_role = role_a.model_copy(deep=True)
                new_roles.append(new_role)
                if role_a.tools:
                    used_tools.update(role_a.tools)

                # Crossover prompts at section level
                new_prompt = self._crossover_prompt_sections(
                    parser, prompt_a, prompt_b, role_a_id, role_b_id,
                    role_a.promptRef, COIN_FLIP_THRESHOLD, PromptSectionType, random
                )
                if new_prompt:
                    new_role_prompts[new_role.promptRef] = new_prompt

            elif role_type in role_types_a:
                # Only parent A has this role - inherit unchanged
                role_a_id = role_types_a[role_type]
                role_a = roles_a[role_a_id]
                new_role = role_a.model_copy(deep=True)
                new_roles.append(new_role)
                if role_a.tools:
                    used_tools.update(role_a.tools)
                prompt_a = parent_a_genome.rolePrompts.get(role_a.promptRef)
                if prompt_a:
                    new_role_prompts[new_role.promptRef] = prompt_a.model_copy(deep=True)
            else:
                # Only parent B has this role - inherit unchanged
                role_b_id = role_types_b[role_type]
                role_b = roles_b[role_b_id]
                new_role = role_b.model_copy(deep=True)
                new_roles.append(new_role)
                if role_b.tools:
                    used_tools.update(role_b.tools)
                prompt_b = parent_b_genome.rolePrompts.get(role_b.promptRef)
                if prompt_b:
                    new_role_prompts[new_role.promptRef] = prompt_b.model_copy(deep=True)

        # Reconstruct graph edges using helper
        new_role_ids = {r.id for r in new_roles}
        new_edges = self._reconstruct_graph_edges(parent_a_config, new_role_ids)

        # Fallback: create linear chain if no edges
        if not new_edges and new_roles:
            new_edges.append(GraphEdge(from_="user_input", to=new_roles[0].id))

        # Build new config using helper
        new_config_dict = self._build_crossover_config(
            parent_a_config, new_roles, new_edges, used_tools
        )

        # Build new genome (specialized for prompt crossover)
        new_genome_dict = parent_a_genome.model_dump()
        new_genome_dict["version"] = new_config_dict["version"]
        new_genome_dict["rolePrompts"] = {
            k: v.model_dump() for k, v in new_role_prompts.items()
        }

        # Create and validate
        new_config = ProcessConfig(**new_config_dict)
        new_genome = PromptGenome(**new_genome_dict)
        self._validate_crossover_result(new_config)

        return new_config, new_genome

    def _crossover_prompt_sections(
        self,
        parser: Any,
        prompt_a: RolePrompt | None,
        prompt_b: RolePrompt | None,
        role_a_id: str,
        role_b_id: str,
        prompt_ref: str,
        coin_flip_threshold: float,
        prompt_section_type: Any,
        random_module: Any,
    ) -> RolePrompt | None:
        """Crossover prompt sections between two prompts.

        Args:
            parser: MarkdownSectionParser instance
            prompt_a: Prompt from parent A (may be None)
            prompt_b: Prompt from parent B (may be None)
            role_a_id: Role ID from parent A (for parsing)
            role_b_id: Role ID from parent B (for parsing)
            prompt_ref: Prompt reference for new prompt
            coin_flip_threshold: Threshold for random selection
            prompt_section_type: PromptSectionType enum
            random_module: random module for selection

        Returns:
            New RolePrompt or None if no prompts available
        """
        if prompt_a and prompt_b:
            # Parse both prompts
            parsed_a = parser.parse(prompt_a, role_a_id)
            parsed_b = parser.parse(prompt_b, role_b_id)

            sections_a = {s.section_type: s for s in parsed_a.sections}
            sections_b = {s.section_type: s for s in parsed_b.sections}

            # Select sections for child
            all_section_types = set(sections_a.keys()) | set(sections_b.keys())
            selected_sections: dict[str, str] = {}

            for section_type in all_section_types:
                section_a = sections_a.get(section_type)
                section_b = sections_b.get(section_type)

                # Immutable sections (CONSTRAINTS) always from parent A
                if section_type == prompt_section_type.CONSTRAINTS:
                    if section_a:
                        selected_sections[section_a.id] = section_a.content
                    elif section_b:
                        selected_sections[section_b.id] = section_b.content
                elif section_a and section_b:
                    # Both have this section - random choice
                    if random_module.random() < coin_flip_threshold:
                        selected_sections[section_a.id] = section_a.content
                    else:
                        selected_sections[section_a.id] = section_b.content
                elif section_a:
                    selected_sections[section_a.id] = section_a.content
                elif section_b:
                    selected_sections[section_b.id] = section_b.content

            # Reconstruct prompt from selected sections
            new_content = parser.reconstruct(parsed_a, selected_sections)
            return RolePrompt(
                id=prompt_a.id,
                content=new_content,
                constraints=prompt_a.constraints,
            )
        if prompt_a:
            return prompt_a.model_copy(deep=True)
        if prompt_b:
            new_prompt = prompt_b.model_copy(deep=True)
            new_prompt.id = prompt_ref
            return new_prompt
        return None

    def _graph_crossover(
        self,
        parent_a_config: ProcessConfig,
        parent_a_genome: PromptGenome,
        parent_b_config: ProcessConfig,
        parent_b_genome: PromptGenome,
    ) -> tuple[ProcessConfig, PromptGenome]:
        """
        Crossover at graph topology level: mix edges from two parents.

        Strategy:
        1. First perform role-level crossover to get base config with mixed roles
        2. Collect all edges from both parents
        3. For each edge, check if both endpoints exist in child
        4. For edges connecting same role types: 50/50 random choice
        5. For unique edges: include if endpoints exist
        6. Validate DAG, create linear fallback if invalid

        Args:
            parent_a_config: First parent ProcessConfig
            parent_a_genome: First parent PromptGenome
            parent_b_config: Second parent ProcessConfig
            parent_b_genome: Second parent PromptGenome

        Returns:
            New (ProcessConfig, PromptGenome) tuple

        Raises:
            ValueError: If crossover produces invalid SOP
        """
        import random

        # Group roles by type using helper
        roles_a, role_types_a = self._group_roles_by_type(parent_a_config)
        roles_b, role_types_b = self._group_roles_by_type(parent_b_config)

        # Build child roles using helper
        new_roles: list[RoleConfig] = []
        source_genome_map: dict[str, PromptGenome] = {}
        used_tools: set[str] = set()

        for role_type in set(role_types_a.keys()) | set(role_types_b.keys()):
            source_role, source_genome = self._select_role_source(
                role_type, role_types_a, role_types_b,
                roles_a, roles_b, parent_a_genome, parent_b_genome
            )
            new_role = source_role.model_copy(deep=True)
            new_roles.append(new_role)
            source_genome_map[new_role.id] = source_genome
            if source_role.tools:
                used_tools.update(source_role.tools)

        new_role_ids = {r.id for r in new_roles}

        # Mix edges from both parents
        new_edges = self._mix_graph_edges(
            parent_a_config, parent_b_config, new_roles, new_role_ids, random
        )

        # Fallback: create linear chain if no edges
        if not new_edges and new_roles:
            new_edges.append(GraphEdge(from_="user_input", to=new_roles[0].id))
            for i in range(len(new_roles) - 1):
                new_edges.append(GraphEdge(from_=new_roles[i].id, to=new_roles[i + 1].id))

        # Build config and genome using helpers
        new_config_dict = self._build_crossover_config(
            parent_a_config, new_roles, new_edges, used_tools
        )
        new_genome_dict = self._build_crossover_genome(
            parent_a_genome, new_roles, source_genome_map, new_config_dict["version"]
        )

        # Create and validate
        new_config = ProcessConfig(**new_config_dict)
        new_genome = PromptGenome(**new_genome_dict)
        self._validate_crossover_result(new_config)

        return new_config, new_genome

    def _mix_graph_edges(
        self,
        parent_a_config: ProcessConfig,
        parent_b_config: ProcessConfig,
        new_roles: list[RoleConfig],
        new_role_ids: set[str],
        random_module: Any,
    ) -> list[GraphEdge]:
        """Mix edges from two parent configs.

        Args:
            parent_a_config: First parent ProcessConfig
            parent_b_config: Second parent ProcessConfig
            new_roles: List of roles for new config
            new_role_ids: Set of role IDs in new config
            random_module: random module for selection

        Returns:
            List of mixed GraphEdges
        """
        COIN_FLIP_THRESHOLD = 0.5

        def edge_key(edge: GraphEdge) -> tuple[str, str]:
            from_str = ",".join(sorted(edge.from_)) if isinstance(edge.from_, list) else edge.from_
            return (from_str, edge.to)

        edges_a = {edge_key(e): e for e in parent_a_config.graph}
        edges_b = {edge_key(e): e for e in parent_b_config.graph}

        new_edges: list[GraphEdge] = []
        all_edge_keys = set(edges_a.keys()) | set(edges_b.keys())

        for key in all_edge_keys:
            edge_a = edges_a.get(key)
            edge_b = edges_b.get(key)

            # Choose edge source
            if edge_a and edge_b:
                chosen_edge = edge_a if random_module.random() < COIN_FLIP_THRESHOLD else edge_b
            elif edge_a:
                chosen_edge = edge_a
            else:
                chosen_edge = edge_b

            if chosen_edge is None:
                continue

            # Map edge endpoints to new roles
            mapped_edge = self._map_edge_to_new_roles(
                chosen_edge, new_roles, new_role_ids
            )
            if mapped_edge:
                new_edges.append(mapped_edge)

        # Remove duplicates
        seen: set[tuple[str, str]] = set()
        unique_edges: list[GraphEdge] = []
        for edge in new_edges:
            key = edge_key(edge)
            if key not in seen:
                seen.add(key)
                unique_edges.append(edge)

        return unique_edges

    def _map_edge_to_new_roles(
        self,
        edge: GraphEdge,
        new_roles: list[RoleConfig],
        new_role_ids: set[str],
    ) -> GraphEdge | None:
        """Map an edge's endpoints to new role IDs.

        Args:
            edge: Edge to map
            new_roles: List of roles in new config
            new_role_ids: Set of role IDs in new config

        Returns:
            Mapped GraphEdge or None if mapping fails
        """
        from_nodes = edge.from_ if isinstance(edge.from_, list) else [edge.from_]
        to_node = edge.to

        # Map from nodes
        valid_from: list[str] = []
        for f in from_nodes:
            if f == "user_input" or f in new_role_ids:
                valid_from.append(f)
            else:
                # Try to find matching role type in child
                f_type = self._get_role_type(f)
                for new_role in new_roles:
                    if self._get_role_type(new_role.id) == f_type:
                        valid_from.append(new_role.id)
                        break

        # Map to node
        valid_to: str | None = None
        if to_node in new_role_ids:
            valid_to = to_node
        else:
            to_type = self._get_role_type(to_node)
            for new_role in new_roles:
                if self._get_role_type(new_role.id) == to_type:
                    valid_to = new_role.id
                    break

        if not valid_from or not valid_to:
            return None

        new_from: str | list[str] = valid_from[0] if len(valid_from) == 1 else valid_from
        return GraphEdge(from_=new_from, to=valid_to, condition=edge.condition)

    def _increment_version(self, version: str, major: bool = False, minor: bool = False) -> str:
        """
        Increment semantic version

        Args:
            version: Current version string (e.g., "1.2.3")
            major: If True, increment major version
            minor: If True, increment minor version
            Otherwise: Increment patch version

        Returns:
            New version string
        """
        parts = version.split(".")
        maj, min_ver, patch = int(parts[0]), int(parts[1]), int(parts[2].split("-")[0])

        if major:
            return f"{maj + 1}.0.0"
        if minor:
            return f"{maj}.{min_ver + 1}.0"
        return f"{maj}.{min_ver}.{patch + 1}"

    def _create_mutation_from_evolution(
        self,
        parent_sop: ProcessConfig,
        parent_genome: PromptGenome,  # noqa: ARG002 - kept for future use
        evolution_result: "PromptEvolutionResult",
        diagnosis: Diagnosis,
    ) -> SOPMutation:
        """
        Create SOPMutation from PromptEvolutionResult.

        Args:
            parent_sop: Parent ProcessConfig
            parent_genome: Parent PromptGenome
            evolution_result: Result from PromptEvolutionOrchestrator
            diagnosis: Original diagnosis

        Returns:
            SOPMutation with evolved prompts
        """

        # Build detailed rationale from evolution result
        evolved_roles = list({change["role_id"] for change in evolution_result.changes_made})
        rationale_parts = [
            diagnosis.rootCauseAnalysis or "Section-based prompt evolution",
            evolution_result.rationale,
            f"Evolved roles: {', '.join(evolved_roles)}",
        ]

        return SOPMutation(
            parentSopId=parent_sop.id,
            parentVersion=parent_sop.version,
            newConfig=parent_sop,  # SOP config doesn't change for prompt mutations
            newPromptGenome=evolution_result.new_prompt_genome,
            rationale=RATIONALE_SEPARATOR.join(rationale_parts),
            mutationType=MutationType.PROMPT_CHANGE,
        )



class DirectorService:
    """
    Orchestrates Diagnostician and Architect for SOP evolution

    Combines diagnosis and mutation proposal in one service
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        meta_config: MetaConfig | None = None,
        model: str = "gpt-5",
        retry_handler: RetryHandler | None = None,
        circuit_breaker_registry: Optional["CircuitBreakerRegistry"] = None,
    ):
        """
        Initialize Director Service

        Args:
            llm_provider: LLM provider
            meta_config: Optional MetaConfig
            model: Model to use for diagnosis and mutation (default: "gpt-5")
            retry_handler: Retry handler for transient failures (creates default if None)
            circuit_breaker_registry: Circuit breaker registry (creates default if None)
        """
        from siare.services.circuit_breaker import get_circuit_breaker_registry

        # Initialize error handling
        self.retry_handler = retry_handler or RetryHandler()
        registry = circuit_breaker_registry or get_circuit_breaker_registry()

        # Create separate circuit breakers for each component
        diagnostician_breaker = registry.get_or_create(
            "diagnostician_llm",
            CircuitBreaker.LLM_CIRCUIT_CONFIG,
        )
        architect_breaker = registry.get_or_create(
            "architect_llm",
            CircuitBreaker.LLM_CIRCUIT_CONFIG,
        )

        # Initialize child components with error handling
        self.diagnostician = Diagnostician(
            llm_provider,
            meta_config,
            model=model,
            retry_handler=self.retry_handler,
            circuit_breaker=diagnostician_breaker,
        )
        self.architect = Architect(
            llm_provider,
            meta_config,
            model=model,
            retry_handler=self.retry_handler,
            circuit_breaker=architect_breaker,
        )

    def _fire_hook(self, hook_name: str, ctx: HookContext, *args: Any, **kwargs: Any) -> Any:
        """Fire an evolution hook from sync context.

        Safely runs async hooks from synchronous code. If no hooks are registered,
        returns immediately with zero overhead. Errors are logged but don't propagate.

        Args:
            hook_name: Name of the hook method (e.g., "on_mutation_complete").
            ctx: Hook context with correlation ID and metadata.
            *args: Arguments for the hook.
            **kwargs: Keyword arguments for the hook.

        Returns:
            Hook result, or None if no hooks registered or hook failed.
        """
        # Fast path: no hooks registered = no overhead
        if HookRegistry.get_evolution_hooks() is None:
            return None

        try:
            # Try to get existing event loop (may be running in async context)
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context - create task but don't await
                loop.create_task(fire_evolution_hook(hook_name, ctx, *args, **kwargs))
                return None
            except RuntimeError:
                # No running loop - create new one for sync context
                return asyncio.run(fire_evolution_hook(hook_name, ctx, *args, **kwargs))
        except Exception as e:
            logger.warning(f"Failed to fire hook {hook_name}: {e}")
            return None

    def propose_improvements(
        self,
        sop_gene: SOPGene,
        sop_config: ProcessConfig,
        prompt_genome: PromptGenome,
        metrics_to_optimize: list[str],
        mutation_types: list[MutationType],
        constraints: dict[str, Any] | None = None,
    ) -> tuple[Diagnosis, SOPMutation]:
        """
        Diagnose and propose mutation in one step

        Args:
            sop_gene: SOPGene to improve
            sop_config: Current ProcessConfig
            prompt_genome: Current PromptGenome
            metrics_to_optimize: Metrics to optimize
            mutation_types: Allowed mutation types
            constraints: Optional constraints

        Returns:
            (Diagnosis, SOPMutation) tuple
        """
        # Create hook context for this evolution cycle
        hook_ctx = HookContext(
            correlation_id=str(uuid.uuid4()),
            metadata={
                "sop_id": sop_config.id,
                "sop_version": sop_config.version,
                "metrics": metrics_to_optimize,
            },
        )

        # Step 1: Diagnose
        diagnosis = self.diagnostician.diagnose(
            sop_gene, sop_config, prompt_genome, metrics_to_optimize
        )

        # Fire mutation start hook (with diagnosis)
        self._fire_hook("on_mutation_start", hook_ctx, sop_config, diagnosis)

        # Step 2: Propose mutation
        mutation = self.architect.propose_mutation(
            diagnosis, sop_config, prompt_genome, mutation_types, constraints
        )

        # Fire mutation complete hook
        self._fire_hook(
            "on_mutation_complete",
            hook_ctx,
            sop_config,  # original
            mutation.newConfig,  # mutated
            mutation.mutationType,
        )

        return diagnosis, mutation
