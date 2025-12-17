"""
TextGrad Strategy Implementation

Textual gradient descent for prompt optimization based on:
- TextGrad (Nature, 2024): https://github.com/zou-group/textgrad

Key features:
- LLM-generated "textual gradients" (critiques)
- Gradient aggregation across traces
- Learning rate controlled updates
- DAG backpropagation to upstream roles
"""

from typing import Any

from siare.core.models import (
    Diagnosis,
    ParsedPrompt,
    ProcessConfig,
    PromptEvolutionResult,
    PromptFeedback,
    PromptGenome,
    PromptOptimizationStrategyType,
    RolePrompt,
    TextGradConfig,
)
from siare.services.llm_provider import LLMMessage, LLMProvider
from siare.services.prompt_evolution.strategies.base import BasePromptOptimizationStrategy

# Constants
TEXT_TRUNCATE_LENGTH = 200
VERSION_PARTS_MIN = 3
CHANGES_DISPLAY_LIMIT = 3
LEARNING_RATE_CONSERVATIVE_THRESHOLD = 0.3
LEARNING_RATE_MODERATE_THRESHOLD = 0.7


class TextualGradient:
    """Represents a textual gradient for a prompt section"""

    def __init__(
        self,
        role_id: str,
        section_id: str,
        critique: str,
        improvement_direction: str,
        magnitude: float = 1.0,
        source_feedback: PromptFeedback | None = None,
    ):
        self.role_id = role_id
        self.section_id = section_id
        self.critique = critique
        self.improvement_direction = improvement_direction
        self.magnitude = magnitude
        self.source_feedback = source_feedback

    def __repr__(self) -> str:
        return (
            f"TextualGradient(role={self.role_id}, section={self.section_id}, "
            f"magnitude={self.magnitude:.2f})"
        )


class TextGradStrategy(BasePromptOptimizationStrategy):
    """
    TextGrad optimization using textual gradient descent.

    Process:
    1. Compute textual gradients from feedback (critique + improvement direction)
    2. Aggregate gradients across multiple feedback items
    3. Apply gradients with learning rate to produce updated prompts
    4. Optionally backpropagate to upstream roles in DAG

    Based on TextGrad (Nature, 2024).
    """

    def __init__(
        self,
        llm_provider: LLMProvider | None = None,
        config: TextGradConfig | None = None,
    ):
        """
        Initialize TextGrad strategy.

        Args:
            llm_provider: LLM provider for gradient computation and application
            config: Strategy configuration
        """
        self.llm_provider = llm_provider
        self.config = config or TextGradConfig()

    @property
    def name(self) -> str:
        return "textgrad"

    @property
    def strategy_type(self) -> PromptOptimizationStrategyType:
        return PromptOptimizationStrategyType.TEXTGRAD

    def requires_population(self) -> bool:
        return False

    def optimize(
        self,
        sop_config: ProcessConfig,
        prompt_genome: PromptGenome,
        feedback: list[PromptFeedback],
        diagnosis: Diagnosis,
        parsed_prompts: dict[str, ParsedPrompt] | None = None,
        constraints: dict[str, Any] | None = None,
    ) -> PromptEvolutionResult:
        """
        Apply textual gradient descent to evolve prompts.

        1. Compute gradients from feedback
        2. Aggregate gradients per role/section
        3. Apply gradients with learning rate
        4. Backpropagate through DAG if configured

        Args:
            sop_config: Current SOP configuration
            prompt_genome: Current prompt genome
            feedback: Structured feedback from LLM critic
            diagnosis: Diagnosis from Diagnostician
            parsed_prompts: Pre-parsed prompts (optional)
            constraints: Evolution constraints

        Returns:
            PromptEvolutionResult with evolved genome
        """
        constraints = constraints or {}
        changes_made: list[dict[str, Any]] = []
        metadata: dict[str, Any] = {
            "strategy": "textgrad",
            "learning_rate": self.config.learning_rate,
            "backprop_depth": self.config.backprop_depth,
        }

        if not feedback:
            return PromptEvolutionResult(
                new_prompt_genome=prompt_genome,
                changes_made=[],
                rationale="No feedback provided for gradient computation",
                strategy_metadata=metadata,
            )

        # Step 1: Compute textual gradients from feedback
        gradients = self._compute_gradients(feedback, diagnosis)

        if not gradients:
            return PromptEvolutionResult(
                new_prompt_genome=prompt_genome,
                changes_made=[],
                rationale="No gradients could be computed from feedback",
                strategy_metadata=metadata,
            )

        # Step 2: Aggregate gradients per role
        aggregated = self._aggregate_gradients(gradients)

        # Step 3: Build DAG for backpropagation
        dag = self._build_dag(sop_config)

        # Step 4: Backpropagate gradients through DAG
        if self.config.backprop_depth > 0:
            aggregated = self._backpropagate(aggregated, dag, self.config.backprop_depth)

        # Step 5: Apply gradients to produce new prompts
        new_role_prompts = dict(prompt_genome.rolePrompts)

        for role_id, role_gradients in aggregated.items():
            if role_id not in prompt_genome.rolePrompts:
                continue

            current_prompt = prompt_genome.rolePrompts[role_id]
            role_constraints: dict[str, Any] = constraints.get(role_id, {})  # type: ignore[assignment]
            must_not_change: list[str] = role_constraints.get("mustNotChange", [])
            parsed = parsed_prompts.get(role_id) if parsed_prompts else None

            # Apply gradients to get new content
            new_content = self._apply_gradients(
                current_content=current_prompt.content,
                gradients=role_gradients,
                parsed_prompt=parsed,
                must_not_change=must_not_change,
            )

            # Validate constraints
            violations = self.validate_constraints(
                current_prompt.content,
                new_content,
                must_not_change,
            )

            if violations:
                # Skip if constraints violated
                continue

            # Record change
            if new_content != current_prompt.content:
                changes_made.append({
                    "role_id": role_id,
                    "section_id": "aggregated",
                    "old": current_prompt.content[:TEXT_TRUNCATE_LENGTH] + "..."
                    if len(current_prompt.content) > TEXT_TRUNCATE_LENGTH
                    else current_prompt.content,
                    "new": new_content[:TEXT_TRUNCATE_LENGTH] + "..."
                    if len(new_content) > TEXT_TRUNCATE_LENGTH
                    else new_content,
                    "gradient_count": len(role_gradients),
                    "avg_magnitude": sum(g.magnitude for g in role_gradients) / len(role_gradients),
                })

                new_role_prompts[role_id] = RolePrompt(
                    id=current_prompt.id,
                    content=new_content,
                    constraints=current_prompt.constraints,
                )

        # Create new genome
        new_genome = PromptGenome(
            id=prompt_genome.id,
            version=self._increment_version(prompt_genome.version),
            rolePrompts=new_role_prompts,
            metadata={
                **(prompt_genome.metadata or {}),
                "textgrad_applied": True,
            },
        )

        metadata["total_gradients"] = len(gradients)
        metadata["roles_updated"] = len(changes_made)

        rationale = self._build_rationale(changes_made, gradients)

        return PromptEvolutionResult(
            new_prompt_genome=new_genome,
            changes_made=changes_made,
            rationale=rationale,
            strategy_metadata=metadata,
        )

    def _compute_gradients(
        self,
        feedback: list[PromptFeedback],
        diagnosis: Diagnosis,
    ) -> list[TextualGradient]:
        """
        Compute textual gradients from feedback.

        Each feedback item produces a gradient with:
        - critique: What's wrong
        - improvement_direction: How to fix it
        - magnitude: Importance/confidence
        """
        gradients: list[TextualGradient] = []

        for fb in feedback:
            # Compute improvement direction
            if fb.suggested_improvement:
                improvement = fb.suggested_improvement
            else:
                improvement = self._derive_improvement(fb)

            # Magnitude based on confidence and failure pattern severity
            magnitude = fb.confidence
            if fb.failure_pattern:
                severity_weights = {
                    "hallucination": 1.5,
                    "safety_violation": 2.0,
                    "reasoning_error": 1.3,
                    "incomplete": 1.0,
                    "irrelevant": 1.0,
                    "timeout": 0.8,
                    "tool_misuse": 1.2,
                    "format_error": 0.7,
                    "context_loss": 1.1,
                }
                weight = severity_weights.get(fb.failure_pattern.value, 1.0)
                magnitude *= weight

            gradient = TextualGradient(
                role_id=fb.role_id,
                section_id=fb.section_id,
                critique=fb.critique,
                improvement_direction=improvement,
                magnitude=magnitude,
                source_feedback=fb,
            )
            gradients.append(gradient)

        return gradients

    def _derive_improvement(self, feedback: PromptFeedback) -> str:
        """Derive improvement direction from critique when not provided."""
        critique = feedback.critique.lower()

        # Pattern-based improvement suggestions
        if "shallow" in critique or "lack depth" in critique:
            return "Add more detailed analysis steps and specific criteria"
        if "missing" in critique or "incomplete" in critique:
            return "Ensure all required elements are addressed comprehensively"
        if "unclear" in critique or "ambiguous" in critique:
            return "Use more precise language and specific examples"
        if "hallucination" in critique or "fabricat" in critique:
            return "Add explicit verification steps and source requirements"
        if "format" in critique or "structure" in critique:
            return "Provide clearer output format specification"
        if "reasoning" in critique or "logic" in critique:
            return "Add step-by-step reasoning requirements"
        return f"Address: {feedback.critique[:100]}"

    def _aggregate_gradients(
        self,
        gradients: list[TextualGradient],
    ) -> dict[str, list[TextualGradient]]:
        """
        Aggregate gradients by role.

        For multiple gradients on same section, keeps highest magnitude
        or merges based on aggregation method.
        """
        aggregated: dict[str, list[TextualGradient]] = {}

        for gradient in gradients:
            role_id = gradient.role_id
            if role_id not in aggregated:
                aggregated[role_id] = []

            # Check for existing gradient on same section
            existing = None
            for i, g in enumerate(aggregated[role_id]):
                if g.section_id == gradient.section_id:
                    existing = (i, g)
                    break

            if existing:
                idx, existing_grad = existing
                if self.config.gradient_aggregation == "mean":
                    # Average the gradients
                    merged = TextualGradient(
                        role_id=role_id,
                        section_id=gradient.section_id,
                        critique=f"{existing_grad.critique}; {gradient.critique}",
                        improvement_direction=f"{existing_grad.improvement_direction}; {gradient.improvement_direction}",
                        magnitude=(existing_grad.magnitude + gradient.magnitude) / 2,
                    )
                    aggregated[role_id][idx] = merged
                elif self.config.gradient_aggregation == "max":
                    # Keep highest magnitude
                    if gradient.magnitude > existing_grad.magnitude:
                        aggregated[role_id][idx] = gradient
                else:  # sum
                    # Sum magnitudes
                    merged = TextualGradient(
                        role_id=role_id,
                        section_id=gradient.section_id,
                        critique=f"{existing_grad.critique}; {gradient.critique}",
                        improvement_direction=f"{existing_grad.improvement_direction}; {gradient.improvement_direction}",
                        magnitude=existing_grad.magnitude + gradient.magnitude,
                    )
                    aggregated[role_id][idx] = merged
            else:
                aggregated[role_id].append(gradient)

        return aggregated

    def _build_dag(self, sop_config: ProcessConfig) -> dict[str, list[str]]:
        """
        Build DAG structure from SOP config.

        Returns dict mapping role_id -> list of upstream role_ids
        """
        dag: dict[str, list[str]] = {}

        for role in sop_config.roles:
            dag[role.id] = []

        for edge in sop_config.graph:
            to_role = edge.to
            from_roles = edge.from_ if isinstance(edge.from_, list) else [edge.from_]

            if to_role not in dag:
                dag[to_role] = []

            dag[to_role].extend(from_roles)

        return dag

    def _backpropagate(
        self,
        gradients: dict[str, list[TextualGradient]],
        dag: dict[str, list[str]],
        depth: int,
    ) -> dict[str, list[TextualGradient]]:
        """
        Backpropagate gradients to upstream roles.

        Gradients flow backwards through the DAG, attenuated by learning rate.
        """
        if depth <= 0:
            return gradients

        result = {k: list(v) for k, v in gradients.items()}  # Deep copy

        for _ in range(depth):
            new_gradients: dict[str, list[TextualGradient]] = {}

            for role_id, role_grads in result.items():
                upstream_roles = dag.get(role_id, [])

                for upstream_role in upstream_roles:
                    if upstream_role not in new_gradients:
                        new_gradients[upstream_role] = []

                    for grad in role_grads:
                        # Create attenuated gradient for upstream
                        backprop_grad = TextualGradient(
                            role_id=upstream_role,
                            section_id="backprop",
                            critique=f"Downstream issue in {role_id}: {grad.critique[:100]}",
                            improvement_direction=f"Improve output to help downstream: {grad.improvement_direction[:100]}",
                            magnitude=grad.magnitude * self.config.learning_rate,
                        )
                        new_gradients[upstream_role].append(backprop_grad)

            # Merge new gradients into result
            for role_id, grads in new_gradients.items():
                if role_id not in result:
                    result[role_id] = []
                result[role_id].extend(grads)

        return result

    def _apply_gradients(
        self,
        current_content: str,
        gradients: list[TextualGradient],
        parsed_prompt: ParsedPrompt | None,
        must_not_change: list[str],
    ) -> str:
        """
        Apply textual gradients to produce updated content.

        Uses LLM if available, otherwise applies heuristic updates.
        """
        if not gradients:
            return current_content

        if self.llm_provider:
            return self._llm_apply_gradients(
                current_content, gradients, must_not_change
            )
        return self._heuristic_apply_gradients(
            current_content, gradients, parsed_prompt, must_not_change
        )

    def _llm_apply_gradients(
        self,
        content: str,
        gradients: list[TextualGradient],
        must_not_change: list[str],
    ) -> str:
        """Use LLM to apply gradients intelligently."""
        assert self.llm_provider is not None, "LLM provider must be set"
        protected_text = "\n".join(f"- {p}" for p in must_not_change) if must_not_change else "None"

        gradient_details = ""
        for i, grad in enumerate(gradients[:5], 1):  # Limit to top 5
            gradient_details += f"\n{i}. [Magnitude: {grad.magnitude:.2f}]"
            gradient_details += f"\n   Issue: {grad.critique[:150]}"
            gradient_details += f"\n   Fix: {grad.improvement_direction[:150]}"

        learning_rate_desc = (
            "conservative" if self.config.learning_rate < LEARNING_RATE_CONSERVATIVE_THRESHOLD
            else "moderate" if self.config.learning_rate < LEARNING_RATE_MODERATE_THRESHOLD
            else "aggressive"
        )

        messages = [
            LLMMessage(
                role="system",
                content=f"""You are an expert prompt engineer applying textual gradients to improve a prompt.

Your task is to apply the suggested improvements while:
1. Preserving protected text exactly as-is
2. Making {learning_rate_desc} changes (learning rate: {self.config.learning_rate})
3. Maintaining the prompt's structure and intent
4. Focusing on the highest-magnitude issues first

Output ONLY the improved prompt, no explanations.""",
            ),
            LLMMessage(
                role="user",
                content=f"""Apply these textual gradients to improve the prompt:

CURRENT PROMPT:
{content}

GRADIENTS TO APPLY:
{gradient_details}

PROTECTED TEXT (must keep exactly):
{protected_text}

Output only the improved prompt:""",
            ),
        ]

        try:
            response = self.llm_provider.complete(
                messages=messages,
                model=self.config.model,
                temperature=0.5,  # Lower temperature for precision
                max_tokens=2000,
            )
            return response.content.strip()
        except Exception:
            # Fallback to heuristic
            return self._heuristic_apply_gradients(
                content, gradients, None, must_not_change
            )

    def _heuristic_apply_gradients(
        self,
        content: str,
        gradients: list[TextualGradient],
        parsed_prompt: ParsedPrompt | None,
        must_not_change: list[str],
    ) -> str:
        """Apply gradients using heuristic rules."""
        lines = content.split("\n")
        modifications: list[dict[str, str]] = []

        # Sort gradients by magnitude (highest first)
        sorted_grads = sorted(gradients, key=lambda g: g.magnitude, reverse=True)

        # Apply top gradients based on learning rate
        max_changes = max(1, int(len(sorted_grads) * self.config.learning_rate))

        for grad in sorted_grads[:max_changes]:
            # Find relevant section
            if parsed_prompt:
                target_section = None
                for section in parsed_prompt.sections:
                    if section.id == grad.section_id or grad.section_id in section.content:
                        target_section = section
                        break

                if target_section and target_section.is_mutable:
                    # Add improvement as a note in that section
                    modifications.append({
                        "type": "add_note",
                        "content": grad.improvement_direction,
                    })
            else:
                # Without parsed prompt, add as general improvement
                modifications.append({
                    "type": "add_note",
                    "content": grad.improvement_direction,
                })

        # Apply modifications
        if modifications:
            # Add improvements at the end of Instructions section or before Constraints
            insertion_point = len(lines)
            for i, line in enumerate(lines):
                if "constraint" in line.lower() or "immutable" in line.lower():
                    insertion_point = i
                    break

            notes_to_add: list[str] = []
            for mod in modifications[:CHANGES_DISPLAY_LIMIT]:  # Limit additions
                if mod["type"] == "add_note" and not any(p in mod["content"] for p in must_not_change):
                    notes_to_add.append(f"- {mod['content']}")

            if notes_to_add:
                lines.insert(insertion_point, "\n### Improvements")
                note: str
                for note in notes_to_add:
                    lines.insert(insertion_point + 1, note)

        return "\n".join(lines)

    def _increment_version(self, version: str) -> str:
        """Increment patch version."""
        try:
            parts = version.split(".")
            if len(parts) >= VERSION_PARTS_MIN:
                parts[2] = str(int(parts[2]) + 1)
                return ".".join(parts)
        except (ValueError, IndexError):
            pass
        return f"{version}.1"

    def _build_rationale(
        self,
        changes_made: list[dict[str, Any]],
        gradients: list[TextualGradient],
    ) -> str:
        """Build human-readable rationale."""
        if not changes_made:
            return f"No changes made despite {len(gradients)} gradient(s) computed"

        rationale_parts = [
            f"TextGrad: Applied {len(gradients)} gradient(s) to {len(changes_made)} role(s)"
        ]

        for change in changes_made[:CHANGES_DISPLAY_LIMIT]:
            grad_count: int = change.get("gradient_count", 0)  # type: ignore[assignment]
            avg_mag: float = change.get("avg_magnitude", 0)  # type: ignore[assignment]
            rationale_parts.append(
                f"- {change['role_id']}: {grad_count} gradient(s), avg magnitude {avg_mag:.2f}"
            )

        if len(changes_made) > CHANGES_DISPLAY_LIMIT:
            rationale_parts.append(f"- ... and {len(changes_made) - CHANGES_DISPLAY_LIMIT} more")

        return "\n".join(rationale_parts)
