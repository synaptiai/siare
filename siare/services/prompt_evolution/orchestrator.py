"""
Prompt Evolution Orchestrator

Coordinates the complete prompt evolution workflow:
1. Parse prompts into sections
2. Extract feedback from evaluations/traces
3. Select optimization strategy
4. Generate and validate mutations
5. Apply valid mutations
6. Track evolution history
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from siare.core.models import (
    Diagnosis,
    EvaluationVector,
    ParsedPrompt,
    ProcessConfig,
    PromptEvolutionOrchestratorConfig,
    PromptEvolutionResult,
    PromptFeedback,
    PromptGenome,
)
from siare.services.prompt_evolution.constraint_validator import ConstraintValidator
from siare.services.prompt_evolution.critic import TraceFeedbackExtractor
from siare.services.prompt_evolution.feedback_extractor import (
    FeedbackArtifactExtractor,
)
from siare.services.prompt_evolution.parser import (
    LLMSectionParser,
    MarkdownSectionParser,
)
from siare.services.prompt_evolution.section_mutator import SectionBasedPromptMutator
from siare.services.prompt_evolution.selector import AdaptiveStrategySelector

if TYPE_CHECKING:
    from siare.services.execution_engine import ExecutionTrace
    from siare.services.llm_provider import LLMProvider


logger = logging.getLogger(__name__)

# Constants
MAX_EVOLUTION_HISTORY = 100
VERSION_PARTS_COUNT = 3


class PromptEvolutionOrchestrator:
    """
    Orchestrates the complete prompt evolution workflow.

    Workflow:
    1. Parse prompts into sections (markdown or LLM-based)
    2. Extract feedback from evaluations and traces
    3. Select optimization strategy based on failure patterns
    4. Run strategy to generate mutations
    5. Validate mutations against constraints
    6. Apply valid mutations and update genome
    7. Track evolution history

    Features:
    - Section-based surgical mutations
    - Multi-source feedback aggregation
    - Adaptive strategy selection
    - Constraint validation
    - Evolution history tracking
    """

    def __init__(
        self,
        llm_provider: LLMProvider | None = None,
        config: PromptEvolutionOrchestratorConfig | None = None,
    ):
        """
        Initialize orchestrator.

        Args:
            llm_provider: LLM provider for optimization strategies
            config: Orchestrator configuration
        """
        self.llm_provider = llm_provider
        self.config = config or PromptEvolutionOrchestratorConfig()

        # Initialize components
        self.parser = (
            LLMSectionParser(llm_provider=llm_provider)
            if self.config.enable_section_parsing
            else MarkdownSectionParser()
        )

        self.feedback_extractor = FeedbackArtifactExtractor()
        self.critic = TraceFeedbackExtractor(llm_provider=llm_provider)

        self.strategy_selector = AdaptiveStrategySelector(
            llm_provider=llm_provider,
            default_strategy=self.config.default_strategy,
        )

        self.section_mutator = SectionBasedPromptMutator(
            validation_mode=self.config.constraint_validation_mode
        )

        self.constraint_validator = ConstraintValidator(
            mode=self.config.constraint_validation_mode
        )

        # Track evolution history
        self._evolution_history: list[dict[str, Any]] = []

    def evolve(
        self,
        sop_config: ProcessConfig,
        prompt_genome: PromptGenome,
        diagnosis: Diagnosis,
        traces: list[ExecutionTrace],
        evaluations: list[EvaluationVector],
        constraints: dict[str, Any] | None = None,
    ) -> PromptEvolutionResult:
        """
        Evolve prompts based on diagnosis and feedback.

        Args:
            sop_config: Current SOP configuration
            prompt_genome: Current prompt genome
            diagnosis: Diagnosis from Diagnostician
            traces: Execution traces
            evaluations: Evaluation vectors
            constraints: Optional evolution constraints

        Returns:
            PromptEvolutionResult with evolved genome and metadata
        """
        # Deep copy genome to avoid mutation
        new_genome = deepcopy(prompt_genome)

        # Step 1: Parse prompts into sections for all roles
        parsed_prompts: dict[str, ParsedPrompt] = {}
        for role in sop_config.roles:
            if role.promptRef and role.promptRef in prompt_genome.rolePrompts:
                prompt = prompt_genome.rolePrompts[role.promptRef]
                parsed = self.parser.parse(prompt=prompt, role_id=role.id)
                parsed_prompts[role.id] = parsed
            else:
                logger.warning(f"Role {role.id} missing prompt ref {role.promptRef}")

        # If no roles could be parsed, skip evolution
        if not parsed_prompts:
            logger.info("No parseable roles found, skipping evolution")
            return PromptEvolutionResult(
                new_prompt_genome=new_genome,
                changes_made=[],
                rationale="No parseable roles found for evolution",
            )

        # Step 2: Extract feedback from multiple sources
        feedback_list: list[PromptFeedback] = []

        # Extract from traces using LLM critic
        if traces:
            trace_feedback = self.critic.extract_feedback(
                traces=traces,
                sop_config=sop_config,
                prompt_genome=prompt_genome,
                evaluations=evaluations,
            )
            feedback_list.extend(trace_feedback)

        # Extract from evaluation artifacts for all roles
        for role_id in parsed_prompts:
            artifact_feedback = self.feedback_extractor.extract(
                evaluations=evaluations,
                role_id=role_id,
            )
            # Convert FeedbackArtifacts to PromptFeedback (role-level, not section-specific)
            for artifact in artifact_feedback:
                feedback_list.append(
                    PromptFeedback(
                        role_id=role_id,
                        section_id="",  # Empty = role-level feedback (no specific section)
                        section_content="",
                        critique=artifact.critique,
                        failure_pattern=artifact.failure_pattern,
                        suggested_improvement=artifact.suggested_fix,
                        confidence=artifact.severity,
                    )
                )

        # Step 3: Select optimization strategy
        strategy_type = self.strategy_selector.select_strategy(
            feedback=feedback_list,
            diagnosis=diagnosis,
        )
        strategy = self.strategy_selector.get_strategy(strategy_type)

        logger.info(f"Selected strategy: {strategy_type.value}")

        # Step 4: Run strategy to generate mutations
        result = strategy.optimize(
            sop_config=sop_config,
            prompt_genome=new_genome,
            feedback=feedback_list,
            diagnosis=diagnosis,
            parsed_prompts=parsed_prompts,
            constraints=constraints,
        )

        # Step 5: Update genome version if changes were made
        if result.changes_made:
            new_genome = result.new_prompt_genome
            # Increment patch version (semantic versioning)
            version_parts = new_genome.version.split(".")
            if len(version_parts) == VERSION_PARTS_COUNT:
                major, minor, patch = version_parts
                new_genome.version = f"{major}.{minor}.{int(patch) + 1}"
            else:
                new_genome.version = "1.0.1"

            # Record evolution history
            self._record_evolution(
                strategy_type=strategy_type.value,
                changes=result.changes_made,
                rationale=result.rationale,
            )

        return PromptEvolutionResult(
            new_prompt_genome=new_genome,
            changes_made=result.changes_made,
            rationale=result.rationale,
            strategy_metadata=result.strategy_metadata,
        )

    def _record_evolution(
        self,
        strategy_type: str,
        changes: list[dict[str, Any]],
        rationale: str,
    ) -> None:
        """Record evolution event in history."""
        self._evolution_history.append({
            "strategy": strategy_type,
            "changes_count": len(changes),
            "rationale": rationale[:200],  # Truncate for storage
        })

        # Keep only last MAX_EVOLUTION_HISTORY events
        if len(self._evolution_history) > MAX_EVOLUTION_HISTORY:
            self._evolution_history = self._evolution_history[-MAX_EVOLUTION_HISTORY:]

    def get_evolution_history(self) -> list[dict[str, Any]]:
        """Get recent evolution history."""
        return list(self._evolution_history)

    def clear_history(self) -> None:
        """Clear evolution history."""
        self._evolution_history.clear()
