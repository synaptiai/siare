"""Section-based prompt mutation utilities."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from siare.core.models import (
    ConstraintViolation,
    ParsedPrompt,
    PromptSection,
    RolePrompt,
    SectionMutation,
    SectionMutationBatch,
)
from siare.services.prompt_evolution.constraint_validator import ConstraintValidator

logger = logging.getLogger(__name__)


class BaseSectionMutator(ABC):
    """Abstract base class for section-level mutation."""

    @abstractmethod
    def apply_mutation(
        self,
        parsed_prompt: ParsedPrompt,
        mutation: SectionMutation,
        original_prompt: RolePrompt,
    ) -> tuple[str, list[ConstraintViolation]]:
        """Apply a single mutation to a parsed prompt.

        Args:
            parsed_prompt: The parsed prompt structure.
            mutation: The mutation to apply.
            original_prompt: Original prompt with constraints.

        Returns:
            Tuple of (reconstructed_content, violations).
        """

    @abstractmethod
    def apply_batch(
        self,
        parsed_prompt: ParsedPrompt,
        batch: SectionMutationBatch,
        original_prompt: RolePrompt,
    ) -> tuple[str, list[ConstraintViolation]]:
        """Apply a batch of mutations atomically.

        Args:
            parsed_prompt: The parsed prompt structure.
            batch: Batch of mutations to apply.
            original_prompt: Original prompt with constraints.

        Returns:
            Tuple of (reconstructed_content, all_violations).
        """


class SectionBasedPromptMutator(BaseSectionMutator):
    """Applies section-level mutations with constraint validation.

    Features:
    - Section-aware mutation application
    - Constraint validation before and after mutation
    - Rollback on constraint violation (strict mode)
    - Preserves immutable sections
    - Maintains markdown structure
    """

    def __init__(self, validation_mode: str = "strict") -> None:
        """Initialize mutator.

        Args:
            validation_mode: "strict" (reject invalid), "warn" (allow with warning).
        """
        self.validation_mode = validation_mode
        self.validator = ConstraintValidator(mode=validation_mode)

    def apply_mutation(
        self,
        parsed_prompt: ParsedPrompt,
        mutation: SectionMutation,
        original_prompt: RolePrompt,
    ) -> tuple[str, list[ConstraintViolation]]:
        """Apply a single mutation to a parsed prompt."""
        # Validate mutation before applying
        violations = self.validator.validate_mutation(
            mutation=mutation,
            original_prompt=original_prompt,
            parsed_prompt=parsed_prompt,
        )

        if violations and self.validation_mode == "strict":
            # Return original content unchanged
            logger.warning(
                f"Mutation rejected due to violations: {[v.violation_description for v in violations]}"
            )
            return self._reconstruct(parsed_prompt.sections), violations

        # Find target section and apply mutation
        updated_sections = self._apply_to_sections(parsed_prompt.sections, mutation)

        # Reconstruct prompt
        result = self._reconstruct(updated_sections)

        return result, violations

    def apply_batch(
        self,
        parsed_prompt: ParsedPrompt,
        batch: SectionMutationBatch,
        original_prompt: RolePrompt,
    ) -> tuple[str, list[ConstraintViolation]]:
        """Apply multiple mutations in batch."""
        all_violations: list[ConstraintViolation] = []
        current_sections = list(parsed_prompt.sections)

        for mutation in batch.mutations:
            # Create temporary ParsedPrompt for validation
            temp_parsed = ParsedPrompt(
                role_id=parsed_prompt.role_id,
                sections=current_sections,
                original_content=self._reconstruct(current_sections),
            )

            violations = self.validator.validate_mutation(
                mutation=mutation,
                original_prompt=original_prompt,
                parsed_prompt=temp_parsed,
            )

            all_violations.extend(violations)

            if not violations or self.validation_mode != "strict":
                # Apply mutation
                current_sections = self._apply_to_sections(current_sections, mutation)

        result = self._reconstruct(current_sections)
        return result, all_violations

    def _apply_to_sections(
        self,
        sections: list[PromptSection],
        mutation: SectionMutation,
    ) -> list[PromptSection]:
        """Apply mutation to matching section."""
        updated_sections: list[PromptSection] = []

        for section in sections:
            if section.id == mutation.section_id:
                # Apply mutation based on type
                new_content = self._apply_mutation_type(
                    original=section.content,
                    mutated=mutation.mutated_content,
                    mutation_type=mutation.mutation_type,
                )

                # Create updated section
                updated_section = PromptSection(
                    id=section.id,
                    content=new_content,
                    section_type=section.section_type,
                    is_mutable=section.is_mutable,
                    parent_role_id=section.parent_role_id,
                )
                updated_sections.append(updated_section)
            else:
                updated_sections.append(section)

        return updated_sections

    def _apply_mutation_type(
        self,
        original: str,
        mutated: str,
        mutation_type: str,
    ) -> str:
        """Apply specific mutation type."""
        if mutation_type == "replace":
            return mutated
        if mutation_type == "append":
            return f"{original}\n{mutated}"
        if mutation_type == "prepend":
            return f"{mutated}\n{original}"
        if mutation_type == "refine":
            # Refine means integrate improvements - default to replace
            return mutated
        logger.warning(
            f"Unknown mutation type: {mutation_type}, defaulting to replace"
        )
        return mutated

    def _reconstruct(self, sections: list[PromptSection]) -> str:
        """Reconstruct prompt from sections.

        Simply concatenates section content with blank lines between them.
        Note: The actual header reconstruction is handled by the parser.
        This is a simplified version for mutation application.
        """
        parts: list[str] = []

        for section in sections:
            # Add content
            parts.append(section.content)

        # Join with double newlines between sections
        return "\n\n".join(parts)
