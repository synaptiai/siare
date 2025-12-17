"""Constraint validation for prompt mutations."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from siare.core.models import (
    ConstraintViolation,
    ParsedPrompt,
    PromptSection,
    RolePrompt,
    SectionMutation,
)


logger = logging.getLogger(__name__)

# Constants
TRUNCATION_LENGTH = 50


class BaseConstraintValidator(ABC):
    """Abstract base class for constraint validators."""

    @abstractmethod
    def validate_mutation(
        self,
        mutation: SectionMutation,
        original_prompt: RolePrompt,
        parsed_prompt: ParsedPrompt,
    ) -> list[ConstraintViolation]:
        """Validate a single mutation against constraints.

        Args:
            mutation: The mutation to validate.
            original_prompt: Original RolePrompt with constraints.
            parsed_prompt: Parsed prompt structure.

        Returns:
            List of constraint violations (empty if valid).
        """

    @abstractmethod
    def validate_batch(
        self,
        mutations: list[SectionMutation],
        original_prompt: RolePrompt,
        parsed_prompt: ParsedPrompt,
    ) -> list[ConstraintViolation]:
        """Validate a batch of mutations.

        Args:
            mutations: List of mutations to validate.
            original_prompt: Original RolePrompt with constraints.
            parsed_prompt: Parsed prompt structure.

        Returns:
            List of all constraint violations.
        """


class ConstraintValidator(BaseConstraintValidator):
    """Validates prompt mutations against constraints.

    Validates:
    1. mustNotChange - Protected text must be preserved
    2. Immutable sections - Cannot modify safety/policy sections
    3. Max/min length - Content length limits
    4. Safety patterns - Detect prompt injection attempts
    """

    # Patterns that indicate prompt injection attempts
    SAFETY_BLACKLIST: frozenset[str] = frozenset([
        "ignore previous instructions",
        "ignore all previous",
        "disregard previous",
        "disregard safety",
        "bypass policy",
        "forget your instructions",
        "new instructions",
        "override safety",
        "ignore safety",
        "ignore your rules",
        "pretend you are",
        "act as if you have no",
        "you are now",
        "from now on you",
    ])

    def __init__(self, mode: str = "strict") -> None:
        """Initialize validator.

        Args:
            mode: Validation mode - "strict" (fail on any violation),
                  "warn" (log warnings but allow), "disabled" (skip validation).
        """
        if mode not in ("strict", "warn", "disabled"):
            raise ValueError(f"Invalid mode: {mode}. Must be strict, warn, or disabled.")
        self.mode = mode

    def validate_mutation(
        self,
        mutation: SectionMutation,
        original_prompt: RolePrompt,
        parsed_prompt: ParsedPrompt,
    ) -> list[ConstraintViolation]:
        """Validate a single mutation against all constraints."""
        if self.mode == "disabled":
            return []

        violations: list[ConstraintViolation] = []

        # 1. Check mustNotChange constraints
        violations.extend(self._validate_must_not_change(mutation, original_prompt))

        # 2. Check immutable section modification
        violations.extend(self._validate_immutable_section(mutation, parsed_prompt))

        # 3. Check length constraints
        violations.extend(
            self._validate_length_constraints(mutation, original_prompt, parsed_prompt)
        )

        # 4. Check safety patterns
        violations.extend(self._validate_safety_patterns(mutation))

        if violations and self.mode == "warn":
            for v in violations:
                logger.warning(
                    f"Constraint violation (warn mode): {v.violation_description}"
                )

        return violations

    def validate_batch(
        self,
        mutations: list[SectionMutation],
        original_prompt: RolePrompt,
        parsed_prompt: ParsedPrompt,
    ) -> list[ConstraintViolation]:
        """Validate all mutations in a batch."""
        all_violations: list[ConstraintViolation] = []
        for mutation in mutations:
            violations = self.validate_mutation(mutation, original_prompt, parsed_prompt)
            all_violations.extend(violations)
        return all_violations

    def _validate_must_not_change(
        self,
        mutation: SectionMutation,
        original_prompt: RolePrompt,
    ) -> list[ConstraintViolation]:
        """Check that mustNotChange text is preserved."""
        violations: list[ConstraintViolation] = []

        if not original_prompt.constraints or not original_prompt.constraints.mustNotChange:
            return violations

        for protected_text in original_prompt.constraints.mustNotChange:
            # Check if protected text was in original and is missing in mutated
            if (
                protected_text in mutation.original_content
                and protected_text not in mutation.mutated_content
            ):
                # Truncate with ellipsis only if text is longer than threshold
                truncated = protected_text[:TRUNCATION_LENGTH] + (
                    "..." if len(protected_text) > TRUNCATION_LENGTH else ""
                )
                violations.append(
                    ConstraintViolation(
                        constraint_type="must_not_change",
                        violation_description=f"Protected text removed: '{truncated}'",
                        section_id=mutation.section_id,
                        role_id=mutation.role_id,
                        severity="error",
                    )
                )

        return violations

    def _validate_immutable_section(
        self,
        mutation: SectionMutation,
        parsed_prompt: ParsedPrompt,
    ) -> list[ConstraintViolation]:
        """Check that immutable sections are not modified."""
        violations: list[ConstraintViolation] = []

        # Find the section being mutated
        target_section: PromptSection | None = None
        for section in parsed_prompt.sections:
            if section.id == mutation.section_id:
                target_section = section
                break

        if target_section is None:
            # Section not found - this is a different kind of error
            return violations

        if (
            not target_section.is_mutable
            and mutation.mutated_content != mutation.original_content
        ):
            # Any change to immutable section is a violation
            # Get section type for description
            section_name = target_section.section_type.value
            violations.append(
                ConstraintViolation(
                    constraint_type="immutable_section",
                    violation_description=f"Attempted to modify immutable section '{section_name}'",
                    section_id=mutation.section_id,
                    role_id=mutation.role_id,
                    severity="error",
                )
            )

        return violations

    def _validate_length_constraints(
        self,
        mutation: SectionMutation,
        original_prompt: RolePrompt,
        parsed_prompt: ParsedPrompt,
    ) -> list[ConstraintViolation]:
        """Check length constraints."""
        violations: list[ConstraintViolation] = []

        if not original_prompt.constraints:
            return violations

        # Calculate new total length if mutation is applied
        current_total = len(parsed_prompt.original_content)
        length_change = len(mutation.mutated_content) - len(mutation.original_content)
        new_total = current_total + length_change

        if (
            original_prompt.constraints.maxLength
            and new_total > original_prompt.constraints.maxLength
        ):
            violations.append(
                ConstraintViolation(
                    constraint_type="max_length",
                    violation_description=f"Content would exceed maxLength: {new_total} > {original_prompt.constraints.maxLength}",
                    section_id=mutation.section_id,
                    role_id=mutation.role_id,
                    severity="error",
                )
            )

        if (
            original_prompt.constraints.minLength
            and new_total < original_prompt.constraints.minLength
        ):
            violations.append(
                ConstraintViolation(
                    constraint_type="min_length",
                    violation_description=f"Content would be below minLength: {new_total} < {original_prompt.constraints.minLength}",
                    section_id=mutation.section_id,
                    role_id=mutation.role_id,
                    severity="error",
                )
            )

        return violations

    def _validate_safety_patterns(
        self,
        mutation: SectionMutation,
    ) -> list[ConstraintViolation]:
        """Check for prompt injection and safety violations."""
        violations: list[ConstraintViolation] = []

        content_lower = mutation.mutated_content.lower()

        for pattern in self.SAFETY_BLACKLIST:
            if pattern in content_lower:
                violations.append(
                    ConstraintViolation(
                        constraint_type="safety_violation",
                        violation_description=f"Detected unsafe pattern: '{pattern}'",
                        section_id=mutation.section_id,
                        role_id=mutation.role_id,
                        severity="error",
                    )
                )
                break  # One safety violation is enough

        return violations
