"""Inject feedback artifacts into prompts during execution."""

from __future__ import annotations

import copy
import logging
import re

from siare.core.models import (
    FeedbackArtifact,
    FeedbackInjectionConfig,
    PromptGenome,
    RolePrompt,
)


logger = logging.getLogger(__name__)

# Severity thresholds for classification
SEVERITY_HIGH_THRESHOLD = 0.8
SEVERITY_MEDIUM_THRESHOLD = 0.5


class FeedbackInjector:
    """Injects feedback artifacts into prompts during execution.

    Injection modes:
    1. prepend - Add feedback section at the start
    2. append - Add feedback section at the end
    3. <section_name> - Add after specific section
    """

    INJECTION_TEMPLATE = """## Previous Execution Feedback
_The following issues were identified in recent executions. Address these in your response._

{feedback_items}

---
"""

    FEEDBACK_ITEM_TEMPLATE = "- **{severity}** {critique}"

    def inject_feedback(
        self,
        prompt_genome: PromptGenome,
        feedback_artifacts: list[FeedbackArtifact],
        role_id: str,
        prompt_ref: str,
        config: FeedbackInjectionConfig | None = None,
    ) -> PromptGenome:
        """Inject feedback into a specific role's prompt.

        Args:
            prompt_genome: The genome to modify.
            feedback_artifacts: Feedback to inject.
            role_id: Role to inject feedback for.
            prompt_ref: Prompt reference key in rolePrompts.
            config: Injection configuration.

        Returns:
            Modified PromptGenome with feedback injected.
        """
        if config is None:
            config = FeedbackInjectionConfig()

        if not config.enabled:
            return prompt_genome

        # Filter feedback for this role
        role_feedback = [
            f for f in feedback_artifacts
            if f.role_id == role_id
        ]

        if not role_feedback:
            return prompt_genome

        # Sort by severity and limit
        role_feedback = sorted(role_feedback, key=lambda f: f.severity, reverse=True)
        role_feedback = role_feedback[:config.max_feedback_items]

        # Build injection content
        injection_content = self._build_injection_content(role_feedback, config)

        # Deep copy genome to avoid mutation
        modified_genome = copy.deepcopy(prompt_genome)

        # Get the prompt to modify
        if prompt_ref not in modified_genome.rolePrompts:
            logger.warning(f"Prompt ref '{prompt_ref}' not found in genome")
            return prompt_genome

        original_prompt = modified_genome.rolePrompts[prompt_ref]
        original_content = original_prompt.content

        # Inject based on position
        new_content = self._inject_content(
            original_content=original_content,
            injection_content=injection_content,
            position=config.injection_position,
        )

        # Update prompt
        modified_genome.rolePrompts[prompt_ref] = RolePrompt(
            id=original_prompt.id,
            content=new_content,
            constraints=original_prompt.constraints,
        )

        return modified_genome

    def _build_injection_content(
        self,
        feedback: list[FeedbackArtifact],
        config: FeedbackInjectionConfig,
    ) -> str:
        """Build formatted feedback injection content."""
        items: list[str] = []

        for artifact in feedback:
            # Format severity indicator
            if artifact.severity >= SEVERITY_HIGH_THRESHOLD:
                severity = "[HIGH]"
            elif artifact.severity >= SEVERITY_MEDIUM_THRESHOLD:
                severity = "[MEDIUM]"
            else:
                severity = "[LOW]"

            item = self.FEEDBACK_ITEM_TEMPLATE.format(
                severity=severity,
                critique=artifact.critique,
            )

            # Add suggestion if configured
            if config.include_suggestions and artifact.suggested_fix:
                item += f"\n  - Suggestion: {artifact.suggested_fix}"

            # Add failure pattern if configured
            if config.include_failure_patterns and artifact.failure_pattern:
                item += f"\n  - Pattern: {artifact.failure_pattern.value}"

            items.append(item)

        feedback_items = "\n".join(items)

        return self.INJECTION_TEMPLATE.format(feedback_items=feedback_items)

    def _inject_content(
        self,
        original_content: str,
        injection_content: str,
        position: str,
    ) -> str:
        """Inject content at specified position."""
        if position == "prepend":
            return f"{injection_content}\n{original_content}"
        if position == "append":
            return f"{original_content}\n\n{injection_content}"
        # Try to find section to insert after
        return self._insert_after_section(
            original_content,
            position,
            injection_content,
        )

    def _insert_after_section(
        self,
        content: str,
        section_name: str,
        injection: str,
    ) -> str:
        """Insert content after a named section."""
        # Find section header pattern
        pattern = rf"^(#{1,6})\s+{re.escape(section_name)}$"
        lines = content.split("\n")

        for i, line in enumerate(lines):
            if re.match(pattern, line, re.IGNORECASE):
                # Find end of section (next header or end)
                j = i + 1
                while j < len(lines):
                    if re.match(r"^#{1,6}\s+", lines[j]):
                        break
                    j += 1

                # Insert before next section
                lines.insert(j, "\n" + injection)
                return "\n".join(lines)

        # Section not found, append to end
        logger.warning(f"Section '{section_name}' not found, appending to end")
        return f"{content}\n\n{injection}"
