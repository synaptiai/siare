"""Prompt diff tracking for evolution analysis.

Captures and compares prompts across evolution generations to show
what changed and how.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from siare.utils.diff import (
    DiffLine,
    compute_prompt_diff,
    format_diff_for_html,
    format_diff_for_markdown,
    has_changes,
)

if TYPE_CHECKING:
    from siare.core.models import PromptGenome


@dataclass
class PromptSnapshot:
    """Snapshot of a prompt at a specific generation."""

    role_id: str
    generation: int
    prompt_text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptDiff:
    """Diff between two versions of a prompt."""

    role_id: str
    from_generation: int
    to_generation: int
    diff_lines: list[DiffLine]
    has_changes: bool
    lines_added: int
    lines_removed: int

    def to_markdown(self) -> str:
        """Format diff as markdown."""
        if not self.has_changes:
            return f"_{self.role_id}: No changes_"

        header = f"### {self.role_id} (Gen {self.from_generation} → {self.to_generation})\n"
        stats = f"*+{self.lines_added} / -{self.lines_removed} lines*\n\n"
        diff_block = "```diff\n" + format_diff_for_markdown(self.diff_lines) + "\n```"
        return header + stats + diff_block

    def to_html(self) -> str:
        """Format diff as HTML."""
        if not self.has_changes:
            return f'<div class="no-changes">{self.role_id}: No changes</div>'

        return f"""
        <div class="prompt-diff">
            <h4>{self.role_id} (Gen {self.from_generation} → {self.to_generation})</h4>
            <div class="diff-stats">+{self.lines_added} / -{self.lines_removed} lines</div>
            <div class="diff-content">
                {format_diff_for_html(self.diff_lines)}
            </div>
        </div>
        """


@dataclass
class ChangeSummary:
    """Summary of all prompt changes."""

    total_roles: int
    roles_changed: int
    total_lines_added: int
    total_lines_removed: int
    most_changed_role: str | None
    change_descriptions: list[str]


class PromptDiffTracker:
    """Tracks prompt changes across evolution generations.

    Usage:
        tracker = PromptDiffTracker()
        tracker.capture_initial(initial_genome)

        # After each evolution generation
        tracker.capture_generation(generation=1, genome=evolved_genome_gen1)
        tracker.capture_generation(generation=2, genome=evolved_genome_gen2)

        # Get diffs
        diffs = tracker.compute_diffs()  # Initial → Final
        gen_diffs = tracker.compute_generation_diffs()  # Per-generation
    """

    def __init__(self) -> None:
        """Initialize the tracker."""
        self._initial_prompts: dict[str, str] = {}
        self._generation_prompts: dict[int, dict[str, str]] = {}
        self._initial_genome: PromptGenome | None = None
        self._final_genome: PromptGenome | None = None

    def capture_initial(self, genome: PromptGenome) -> None:
        """Capture initial prompts from genome.

        Args:
            genome: Initial PromptGenome to capture
        """
        self._initial_genome = genome
        self._initial_prompts = self._extract_prompts(genome)
        self._generation_prompts[0] = self._initial_prompts.copy()

    def set_initial_prompts(self, prompts: dict[str, str]) -> None:
        """Set initial prompts from a dictionary (for checkpoint restore).

        Args:
            prompts: Dict mapping role_id to prompt text
        """
        self._initial_prompts = prompts.copy()
        self._generation_prompts[0] = prompts.copy()

    def capture_generation(self, generation: int, genome: PromptGenome) -> None:
        """Capture prompts at a specific generation.

        Args:
            generation: Generation number (1, 2, 3, ...)
            genome: PromptGenome at this generation
        """
        self._generation_prompts[generation] = self._extract_prompts(genome)
        self._final_genome = genome

    def capture_evolved(self, genome: PromptGenome) -> None:
        """Capture final evolved prompts.

        Convenience method that sets both the final genome and adds it
        to generation history if not already present.

        Args:
            genome: Final evolved PromptGenome
        """
        self._final_genome = genome
        # Add to generation history at next available generation
        if self._generation_prompts:
            next_gen = max(self._generation_prompts.keys()) + 1
        else:
            next_gen = 1
        self._generation_prompts[next_gen] = self._extract_prompts(genome)

    def _extract_prompts(self, genome: PromptGenome) -> dict[str, str]:
        """Extract prompt texts from genome.

        Args:
            genome: PromptGenome to extract from

        Returns:
            Dict mapping role_id to prompt text
        """
        prompts: dict[str, str] = {}
        for role_id, role_prompt in genome.rolePrompts.items():
            # RolePrompt has id and content fields
            prompts[role_id] = role_prompt.content
        return prompts

    def compute_diffs(self) -> dict[str, PromptDiff]:
        """Compute diffs between initial and final prompts.

        Returns:
            Dict mapping role_id to PromptDiff
        """
        if not self._initial_prompts:
            return {}

        final_prompts = self._get_final_prompts()
        diffs: dict[str, PromptDiff] = {}

        # Get all roles from both initial and final
        all_roles = set(self._initial_prompts.keys()) | set(final_prompts.keys())

        for role_id in all_roles:
            old_text = self._initial_prompts.get(role_id, "")
            new_text = final_prompts.get(role_id, "")

            diff_lines = compute_prompt_diff(old_text, new_text)
            lines_added = sum(1 for d in diff_lines if d["type"] == "add")
            lines_removed = sum(1 for d in diff_lines if d["type"] == "remove")

            diffs[role_id] = PromptDiff(
                role_id=role_id,
                from_generation=0,
                to_generation=max(self._generation_prompts.keys()) if self._generation_prompts else 0,
                diff_lines=diff_lines,
                has_changes=has_changes(diff_lines),
                lines_added=lines_added,
                lines_removed=lines_removed,
            )

        return diffs

    def compute_generation_diffs(self) -> list[dict[str, PromptDiff]]:
        """Compute diffs between consecutive generations.

        Returns:
            List of diffs per generation transition (gen 0→1, 1→2, etc.)
        """
        if len(self._generation_prompts) < 2:
            return []

        generations = sorted(self._generation_prompts.keys())
        all_diffs: list[dict[str, PromptDiff]] = []

        for i in range(len(generations) - 1):
            from_gen = generations[i]
            to_gen = generations[i + 1]
            from_prompts = self._generation_prompts[from_gen]
            to_prompts = self._generation_prompts[to_gen]

            gen_diffs: dict[str, PromptDiff] = {}
            all_roles = set(from_prompts.keys()) | set(to_prompts.keys())

            for role_id in all_roles:
                old_text = from_prompts.get(role_id, "")
                new_text = to_prompts.get(role_id, "")

                diff_lines = compute_prompt_diff(old_text, new_text)
                lines_added = sum(1 for d in diff_lines if d["type"] == "add")
                lines_removed = sum(1 for d in diff_lines if d["type"] == "remove")

                gen_diffs[role_id] = PromptDiff(
                    role_id=role_id,
                    from_generation=from_gen,
                    to_generation=to_gen,
                    diff_lines=diff_lines,
                    has_changes=has_changes(diff_lines),
                    lines_added=lines_added,
                    lines_removed=lines_removed,
                )

            all_diffs.append(gen_diffs)

        return all_diffs

    def _get_final_prompts(self) -> dict[str, str]:
        """Get final prompts from most recent generation."""
        if not self._generation_prompts:
            return {}
        max_gen = max(self._generation_prompts.keys())
        return self._generation_prompts[max_gen]

    def summarize_changes(self) -> ChangeSummary:
        """Summarize all changes across evolution.

        Returns:
            ChangeSummary with aggregate statistics
        """
        diffs = self.compute_diffs()

        if not diffs:
            return ChangeSummary(
                total_roles=0,
                roles_changed=0,
                total_lines_added=0,
                total_lines_removed=0,
                most_changed_role=None,
                change_descriptions=[],
            )

        total_roles = len(diffs)
        roles_changed = sum(1 for d in diffs.values() if d.has_changes)
        total_added = sum(d.lines_added for d in diffs.values())
        total_removed = sum(d.lines_removed for d in diffs.values())

        # Find most changed role
        most_changed = max(
            diffs.values(),
            key=lambda d: d.lines_added + d.lines_removed,
            default=None,
        )

        # Generate descriptions
        descriptions: list[str] = []
        for role_id, diff in diffs.items():
            if diff.has_changes:
                descriptions.append(
                    f"{role_id}: +{diff.lines_added}/-{diff.lines_removed} lines"
                )

        return ChangeSummary(
            total_roles=total_roles,
            roles_changed=roles_changed,
            total_lines_added=total_added,
            total_lines_removed=total_removed,
            most_changed_role=most_changed.role_id if most_changed and most_changed.has_changes else None,
            change_descriptions=descriptions,
        )

    def get_initial_prompts(self) -> dict[str, str]:
        """Get initial prompts.

        Returns:
            Dict mapping role_id to initial prompt text
        """
        return self._initial_prompts.copy()

    def get_evolved_prompts(self) -> dict[str, str]:
        """Get final evolved prompts.

        Returns:
            Dict mapping role_id to evolved prompt text
        """
        return self._get_final_prompts()

    def get_generation_count(self) -> int:
        """Get number of generations tracked.

        Returns:
            Number of generations (including initial gen 0)
        """
        return len(self._generation_prompts)
