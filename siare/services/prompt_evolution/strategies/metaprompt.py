"""
MetaPrompt Strategy Implementation

LLM meta-analysis for targeted prompt improvements based on:
- Arize Prompt Learning SDK approach
- Meta-prompt patterns for self-improvement

Key features:
- Comprehensive context building from traces and feedback
- LLM identifies failing sections and proposes improvements
- Direct section-level updates respecting constraints
- Fast, single-shot improvement for quick fixes
"""

from typing import Any, Optional, cast

from siare.core.models import (
    Diagnosis,
    MetaPromptConfig,
    ParsedPrompt,
    ProcessConfig,
    PromptEvolutionResult,
    PromptFeedback,
    PromptGenome,
    PromptOptimizationStrategyType,
    RolePrompt,
)
from siare.services.llm_provider import LLMMessage, LLMProvider
from siare.services.prompt_evolution.strategies.base import BasePromptOptimizationStrategy


# Constants
TEXT_TRUNCATE_LENGTH = 200
VERSION_PARTS_MIN = 3
CHANGES_DISPLAY_LIMIT = 3


class MetaPromptStrategy(BasePromptOptimizationStrategy):
    """
    MetaPrompt optimization using LLM meta-analysis.

    Process:
    1. Build comprehensive context from feedback and diagnosis
    2. Ask LLM to analyze the prompt and identify improvements
    3. LLM proposes specific changes to sections
    4. Apply changes while respecting constraints

    Best for: Quick, targeted fixes when you need fast iteration.
    """

    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        config: Optional[MetaPromptConfig] = None,
    ):
        """
        Initialize MetaPrompt strategy.

        Args:
            llm_provider: LLM provider for meta-analysis
            config: Strategy configuration
        """
        self.llm_provider = llm_provider
        self.config = config or MetaPromptConfig()

    @property
    def name(self) -> str:
        return "metaprompt"

    @property
    def strategy_type(self) -> PromptOptimizationStrategyType:
        return PromptOptimizationStrategyType.METAPROMPT

    def requires_population(self) -> bool:
        return False

    def optimize(
        self,
        sop_config: ProcessConfig,  # noqa: ARG002
        prompt_genome: PromptGenome,
        feedback: list[PromptFeedback],
        diagnosis: Diagnosis,
        parsed_prompts: Optional[dict[str, ParsedPrompt]] = None,
        constraints: Optional[dict[str, Any]] = None,
    ) -> PromptEvolutionResult:
        """
        Apply LLM meta-analysis to evolve prompts.

        1. Build context from feedback and diagnosis
        2. Query LLM for improvements
        3. Apply suggested changes

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
        metadata = {
            "strategy": "metaprompt",
            "analysis_depth": self.config.analysis_depth,
            "improvement_count": self.config.improvement_count,
        }

        if not feedback:
            return PromptEvolutionResult(
                new_prompt_genome=prompt_genome,
                changes_made=[],
                rationale="No feedback provided for meta-analysis",
                strategy_metadata=metadata,
            )

        # Group feedback by role
        roles_to_improve = self._group_feedback_by_role(feedback)

        if not roles_to_improve:
            return PromptEvolutionResult(
                new_prompt_genome=prompt_genome,
                changes_made=[],
                rationale="No roles identified for improvement",
                strategy_metadata=metadata,
            )

        # Create new role prompts
        new_role_prompts = dict(prompt_genome.rolePrompts)

        for role_id, role_feedback in roles_to_improve.items():
            if role_id not in prompt_genome.rolePrompts:
                continue

            current_prompt = prompt_genome.rolePrompts[role_id]
            role_constraints: dict[str, Any] = constraints.get(role_id, {})  # type: ignore
            must_not_change: list[str] = role_constraints.get("mustNotChange", [])  # type: ignore
            parsed = parsed_prompts.get(role_id) if parsed_prompts else None

            # Get improvements from LLM or heuristics
            improvements = self._get_improvements(
                current_prompt=current_prompt,
                feedback=role_feedback,
                diagnosis=diagnosis,
                parsed_prompt=parsed,
                must_not_change=must_not_change,
            )

            if not improvements:
                continue

            # Apply improvements
            new_content = self._apply_improvements(
                current_content=current_prompt.content,
                improvements=improvements,
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
                continue

            # Record change
            if new_content != current_prompt.content:
                changes_made.append({
                    "role_id": role_id,
                    "section_id": "meta_improved",
                    "old": current_prompt.content[:TEXT_TRUNCATE_LENGTH] + "..."
                    if len(current_prompt.content) > TEXT_TRUNCATE_LENGTH
                    else current_prompt.content,
                    "new": new_content[:TEXT_TRUNCATE_LENGTH] + "..."
                    if len(new_content) > TEXT_TRUNCATE_LENGTH
                    else new_content,
                    "improvement_count": len(improvements),
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
                "metaprompt_applied": True,
            },
        )

        metadata["roles_improved"] = len(changes_made)

        rationale = self._build_rationale(changes_made, roles_to_improve)

        return PromptEvolutionResult(
            new_prompt_genome=new_genome,
            changes_made=changes_made,
            rationale=rationale,
            strategy_metadata=metadata,
        )

    def _group_feedback_by_role(
        self,
        feedback: list[PromptFeedback],
    ) -> dict[str, list[PromptFeedback]]:
        """Group feedback items by role_id."""
        grouped: dict[str, list[PromptFeedback]] = {}
        for fb in feedback:
            if fb.role_id not in grouped:
                grouped[fb.role_id] = []
            grouped[fb.role_id].append(fb)
        return grouped

    def _get_improvements(
        self,
        current_prompt: RolePrompt,
        feedback: list[PromptFeedback],
        diagnosis: Diagnosis,
        parsed_prompt: Optional[ParsedPrompt],
        must_not_change: list[str],
    ) -> list[dict[str, Any]]:
        """
        Get improvement suggestions from LLM or heuristics.

        Returns list of improvements:
        [{"target": "section_name", "change": "what to change", "reason": "why"}]
        """
        if self.llm_provider:
            return self._llm_get_improvements(
                current_prompt, feedback, diagnosis, must_not_change
            )
        return self._heuristic_get_improvements(
            current_prompt, feedback, diagnosis, parsed_prompt
        )

    def _llm_get_improvements(
        self,
        current_prompt: RolePrompt,
        feedback: list[PromptFeedback],
        diagnosis: Diagnosis,
        must_not_change: list[str],
    ) -> list[dict[str, Any]]:
        """Use LLM to analyze and suggest improvements."""
        protected_text = "\n".join(f"- {p}" for p in must_not_change) if must_not_change else "None"

        feedback_summary = ""
        for i, fb in enumerate(feedback[:5], 1):
            pattern = fb.failure_pattern.value if fb.failure_pattern else "issue"
            feedback_summary += f"\n{i}. [{pattern}] {fb.critique[:150]}"
            if fb.suggested_improvement:
                feedback_summary += f"\n   Suggestion: {fb.suggested_improvement[:100]}"

        diagnosis_context = f"""
Primary Weakness: {diagnosis.primaryWeakness}
Root Cause: {diagnosis.rootCauseAnalysis}
Recommendations: {', '.join(diagnosis.recommendations[:3])}
"""

        analysis_instruction = (
            "Provide a thorough analysis with detailed improvements"
            if self.config.analysis_depth == "detailed"
            else "Focus on the most critical improvements"
        )

        messages = [
            LLMMessage(
                role="system",
                content=f"""You are an expert prompt engineer analyzing a prompt for improvements.

Your task:
1. Review the prompt and feedback carefully
2. Identify the {self.config.improvement_count} most important improvements
3. Provide specific, actionable changes

{analysis_instruction}

Output format (JSON array):
[
  {{"target": "section_name", "change": "specific change to make", "reason": "why this helps"}}
]

Rules:
- Protected text must not be removed or modified
- Focus on the highest-impact improvements
- Be specific about what to change""",
            ),
            LLMMessage(
                role="user",
                content=f"""Analyze this prompt and suggest improvements:

CURRENT PROMPT:
{current_prompt.content}

FEEDBACK RECEIVED:
{feedback_summary}

DIAGNOSIS:
{diagnosis_context}

PROTECTED TEXT (must not change):
{protected_text}

Provide {self.config.improvement_count} improvements as JSON:""",
            ),
        ]

        try:
            if self.llm_provider is not None:
                response = self.llm_provider.complete(
                    messages=messages,
                    model=self.config.model,
                    temperature=0.3,  # Lower for consistency
                    max_tokens=1500,
                )

                # Parse JSON response
                return self._parse_improvements_response(response.content)
        except Exception:  # noqa: BLE001
            pass
        # Fallback to heuristics
        return self._heuristic_get_improvements(
            current_prompt, feedback, diagnosis, None
        )

    def _parse_improvements_response(self, response: str) -> list[dict[str, Any]]:
        """Parse LLM response into list of improvements."""
        import json
        import re

        # Try to extract JSON from response
        try:
            # Look for JSON array
            json_match = re.search(r"\[[\s\S]*\]", response)
            if json_match:
                parsed = json.loads(json_match.group())
                # Validate structure: must be list of dicts with expected fields
                if isinstance(parsed, list):
                    valid_improvements: list[dict[str, str]] = []
                    parsed_list = cast("list[Any]", parsed)
                    for item in parsed_list:
                        if not isinstance(item, dict):
                            continue
                        # Ensure required fields exist with defaults
                        item_dict = cast("dict[str, Any]", item)
                        target = str(item_dict.get("target", "general"))
                        change = str(item_dict.get("change") or item_dict.get("suggestion", ""))
                        reason = str(item_dict.get("reason") or item_dict.get("rationale", ""))
                        improvement: dict[str, str] = {
                            "target": target,
                            "change": change,
                            "reason": reason,
                        }
                        # Only include if there's actual change content
                        if improvement["change"]:
                            valid_improvements.append(improvement)
                    if valid_improvements:
                        return valid_improvements
        except (json.JSONDecodeError, TypeError):
            pass

        # Fallback: create single improvement from text
        return [{
            "target": "general",
            "change": response[:500],
            "reason": "LLM meta-analysis suggestion"
        }]

    def _heuristic_get_improvements(
        self,
        current_prompt: RolePrompt,  # noqa: ARG002
        feedback: list[PromptFeedback],
        diagnosis: Diagnosis,
        parsed_prompt: Optional[ParsedPrompt],  # noqa: ARG002
    ) -> list[dict[str, Any]]:
        """Generate improvements using heuristic rules."""
        improvements: list[dict[str, Any]] = []

        # Sort feedback by confidence
        sorted_feedback = sorted(feedback, key=lambda f: f.confidence, reverse=True)

        for fb in sorted_feedback[: self.config.improvement_count]:
            improvement = {
                "target": fb.section_id or "general",
                "reason": fb.critique[:200],
            }

            if fb.suggested_improvement:
                improvement["change"] = fb.suggested_improvement
            elif fb.failure_pattern:
                # Pattern-based improvements
                pattern_changes = {
                    "hallucination": "Add explicit verification step: 'Verify all claims against source material'",
                    "incomplete": "Add completeness check: 'Ensure all required elements are addressed'",
                    "irrelevant": "Add relevance filter: 'Focus only on directly relevant information'",
                    "timeout": "Add efficiency note: 'Prioritize key information, be concise'",
                    "tool_misuse": "Clarify tool usage: 'Use tools appropriately for their intended purpose'",
                    "format_error": "Add format specification: 'Follow the output format exactly'",
                    "reasoning_error": "Add reasoning structure: 'Show step-by-step reasoning'",
                    "context_loss": "Add context preservation: 'Maintain context from previous steps'",
                    "safety_violation": "Reinforce safety: 'Adhere to all safety guidelines'",
                }
                improvement["change"] = pattern_changes.get(
                    fb.failure_pattern.value,
                    f"Address: {fb.critique[:100]}"
                )
            else:
                improvement["change"] = f"Improve based on: {fb.critique[:100]}"

            improvements.append(improvement)

        # Add diagnosis-based improvement if space
        if len(improvements) < self.config.improvement_count:
            improvements.append({
                "target": "general",
                "change": f"Address primary weakness: {diagnosis.primaryWeakness[:150]}",
                "reason": diagnosis.rootCauseAnalysis[:150],
            })

        return improvements

    def _apply_improvements(
        self,
        current_content: str,
        improvements: list[dict[str, Any]],
        parsed_prompt: Optional[ParsedPrompt],
        must_not_change: list[str],
    ) -> str:
        """Apply improvements to produce new content."""
        if self.llm_provider:
            return self._llm_apply_improvements(
                current_content, improvements, must_not_change
            )
        return self._heuristic_apply_improvements(
            current_content, improvements, parsed_prompt, must_not_change
        )

    def _llm_apply_improvements(
        self,
        content: str,
        improvements: list[dict[str, Any]],
        must_not_change: list[str],
    ) -> str:
        """Use LLM to apply improvements."""
        protected_text = "\n".join(f"- {p}" for p in must_not_change) if must_not_change else "None"

        improvements_text = ""
        for i, imp in enumerate(improvements, 1):
            improvements_text += f"\n{i}. Target: {imp.get('target', 'general')}"
            improvements_text += f"\n   Change: {imp.get('change', '')}"
            improvements_text += f"\n   Reason: {imp.get('reason', '')}"

        messages = [
            LLMMessage(
                role="system",
                content="""You are an expert prompt engineer applying improvements to a prompt.

Your task:
1. Apply the suggested improvements to the prompt
2. Preserve all protected text exactly
3. Maintain the prompt's structure and intent
4. Make targeted, surgical changes

Output ONLY the improved prompt, no explanations.""",
            ),
            LLMMessage(
                role="user",
                content=f"""Apply these improvements to the prompt:

CURRENT PROMPT:
{content}

IMPROVEMENTS TO APPLY:
{improvements_text}

PROTECTED TEXT (must keep exactly):
{protected_text}

Output only the improved prompt:""",
            ),
        ]

        try:
            if self.llm_provider is not None:
                response = self.llm_provider.complete(
                    messages=messages,
                    model=self.config.model,
                    temperature=0.3,
                    max_tokens=2000,
                )
                return response.content.strip()
            return content
        except Exception:  # noqa: BLE001
            return self._heuristic_apply_improvements(
                content, improvements, None, must_not_change
            )

    def _heuristic_apply_improvements(
        self,
        content: str,
        improvements: list[dict[str, Any]],
        parsed_prompt: Optional[ParsedPrompt],  # noqa: ARG002
        must_not_change: list[str],
    ) -> str:
        """Apply improvements using heuristic rules."""
        lines = content.split("\n")

        # Find a good insertion point (before constraints/immutable sections)
        insertion_point = len(lines)
        for i, line in enumerate(lines):
            if any(kw in line.lower() for kw in ["constraint", "immutable", "prohibited", "safety"]):
                insertion_point = i
                break

        # Build improvement section
        improvement_lines = ["\n### Applied Improvements"]
        for imp in improvements[:3]:  # Limit to 3
            change = imp.get("change", "")
            # Check not in protected text
            if not any(p in change for p in must_not_change):
                improvement_lines.append(f"- {change}")

        if len(improvement_lines) > 1:  # Has actual improvements
            # Insert improvements
            for i, line in enumerate(improvement_lines):
                lines.insert(insertion_point + i, line)

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
        roles_to_improve: dict[str, list[PromptFeedback]],
    ) -> str:
        """Build human-readable rationale."""
        if not changes_made:
            total_feedback = sum(len(v) for v in roles_to_improve.values())
            return f"No changes made despite {total_feedback} feedback item(s)"

        rationale_parts = [
            f"MetaPrompt: Applied improvements to {len(changes_made)} role(s)"
        ]

        for change in changes_made[:CHANGES_DISPLAY_LIMIT]:
            imp_count = change.get("improvement_count", 0)
            rationale_parts.append(
                f"- {change['role_id']}: {imp_count} improvement(s) applied"
            )

        if len(changes_made) > CHANGES_DISPLAY_LIMIT:
            rationale_parts.append(f"- ... and {len(changes_made) - CHANGES_DISPLAY_LIMIT} more")

        return "\n".join(rationale_parts)
