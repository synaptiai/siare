"""
LLM Critic for Prompt Evolution

Extracts structured feedback from execution traces using LLM analysis.
Classifies failure patterns and generates actionable improvement suggestions.
"""

import contextlib
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

from siare.core.models import (
    EvaluationArtifacts,
    EvaluationVector,
    FailurePattern,
    ProcessConfig,
    PromptFeedback,
    PromptGenome,
    PromptSection,
    PromptSectionType,
)
from siare.services.execution_engine import ExecutionTrace
from siare.services.prompt_evolution.parser import MarkdownSectionParser

if TYPE_CHECKING:
    from siare.services.llm_provider import LLMProvider


logger = logging.getLogger(__name__)

# Configuration constants
POOR_PERFORMANCE_THRESHOLD = 0.5
PROMPT_EXCERPT_MAX_LENGTH = 1000
IO_MAX_LENGTH = 500


# Prompts for LLM-based failure classification
FAILURE_CLASSIFICATION_PROMPT = """Analyze the following execution trace and classify the failure pattern.

## Role Information
Role ID: {role_id}
Role Prompt (excerpt):
```
{prompt_excerpt}
```

## Execution Data
Input:
{inputs}

Output:
{outputs}

Error (if any): {error}

## Available Failure Patterns
- HALLUCINATION: Made up information not in the source data
- INCOMPLETE: Missing required information or partial response
- IRRELEVANT: Off-topic response that doesn't address the task
- TIMEOUT: Execution exceeded time limits
- TOOL_MISUSE: Incorrect tool usage or wrong tool selection
- FORMAT_ERROR: Response format doesn't match expected structure
- REASONING_ERROR: Logical errors or incorrect conclusions
- CONTEXT_LOSS: Lost track of context or earlier information
- SAFETY_VIOLATION: Safety policy breach

## Task
1. Identify the primary failure pattern (one of the above)
2. Explain why this pattern applies
3. Suggest specific improvements to the prompt

Respond in this format:
PATTERN: <pattern_name>
CONFIDENCE: <0.0-1.0>
REASONING: <explanation>
SUGGESTION: <specific prompt improvement>
"""

SECTION_FEEDBACK_PROMPT = """Given the failure analysis and the prompt section, generate targeted feedback.

## Failure Analysis
Pattern: {failure_pattern}
Reasoning: {failure_reasoning}

## Prompt Section
Section Type: {section_type}
Section Content:
```
{section_content}
```

## Task
Analyze how this section may have contributed to the failure and suggest specific improvements.

Respond in this format:
CRITIQUE: <detailed critique of what went wrong in this section>
IMPROVEMENT: <specific rewritten content for this section>
"""


class BaseLLMCritic(ABC):
    """Abstract base class for LLM-based feedback extraction"""

    @abstractmethod
    def extract_feedback(
        self,
        traces: list[ExecutionTrace],
        sop_config: ProcessConfig,
        prompt_genome: PromptGenome,
        evaluations: list[EvaluationVector] | None = None,
        artifacts: EvaluationArtifacts | None = None,
    ) -> list[PromptFeedback]:
        """
        Extract structured feedback from execution traces.

        Args:
            traces: Execution traces with node_executions, tool_calls, errors
            sop_config: SOP configuration for context
            prompt_genome: Current prompts for each role
            evaluations: Optional evaluation vectors with metric scores
            artifacts: Optional evaluation artifacts with existing feedback

        Returns:
            List of PromptFeedback with role-level attribution
        """


class TraceFeedbackExtractor(BaseLLMCritic):
    """
    Extracts structured feedback from execution traces using LLM analysis.

    Pipeline:
    1. Parse traces to identify failing/poor-performing nodes
    2. Correlate failures with role prompts
    3. Use LLM to classify failure patterns
    4. Generate actionable suggestions per prompt section
    """

    def __init__(
        self,
        llm_provider: Optional["LLMProvider"] = None,
        model: str = "gpt-4",
        temperature: float = 0.3,
    ):
        """
        Initialize the feedback extractor.

        Args:
            llm_provider: LLM provider for classification (optional for rule-based)
            model: Model to use for classification
            temperature: LLM temperature (lower = more deterministic)
        """
        self.llm_provider: LLMProvider | None = llm_provider
        self.model = model
        self.temperature = temperature
        self.parser = MarkdownSectionParser()

    def extract_feedback(
        self,
        traces: list[ExecutionTrace],
        sop_config: ProcessConfig,
        prompt_genome: PromptGenome,
        evaluations: list[EvaluationVector] | None = None,
        artifacts: EvaluationArtifacts | None = None,
    ) -> list[PromptFeedback]:
        """
        Extract structured feedback from execution traces.

        Args:
            traces: Execution traces
            sop_config: SOP configuration
            prompt_genome: Current prompts
            evaluations: Optional evaluation vectors
            artifacts: Optional evaluation artifacts

        Returns:
            List of PromptFeedback objects
        """
        feedback_list: list[PromptFeedback] = []

        # Step 1: Identify failing/poor-performing nodes
        failing_nodes = self._identify_failing_nodes(traces, evaluations)

        # Step 2: For each failing node, analyze and classify
        for node_info in failing_nodes:
            role_id = node_info["role_id"]

            # Get prompt for this role
            role_prompt = self._get_role_prompt(role_id, sop_config, prompt_genome)
            if not role_prompt:
                logger.warning(f"No prompt found for role {role_id}")
                continue

            # Parse prompt into sections
            parsed_prompt = self.parser.parse(role_prompt, role_id)

            # Classify failure pattern
            failure_info = self._classify_failure(node_info, role_prompt.content)

            # Generate feedback for mutable sections
            for section in parsed_prompt.sections:
                if section.is_mutable:
                    section_feedback = self._generate_section_feedback(
                        section=section,
                        failure_info=failure_info,
                        node_info=node_info,
                    )
                    feedback_list.append(section_feedback)

        # Step 3: Incorporate existing artifacts if available
        if artifacts:
            feedback_list.extend(
                self._incorporate_existing_artifacts(artifacts, prompt_genome, sop_config)
            )

        # Step 4: Aggregate and deduplicate
        return self._aggregate_feedback(feedback_list)

    def _identify_failing_nodes(
        self,
        traces: list[ExecutionTrace],
        evaluations: list[EvaluationVector] | None,
    ) -> list[dict[str, Any]]:
        """
        Identify nodes that failed or performed poorly.

        Args:
            traces: Execution traces
            evaluations: Optional evaluation vectors

        Returns:
            List of node info dicts with role_id, inputs, outputs, errors
        """
        failing_nodes: list[dict[str, Any]] = []

        for trace in traces:
            # Check explicit errors
            for error in trace.errors:
                failing_nodes.append({
                    "role_id": error["role_id"],
                    "error": error["error"],
                    "inputs": self._find_node_inputs(trace, error["role_id"]),
                    "outputs": self._find_node_outputs(trace, error["role_id"]),
                    "trace_id": trace.run_id,
                    "failure_source": "explicit_error",
                })

            # If we have evaluations, check for low scores
            if evaluations:
                eval_for_trace = next(
                    (e for e in evaluations if e.runId == trace.run_id), None
                )
                if eval_for_trace:
                    # Check if average score is below threshold
                    avg_score = sum(m.score for m in eval_for_trace.metrics) / len(
                        eval_for_trace.metrics
                    )
                    if avg_score < POOR_PERFORMANCE_THRESHOLD:
                        # Attribute to all nodes (we don't know which specifically failed)
                        for node_exec in trace.node_executions:
                            if node_exec["role_id"] not in [
                                n["role_id"] for n in failing_nodes
                            ]:
                                failing_nodes.append({
                                    "role_id": node_exec["role_id"],
                                    "error": None,
                                    "inputs": node_exec.get("inputs", {}),
                                    "outputs": node_exec.get("outputs", {}),
                                    "trace_id": trace.run_id,
                                    "failure_source": "low_evaluation_score",
                                    "eval_score": avg_score,
                                })

        return failing_nodes

    def _find_node_inputs(
        self, trace: ExecutionTrace, role_id: str
    ) -> dict[str, Any]:
        """Find inputs for a specific role in trace"""
        for node_exec in trace.node_executions:
            if node_exec["role_id"] == role_id:
                return node_exec.get("inputs", {})
        return {}

    def _find_node_outputs(
        self, trace: ExecutionTrace, role_id: str
    ) -> dict[str, Any]:
        """Find outputs for a specific role in trace"""
        for node_exec in trace.node_executions:
            if node_exec["role_id"] == role_id:
                return node_exec.get("outputs", {})
        return {}

    def _get_role_prompt(
        self,
        role_id: str,
        sop_config: ProcessConfig,
        prompt_genome: PromptGenome,
    ):
        """
        Get the RolePrompt for a specific role.

        Args:
            role_id: Role ID
            sop_config: SOP config with role definitions
            prompt_genome: Prompt genome with role prompts

        Returns:
            RolePrompt or None if not found
        """
        # Find the role config
        role_config = next(
            (r for r in sop_config.roles if r.id == role_id), None
        )
        if not role_config:
            return None

        # Get prompt from genome
        prompt_ref = role_config.promptRef
        return prompt_genome.rolePrompts.get(prompt_ref)

    def _classify_failure(
        self,
        node_info: dict[str, Any],
        prompt_content: str,
    ) -> dict[str, Any]:
        """
        Classify the failure pattern using LLM or heuristics.

        Args:
            node_info: Node execution info
            prompt_content: Full prompt content

        Returns:
            Dict with pattern, confidence, reasoning, suggestion
        """
        # If no LLM provider, use heuristic classification
        if not self.llm_provider:
            return self._heuristic_classify_failure(node_info)

        # Use LLM for classification
        return self._llm_classify_failure(node_info, prompt_content)

    def _heuristic_classify_failure(
        self, node_info: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Classify failure using rule-based heuristics.

        Args:
            node_info: Node execution info

        Returns:
            Classification result
        """
        error = node_info.get("error", "") or ""
        error_lower = error.lower()
        outputs = node_info.get("outputs", {})
        output_str = str(outputs).lower() if outputs else ""

        # Pattern detection heuristics
        if "timeout" in error_lower or "time" in error_lower:
            return {
                "pattern": FailurePattern.TIMEOUT,
                "confidence": 0.9,
                "reasoning": "Execution exceeded time limits",
                "suggestion": "Add time management instructions to the prompt",
            }

        if "tool" in error_lower or "api" in error_lower:
            return {
                "pattern": FailurePattern.TOOL_MISUSE,
                "confidence": 0.8,
                "reasoning": "Tool invocation error detected",
                "suggestion": "Add clearer tool usage guidelines",
            }

        if "format" in error_lower or "parse" in error_lower or "json" in error_lower:
            return {
                "pattern": FailurePattern.FORMAT_ERROR,
                "confidence": 0.85,
                "reasoning": "Output format doesn't match expected structure",
                "suggestion": "Add explicit output format examples",
            }

        if "safety" in error_lower or "policy" in error_lower or "refuse" in error_lower:
            return {
                "pattern": FailurePattern.SAFETY_VIOLATION,
                "confidence": 0.9,
                "reasoning": "Safety policy triggered",
                "suggestion": "Review and clarify safety guidelines",
            }

        # Check output content for hallucination indicators
        if not outputs or output_str in ["none", "{}", "null", ""]:
            return {
                "pattern": FailurePattern.INCOMPLETE,
                "confidence": 0.7,
                "reasoning": "Empty or missing output",
                "suggestion": "Add completeness requirements to instructions",
            }

        # Default to reasoning error for low evaluation scores
        if node_info.get("failure_source") == "low_evaluation_score":
            return {
                "pattern": FailurePattern.REASONING_ERROR,
                "confidence": 0.5,
                "reasoning": "Low evaluation score without explicit error",
                "suggestion": "Review reasoning steps and add verification",
            }

        # Fallback
        return {
            "pattern": FailurePattern.INCOMPLETE,
            "confidence": 0.4,
            "reasoning": "Unable to determine specific failure pattern",
            "suggestion": "Review and clarify prompt instructions",
        }

    def _llm_classify_failure(
        self,
        node_info: dict[str, Any],
        prompt_content: str,
    ) -> dict[str, Any]:
        """
        Classify failure using LLM analysis.

        Args:
            node_info: Node execution info
            prompt_content: Full prompt content

        Returns:
            Classification result
        """
        # Build classification prompt
        prompt = FAILURE_CLASSIFICATION_PROMPT.format(
            role_id=node_info["role_id"],
            prompt_excerpt=prompt_content[:PROMPT_EXCERPT_MAX_LENGTH],
            inputs=str(node_info.get("inputs", {}))[:IO_MAX_LENGTH],
            outputs=str(node_info.get("outputs", {}))[:IO_MAX_LENGTH],
            error=node_info.get("error", "None"),
        )

        try:
            # Call LLM
            from siare.services.llm_provider import LLMMessage
            response = self.llm_provider.complete(  # type: ignore[union-attr]
                model=self.model,
                messages=[LLMMessage(role="user", content=prompt)],  # type: ignore[misc]
                temperature=self.temperature,
            )

            # Parse response
            return self._parse_classification_response(response.content)

        except Exception as e:
            logger.warning(f"LLM classification failed: {e}, falling back to heuristics")
            return self._heuristic_classify_failure(node_info)

    def _parse_classification_response(self, response: str) -> dict[str, Any]:
        """Parse LLM classification response into structured dict"""
        lines = response.strip().split("\n")
        result = {
            "pattern": FailurePattern.INCOMPLETE,
            "confidence": 0.5,
            "reasoning": "",
            "suggestion": "",
        }

        for line in lines:
            if line.startswith("PATTERN:"):
                pattern_str = line.replace("PATTERN:", "").strip().upper()
                with contextlib.suppress(ValueError):
                    result["pattern"] = FailurePattern(pattern_str.lower())
            elif line.startswith("CONFIDENCE:"):
                with contextlib.suppress(ValueError):
                    result["confidence"] = float(
                        line.replace("CONFIDENCE:", "").strip()
                    )
            elif line.startswith("REASONING:"):
                result["reasoning"] = line.replace("REASONING:", "").strip()
            elif line.startswith("SUGGESTION:"):
                result["suggestion"] = line.replace("SUGGESTION:", "").strip()

        return result

    def _generate_section_feedback(
        self,
        section: PromptSection,
        failure_info: dict[str, Any],
        node_info: dict[str, Any],  # noqa: ARG002
    ) -> PromptFeedback:
        """
        Generate feedback for a specific prompt section.

        Args:
            section: PromptSection to generate feedback for
            failure_info: Classified failure information
            node_info: Node execution info

        Returns:
            PromptFeedback object
        """
        # Build critique based on failure pattern and section type
        critique = self._build_section_critique(section, failure_info)
        suggestion = self._build_section_suggestion(section, failure_info)

        return PromptFeedback(
            role_id=section.parent_role_id,
            section_id=section.id,
            section_content=section.content,
            critique=critique,
            failure_pattern=failure_info.get("pattern"),
            suggested_improvement=suggestion,
            confidence=failure_info.get("confidence", 0.5),
        )

    def _build_section_critique(
        self,
        section: PromptSection,
        failure_info: dict[str, Any],
    ) -> str:
        """Build critique for a section based on failure pattern"""
        pattern = failure_info.get("pattern", FailurePattern.INCOMPLETE)
        reasoning = failure_info.get("reasoning", "")

        section_type_critiques = {
            PromptSectionType.INSTRUCTIONS: {
                FailurePattern.HALLUCINATION: "Instructions may not sufficiently constrain the model to use only provided information.",
                FailurePattern.INCOMPLETE: "Instructions may lack explicit completeness requirements.",
                FailurePattern.REASONING_ERROR: "Instructions may not provide clear reasoning steps.",
                FailurePattern.FORMAT_ERROR: "Instructions may not specify expected output format.",
            },
            PromptSectionType.EXAMPLES: {
                FailurePattern.FORMAT_ERROR: "Examples may not demonstrate the expected output format clearly.",
                FailurePattern.INCOMPLETE: "Examples may not show complete responses.",
            },
            PromptSectionType.OBJECTIVE: {
                FailurePattern.IRRELEVANT: "Objective may not clearly define the task scope.",
                FailurePattern.INCOMPLETE: "Objective may not specify all required outputs.",
            },
        }

        # Get section-specific critique
        type_critiques = section_type_critiques.get(section.section_type, {})
        specific_critique = type_critiques.get(
            pattern, f"Section may contribute to {pattern.value} issues."
        )

        return f"{specific_critique} {reasoning}".strip()

    def _build_section_suggestion(
        self,
        section: PromptSection,  # noqa: ARG002
        failure_info: dict[str, Any],
    ) -> str:
        """Build improvement suggestion for a section"""
        pattern = failure_info.get("pattern", FailurePattern.INCOMPLETE)
        general_suggestion = failure_info.get("suggestion", "")

        pattern_suggestions = {
            FailurePattern.HALLUCINATION: "Add explicit instruction: 'Only use information from the provided sources. Do not make up facts.'",
            FailurePattern.INCOMPLETE: "Add: 'Ensure your response addresses all required points. Check for completeness before responding.'",
            FailurePattern.IRRELEVANT: "Add: 'Stay focused on the specific task. Do not provide tangential information.'",
            FailurePattern.FORMAT_ERROR: "Add explicit format specification with example output structure.",
            FailurePattern.REASONING_ERROR: "Add: 'Think step by step. Verify each conclusion before proceeding.'",
            FailurePattern.TOOL_MISUSE: "Add clear tool usage guidelines with when to use each tool.",
            FailurePattern.CONTEXT_LOSS: "Add: 'Refer back to the original context throughout your response.'",
        }

        specific_suggestion = pattern_suggestions.get(pattern, general_suggestion)
        return specific_suggestion or general_suggestion

    def _incorporate_existing_artifacts(
        self,
        artifacts: EvaluationArtifacts,
        prompt_genome: PromptGenome,
        sop_config: ProcessConfig,
    ) -> list[PromptFeedback]:
        """
        Incorporate existing evaluation artifacts into feedback.

        Args:
            artifacts: Existing evaluation artifacts
            prompt_genome: Prompt genome
            sop_config: SOP config

        Returns:
            Additional feedback from artifacts
        """
        additional_feedback: list[PromptFeedback] = []

        # Process LLM feedback from artifacts
        if artifacts.llmFeedback:
            for metric_id, critique in artifacts.llmFeedback.items():
                # Create generic feedback for all roles
                for role_config in sop_config.roles:
                    role_prompt = prompt_genome.rolePrompts.get(role_config.promptRef)
                    if role_prompt:
                        parsed = self.parser.parse(role_prompt, role_config.id)
                        for section in parsed.sections:
                            if section.is_mutable:
                                additional_feedback.append(
                                    PromptFeedback(
                                        role_id=role_config.id,
                                        section_id=section.id,
                                        section_content=section.content,
                                        critique=f"[{metric_id}] {critique}",
                                        failure_pattern=None,  # Unknown from artifacts
                                        suggested_improvement=None,
                                        confidence=0.6,
                                    )
                                )

        # Process failure modes from artifacts
        if artifacts.failureModes:
            for failure_mode in artifacts.failureModes:
                # Map string to FailurePattern if possible
                pattern = self._map_failure_mode_string(failure_mode)
                # Add to feedback (generic, applies to all roles)
                for role_config in sop_config.roles:
                    role_prompt = prompt_genome.rolePrompts.get(role_config.promptRef)
                    if role_prompt:
                        parsed = self.parser.parse(role_prompt, role_config.id)
                        for section in parsed.sections:
                            if section.is_mutable:
                                additional_feedback.append(
                                    PromptFeedback(
                                        role_id=role_config.id,
                                        section_id=section.id,
                                        section_content=section.content,
                                        critique=f"Failure mode detected: {failure_mode}",
                                        failure_pattern=pattern,
                                        suggested_improvement=None,
                                        confidence=0.5,
                                    )
                                )

        return additional_feedback

    def _map_failure_mode_string(self, failure_mode: str) -> FailurePattern | None:
        """Map a failure mode string to FailurePattern enum"""
        failure_lower = failure_mode.lower()

        mapping = {
            "hallucination": FailurePattern.HALLUCINATION,
            "incomplete": FailurePattern.INCOMPLETE,
            "irrelevant": FailurePattern.IRRELEVANT,
            "timeout": FailurePattern.TIMEOUT,
            "tool": FailurePattern.TOOL_MISUSE,
            "format": FailurePattern.FORMAT_ERROR,
            "reasoning": FailurePattern.REASONING_ERROR,
            "context": FailurePattern.CONTEXT_LOSS,
            "safety": FailurePattern.SAFETY_VIOLATION,
        }

        for key, pattern in mapping.items():
            if key in failure_lower:
                return pattern

        return None

    def _aggregate_feedback(
        self, feedback_list: list[PromptFeedback]
    ) -> list[PromptFeedback]:
        """
        Aggregate and deduplicate feedback.

        Combines feedback for the same section, keeping highest confidence.

        Args:
            feedback_list: Raw feedback list

        Returns:
            Aggregated feedback list
        """
        # Group by (role_id, section_id)
        grouped: dict[tuple[str, str], list[PromptFeedback]] = {}

        for feedback in feedback_list:
            key = (feedback.role_id, feedback.section_id)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(feedback)

        # Keep highest confidence feedback per section
        aggregated: list[PromptFeedback] = []

        for _key, group in grouped.items():
            if len(group) == 1:
                aggregated.append(group[0])
            else:
                # Sort by confidence and keep highest
                sorted_group = sorted(
                    group, key=lambda f: f.confidence, reverse=True
                )
                best = sorted_group[0]

                # Combine critiques from top 3
                combined_critique = "\n".join(
                    f.critique for f in sorted_group[:3] if f.critique
                )

                aggregated.append(
                    PromptFeedback(
                        role_id=best.role_id,
                        section_id=best.section_id,
                        section_content=best.section_content,
                        critique=combined_critique or best.critique,
                        failure_pattern=best.failure_pattern,
                        suggested_improvement=best.suggested_improvement,
                        confidence=best.confidence,
                    )
                )

        return aggregated
