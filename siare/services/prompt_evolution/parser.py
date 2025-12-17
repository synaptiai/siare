"""
Prompt Section Parser

Parses prompts into structured sections for surgical mutations.
Identifies mutable vs immutable sections based on keywords and constraints.
"""

import json
import logging
import re
import uuid
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, Optional

from siare.core.models import (
    ParsedPrompt,
    PromptSection,
    PromptSectionType,
    RolePrompt,
)


logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from siare.services.llm_provider import LLMProvider


class BasePromptSectionParser(ABC):
    """Abstract base class for prompt section parsers"""

    @abstractmethod
    def parse(self, prompt: RolePrompt, role_id: str) -> ParsedPrompt:
        """
        Parse a prompt into sections.

        Args:
            prompt: RolePrompt to parse
            role_id: ID of the role this prompt belongs to

        Returns:
            ParsedPrompt with sections
        """

    @abstractmethod
    def reconstruct(
        self,
        parsed: ParsedPrompt,
        section_updates: Optional[dict[str, str]] = None,
    ) -> str:
        """
        Reconstruct prompt from sections with optional updates.

        Args:
            parsed: ParsedPrompt with sections
            section_updates: Optional dict of section_id -> new_content

        Returns:
            Reconstructed prompt string
        """


class MarkdownSectionParser(BasePromptSectionParser):
    """
    Parses prompts using markdown headers as section delimiters.

    Expected format:
    ```
    # Role
    You are a research analyst...

    ## Instructions
    1. First, analyze...

    ## Constraints [IMMUTABLE]
    - Never fabricate citations

    ## Examples
    Input: ...
    ```

    Sections are identified by markdown headers (# to ######).
    Sections marked with [IMMUTABLE] or containing safety keywords are protected.
    """

    # Keywords that indicate immutable sections (case-insensitive)
    IMMUTABLE_KEYWORDS = frozenset(
        [
            "safety",
            "policy",
            "regulation",
            "compliance",
            "must not",
            "prohibited",
            "required",
            "constraint",
            "restriction",
            "never",
            "always",
            "mandatory",
        ]
    )

    # Mapping of header keywords to section types
    SECTION_TYPE_KEYWORDS: ClassVar[dict[PromptSectionType, list[str]]] = {
        PromptSectionType.ROLE_DEFINITION: ["role", "persona", "identity", "you are"],
        PromptSectionType.OBJECTIVE: ["objective", "goal", "task", "purpose", "mission"],
        PromptSectionType.INSTRUCTIONS: [
            "instruction",
            "step",
            "procedure",
            "how to",
            "process",
        ],
        PromptSectionType.CONSTRAINTS: [
            "constraint",
            "rule",
            "policy",
            "restriction",
            "limit",
            "safety",
        ],
        PromptSectionType.EXAMPLES: ["example", "sample", "demonstration", "few-shot"],
        PromptSectionType.FORMAT: ["format", "output", "response", "structure"],
        PromptSectionType.CONTEXT: ["context", "background", "information", "knowledge"],
    }

    def __init__(self):
        """Initialize parser with compiled regex patterns"""
        # Pattern for markdown headers (# to ######)
        self._header_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
        # Pattern for [IMMUTABLE] marker
        self._immutable_marker_pattern = re.compile(r"\[IMMUTABLE\]", re.IGNORECASE)

    def parse(self, prompt: RolePrompt, role_id: str) -> ParsedPrompt:
        """
        Parse a prompt into sections based on markdown headers.

        Args:
            prompt: RolePrompt to parse
            role_id: ID of the role this prompt belongs to

        Returns:
            ParsedPrompt with sections
        """
        content = prompt.content
        constraints = prompt.constraints

        # Extract mustNotChange constraints
        immutable_constraints = []
        if constraints and constraints.mustNotChange:
            immutable_constraints = constraints.mustNotChange

        # Find all headers
        headers = list(self._header_pattern.finditer(content))

        if not headers:
            # No headers found - treat entire content as single fallback section
            section = PromptSection(
                id=str(uuid.uuid4()),
                content=content.strip(),
                is_mutable=self._is_content_mutable(content, immutable_constraints),
                section_type=PromptSectionType.FALLBACK,
                parent_role_id=role_id,
            )
            return ParsedPrompt(
                role_id=role_id,
                original_content=content,
                sections=[section],
                immutable_constraints=immutable_constraints,
            )

        sections: list[PromptSection] = []

        for i, match in enumerate(headers):
            header_level = len(match.group(1))
            header_text = match.group(2).strip()
            header_end = match.end()

            # Find section content (until next header or end of content)
            section_end = headers[i + 1].start() if i + 1 < len(headers) else len(content)

            section_content = content[header_end:section_end].strip()

            # Determine section type
            section_type = self._infer_section_type(header_text)

            # Check if section is mutable
            is_mutable = self._is_section_mutable(
                header_text=header_text,
                section_content=section_content,
                constraints=immutable_constraints,
            )

            section = PromptSection(
                id=str(uuid.uuid4()),
                content=section_content,
                is_mutable=is_mutable,
                section_type=section_type,
                parent_role_id=role_id,
            )
            sections.append(section)

            # Store header for reconstruction
            section._header_text = header_text  # type: ignore
            section._header_level = header_level  # type: ignore

        return ParsedPrompt(
            role_id=role_id,
            original_content=content,
            sections=sections,
            immutable_constraints=immutable_constraints,
        )

    def reconstruct(
        self,
        parsed: ParsedPrompt,
        section_updates: Optional[dict[str, str]] = None,
    ) -> str:
        """
        Reconstruct prompt from sections.

        Args:
            parsed: ParsedPrompt with sections
            section_updates: Optional dict of section_id -> new_content

        Returns:
            Reconstructed prompt string
        """
        if not parsed.sections:
            return ""

        # Check if sections have header info (structured) or are fallback
        if len(parsed.sections) == 1 and parsed.sections[0].section_type == PromptSectionType.FALLBACK:
            # Single fallback section - return content directly
            section = parsed.sections[0]
            if section_updates and section.id in section_updates:
                return section_updates[section.id]
            return section.content

        parts: list[str] = []
        for section in parsed.sections:
            # Get header info if available
            header_text = getattr(section, "_header_text", section.section_type.value.replace("_", " ").title())
            header_level = getattr(section, "_header_level", 2)

            # Build header
            header = "#" * header_level + " " + header_text

            # Get content (apply update if provided)
            content = section.content
            if section_updates and section.id in section_updates:
                content = section_updates[section.id]

            parts.append(header)
            if content:
                parts.append(content)
            parts.append("")  # Blank line between sections

        return "\n".join(parts).strip()

    def _infer_section_type(self, header_text: str) -> PromptSectionType:
        """
        Infer section type from header text.

        Args:
            header_text: Header text (without # symbols)

        Returns:
            Inferred PromptSectionType
        """
        header_lower = header_text.lower()

        for section_type, keywords in self.SECTION_TYPE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in header_lower:
                    return section_type

        return PromptSectionType.FALLBACK

    def _is_section_mutable(
        self,
        header_text: str,
        section_content: str,
        constraints: list[str],
    ) -> bool:
        """
        Determine if a section can be mutated.

        Section is IMMUTABLE if:
        1. Header contains [IMMUTABLE] marker
        2. Header contains immutable keywords
        3. Section content contains protected text from constraints

        Args:
            header_text: Header text
            section_content: Section body content
            constraints: Protected text constraints

        Returns:
            True if section can be mutated
        """
        # Check for [IMMUTABLE] marker
        if self._immutable_marker_pattern.search(header_text):
            return False

        # Check for immutable keywords in header
        header_lower = header_text.lower()
        for keyword in self.IMMUTABLE_KEYWORDS:
            if keyword in header_lower:
                return False

        # Check if section contains protected text
        return all(protected_text not in section_content for protected_text in constraints)

    def _is_content_mutable(
        self,
        content: str,
        constraints: list[str],
    ) -> bool:
        """
        Check if unstructured content is mutable.

        Args:
            content: Content to check
            constraints: Protected text constraints

        Returns:
            True if content can be mutated
        """
        # Check for immutable keywords
        content_lower = content.lower()
        for keyword in self.IMMUTABLE_KEYWORDS:
            if keyword in content_lower:
                return False

        # Check for protected text
        return all(protected_text not in content for protected_text in constraints)


class LLMSectionParser(BasePromptSectionParser):
    """
    LLM-based section parser for prompts without clear markdown structure.

    Uses LLM to intelligently identify logical sections in unstructured prompts.
    Fallback when MarkdownSectionParser cannot find headers.
    """

    def __init__(
        self,
        llm_provider: Optional["LLMProvider"] = None,
        model: str = "gpt-4",
    ):
        """
        Initialize LLM-based parser.

        Args:
            llm_provider: LLM provider for section identification
            model: Model to use
        """
        self.llm_provider: Optional[LLMProvider] = llm_provider
        self.model = model
        self._markdown_parser = MarkdownSectionParser()

    def parse(self, prompt: RolePrompt, role_id: str) -> ParsedPrompt:
        """
        Parse prompt using LLM to identify sections.

        First tries markdown parsing, falls back to LLM if no structure found.

        Args:
            prompt: RolePrompt to parse
            role_id: Role ID

        Returns:
            ParsedPrompt with sections
        """
        # Try markdown parsing first
        parsed = self._markdown_parser.parse(prompt, role_id)

        # If we got structured sections, return them
        if len(parsed.sections) > 1 or (
            parsed.sections and parsed.sections[0].section_type != PromptSectionType.FALLBACK
        ):
            return parsed

        # Try LLM-based parsing for unstructured prompts
        return self._llm_parse(prompt, role_id, fallback_parsed=parsed)

    def reconstruct(
        self,
        parsed: ParsedPrompt,
        section_updates: Optional[dict[str, str]] = None,
    ) -> str:
        """Delegate to markdown parser for reconstruction"""
        return self._markdown_parser.reconstruct(parsed, section_updates)

    def _llm_parse(
        self,
        prompt: RolePrompt,
        role_id: str,
        fallback_parsed: ParsedPrompt,
    ) -> ParsedPrompt:
        """Use LLM to identify logical sections in unstructured prompt.

        Called when markdown parsing finds no section headers.

        Args:
            prompt: The prompt to parse.
            role_id: Role ID for attribution.
            fallback_parsed: Fallback result if LLM parsing fails.

        Returns:
            ParsedPrompt with LLM-identified sections.
        """
        if self.llm_provider is None:
            logger.debug("No LLM provider, using fallback")
            return fallback_parsed

        llm_prompt = f"""Analyze this prompt and identify its logical sections.

PROMPT:
{prompt.content}

For each section, identify:
1. The section type: role_definition, objective, instructions, constraints, examples, format, context
2. The exact content of that section
3. Whether it contains safety/policy content that should be immutable

Output JSON format:
{{
    "sections": [
        {{"type": "<section_type>", "content": "<exact text from prompt>", "is_immutable": <true/false>}}
    ]
}}

Rules:
- Extract actual text from the prompt, don't summarize
- Mark as immutable if contains safety rules, policies, or constraints that must not change
- Order sections as they appear in the prompt"""

        try:
            from siare.services.llm_provider import LLMMessage

            response = self.llm_provider.complete(
                messages=[LLMMessage(role="user", content=llm_prompt)],
                model=self.model,
                temperature=0.1,  # Low temperature for consistent parsing
            )

            # Parse JSON response
            result = json.loads(response.content)

            sections: list[PromptSection] = []
            for section_data in result.get("sections", []):
                section_type = self._map_section_type(section_data.get("type", ""))
                is_immutable = section_data.get("is_immutable", False)

                section = PromptSection(
                    id=str(uuid.uuid4()),
                    content=section_data.get("content", ""),
                    is_mutable=not is_immutable,
                    section_type=section_type,
                    parent_role_id=role_id,
                )
                sections.append(section)

            if sections:
                return ParsedPrompt(
                    role_id=role_id,
                    sections=sections,
                    original_content=prompt.content,
                    immutable_constraints=fallback_parsed.immutable_constraints,
                )

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"LLM parsing failed: {e}")

        return fallback_parsed

    def _map_section_type(self, type_str: str) -> PromptSectionType:
        """Map string type to PromptSectionType enum.

        Args:
            type_str: String representation of section type.

        Returns:
            Corresponding PromptSectionType enum value.
        """
        mapping = {
            "role_definition": PromptSectionType.ROLE_DEFINITION,
            "objective": PromptSectionType.OBJECTIVE,
            "instructions": PromptSectionType.INSTRUCTIONS,
            "constraints": PromptSectionType.CONSTRAINTS,
            "examples": PromptSectionType.EXAMPLES,
            "format": PromptSectionType.FORMAT,
            "context": PromptSectionType.CONTEXT,
        }
        return mapping.get(type_str.lower(), PromptSectionType.FALLBACK)
