"""
EvoPrompt Strategy Implementation

Evolutionary algorithms (GA/DE) for prompt optimization based on:
- EvoPrompt (ICLR 2024): https://github.com/beeevita/EvoPrompt

Key features:
- Population of prompt variants per role
- LLM-guided crossover and mutation operators
- Tournament/roulette selection based on fitness
- Section-aware evolution respecting immutability constraints
"""

import random
import re
from typing import Any

from siare.core.models import (
    Diagnosis,
    EvoPromptConfig,
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
MIN_LINE_LENGTH_REPHRASE = 10
MIN_WORDS_FOR_SWAP = 3
MIN_POPULATION_FOR_CROSSOVER = 2
CROSSOVER_SECTION_PROBABILITY = 0.5


class PromptVariant:
    """A single prompt variant in the population"""

    def __init__(
        self,
        role_id: str,
        content: str,
        fitness: float = 0.0,
        generation: int = 0,
        parent_ids: list[str] | None = None,
    ):
        self.id = f"{role_id}-gen{generation}-{random.randint(1000, 9999)}"  # noqa: S311
        self.role_id = role_id
        self.content = content
        self.fitness = fitness
        self.generation = generation
        self.parent_ids = parent_ids or []

    def __repr__(self) -> str:
        return f"PromptVariant(id={self.id}, fitness={self.fitness:.3f})"


class EvoPromptStrategy(BasePromptOptimizationStrategy):
    """
    EvoPrompt optimization using evolutionary algorithms.

    Maintains a population of prompt variants and evolves them using:
    - LLM-guided mutation: Generate improved variants based on feedback
    - LLM-guided crossover: Combine successful traits from parents
    - Tournament selection: Select fittest candidates for reproduction

    Based on EvoPrompt (ICLR 2024) with adaptations for SIARE.
    """

    def __init__(
        self,
        llm_provider: LLMProvider | None = None,
        config: EvoPromptConfig | None = None,
    ):
        """
        Initialize EvoPrompt strategy.

        Args:
            llm_provider: LLM provider for mutation/crossover operations
            config: Strategy configuration
        """
        self.llm_provider = llm_provider
        self.config = config or EvoPromptConfig()

        # Population storage: role_id -> list[PromptVariant]
        self._populations: dict[str, list[PromptVariant]] = {}
        self._generation = 0

    @property
    def name(self) -> str:
        return "evoprompt"

    @property
    def strategy_type(self) -> PromptOptimizationStrategyType:
        return PromptOptimizationStrategyType.EVOPROMPT

    def requires_population(self) -> bool:
        return True

    def optimize(
        self,
        sop_config: ProcessConfig,  # noqa: ARG002
        prompt_genome: PromptGenome,
        feedback: list[PromptFeedback],
        diagnosis: Diagnosis,
        parsed_prompts: dict[str, ParsedPrompt] | None = None,
        constraints: dict[str, Any] | None = None,
    ) -> PromptEvolutionResult:
        """
        Evolve prompts using evolutionary algorithms.

        1. Initialize population if needed
        2. Evaluate fitness based on feedback
        3. Select parents via tournament selection
        4. Generate offspring via crossover and mutation
        5. Return best variant

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
            "strategy": "evoprompt",
            "algorithm": self.config.algorithm,
            "generation": self._generation,
        }

        # Get roles that need optimization based on feedback
        roles_to_optimize = self._identify_roles_to_optimize(feedback, diagnosis)

        if not roles_to_optimize:
            # No roles need optimization - return unchanged
            return PromptEvolutionResult(
                new_prompt_genome=prompt_genome,
                changes_made=[],
                rationale="No roles identified for optimization based on feedback",
                strategy_metadata=metadata,
            )

        # Create new genome with evolved prompts
        new_role_prompts = dict(prompt_genome.rolePrompts)

        for role_id in roles_to_optimize:
            if role_id not in prompt_genome.rolePrompts:
                continue

            current_prompt = prompt_genome.rolePrompts[role_id]
            role_feedback = [f for f in feedback if f.role_id == role_id]
            role_constraints: dict[str, Any] = constraints.get(role_id, {})  # type: ignore
            must_not_change: list[str] = role_constraints.get("mustNotChange", [])  # type: ignore

            # Get or initialize population for this role
            population = self._get_or_init_population(role_id, current_prompt.content)

            # Update fitness based on feedback
            self._update_fitness(population, role_feedback)

            # Evolve population
            new_population = self._evolve_population(
                population=population,
                feedback=role_feedback,
                parsed_prompt=parsed_prompts.get(role_id) if parsed_prompts else None,
                must_not_change=must_not_change,
            )

            # Store new population
            self._populations[role_id] = new_population

            # Select best variant
            best_variant = max(new_population, key=lambda v: v.fitness)

            # Validate constraints
            violations = self.validate_constraints(
                current_prompt.content,
                best_variant.content,
                must_not_change,
            )

            if violations:
                # Keep original if constraints violated
                continue

            # Record change
            if best_variant.content != current_prompt.content:
                changes_made.append({
                    "role_id": role_id,
                    "section_id": "full_prompt",
                    "old": current_prompt.content[:TEXT_TRUNCATE_LENGTH] + "..."
                    if len(current_prompt.content) > TEXT_TRUNCATE_LENGTH
                    else current_prompt.content,
                    "new": best_variant.content[:TEXT_TRUNCATE_LENGTH] + "..."
                    if len(best_variant.content) > TEXT_TRUNCATE_LENGTH
                    else best_variant.content,
                    "variant_id": best_variant.id,
                    "fitness": best_variant.fitness,
                })

                new_role_prompts[role_id] = RolePrompt(
                    id=current_prompt.id,
                    content=best_variant.content,
                    constraints=current_prompt.constraints,
                )

        # Increment generation
        self._generation += 1
        metadata["generation"] = self._generation
        population_sizes: dict[str, int] = {
            role_id: len(pop) for role_id, pop in self._populations.items()
        }
        metadata["population_sizes"] = population_sizes

        # Create new genome
        new_genome = PromptGenome(
            id=prompt_genome.id,
            version=self._increment_version(prompt_genome.version),
            rolePrompts=new_role_prompts,
            metadata={
                **(prompt_genome.metadata or {}),
                "evoprompt_generation": self._generation,
            },
        )

        rationale = self._build_rationale(changes_made, roles_to_optimize)

        return PromptEvolutionResult(
            new_prompt_genome=new_genome,
            changes_made=changes_made,
            rationale=rationale,
            strategy_metadata=metadata,
        )

    def _identify_roles_to_optimize(
        self,
        feedback: list[PromptFeedback],
        diagnosis: Diagnosis,
    ) -> list[str]:
        """Identify roles that need optimization based on feedback and diagnosis."""
        # Get roles with feedback
        roles_with_feedback = {f.role_id for f in feedback}

        # Get roles mentioned in diagnosis
        diagnosed_roles: set[str] = set()

        # Check primary weakness
        if diagnosis.primaryWeakness and "role:" in diagnosis.primaryWeakness.lower():
            match = re.search(r"role:\s*(\S+)", diagnosis.primaryWeakness, re.IGNORECASE)
            if match:
                diagnosed_roles.add(match.group(1))

        # Check secondary weaknesses
        if diagnosis.secondaryWeaknesses:
            for weak in diagnosis.secondaryWeaknesses:
                if "role:" in weak.lower():
                    match = re.search(r"role:\s*(\S+)", weak, re.IGNORECASE)
                    if match:
                        diagnosed_roles.add(match.group(1))

        return list(roles_with_feedback | diagnosed_roles)

    def _get_or_init_population(
        self,
        role_id: str,
        initial_content: str,
    ) -> list[PromptVariant]:
        """Get existing population or initialize new one."""
        if self._populations.get(role_id):
            return self._populations[role_id]

        # Initialize population with copies of initial prompt
        population: list[PromptVariant] = []
        for _ in range(self.config.population_size):
            variant = PromptVariant(
                role_id=role_id,
                content=initial_content,
                fitness=0.5,  # Neutral initial fitness
                generation=0,
            )
            population.append(variant)

        return population

    def _update_fitness(
        self,
        population: list[PromptVariant],
        feedback: list[PromptFeedback],
    ) -> None:
        """Update fitness scores based on feedback."""
        if not feedback:
            return

        # Calculate average confidence from feedback (inverse as penalty)
        avg_confidence = sum(f.confidence for f in feedback) / len(feedback)

        # Count failure patterns
        failure_count = sum(1 for f in feedback if f.failure_pattern is not None)
        failure_penalty = failure_count * 0.1

        # Fitness is inverse of problems: fewer issues = higher fitness
        base_fitness = 1.0 - avg_confidence  # Lower confidence in critique = better
        adjusted_fitness = max(0.0, base_fitness - failure_penalty)

        # Update fitness for current generation variants
        for variant in population:
            if variant.generation == self._generation:
                variant.fitness = adjusted_fitness

    def _evolve_population(
        self,
        population: list[PromptVariant],
        feedback: list[PromptFeedback],
        parsed_prompt: ParsedPrompt | None,
        must_not_change: list[str],
    ) -> list[PromptVariant]:
        """
        Evolve population through selection, crossover, and mutation.

        Uses LLM-guided operators when provider available, otherwise
        falls back to heuristic operators.
        """
        new_population: list[PromptVariant] = []

        # Elitism: keep top performers
        sorted_pop = sorted(population, key=lambda v: v.fitness, reverse=True)
        elite_count = max(1, self.config.population_size // 4)
        elites = sorted_pop[:elite_count]
        new_population.extend(elites)

        # Generate offspring to fill population
        while len(new_population) < self.config.population_size:
            if random.random() < self.config.crossover_rate and len(population) >= MIN_POPULATION_FOR_CROSSOVER:  # noqa: S311
                # Crossover
                parent1 = self._tournament_select(population)
                parent2 = self._tournament_select(population)

                offspring = self._crossover(
                    parent1, parent2, feedback, parsed_prompt, must_not_change
                )
                new_population.append(offspring)

            else:
                # Mutation
                parent = self._tournament_select(population)

                if random.random() < self.config.mutation_rate:  # noqa: S311
                    offspring = self._mutate(
                        parent, feedback, parsed_prompt, must_not_change
                    )
                else:
                    # Clone without mutation
                    offspring = PromptVariant(
                        role_id=parent.role_id,
                        content=parent.content,
                        fitness=parent.fitness * 0.9,  # Slight decay
                        generation=self._generation + 1,
                        parent_ids=[parent.id],
                    )
                new_population.append(offspring)

        return new_population[: self.config.population_size]

    def _tournament_select(
        self,
        population: list[PromptVariant],
        tournament_size: int = 3,
    ) -> PromptVariant:
        """Select parent via tournament selection."""
        tournament = random.sample(
            population, min(tournament_size, len(population))
        )
        return max(tournament, key=lambda v: v.fitness)

    def _roulette_select(self, population: list[PromptVariant]) -> PromptVariant:
        """Select parent via fitness-proportionate (roulette) selection."""
        total_fitness = sum(max(0.01, v.fitness) for v in population)
        pick = random.uniform(0, total_fitness)  # noqa: S311

        current = 0
        for variant in population:
            current += max(0.01, variant.fitness)
            if current >= pick:
                return variant

        return population[-1]

    def _crossover(
        self,
        parent1: PromptVariant,
        parent2: PromptVariant,
        feedback: list[PromptFeedback],
        parsed_prompt: ParsedPrompt | None,
        must_not_change: list[str],
    ) -> PromptVariant:
        """
        Crossover two parents to create offspring.

        Uses LLM-guided crossover if provider available.
        """
        if self.llm_provider:
            offspring_content = self._llm_crossover(
                parent1.content, parent2.content, feedback, must_not_change
            )
        else:
            offspring_content = self._heuristic_crossover(
                parent1.content, parent2.content, parsed_prompt
            )

        return PromptVariant(
            role_id=parent1.role_id,
            content=offspring_content,
            fitness=(parent1.fitness + parent2.fitness) / 2,  # Inherit avg fitness
            generation=self._generation + 1,
            parent_ids=[parent1.id, parent2.id],
        )

    def _mutate(
        self,
        parent: PromptVariant,
        feedback: list[PromptFeedback],
        parsed_prompt: ParsedPrompt | None,
        must_not_change: list[str],
    ) -> PromptVariant:
        """
        Mutate a parent to create offspring.

        Uses LLM-guided mutation if provider available.
        """
        if self.llm_provider:
            mutated_content = self._llm_mutate(
                parent.content, feedback, must_not_change
            )
        else:
            mutated_content = self._heuristic_mutate(
                parent.content, feedback, parsed_prompt
            )

        return PromptVariant(
            role_id=parent.role_id,
            content=mutated_content,
            fitness=parent.fitness * 0.9,  # Slight fitness penalty for mutation
            generation=self._generation + 1,
            parent_ids=[parent.id],
        )

    def _llm_crossover(
        self,
        parent1_content: str,
        parent2_content: str,
        feedback: list[PromptFeedback],
        must_not_change: list[str],
    ) -> str:
        """Use LLM to intelligently combine two parent prompts."""
        protected_text = "\n".join(f"- {p}" for p in must_not_change) if must_not_change else "None"

        feedback_summary = ""
        if feedback:
            feedback_items = [
                f"- {f.failure_pattern.value if f.failure_pattern else 'issue'}: {f.critique[:100]}"
                for f in feedback[:3]
            ]
            feedback_summary = "\n".join(feedback_items)

        messages = [
            LLMMessage(
                role="system",
                content="""You are an expert prompt engineer performing crossover on two prompt variants.
Your task is to combine the best elements from both parents to create an improved offspring.

Rules:
1. Preserve any protected text exactly as-is
2. Combine complementary strengths from both parents
3. Address the feedback issues where possible
4. Maintain coherent structure and flow
5. Output ONLY the new prompt, no explanations""",
            ),
            LLMMessage(
                role="user",
                content=f"""Combine these two prompt variants:

PARENT 1:
{parent1_content}

PARENT 2:
{parent2_content}

PROTECTED TEXT (must keep exactly):
{protected_text}

FEEDBACK TO ADDRESS:
{feedback_summary if feedback_summary else "No specific feedback"}

Output only the combined prompt:""",
            ),
        ]

        try:
            if self.llm_provider is not None:
                response = self.llm_provider.complete(
                    messages=messages,
                    model=self.config.model if hasattr(self.config, "model") else "gpt-4",
                    temperature=0.7,
                    max_tokens=2000,
                )
                return response.content.strip()
        except Exception:
            pass
        # Fallback to heuristic
        return self._heuristic_crossover(parent1_content, parent2_content, None)

    def _llm_mutate(
        self,
        content: str,
        feedback: list[PromptFeedback],
        must_not_change: list[str],
    ) -> str:
        """Use LLM to intelligently mutate a prompt based on feedback."""
        protected_text = "\n".join(f"- {p}" for p in must_not_change) if must_not_change else "None"

        feedback_details = ""
        if feedback:
            for f in feedback[:3]:
                pattern = f.failure_pattern.value if f.failure_pattern else "issue"
                feedback_details += f"\n- {pattern}: {f.critique}"
                if f.suggested_improvement:
                    feedback_details += f"\n  Suggestion: {f.suggested_improvement}"

        messages = [
            LLMMessage(
                role="system",
                content="""You are an expert prompt engineer performing mutation on a prompt.
Your task is to improve the prompt by addressing the feedback while preserving what works.

Rules:
1. Preserve any protected text exactly as-is
2. Focus on addressing the specific feedback issues
3. Make targeted improvements, not wholesale rewrites
4. Maintain the overall structure and intent
5. Output ONLY the improved prompt, no explanations""",
            ),
            LLMMessage(
                role="user",
                content=f"""Mutate this prompt to address the feedback:

CURRENT PROMPT:
{content}

PROTECTED TEXT (must keep exactly):
{protected_text}

FEEDBACK TO ADDRESS:
{feedback_details if feedback_details else "General improvement needed"}

Output only the improved prompt:""",
            ),
        ]

        try:
            if self.llm_provider is not None:
                response = self.llm_provider.complete(
                    messages=messages,
                    model=self.config.model if hasattr(self.config, "model") else "gpt-4",
                    temperature=0.8,  # Slightly higher for more variation
                    max_tokens=2000,
                )
                return response.content.strip()
        except Exception:
            pass
        # Fallback to heuristic
        return self._heuristic_mutate(content, feedback, None)

    def _heuristic_crossover(
        self,
        parent1_content: str,
        parent2_content: str,
        parsed_prompt: ParsedPrompt | None,
    ) -> str:
        """Heuristic crossover when LLM not available."""
        if parsed_prompt and len(parsed_prompt.sections) > 1:
            # Section-based crossover
            return self._section_crossover(parent1_content, parent2_content, parsed_prompt)
        # Line-based crossover
        return self._line_crossover(parent1_content, parent2_content)

    def _section_crossover(
        self,
        parent1_content: str,
        parent2_content: str,
        parsed_prompt: ParsedPrompt,  # noqa: ARG002
    ) -> str:
        """Crossover at section boundaries."""
        # Split both parents by markdown headers
        parent1_sections = self._split_by_headers(parent1_content)
        parent2_sections = self._split_by_headers(parent2_content)

        # Randomly select sections from each parent
        result_sections: list[str] = []
        max_sections = max(len(parent1_sections), len(parent2_sections))

        for i in range(max_sections):
            if random.random() < CROSSOVER_SECTION_PROBABILITY:  # noqa: S311
                if i < len(parent1_sections):
                    result_sections.append(parent1_sections[i])
                elif i < len(parent2_sections):
                    result_sections.append(parent2_sections[i])
            elif i < len(parent2_sections):
                result_sections.append(parent2_sections[i])
            elif i < len(parent1_sections):
                result_sections.append(parent1_sections[i])

        return "\n\n".join(result_sections)

    def _line_crossover(
        self,
        parent1_content: str,
        parent2_content: str,
    ) -> str:
        """Simple line-based crossover."""
        lines1 = parent1_content.split("\n")
        lines2 = parent2_content.split("\n")

        # Ensure we have at least one line from each
        if not lines1 and not lines2:
            return parent1_content or parent2_content

        # Single-point crossover with safeguards
        crossover_point1 = random.randint(0, len(lines1)) if lines1 else 0  # noqa: S311
        crossover_point2 = random.randint(0, len(lines2)) if lines2 else 0  # noqa: S311

        offspring_lines = lines1[:crossover_point1] + lines2[crossover_point2:]

        # Ensure non-empty result
        if not offspring_lines or not any(line.strip() for line in offspring_lines):
            # Fall back to combining both parents
            offspring_lines = lines1 + lines2 if lines1 and lines2 else lines1 or lines2

        return "\n".join(offspring_lines)

    def _heuristic_mutate(
        self,
        content: str,
        feedback: list[PromptFeedback],  # noqa: ARG002
        parsed_prompt: ParsedPrompt | None,  # noqa: ARG002
    ) -> str:
        """Heuristic mutation when LLM not available."""
        lines = content.split("\n")

        # Find mutable lines (skip headers and protected content)
        mutable_indices: list[int] = []
        for i, line in enumerate(lines):
            if not line.strip().startswith("#") and line.strip():
                mutable_indices.append(i)

        if not mutable_indices:
            return content

        # Apply random mutations
        mutations_applied = 0
        max_mutations = max(1, len(mutable_indices) // 4)

        while mutations_applied < max_mutations:
            idx = random.choice(mutable_indices)  # noqa: S311
            line = lines[idx]

            mutation_type = random.choice(["rephrase", "expand", "clarify"])  # noqa: S311

            if mutation_type == "rephrase" and len(line) > MIN_LINE_LENGTH_REPHRASE:
                # Simple word shuffle for rephrasing
                words: list[str] = line.split()
                if len(words) > MIN_WORDS_FOR_SWAP:
                    # Swap two adjacent words
                    swap_idx = random.randint(0, len(words) - 2)  # noqa: S311
                    words[swap_idx], words[swap_idx + 1] = (
                        words[swap_idx + 1],
                        words[swap_idx],
                    )
                    lines[idx] = " ".join(words)

            elif mutation_type == "expand":
                # Add emphasis or clarification marker
                if not line.strip().startswith("-"):
                    lines[idx] = f"- {line.strip()}"
                else:
                    lines[idx] = f"{line} (important)"

            elif mutation_type == "clarify":
                # Add specificity marker
                if "specific" not in line.lower():
                    lines[idx] = line.rstrip() + " Be specific."

            mutations_applied += 1

        return "\n".join(lines)

    def _split_by_headers(self, content: str) -> list[str]:
        """Split content by markdown headers."""
        sections: list[str] = []
        current_section: list[str] = []

        for line in content.split("\n"):
            if line.strip().startswith("#"):
                if current_section:
                    sections.append("\n".join(current_section))
                current_section = [line]
            else:
                current_section.append(line)

        if current_section:
            sections.append("\n".join(current_section))

        return sections if sections else [content]

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
        roles_optimized: list[str],
    ) -> str:
        """Build human-readable rationale for changes."""
        if not changes_made:
            return f"No changes made to roles: {', '.join(roles_optimized)}"

        rationale_parts = [
            f"EvoPrompt (gen {self._generation}): Evolved {len(changes_made)} prompt(s)"
        ]

        for change in changes_made[:CHANGES_DISPLAY_LIMIT]:
            rationale_parts.append(
                f"- {change['role_id']}: variant {change.get('variant_id', 'unknown')} "
                f"(fitness: {change.get('fitness', 0):.2f})"
            )

        if len(changes_made) > CHANGES_DISPLAY_LIMIT:
            rationale_parts.append(f"- ... and {len(changes_made) - CHANGES_DISPLAY_LIMIT} more")

        return "\n".join(rationale_parts)

    # Population management methods for external use

    def get_population(self, role_id: str) -> list[PromptVariant]:
        """Get current population for a role."""
        return self._populations.get(role_id, [])

    def set_population(
        self,
        role_id: str,
        variants: list[tuple[str, float]],
    ) -> None:
        """
        Set population from external source (e.g., GenePool).

        Args:
            role_id: Role identifier
            variants: List of (content, fitness) tuples
        """
        population = [
            PromptVariant(
                role_id=role_id,
                content=content,
                fitness=fitness,
                generation=self._generation,
            )
            for content, fitness in variants
        ]
        self._populations[role_id] = population

    def get_best_variant(self, role_id: str) -> PromptVariant | None:
        """Get best variant for a role."""
        population = self._populations.get(role_id, [])
        if not population:
            return None
        return max(population, key=lambda v: v.fitness)

    def reset_population(self, role_id: str | None = None) -> None:
        """Reset population(s) to empty."""
        if role_id:
            self._populations.pop(role_id, None)
        else:
            self._populations.clear()
            self._generation = 0
