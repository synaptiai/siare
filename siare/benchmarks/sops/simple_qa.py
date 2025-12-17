"""Simple Q&A SOP for benchmark evaluation.

A minimal single-role SOP that takes a question and produces an answer.
Designed for quick evaluation with local LLMs like Ollama.
"""

from siare.core.models import (
    GraphEdge,
    ProcessConfig,
    PromptGenome,
    RoleConfig,
    RoleInput,
    RolePrompt,
)


def create_benchmark_sop(model: str = "llama3.2:1b") -> ProcessConfig:
    """Create a minimal Q&A SOP for benchmarking.

    Args:
        model: Model identifier (e.g., "llama3.2:1b", "gpt-4o-mini")

    Returns:
        ProcessConfig ready for benchmark execution
    """
    return ProcessConfig(
        id="benchmark_qa",
        version="1.0.0",
        description="Minimal Q&A pipeline for benchmark evaluation",
        models={model: model},  # Use actual model name as key
        tools=[],
        roles=[
            RoleConfig(
                id="answerer",
                model=model,  # Use actual model name directly
                promptRef="answerer_prompt",
                inputs=[RoleInput(from_="user_input")],
                outputs=["answer"],
            ),
        ],
        graph=[
            GraphEdge(from_="user_input", to="answerer"),
        ],
    )


def create_benchmark_genome() -> PromptGenome:
    """Create prompt genome for benchmark SOP.

    Returns:
        PromptGenome with answerer prompt
    """
    return PromptGenome(
        id="benchmark_qa_genome",
        version="1.0.0",
        rolePrompts={
            "answerer_prompt": RolePrompt(
                id="answerer_prompt",
                content="""You are a question-answering assistant.

Given the question below, provide a clear, accurate, and concise answer.
Focus on factual accuracy. If you're unsure, say so.

Question: {query}

Answer:""",
            ),
        },
    )
