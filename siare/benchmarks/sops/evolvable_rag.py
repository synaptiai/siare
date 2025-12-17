"""Evolvable RAG SOP with configurable retrieval parameters.

This SOP is designed for benchmark evolution experiments:
1. Has retrieval capability via vector_search tool
2. Exposes evolvable parameters (top_k, similarity_threshold)
3. Provides baseline configs (poor, reasonable, optimal)
4. Defines parameter bounds for Director mutations

The goal is to demonstrate that evolution can discover better
retrieval configurations than hand-tuned baselines.
"""

from typing import Any

from siare.core.models import (
    GraphEdge,
    ProcessConfig,
    PromptGenome,
    RoleConfig,
    RoleInput,
    RolePrompt,
)

# Parameter bounds for evolution - Director can mutate within these ranges
EVOLVABLE_PARAM_BOUNDS: dict[str, dict[str, Any]] = {
    "top_k": {
        "min": 3,
        "max": 50,
        "type": "int",
        "description": "Number of documents to retrieve",
    },
    "similarity_threshold": {
        "min": 0.3,
        "max": 0.9,
        "type": "float",
        "description": "Minimum similarity score for retrieved documents",
    },
    "rerank_top_n": {
        "min": 0,
        "max": 20,
        "type": "int",
        "description": "Number of documents to rerank (0 = disabled)",
    },
}


def create_poor_baseline_config() -> dict[str, Any]:
    """Create intentionally suboptimal baseline configuration.

    This config retrieves too many irrelevant documents with
    a low similarity threshold, making it hard to answer correctly.

    Returns:
        Dictionary of poor baseline parameters
    """
    return {
        "top_k": 50,  # Too many documents = noise
        "similarity_threshold": 0.3,  # Too permissive = irrelevant docs
        "model": "llama3.2:1b",  # Smaller model
    }


def create_reasonable_baseline_config() -> dict[str, Any]:
    """Create hand-tuned reasonable baseline configuration.

    This is what a human might configure without optimization.

    Returns:
        Dictionary of reasonable baseline parameters
    """
    return {
        "top_k": 10,
        "similarity_threshold": 0.6,
        "model": "llama3.1:8b",
    }


def create_optimal_baseline_config() -> dict[str, Any]:
    """Create known-good configuration (target for evolution).

    This represents what we hope evolution can discover.

    Returns:
        Dictionary of optimal parameters
    """
    return {
        "top_k": 8,
        "similarity_threshold": 0.7,
        "model": "llama3.1:8b",
    }


def create_evolvable_rag_sop(
    model: str = "llama3.1:8b",
    top_k: int = 10,
    similarity_threshold: float = 0.5,
    index_name: str = "frames_corpus",
    rerank_top_n: int = 0,
) -> ProcessConfig:
    """Create a RAG SOP with evolvable retrieval parameters.

    Args:
        model: Model identifier for LLM roles
        top_k: Number of documents to retrieve
        similarity_threshold: Minimum similarity score for retrieval
        index_name: Name of the vector index to search
        rerank_top_n: Number of docs to rerank (0 = disabled)

    Returns:
        ProcessConfig ready for benchmark execution and evolution
    """
    roles = [
        # Retriever role - uses vector_search tool
        RoleConfig(
            id="retriever",
            model=model,
            tools=["vector_search"],
            promptRef="retriever_prompt",
            inputs=[RoleInput(from_="user_input")],
            outputs=["retrieved_context", "retrieved_doc_ids"],
            params={
                "top_k": top_k,
                "similarity_threshold": similarity_threshold,
                "index_name": index_name,
                "rerank_top_n": rerank_top_n,
                # Mark which params are evolvable
                "_evolvable": ["top_k", "similarity_threshold", "rerank_top_n"],
            },
        ),
        # Answerer role - generates answer from retrieved context
        RoleConfig(
            id="answerer",
            model=model,
            promptRef="answerer_prompt",
            inputs=[
                RoleInput(from_="user_input"),
                RoleInput(from_="retriever"),
            ],
            outputs=["answer"],
        ),
    ]

    graph = [
        GraphEdge(from_="user_input", to="retriever"),
        GraphEdge(from_="retriever", to="answerer"),
    ]

    return ProcessConfig(
        id="evolvable_rag",
        version="1.0.0",
        description=f"Evolvable RAG pipeline (top_k={top_k}, threshold={similarity_threshold})",
        models={model: model},
        tools=["vector_search"],
        roles=roles,
        graph=graph,
        hyperparameters={
            "top_k": top_k,
            "similarity_threshold": similarity_threshold,
            "rerank_top_n": rerank_top_n,
            "evolvable_param_bounds": EVOLVABLE_PARAM_BOUNDS,
        },
    )


def create_evolvable_rag_genome(
    retrieval_style: str = "generic",
) -> PromptGenome:
    """Create prompt genome for evolvable RAG SOP.

    Args:
        retrieval_style: "generic" (evolvable) or "optimized" (target)

    Returns:
        PromptGenome with retriever and answerer prompts
    """
    if retrieval_style == "generic":
        # Generic prompts that evolution should improve
        retriever_prompt = """You are a document retrieval assistant.

Given the question below, use the vector_search tool to find relevant documents.

Question: {query}

Search for documents that might contain the answer. Return the retrieved documents."""

        answerer_prompt = """You are a question-answering assistant.

Context from retrieved documents:
{retrieved_context}

Question: {query}

Based on the context above, provide an answer to the question.

Answer:"""

    else:  # optimized
        # Better prompts (what evolution might discover)
        retriever_prompt = """You are an expert document retrieval assistant specialized in finding precise evidence.

Question to answer: {query}

TASK: Use the vector_search tool to find documents containing facts needed to answer this question.

Search strategy:
1. Identify key entities and concepts in the question
2. Search for documents mentioning these entities
3. Prioritize documents with specific factual claims

Execute the search and return retrieved documents with their relevance scores."""

        answerer_prompt = """You are a precise question-answering assistant that synthesizes information from multiple sources.

RETRIEVED EVIDENCE:
{retrieved_context}

QUESTION: {query}

INSTRUCTIONS:
1. Identify which retrieved documents contain relevant information
2. Extract specific facts that help answer the question
3. Synthesize a clear, concise answer
4. If the evidence is insufficient, state what information is missing

Provide a direct answer based ONLY on the retrieved evidence. Do not use prior knowledge.

ANSWER:"""

    return PromptGenome(
        id="evolvable_rag_genome",
        version="1.0.0",
        rolePrompts={
            "retriever_prompt": RolePrompt(
                id="retriever_prompt",
                content=retriever_prompt,
            ),
            "answerer_prompt": RolePrompt(
                id="answerer_prompt",
                content=answerer_prompt,
            ),
        },
    )


def get_sop_retrieval_params(sop: ProcessConfig) -> dict[str, Any]:
    """Extract retrieval parameters from SOP for comparison.

    Args:
        sop: ProcessConfig to extract params from

    Returns:
        Dictionary of retrieval parameters
    """
    retriever = next((r for r in sop.roles if r.id == "retriever"), None)
    if not retriever:
        return {}

    return {
        "top_k": retriever.params.get("top_k"),
        "similarity_threshold": retriever.params.get("similarity_threshold"),
        "rerank_top_n": retriever.params.get("rerank_top_n", 0),
    }
