"""RAG SOP with vector search retrieval for benchmark evaluation.

A two-role SOP that:
1. Retrieves relevant documents using vector search
2. Generates an answer grounded in retrieved context

Designed for BEIR-style retrieval benchmarks and general RAG evaluation.
"""


from siare.core.models import (
    GraphEdge,
    ProcessConfig,
    PromptGenome,
    RoleConfig,
    RoleInput,
    RolePrompt,
)


def create_rag_sop(
    model: str = "llama3.2:1b",
    top_k: int = 10,
    index_name: str = "benchmark_corpus",
) -> ProcessConfig:
    """Create a RAG SOP with vector search retrieval.

    Args:
        model: Model identifier for the answerer role
        top_k: Number of documents to retrieve
        index_name: Name of the vector index to search

    Returns:
        ProcessConfig ready for benchmark execution
    """
    return ProcessConfig(
        id="rag_retriever",
        version="1.0.0",
        description="RAG pipeline with vector search for benchmarks",
        models={model: model},
        tools=["vector_search"],
        roles=[
            RoleConfig(
                id="retriever",
                model=model,
                tools=["vector_search"],
                promptRef="retriever_prompt",
                inputs=[RoleInput(from_="user_input")],
                outputs=["retrieved_context", "doc_ids"],
                params={
                    "top_k": top_k,
                    "index_name": index_name,
                },
            ),
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
        ],
        graph=[
            GraphEdge(from_="user_input", to="retriever"),
            GraphEdge(from_="retriever", to="answerer"),
        ],
    )


def create_rag_genome(top_k: int = 10) -> PromptGenome:
    """Create prompt genome for RAG SOP.

    Args:
        top_k: Number of documents mentioned in prompts

    Returns:
        PromptGenome with retriever and answerer prompts
    """
    return PromptGenome(
        id="rag_retriever_genome",
        version="1.0.0",
        rolePrompts={
            "retriever_prompt": RolePrompt(
                id="retriever_prompt",
                content=f"""You are a document retrieval assistant.

Given the question below, use the vector_search tool to find the {top_k} most relevant documents.

Question: {{query}}

Call the vector_search tool with the question as the query. Return the retrieved documents as context for answering.""",
            ),
            "answerer_prompt": RolePrompt(
                id="answerer_prompt",
                content="""You are a question-answering assistant that uses retrieved documents.

Retrieved Context:
{retrieved_context}

Question: {query}

Based ONLY on the retrieved context above, provide a clear, accurate, and concise answer.
If the context doesn't contain the answer, say "I cannot find this information in the provided documents."

Answer:""",
            ),
        },
    )
