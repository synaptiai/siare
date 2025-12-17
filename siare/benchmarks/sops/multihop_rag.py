"""Multi-hop RAG SOP for complex reasoning benchmarks.

A multi-step SOP that:
1. Decomposes complex questions into sub-queries
2. Retrieves information for each sub-query (using web search for Wikipedia)
3. Aggregates evidence across multiple hops
4. Synthesizes a final answer

Designed for FRAMES-style multi-hop reasoning benchmarks.
"""

from siare.core.models import (
    GraphEdge,
    ProcessConfig,
    PromptGenome,
    RoleConfig,
    RoleInput,
    RolePrompt,
)


def create_multihop_sop(
    model: str = "llama3.2:1b",
    max_hops: int = 3,
    search_provider: str = "duckduckgo",
) -> ProcessConfig:
    """Create a multi-hop RAG SOP for complex reasoning.

    Args:
        model: Model identifier for all roles
        max_hops: Maximum number of retrieval iterations
        search_provider: Web search provider for Wikipedia access

    Returns:
        ProcessConfig ready for multi-hop benchmark execution
    """
    roles = [
        # Step 1: Decompose the complex question
        RoleConfig(
            id="query_decomposer",
            model=model,
            promptRef="decomposer_prompt",
            inputs=[RoleInput(from_="user_input")],
            outputs=["sub_queries", "reasoning_plan"],
        ),
        # Step 2: Retrieve information for sub-queries
        RoleConfig(
            id="retriever",
            model=model,
            tools=["web_search"],
            promptRef="retriever_prompt",
            inputs=[
                RoleInput(from_="user_input"),
                RoleInput(from_="query_decomposer"),
            ],
            outputs=["retrieved_evidence", "sources"],
            params={
                "max_results": 5,
                "search_provider": search_provider,
            },
        ),
        # Step 3: Analyze and extract relevant facts
        RoleConfig(
            id="fact_extractor",
            model=model,
            promptRef="extractor_prompt",
            inputs=[
                RoleInput(from_="user_input"),
                RoleInput(from_="query_decomposer"),
                RoleInput(from_="retriever"),
            ],
            outputs=["extracted_facts", "missing_info"],
        ),
        # Step 4: Synthesize final answer
        RoleConfig(
            id="synthesizer",
            model=model,
            promptRef="synthesizer_prompt",
            inputs=[
                RoleInput(from_="user_input"),
                RoleInput(from_="query_decomposer"),
                RoleInput(from_="fact_extractor"),
            ],
            outputs=["answer", "confidence", "reasoning_trace"],
        ),
    ]

    graph = [
        GraphEdge(from_="user_input", to="query_decomposer"),
        GraphEdge(from_="query_decomposer", to="retriever"),
        GraphEdge(from_="retriever", to="fact_extractor"),
        GraphEdge(from_="fact_extractor", to="synthesizer"),
    ]

    return ProcessConfig(
        id="multihop_rag",
        version="1.0.0",
        description=f"Multi-hop RAG pipeline ({max_hops} hops max) for complex reasoning",
        models={model: model},
        tools=["web_search"],
        roles=roles,
        graph=graph,
        hyperparameters={
            "max_hops": max_hops,
        },
    )


def create_multihop_genome() -> PromptGenome:
    """Create prompt genome for multi-hop RAG SOP.

    Returns:
        PromptGenome with prompts for all roles
    """
    return PromptGenome(
        id="multihop_rag_genome",
        version="1.0.0",
        rolePrompts={
            "decomposer_prompt": RolePrompt(
                id="decomposer_prompt",
                content="""You are a query decomposition expert.

Given a complex question that requires multiple pieces of information, break it down into simpler sub-questions that can each be answered with a single search.

Complex Question: {query}

Decompose this into 2-4 sub-questions. For each sub-question:
1. State what specific fact we need to find
2. Suggest a search query to find it

Format your response as:
SUB-QUERY 1: [search query]
LOOKING FOR: [what fact we need]

SUB-QUERY 2: [search query]
LOOKING FOR: [what fact we need]

...""",
            ),
            "retriever_prompt": RolePrompt(
                id="retriever_prompt",
                content="""You are a research assistant with web search access.

Original Question: {query}

Sub-queries to investigate:
{sub_queries}

Use the web_search tool to find information for each sub-query. Search for Wikipedia articles and authoritative sources.

For each sub-query, perform a search and record the relevant findings.""",
            ),
            "extractor_prompt": RolePrompt(
                id="extractor_prompt",
                content="""You are a fact extraction expert.

Original Question: {query}

Research Plan:
{reasoning_plan}

Retrieved Evidence:
{retrieved_evidence}

Extract the specific facts needed to answer the original question.

For each fact:
- State the fact clearly
- Note which source it came from
- Rate confidence (HIGH/MEDIUM/LOW)

If any required information is missing, note what we still need to find.

EXTRACTED FACTS:
1. [fact] (Source: [source], Confidence: [level])
...""",
            ),
            "synthesizer_prompt": RolePrompt(
                id="synthesizer_prompt",
                content="""You are a synthesis expert who combines multiple facts into a coherent answer.

Original Question: {query}

Reasoning Plan:
{reasoning_plan}

Extracted Facts:
{extracted_facts}

Using ONLY the extracted facts above, synthesize a final answer to the original question.

Show your reasoning step by step:
1. What facts are relevant?
2. How do they connect?
3. What is the final answer?

If the facts are insufficient, explain what's missing and give the best possible answer with caveats.

REASONING:
[your step-by-step reasoning]

FINAL ANSWER:
[your concise answer]""",
            ),
        },
    )
