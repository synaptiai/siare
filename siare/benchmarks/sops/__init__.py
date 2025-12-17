"""Pre-built SOPs for benchmark evaluation.

SOPs:
- simple_qa: Minimal Q&A (no retrieval) for baseline testing
- rag_retriever: Vector search retrieval + generation
- multihop_rag: Multi-step retrieval for complex reasoning
- evolvable_rag: RAG with configurable, evolvable parameters
"""

from siare.benchmarks.sops.evolvable_rag import (
    EVOLVABLE_PARAM_BOUNDS,
    create_evolvable_rag_genome,
    create_evolvable_rag_sop,
    create_optimal_baseline_config,
    create_poor_baseline_config,
    create_reasonable_baseline_config,
    get_sop_retrieval_params,
)
from siare.benchmarks.sops.multihop_rag import (
    create_multihop_genome,
    create_multihop_sop,
)
from siare.benchmarks.sops.rag_retriever import (
    create_rag_genome,
    create_rag_sop,
)
from siare.benchmarks.sops.simple_qa import (
    create_benchmark_genome,
    create_benchmark_sop,
)


__all__ = [
    "create_benchmark_genome",
    "create_benchmark_sop",
    "create_multihop_genome",
    "create_multihop_sop",
    "create_rag_genome",
    "create_rag_sop",
    # Evolvable RAG
    "EVOLVABLE_PARAM_BOUNDS",
    "create_evolvable_rag_genome",
    "create_evolvable_rag_sop",
    "create_optimal_baseline_config",
    "create_poor_baseline_config",
    "create_reasonable_baseline_config",
    "get_sop_retrieval_params",
]
