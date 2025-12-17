"""Tests for EvolutionScheduler"""

import logging

import pytest

from siare.core.models import (
    MutationType,
    ProcessConfig,
    RoleConfig,
    SOPGene,
)
from siare.services import (
    ConfigStore,
    DirectorService,
    EvaluationService,
    EvolutionScheduler,
    ExecutionEngine,
    GenePool,
    QDGridManager,
)
from tests.mocks import MockLLMProvider


@pytest.fixture
def temp_storage(tmp_path):
    """Create temporary storage directory"""
    return tmp_path


@pytest.fixture
def scheduler(temp_storage):
    """Create a test scheduler with all dependencies"""
    config_store = ConfigStore(storage_path=temp_storage / "configs")
    gene_pool = GenePool(storage_path=temp_storage / "genes")
    qd_grid = QDGridManager()
    llm_provider = MockLLMProvider()
    execution_engine = ExecutionEngine(llm_provider=llm_provider)
    evaluation_service = EvaluationService(llm_provider=llm_provider)
    director_service = DirectorService(llm_provider=llm_provider)

    scheduler = EvolutionScheduler(
        gene_pool=gene_pool,
        config_store=config_store,
        execution_engine=execution_engine,
        evaluation_service=evaluation_service,
        director_service=director_service,
        qd_grid=qd_grid,
    )

    return scheduler


def test_generate_offspring_logs_on_missing_parent(scheduler, caplog):
    """Test that _generate_offspring logs error when parent gene not found."""
    with caplog.at_level(logging.ERROR):
        result = scheduler._generate_offspring(
            parent_id="nonexistent",
            parent_version="1.0.0",
            mutation_types=[MutationType.PROMPT_CHANGE],
            constraints=None,
        )

    assert result is None
    assert "Parent gene not found" in caplog.text
    assert "nonexistent" in caplog.text
    assert "1.0.0" in caplog.text


def test_generate_offspring_logs_on_missing_sop(scheduler, caplog):
    """Test that _generate_offspring logs error when parent SOP not found."""
    # Create a gene without a corresponding SOP in config store
    gene = SOPGene(
        sopId="orphan_sop",
        version="1.0.0",
        promptGenomeId="genome1",
        promptGenomeVersion="1.0.0",
        configSnapshot=ProcessConfig(
            id="orphan_sop",
            version="1.0.0",
            models={},
            tools=[],
            roles=[],
            graph=[],
        ),
        evaluations=[],
        aggregatedMetrics={},
    )
    scheduler.gene_pool.add_sop_gene(gene)
    # Note: SOP is NOT saved to config_store, so it will be missing

    with caplog.at_level(logging.ERROR):
        result = scheduler._generate_offspring(
            parent_id="orphan_sop",
            parent_version="1.0.0",
            mutation_types=[MutationType.PROMPT_CHANGE],
            constraints=None,
        )

    assert result is None
    assert "Parent SOP not found" in caplog.text
    assert "orphan_sop" in caplog.text


def test_generate_offspring_logs_on_missing_genome(scheduler, caplog):
    """Test that _generate_offspring logs error when parent genome not found."""
    # Create SOP and gene, but no genome
    sop = ProcessConfig(
        id="sop_no_genome",
        version="1.0.0",
        models={},
        tools=[],
        roles=[
            RoleConfig(
                id="agent",
                model="gpt-4",
                promptRef="agent_prompt",
            )
        ],
        graph=[],
    )
    scheduler.config_store.save_sop(sop)

    gene = SOPGene(
        sopId="sop_no_genome",
        version="1.0.0",
        promptGenomeId="missing_genome",
        promptGenomeVersion="1.0.0",
        configSnapshot=sop,
        evaluations=[],
        aggregatedMetrics={},
    )
    scheduler.gene_pool.add_sop_gene(gene)
    # Note: Genome is NOT saved to config_store, so it will be missing

    with caplog.at_level(logging.ERROR):
        result = scheduler._generate_offspring(
            parent_id="sop_no_genome",
            parent_version="1.0.0",
            mutation_types=[MutationType.PROMPT_CHANGE],
            constraints=None,
        )

    assert result is None
    assert "Parent genome not found" in caplog.text
    assert "missing_genome" in caplog.text


def test_step_logs_warning_when_no_parents_selected(scheduler, caplog):
    """Test that step() logs warning with context when no parents are selected."""
    from siare.core.models import (
        EvolutionConstraints,
        EvolutionJob,
        EvolutionJobStatus,
        EvolutionPhase,
        SelectionStrategy,
        TaskSet,
    )

    # Create minimal job
    job = EvolutionJob(
        id="test_job",
        domain="test",
        baseSops=[],
        taskSet=TaskSet(id="test_tasks", domain="test", version="1.0.0", tasks=[]),
        metricsToOptimize=["quality"],
        qualityScoreWeights={"quality": 1.0},
        constraints=EvolutionConstraints(),
        status=EvolutionJobStatus.PENDING,
        phases=[
            EvolutionPhase(
                name="test_phase",
                maxGenerations=10,
                parentsPerGeneration=2,
                selectionStrategy=SelectionStrategy.TOURNAMENT,
                allowedMutationTypes=[MutationType.PROMPT_CHANGE],
            )
        ],
    )

    scheduler.start_job(job)

    # Empty gene pool - no parents can be selected
    assert len(list(scheduler.gene_pool.list_sop_genes())) == 0

    with caplog.at_level(logging.WARNING):
        result = scheduler.step()

    # Verify warning is logged with context
    assert "No parents selected" in caplog.text
    assert "generation 0" in caplog.text
    assert "TOURNAMENT" in caplog.text or "SelectionStrategy.TOURNAMENT" in caplog.text
    assert "Gene pool size: 0" in caplog.text

    # Verify result has proper status and warning
    assert result["status"] == "no_parents"
    assert result["offspring_count"] == 0
    assert "warning" in result
    assert "Selection returned no parents" in result["warning"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
