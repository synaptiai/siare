"""Tests for core data models"""

import pytest

from siare.core.models import (
    ConstraintViolation,
    DomainPackage,
    EvolutionJob,
    EvolutionJobStatus,
    EvolutionPhase,
    FailurePattern,
    FeedbackArtifact,
    FeedbackInjectionConfig,
    GraphEdge,
    MetricConfig,
    MetricType,
    MutationType,
    ProcessConfig,
    PromptEvolutionOrchestratorConfig,
    PromptGenome,
    PromptOptimizationStrategyType,
    RoleConfig,
    RolePrompt,
    SectionMutation,
    SectionMutationBatch,
    SelectionStrategy,
)


def test_process_config_creation():
    """Test creating a valid ProcessConfig"""
    sop = ProcessConfig(
        id="test_sop",
        version="1.0.0",
        models={"gpt-4": "gpt-5"},
        tools=["vector_search"],
        roles=[
            RoleConfig(
                id="planner",
                model="gpt-4",
                promptRef="planner_prompt",
                inputs=[{"from": "user_input"}],
                outputs=["plan"],
            )
        ],
        graph=[
            GraphEdge(from_="user_input", to="planner")
        ],
    )

    assert sop.id == "test_sop"
    assert sop.version == "1.0.0"
    assert len(sop.roles) == 1
    assert sop.roles[0].id == "planner"


def test_prompt_genome_creation():
    """Test creating a PromptGenome"""
    genome = PromptGenome(
        id="test_genome",
        version="1.0.0",
        rolePrompts={
            "planner_prompt": RolePrompt(
                id="planner_prompt",
                content="You are a planning agent.",
            )
        },
    )

    assert genome.id == "test_genome"
    assert "planner_prompt" in genome.rolePrompts


def test_metric_config_validation():
    """Test MetricConfig validation"""
    # Valid LLM judge metric
    metric = MetricConfig(
        id="factuality",
        type=MetricType.LLM_JUDGE,
        model="gpt-4",
        promptRef="judge_prompt",
        inputs=["response", "context"],
    )
    assert metric.type == MetricType.LLM_JUDGE

    # Invalid: LLM judge without model should fail
    with pytest.raises(ValueError):
        MetricConfig(
            id="bad_metric",
            type=MetricType.LLM_JUDGE,
            inputs=["response"],
        )


def test_evolution_job_validation():
    """Test EvolutionJob validation"""
    # Valid evolution job
    job = EvolutionJob(
        id="test_job",
        domain="healthcare",
        baseSops=[{"sopId": "sop1", "sopVersion": "1.0", "promptGenomeId": "pg1", "promptGenomeVersion": "1.0"}],
        taskSet={
            "id": "tasks1",
            "domain": "healthcare",
            "tasks": [],
            "version": "1.0.0",
        },
        metricsToOptimize=["accuracy", "safety"],
        qualityScoreWeights={"accuracy": 0.6, "safety": 0.4},
        constraints={},
        phases=[
            EvolutionPhase(
                name="prompt_only",
                allowedMutationTypes=[MutationType.PROMPT_CHANGE],
                selectionStrategy=SelectionStrategy.PARETO,
                parentsPerGeneration=5,
                maxGenerations=10,
            )
        ],
        status=EvolutionJobStatus.PENDING,
    )

    assert job.id == "test_job"
    assert len(job.phases) == 1

    # Invalid: weights don't sum to 1.0
    with pytest.raises(ValueError):
        EvolutionJob(
            id="bad_job",
            domain="healthcare",
            baseSops=[],
            taskSet={"id": "t1", "domain": "d", "tasks": [], "version": "1.0"},
            metricsToOptimize=["m1"],
            qualityScoreWeights={"m1": 0.5},  # Should sum to 1.0
            constraints={},
            phases=[
                EvolutionPhase(
                    name="test",
                    allowedMutationTypes=[],
                    selectionStrategy=SelectionStrategy.PARETO,
                    parentsPerGeneration=1,
                    maxGenerations=1,
                )
            ],
            status=EvolutionJobStatus.PENDING,
        )


def test_domain_package_semver_validation():
    """Test DomainPackage semantic version validation"""
    # Valid semver
    pkg = DomainPackage(
        id="healthcare_pkg",
        name="Healthcare Domain",
        version="1.2.3",
        sopTemplates=["sop1"],
        promptGenomes=["pg1"],
        metaConfigs=["mc1"],
        toolConfigs=["tc1"],
        metricConfigs=["m1"],
        evaluationTasks=["t1"],
        domainConfig={},
    )
    assert pkg.version == "1.2.3"

    # Invalid semver
    with pytest.raises(ValueError):
        DomainPackage(
            id="bad_pkg",
            name="Bad Package",
            version="v1.2",  # Invalid format
            sopTemplates=["sop1"],
            promptGenomes=["pg1"],
            metaConfigs=["mc1"],
            toolConfigs=["tc1"],
            metricConfigs=["m1"],
            evaluationTasks=["t1"],
            domainConfig={},
        )


def test_section_mutation_model_validation():
    """Test SectionMutation model validates fields correctly."""
    mutation = SectionMutation(
        section_id="instructions",
        role_id="analyst",
        original_content="Analyze data",
        mutated_content="Analyze data thoroughly",
        mutation_type="replace",
        rationale="Add specificity",
        source_strategy=PromptOptimizationStrategyType.TEXTGRAD,
        confidence=0.8,
    )

    assert mutation.section_id == "instructions"
    assert mutation.confidence == 0.8
    assert mutation.source_strategy == PromptOptimizationStrategyType.TEXTGRAD
    assert mutation.mutation_type == "replace"


def test_section_mutation_confidence_bounds():
    """Test SectionMutation confidence must be 0-1."""
    with pytest.raises(ValueError):
        SectionMutation(
            section_id="test",
            role_id="test",
            original_content="a",
            mutated_content="b",
            mutation_type="replace",
            rationale="test",
            source_strategy=PromptOptimizationStrategyType.TEXTGRAD,
            confidence=1.5,  # Invalid: > 1.0
        )


def test_section_mutation_batch_model():
    """Test SectionMutationBatch model."""
    mutation1 = SectionMutation(
        section_id="instructions",
        role_id="analyst",
        original_content="Analyze data",
        mutated_content="Analyze data thoroughly",
        mutation_type="replace",
        rationale="Add specificity",
        source_strategy=PromptOptimizationStrategyType.TEXTGRAD,
        confidence=0.8,
    )

    mutation2 = SectionMutation(
        section_id="examples",
        role_id="analyst",
        original_content="Example 1",
        mutated_content="Example 1: Detailed example",
        mutation_type="replace",
        rationale="Add detail",
        source_strategy=PromptOptimizationStrategyType.EVOPROMPT,
        confidence=0.7,
    )

    batch = SectionMutationBatch(
        prompt_genome_id="genome-123",
        prompt_genome_version="1.2.0",
        mutations=[mutation1, mutation2],
        applied=False,
    )

    assert batch.prompt_genome_id == "genome-123"
    assert len(batch.mutations) == 2
    assert batch.applied is False
    assert batch.batch_id  # Should have auto-generated ID


def test_feedback_artifact_model():
    """Test FeedbackArtifact model."""
    artifact = FeedbackArtifact(
        source_type="llm_judge",
        role_id="analyst",
        metric_id="accuracy",
        critique="Response lacked specificity",
        severity=0.7,
        failure_pattern=FailurePattern.INCOMPLETE,
        suggested_fix="Add more detail to instructions",
    )

    assert artifact.source_type == "llm_judge"
    assert artifact.failure_pattern == FailurePattern.INCOMPLETE
    assert artifact.severity == 0.7
    assert artifact.role_id == "analyst"


def test_feedback_artifact_severity_bounds():
    """Test FeedbackArtifact severity must be 0-1."""
    with pytest.raises(ValueError):
        FeedbackArtifact(
            source_type="trace",
            role_id="test",
            critique="test",
            severity=2.0,  # Invalid: > 1.0
        )


def test_constraint_violation_model():
    """Test ConstraintViolation model."""
    violation = ConstraintViolation(
        constraint_type="must_not_change",
        violation_description="Protected text 'SAFETY:' was removed",
        section_id="safety_policy",
        role_id="analyst",
        severity="error",
    )

    assert violation.constraint_type == "must_not_change"
    assert violation.severity == "error"
    assert violation.section_id == "safety_policy"


def test_feedback_injection_config_model():
    """Test FeedbackInjectionConfig model."""
    config = FeedbackInjectionConfig(
        enabled=True,
        injection_position="append",
        max_feedback_items=5,
        include_failure_patterns=True,
        include_suggestions=False,
    )

    assert config.enabled is True
    assert config.max_feedback_items == 5
    assert config.include_suggestions is False


def test_feedback_injection_config_bounds():
    """Test FeedbackInjectionConfig max_feedback_items bounds."""
    with pytest.raises(ValueError):
        FeedbackInjectionConfig(max_feedback_items=0)  # Below minimum

    with pytest.raises(ValueError):
        FeedbackInjectionConfig(max_feedback_items=11)  # Above maximum


def test_prompt_evolution_orchestrator_config_model():
    """Test PromptEvolutionOrchestratorConfig model."""
    config = PromptEvolutionOrchestratorConfig(
        default_strategy=PromptOptimizationStrategyType.METAPROMPT,
        enable_section_parsing=True,
        enable_feedback_injection=True,
        max_mutations_per_role=5,
        constraint_validation_mode="strict",
        fallback_to_full_prompt=False,
    )

    assert config.default_strategy == PromptOptimizationStrategyType.METAPROMPT
    assert config.max_mutations_per_role == 5
    assert config.constraint_validation_mode == "strict"
    assert config.fallback_to_full_prompt is False


def test_prompt_evolution_orchestrator_config_bounds():
    """Test PromptEvolutionOrchestratorConfig max_mutations_per_role bounds."""
    with pytest.raises(ValueError):
        PromptEvolutionOrchestratorConfig(max_mutations_per_role=0)  # Below minimum

    with pytest.raises(ValueError):
        PromptEvolutionOrchestratorConfig(max_mutations_per_role=11)  # Above maximum


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
