"""Core data models for SIARE"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, ClassVar, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


__all__ = [
    "Action",
    # Enums
    "AggregationMethod",
    "AggregatedMetric",
    # Models - Approval Workflow
    "ApprovalDecision",
    "ApprovalStage",
    "ApprovalStageStatus",
    "ApprovalType",
    "ApprovalWorkflow",
    "ApprovalWorkflowInstance",
    "BudgetLimit",
    "BudgetUsage",
    "CircuitBreakerConfig",
    "CircuitState",
    # Models - Deployment
    "DeploymentState",
    "DeploymentVersion",
    # Models - Director
    "Diagnosis",
    "DomainConfig",
    "DomainDependency",
    # Models - Domains
    "DomainPackage",
    "ErrorCategory",
    "ErrorContext",
    "ErrorSeverity",
    "EvaluationArtifacts",
    "EvaluationVector",
    "EvolutionConstraints",
    # Models - Evolution
    "EvolutionJob",
    "EvolutionJobStatus",
    "EvolutionPhase",
    "GraphEdge",
    "KillSwitchResult",
    # Models - Meta Evolution
    "MetaConfig",
    "MetaGene",
    "MetaMutation",
    # Models - Metrics
    "MetricConfig",
    "MetricResult",
    "MetricSource",
    "MetricType",
    "MutationType",
    "OutlierInfo",
    "ParetoFlags",
    "Permission",
    "PermissionCondition",
    "PermissionScope",
    # Models - Process Config (SOP)
    "ProcessConfig",
    "PromptConstraints",
    # Models - Prompt Genome
    "PromptGenome",
    # Models - QD Grid
    "QDCell",
    "QDFeatures",
    "ResourceType",
    "RetryConfig",
    "RollbackResult",
    "Role",
    "RoleConfig",
    "RoleInput",
    "RolePrompt",
    "SafetyValidationReport",
    # Models - Gene Pool
    "SOPDeployment",
    "SOPGene",
    "SOPMutation",
    "SelectionStrategy",
    "StatisticalTestResult",
    "StopConditions",
    "Task",
    "TaskMetadata",
    # Models - Tasks
    "TaskSet",
    # Models - Tools
    "ToolConfig",
    "ToolType",
    # Models - Users & Permissions
    "User",
    "UserPreferences",
    "UserStatus",
    # Models - Validation
    "ValidationCheckResult",
    "ValidationError",
    "ValidationResult",
    # Models - Prompt Evolution
    "PromptOptimizationStrategyType",
    "FailurePattern",
    "PromptSectionType",
    "PromptFeedback",
    "PromptSection",
    "ParsedPrompt",
    "PromptEvolutionResult",
    "SectionMutation",
    "SectionMutationBatch",
    "FeedbackArtifact",
    "ConstraintViolation",
    "FeedbackInjectionConfig",
    "PromptEvolutionOrchestratorConfig",
    "TextGradConfig",
    "EvoPromptConfig",
    "MetaPromptConfig",
]


# ============================================================================
# Helper Functions
# ============================================================================


def _utc_now_iso() -> str:
    """Return current UTC timestamp in ISO format"""
    return datetime.now(timezone.utc).isoformat()


# ============================================================================
# Type Aliases
# ============================================================================

ModelRef = str  # e.g., "gpt-5", "llama-3-70b"
ToolRef = str  # Tool configuration ID
PromptRef = str  # Reference to prompt in PromptGenome


# ============================================================================
# ProcessConfig (SOP) Models
# ============================================================================


class RoleInput(BaseModel):
    """Input configuration for a role"""

    from_: Union[str, list[str]] = Field(
        validation_alias="from", serialization_alias="from"
    )
    fields: Optional[list[str]] = None

    model_config = ConfigDict(populate_by_name=True, by_alias=True)  # type: ignore[call-arg]


class RoleConfig(BaseModel):
    """Configuration for a single role in the SOP"""

    id: str
    model: ModelRef
    tools: Optional[list[ToolRef]] = None
    promptRef: PromptRef
    inputs: Optional[list[RoleInput]] = None
    outputs: Optional[list[str]] = None
    params: Optional[dict[str, Any]] = None


class GraphEdge(BaseModel):
    """Edge in the execution graph"""

    from_: Union[str, list[str]] = Field(
        validation_alias="from", serialization_alias="from"
    )
    to: str
    condition: Optional[str] = None

    model_config = ConfigDict(populate_by_name=True, by_alias=True)  # type: ignore[call-arg]


class ProcessConfig(BaseModel):
    """SOP (Standard Operating Procedure) configuration"""

    id: str
    version: str
    description: Optional[str] = None
    models: dict[str, ModelRef]
    tools: list[ToolRef]
    roles: list[RoleConfig]
    graph: list[GraphEdge]
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    metadata: Optional[dict[str, Any]] = None


# ============================================================================
# PromptGenome & MetaConfig Models
# ============================================================================


class PromptConstraints(BaseModel):
    """Constraints for prompt evolution"""

    mustNotChange: Optional[list[str]] = None
    allowedChanges: Optional[list[str]] = None
    domainTips: Optional[list[str]] = None
    maxLength: Optional[int] = None
    minLength: Optional[int] = None


class RolePrompt(BaseModel):
    """Prompt definition for a role"""

    id: str
    content: str
    constraints: Optional[PromptConstraints] = None


class PromptGenome(BaseModel):
    """Collection of prompts for all roles in an SOP"""

    id: str
    version: str
    rolePrompts: dict[str, RolePrompt]  # promptRef -> RolePrompt
    metadata: Optional[dict[str, Any]] = None


class MetaConfig(BaseModel):
    """Meta-configuration for Director and Judges"""

    id: str
    version: str
    directorPrompt: RolePrompt
    judgePrompts: dict[str, RolePrompt]  # metricId -> judge prompt
    globalAgentArchetypes: Optional[dict[str, RolePrompt]] = None
    metadata: Optional[dict[str, Any]] = None


# ============================================================================
# Tool & Adapter Config
# ============================================================================


class ToolType(str, Enum):
    """Types of tools/adapters"""

    VECTOR_SEARCH = "vector_search"
    SQL = "sql"
    API = "api"
    FILE_SYSTEM = "file_system"
    WEB_SEARCH = "web_search"
    CUSTOM = "custom"


class ToolConfig(BaseModel):
    """Configuration for a tool/adapter"""

    id: str
    type: ToolType
    config: dict[str, Any]


# ============================================================================
# Metric, Evaluation & Artifacts Models
# ============================================================================


class MetricType(str, Enum):
    """Types of metrics"""

    LLM_JUDGE = "llm_judge"
    PROGRAMMATIC = "programmatic"
    RUNTIME = "runtime"
    HUMAN = "human"


class AggregationMethod(str, Enum):
    """Methods for aggregating metrics across tasks"""

    MEAN = "mean"
    MEDIAN = "median"
    MIN = "min"
    MAX = "max"
    P95 = "p95"
    WEIGHTED = "weighted"


class MetricConfig(BaseModel):
    """Configuration for a metric"""

    id: str
    type: MetricType
    model: Optional[ModelRef] = None
    promptRef: Optional[str] = None
    fnRef: Optional[str] = None
    inputs: list[str]
    aggregationMethod: AggregationMethod = AggregationMethod.MEAN
    weight: float = 1.0

    @model_validator(mode="after")
    def validate_metric_config(self) -> MetricConfig:
        """Validate metric configuration based on type"""
        if self.type == MetricType.LLM_JUDGE:
            if not self.model or not self.promptRef:
                raise ValueError("llm_judge metrics must have model and promptRef")
        elif self.type in [MetricType.PROGRAMMATIC, MetricType.RUNTIME] and not self.fnRef:
            raise ValueError(f"{self.type} metrics must have fnRef")
        return self


class MetricSource(str, Enum):
    """Source of metric evaluation"""

    LLM = "llm"
    PROGRAMMATIC = "programmatic"
    RUNTIME = "runtime"
    HUMAN = "human"


class MetricResult(BaseModel):
    """Result of a single metric evaluation"""

    metricId: str
    score: float  # Normalized 0-1
    rawValue: Optional[Any] = None
    reasoning: Optional[str] = None
    source: MetricSource


class EvaluationArtifacts(BaseModel):
    """Artifacts from evaluation for feedback"""

    llmFeedback: Optional[dict[str, str]] = None  # metricId -> critique
    failureModes: Optional[list[str]] = None
    toolErrors: Optional[list[str]] = None
    traceRefs: Optional[list[str]] = None
    extra: dict[str, Any] = Field(default_factory=dict)


class EvaluationVector(BaseModel):
    """Complete evaluation of an SOP run"""

    sopId: str
    sopVersion: str
    promptGenomeId: str
    promptGenomeVersion: str
    runId: str
    metrics: list[MetricResult]
    artifacts: Optional[EvaluationArtifacts] = None
    timestamp: str = Field(default_factory=_utc_now_iso)
    taskMetadata: Optional[dict[str, Any]] = None


# ============================================================================
# Statistical Aggregation Models
# ============================================================================


class OutlierInfo(BaseModel):
    """Information about detected outliers in metric data"""

    indices: list[int]  # Indices of outlier samples
    values: list[float]  # Outlier values
    method: str  # Detection method used (e.g., "iqr", "zscore")
    threshold: Optional[float] = None  # Threshold used for detection


class AggregatedMetric(BaseModel):
    """Statistically aggregated metric with confidence intervals"""

    metricId: str
    mean: float
    median: float
    trimmedMean: Optional[float] = None  # 10% trimmed mean
    confidenceInterval: Optional[tuple[float, float]] = None  # (lower, upper) 95% CI
    standardDeviation: Optional[float] = None
    standardError: Optional[float] = None
    sampleSize: int
    outliers: Optional[OutlierInfo] = None
    aggregationMethod: AggregationMethod
    rawValues: Optional[list[float]] = None  # Optional storage of raw values


class StatisticalTestResult(BaseModel):
    """Result of statistical hypothesis test"""

    testType: str  # "mannwhitneyu", "wilcoxon", "ttest"
    statistic: float
    pValue: float
    isSignificant: bool  # p < 0.05
    effectSize: Optional[float] = None  # Cohen's d or rank-biserial correlation
    confidenceLevel: float = 0.95
    hypothesis: str  # Description of the hypothesis tested


# ============================================================================
# Error Handling Models
# ============================================================================


class ErrorCategory(str, Enum):
    """Category of error for classification and handling"""

    TRANSIENT = "transient"  # Temporary, retry may succeed
    PERMANENT = "permanent"  # Will not succeed on retry
    DEGRADED = "degraded"  # Partial success possible
    CRITICAL = "critical"  # System integrity at risk


class ErrorSeverity(str, Enum):
    """Severity level of error"""

    LOW = "low"  # Log and continue
    MEDIUM = "medium"  # Log, alert, continue with degradation
    HIGH = "high"  # Halt operation, alert, attempt recovery
    CRITICAL = "critical"  # Halt system, alert, manual intervention required


class CircuitState(str, Enum):
    """Circuit breaker state"""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Too many failures, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


class ErrorContext(BaseModel):
    """Context information for an error"""

    category: ErrorCategory
    severity: ErrorSeverity
    component: str  # Which component failed
    operation: str  # What operation failed
    errorMessage: str
    stackTrace: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None
    timestamp: str = Field(default_factory=_utc_now_iso)
    retryable: bool


class RetryConfig(BaseModel):
    """Configuration for retry behavior"""

    max_attempts: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    exponential_base: float = 2.0
    jitter: bool = True

    @field_validator("max_attempts")
    @classmethod
    def validate_max_attempts(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_attempts must be at least 1")
        return v

    @field_validator("base_delay", "max_delay")
    @classmethod
    def validate_delays(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Delay must be positive")
        return v

    @field_validator("exponential_base")
    @classmethod
    def validate_exponential_base(cls, v: float) -> float:
        if v < 1.0:
            raise ValueError("exponential_base must be >= 1.0")
        return v


class CircuitBreakerConfig(BaseModel):
    """Configuration for circuit breaker"""

    failure_threshold: int = 5  # failures before opening
    timeout: int = 60  # seconds before retry attempt
    half_open_max_calls: int = 3  # test calls in HALF_OPEN state

    @field_validator("failure_threshold", "timeout", "half_open_max_calls")
    @classmethod
    def validate_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("Value must be positive")
        return v


# ============================================================================
# Director Diagnosis & Mutations
# ============================================================================


class Diagnosis(BaseModel):
    """Director's diagnosis of SOP performance"""

    primaryWeakness: str
    secondaryWeaknesses: Optional[list[str]] = None
    strengths: Optional[list[str]] = None
    rootCauseAnalysis: str
    recommendations: list[str]
    referencedArtifacts: Optional[list[str]] = None


class MutationType(str, Enum):
    """Types of SOP mutations"""

    PARAM_TWEAK = "param_tweak"
    PROMPT_CHANGE = "prompt_change"
    ADD_ROLE = "add_role"
    REMOVE_ROLE = "remove_role"
    REWIRE_GRAPH = "rewire_graph"
    CROSSOVER = "crossover"
    META_PROMPT_CHANGE = "meta_prompt_change"


class SOPMutation(BaseModel):
    """Proposed mutation to an SOP"""

    parentSopId: str
    parentVersion: str
    newConfig: ProcessConfig
    newPromptGenome: Optional[PromptGenome] = None
    rationale: str
    mutationType: MutationType


class MetaMutation(BaseModel):
    """Mutation to MetaConfig"""

    parentMetaId: str
    parentMetaVersion: str
    newMetaConfig: MetaConfig
    rationale: str


# ============================================================================
# Gene Pool, QD Features & Pareto
# ============================================================================


class QDFeatures(BaseModel):
    """Quality-Diversity features for an SOP"""

    complexity: Optional[float] = None
    diversityEmbedding: Optional[list[float]] = None
    domainFeatures: Optional[dict[str, float]] = None
    featureVersion: Optional[str] = None


class ParetoFlags(BaseModel):
    """Pareto frontier membership flags"""

    isParetoOptimal: bool
    paretoSetId: Optional[str] = None


class QDCell(BaseModel):
    """QD Grid cell assignment"""

    cellId: str
    isCellElite: Optional[bool] = None


class SOPGene(BaseModel):
    """Gene pool entry for an SOP"""

    sopId: str
    version: str
    parent: Optional[dict[str, str]] = None  # {sopId, version}
    promptGenomeId: str
    promptGenomeVersion: str
    configSnapshot: ProcessConfig
    evaluations: list[EvaluationVector]
    aggregatedMetrics: dict[str, AggregatedMetric]  # Full statistical metadata
    qdFeatures: Optional[QDFeatures] = None
    frontierFlags: Optional[ParetoFlags] = None
    qdCell: Optional[QDCell] = None
    tags: Optional[list[str]] = None
    generation: Optional[int] = None  # Generation number for temporal queries (RECENT strategy)
    createdAt: str = Field(default_factory=_utc_now_iso)

    def get_metric_mean(self, metric_id: str) -> float:
        """
        Get mean value for a metric (backward compatibility helper)

        Args:
            metric_id: Metric identifier

        Returns:
            Mean value, or 0.0 if metric not found
        """
        if metric_id not in self.aggregatedMetrics:
            return 0.0
        return self.aggregatedMetrics[metric_id].mean

    def get_metric_ci_width(self, metric_id: str) -> float:
        """
        Get confidence interval width (measure of uncertainty)

        Args:
            metric_id: Metric identifier

        Returns:
            CI width (upper - lower), or inf if CI not available
        """
        if metric_id not in self.aggregatedMetrics:
            return float("inf")

        agg = self.aggregatedMetrics[metric_id]
        if agg.confidenceInterval is None:
            return float("inf")

        lower, upper = agg.confidenceInterval
        return upper - lower

    def get_metric_confidence_interval(
        self, metric_id: str
    ) -> Optional[tuple[float, float]]:
        """
        Get confidence interval for a metric as (lower, upper) tuple.

        Args:
            metric_id: Metric identifier

        Returns:
            Tuple of (lower, upper) bounds, or None if CI not available
        """
        if metric_id not in self.aggregatedMetrics:
            return None

        agg = self.aggregatedMetrics[metric_id]
        return agg.confidenceInterval


class MetaGene(BaseModel):
    """Gene pool entry for MetaConfig"""

    metaId: str
    version: str
    parent: Optional[dict[str, str]] = None
    configSnapshot: MetaConfig
    metaMetrics: dict[str, float]
    createdAt: str = Field(default_factory=_utc_now_iso)


# ============================================================================
# Evolution Job & Task Models
# ============================================================================


class TaskMetadata(BaseModel):
    """Metadata for a task"""

    category: Optional[str] = None
    difficulty: Optional[str] = None
    importance: Optional[float] = 1.0
    source: Optional[str] = None
    tags: Optional[list[str]] = None


class Task(BaseModel):
    """Single evaluation task"""

    id: str
    input: dict[str, Any]
    groundTruth: Optional[dict[str, Any]] = None
    metadata: Optional[TaskMetadata] = None
    weight: float = 1.0


class TaskSet(BaseModel):
    """Collection of tasks for evaluation"""

    id: str
    domain: str
    description: Optional[str] = None
    tasks: list[Task]
    createdAt: str = Field(default_factory=_utc_now_iso)
    version: str


class BudgetLimit(BaseModel):
    """Budget constraints"""

    maxCost: Optional[float] = None
    maxEvaluations: Optional[int] = None
    maxLLMCalls: Optional[int] = None
    maxWallTime: Optional[int] = None  # seconds

    @field_validator("maxCost", "maxEvaluations", "maxLLMCalls", "maxWallTime")
    @classmethod
    def validate_positive(cls, v: Optional[Union[int, float]]) -> Optional[Union[int, float]]:
        if v is not None and v < 0:
            raise ValueError("Budget limits must be positive")
        return v


class BudgetUsage(BaseModel):
    """Current budget usage"""

    cost: float = 0.0
    evaluations: int = 0
    llmCalls: int = 0
    wallTime: int = 0  # seconds


class EvolutionConstraints(BaseModel):
    """Constraints for evolution"""

    safetyMetrics: Optional[list[dict[str, Any]]] = None  # {metricId, minValue}
    budgetLimit: Optional[BudgetLimit] = None
    mandatoryRoles: Optional[list[str]] = None
    maxRoles: Optional[int] = None
    maxEdges: Optional[int] = None
    allowedTools: Optional[list[str]] = None
    disallowedMutationTypes: Optional[list[MutationType]] = None


class SelectionStrategy(str, Enum):
    """Selection strategies for evolution"""

    PARETO = "pareto"
    QD_UNIFORM = "qd_uniform"
    QD_QUALITY_WEIGHTED = "qd_quality_weighted"
    QD_CURIOSITY = "qd_curiosity"  # UCB-based curiosity-driven selection
    RECENT = "recent"  # Momentum-based selection from recent generations
    TOURNAMENT = "tournament"
    HYBRID = "hybrid"  # Composite strategy mixing multiple approaches


# ============================================================================
# Selection Strategy Configuration Models
# ============================================================================


class QDCuriosityConfig(BaseModel):
    """Configuration for QD_CURIOSITY selection strategy (UCB-based exploration)"""

    explorationConstant: float = Field(
        default=1.0,
        ge=0.0,
        description="UCB exploration constant C (higher = more exploration)",
    )
    temperature: float = Field(
        default=1.0,
        gt=0.0,
        description="Softmax temperature for sampling (higher = more uniform)",
    )
    normalizeQuality: bool = Field(
        default=True,
        description="Normalize quality scores to [0,1] before UCB calculation",
    )


class RecentSelectionConfig(BaseModel):
    """Configuration for RECENT selection strategy (momentum-based)"""

    lookbackWindow: int = Field(
        default=3,
        ge=1,
        description="Number of recent generations to consider",
    )
    minQualityThreshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum quality threshold for candidates (weighted_aggregate)",
    )
    samplingMethod: Literal["uniform", "quality_weighted"] = Field(
        default="quality_weighted",
        description="How to sample from recent candidates",
    )
    fallbackOnEmpty: bool = Field(
        default=True,
        description="If no recent genes meet criteria, relax constraints",
    )


class HybridStrategyComponent(BaseModel):
    """Single component of a hybrid selection strategy"""

    strategyType: SelectionStrategy = Field(
        description="Type of selection strategy",
    )
    weight: float = Field(
        ge=0.0,
        le=1.0,
        description="Weight for this strategy (must sum to 1.0 across all components)",
    )
    config: Optional[dict[str, Any]] = Field(
        default=None,
        description="Strategy-specific configuration",
    )


class HybridSelectionConfig(BaseModel):
    """Configuration for HYBRID selection strategy (composite mixing)"""

    # Weight sum tolerance constants (for floating-point comparison)
    _WEIGHT_SUM_TOLERANCE_LOW: ClassVar[float] = 0.99
    _WEIGHT_SUM_TOLERANCE_HIGH: ClassVar[float] = 1.01

    components: list[HybridStrategyComponent] = Field(
        min_length=2,
        description="List of strategies to mix (must have at least 2)",
    )
    deduplication: bool = Field(
        default=True,
        description="Remove duplicate selections across strategies",
    )
    fallbackStrategy: SelectionStrategy = Field(
        default=SelectionStrategy.QD_QUALITY_WEIGHTED,
        description="Strategy to use for filling remaining budget",
    )

    @model_validator(mode="after")
    def validate_weights_sum_to_one(self) -> HybridSelectionConfig:
        """Ensure component weights sum to approximately 1.0"""
        total_weight = sum(comp.weight for comp in self.components)
        if not (self._WEIGHT_SUM_TOLERANCE_LOW <= total_weight <= self._WEIGHT_SUM_TOLERANCE_HIGH):
            raise ValueError(f"Component weights must sum to 1.0 (±0.01), got {total_weight:.4f}")
        return self

    @model_validator(mode="after")
    def validate_no_hybrid_recursion(self) -> HybridSelectionConfig:
        """Prevent HYBRID strategies from containing other HYBRID strategies"""
        for comp in self.components:
            if comp.strategyType == SelectionStrategy.HYBRID:
                raise ValueError(
                    "HYBRID strategy cannot contain other HYBRID strategies (no recursion)"
                )
        return self


class SelectionStrategyConfig(BaseModel):
    """Unified configuration for all selection strategies"""

    strategyType: SelectionStrategy

    # Strategy-specific configs (only one should be set based on strategyType)
    qdCuriosityConfig: Optional[QDCuriosityConfig] = None
    recentConfig: Optional[RecentSelectionConfig] = None
    hybridConfig: Optional[HybridSelectionConfig] = None

    # Generic config for other strategies (PARETO, TOURNAMENT, QD_UNIFORM, QD_QUALITY_WEIGHTED)
    genericConfig: Optional[dict[str, Any]] = None

    @model_validator(mode="after")
    def validate_config_matches_strategy(self) -> SelectionStrategyConfig:
        """Ensure the correct config is set for the strategy type and enforce mutual exclusivity"""
        # Define which config should be set for each strategy type
        config_mapping = {
            SelectionStrategy.QD_CURIOSITY: "qdCuriosityConfig",
            SelectionStrategy.RECENT: "recentConfig",
            SelectionStrategy.HYBRID: "hybridConfig",
            SelectionStrategy.PARETO: "genericConfig",
            SelectionStrategy.TOURNAMENT: "genericConfig",
            SelectionStrategy.QD_UNIFORM: "genericConfig",
            SelectionStrategy.QD_QUALITY_WEIGHTED: "genericConfig",
        }

        expected_config = config_mapping.get(self.strategyType)
        if not expected_config:
            raise ValueError(f"Unknown strategy type: {self.strategyType}")

        # Check mutual exclusivity: only the expected config should be set
        all_configs = {
            "qdCuriosityConfig": self.qdCuriosityConfig,
            "recentConfig": self.recentConfig,
            "hybridConfig": self.hybridConfig,
            "genericConfig": self.genericConfig,
        }

        # Find configs that are set but shouldn't be
        unexpected_configs = [
            name
            for name, value in all_configs.items()
            if name != expected_config and value is not None
        ]

        if unexpected_configs:
            raise ValueError(
                f"Strategy {self.strategyType.value} expects only '{expected_config}' "
                f"to be set, but found: {', '.join(unexpected_configs)}"
            )

        # Ensure required config is present (with defaults where applicable)
        if self.strategyType == SelectionStrategy.QD_CURIOSITY:
            if self.qdCuriosityConfig is None:
                self.qdCuriosityConfig = QDCuriosityConfig()
        elif self.strategyType == SelectionStrategy.RECENT:
            if self.recentConfig is None:
                self.recentConfig = RecentSelectionConfig()
        elif self.strategyType == SelectionStrategy.HYBRID:
            if self.hybridConfig is None:
                raise ValueError("HYBRID strategy requires hybridConfig to be specified")

        return self


class EvolutionPhase(BaseModel):
    """Single phase of evolution"""

    name: str
    allowedMutationTypes: list[MutationType]
    selectionStrategy: SelectionStrategy
    parentsPerGeneration: int
    maxGenerations: int
    budgetPerGeneration: Optional[BudgetLimit] = None
    convergence: Optional[dict[str, Any]] = None  # {patience, improvementThreshold}

    # Optional detailed strategy configuration
    selectionStrategyConfig: Optional[SelectionStrategyConfig] = None


class EvolutionJobStatus(str, Enum):
    """Status of evolution job"""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StopConditions(BaseModel):
    """Conditions to stop evolution"""

    maxTotalGenerations: int
    maxBudget: BudgetLimit
    targetQuality: Optional[float] = None
    minDiversity: Optional[float] = None


class EvolutionJob(BaseModel):
    """Evolution job configuration"""

    # Validation tolerance for weight sum (allows floating-point imprecision)
    _WEIGHT_SUM_TOLERANCE_LOW: ClassVar[float] = 0.99
    _WEIGHT_SUM_TOLERANCE_HIGH: ClassVar[float] = 1.01

    id: str
    domain: str
    baseSops: list[dict[str, str]]  # {sopId, sopVersion, promptGenomeId, promptGenomeVersion}
    taskSet: TaskSet
    metricsToOptimize: list[str]
    qualityScoreWeights: dict[str, float]
    constraints: EvolutionConstraints
    phases: list[EvolutionPhase]
    status: EvolutionJobStatus
    currentPhaseIndex: int = 0
    currentGeneration: int = 0
    budgetUsed: BudgetUsage = Field(default_factory=BudgetUsage)
    bestSopSoFar: Optional[dict[str, Any]] = None
    createdBy: Optional[str] = None
    createdAt: str = Field(default_factory=_utc_now_iso)
    startedAt: Optional[str] = None
    completedAt: Optional[str] = None
    estimatedCompletionAt: Optional[str] = None
    config: Optional[dict[str, Any]] = None  # {qdGridConfig, aggregationConfig, stopConditions}

    @field_validator("currentPhaseIndex")
    @classmethod
    def validate_current_phase_index(cls, v: int) -> int:
        if v < 0:
            raise ValueError("currentPhaseIndex must be >= 0")
        return v

    @field_validator("metricsToOptimize")
    @classmethod
    def validate_metrics_not_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("metricsToOptimize cannot be empty")
        return v

    @field_validator("qualityScoreWeights")
    @classmethod
    def validate_weights_sum_to_one(cls, v: dict[str, float]) -> dict[str, float]:
        total = sum(v.values())
        if not (cls._WEIGHT_SUM_TOLERANCE_LOW <= total <= cls._WEIGHT_SUM_TOLERANCE_HIGH):
            raise ValueError(f"Quality score weights must sum to 1.0, got {total}")
        return v

    @field_validator("phases")
    @classmethod
    def validate_phases_not_empty(cls, v: list[EvolutionPhase]) -> list[EvolutionPhase]:
        if not v:
            raise ValueError("Must have at least one evolution phase")
        return v

    @model_validator(mode="after")
    def validate_current_phase_in_range(self) -> EvolutionJob:
        if self.currentPhaseIndex < 0 or self.currentPhaseIndex >= len(self.phases):
            raise ValueError(
                f"currentPhaseIndex {self.currentPhaseIndex} out of range for {len(self.phases)} phases"
            )
        return self


# ============================================================================
# Domain Package Models
# ============================================================================


class DomainDependency(BaseModel):
    """Dependency on another domain package"""

    packageId: str
    version: str
    reason: Optional[str] = None


class DomainConfig(BaseModel):
    """Domain-specific configuration"""

    defaultEvolutionConfig: Optional[dict[str, Any]] = None
    aggregationConfig: Optional[dict[str, Any]] = None
    domainFeatureExtractors: Optional[dict[str, str]] = None
    recommendedConstraints: Optional[EvolutionConstraints] = None


class DomainPackage(BaseModel):
    """Complete domain package"""

    id: str
    name: str
    version: str
    description: Optional[str] = None
    sopTemplates: list[str]
    promptGenomes: list[str]
    metaConfigs: list[str]
    toolConfigs: list[str]
    metricConfigs: list[str]
    evaluationTasks: list[str]
    humanFeedbackProtocols: Optional[list[str]] = None
    customCode: Optional[list[str]] = None
    domainConfig: DomainConfig
    maintainer: Optional[str] = None
    documentation: Optional[str] = None
    exampleUseCases: Optional[list[str]] = None
    tags: Optional[list[str]] = None
    createdAt: str = Field(default_factory=_utc_now_iso)
    updatedAt: str = Field(default_factory=_utc_now_iso)
    dependencies: Optional[list[DomainDependency]] = None

    @field_validator("version")
    @classmethod
    def validate_semver(cls, v: str) -> str:
        pattern = r"^\d+\.\d+\.\d+(-[a-zA-Z0-9.]+)?$"
        if not re.match(pattern, v):
            raise ValueError(f"Invalid semantic version: {v}")
        return v

    @field_validator("sopTemplates", "promptGenomes", "toolConfigs", "metricConfigs")
    @classmethod
    def validate_not_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("Core components cannot be empty")
        return v


# ============================================================================
# User & RBAC Models
# ============================================================================


class ResourceType(str, Enum):
    """Resource types for RBAC"""

    SOP = "sop"
    PROMPT_GENOME = "prompt_genome"
    META_CONFIG = "meta_config"
    DOMAIN = "domain"
    GENE_POOL = "gene_pool"
    EVOLUTION_JOB = "evolution_job"
    TASK_SET = "task_set"
    USER = "user"
    ROLE = "role"


class Action(str, Enum):
    """Actions for RBAC"""

    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    EXECUTE = "execute"
    DEPLOY = "deploy"
    APPROVE = "approve"
    EMERGENCY_STOP = "emergency_stop"


class PermissionScope(BaseModel):
    """Scope for permissions"""

    domainIds: Optional[list[str]] = None
    sopIds: Optional[list[str]] = None
    resourceIds: Optional[list[str]] = None
    attributes: Optional[dict[str, Any]] = None


class PermissionCondition(BaseModel):
    """Conditional permissions"""

    type: str  # "time", "ip", "mfa", "custom"
    params: dict[str, Any]


class Permission(BaseModel):
    """Single permission"""

    resource: ResourceType
    action: Action
    scope: Optional[PermissionScope] = None
    conditions: Optional[list[PermissionCondition]] = None


class Role(BaseModel):
    """RBAC role"""

    id: str
    name: str
    description: Optional[str] = None
    permissions: list[Permission]
    inheritsFrom: Optional[list[str]] = None
    createdAt: str = Field(default_factory=_utc_now_iso)
    updatedAt: str = Field(default_factory=_utc_now_iso)

    @field_validator("permissions")
    @classmethod
    def validate_permissions_not_empty(cls, v: list[Permission]) -> list[Permission]:
        if not v:
            raise ValueError("Role must have at least one permission")
        return v


class UserStatus(str, Enum):
    """User account status"""

    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"


class UserPreferences(BaseModel):
    """User preferences"""

    defaultDomain: Optional[str] = None
    notificationSettings: Optional[dict[str, bool]] = None
    uiSettings: Optional[dict[str, Any]] = None


class User(BaseModel):
    """User account"""

    id: str
    email: str
    name: Optional[str] = None
    authProvider: str = "local"  # "local", "oauth", "saml"
    hashedPassword: Optional[str] = None
    oauthProviderId: Optional[str] = None
    roles: list[str]
    domains: list[str]
    status: UserStatus
    emailVerified: bool = False
    createdAt: str = Field(default_factory=_utc_now_iso)
    updatedAt: str = Field(default_factory=_utc_now_iso)
    lastLoginAt: Optional[str] = None
    preferences: Optional[UserPreferences] = None

    # Rate limiting and quota fields
    tier: str = "free"  # "free", "standard", "premium", "enterprise"
    customLimits: Optional[dict[str, int]] = None  # Override limits for specific user
    quotaExempt: bool = False  # Admin flag to bypass all limits

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        if not re.match(pattern, v):
            raise ValueError(f"Invalid email format: {v}")
        return v

    @field_validator("roles")
    @classmethod
    def validate_roles_not_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("User must have at least one role")
        return v


# ============================================================================
# Validation Models
# ============================================================================


class ValidationError(BaseModel):
    """Validation error details"""

    code: str
    message: str
    field: Optional[str] = None
    severity: str  # "ERROR", "WARNING", "INFO"
    details: Optional[dict[str, Any]] = None


class ValidationResult(BaseModel):
    """Validation result with errors list"""

    valid: bool
    errors: list[ValidationError] = Field(
        default_factory=list  # type: ignore[var-annotated]
    )
    validated_at: str = Field(default_factory=_utc_now_iso)


# ============================================================================
# Deployment & Governance Models
# ============================================================================


class DeploymentState(str, Enum):
    """Deployment lifecycle states"""

    CREATED = "created"  # Deployment request created
    PENDING = "pending"  # Waiting for validation
    VALIDATING = "validating"  # Safety validation in progress
    APPROVED = "approved"  # Validation passed
    DEPLOYING = "deploying"  # Deployment in progress
    DEPLOYED = "deployed"  # Deployment complete
    ACTIVE = "active"  # SOP actively serving requests
    DEGRADED = "degraded"  # Performance degraded below threshold
    EMERGENCY = "emergency"  # Kill-switch activated
    ROLLED_BACK = "rolled_back"  # Rolled back to previous version
    REJECTED = "rejected"  # Validation failed
    DENIED = "denied"  # Manual approval denied
    CANCELED = "canceled"  # Deployment canceled
    EXPIRED = "expired"  # Deployment expired (not activated in time)
    INACTIVE = "inactive"  # Manually deactivated


class SOPDeployment(BaseModel):
    """SOP deployment request and status"""

    id: str
    sop_id: str
    sop_version: str
    environment: str  # "development", "staging", "production"
    status: DeploymentState
    requested_by: str
    created_at: str = Field(default_factory=_utc_now_iso)
    deployed_at: Optional[str] = None
    deactivated_at: Optional[str] = None
    workflow_instance_id: Optional[str] = None
    validation_report: Optional[SafetyValidationReport] = None
    domain_id: str


class DeploymentVersion(BaseModel):
    """Immutable deployment version record"""

    version_id: str
    sop_id: str
    sop_config: ProcessConfig
    environment: str
    deployed_at: str
    deployed_by: str
    baseline_metrics: dict[str, AggregatedMetric]  # Changed from EvaluationVector
    safety_constraints: EvolutionConstraints
    previous_version_id: Optional[str] = None
    is_rollback: bool = False
    status: str  # "active", "superseded", "rolled_back"


class ValidationCheckResult(BaseModel):
    """Result of a single safety validation check"""

    passed: bool
    message: Optional[str] = None
    requires_approval: bool = False
    warnings: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class SafetyValidationReport(BaseModel):
    """Complete safety validation report for deployment"""

    sop_id: str
    sop_version: str
    timestamp: str = Field(default_factory=_utc_now_iso)
    overall_passed: bool
    checks: dict[str, ValidationCheckResult]
    requires_human_review: bool
    approver_notes: Optional[str] = None


class KillSwitchResult(BaseModel):
    """Result of kill-switch activation"""

    success: bool
    kill_switch_id: str
    rollback_success: Optional[bool] = None
    incident_report_id: Optional[str] = None
    error: Optional[str] = None
    message: str


class RollbackResult(BaseModel):
    """Result of rollback operation"""

    success: bool
    rollback_id: Optional[str] = None
    previous_version_id: Optional[str] = None
    new_version_id: Optional[str] = None
    error: Optional[str] = None
    message: Optional[str] = None


# ============================================================================
# Approval Workflow Models
# ============================================================================


class ApprovalType(str, Enum):
    """Types of approval stages"""

    AUTO_VALIDATION = "auto_validation"  # Automated safety checks
    DOMAIN_EXPERT = "domain_expert"  # Domain expert review
    PRODUCTION_APPROVAL = "production_approval"  # Final production sign-off
    SECURITY_REVIEW = "security_review"  # Security team review


class ApprovalStageStatus(str, Enum):
    """Status of an approval stage"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    APPROVED = "approved"
    REJECTED = "rejected"
    SKIPPED = "skipped"
    TIMED_OUT = "timed_out"
    ESCALATED = "escalated"  # Escalated due to timeout


class ApprovalDecision(BaseModel):
    """Individual approval decision by an approver"""

    stage_id: str
    approver_id: str
    decision: Literal["approve", "reject"]
    reason: Optional[str] = None
    timestamp: str = Field(default_factory=_utc_now_iso)
    conditions: Optional[list[str]] = None  # Conditional approval terms


class ApprovalStage(BaseModel):
    """Definition of an approval stage in a workflow"""

    id: str
    name: str
    approval_type: ApprovalType
    required_roles: list[str]  # Roles that can approve this stage
    min_approvers: int = 1  # Minimum approvals needed
    timeout_hours: Optional[int] = None  # Auto-timeout (None = no timeout)
    escalation_roles: list[str] = Field(default_factory=list)  # Escalation chain on timeout
    conditions: Optional[dict[str, Any]] = None  # Conditions to trigger this stage
    order: int  # Execution order (0-indexed)


class ApprovalWorkflow(BaseModel):
    """Approval workflow template defining stages for an environment"""

    id: str
    name: str
    description: Optional[str] = None
    environment: str  # Which environment this workflow applies to
    stages: list[ApprovalStage]
    created_at: str = Field(default_factory=_utc_now_iso)
    active: bool = True

    @field_validator("stages")
    @classmethod
    def validate_stages_not_empty(cls, v: list[ApprovalStage]) -> list[ApprovalStage]:
        if not v:
            raise ValueError("Workflow must have at least one stage")
        return v


class ApprovalWorkflowInstance(BaseModel):
    """Runtime instance of an approval workflow for a specific deployment"""

    id: str
    workflow_id: str
    deployment_id: str
    current_stage_index: int = 0
    stage_statuses: dict[str, ApprovalStageStatus]  # stage_id -> status
    stage_decisions: dict[str, list[ApprovalDecision]]  # stage_id -> list of decisions
    started_at: str = Field(default_factory=_utc_now_iso)
    completed_at: Optional[str] = None
    overall_status: Literal["pending", "approved", "rejected", "timed_out", "escalated"] = (
        "pending"
    )
    escalation_count: int = 0  # Number of times workflow was escalated


# ============================================================================
# Prompt Evolution Models
# ============================================================================


class PromptOptimizationStrategyType(str, Enum):
    """Types of prompt optimization strategies"""

    TEXTGRAD = "textgrad"  # Textual gradient descent with LLM critiques
    EVOPROMPT = "evoprompt"  # Evolutionary algorithms (GA/DE)
    METAPROMPT = "metaprompt"  # LLM meta-analysis for targeted improvements


class FailurePattern(str, Enum):
    """Categories of failure patterns detected in execution traces"""

    HALLUCINATION = "hallucination"  # Made up information
    INCOMPLETE = "incomplete"  # Missing required information
    IRRELEVANT = "irrelevant"  # Off-topic response
    TIMEOUT = "timeout"  # Exceeded time limit
    TOOL_MISUSE = "tool_misuse"  # Incorrect tool usage
    FORMAT_ERROR = "format_error"  # Wrong output format
    REASONING_ERROR = "reasoning_error"  # Logical errors
    CONTEXT_LOSS = "context_loss"  # Lost track of context
    SAFETY_VIOLATION = "safety_violation"  # Safety policy breach


class PromptSectionType(str, Enum):
    """Types of sections within a prompt"""

    ROLE_DEFINITION = "role_definition"  # Who the agent is
    OBJECTIVE = "objective"  # What the agent should accomplish
    INSTRUCTIONS = "instructions"  # Step-by-step guidance
    CONSTRAINTS = "constraints"  # Rules and limitations (often immutable)
    EXAMPLES = "examples"  # Few-shot examples
    FORMAT = "format"  # Output format specification
    CONTEXT = "context"  # Background information
    FALLBACK = "fallback"  # Unstructured/legacy content


class PromptFeedback(BaseModel):
    """Structured feedback for a prompt section from LLM critic"""

    role_id: str
    section_id: str
    section_content: str
    critique: str  # Detailed analysis of what went wrong
    failure_pattern: Optional[FailurePattern] = None
    suggested_improvement: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)


class PromptSection(BaseModel):
    """Parsed section of a prompt"""

    id: str
    content: str
    is_mutable: bool = True  # False for safety/constraint sections
    section_type: PromptSectionType
    parent_role_id: str


class ParsedPrompt(BaseModel):
    """Prompt parsed into structured sections"""

    role_id: str
    original_content: str
    sections: list[PromptSection]
    immutable_constraints: list[str] = Field(default_factory=list)


class PromptEvolutionResult(BaseModel):
    """Result from applying a prompt optimization strategy"""

    new_prompt_genome: PromptGenome
    changes_made: list[dict[str, Any]]  # [{role_id, section_id, old_content, new_content}]
    rationale: str
    strategy_metadata: dict[str, Any] = Field(default_factory=dict)


class SectionMutation(BaseModel):
    """Individual mutation to a prompt section."""

    section_id: str = Field(description="ID of the section being mutated")
    role_id: str = Field(description="ID of the role whose prompt contains this section")
    original_content: str = Field(description="Original section content before mutation")
    mutated_content: str = Field(description="New section content after mutation")
    mutation_type: str = Field(description="Type of mutation: replace, append, prepend, refine")
    rationale: str = Field(description="Reason for the mutation")
    source_strategy: PromptOptimizationStrategyType = Field(description="Strategy that generated this mutation")
    confidence: float = Field(ge=0.0, le=1.0, default=0.5, description="Confidence in mutation quality")


class SectionMutationBatch(BaseModel):
    """Collection of mutations to apply atomically."""

    batch_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    prompt_genome_id: str
    prompt_genome_version: str
    mutations: list[SectionMutation] = Field(default_factory=list)
    created_at: str = Field(default_factory=_utc_now_iso)
    applied: bool = False


class FeedbackArtifact(BaseModel):
    """Structured artifact extracted from evaluation results."""

    source_type: str = Field(description="Source: llm_judge, trace, metric, user")
    role_id: str = Field(description="Role this feedback applies to")
    metric_id: Optional[str] = Field(default=None, description="Metric ID if from evaluation")
    critique: str = Field(description="Description of the issue")
    severity: float = Field(ge=0.0, le=1.0, default=0.5, description="Severity 0-1")
    failure_pattern: Optional[FailurePattern] = Field(default=None, description="Classified failure pattern")
    suggested_fix: Optional[str] = Field(default=None, description="Suggested improvement")
    trace_ref: Optional[str] = Field(default=None, description="Reference to execution trace")


class ConstraintViolation(BaseModel):
    """Record of a constraint violation detected during mutation."""

    constraint_type: str = Field(description="Type: must_not_change, max_length, keyword_required")
    violation_description: str = Field(description="Human-readable description")
    section_id: Optional[str] = Field(default=None, description="Section ID if applicable")
    role_id: str = Field(description="Role ID where violation occurred")
    severity: str = Field(default="error", description="error or warning")


class FeedbackInjectionConfig(BaseModel):
    """Configuration for runtime feedback injection."""

    enabled: bool = Field(default=True)
    injection_position: str = Field(default="append", description="prepend, append, or section name")
    max_feedback_items: int = Field(default=3, ge=1, le=10)
    include_failure_patterns: bool = Field(default=True)
    include_suggestions: bool = Field(default=True)


class PromptEvolutionOrchestratorConfig(BaseModel):
    """Configuration for the orchestration workflow."""

    default_strategy: PromptOptimizationStrategyType = Field(default=PromptOptimizationStrategyType.EVOPROMPT)
    enable_section_parsing: bool = Field(default=True)
    enable_feedback_injection: bool = Field(default=True)
    max_mutations_per_role: int = Field(default=3, ge=1, le=10)
    constraint_validation_mode: str = Field(default="strict", description="strict, warn, or disabled")
    fallback_to_full_prompt: bool = Field(default=True)


class TextGradConfig(BaseModel):
    """Configuration for TextGrad optimization strategy"""

    learning_rate: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="How much to incorporate gradient suggestions (0=ignore, 1=full)",
    )
    backprop_depth: int = Field(
        default=2,
        ge=1,
        description="How many upstream roles to backpropagate gradients through",
    )
    gradient_aggregation: str = Field(
        default="mean",
        description="How to aggregate gradients across traces: mean, weighted, majority",
    )
    model: str = Field(default="gpt-4", description="LLM model for gradient computation")
    temperature: float = Field(
        default=0.5,
        ge=0.0,
        le=2.0,
        description="Temperature for LLM gradient generation",
    )


class EvoPromptConfig(BaseModel):
    """Configuration for EvoPrompt optimization strategy"""

    population_size: int = Field(
        default=8,
        ge=2,
        description="Number of prompt variants in population",
    )
    mutation_rate: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Probability of mutation per generation",
    )
    crossover_rate: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Probability of crossover between parents",
    )
    selection_method: str = Field(
        default="tournament",
        description="Parent selection: tournament, roulette",
    )
    tournament_size: int = Field(
        default=3,
        ge=2,
        description="Tournament size for selection",
    )
    algorithm: str = Field(
        default="GA",
        description="Evolutionary algorithm: GA (Genetic Algorithm), DE (Differential Evolution)",
    )
    model: str = Field(default="gpt-4", description="LLM model for crossover/mutation")


class MetaPromptConfig(BaseModel):
    """Configuration for MetaPrompt optimization strategy"""

    analysis_depth: str = Field(
        default="detailed",
        description="Analysis depth: quick, detailed, exhaustive",
    )
    improvement_count: int = Field(
        default=3,
        ge=1,
        description="Number of improvement alternatives to generate",
    )
    focus_on_failures: bool = Field(
        default=True,
        description="Prioritize addressing detected failure patterns",
    )
    model: str = Field(default="gpt-4", description="LLM model for meta-analysis")
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Temperature for improvement generation",
    )
