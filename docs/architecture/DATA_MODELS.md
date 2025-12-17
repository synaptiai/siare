# Complete Data Models Specification

**Version:** 1.1
**Status:** Production Reference
**Purpose:** Complete data model definitions for all SIARE entities, including validation rules and relationships

> **Open-Core Architecture**: SIARE follows an open-core model:
> - **siare** (this package): Core models for evolution, execution, evaluation, prompt optimization
> - **siare-cloud** (enterprise): User/RBAC models, deployment/approval models, audit models
>
> Sections marked with 🔒 describe enterprise models not included in this open-source package.

---

## 1. Overview

This document defines all data models for SIARE, including:
1. **Core models** (ProcessConfig, PromptGenome, MetaConfig, etc.) - defined in ARCHITECTURE.md
2. **Evolution models** (EvolutionJob, TaskSet, constraints, phases)
3. **Domain models** (DomainPackage, RoleTemplate, RoleLibrary)
4. **User & RBAC models** 🔒 (User, Role, Permission, Authorization) - *Enterprise*
5. **Deployment models** 🔒 (DeploymentRequest, ApprovalStage, ApprovalWorkflow) - *Enterprise*
6. **Prompt evolution models** (PromptEvolutionConfig, PromptSection, PromptFeedback)
7. **Selection strategy models** (SelectionStrategyConfig, strategy-specific params)
8. **Validation rules** for all models
9. **Cross-model relationships** and referential integrity

---

## 2. Evolution Job Models

### 2.1 EvolutionJob

```typescript
interface EvolutionJob {
  id: string;                      // Unique job ID
  domain: string;                  // Domain package ID

  // Initial configuration
  baseSops: Array<{                // Starting SOPs to evolve from
    sopId: string;
    sopVersion: string;
    promptGenomeId: string;
    promptGenomeVersion: string;
  }>;

  taskSet: TaskSet;                // Tasks to evaluate on

  // Optimization configuration
  metricsToOptimize: string[];     // MetricConfig IDs
  qualityScoreWeights: Record<string, number>; // For single quality score

  // Constraints
  constraints: EvolutionConstraints;

  // Phase configuration
  phases: EvolutionPhase[];

  // State
  status: EvolutionJobStatus;
  currentPhaseIndex: number;
  currentGeneration: number;

  // Results tracking
  budgetUsed: BudgetUsage;
  bestSopSoFar?: {
    sopId: string;
    sopVersion: string;
    qualityScore: number;
    generation: number;
  };

  // Metadata
  createdBy: string;               // User ID
  createdAt: string;
  startedAt?: string;
  completedAt?: string;
  estimatedCompletionAt?: string;

  // Configuration
  config: {
    qdGridConfig: QDGridConfig;
    aggregationConfig: MetricAggregationConfig;
    stopConditions: StopConditions;
  };
}

enum EvolutionJobStatus {
  PENDING = "pending",
  RUNNING = "running",
  PAUSED = "paused",
  COMPLETED = "completed",
  FAILED = "failed",
  CANCELLED = "cancelled"
}

interface EvolutionConstraints {
  // Safety constraints (hard limits)
  safetyMetrics?: Array<{
    metricId: string;
    minValue: number;              // Must exceed this
  }>;

  // Budget constraints
  budgetLimit?: BudgetLimit;

  // Structural constraints
  mandatoryRoles?: string[];       // Roles that cannot be removed
  maxRoles?: number;               // Maximum roles in SOP
  maxEdges?: number;               // Maximum edges in graph
  allowedTools?: string[];         // Whitelist of allowed tools

  // Mutation constraints
  disallowedMutationTypes?: MutationType[];
}

interface BudgetLimit {
  maxCost?: number;                // In USD
  maxEvaluations?: number;         // Total evaluation runs
  maxLLMCalls?: number;            // Total LLM API calls
  maxWallTime?: number;            // In seconds
}

interface BudgetUsage {
  cost: number;                    // Current spend in USD
  evaluations: number;             // Evaluations run so far
  llmCalls: number;                // LLM calls made
  wallTime: number;                // Elapsed time in seconds
}

interface EvolutionPhase {
  name: string;
  allowedMutationTypes: MutationType[];
  selectionStrategy: SelectionStrategy;
  parentsPerGeneration: number;
  maxGenerations: number;
  budgetPerGeneration?: BudgetLimit;

  // Convergence criteria for this phase
  convergence?: {
    patience: number;              // Generations without improvement
    improvementThreshold: number;  // Minimum improvement to count as progress
  };
}

interface StopConditions {
  maxTotalGenerations: number;
  maxBudget: BudgetLimit;
  targetQuality?: number;          // Stop if quality exceeds this
  minDiversity?: number;           // Stop if diversity drops below this
}
```

### 2.2 TaskSet

```typescript
interface TaskSet {
  id: string;
  domain: string;
  description?: string;

  tasks: Task[];

  // Metadata
  createdAt: string;
  version: string;
}

interface Task {
  id: string;
  input: Record<string, any>;      // Task-specific input fields
  groundTruth?: Record<string, any>; // For programmatic metrics
  metadata?: TaskMetadata;
}

interface TaskMetadata {
  category?: string;               // For stratified aggregation
  difficulty?: "easy" | "medium" | "hard";
  importance?: number;             // Weight for aggregation (default: 1.0)
  source?: string;                 // Where task came from
  tags?: string[];
  [key: string]: any;              // Extensible
}
```

### 2.3 Validation Rules for EvolutionJob

```python
from pydantic import BaseModel, validator, root_validator
from typing import List, Dict, Optional
from enum import Enum

class EvolutionJobValidation(BaseModel):
    id: str
    domain: str
    baseSops: List[Dict]
    taskSet: Dict
    metricsToOptimize: List[str]
    qualityScoreWeights: Dict[str, float]
    constraints: Dict
    phases: List[Dict]
    status: str
    currentPhaseIndex: int
    currentGeneration: int

    @validator('metricsToOptimize')
    def validate_metrics_not_empty(cls, v):
        if not v:
            raise ValueError("metricsToOptimize cannot be empty")
        return v

    @validator('qualityScoreWeights')
    def validate_weights_sum_to_one(cls, v):
        total = sum(v.values())
        if not (0.99 <= total <= 1.01):  # Allow small floating point error
            raise ValueError(f"Quality score weights must sum to 1.0, got {total}")
        return v

    @validator('phases')
    def validate_phases_not_empty(cls, v):
        if not v:
            raise ValueError("Must have at least one evolution phase")
        return v

    @root_validator
    def validate_quality_weights_match_metrics(cls, values):
        """Ensure quality score weights reference valid metrics"""
        metrics = set(values.get('metricsToOptimize', []))
        weights = set(values.get('qualityScoreWeights', {}).keys())

        if not weights.issubset(metrics):
            extra = weights - metrics
            raise ValueError(f"Quality weights reference undefined metrics: {extra}")

        return values

    @root_validator
    def validate_current_phase_in_range(cls, values):
        """Ensure currentPhaseIndex is valid"""
        current_idx = values.get('currentPhaseIndex', 0)
        phases = values.get('phases', [])

        if current_idx < 0 or current_idx >= len(phases):
            raise ValueError(f"currentPhaseIndex {current_idx} out of range for {len(phases)} phases")

        return values

    @validator('constraints')
    def validate_constraints(cls, v):
        """Validate constraint structure"""
        if 'budgetLimit' in v:
            budget = v['budgetLimit']
            if 'maxCost' in budget and budget['maxCost'] <= 0:
                raise ValueError("maxCost must be positive")
            if 'maxEvaluations' in budget and budget['maxEvaluations'] <= 0:
                raise ValueError("maxEvaluations must be positive")

        if 'safetyMetrics' in v:
            for safety in v['safetyMetrics']:
                if 'metricId' not in safety or 'minValue' not in safety:
                    raise ValueError("safetyMetrics must have metricId and minValue")
                if not (0 <= safety['minValue'] <= 1):
                    raise ValueError("safety minValue must be in [0, 1]")

        return v
```

---

## 3. Domain Package Models

### 3.1 DomainPackage

```typescript
interface DomainPackage {
  id: string;                      // Unique domain package ID
  name: string;                    // Human-readable name
  version: string;                 // Semantic version (e.g., "1.0.0")
  description?: string;

  // Core components
  sopTemplates: SopTemplateRef[];
  promptGenomes: PromptGenomeRef[];
  metaConfigs: MetaConfigRef[];
  toolConfigs: ToolConfigRef[];
  metricConfigs: MetricConfigRef[];

  // Evaluation tasks
  evaluationTasks: TaskSetRef[];

  // Optional components
  humanFeedbackProtocols?: HumanFeedbackProtocolRef[];
  customCode?: CustomCodeRef[];    // Domain-specific evaluators, tools

  // Domain-specific configuration
  domainConfig: DomainConfig;

  // Metadata
  maintainer?: string;
  documentation?: string;          // URL or markdown
  exampleUseCases?: string[];
  tags?: string[];

  createdAt: string;
  updatedAt: string;

  // Dependencies
  dependencies?: DomainDependency[];
}

type SopTemplateRef = string;      // SOP ID
type PromptGenomeRef = string;     // PromptGenome ID
type MetaConfigRef = string;       // MetaConfig ID
type ToolConfigRef = string;       // ToolConfig ID
type MetricConfigRef = string;     // MetricConfig ID
type TaskSetRef = string;          // TaskSet ID
type HumanFeedbackProtocolRef = string;
type CustomCodeRef = string;       // Python module path

interface DomainConfig {
  // Default evolution settings for this domain
  defaultEvolutionConfig?: {
    phases: EvolutionPhase[];
    qdGridConfig: QDGridConfig;
    qualityScoreWeights: Record<string, number>;
  };

  // Default aggregation settings
  aggregationConfig?: MetricAggregationConfig;

  // Domain-specific features for QD Grid
  domainFeatureExtractors?: Record<string, string>; // name -> Python function path

  // Recommended constraints
  recommendedConstraints?: EvolutionConstraints;
}

interface DomainDependency {
  packageId: string;
  version: string;                 // Semver constraint (e.g., "^1.0.0")
  reason?: string;                 // Why dependency is needed
}
```

### 3.2 RoleTemplate (for add_role mutations)

```typescript
interface RoleTemplate {
  id: string;
  name: string;                    // "critic", "fact_checker", "summarizer", etc.
  description: string;

  // Default configuration
  defaultModel: string;
  tools: string[];                 // ToolConfig IDs
  systemPrompt: string;

  // I/O specifications
  defaultInputs?: Array<{
    from: string;                  // Role ID or "user_input"
    fields?: string[];
  }>;
  defaultOutputs?: string[];

  defaultParams?: Record<string, any>;

  // Constraints
  constraints?: PromptConstraints;

  // Metadata
  category?: string;               // "analysis", "critique", "retrieval", "synthesis"
  complexity?: number;             // Estimated complexity (for QD features)
}

interface RoleLibrary {
  domain: string;
  templates: Record<string, RoleTemplate>;  // name -> template
}
```

### 3.3 Validation Rules for DomainPackage

```python
class DomainPackageValidation(BaseModel):
    id: str
    name: str
    version: str
    sopTemplates: List[str]
    promptGenomes: List[str]
    metaConfigs: List[str]
    toolConfigs: List[str]
    metricConfigs: List[str]
    evaluationTasks: List[str]
    domainConfig: Dict

    @validator('version')
    def validate_semver(cls, v):
        """Validate semantic version format"""
        import re
        pattern = r'^\d+\.\d+\.\d+(-[a-zA-Z0-9.]+)?$'
        if not re.match(pattern, v):
            raise ValueError(f"Invalid semantic version: {v}")
        return v

    @validator('sopTemplates', 'promptGenomes', 'toolConfigs', 'metricConfigs')
    def validate_not_empty(cls, v, field):
        """Core components cannot be empty"""
        if not v:
            raise ValueError(f"{field.name} cannot be empty")
        return v

    @validator('evaluationTasks')
    def validate_has_tasks(cls, v):
        """Must have at least one evaluation task set"""
        if not v:
            raise ValueError("Domain package must have at least one evaluation task set")
        return v

    @root_validator
    def validate_domain_config(cls, values):
        """Validate domain-specific configuration"""
        domain_config = values.get('domainConfig', {})

        if 'defaultEvolutionConfig' in domain_config:
            evo_config = domain_config['defaultEvolutionConfig']

            # Validate phases
            if 'phases' in evo_config and not evo_config['phases']:
                raise ValueError("defaultEvolutionConfig.phases cannot be empty")

            # Validate quality score weights
            if 'qualityScoreWeights' in evo_config:
                total = sum(evo_config['qualityScoreWeights'].values())
                if not (0.99 <= total <= 1.01):
                    raise ValueError("qualityScoreWeights must sum to 1.0")

        return values
```

---

## 4. User & RBAC Models 🔒

> **Enterprise Feature**: User management and RBAC are available in siare-cloud.

### 4.1 User

```typescript
interface User {
  id: string;                      // Unique user ID (UUID)
  email: string;
  name?: string;

  // Authentication
  authProvider: "local" | "oauth" | "saml";
  hashedPassword?: string;         // Only for local auth
  oauthProviderId?: string;

  // Authorization
  roles: string[];                 // Role IDs
  domains: string[];               // Domain package IDs user can access

  // Status
  status: "active" | "inactive" | "suspended";
  emailVerified: boolean;

  // Metadata
  createdAt: string;
  updatedAt: string;
  lastLoginAt?: string;

  // Preferences
  preferences?: UserPreferences;
}

interface UserPreferences {
  defaultDomain?: string;
  notificationSettings?: {
    emailOnJobComplete: boolean;
    emailOnJobFailed: boolean;
  };
  uiSettings?: Record<string, any>;
}
```

### 4.2 Role & Permission

```typescript
interface Role {
  id: string;
  name: string;                    // "admin", "domain_owner", "viewer", etc.
  description?: string;

  permissions: Permission[];

  // Inheritance
  inheritsFrom?: string[];         // Other role IDs

  createdAt: string;
  updatedAt: string;
}

interface Permission {
  resource: ResourceType;
  action: Action;
  scope?: PermissionScope;

  // Conditional permissions
  conditions?: PermissionCondition[];
}

enum ResourceType {
  SOP = "sop",
  PROMPT_GENOME = "prompt_genome",
  META_CONFIG = "meta_config",
  DOMAIN = "domain",
  GENE_POOL = "gene_pool",
  EVOLUTION_JOB = "evolution_job",
  TASK_SET = "task_set",
  USER = "user",
  ROLE = "role"
}

enum Action {
  READ = "read",
  WRITE = "write",
  DELETE = "delete",
  EXECUTE = "execute",            // For running jobs
  DEPLOY = "deploy",              // For deploying SOPs to production
  APPROVE = "approve"             // For human-in-the-loop approvals
}

interface PermissionScope {
  // Scope to specific resources
  domainIds?: string[];            // Only these domains
  sopIds?: string[];               // Only these SOPs
  resourceIds?: string[];          // Generic resource filter

  // Attribute-based access control (ABAC)
  attributes?: Record<string, any>;
}

interface PermissionCondition {
  type: "time" | "ip" | "mfa" | "custom";
  params: Record<string, any>;
}

// Predefined roles
const PREDEFINED_ROLES: Record<string, Role> = {
  admin: {
    id: "role_admin",
    name: "Administrator",
    description: "Full system access",
    permissions: [
      { resource: ResourceType.SOP, action: Action.READ },
      { resource: ResourceType.SOP, action: Action.WRITE },
      { resource: ResourceType.SOP, action: Action.DELETE },
      { resource: ResourceType.SOP, action: Action.DEPLOY },
      { resource: ResourceType.DOMAIN, action: Action.READ },
      { resource: ResourceType.DOMAIN, action: Action.WRITE },
      { resource: ResourceType.EVOLUTION_JOB, action: Action.EXECUTE },
      { resource: ResourceType.USER, action: Action.READ },
      { resource: ResourceType.USER, action: Action.WRITE },
      { resource: ResourceType.ROLE, action: Action.READ },
      { resource: ResourceType.ROLE, action: Action.WRITE }
    ],
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString()
  },

  domain_owner: {
    id: "role_domain_owner",
    name: "Domain Owner",
    description: "Full access to specific domains",
    permissions: [
      {
        resource: ResourceType.SOP,
        action: Action.READ,
        scope: { domainIds: ["$user.domains"] }  // Scoped to user's domains
      },
      {
        resource: ResourceType.SOP,
        action: Action.WRITE,
        scope: { domainIds: ["$user.domains"] }
      },
      {
        resource: ResourceType.EVOLUTION_JOB,
        action: Action.EXECUTE,
        scope: { domainIds: ["$user.domains"] }
      },
      {
        resource: ResourceType.SOP,
        action: Action.DEPLOY,
        scope: { domainIds: ["$user.domains"] }
      }
    ],
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString()
  },

  researcher: {
    id: "role_researcher",
    name: "Researcher",
    description: "Can run evolution jobs but not deploy",
    permissions: [
      {
        resource: ResourceType.SOP,
        action: Action.READ,
        scope: { domainIds: ["$user.domains"] }
      },
      {
        resource: ResourceType.SOP,
        action: Action.WRITE,
        scope: { domainIds: ["$user.domains"] }
      },
      {
        resource: ResourceType.EVOLUTION_JOB,
        action: Action.EXECUTE,
        scope: { domainIds: ["$user.domains"] }
      }
      // Note: No DEPLOY permission
    ],
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString()
  },

  viewer: {
    id: "role_viewer",
    name: "Viewer",
    description: "Read-only access",
    permissions: [
      {
        resource: ResourceType.SOP,
        action: Action.READ,
        scope: { domainIds: ["$user.domains"] }
      },
      {
        resource: ResourceType.GENE_POOL,
        action: Action.READ,
        scope: { domainIds: ["$user.domains"] }
      },
      {
        resource: ResourceType.EVOLUTION_JOB,
        action: Action.READ,
        scope: { domainIds: ["$user.domains"] }
      }
    ],
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString()
  }
};
```

### 4.3 Authorization Logic

```python
class AuthorizationService:
    """Check if user has permission to perform action"""

    def __init__(self, user_repo, role_repo):
        self.user_repo = user_repo
        self.role_repo = role_repo

    def check_permission(
        self,
        user: User,
        resource_type: ResourceType,
        action: Action,
        resource_id: Optional[str] = None,
        resource_metadata: Optional[Dict] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if user has permission

        Returns:
            (has_permission, denial_reason)
        """
        # Get user's roles
        roles = [self.role_repo.get(role_id) for role_id in user.roles]

        # Check each role's permissions
        for role in roles:
            for permission in role.permissions:
                if permission.resource != resource_type:
                    continue
                if permission.action != action:
                    continue

                # Check scope
                if permission.scope:
                    if not self._check_scope(user, permission.scope, resource_id, resource_metadata):
                        continue

                # Check conditions
                if permission.conditions:
                    if not self._check_conditions(user, permission.conditions):
                        continue

                # Permission granted!
                return True, None

        # No matching permission found
        return False, f"User {user.id} lacks permission: {action} on {resource_type}"

    def _check_scope(
        self,
        user: User,
        scope: PermissionScope,
        resource_id: Optional[str],
        resource_metadata: Optional[Dict]
    ) -> bool:
        """Check if resource matches permission scope"""

        # Domain-based scoping
        if scope.domainIds:
            # Special variable: $user.domains
            allowed_domains = set()
            for domain_pattern in scope.domainIds:
                if domain_pattern == "$user.domains":
                    allowed_domains.update(user.domains)
                else:
                    allowed_domains.add(domain_pattern)

            # Check if resource's domain is in allowed set
            if resource_metadata and "domain" in resource_metadata:
                resource_domain = resource_metadata["domain"]
                if resource_domain not in allowed_domains:
                    return False

        # Resource ID scoping
        if scope.resourceIds and resource_id:
            if resource_id not in scope.resourceIds:
                return False

        # Attribute-based scoping
        if scope.attributes and resource_metadata:
            for attr_key, attr_value in scope.attributes.items():
                if resource_metadata.get(attr_key) != attr_value:
                    return False

        return True

    def _check_conditions(self, user: User, conditions: List[PermissionCondition]) -> bool:
        """Check if conditional permissions are satisfied"""
        for condition in conditions:
            if condition.type == "mfa":
                # Check if user has MFA enabled
                # (Implementation depends on auth system)
                pass

            elif condition.type == "time":
                # Check if current time is within allowed window
                import datetime
                now = datetime.datetime.utcnow()
                # Parse time window from condition.params
                pass

            # Add other condition types as needed

        return True  # All conditions satisfied
```

### 4.4 Validation Rules for User & Role

```python
class UserValidation(BaseModel):
    id: str
    email: str
    roles: List[str]
    domains: List[str]
    status: str
    emailVerified: bool

    @validator('email')
    def validate_email(cls, v):
        """Validate email format"""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(pattern, v):
            raise ValueError(f"Invalid email format: {v}")
        return v

    @validator('status')
    def validate_status(cls, v):
        """Validate status is valid"""
        valid_statuses = ["active", "inactive", "suspended"]
        if v not in valid_statuses:
            raise ValueError(f"Invalid status: {v}. Must be one of {valid_statuses}")
        return v

    @validator('roles')
    def validate_roles_not_empty(cls, v):
        """User must have at least one role"""
        if not v:
            raise ValueError("User must have at least one role")
        return v


class RoleValidation(BaseModel):
    id: str
    name: str
    permissions: List[Dict]

    @validator('permissions')
    def validate_permissions_not_empty(cls, v):
        """Role must have at least one permission"""
        if not v:
            raise ValueError("Role must have at least one permission")
        return v

    @validator('permissions')
    def validate_permission_structure(cls, v):
        """Validate each permission has required fields"""
        for perm in v:
            if 'resource' not in perm:
                raise ValueError("Permission missing 'resource' field")
            if 'action' not in perm:
                raise ValueError("Permission missing 'action' field")

            # Validate resource and action are valid enum values
            valid_resources = ["sop", "prompt_genome", "meta_config", "domain", "gene_pool",
                             "evolution_job", "task_set", "user", "role"]
            valid_actions = ["read", "write", "delete", "execute", "deploy", "approve"]

            if perm['resource'] not in valid_resources:
                raise ValueError(f"Invalid resource: {perm['resource']}")
            if perm['action'] not in valid_actions:
                raise ValueError(f"Invalid action: {perm['action']}")

        return v
```

---

## 5. Deployment & Approval Models 🔒

> **Enterprise Feature**: Deployment workflows and approval chains are available in siare-cloud.

### 5.1 DeploymentRequest

```typescript
interface DeploymentRequest {
  id: string;                      // Unique deployment ID
  sopId: string;                   // SOP being deployed
  sopVersion: string;              // Specific version

  // Target environment
  environment: "development" | "staging" | "production";
  targetConfig?: Record<string, any>; // Environment-specific config

  // State machine
  status: DeploymentStatus;

  // Approval workflow
  approvalChain: ApprovalStage[];
  currentStageIndex: number;

  // Metadata
  requestedBy: string;             // User ID
  requestedAt: string;
  approvedAt?: string;
  deployedAt?: string;

  // Audit
  auditLog: DeploymentAuditEntry[];
}

enum DeploymentStatus {
  CREATED = "created",
  PENDING = "pending",
  VALIDATING = "validating",
  APPROVED = "approved",
  DEPLOYING = "deploying",
  DEPLOYED = "deployed",
  ACTIVE = "active",
  REJECTED = "rejected",
  DENIED = "denied",
  ROLLED_BACK = "rolled_back"
}

interface DeploymentAuditEntry {
  timestamp: string;
  action: string;
  actor: string;                   // User ID or "system"
  details?: Record<string, any>;
}
```

### 5.2 ApprovalStage

```typescript
interface ApprovalStage {
  id: string;
  name: string;
  type: ApprovalType;

  // Configuration
  requiredRoles: string[];         // Role IDs that can approve
  quorum?: number;                 // For MULTI: how many approvers needed
  timeout?: number;                // Seconds before escalation

  // State
  status: "pending" | "approved" | "rejected" | "skipped";
  decisions: ApprovalDecision[];
}

enum ApprovalType {
  AUTO = "auto",                   // Automated validation
  SINGLE = "single",               // One approver required
  MULTI = "multi",                 // Multiple approvers (quorum)
  UNANIMOUS = "unanimous"          // All required approvers
}

interface ApprovalDecision {
  approver: string;                // User ID
  decision: "approve" | "reject" | "abstain";
  comment?: string;
  conditions?: string[];           // Conditional approval
  timestamp: string;
}

interface ApprovalWorkflowConfig {
  environment: string;
  stages: ApprovalStageConfig[];
}

interface ApprovalStageConfig {
  name: string;
  type: ApprovalType;
  requiredRoles: string[];
  quorum?: number;
  timeout?: number;
  escalationPolicy?: {
    escalateTo: string[];          // User IDs
    afterSeconds: number;
  };
}
```

### 5.3 Validation Rules for Deployment

```python
class DeploymentRequestValidation(BaseModel):
    id: str
    sopId: str
    sopVersion: str
    environment: str
    status: str
    approvalChain: List[Dict]
    currentStageIndex: int
    requestedBy: str
    requestedAt: str

    @validator('environment')
    def validate_environment(cls, v):
        valid_envs = ["development", "staging", "production"]
        if v not in valid_envs:
            raise ValueError(f"Invalid environment: {v}")
        return v

    @validator('approvalChain')
    def validate_approval_chain(cls, v, values):
        """Production requires approval chain"""
        env = values.get('environment')
        if env == 'production' and not v:
            raise ValueError("Production deployments require approval chain")
        return v

    @root_validator
    def validate_current_stage_in_range(cls, values):
        """Ensure currentStageIndex is valid"""
        current_idx = values.get('currentStageIndex', 0)
        chain = values.get('approvalChain', [])

        if chain and (current_idx < 0 or current_idx >= len(chain)):
            raise ValueError(f"currentStageIndex {current_idx} out of range")

        return values
```

---

## 6. Prompt Evolution Models

### 6.1 PromptEvolutionStrategy

```typescript
interface PromptEvolutionConfig {
  strategy: StrategyType;
  strategyConfig: TextGradConfig | EvoPromptConfig | MetaPromptConfig;

  // Common settings
  maxIterations: number;
  improvementThreshold: number;    // Minimum improvement to continue

  // Constraints
  mustNotChange: string[];         // Section IDs that cannot be modified
  maxSectionLength?: number;
}

type StrategyType = "textgrad" | "evoprompt" | "metaprompt" | "adaptive";

interface TextGradConfig {
  learningRate: number;            // 0.0 to 1.0
  momentumDecay: number;
  criticModel: string;             // Model for generating critiques
  maxCritiqueLength: number;
}

interface EvoPromptConfig {
  populationSize: number;
  mutationRate: number;            // 0.0 to 1.0
  crossoverRate: number;           // 0.0 to 1.0
  selectionPressure: number;
  algorithm: "ga" | "de";          // Genetic Algorithm or Differential Evolution
}

interface MetaPromptConfig {
  metaModel: string;               // Model for meta-analysis
  analysisDepth: "shallow" | "deep";
  structuralChangesAllowed: boolean;
}

interface AdaptiveConfig {
  strategies: StrategyType[];      // Available strategies
  selectionCriteria: "failure_type" | "iteration_count" | "improvement_rate";
  fallbackStrategy: StrategyType;
}
```

### 6.2 PromptSection

```typescript
interface PromptSection {
  id: string;                      // Unique section identifier
  type: SectionType;
  content: string;
  metadata?: Record<string, any>;

  // Evolution tracking
  version: number;
  lastModified?: string;
  modifiedBy?: string;             // Strategy that modified it
}

enum SectionType {
  SYSTEM_CONTEXT = "system_context",
  ROLE_DEFINITION = "role_definition",
  CONSTRAINTS = "constraints",
  OUTPUT_FORMAT = "output_format",
  EXAMPLES = "examples",
  CUSTOM = "custom"
}

interface ParsedPrompt {
  sections: PromptSection[];
  parseMethod: "markdown" | "llm" | "heuristic";
  parseConfidence: number;         // 0.0 to 1.0
}
```

### 6.3 PromptFeedback

```typescript
interface PromptFeedback {
  promptId: string;
  promptVersion: string;

  // Feedback source
  source: "llm_critic" | "human" | "metric";

  // Critique details
  critique: string;
  affectedSections: string[];      // Section IDs
  severity: "low" | "medium" | "high";

  // Actionable suggestions
  suggestions: PromptSuggestion[];

  // Metadata
  timestamp: string;
  evaluationContext?: {
    taskId: string;
    score: number;
    failureType?: string;
  };
}

interface PromptSuggestion {
  sectionId: string;
  changeType: "replace" | "append" | "prepend" | "delete";
  suggestedContent: string;
  rationale: string;
  confidence: number;              // 0.0 to 1.0
}
```

### 6.4 Validation Rules for Prompt Evolution

```python
class PromptEvolutionConfigValidation(BaseModel):
    strategy: str
    maxIterations: int
    improvementThreshold: float
    mustNotChange: List[str]

    @validator('strategy')
    def validate_strategy(cls, v):
        valid_strategies = ["textgrad", "evoprompt", "metaprompt", "adaptive"]
        if v not in valid_strategies:
            raise ValueError(f"Invalid strategy: {v}")
        return v

    @validator('maxIterations')
    def validate_max_iterations(cls, v):
        if v < 1 or v > 100:
            raise ValueError("maxIterations must be between 1 and 100")
        return v

    @validator('improvementThreshold')
    def validate_threshold(cls, v):
        if v < 0 or v > 1:
            raise ValueError("improvementThreshold must be in [0, 1]")
        return v
```

---

## 7. Selection Strategy Models

### 7.1 Selection Strategy Configuration

```typescript
interface SelectionStrategyConfig {
  type: SelectionStrategyType;
  params: TournamentParams | RouletteParams | RankParams | TruncationParams |
          LexicaseParams | SUSParams;
}

enum SelectionStrategyType {
  TOURNAMENT = "tournament",
  ROULETTE_WHEEL = "roulette_wheel",
  RANK_BASED = "rank_based",
  ELITIST = "elitist",
  FITNESS_PROPORTIONATE = "fitness_proportionate",
  TRUNCATION = "truncation",
  STOCHASTIC_UNIVERSAL_SAMPLING = "stochastic_universal_sampling",
  LEXICASE = "lexicase"
}

interface TournamentParams {
  tournamentSize: number;          // k competitors per tournament
  selectionProbability?: number;   // Probability of selecting winner (default: 1.0)
}

interface RouletteParams {
  fitnessScaling?: "linear" | "exponential" | "rank";
  scalingFactor?: number;
}

interface RankParams {
  selectionPressure: number;       // 1.0 to 2.0 (higher = more pressure)
  linearBias?: number;
}

interface TruncationParams {
  truncationRatio: number;         // Top k% to select (0.0 to 1.0)
}

interface LexicaseParams {
  epsilon?: number;                // Epsilon-lexicase threshold
  shuffleMetrics: boolean;         // Whether to randomize metric order
}

interface SUSParams {
  numPointers: number;             // Number of evenly-spaced selection points
}
```

### 7.2 Validation Rules for Selection Strategy

```python
class SelectionStrategyValidation(BaseModel):
    type: str
    params: Dict

    @validator('type')
    def validate_strategy_type(cls, v):
        valid_types = [
            "tournament", "roulette_wheel", "rank_based", "elitist",
            "fitness_proportionate", "truncation",
            "stochastic_universal_sampling", "lexicase"
        ]
        if v not in valid_types:
            raise ValueError(f"Invalid selection strategy: {v}")
        return v

    @root_validator
    def validate_params_for_type(cls, values):
        strategy_type = values.get('type')
        params = values.get('params', {})

        if strategy_type == 'tournament':
            if 'tournamentSize' not in params:
                raise ValueError("Tournament requires tournamentSize")
            if params['tournamentSize'] < 2:
                raise ValueError("tournamentSize must be >= 2")

        elif strategy_type == 'truncation':
            if 'truncationRatio' not in params:
                raise ValueError("Truncation requires truncationRatio")
            ratio = params['truncationRatio']
            if ratio <= 0 or ratio > 1:
                raise ValueError("truncationRatio must be in (0, 1]")

        elif strategy_type == 'rank_based':
            if 'selectionPressure' in params:
                sp = params['selectionPressure']
                if sp < 1.0 or sp > 2.0:
                    raise ValueError("selectionPressure must be in [1.0, 2.0]")

        return values
```

---

## 8. Cross-Model Validation

### 8.1 Referential Integrity Checks

```python
class ReferentialIntegrityChecker:
    """Validate references between models"""

    def __init__(self, config_store):
        self.config_store = config_store

    def validate_evolution_job(self, job: EvolutionJob) -> List[str]:
        """
        Validate all references in EvolutionJob

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check base SOPs exist
        for base_sop in job.baseSops:
            sop = self.config_store.get_sop(base_sop.sopId, base_sop.sopVersion)
            if not sop:
                errors.append(f"Base SOP not found: {base_sop.sopId}:{base_sop.sopVersion}")

            # Check corresponding PromptGenome exists
            prompt_genome = self.config_store.get_prompt_genome(
                base_sop.promptGenomeId,
                base_sop.promptGenomeVersion
            )
            if not prompt_genome:
                errors.append(
                    f"PromptGenome not found: {base_sop.promptGenomeId}:{base_sop.promptGenomeVersion}"
                )

        # Check metrics exist
        for metric_id in job.metricsToOptimize:
            metric = self.config_store.get_metric(metric_id)
            if not metric:
                errors.append(f"Metric not found: {metric_id}")

        # Check domain package exists
        domain_pkg = self.config_store.get_domain_package(job.domain)
        if not domain_pkg:
            errors.append(f"Domain package not found: {job.domain}")

        # Check task set exists
        task_set = self.config_store.get_task_set(job.taskSet.id)
        if not task_set:
            errors.append(f"Task set not found: {job.taskSet.id}")

        return errors

    def validate_domain_package(self, pkg: DomainPackage) -> List[str]:
        """Validate all references in DomainPackage"""
        errors = []

        # Check SOP templates exist
        for sop_id in pkg.sopTemplates:
            if not self.config_store.get_sop(sop_id):
                errors.append(f"SOP template not found: {sop_id}")

        # Check PromptGenomes exist
        for pg_id in pkg.promptGenomes:
            if not self.config_store.get_prompt_genome(pg_id):
                errors.append(f"PromptGenome not found: {pg_id}")

        # Check MetaConfigs exist
        for mc_id in pkg.metaConfigs:
            if not self.config_store.get_meta_config(mc_id):
                errors.append(f"MetaConfig not found: {mc_id}")

        # Check ToolConfigs exist
        for tool_id in pkg.toolConfigs:
            if not self.config_store.get_tool(tool_id):
                errors.append(f"ToolConfig not found: {tool_id}")

        # Check MetricConfigs exist
        for metric_id in pkg.metricConfigs:
            if not self.config_store.get_metric(metric_id):
                errors.append(f"MetricConfig not found: {metric_id}")

        # Check TaskSets exist
        for task_set_id in pkg.evaluationTasks:
            if not self.config_store.get_task_set(task_set_id):
                errors.append(f"TaskSet not found: {task_set_id}")

        # Check dependencies
        if pkg.dependencies:
            for dep in pkg.dependencies:
                dep_pkg = self.config_store.get_domain_package(dep.packageId)
                if not dep_pkg:
                    errors.append(f"Dependency package not found: {dep.packageId}")
                else:
                    # Check version compatibility
                    if not self._check_version_compatible(dep_pkg.version, dep.version):
                        errors.append(
                            f"Dependency version incompatible: {dep.packageId} "
                            f"requires {dep.version}, found {dep_pkg.version}"
                        )

        return errors

    def _check_version_compatible(self, actual: str, required: str) -> bool:
        """Check if versions are compatible using semver rules"""
        # Simplified: exact match or compatible major version
        # Full implementation would use semver library
        actual_parts = actual.split('.')
        required_parts = required.lstrip('^~').split('.')

        # Same major version
        if actual_parts[0] == required_parts[0]:
            return True

        return False
```

---

## 9. Validation Summary

### 9.1 All Model Validations

| Model | Validation Rules | Priority |
|-------|------------------|----------|
| **EvolutionJob** | Metrics exist, weights sum to 1, phases not empty, references valid | HIGH |
| **DomainPackage** | Semver format, core components exist, dependencies satisfied | HIGH |
| **User** | Email format, at least one role, status valid | HIGH |
| **Role** | At least one permission, resource/action valid | MEDIUM |
| **ProcessConfig** | DAG is acyclic, roles have valid tools/prompts | HIGH |
| **PromptGenome** | All promptRefs used in ProcessConfigs exist | MEDIUM |
| **TaskSet** | At least one task, input/groundTruth valid | MEDIUM |
| **MetricConfig** | Required fields based on type (llm_judge needs promptRef) | HIGH |
| **DeploymentRequest** | Valid environment, production requires approval chain | HIGH |
| **ApprovalStage** | Valid type, required roles not empty, quorum valid | MEDIUM |
| **PromptEvolutionConfig** | Valid strategy, iterations in range, threshold in [0,1] | HIGH |
| **SelectionStrategy** | Valid type, params match type requirements | MEDIUM |

### 9.2 Validation Enforcement Points

```python
class ValidationEnforcer:
    """Enforce validation at write-time"""

    def create_evolution_job(self, job: EvolutionJob) -> Tuple[bool, List[str]]:
        """
        Create evolution job with validation

        Returns:
            (success, errors)
        """
        # 1. Schema validation
        try:
            EvolutionJobValidation(**job.dict())
        except Exception as e:
            return False, [f"Schema validation failed: {str(e)}"]

        # 2. Referential integrity
        ref_errors = ReferentialIntegrityChecker(self.config_store).validate_evolution_job(job)
        if ref_errors:
            return False, ref_errors

        # 3. Business logic validation
        business_errors = self._validate_business_logic(job)
        if business_errors:
            return False, business_errors

        # All validations passed
        self.job_store.save(job)
        return True, []

    def _validate_business_logic(self, job: EvolutionJob) -> List[str]:
        """Business logic validation"""
        errors = []

        # Check budget is reasonable
        if job.constraints.budgetLimit:
            if job.constraints.budgetLimit.maxCost and job.constraints.budgetLimit.maxCost < 1.0:
                errors.append("Budget too low: minimum $1.00")

        # Check phase sequence makes sense
        for i, phase in enumerate(job.phases):
            if phase.parentsPerGeneration <= 0:
                errors.append(f"Phase {i}: parentsPerGeneration must be positive")
            if phase.maxGenerations <= 0:
                errors.append(f"Phase {i}: maxGenerations must be positive")

        return errors
```

---

## 10. Implementation Checklist

**Core (siare):**
- [x] Implement EvolutionJob model with Pydantic validation
- [x] Implement DomainPackage model with validation
- [x] Implement TaskSet and Task models
- [x] Implement RoleTemplate and RoleLibrary models
- [x] Implement ReferentialIntegrityChecker
- [x] Implement ValidationEnforcer
- [x] Add unit tests for all validation rules

**Enterprise (siare-cloud):**
- [x] Implement User, Role, Permission models
- [x] Implement AuthorizationService
- [x] Add integration tests for cross-model validation
- [x] Add tests for RBAC authorization logic
- [x] Add tests for scope and condition checking
- [x] Document all validation error messages
- [x] Create migration scripts for predefined roles

---

## 11. Future Extensions

### 11.1 Advanced RBAC Features 🔒

> Available in siare-cloud enterprise edition.

- **Attribute-Based Access Control (ABAC)**: Fine-grained permissions based on resource attributes
- **Time-based permissions**: Temporary access grants
- **Delegation**: Users can delegate permissions to others
- **Audit logging**: Track all authorization decisions

### 11.2 Domain Package Management

- **Package registry**: Central registry for domain packages
- **Version resolution**: Automatic resolution of dependency versions
- **Package publishing**: Workflow for publishing new domain packages
- **Package deprecation**: Graceful deprecation of old versions

---

**Last Updated:** 2025-12-17
