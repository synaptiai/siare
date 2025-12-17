"""
Core Constants

Centralized constants for magic values used across the codebase.
Organized by domain for clarity.
"""

# =============================================================================
# Statistical Thresholds
# =============================================================================

# Minimum sample sizes for statistical tests
MIN_SAMPLES_VARIANCE = 2        # Minimum for variance calculation
MIN_SAMPLES_SKEWNESS = 3        # Minimum for skewness calculation
MIN_SAMPLES_KURTOSIS = 4        # Minimum for kurtosis calculation
MIN_SAMPLES_NORMALITY = 5       # Minimum for normality tests
MIN_SAMPLES_TTEST = 2           # Minimum for t-test per group
MIN_SAMPLES_MANNWHITNEY = 3     # Minimum for Mann-Whitney test

# Statistical confidence/significance
DEFAULT_ALPHA = 0.05            # Default significance level
CONFIDENCE_LEVEL_95 = 0.95      # 95% confidence
MEDIAN_POSITION = 0.5           # Position of median in sorted data

# Weight validation bounds
WEIGHT_SUM_LOWER_BOUND = 0.99   # Weights must sum to >= this
WEIGHT_SUM_UPPER_BOUND = 1.01   # Weights must sum to <= this

# =============================================================================
# Multiple Comparison Thresholds
# =============================================================================

MIN_COMPARISONS_BONFERRONI = 5  # Min for Bonferroni correction
MAX_COMPARISONS_BONFERRONI = 20 # Max before switching to Holm
MIN_GROUPS_TUKEY = 3            # Min groups for Tukey's HSD
MIN_SAMPLES_PER_GROUP = 3       # Min samples per group for tests
MAX_SAMPLES_PER_GROUP = 10      # Max before using asymptotic

# =============================================================================
# Service Configuration
# =============================================================================

# Retry and timeout settings
MAX_RETRY_ATTEMPTS = 3          # Default max retries
DEFAULT_CACHE_TTL = 3600        # Default cache TTL in seconds

# Safety thresholds
SAFETY_CRITICAL_THRESHOLD = 1000  # Threshold for safety critical operations
MIN_PARENTS_FOR_CROSSOVER = 2     # Minimum parents for crossover operation

# Adapter settings
MIN_SEARCH_RESULTS = 5          # Minimum search results to return
