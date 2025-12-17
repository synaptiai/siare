"""Statistical utilities for robust metric aggregation"""

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
from scipy import stats

from siare.core.constants import (
    DEFAULT_ALPHA,
    MEDIAN_POSITION,
    MIN_SAMPLES_KURTOSIS,
    MIN_SAMPLES_MANNWHITNEY,
    MIN_SAMPLES_NORMALITY,
    MIN_SAMPLES_SKEWNESS,
    MIN_SAMPLES_TTEST,
    MIN_SAMPLES_VARIANCE,
)
from siare.core.models import (
    AggregatedMetric,
    AggregationMethod,
    OutlierInfo,
    StatisticalTestResult,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Bootstrap Confidence Intervals
# ============================================================================


def bootstrap_confidence_interval(
    data: list[float],
    statistic_fn: Callable[[Any], float] = np.mean,  # type: ignore[assignment]
    confidence_level: float = 0.95,
    n_bootstrap: int = 10000,
    random_seed: int | None = None,
) -> tuple[float, float]:
    """
    Calculate bootstrap confidence interval for a statistic.

    Uses percentile bootstrap method for robust CI estimation.

    Args:
        data: Sample data
        statistic_fn: Function to compute statistic (default: mean)
        confidence_level: Confidence level (0-1), default 0.95
        n_bootstrap: Number of bootstrap samples, default 10000
        random_seed: Random seed for reproducibility

    Returns:
        (lower_bound, upper_bound) confidence interval
    """
    if len(data) < MIN_SAMPLES_VARIANCE:
        raise ValueError("Need at least 2 samples for bootstrap CI")

    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")

    rng = np.random.RandomState(random_seed)
    data_array = np.array(data)
    n = len(data_array)

    # Generate bootstrap samples
    bootstrap_statistics = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(data_array, size=n, replace=True)
        bootstrap_statistics[i] = statistic_fn(sample)

    # Calculate percentile-based confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    lower_bound = np.percentile(bootstrap_statistics, lower_percentile)
    upper_bound = np.percentile(bootstrap_statistics, upper_percentile)

    return (float(lower_bound), float(upper_bound))


# ============================================================================
# Outlier Detection
# ============================================================================


def detect_outliers_iqr(
    data: list[float],
    iqr_multiplier: float = 1.5,
) -> OutlierInfo:
    """
    Detect outliers using Interquartile Range (IQR) method.

    Outliers are values outside [Q1 - k*IQR, Q3 + k*IQR] where k is the multiplier.

    Args:
        data: Sample data
        iqr_multiplier: IQR multiplier (default 1.5, Tukey's fence)

    Returns:
        OutlierInfo with detected outliers
    """
    if len(data) < MIN_SAMPLES_KURTOSIS:
        # Cannot compute IQR with fewer than 4 samples
        return OutlierInfo(indices=[], values=[], method="iqr", threshold=iqr_multiplier)

    data_array = np.array(data)
    q1 = np.percentile(data_array, 25)
    q3 = np.percentile(data_array, 75)
    iqr = q3 - q1

    lower_fence = q1 - iqr_multiplier * iqr
    upper_fence = q3 + iqr_multiplier * iqr

    # Find outliers
    outlier_mask = (data_array < lower_fence) | (data_array > upper_fence)
    outlier_indices = np.where(outlier_mask)[0].tolist()
    outlier_values = data_array[outlier_mask].tolist()

    logger.debug(
        f"IQR outlier detection: Q1={q1:.4f}, Q3={q3:.4f}, IQR={iqr:.4f}, "
        f"fences=[{lower_fence:.4f}, {upper_fence:.4f}], found {len(outlier_indices)} outliers"
    )

    return OutlierInfo(
        indices=outlier_indices,
        values=outlier_values,
        method="iqr",
        threshold=iqr_multiplier,
    )


def detect_outliers_zscore(
    data: list[float],
    threshold: float = 3.0,
) -> OutlierInfo:
    """
    Detect outliers using Z-score method.

    Outliers are values with |z-score| > threshold.

    Args:
        data: Sample data
        threshold: Z-score threshold (default 3.0)

    Returns:
        OutlierInfo with detected outliers
    """
    if len(data) < MIN_SAMPLES_SKEWNESS:
        return OutlierInfo(indices=[], values=[], method="zscore", threshold=threshold)

    data_array = np.array(data)
    mean = np.mean(data_array)
    std = np.std(data_array, ddof=1)

    if std == 0:
        # No variation, no outliers
        return OutlierInfo(indices=[], values=[], method="zscore", threshold=threshold)

    z_scores = np.abs((data_array - mean) / std)
    outlier_mask = z_scores > threshold
    outlier_indices = np.where(outlier_mask)[0].tolist()
    outlier_values = data_array[outlier_mask].tolist()

    logger.debug(
        f"Z-score outlier detection: mean={mean:.4f}, std={std:.4f}, "
        f"threshold={threshold}, found {len(outlier_indices)} outliers"
    )

    return OutlierInfo(
        indices=outlier_indices,
        values=outlier_values,
        method="zscore",
        threshold=threshold,
    )


# ============================================================================
# Robust Aggregation Functions
# ============================================================================


def trimmed_mean(data: list[float], trim_proportion: float = 0.1) -> float:
    """
    Calculate trimmed mean by removing extreme values.

    Args:
        data: Sample data
        trim_proportion: Proportion to trim from each tail (0-0.5)

    Returns:
        Trimmed mean
    """
    if not 0 <= trim_proportion < MEDIAN_POSITION:
        raise ValueError("trim_proportion must be between 0 and 0.5")

    if len(data) < MIN_SAMPLES_VARIANCE:
        return float(np.mean(data))

    return float(stats.trim_mean(data, trim_proportion))  # type: ignore[arg-type]


def aggregate_with_statistics(
    metric_id: str,
    values: list[float],
    aggregation_method: AggregationMethod = AggregationMethod.MEAN,
    compute_confidence_interval: bool = True,
    detect_outliers: bool = True,
    outlier_method: str = "iqr",
    trim_proportion: float = 0.1,
) -> AggregatedMetric:
    """
    Aggregate metric values with statistical rigor.

    Args:
        metric_id: Metric identifier
        values: Metric values to aggregate
        aggregation_method: Primary aggregation method
        compute_confidence_interval: Whether to compute bootstrap CI
        detect_outliers: Whether to detect outliers
        outlier_method: Outlier detection method ("iqr" or "zscore")
        trim_proportion: Proportion for trimmed mean (0-0.5)

    Returns:
        AggregatedMetric with statistics
    """
    if not values:
        raise ValueError("Cannot aggregate empty values list")

    values_array = np.array(values)
    n = len(values)

    # Basic statistics
    mean = float(np.mean(values_array))
    median = float(np.median(values_array))
    std_dev = float(np.std(values_array, ddof=1)) if n > 1 else 0.0
    std_error = std_dev / np.sqrt(n) if n > 1 else 0.0

    # Trimmed mean
    trim_mean = None
    if n >= MIN_SAMPLES_KURTOSIS:  # Need at least 4 samples for meaningful trimming
        trim_mean = trimmed_mean(values, trim_proportion)

    # Confidence interval
    ci = None
    if compute_confidence_interval and n >= MIN_SAMPLES_VARIANCE:
        try:
            ci = bootstrap_confidence_interval(
                values,
                confidence_level=0.95,
                n_bootstrap=10000,
            )
        except Exception as e:
            logger.warning(f"Failed to compute CI for {metric_id}: {e}")

    # Outlier detection
    outliers = None
    if detect_outliers and n >= MIN_SAMPLES_KURTOSIS:
        try:
            if outlier_method == "iqr":
                outliers = detect_outliers_iqr(values)
            elif outlier_method == "zscore":
                outliers = detect_outliers_zscore(values)
            else:
                logger.warning(f"Unknown outlier method: {outlier_method}")
        except Exception as e:
            logger.warning(f"Failed to detect outliers for {metric_id}: {e}")

    return AggregatedMetric(
        metricId=metric_id,
        mean=mean,
        median=median,
        trimmedMean=trim_mean,
        confidenceInterval=ci,
        standardDeviation=std_dev,
        standardError=std_error,
        sampleSize=n,
        outliers=outliers,
        aggregationMethod=aggregation_method,
        rawValues=values,
    )


# ============================================================================
# Statistical Hypothesis Testing
# ============================================================================


def mann_whitney_u_test(
    sample_a: list[float],
    sample_b: list[float],
    alternative: str = "two-sided",
) -> StatisticalTestResult:
    """
    Perform Mann-Whitney U test (non-parametric comparison of two samples).

    Tests whether distributions of two independent samples are different.

    Args:
        sample_a: First sample
        sample_b: Second sample
        alternative: "two-sided", "less", or "greater"

    Returns:
        StatisticalTestResult
    """
    if len(sample_a) < MIN_SAMPLES_MANNWHITNEY or len(sample_b) < MIN_SAMPLES_MANNWHITNEY:
        raise ValueError("Need at least 3 samples in each group for Mann-Whitney U test")

    result = stats.mannwhitneyu(  # type: ignore[arg-type]
        sample_a,
        sample_b,
        alternative=alternative,
    )
    statistic: float = float(result.statistic)  # type: ignore[attr-defined]
    p_value: float = float(result.pvalue)  # type: ignore[attr-defined]

    # Compute rank-biserial correlation as effect size
    n1: int = len(sample_a)
    n2: int = len(sample_b)
    u1: float = statistic
    rank_biserial: float = 1 - (2 * u1) / (n1 * n2)

    is_significant: bool = p_value < DEFAULT_ALPHA

    hypothesis = f"Sample A and Sample B have {'different' if alternative == 'two-sided' else 'ordered'} distributions"

    return StatisticalTestResult(
        testType="mannwhitneyu",
        statistic=statistic,
        pValue=p_value,
        isSignificant=is_significant,
        effectSize=rank_biserial,
        confidenceLevel=0.95,
        hypothesis=hypothesis,
    )


def wilcoxon_signed_rank_test(
    sample_a: list[float],
    sample_b: list[float],
    alternative: str = "two-sided",
) -> StatisticalTestResult:
    """
    Perform Wilcoxon signed-rank test (non-parametric paired comparison).

    Tests whether paired samples have different distributions.

    Args:
        sample_a: First paired sample
        sample_b: Second paired sample
        alternative: "two-sided", "less", or "greater"

    Returns:
        StatisticalTestResult
    """
    if len(sample_a) != len(sample_b):
        raise ValueError("Wilcoxon test requires equal-length paired samples")

    if len(sample_a) < MIN_SAMPLES_NORMALITY:
        raise ValueError("Need at least 5 paired samples for Wilcoxon test")

    result = stats.wilcoxon(  # type: ignore[arg-type]
        sample_a,
        sample_b,
        alternative=alternative,
    )
    statistic: float = float(result.statistic)  # type: ignore[attr-defined]
    p_value: float = float(result.pvalue)  # type: ignore[attr-defined]

    is_significant: bool = p_value < DEFAULT_ALPHA

    # Compute effect size (matched-pairs rank-biserial correlation)
    differences = np.array(sample_a) - np.array(sample_b)
    non_zero_diffs = differences[differences != 0]
    n = len(non_zero_diffs)

    if n > 0:
        # Simplified rank-biserial for paired data
        pos_ranks = np.sum(np.abs(non_zero_diffs[non_zero_diffs > 0]))
        total_ranks = n * (n + 1) / 2
        rank_biserial = (pos_ranks / total_ranks - 0.5) * 2
    else:
        rank_biserial = 0.0

    hypothesis = f"Paired samples have {'different' if alternative == 'two-sided' else 'ordered'} distributions"

    return StatisticalTestResult(
        testType="wilcoxon",
        statistic=statistic,
        pValue=p_value,
        isSignificant=is_significant,
        effectSize=float(rank_biserial),
        confidenceLevel=0.95,
        hypothesis=hypothesis,
    )


def independent_t_test(
    sample_a: list[float],
    sample_b: list[float],
    equal_variance: bool = False,
) -> StatisticalTestResult:
    """
    Perform independent samples t-test (parametric comparison).

    Tests whether means of two independent samples are different.
    Assumes approximately normal distributions.

    Args:
        sample_a: First sample
        sample_b: Second sample
        equal_variance: Whether to assume equal variances (Welch's t-test if False)

    Returns:
        StatisticalTestResult
    """
    if len(sample_a) < MIN_SAMPLES_TTEST or len(sample_b) < MIN_SAMPLES_TTEST:
        raise ValueError("Need at least 2 samples in each group for t-test")

    result = stats.ttest_ind(  # type: ignore[arg-type]
        sample_a,
        sample_b,
        equal_var=equal_variance,
    )
    statistic: float = float(result.statistic)  # type: ignore[attr-defined]
    p_value: float = float(result.pvalue)  # type: ignore[attr-defined]

    # Compute Cohen's d as effect size
    mean_a = np.mean(sample_a)
    mean_b = np.mean(sample_b)
    pooled_std = np.sqrt(
        ((len(sample_a) - 1) * np.var(sample_a, ddof=1) + (len(sample_b) - 1) * np.var(sample_b, ddof=1))
        / (len(sample_a) + len(sample_b) - 2)
    )

    cohens_d = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0.0

    is_significant: bool = p_value < DEFAULT_ALPHA

    test_name = "welch_ttest" if not equal_variance else "ttest"
    hypothesis = "Sample A and Sample B have different means"

    return StatisticalTestResult(
        testType=test_name,
        statistic=statistic,
        pValue=p_value,
        isSignificant=is_significant,
        effectSize=float(cohens_d),
        confidenceLevel=0.95,
        hypothesis=hypothesis,
    )


def compare_sop_performance(
    sop_a_metrics: dict[str, list[float]],
    sop_b_metrics: dict[str, list[float]],
    paired: bool = False,
    use_parametric: bool = False,
) -> dict[str, StatisticalTestResult]:
    """
    Compare performance of two SOPs across multiple metrics.

    Args:
        sop_a_metrics: Metric values for SOP A {metric_id: [values]}
        sop_b_metrics: Metric values for SOP B {metric_id: [values]}
        paired: Whether samples are paired (same tasks evaluated)
        use_parametric: Whether to use parametric tests (t-test vs Mann-Whitney/Wilcoxon)

    Returns:
        Dictionary of test results by metric ID
    """
    results: dict[str, StatisticalTestResult] = {}

    for metric_id in sop_a_metrics:
        if metric_id not in sop_b_metrics:
            logger.warning(f"Metric {metric_id} missing from SOP B, skipping comparison")
            continue

        values_a = sop_a_metrics[metric_id]
        values_b = sop_b_metrics[metric_id]

        try:
            if paired:
                if use_parametric:
                    # Paired t-test
                    if len(values_a) != len(values_b):
                        logger.warning(f"Paired test requires equal lengths for {metric_id}")
                        continue
                    result = stats.ttest_rel(values_a, values_b)  # type: ignore[arg-type]
                    stat: float = float(result.statistic)  # type: ignore[attr-defined]
                    p_val: float = float(result.pvalue)  # type: ignore[attr-defined]
                    test_result = StatisticalTestResult(
                        testType="paired_ttest",
                        statistic=stat,
                        pValue=p_val,
                        isSignificant=p_val < DEFAULT_ALPHA,
                        confidenceLevel=0.95,
                        hypothesis=f"SOP A and SOP B have different mean {metric_id}",
                    )
                else:
                    # Wilcoxon signed-rank test
                    test_result = wilcoxon_signed_rank_test(values_a, values_b)
            elif use_parametric:
                # Independent t-test
                test_result = independent_t_test(values_a, values_b, equal_variance=False)
            else:
                # Mann-Whitney U test
                test_result = mann_whitney_u_test(values_a, values_b)

            results[metric_id] = test_result

        except Exception as e:
            logger.error(f"Failed to compare metric {metric_id}: {e}")
            continue

    return results
