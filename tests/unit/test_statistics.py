"""Tests for statistical utilities"""

import numpy as np
import pytest

from siare.core.models import AggregationMethod
from siare.utils.statistics import (
    aggregate_with_statistics,
    bootstrap_confidence_interval,
    compare_sop_performance,
    detect_outliers_iqr,
    detect_outliers_zscore,
    independent_t_test,
    mann_whitney_u_test,
    trimmed_mean,
    wilcoxon_signed_rank_test,
)


# ============================================================================
# Bootstrap Confidence Interval Tests
# ============================================================================


def test_bootstrap_ci_valid_data():
    """Test bootstrap CI with valid data"""
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    lower, upper = bootstrap_confidence_interval(data, confidence_level=0.95, random_seed=42)

    # CI should contain the true mean
    true_mean = np.mean(data)
    assert lower < true_mean < upper

    # CI should be reasonable width
    assert upper - lower < 5.0


def test_bootstrap_ci_different_statistics():
    """Test bootstrap CI with different statistics (median)"""
    data = [1.0, 2.0, 3.0, 4.0, 100.0]  # Outlier
    lower, upper = bootstrap_confidence_interval(data, statistic_fn=np.median, random_seed=42)

    true_median = np.median(data)
    assert lower <= true_median <= upper


def test_bootstrap_ci_insufficient_samples():
    """Test bootstrap CI with insufficient samples"""
    with pytest.raises(ValueError, match="Need at least 2 samples"):
        bootstrap_confidence_interval([1.0], confidence_level=0.95)


def test_bootstrap_ci_invalid_confidence_level():
    """Test bootstrap CI with invalid confidence level"""
    data = [1.0, 2.0, 3.0]

    with pytest.raises(ValueError, match="confidence_level must be between 0 and 1"):
        bootstrap_confidence_interval(data, confidence_level=1.5)

    with pytest.raises(ValueError, match="confidence_level must be between 0 and 1"):
        bootstrap_confidence_interval(data, confidence_level=-0.1)


def test_bootstrap_ci_reproducibility():
    """Test bootstrap CI is reproducible with same seed"""
    data = [1.0, 2.0, 3.0, 4.0, 5.0]

    ci1 = bootstrap_confidence_interval(data, random_seed=42)
    ci2 = bootstrap_confidence_interval(data, random_seed=42)

    assert ci1 == ci2


# ============================================================================
# Outlier Detection Tests
# ============================================================================


def test_detect_outliers_iqr_no_outliers():
    """Test IQR outlier detection with no outliers"""
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    outlier_info = detect_outliers_iqr(data)

    assert len(outlier_info.indices) == 0
    assert len(outlier_info.values) == 0
    assert outlier_info.method == "iqr"


def test_detect_outliers_iqr_with_outliers():
    """Test IQR outlier detection with clear outliers"""
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0]
    outlier_info = detect_outliers_iqr(data)

    assert len(outlier_info.indices) > 0
    assert 100.0 in outlier_info.values
    assert outlier_info.method == "iqr"
    assert outlier_info.threshold == 1.5


def test_detect_outliers_iqr_insufficient_samples():
    """Test IQR outlier detection with insufficient samples"""
    data = [1.0, 2.0, 3.0]
    outlier_info = detect_outliers_iqr(data)

    # Should return empty outlier info
    assert len(outlier_info.indices) == 0


def test_detect_outliers_zscore_no_outliers():
    """Test Z-score outlier detection with no outliers"""
    data = list(np.random.normal(0, 1, 100))
    outlier_info = detect_outliers_zscore(data, threshold=3.0)

    # With threshold=3, very few outliers expected in normal data
    assert len(outlier_info.indices) < 5
    assert outlier_info.method == "zscore"


def test_detect_outliers_zscore_with_outliers():
    """Test Z-score outlier detection with clear outliers"""
    data = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 100.0]
    outlier_info = detect_outliers_zscore(data, threshold=3.0)

    assert len(outlier_info.indices) > 0
    assert 100.0 in outlier_info.values


def test_detect_outliers_zscore_no_variation():
    """Test Z-score outlier detection with no variation"""
    data = [5.0, 5.0, 5.0, 5.0]
    outlier_info = detect_outliers_zscore(data)

    # No variation means no outliers
    assert len(outlier_info.indices) == 0


# ============================================================================
# Trimmed Mean Tests
# ============================================================================


def test_trimmed_mean_basic():
    """Test trimmed mean with basic data"""
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    result = trimmed_mean(data, trim_proportion=0.1)

    # Trimmed mean should be close to mean for symmetric data
    assert abs(result - 5.5) < 0.5


def test_trimmed_mean_with_outliers():
    """Test trimmed mean removes outlier influence"""
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1000.0]

    # Regular mean is heavily influenced
    regular_mean = np.mean(data)
    assert regular_mean > 100

    # Trimmed mean is more robust
    trimmed = trimmed_mean(data, trim_proportion=0.1)
    assert trimmed < 10


def test_trimmed_mean_invalid_proportion():
    """Test trimmed mean with invalid proportion"""
    data = [1.0, 2.0, 3.0]

    with pytest.raises(ValueError, match="trim_proportion must be between 0 and 0.5"):
        trimmed_mean(data, trim_proportion=0.6)

    with pytest.raises(ValueError, match="trim_proportion must be between 0 and 0.5"):
        trimmed_mean(data, trim_proportion=-0.1)


# ============================================================================
# Aggregate with Statistics Tests
# ============================================================================


def test_aggregate_with_statistics_basic():
    """Test aggregate_with_statistics with basic data"""
    data = [0.8, 0.85, 0.9, 0.82, 0.88, 0.91, 0.87, 0.84]
    result = aggregate_with_statistics("accuracy", data)

    assert result.metricId == "accuracy"
    assert result.mean == pytest.approx(np.mean(data), rel=1e-6)
    assert result.median == pytest.approx(np.median(data), rel=1e-6)
    assert result.sampleSize == len(data)
    assert result.standardDeviation is not None
    assert result.standardError is not None


def test_aggregate_with_statistics_with_ci():
    """Test aggregate_with_statistics computes confidence intervals"""
    data = [0.8, 0.85, 0.9, 0.82, 0.88, 0.91, 0.87, 0.84]
    result = aggregate_with_statistics("accuracy", data, compute_confidence_interval=True)

    assert result.confidenceInterval is not None
    lower, upper = result.confidenceInterval
    assert lower < result.mean < upper


def test_aggregate_with_statistics_with_outliers():
    """Test aggregate_with_statistics detects outliers"""
    data = [0.8, 0.85, 0.9, 0.82, 0.88, 0.91, 0.01]  # 0.01 is outlier
    result = aggregate_with_statistics("accuracy", data, detect_outliers=True)

    assert result.outliers is not None
    assert len(result.outliers.indices) > 0


def test_aggregate_with_statistics_small_sample():
    """Test aggregate_with_statistics with small sample"""
    data = [0.8, 0.9]
    result = aggregate_with_statistics("accuracy", data, compute_confidence_interval=True)

    assert result.sampleSize == 2
    # Should still compute CI even with small sample
    assert result.confidenceInterval is not None


def test_aggregate_with_statistics_empty_data():
    """Test aggregate_with_statistics with empty data"""
    with pytest.raises(ValueError, match="Cannot aggregate empty values list"):
        aggregate_with_statistics("accuracy", [])


def test_aggregate_with_statistics_trimmed_mean():
    """Test aggregate_with_statistics computes trimmed mean"""
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0]
    result = aggregate_with_statistics("metric", data)

    # Trimmed mean should be more robust than mean
    assert result.trimmedMean is not None
    assert result.trimmedMean < result.mean


# ============================================================================
# Hypothesis Testing Tests
# ============================================================================


def test_mann_whitney_u_test_different_distributions():
    """Test Mann-Whitney U test with different distributions"""
    sample_a = [1.0, 2.0, 3.0, 4.0, 5.0]
    sample_b = [6.0, 7.0, 8.0, 9.0, 10.0]

    result = mann_whitney_u_test(sample_a, sample_b, alternative="two-sided")

    assert result.testType == "mannwhitneyu"
    assert result.pValue < 0.05  # Should be significant
    assert result.isSignificant is True
    assert result.effectSize is not None


def test_mann_whitney_u_test_same_distribution():
    """Test Mann-Whitney U test with same distribution"""
    np.random.seed(42)
    sample_a = list(np.random.normal(5, 1, 20))
    sample_b = list(np.random.normal(5, 1, 20))

    result = mann_whitney_u_test(sample_a, sample_b, alternative="two-sided")

    # Likely not significant
    assert result.pValue > 0.01


def test_mann_whitney_u_test_insufficient_samples():
    """Test Mann-Whitney U test with insufficient samples"""
    with pytest.raises(ValueError, match="Need at least 3 samples"):
        mann_whitney_u_test([1.0, 2.0], [3.0, 4.0])


def test_wilcoxon_signed_rank_test_paired():
    """Test Wilcoxon signed-rank test with paired data"""
    sample_a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    sample_b = [1.1, 2.2, 3.1, 4.3, 5.2, 6.1, 7.2, 8.1]

    result = wilcoxon_signed_rank_test(sample_a, sample_b, alternative="two-sided")

    assert result.testType == "wilcoxon"
    # Small differences, may or may not be significant
    assert result.pValue is not None


def test_wilcoxon_signed_rank_test_unequal_lengths():
    """Test Wilcoxon signed-rank test with unequal lengths"""
    with pytest.raises(ValueError, match="equal-length paired samples"):
        wilcoxon_signed_rank_test([1.0, 2.0, 3.0], [1.0, 2.0])


def test_wilcoxon_signed_rank_test_insufficient_samples():
    """Test Wilcoxon signed-rank test with insufficient samples"""
    with pytest.raises(ValueError, match="Need at least 5 paired samples"):
        wilcoxon_signed_rank_test([1.0, 2.0], [1.1, 2.1])


def test_independent_t_test_different_means():
    """Test independent t-test with different means"""
    np.random.seed(42)
    sample_a = list(np.random.normal(5, 1, 30))
    sample_b = list(np.random.normal(7, 1, 30))

    result = independent_t_test(sample_a, sample_b, equal_variance=False)

    assert result.testType == "welch_ttest"
    assert result.pValue < 0.001  # Should be highly significant
    assert result.isSignificant is True
    assert result.effectSize is not None  # Cohen's d


def test_independent_t_test_same_mean():
    """Test independent t-test with same mean"""
    np.random.seed(42)
    sample_a = list(np.random.normal(5, 1, 30))
    sample_b = list(np.random.normal(5, 1, 30))

    result = independent_t_test(sample_a, sample_b)

    # Likely not significant
    assert result.pValue > 0.01


def test_independent_t_test_insufficient_samples():
    """Test independent t-test with insufficient samples"""
    with pytest.raises(ValueError, match="Need at least 2 samples"):
        independent_t_test([1.0], [2.0])


# ============================================================================
# Compare SOP Performance Tests
# ============================================================================


def test_compare_sop_performance_basic():
    """Test comparing SOP performance across metrics"""
    sop_a_metrics = {
        "accuracy": [0.8, 0.85, 0.9, 0.82, 0.88],
        "latency": [100.0, 110.0, 95.0, 105.0, 98.0],
    }
    sop_b_metrics = {
        "accuracy": [0.7, 0.72, 0.75, 0.68, 0.71],
        "latency": [120.0, 125.0, 115.0, 118.0, 122.0],
    }

    results = compare_sop_performance(sop_a_metrics, sop_b_metrics, paired=False, use_parametric=False)

    assert "accuracy" in results
    assert "latency" in results
    assert results["accuracy"].testType == "mannwhitneyu"
    assert results["accuracy"].isSignificant is True  # SOP A is clearly better


def test_compare_sop_performance_paired():
    """Test comparing SOP performance with paired data"""
    sop_a_metrics = {
        "accuracy": [0.8, 0.85, 0.9, 0.82, 0.88, 0.87, 0.84, 0.86],
    }
    sop_b_metrics = {
        "accuracy": [0.79, 0.84, 0.88, 0.81, 0.87, 0.86, 0.83, 0.85],
    }

    results = compare_sop_performance(sop_a_metrics, sop_b_metrics, paired=True, use_parametric=False)

    assert "accuracy" in results
    assert results["accuracy"].testType == "wilcoxon"


def test_compare_sop_performance_parametric():
    """Test comparing SOP performance with parametric tests"""
    np.random.seed(42)
    sop_a_metrics = {
        "accuracy": list(np.random.normal(0.85, 0.05, 30)),
    }
    sop_b_metrics = {
        "accuracy": list(np.random.normal(0.75, 0.05, 30)),
    }

    results = compare_sop_performance(sop_a_metrics, sop_b_metrics, paired=False, use_parametric=True)

    assert "accuracy" in results
    assert results["accuracy"].testType == "welch_ttest"
    assert results["accuracy"].isSignificant is True


def test_compare_sop_performance_missing_metric():
    """Test comparing SOP performance with missing metric in one SOP"""
    sop_a_metrics = {
        "accuracy": [0.8, 0.85, 0.9],
        "latency": [100.0, 110.0, 95.0],
    }
    sop_b_metrics = {
        "accuracy": [0.7, 0.72, 0.75],
        # latency missing
    }

    results = compare_sop_performance(sop_a_metrics, sop_b_metrics)

    # Only accuracy should be compared
    assert "accuracy" in results
    assert "latency" not in results


def test_compare_sop_performance_no_common_metrics():
    """Test comparing SOP performance with no common metrics"""
    sop_a_metrics = {
        "accuracy": [0.8, 0.85, 0.9],
    }
    sop_b_metrics = {
        "latency": [100.0, 110.0, 95.0],
    }

    results = compare_sop_performance(sop_a_metrics, sop_b_metrics)

    # No common metrics
    assert len(results) == 0


# ============================================================================
# Integration Tests
# ============================================================================


def test_full_statistical_pipeline():
    """Test full statistical aggregation pipeline"""
    # Simulate multiple evaluations
    metric_values = [0.82, 0.85, 0.88, 0.84, 0.87, 0.90, 0.83, 0.86, 0.89, 0.01]  # One outlier

    # Aggregate with all features
    result = aggregate_with_statistics(
        metric_id="accuracy",
        values=metric_values,
        aggregation_method=AggregationMethod.MEAN,
        compute_confidence_interval=True,
        detect_outliers=True,
        outlier_method="iqr",
    )

    # Check all components are present
    assert result.metricId == "accuracy"
    assert result.mean is not None
    assert result.median is not None
    assert result.trimmedMean is not None
    assert result.confidenceInterval is not None
    assert result.standardDeviation is not None
    assert result.standardError is not None
    assert result.sampleSize == len(metric_values)
    assert result.outliers is not None
    assert len(result.outliers.indices) > 0  # Should detect 0.01 as outlier
    assert 0.01 in result.outliers.values

    # Trimmed mean should be higher than mean (outlier pulls down)
    assert result.trimmedMean > result.mean


def test_statistical_comparison_workflow():
    """Test complete statistical comparison workflow"""
    np.random.seed(42)

    # Generate realistic metric data for two SOPs
    sop_a_metrics = {
        "accuracy": list(np.random.beta(8, 2, 50)),  # High accuracy
        "latency": list(np.random.gamma(2, 50, 50)),  # Moderate latency
        "cost": list(np.random.gamma(2, 0.01, 50)),  # Low cost
    }

    sop_b_metrics = {
        "accuracy": list(np.random.beta(6, 4, 50)),  # Lower accuracy
        "latency": list(np.random.gamma(2, 40, 50)),  # Lower latency
        "cost": list(np.random.gamma(2, 0.015, 50)),  # Higher cost
    }

    # Compare SOPs
    results = compare_sop_performance(
        sop_a_metrics,
        sop_b_metrics,
        paired=False,
        use_parametric=False,
    )

    # All three metrics should be compared
    assert len(results) == 3
    assert all(metric in results for metric in ["accuracy", "latency", "cost"])

    # All tests should have valid results
    for metric_id, test_result in results.items():
        assert test_result.pValue is not None
        assert 0 <= test_result.pValue <= 1
        assert test_result.statistic is not None
        assert test_result.isSignificant == (test_result.pValue < 0.05)
