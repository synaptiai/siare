"""Multiple comparison correction for statistical tests"""

import logging
from enum import Enum
from typing import Any

import numpy as np
import numpy.typing as npt

from siare.core.constants import (
    MAX_COMPARISONS_BONFERRONI,
    MAX_SAMPLES_PER_GROUP,
    MIN_COMPARISONS_BONFERRONI,
    MIN_GROUPS_TUKEY,
)
from siare.core.models import StatisticalTestResult

logger = logging.getLogger(__name__)


# Type alias for correction results (is_significant, corrected_p_values)
CorrectionResult = tuple[npt.NDArray[np.bool_], npt.NDArray[np.floating[Any]]]


class CorrectionMethod(str, Enum):
    """Multiple comparison correction methods"""

    BONFERRONI = "bonferroni"  # Most conservative
    HOLM = "holm"  # Holm-Bonferroni (less conservative)
    SIDAK = "sidak"  # Similar to Bonferroni
    FDR_BH = "fdr_bh"  # Benjamini-Hochberg (controls false discovery rate)
    NONE = "none"  # No correction


def _bonferroni_correction(
    p_values: npt.NDArray[np.floating[Any]], n_tests: int, alpha: float
) -> CorrectionResult:
    """Apply Bonferroni correction (most conservative).

    Args:
        p_values: Array of p-values
        n_tests: Number of tests
        alpha: Significance level

    Returns:
        Tuple of (is_significant, corrected_p_values)
    """
    adjusted_alpha = alpha / n_tests
    is_significant = p_values < adjusted_alpha
    corrected_p_values = np.minimum(p_values * n_tests, 1.0)
    return is_significant, corrected_p_values


def _holm_correction(
    p_values: npt.NDArray[np.floating[Any]], n_tests: int, alpha: float
) -> CorrectionResult:
    """Apply Holm-Bonferroni step-down correction.

    Args:
        p_values: Array of p-values
        n_tests: Number of tests
        alpha: Significance level

    Returns:
        Tuple of (is_significant, corrected_p_values)
    """
    sorted_indices = np.argsort(p_values)
    is_significant = np.zeros(n_tests, dtype=bool)
    corrected_p_values = p_values.copy()

    for i, idx in enumerate(sorted_indices):
        threshold = alpha / (n_tests - i)
        if p_values[idx] < threshold:
            is_significant[idx] = True
            corrected_p_values[idx] = p_values[idx] * (n_tests - i)
        else:
            break

    corrected_p_values = np.minimum(corrected_p_values, 1.0)
    return is_significant, corrected_p_values


def _sidak_correction(
    p_values: npt.NDArray[np.floating[Any]], n_tests: int, alpha: float
) -> CorrectionResult:
    """Apply Šidák correction (assumes independence).

    Args:
        p_values: Array of p-values
        n_tests: Number of tests
        alpha: Significance level

    Returns:
        Tuple of (is_significant, corrected_p_values)
    """
    adjusted_alpha = 1 - (1 - alpha) ** (1 / n_tests)
    is_significant = p_values < adjusted_alpha
    corrected_p_values = 1 - (1 - p_values) ** n_tests
    return is_significant, corrected_p_values


def _fdr_bh_correction(
    p_values: npt.NDArray[np.floating[Any]], n_tests: int, alpha: float
) -> CorrectionResult:
    """Apply Benjamini-Hochberg FDR correction.

    Args:
        p_values: Array of p-values
        n_tests: Number of tests
        alpha: Significance level

    Returns:
        Tuple of (is_significant, corrected_p_values)
    """
    sorted_indices = np.argsort(p_values)
    is_significant = np.zeros(n_tests, dtype=bool)
    corrected_p_values = p_values.copy()

    # Step-up procedure
    for i in range(n_tests - 1, -1, -1):
        idx = sorted_indices[i]
        threshold = (i + 1) / n_tests * alpha
        if p_values[idx] <= threshold:
            is_significant[idx] = True
            for j in range(i + 1):
                is_significant[sorted_indices[j]] = True
            break

    # Correct p-values (BH adjustment)
    for i in range(n_tests):
        rank = int(np.where(sorted_indices == i)[0][0] + 1)
        corrected_p_values[i] = min(p_values[i] * n_tests / rank, 1.0)

    return is_significant, corrected_p_values


def _build_corrected_result(
    original: StatisticalTestResult,
    method: CorrectionMethod,
    corrected_p_value: float,
    is_significant: bool,
) -> StatisticalTestResult:
    """Build a corrected StatisticalTestResult.

    Args:
        original: Original test result
        method: Correction method used
        corrected_p_value: Adjusted p-value
        is_significant: Corrected significance flag

    Returns:
        New StatisticalTestResult with corrections applied
    """
    return StatisticalTestResult(
        testType=f"{original.testType}_{method.value}",
        statistic=original.statistic,
        pValue=corrected_p_value,
        isSignificant=is_significant,
        effectSize=original.effectSize,
        confidenceLevel=original.confidenceLevel,
        hypothesis=original.hypothesis,
    )


# Dispatch table for correction methods
_CORRECTION_METHODS = {
    CorrectionMethod.BONFERRONI: _bonferroni_correction,
    CorrectionMethod.HOLM: _holm_correction,
    CorrectionMethod.SIDAK: _sidak_correction,
    CorrectionMethod.FDR_BH: _fdr_bh_correction,
}


def correct_multiple_comparisons(
    test_results: dict[str, StatisticalTestResult],
    method: CorrectionMethod = CorrectionMethod.HOLM,
    alpha: float = 0.05,
) -> dict[str, StatisticalTestResult]:
    """
    Apply multiple comparison correction to statistical test results

    When testing N hypotheses, probability of at least one false positive
    increases. Correction methods adjust p-values or significance threshold.

    Args:
        test_results: Dictionary of test results (metric_id -> result)
        method: Correction method
        alpha: Family-wise error rate (default 0.05)

    Returns:
        Corrected test results with updated isSignificant flags

    Example:
        # Compare SOP A vs SOP B across 10 metrics
        raw_results = evaluation_service.compare_sops(evals_a, evals_b)
        corrected = correct_multiple_comparisons(
            raw_results["statistical_tests"],
            method=CorrectionMethod.HOLM
        )
        # Now only truly significant differences remain
    """
    if not test_results:
        return {}

    if method == CorrectionMethod.NONE:
        return test_results

    # Get correction function from dispatch table
    correction_fn = _CORRECTION_METHODS.get(method)
    if correction_fn is None:
        raise ValueError(f"Unknown correction method: {method}")

    # Extract p-values and metric IDs
    metric_ids = list(test_results.keys())
    p_values = np.array([test_results[m_id].pValue for m_id in metric_ids])
    n_tests = len(p_values)

    # Apply correction
    is_significant, corrected_p_values = correction_fn(p_values, n_tests, alpha)

    # Build corrected results
    corrected_results = {
        metric_id: _build_corrected_result(
            original=test_results[metric_id],
            method=method,
            corrected_p_value=float(corrected_p_values[i]),
            is_significant=bool(is_significant[i]),
        )
        for i, metric_id in enumerate(metric_ids)
    }

    logger.info(
        f"Multiple comparison correction ({method.value}): "
        f"{n_tests} tests, {is_significant.sum()} significant after correction"
    )

    return corrected_results


# Recommendation rules per context: list of (threshold, method) pairs
# For each context, iterate through and return first method where n <= threshold
_RECOMMENDATION_RULES: dict[str, list[tuple[int, CorrectionMethod]]] = {
    "exploratory": [
        # More permissive for hypothesis generation
        (MIN_COMPARISONS_BONFERRONI, CorrectionMethod.NONE),
        (MAX_COMPARISONS_BONFERRONI, CorrectionMethod.FDR_BH),
    ],
    "confirmatory": [
        # Strict control for final decisions
        (MIN_GROUPS_TUKEY, CorrectionMethod.BONFERRONI),
    ],
    "general": [
        (MIN_GROUPS_TUKEY, CorrectionMethod.NONE),
        (MAX_SAMPLES_PER_GROUP, CorrectionMethod.HOLM),
    ],
}

# Default fallback methods per context when no threshold matches
_FALLBACK_METHODS: dict[str, CorrectionMethod] = {
    "exploratory": CorrectionMethod.HOLM,
    "confirmatory": CorrectionMethod.HOLM,
    "general": CorrectionMethod.FDR_BH,
}


def _get_method_for_context(
    n_comparisons: int, context: str
) -> CorrectionMethod:
    """Get recommended correction method for a specific context.

    Args:
        n_comparisons: Number of statistical tests
        context: Context type

    Returns:
        Recommended correction method
    """
    rules = _RECOMMENDATION_RULES.get(context, _RECOMMENDATION_RULES["general"])
    for threshold, method in rules:
        if n_comparisons <= threshold:
            return method
    return _FALLBACK_METHODS.get(context, CorrectionMethod.FDR_BH)


def recommend_correction_method(
    n_comparisons: int, context: str = "general"
) -> CorrectionMethod:
    """
    Recommend correction method based on number of comparisons

    Args:
        n_comparisons: Number of statistical tests performed
        context: Context ("exploratory", "confirmatory", "general")

    Returns:
        Recommended correction method
    """
    if n_comparisons == 1:
        return CorrectionMethod.NONE
    return _get_method_for_context(n_comparisons, context)
