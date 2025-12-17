"""Shared test helpers for unit tests"""

from siare.core.models import (
    AggregatedMetric,
    AggregationMethod,
    ProcessConfig,
    SOPGene,
)


def create_test_gene(
    sop_id: str,
    version: str,
    config: ProcessConfig,
    quality_score: float = 0.8,
    generation: int | None = None,
) -> SOPGene:
    """
    Helper to create a test SOPGene with proper aggregated metrics

    Args:
        sop_id: SOP identifier
        version: Version string
        config: ProcessConfig snapshot
        quality_score: Quality score for metrics (0.0-1.0)
        generation: Optional generation number

    Returns:
        Configured SOPGene
    """
    # Create aggregated metrics with proper AggregatedMetric objects
    aggregated_metrics = {
        "accuracy": AggregatedMetric(
            metricId="accuracy",
            mean=quality_score,
            median=quality_score,
            trimmedMean=quality_score,
            confidenceInterval=(quality_score - 0.05, quality_score + 0.05),
            standardDeviation=0.02,
            standardError=0.01,
            sampleSize=10,
            outliers=None,
            aggregationMethod=AggregationMethod.MEAN,
            rawValues=None,
        ),
        "weighted_aggregate": AggregatedMetric(
            metricId="weighted_aggregate",
            mean=quality_score,
            median=quality_score,
            trimmedMean=quality_score,
            confidenceInterval=(quality_score - 0.05, quality_score + 0.05),
            standardDeviation=0.02,
            standardError=0.01,
            sampleSize=10,
            outliers=None,
            aggregationMethod=AggregationMethod.WEIGHTED,
            rawValues=None,
        ),
    }

    return SOPGene(
        sopId=sop_id,
        version=version,
        promptGenomeId="test-genome",
        promptGenomeVersion="1.0.0",
        configSnapshot=config,
        evaluations=[],
        aggregatedMetrics=aggregated_metrics,
        generation=generation,
    )
