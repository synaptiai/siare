"""Gene Pool Service - Manages SOPGenes and Pareto frontier"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from siare.core.models import (
    MetaGene,
    ParetoFlags,
    SOPGene,
)
from siare.services.retry_handler import RetryHandler
from siare.utils.file_utils import atomic_write_json

logger = logging.getLogger(__name__)


class GenePool:
    """
    Manages population of SOPGenes and MetaGenes

    Features:
    - Persistent storage of genes
    - Pareto frontier calculation and tracking
    - Gene querying and filtering
    - Integration with QD Grid
    """

    def __init__(
        self,
        storage_path: str | None = None,
        retry_handler: RetryHandler | None = None,
    ):
        """
        Initialize Gene Pool

        Args:
            storage_path: Optional path for JSON persistence
            retry_handler: Retry handler for file I/O operations (creates default if None)
        """
        self.storage_path = Path(storage_path) if storage_path else None
        self.retry_handler = retry_handler or RetryHandler()

        # Gene storage: {sopId: {version: SOPGene}}
        self._sop_genes: dict[str, dict[str, SOPGene]] = {}
        self._meta_genes: dict[str, dict[str, MetaGene]] = {}

        # Pareto frontier tracking
        self._pareto_sets: dict[str, set[tuple[str, str]]] = {}  # setId -> {(sopId, version)}
        self._current_pareto_set_id: str = "default"

        # Generation tracking for temporal queries (RECENT strategy)
        self._generation_index: dict[
            int, list[tuple[str, str]]
        ] = {}  # generation -> [(sopId, version)]
        self._current_generation: int = 0  # Tracks latest generation

        # Statistics
        self.stats = {
            "total_sop_genes": 0,
            "total_meta_genes": 0,
            "pareto_optimal_count": 0,
            "generations_tracked": 0,
        }

        # Load from disk if path provided
        if self.storage_path:
            self._load_from_disk()

    # =========================================================================
    # SOPGene Management
    # =========================================================================

    def add_sop_gene(self, gene: SOPGene, generation: int | None = None) -> None:
        """
        Add a SOPGene to the pool with generation tracking

        Args:
            gene: SOPGene to add
            generation: Generation number (uses current if None)
        """
        if gene.sopId not in self._sop_genes:
            self._sop_genes[gene.sopId] = {}

        self._sop_genes[gene.sopId][gene.version] = gene

        # Track generation
        if generation is None:
            generation = self._current_generation

        # Store generation in gene if not already set
        if not hasattr(gene, "generation") or gene.generation is None:
            gene.generation = generation

        # Update generation index (avoid duplicates)
        if generation not in self._generation_index:
            self._generation_index[generation] = []

        gene_key = (gene.sopId, gene.version)
        if gene_key not in self._generation_index[generation]:
            self._generation_index[generation].append(gene_key)

        # Update current generation
        self._current_generation = max(self._current_generation, generation)

        # Update stats
        self.stats["total_sop_genes"] = sum(len(v) for v in self._sop_genes.values())
        self.stats["generations_tracked"] = len(self._generation_index)

        self._persist()

    def get_sop_gene(self, sop_id: str, version: str | None = None) -> SOPGene | None:
        """
        Get a SOPGene

        Args:
            sop_id: SOP identifier
            version: Version (if None, returns latest)

        Returns:
            SOPGene or None
        """
        if sop_id not in self._sop_genes:
            return None

        versions = self._sop_genes[sop_id]
        if version:
            return versions.get(version)

        # Return latest version
        if not versions:
            return None
        latest_version = max(versions.keys(), key=self._version_key)
        return versions[latest_version]

    def list_sop_genes(
        self,
        pareto_optimal_only: bool = False,
        pareto_set_id: str | None = None,
        tags: list[str] | None = None,
        min_quality: float | None = None,
    ) -> list[SOPGene]:
        """
        List SOPGenes with optional filters

        Args:
            pareto_optimal_only: Only return Pareto optimal genes
            pareto_set_id: Filter by Pareto set
            tags: Filter by tags
            min_quality: Minimum quality threshold

        Returns:
            List of SOPGenes
        """
        genes: list[SOPGene] = []

        for _sop_id, versions in self._sop_genes.items():
            for _version, gene in versions.items():
                # Apply filters
                if pareto_optimal_only:
                    if not gene.frontierFlags or not gene.frontierFlags.isParetoOptimal:
                        continue
                    if pareto_set_id and gene.frontierFlags.paretoSetId != pareto_set_id:
                        continue

                if tags:
                    if not gene.tags or not any(tag in gene.tags for tag in tags):
                        continue

                if min_quality is not None:
                    # Check if any metric meets threshold (use mean values)
                    max_metric = (
                        max(agg.mean for agg in gene.aggregatedMetrics.values())
                        if gene.aggregatedMetrics
                        else 0.0
                    )
                    if max_metric < min_quality:
                        continue

                genes.append(gene)

        return genes

    def get_pareto_frontier(
        self, metric_ids: list[str], pareto_set_id: str | None = None
    ) -> list[SOPGene]:
        """
        Get Pareto frontier for specified metrics

        Args:
            metric_ids: Metrics to use for Pareto calculation
            pareto_set_id: Specific Pareto set (uses current if None)

        Returns:
            List of Pareto optimal SOPGenes
        """
        set_id = pareto_set_id or self._current_pareto_set_id

        # Get all genes with frontier flags matching this set
        pareto_genes: list[SOPGene] = []
        for gene in self.list_sop_genes():
            if gene.frontierFlags and gene.frontierFlags.isParetoOptimal:
                if gene.frontierFlags.paretoSetId == set_id:
                    pareto_genes.append(gene)

        return pareto_genes

    def update_pareto_frontier(
        self,
        metric_ids: list[str],
        pareto_set_id: str | None = None,
        maximize_all: bool = True,
    ) -> int:
        """
        Recalculate Pareto frontier for all genes

        Args:
            metric_ids: Metrics to use for Pareto calculation
            pareto_set_id: Set ID for this frontier (creates new if None)
            maximize_all: If True, maximize all metrics (else minimize)

        Returns:
            Number of Pareto optimal genes found
        """
        # Create new Pareto set ID if not provided
        if pareto_set_id is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            pareto_set_id = f"pareto_{timestamp}"

        self._current_pareto_set_id = pareto_set_id

        # Collect all genes with metrics
        all_genes = self.list_sop_genes()

        # Extract metric vectors
        gene_metrics: list[list[float]] = []
        gene_refs: list[tuple[str, str]] = []

        for gene in all_genes:
            # Get metrics for this gene (use mean values for Pareto comparison)
            metrics = [gene.get_metric_mean(m_id) for m_id in metric_ids]

            # Skip if missing metrics
            if any(
                m == 0.0 and m_id not in gene.aggregatedMetrics
                for m, m_id in zip(metrics, metric_ids, strict=False)
            ):
                continue

            gene_metrics.append(metrics)
            gene_refs.append((gene.sopId, gene.version))

        if not gene_metrics:
            return 0

        # Calculate Pareto frontier with CI tie-breaking
        pareto_indices = self._calculate_pareto_frontier(
            np.array(gene_metrics),
            maximize=maximize_all,
            gene_refs=gene_refs,
            metric_ids=metric_ids,
        )

        # Update genes with Pareto flags
        pareto_genes: set[tuple[str, str]] = set()
        for idx in pareto_indices:
            sop_id, version = gene_refs[idx]
            pareto_genes.add((sop_id, version))

        # Update all genes
        for gene in all_genes:
            gene_ref = (gene.sopId, gene.version)

            if gene_ref in pareto_genes:
                gene.frontierFlags = ParetoFlags(isParetoOptimal=True, paretoSetId=pareto_set_id)
            else:
                # Mark as non-Pareto
                gene.frontierFlags = ParetoFlags(isParetoOptimal=False, paretoSetId=None)

        # Store Pareto set
        self._pareto_sets[pareto_set_id] = pareto_genes

        # Update stats
        self.stats["pareto_optimal_count"] = len(pareto_genes)

        self._persist()

        return len(pareto_genes)

    def _calculate_pareto_frontier(
        self,
        metrics: NDArray[np.float64],
        maximize: bool = True,
        gene_refs: list[tuple[str, str]] | None = None,
        metric_ids: list[str] | None = None,
        mean_tolerance: float = 0.01,
        use_ci_tiebreaker: bool = True,
    ) -> list[int]:
        """
        Calculate Pareto frontier with optional CI tie-breaking.

        When two genes have similar means (within tolerance), prefer the one
        with narrower confidence intervals (more reliable performance).

        Args:
            metrics: (N, M) array of metric values
            maximize: If True, maximize all objectives
            gene_refs: Optional list of (sopId, version) tuples for CI lookup
            metric_ids: Optional metric IDs for CI lookup
            mean_tolerance: Relative tolerance for considering means "similar" (default 1%)
            use_ci_tiebreaker: Whether to use CI tie-breaking (default True)

        Returns:
            List of indices of Pareto optimal solutions
        """
        n_points = metrics.shape[0]

        if n_points == 0:
            return []

        # For minimization, negate metrics
        if not maximize:
            metrics = -metrics

        # Track which points are dominated
        is_pareto = np.ones(n_points, dtype=bool)

        for i in range(n_points):
            if not is_pareto[i]:
                continue

            # Check if point i dominates any other points
            # A dominates B if A >= B in all dimensions and A > B in at least one
            for j in range(n_points):
                if i == j or not is_pareto[j]:
                    continue

                # Standard Pareto dominance check
                if np.all(metrics[i] >= metrics[j]) and np.any(metrics[i] > metrics[j]):
                    is_pareto[j] = False
                    continue

                # CI Tie-breaking for "tied" genes (means within tolerance)
                if use_ci_tiebreaker and gene_refs and metric_ids:
                    # Calculate relative difference for all metrics
                    max_vals = np.maximum(np.abs(metrics[i]), np.abs(metrics[j]))
                    max_vals = np.where(max_vals == 0, 1.0, max_vals)
                    relative_diff = np.abs(metrics[i] - metrics[j]) / max_vals

                    # If all metrics are within tolerance, use CI width to tie-break
                    if np.all(relative_diff <= mean_tolerance):
                        ci_width_i = self._get_ci_width_for_comparison(
                            gene_refs[i], metric_ids
                        )
                        ci_width_j = self._get_ci_width_for_comparison(
                            gene_refs[j], metric_ids
                        )

                        # Prefer narrower CI (more reliable estimate)
                        if ci_width_j < ci_width_i and ci_width_j != float("inf"):
                            # j has narrower CI, so i is "dominated" by tie-breaker
                            is_pareto[i] = False
                            break
                        if ci_width_i < ci_width_j and ci_width_i != float("inf"):
                            # i has narrower CI, so j is "dominated" by tie-breaker
                            is_pareto[j] = False

        return list(np.where(is_pareto)[0])

    def _get_ci_width_for_comparison(
        self,
        gene_ref: tuple[str, str],
        metric_ids: list[str],
    ) -> float:
        """
        Get average CI width for a gene across specified metrics.

        Used for tie-breaking in Pareto frontier calculation when genes have
        similar mean performance. Narrower CIs indicate more reliable estimates.

        Args:
            gene_ref: (sopId, version) tuple identifying the gene
            metric_ids: Metrics to average CI width across

        Returns:
            Average CI width (inf if any CI is unavailable)
        """
        sop_id, version = gene_ref
        gene = self.get_sop_gene(sop_id, version)

        if gene is None:
            return float("inf")

        total_width = 0.0
        count = 0

        for metric_id in metric_ids:
            width = gene.get_metric_ci_width(metric_id)
            if width == float("inf"):
                return float("inf")  # If any CI is missing, can't use CI tie-breaking
            total_width += width
            count += 1

        return total_width / count if count > 0 else float("inf")

    # =========================================================================
    # MetaGene Management
    # =========================================================================

    def add_meta_gene(self, gene: MetaGene) -> None:
        """Add a MetaGene to the pool"""
        if gene.metaId not in self._meta_genes:
            self._meta_genes[gene.metaId] = {}

        self._meta_genes[gene.metaId][gene.version] = gene
        self.stats["total_meta_genes"] = sum(len(v) for v in self._meta_genes.values())

        self._persist()

    def get_meta_gene(self, meta_id: str, version: str | None = None) -> MetaGene | None:
        """Get a MetaGene"""
        if meta_id not in self._meta_genes:
            return None

        versions = self._meta_genes[meta_id]
        if version:
            return versions.get(version)

        # Return latest
        if not versions:
            return None
        latest_version = max(versions.keys(), key=self._version_key)
        return versions[latest_version]

    def list_meta_genes(self) -> list[MetaGene]:
        """List all MetaGenes"""
        genes: list[MetaGene] = []
        for _meta_id, versions in self._meta_genes.items():
            for _version, gene in versions.items():
                genes.append(gene)
        return genes

    # =========================================================================
    # Genealogy Tracking
    # =========================================================================

    def get_lineage(self, sop_id: str, version: str, max_depth: int = 10) -> list[SOPGene]:
        """
        Get lineage (ancestors) of a SOPGene

        Args:
            sop_id: SOP identifier
            version: Version
            max_depth: Maximum depth to traverse

        Returns:
            List of SOPGenes in lineage (oldest to newest)
        """
        lineage: list[SOPGene] = []
        current = self.get_sop_gene(sop_id, version)
        depth = 0

        while current and depth < max_depth:
            lineage.append(current)

            # Get parent
            if current.parent:
                parent_id = current.parent.get("sopId")
                parent_version = current.parent.get("version")
                if parent_id and parent_version:
                    current = self.get_sop_gene(parent_id, parent_version)
                else:
                    break
            else:
                break

            depth += 1

        return list(reversed(lineage))  # Oldest to newest

    def get_descendants(self, sop_id: str, version: str) -> list[SOPGene]:
        """
        Get all descendants of a SOPGene

        Args:
            sop_id: SOP identifier
            version: Version

        Returns:
            List of descendant SOPGenes
        """
        descendants: list[SOPGene] = []

        # Search all genes for children
        for gene in self.list_sop_genes():
            if gene.parent:
                parent_id = gene.parent.get("sopId")
                parent_version = gene.parent.get("version")
                if parent_id == sop_id and parent_version == version:
                    descendants.append(gene)

        return descendants

    # =========================================================================
    # Statistics & Analysis
    # =========================================================================

    def get_stats(self) -> dict[str, Any]:
        """Get gene pool statistics"""
        return self.stats.copy()

    def get_diversity_stats(self) -> dict[str, Any]:
        """Calculate diversity statistics"""
        genes = self.list_sop_genes()

        if not genes:
            return {
                "unique_sop_ids": 0,
                "total_versions": 0,
                "avg_versions_per_sop": 0.0,
                "complexity_range": (0.0, 0.0),
                "avg_complexity": 0.0,
            }

        unique_sops = {g.sopId for g in genes}

        # Calculate complexity stats
        complexities = [
            g.qdFeatures.complexity
            for g in genes
            if g.qdFeatures and g.qdFeatures.complexity is not None
        ]

        return {
            "unique_sop_ids": len(unique_sops),
            "total_versions": len(genes),
            "avg_versions_per_sop": len(genes) / len(unique_sops) if unique_sops else 0.0,
            "complexity_range": (min(complexities), max(complexities))
            if complexities
            else (0.0, 0.0),
            "avg_complexity": sum(complexities) / len(complexities) if complexities else 0.0,
        }

    # =========================================================================
    # Generation Tracking (for RECENT selection strategy)
    # =========================================================================

    def increment_generation(self) -> int:
        """
        Increment generation counter

        Returns:
            New generation number
        """
        self._current_generation += 1
        return self._current_generation

    def get_current_generation(self) -> int:
        """Get current generation number"""
        return self._current_generation

    def get_genes_from_generation(self, generation: int) -> list[SOPGene]:
        """
        Get all genes from a specific generation

        Args:
            generation: Generation number

        Returns:
            List of SOPGenes from that generation
        """
        if generation not in self._generation_index:
            return []

        genes: list[SOPGene] = []
        for sop_id, version in self._generation_index[generation]:
            gene = self.get_sop_gene(sop_id, version)
            if gene:
                genes.append(gene)

        return genes

    def get_genes_from_recent_generations(
        self, lookback: int, min_quality: float | None = None
    ) -> list[SOPGene]:
        """
        Get genes from last N generations

        Args:
            lookback: Number of generations to look back
            min_quality: Optional minimum quality threshold (weighted_aggregate)

        Returns:
            List of SOPGenes from recent generations
        """
        cutoff_gen = max(0, self._current_generation - lookback)

        genes: list[SOPGene] = []
        for gen in range(cutoff_gen, self._current_generation + 1):
            genes.extend(self.get_genes_from_generation(gen))

        # Apply quality filter if specified
        if min_quality is not None:
            genes = [
                g
                for g in genes
                if "weighted_aggregate" in g.aggregatedMetrics
                and g.aggregatedMetrics["weighted_aggregate"].mean >= min_quality
            ]

        return genes

    def get_generation_stats(self) -> dict[str, Any]:
        """Get statistics about generation distribution"""
        if not self._generation_index:
            return {
                "current_generation": self._current_generation,
                "total_generations": 0,
                "genes_per_generation": {},
                "avg_genes_per_generation": 0.0,
            }

        genes_per_gen = {gen: len(genes) for gen, genes in self._generation_index.items()}

        return {
            "current_generation": self._current_generation,
            "total_generations": len(self._generation_index),
            "genes_per_generation": genes_per_gen,
            "avg_genes_per_generation": sum(genes_per_gen.values()) / len(genes_per_gen)
            if genes_per_gen
            else 0.0,
        }

    # =========================================================================
    # Persistence
    # =========================================================================

    def _persist(self) -> None:
        """
        Persist gene pool to disk with atomic writes

        All data is written atomically to prevent partial state corruption.
        Uses retry handler to handle transient failures.

        Raises:
            RuntimeError: If persistence fails after all retries
        """
        if not self.storage_path:
            return

        try:
            # Create directory with retry
            storage_path = self.storage_path  # Local variable for type narrowing
            self.retry_handler.execute_with_retry(
                lambda: storage_path.mkdir(parents=True, exist_ok=True),
                retry_config=RetryHandler.CONFIG_RETRY_CONFIG,
                component="GenePool",
                operation="mkdir",
            )

            # Prepare all data first (fail fast if serialization issues)
            sop_genes_data: dict[str, dict[str, Any]] = {}
            for sop_id, versions in self._sop_genes.items():
                sop_genes_data[sop_id] = {
                    version: gene.model_dump(mode="json") for version, gene in versions.items()
                }

            meta_genes_data: dict[str, dict[str, Any]] = {}
            for meta_id, versions in self._meta_genes.items():
                meta_genes_data[meta_id] = {
                    version: gene.model_dump(mode="json") for version, gene in versions.items()
                }

            pareto_sets_data: dict[str, list[tuple[str, str]]] = {
                set_id: list(genes)  # Convert set to list for JSON
                for set_id, genes in self._pareto_sets.items()
            }

            # Atomically write all files
            # If any write fails, previous files remain intact
            atomic_write_json(
                self.storage_path / "sop_genes.json", sop_genes_data, self.retry_handler, "GenePool"
            )
            atomic_write_json(
                self.storage_path / "meta_genes.json",
                meta_genes_data,
                self.retry_handler,
                "GenePool",
            )
            atomic_write_json(
                self.storage_path / "pareto_sets.json",
                pareto_sets_data,
                self.retry_handler,
                "GenePool",
            )
            atomic_write_json(
                self.storage_path / "stats.json", self.stats, self.retry_handler, "GenePool"
            )

            logger.debug(f"Successfully persisted gene pool to {self.storage_path}")

        except Exception as e:
            logger.exception("Failed to persist gene pool")
            raise RuntimeError(f"Gene pool persistence failed: {e}") from e

    def _load_json_file(
        self, file_path: Path, operation_name: str
    ) -> dict[str, Any] | None:
        """Load JSON file with retry handling.

        Args:
            file_path: Path to JSON file
            operation_name: Name for logging

        Returns:
            Parsed JSON data or None if file doesn't exist
        """
        if not file_path.exists():
            return None

        def _load() -> dict[str, Any]:
            with file_path.open() as f:
                return json.load(f)

        return self.retry_handler.execute_with_retry(
            _load,
            retry_config=RetryHandler.CONFIG_RETRY_CONFIG,
            component="GenePool",
            operation=operation_name,
        )

    def _load_sop_genes_from_data(self, data: dict[str, Any]) -> tuple[int, int]:
        """Load SOPGenes from parsed JSON data.

        Args:
            data: Parsed JSON data

        Returns:
            Tuple of (loaded_count, error_count)
        """
        loaded_count = 0
        error_count = 0

        for sop_id, versions in data.items():
            self._sop_genes[sop_id] = {}
            for version, gene_data in versions.items():
                try:
                    self._sop_genes[sop_id][version] = SOPGene(**gene_data)
                    loaded_count += 1
                except Exception as e:
                    error_count += 1
                    logger.warning(f"Failed to load SOPGene {sop_id}@{version}: {e}")

        return loaded_count, error_count

    def _load_meta_genes_from_data(self, data: dict[str, Any]) -> tuple[int, int]:
        """Load MetaGenes from parsed JSON data.

        Args:
            data: Parsed JSON data

        Returns:
            Tuple of (loaded_count, error_count)
        """
        loaded_count = 0
        error_count = 0

        for meta_id, versions in data.items():
            self._meta_genes[meta_id] = {}
            for version, gene_data in versions.items():
                try:
                    self._meta_genes[meta_id][version] = MetaGene(**gene_data)
                    loaded_count += 1
                except Exception as e:
                    error_count += 1
                    logger.warning(f"Failed to load MetaGene {meta_id}@{version}: {e}")

        return loaded_count, error_count

    def _load_pareto_sets_from_data(self, data: dict[str, Any]) -> int:
        """Load Pareto sets from parsed JSON data.

        Args:
            data: Parsed JSON data

        Returns:
            Number of loaded sets
        """
        for set_id, genes_list in data.items():
            try:
                self._pareto_sets[set_id] = {tuple(g) for g in genes_list}
            except Exception as e:
                logger.warning(f"Failed to load Pareto set {set_id}: {e}")

        return len(self._pareto_sets)

    def _validate_loaded_state(self) -> None:
        """Validate and fix loaded state consistency."""
        actual_sop_count = sum(len(v) for v in self._sop_genes.values())
        if self.stats.get("total_sop_genes") != actual_sop_count:
            logger.warning(
                f"Stats mismatch: expected {self.stats.get('total_sop_genes')} SOPGenes, "
                f"loaded {actual_sop_count}. Updating stats."
            )
            self.stats["total_sop_genes"] = actual_sop_count

    def _load_from_disk(self) -> None:
        """
        Load gene pool from disk with error handling.

        Validates loaded data and logs warnings for corrupted entries.
        Uses retry handler to handle transient file system failures.

        Raises:
            RuntimeError: If critical files cannot be loaded
        """
        if not self.storage_path or not self.storage_path.exists():
            logger.debug("No storage path configured or path doesn't exist, skipping load")
            return

        try:
            # Load SOPGenes (critical)
            sop_data = self._load_json_file(
                self.storage_path / "sop_genes.json", "load_sop_genes"
            )
            if sop_data:
                loaded, errors = self._load_sop_genes_from_data(sop_data)
                logger.info(f"Loaded {loaded} SOPGenes ({errors} errors)")

            # Load MetaGenes (non-critical)
            try:
                meta_data = self._load_json_file(
                    self.storage_path / "meta_genes.json", "load_meta_genes"
                )
                if meta_data:
                    loaded, errors = self._load_meta_genes_from_data(meta_data)
                    logger.info(f"Loaded {loaded} MetaGenes ({errors} errors)")
            except Exception as e:
                logger.warning(f"Failed to load meta_genes.json: {e}")

            # Load Pareto sets (non-critical)
            try:
                pareto_data = self._load_json_file(
                    self.storage_path / "pareto_sets.json", "load_pareto_sets"
                )
                if pareto_data:
                    count = self._load_pareto_sets_from_data(pareto_data)
                    logger.info(f"Loaded {count} Pareto sets")
            except Exception as e:
                logger.warning(f"Failed to load pareto_sets.json: {e}")

            # Load stats (non-critical)
            try:
                stats_data = self._load_json_file(
                    self.storage_path / "stats.json", "load_stats"
                )
                if stats_data:
                    self.stats = stats_data
                    logger.debug(f"Loaded stats: {self.stats}")
            except Exception as e:
                logger.warning(f"Failed to load stats.json: {e}")

            # Validate loaded state
            self._validate_loaded_state()

        except Exception as e:
            logger.exception("Critical failure loading gene pool from disk")
            raise RuntimeError(f"Failed to load gene pool: {e}") from e

    def _version_key(self, version: str) -> tuple[int, int, int, str]:
        """Convert version string to tuple for comparison.

        Handles versions like "1.0.0", "1.1.0-abc123", etc.
        The suffix is included as a tiebreaker for versions with same
        major.minor.patch numbers.
        """
        try:
            # Split off any suffix after dash
            base, _, suffix = version.partition("-")
            parts = base.split(".")

            major = int(parts[0]) if len(parts) > 0 else 0
            minor = int(parts[1]) if len(parts) > 1 else 0
            patch = int(parts[2]) if len(parts) > 2 else 0

            return (major, minor, patch, suffix)
        except (ValueError, AttributeError):
            return (0, 0, 0, "")

    def clear_all(self) -> None:
        """Clear all data (for testing)"""
        self._sop_genes.clear()
        self._meta_genes.clear()
        self._pareto_sets.clear()
        self.stats = {
            "total_sop_genes": 0,
            "total_meta_genes": 0,
            "pareto_optimal_count": 0,
            "generations_tracked": 0,
        }
