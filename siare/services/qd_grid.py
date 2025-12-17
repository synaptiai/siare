"""Quality-Diversity Grid - Maintains diverse, high-quality SOPs"""

import logging
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import numpy.typing as npt
from sklearn.decomposition import PCA

from siare.core.models import (
    ProcessConfig,
    PromptGenome,
    QDCell,
    QDFeatures,
    SOPGene,
)
from siare.utils.sampling import quality_weighted_sample


logger = logging.getLogger(__name__)


# ============================================================================
# Sentence Transformers Integration
# ============================================================================

# Sentence-transformers model (lazy loaded, thread-safe)
_embedding_model: Any = None
_embedding_model_lock = threading.Lock()
_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

try:
    from sentence_transformers import SentenceTransformer
    sentence_transformers_available = True
except ImportError:
    SentenceTransformer = None  # type: ignore
    sentence_transformers_available = False


def _get_embedding_model() -> Any:
    """
    Lazy load sentence-transformers model (thread-safe).

    Uses double-checked locking pattern to avoid lock contention
    after initialization while ensuring thread safety during init.
    """
    global _embedding_model
    if _embedding_model is None:
        with _embedding_model_lock:
            # Double-check after acquiring lock
            if _embedding_model is None and sentence_transformers_available and SentenceTransformer is not None:
                _embedding_model = SentenceTransformer(_EMBEDDING_MODEL_NAME)
                logger.info(f"Loaded embedding model: {_EMBEDDING_MODEL_NAME}")
    return _embedding_model


# ============================================================================
# Constants
# ============================================================================

# Complexity feature weights
COMPLEXITY_WEIGHT_NUM_ROLES = 0.4
COMPLEXITY_WEIGHT_AVG_DEPTH = 0.3
COMPLEXITY_WEIGHT_NUM_EDGES = 0.2
COMPLEXITY_WEIGHT_AVG_PROMPT = 0.1

# Normalization factors for complexity calculation
NORM_FACTOR_MAX_ROLES = 10.0
NORM_FACTOR_MAX_DEPTH = 5.0
NORM_FACTOR_MAX_EDGES = 20.0
NORM_FACTOR_MAX_PROMPT_LENGTH = 2000.0


# ============================================================================
# Cell ID Helper
# ============================================================================


@dataclass(frozen=True)
class CellID:
    """
    Structured representation of a QD Grid cell ID.

    Provides safe parsing and formatting of cell IDs like "complexity_5-diversity_0_7".
    """

    complexity_bin: int
    diversity_bins: tuple[int, ...]

    # Pattern for parsing cell IDs: "complexity_X-diversity_0_Y[-diversity_1_Z...]"
    _PATTERN = re.compile(r"complexity_(\d+)(?:-diversity_\d+_(\d+))+")

    @classmethod
    def from_string(cls, cell_id: str) -> "CellID":
        """
        Parse cell ID string into structured CellID.

        Args:
            cell_id: String like "complexity_5-diversity_0_7"

        Returns:
            CellID instance

        Raises:
            ValueError: If cell_id format is invalid
        """
        # Extract complexity bin
        if not cell_id.startswith("complexity_"):
            raise ValueError(f"Invalid cell ID format: {cell_id}")

        parts = cell_id.split("-")
        if len(parts) < 2:
            raise ValueError(f"Invalid cell ID format (missing diversity): {cell_id}")

        try:
            comp_bin = int(parts[0].split("_")[1])
        except (IndexError, ValueError) as e:
            raise ValueError(f"Invalid complexity bin in cell ID: {cell_id}") from e

        # Extract diversity bins
        div_bins: list[int] = []
        for part in parts[1:]:
            if not part.startswith("diversity_"):
                raise ValueError(f"Invalid diversity part in cell ID: {part}")
            try:
                # Format: diversity_<dim>_<bin>
                bin_idx = int(part.split("_")[2])
                div_bins.append(bin_idx)
            except (IndexError, ValueError) as e:
                raise ValueError(f"Invalid diversity bin in cell ID: {part}") from e

        return cls(complexity_bin=comp_bin, diversity_bins=tuple(div_bins))

    def __str__(self) -> str:
        """Generate cell ID string from structured data."""
        parts = [f"complexity_{self.complexity_bin}"]
        for i, bin_idx in enumerate(self.diversity_bins):
            parts.append(f"diversity_{i}_{bin_idx}")
        return "-".join(parts)


# ============================================================================
# Feature Calculation
# ============================================================================


def calculate_complexity(sop: ProcessConfig, prompt_genome: PromptGenome) -> float:
    """
    Calculate structural complexity feature

    Formula from QD_ALGORITHM_SPEC.md:
    complexity = 0.4 * numRoles + 0.3 * avgChainDepth + 0.2 * numEdges + 0.1 * avgPromptLength
    Normalized to [0, 1]
    """
    num_roles = len(sop.roles)
    num_edges = len(sop.graph)

    # Calculate average chain depth via topological sort
    avg_depth = _calculate_avg_chain_depth(sop)

    # Calculate average prompt length
    prompt_lengths: list[int] = []
    for role in sop.roles:
        if role.promptRef in prompt_genome.rolePrompts:
            prompt = prompt_genome.rolePrompts[role.promptRef].content
            prompt_lengths.append(len(prompt))

    avg_prompt_len = sum(prompt_lengths) / len(prompt_lengths) if prompt_lengths else 0.0

    # Normalize each component
    norm_roles = min(num_roles / NORM_FACTOR_MAX_ROLES, 1.0)
    norm_depth = min(avg_depth / NORM_FACTOR_MAX_DEPTH, 1.0)
    norm_edges = min(num_edges / NORM_FACTOR_MAX_EDGES, 1.0)
    norm_prompt = min(avg_prompt_len / NORM_FACTOR_MAX_PROMPT_LENGTH, 1.0)

    return (
        COMPLEXITY_WEIGHT_NUM_ROLES * norm_roles
        + COMPLEXITY_WEIGHT_AVG_DEPTH * norm_depth
        + COMPLEXITY_WEIGHT_NUM_EDGES * norm_edges
        + COMPLEXITY_WEIGHT_AVG_PROMPT * norm_prompt
    )


def _calculate_avg_chain_depth(sop: ProcessConfig) -> float:
    """Calculate average depth of execution chains in DAG"""
    from collections import defaultdict, deque

    # Build adjacency list
    adj: dict[str, list[str]] = defaultdict(list)
    in_degree: dict[str, int] = defaultdict(int)
    nodes: set[str] = set()

    for edge in sop.graph:
        from_nodes = edge.from_ if isinstance(edge.from_, list) else [edge.from_]
        to_node = edge.to
        nodes.update(from_nodes)
        nodes.add(to_node)

        for from_node in from_nodes:
            adj[from_node].append(to_node)
            in_degree[to_node] += 1

    # Topological sort with level tracking
    queue: deque[tuple[str, int]] = deque([(node, 0) for node in nodes if in_degree[node] == 0])
    depths: list[int] = []

    while queue:
        node, depth = queue.popleft()
        depths.append(depth)

        for neighbor in adj[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append((neighbor, depth + 1))

    return sum(depths) / len(depths) if depths else 1.0


def calculate_diversity_embedding(
    sop: ProcessConfig,
    prompt_genome: PromptGenome,
) -> npt.NDArray[np.float64]:
    """
    Generate embedding representing prompt diversity

    Uses sentence-transformers (all-MiniLM-L6-v2) for semantic embeddings.
    Raises ImportError if sentence-transformers not available.

    Args:
        sop: ProcessConfig to embed
        prompt_genome: PromptGenome with prompt contents

    Returns:
        Normalized embedding vector

    Raises:
        ImportError: If sentence-transformers not available
    """
    # Concatenate all prompts
    prompt_texts: list[str] = []
    for role in sop.roles:
        if role.promptRef in prompt_genome.rolePrompts:
            prompt = prompt_genome.rolePrompts[role.promptRef].content
            prompt_texts.append(f"[{role.id}] {prompt}")

    # Add tool information
    tool_text = " ".join([f"tool:{tool}" for tool in sop.tools])
    prompt_texts.append(tool_text)

    combined = "\n\n".join(prompt_texts)

    # Generate semantic embedding using sentence-transformers (384 dims)
    embedding = _simple_text_embedding(combined, dim=384)

    # Normalize to unit length
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return embedding


def _simple_text_embedding(text: str, dim: int = 384) -> npt.NDArray[np.float64]:
    """
    Generate text embedding using sentence-transformers.

    Args:
        text: Text to embed
        dim: Expected embedding dimension (384 for all-MiniLM-L6-v2)

    Returns:
        Embedding vector of shape (dim,)

    Raises:
        ImportError: If sentence-transformers not available
    """
    model = _get_embedding_model()

    if model is not None:
        # Use real sentence-transformers
        embedding = model.encode(text, convert_to_numpy=True)
        # Truncate/pad if dimensions don't match
        if len(embedding) != dim:
            if len(embedding) > dim:
                embedding = embedding[:dim]
            else:
                embedding = np.pad(embedding, (0, dim - len(embedding)))
        return embedding

    # No model available - FAIL LOUDLY (no silent fallback)
    raise ImportError(
        "sentence-transformers required for embeddings. "
        "Install with: pip install sentence-transformers"
    )


# ============================================================================
# Embedding Reduction
# ============================================================================


class EmbeddingReducer:
    """Reduces high-dimensional embeddings to lower dimensions for QD Grid"""

    def __init__(self, method: str = "PCA", target_dimensions: int = 2):
        """
        Initialize reducer

        Args:
            method: "PCA" or "UMAP"
            target_dimensions: Target number of dimensions
        """
        self.method = method
        self.target_dimensions = target_dimensions
        self.reducer: Any = None
        self.is_fitted = False

    def fit(self, embeddings: npt.NDArray[np.float64]) -> None:
        """
        Fit reducer on embeddings

        Args:
            embeddings: (N, D) array of embeddings
        """
        if self.method == "PCA":
            self.reducer = PCA(n_components=self.target_dimensions)
            self.reducer.fit(embeddings)
            self.is_fitted = True
        elif self.method == "UMAP":
            # Requires umap-learn package
            try:
                from umap import UMAP  # type: ignore

                self.reducer = UMAP(n_components=self.target_dimensions)  # type: ignore
                self.reducer.fit(embeddings)  # type: ignore
                self.is_fitted = True
            except ImportError:
                raise ImportError("UMAP requires umap-learn package: pip install umap-learn")
        else:
            raise ValueError(f"Unknown reduction method: {self.method}")

    def transform(self, embedding: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Transform single embedding

        Args:
            embedding: (D,) array

        Returns:
            (target_dimensions,) array
        """
        if not self.is_fitted:
            raise RuntimeError("Reducer not fitted. Call fit() first.")

        # Reshape for sklearn
        embedding_2d = embedding.reshape(1, -1)
        reduced = self.reducer.transform(embedding_2d)
        return reduced.flatten()

    def fit_transform(self, embeddings: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Fit and transform in one step"""
        self.fit(embeddings)
        return self.reducer.transform(embeddings)


# ============================================================================
# Binning
# ============================================================================


def bin_scalar(value: float, num_bins: int, value_range: tuple[float, float]) -> int:
    """
    Map continuous scalar to discrete bin

    Args:
        value: Scalar value
        num_bins: Number of bins
        value_range: (min, max) expected range

    Returns:
        Bin index in [0, num_bins-1]
    """
    min_val, max_val = value_range

    # Validate bounds
    if num_bins <= 0:
        raise ValueError(f"num_bins must be positive, got {num_bins}")

    if min_val >= max_val:
        raise ValueError(f"min_val must be less than max_val, got min={min_val}, max={max_val}")

    # Clamp to range
    value = max(min_val, min(max_val, value))

    # Normalize to [0, 1]
    normalized = (value - min_val) / (max_val - min_val) if max_val > min_val else 0.0

    # Map to bin
    bin_idx = int(normalized * num_bins)
    return min(bin_idx, num_bins - 1)  # Handle edge case


def calculate_cell_id(
    complexity: float,
    reduced_embedding: npt.NDArray[np.float64],
    complexity_bins: int = 10,
    embedding_bins: int = 10,
) -> str:
    """
    Calculate cell ID for QD Grid

    Args:
        complexity: Complexity feature [0, 1]
        reduced_embedding: Reduced embedding (1D or 2D)
        complexity_bins: Number of bins for complexity
        embedding_bins: Number of bins per embedding dimension

    Returns:
        Cell ID string like "complexity_5-diversity_0_7"
    """
    # Bin complexity
    comp_bin = bin_scalar(complexity, complexity_bins, (0.0, 1.0))

    # Bin embedding components
    # Assume embedding components are roughly in [-3, 3] after PCA
    emb_bins: list[str] = []
    for i, val in enumerate(reduced_embedding):
        bin_idx = bin_scalar(float(val), embedding_bins, (-3.0, 3.0))
        emb_bins.append(f"diversity_{i}_{bin_idx}")

    # Construct cell ID
    return f"complexity_{comp_bin}-" + "-".join(emb_bins)


# ============================================================================
# QD Grid Manager
# ============================================================================


class QDGridManager:
    """
    Manages Quality-Diversity Grid

    Maintains elites in each cell for diverse, high-quality SOP population.
    Requires sentence-transformers for embeddings.
    """

    def __init__(
        self,
        complexity_bins: int = 10,
        embedding_dimensions: int = 2,
        embedding_bins: int = 10,
    ):
        """
        Initialize QD Grid

        Args:
            complexity_bins: Number of bins for complexity feature
            embedding_dimensions: Number of embedding dimensions after reduction
            embedding_bins: Number of bins per embedding dimension
        """
        self.complexity_bins = complexity_bins
        self.embedding_dimensions = embedding_dimensions
        self.embedding_bins = embedding_bins

        # Grid storage: cell_id -> elite SOPGene
        self.cells: dict[str, dict[str, Any]] = {}

        # Embedding reducer
        self.embedding_reducer = EmbeddingReducer(
            method="PCA", target_dimensions=embedding_dimensions
        )

        # Visit tracking for curiosity-driven selection
        self.cell_visit_counts: dict[str, int] = {}  # cell_id -> visit count
        self.total_selections: int = 0  # Total selections across all cells

        # Statistics
        self.stats: dict[str, Any] = {
            "total_sops_evaluated": 0,
            "occupied_cells": 0,
            "coverage": 0.0,
            "average_elite_quality": 0.0,
        }

        # Total possible cells
        self.total_cells = complexity_bins * (embedding_bins**embedding_dimensions)

    def bootstrap(self, initial_sops: list[tuple[ProcessConfig, PromptGenome]]) -> None:
        """
        Bootstrap grid with initial SOPs to fit embedding reducer

        Args:
            initial_sops: List of (ProcessConfig, PromptGenome) pairs
        """
        if len(initial_sops) < self.embedding_dimensions:
            logger.warning(
                f"Only {len(initial_sops)} SOPs for bootstrap. "
                f"Need at least {self.embedding_dimensions} for PCA. Skipping bootstrap."
            )
            return

        # Calculate embeddings
        embeddings: list[npt.NDArray[np.float64]] = []
        for sop, genome in initial_sops:
            emb = calculate_diversity_embedding(sop, genome)
            embeddings.append(emb)

        if embeddings:
            embeddings_array = np.array(embeddings)
            self.embedding_reducer.fit(embeddings_array)
            logger.info(f"QD Grid bootstrapped with {len(embeddings)} SOPs")

    def add_sop(
        self,
        sop_gene: SOPGene,
        sop_config: ProcessConfig,
        prompt_genome: PromptGenome,
        quality_metric: str = "weighted_aggregate",
    ) -> tuple[bool, str]:
        """
        Add SOP to grid, potentially replacing elite

        Args:
            sop_gene: SOPGene to add
            sop_config: ProcessConfig
            prompt_genome: PromptGenome
            quality_metric: Metric to use for quality comparison

        Returns:
            (was_added, cell_id) - True if SOP became elite
        """
        # Calculate QD features
        complexity = calculate_complexity(sop_config, prompt_genome)
        diversity_embedding = calculate_diversity_embedding(sop_config, prompt_genome)

        # Reduce embedding
        if not self.embedding_reducer.is_fitted:
            # If reducer not fitted, use raw embedding (limited dimensions)
            # This is a fallback for when we don't have enough SOPs to fit PCA
            reduced_embedding = diversity_embedding[: self.embedding_dimensions]
        else:
            reduced_embedding = self.embedding_reducer.transform(diversity_embedding)

        # Calculate cell ID
        cell_id = calculate_cell_id(
            complexity,
            reduced_embedding,
            self.complexity_bins,
            self.embedding_bins,
        )

        # Get quality score (use mean value)
        quality = sop_gene.get_metric_mean(quality_metric)

        # Check if cell exists and if SOP is better
        was_added = False

        if cell_id not in self.cells or quality > self.cells[cell_id]["quality"]:
            # New elite!
            self.cells[cell_id] = {
                "sopId": sop_gene.sopId,
                "sopVersion": sop_gene.version,
                "quality": quality,
                "complexity": complexity,
                "reduced_embedding": reduced_embedding.tolist(),
                "evaluatedAt": datetime.now(timezone.utc).isoformat(),
            }
            was_added = True

            # Update SOPGene with QD info
            sop_gene.qdFeatures = QDFeatures(
                complexity=complexity,
                diversityEmbedding=diversity_embedding.tolist(),
                featureVersion="mvp-1.0",
            )
            sop_gene.qdCell = QDCell(
                cellId=cell_id,
                isCellElite=True,
            )

        # Update stats
        self.stats["total_sops_evaluated"] += 1
        self.stats["occupied_cells"] = len(self.cells)
        self.stats["coverage"] = self.stats["occupied_cells"] / self.total_cells

        if self.cells:
            qualities = [cell["quality"] for cell in self.cells.values()]
            self.stats["average_elite_quality"] = sum(qualities) / len(qualities)

        return was_added, cell_id

    def get_elite(self, cell_id: str) -> Optional[dict[str, Any]]:
        """Get elite SOP for a specific cell"""
        return self.cells.get(cell_id)

    def get_all_elites(self) -> list[dict[str, Any]]:
        """Get all elite SOPs across grid"""
        return list(self.cells.values())

    def sample_for_evolution(self, strategy: str = "uniform", n: int = 10) -> list[tuple[str, str]]:
        """
        Sample SOPs from grid for evolution

        Args:
            strategy: Sampling strategy ("uniform", "quality_weighted")
            n: Number of SOPs to sample

        Returns:
            List of (sopId, sopVersion) tuples
        """
        elites = self.get_all_elites()
        if not elites:
            return []

        n = min(n, len(elites))

        if strategy == "uniform":
            # Uniform random sampling
            indices = np.random.choice(len(elites), size=n, replace=False)
            sampled: list[dict[str, Any]] = [elites[int(i)] for i in indices]
            return [(e["sopId"], e["sopVersion"]) for e in sampled]

        if strategy == "quality_weighted":
            # Sample proportional to quality using shared utility
            sampled = quality_weighted_sample(
                elites, n, quality_extractor=lambda e: e["quality"]
            )
            return [(e["sopId"], e["sopVersion"]) for e in sampled]

        raise ValueError(f"Unknown sampling strategy: {strategy}")

    def get_stats(self) -> dict[str, Any]:
        """Get grid statistics"""
        return self.stats.copy()

    def visualize_grid(self) -> dict[str, Any]:
        """
        Get grid visualization data

        Returns dict with cell occupancy and quality for visualization
        """
        MAX_VISUALIZATION_DIMENSIONS = 2
        if self.embedding_dimensions > MAX_VISUALIZATION_DIMENSIONS:
            raise NotImplementedError("Visualization only supported for 2D embeddings")

        # Build grid matrix
        grid = np.zeros((self.complexity_bins, self.embedding_bins))

        for cell_id_str, elite in self.cells.items():
            # Parse cell ID using CellID class for safe parsing
            try:
                cell_id = CellID.from_string(cell_id_str)
                comp_bin = cell_id.complexity_bin
                emb_bin = cell_id.diversity_bins[0]  # First diversity dimension
                grid[comp_bin, emb_bin] = elite["quality"]
            except (ValueError, IndexError) as e:
                logger.warning(f"Skipping malformed cell ID '{cell_id_str}': {e}")

        return {
            "grid": grid.tolist(),
            "complexity_bins": self.complexity_bins,
            "embedding_bins": self.embedding_bins,
            "occupied_cells": self.stats["occupied_cells"],
            "coverage": self.stats["coverage"],
        }

    # ========================================================================
    # Visit Tracking and Curiosity-Driven Selection (UCB)
    # ========================================================================

    def track_selection(self, cell_id: str) -> None:
        """
        Track that a cell was selected for parent selection

        Args:
            cell_id: QD cell identifier
        """
        if cell_id not in self.cell_visit_counts:
            self.cell_visit_counts[cell_id] = 0

        self.cell_visit_counts[cell_id] += 1
        self.total_selections += 1

    def get_visit_count(self, cell_id: str) -> int:
        """
        Get number of times cell has been selected

        Args:
            cell_id: Cell identifier

        Returns:
            Visit count (0 if never visited)
        """
        return self.cell_visit_counts.get(cell_id, 0)

    def get_ucb_scores(
        self, exploration_constant: float = 1.0, normalize_quality: bool = True
    ) -> dict[str, float]:
        """
        Calculate UCB (Upper Confidence Bound) scores for all occupied cells

        UCB formula: score = quality + C * sqrt(log(total_visits) / cell_visits)

        Args:
            exploration_constant: C in UCB formula (higher = more exploration)
            normalize_quality: If True, normalize qualities to [0,1]

        Returns:
            Dictionary of {cell_id: ucb_score}
        """
        import math

        ucb_scores: dict[str, float] = {}
        elites = self.get_all_elites()

        if not elites:
            return {}

        # Normalize qualities if requested
        min_q: float
        quality_range: float
        if normalize_quality:
            qualities: list[Any] = [e["quality"] for e in elites]
            min_q, max_q = min(qualities), max(qualities)
            quality_range = max_q - min_q if max_q > min_q else 1.0
        else:
            min_q, quality_range = 0.0, 1.0

        for elite in elites:
            cell_id: Optional[str] = elite.get("cellId")
            if not cell_id:
                # Backward compatibility: construct cell ID from elite data
                # This matches the format from add_to_grid()
                features: dict[str, Any] = elite.get("features", {})
                cell_id = f"complexity_{int(features.get('complexity', 0) * self.complexity_bins)}"
                if features.get("diversityEmbedding"):
                    emb_bins: list[int] = [
                        int(float(val) * self.embedding_bins)
                        for val in features["diversityEmbedding"][: self.embedding_dimensions]
                    ]
                    cell_id += f"-diversity_{'_'.join(map(str, emb_bins))}"

            quality: Any = elite["quality"]

            # Normalize quality
            if normalize_quality and quality_range > 0:
                quality = (quality - min_q) / quality_range

            visit_count = self.get_visit_count(cell_id)

            ucb: float
            if visit_count == 0:
                # Unvisited cells: maximum UCB (infinite exploration bonus)
                ucb = float("inf")
            elif self.total_selections == 0:
                # No selections yet: use quality only
                ucb = float(quality)
            else:
                # UCB formula
                exploration_bonus = exploration_constant * math.sqrt(
                    math.log(self.total_selections) / visit_count
                )
                ucb = float(quality) + exploration_bonus

            ucb_scores[cell_id] = ucb

        return ucb_scores

    def sample_by_ucb(
        self,
        count: int,
        exploration_constant: float = 1.0,
        temperature: float = 1.0,
    ) -> list[tuple[str, str]]:
        """
        Sample cells using UCB-based curiosity

        Args:
            count: Number of cells to sample
            exploration_constant: C parameter (higher = more exploration)
            temperature: Softmax temperature (higher = more uniform)

        Returns:
            List of (sopId, sopVersion) tuples from selected cells
        """
        ucb_scores = self.get_ucb_scores(exploration_constant)

        if not ucb_scores:
            return []

        # Handle infinite UCB scores (unvisited cells)
        infinite_cells: list[str] = [cid for cid, score in ucb_scores.items() if score == float("inf")]

        selected_cells: list[str] = []

        if infinite_cells:
            # Prioritize unvisited cells
            n_infinite = min(count, len(infinite_cells))
            selected_cells = list(np.random.choice(infinite_cells, size=n_infinite, replace=False))

            # If we still need more, sample from visited cells
            if n_infinite < count:
                remaining = count - n_infinite
                visited_scores: dict[str, float] = {
                    cid: score for cid, score in ucb_scores.items() if score != float("inf")
                }

                if visited_scores:
                    selected_cells.extend(
                        self._sample_by_scores(visited_scores, remaining, temperature)
                    )
        else:
            # All cells visited: sample by UCB scores
            selected_cells = self._sample_by_scores(ucb_scores, count, temperature)

        # Track selections
        for cell_id in selected_cells:
            self.track_selection(cell_id)

        # Return elite SOPs from selected cells
        result: list[tuple[str, str]] = []
        for cell_id in selected_cells:
            elite = self.cells.get(cell_id)
            if elite:
                result.append((elite["sopId"], elite["sopVersion"]))

        return result

    def _sample_by_scores(
        self, scores: dict[str, float], count: int, temperature: float
    ) -> list[str]:
        """
        Sample cell IDs proportional to scores (softmax)

        Args:
            scores: Dictionary of cell_id -> score
            count: Number to sample
            temperature: Softmax temperature

        Returns:
            List of sampled cell IDs
        """
        cell_ids = list(scores.keys())
        score_values = np.array([scores[cid] for cid in cell_ids])

        # Softmax with temperature
        exp_scores = np.exp(score_values / temperature)
        probs = exp_scores / exp_scores.sum()

        # Sample without replacement
        n = min(count, len(cell_ids))
        indices = np.random.choice(len(cell_ids), size=n, p=probs, replace=False)

        return [cell_ids[i] for i in indices]

    def reset_visit_counts(self) -> None:
        """Reset visit tracking (e.g., between evolution jobs)"""
        self.cell_visit_counts.clear()
        self.total_selections = 0

    def get_visit_stats(self) -> dict[str, Any]:
        """
        Get visit count statistics

        Returns:
            Dictionary with visit statistics
        """
        if not self.cell_visit_counts:
            return {
                "total_selections": 0,
                "cells_visited": 0,
                "cells_unvisited": len(self.cells),
                "avg_visits_per_cell": 0.0,
                "max_visits": 0,
                "min_visits": 0,
            }

        visits = list(self.cell_visit_counts.values())
        unvisited = len(self.cells) - len(self.cell_visit_counts)

        return {
            "total_selections": self.total_selections,
            "cells_visited": len(self.cell_visit_counts),
            "cells_unvisited": unvisited,
            "avg_visits_per_cell": sum(visits) / len(visits) if visits else 0.0,
            "max_visits": max(visits) if visits else 0,
            "min_visits": min(visits) if visits else 0,
        }
