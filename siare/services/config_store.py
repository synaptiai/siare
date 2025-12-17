"""Config Store Service - Manages versioned configuration storage"""

import json
from pathlib import Path
from typing import Any, Optional

from siare.core.models import (
    DomainPackage,
    MetaConfig,
    MetricConfig,
    ProcessConfig,
    PromptGenome,
    TaskSet,
    ToolConfig,
)


class ConfigStore:
    """
    Stores and retrieves versioned configurations

    For MVP: In-memory storage with optional JSON persistence
    For Production: Replace with proper database (PostgreSQL + S3)
    """

    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize config store

        Args:
            storage_path: Optional path for JSON file persistence
        """
        self.storage_path = Path(storage_path) if storage_path else None

        # In-memory stores: {id: {version: config}}
        self._sops: dict[str, dict[str, ProcessConfig]] = {}
        self._prompt_genomes: dict[str, dict[str, PromptGenome]] = {}
        self._meta_configs: dict[str, dict[str, MetaConfig]] = {}
        self._tools: dict[str, ToolConfig] = {}  # Tools typically not versioned
        self._metrics: dict[str, MetricConfig] = {}  # Metrics typically not versioned
        self._task_sets: dict[str, dict[str, TaskSet]] = {}
        self._domain_packages: dict[str, dict[str, DomainPackage]] = {}

        # Load from disk if path provided
        if self.storage_path:
            self._load_from_disk()

    # ========================================================================
    # ProcessConfig (SOP) Methods
    # ========================================================================

    def save_sop(self, sop: ProcessConfig) -> None:
        """Save a ProcessConfig"""
        if sop.id not in self._sops:
            self._sops[sop.id] = {}
        self._sops[sop.id][sop.version] = sop
        self._persist()

    def get_sop(self, sop_id: str, version: Optional[str] = None) -> Optional[ProcessConfig]:
        """
        Get a ProcessConfig

        Args:
            sop_id: SOP identifier
            version: Version string (if None, returns latest)

        Returns:
            ProcessConfig or None if not found
        """
        if sop_id not in self._sops:
            return None

        versions = self._sops[sop_id]
        if version:
            return versions.get(version)

        # Return latest version
        if not versions:
            return None
        latest_version = max(versions.keys(), key=self._version_key)
        return versions[latest_version]

    def list_sops(self, domain: Optional[str] = None) -> list[tuple[str, str]]:
        """
        List all SOPs

        Args:
            domain: Optional domain filter

        Returns:
            List of (sop_id, version) tuples
        """
        result: list[tuple[str, str]] = []
        for sop_id, versions in self._sops.items():
            for version in versions:
                # Domain filtering could be added via metadata
                result.append((sop_id, version))
        return result

    def get_sop_versions(self, sop_id: str) -> list[str]:
        """Get all versions of an SOP"""
        if sop_id not in self._sops:
            return []
        return sorted(self._sops[sop_id].keys(), key=self._version_key)

    # ========================================================================
    # PromptGenome Methods
    # ========================================================================

    def save_prompt_genome(self, genome: PromptGenome) -> None:
        """Save a PromptGenome"""
        if genome.id not in self._prompt_genomes:
            self._prompt_genomes[genome.id] = {}
        self._prompt_genomes[genome.id][genome.version] = genome
        self._persist()

    def get_prompt_genome(
        self, genome_id: str, version: Optional[str] = None
    ) -> Optional[PromptGenome]:
        """Get a PromptGenome"""
        if genome_id not in self._prompt_genomes:
            return None

        versions = self._prompt_genomes[genome_id]
        if version:
            return versions.get(version)

        # Return latest
        if not versions:
            return None
        latest_version = max(versions.keys(), key=self._version_key)
        return versions[latest_version]

    def list_prompt_genomes(self) -> list[tuple[str, str]]:
        """List all PromptGenomes"""
        result: list[tuple[str, str]] = []
        for genome_id, versions in self._prompt_genomes.items():
            for version in versions:
                result.append((genome_id, version))
        return result

    # ========================================================================
    # MetaConfig Methods
    # ========================================================================

    def save_meta_config(self, meta: MetaConfig) -> None:
        """Save a MetaConfig"""
        if meta.id not in self._meta_configs:
            self._meta_configs[meta.id] = {}
        self._meta_configs[meta.id][meta.version] = meta
        self._persist()

    def get_meta_config(self, meta_id: str, version: Optional[str] = None) -> Optional[MetaConfig]:
        """Get a MetaConfig"""
        if meta_id not in self._meta_configs:
            return None

        versions = self._meta_configs[meta_id]
        if version:
            return versions.get(version)

        # Return latest
        if not versions:
            return None
        latest_version = max(versions.keys(), key=self._version_key)
        return versions[latest_version]

    # ========================================================================
    # ToolConfig Methods
    # ========================================================================

    def save_tool(self, tool: ToolConfig) -> None:
        """Save a ToolConfig"""
        self._tools[tool.id] = tool
        self._persist()

    def get_tool(self, tool_id: str) -> Optional[ToolConfig]:
        """Get a ToolConfig"""
        return self._tools.get(tool_id)

    def list_tools(self) -> list[str]:
        """List all tool IDs"""
        return list(self._tools.keys())

    # ========================================================================
    # MetricConfig Methods
    # ========================================================================

    def save_metric(self, metric: MetricConfig) -> None:
        """Save a MetricConfig"""
        self._metrics[metric.id] = metric
        self._persist()

    def get_metric(self, metric_id: str) -> Optional[MetricConfig]:
        """Get a MetricConfig"""
        return self._metrics.get(metric_id)

    def list_metrics(self) -> list[str]:
        """List all metric IDs"""
        return list(self._metrics.keys())

    # ========================================================================
    # TaskSet Methods
    # ========================================================================

    def save_task_set(self, task_set: TaskSet) -> None:
        """Save a TaskSet"""
        if task_set.id not in self._task_sets:
            self._task_sets[task_set.id] = {}
        self._task_sets[task_set.id][task_set.version] = task_set
        self._persist()

    def get_task_set(self, task_set_id: str, version: Optional[str] = None) -> Optional[TaskSet]:
        """Get a TaskSet"""
        if task_set_id not in self._task_sets:
            return None

        versions = self._task_sets[task_set_id]
        if version:
            return versions.get(version)

        # Return latest
        if not versions:
            return None
        latest_version = max(versions.keys(), key=self._version_key)
        return versions[latest_version]

    # ========================================================================
    # DomainPackage Methods
    # ========================================================================

    def save_domain_package(self, package: DomainPackage) -> None:
        """Save a DomainPackage"""
        if package.id not in self._domain_packages:
            self._domain_packages[package.id] = {}
        self._domain_packages[package.id][package.version] = package
        self._persist()

    def get_domain_package(
        self, package_id: str, version: Optional[str] = None
    ) -> Optional[DomainPackage]:
        """Get a DomainPackage"""
        if package_id not in self._domain_packages:
            return None

        versions = self._domain_packages[package_id]
        if version:
            return versions.get(version)

        # Return latest
        if not versions:
            return None
        latest_version = max(versions.keys(), key=self._version_key)
        return versions[latest_version]

    def list_domain_packages(self) -> list[tuple[str, str]]:
        """List all DomainPackages"""
        result: list[tuple[str, str]] = []
        for package_id, versions in self._domain_packages.items():
            for version in versions:
                result.append((package_id, version))
        return result

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def _version_key(self, version: str) -> tuple[int, ...]:
        """
        Convert version string to tuple for comparison

        Example: "1.2.3" -> (1, 2, 3)
        """
        try:
            parts = version.split("-")[0]  # Handle "1.2.3-beta"
            return tuple(int(x) for x in parts.split("."))
        except (ValueError, AttributeError):
            return (0,)

    def _persist(self) -> None:
        """Persist to disk if storage path configured"""
        if not self.storage_path:
            return

        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Save each collection to separate JSON files
        self._save_collection("sops", self._sops)
        self._save_collection("prompt_genomes", self._prompt_genomes)
        self._save_collection("meta_configs", self._meta_configs)
        self._save_single("tools", self._tools)
        self._save_single("metrics", self._metrics)
        self._save_collection("task_sets", self._task_sets)
        self._save_collection("domain_packages", self._domain_packages)

    def _save_collection(
        self,
        name: str,
        data: dict[str, dict[str, Any]],
    ) -> None:
        """Save versioned collection to JSON"""
        if self.storage_path is None:
            return
        file_path = self.storage_path / f"{name}.json"
        serialized: dict[str, dict[str, Any]] = {}

        for item_id, versions in data.items():
            serialized[item_id] = {
                version: obj.model_dump(mode="json") for version, obj in versions.items()
            }

        with open(file_path, "w") as f:
            json.dump(serialized, f, indent=2)

    def _save_single(self, name: str, data: dict[str, Any]) -> None:
        """Save non-versioned collection to JSON"""
        if self.storage_path is None:
            return
        file_path = self.storage_path / f"{name}.json"
        serialized: dict[str, Any] = {item_id: obj.model_dump(mode="json") for item_id, obj in data.items()}

        with open(file_path, "w") as f:
            json.dump(serialized, f, indent=2)

    def _load_from_disk(self) -> None:
        """Load all data from disk"""
        if not self.storage_path or not self.storage_path.exists():
            return

        # Load each collection
        self._load_collection("sops", self._sops, ProcessConfig)
        self._load_collection("prompt_genomes", self._prompt_genomes, PromptGenome)
        self._load_collection("meta_configs", self._meta_configs, MetaConfig)
        self._load_single("tools", self._tools, ToolConfig)
        self._load_single("metrics", self._metrics, MetricConfig)
        self._load_collection("task_sets", self._task_sets, TaskSet)
        self._load_collection("domain_packages", self._domain_packages, DomainPackage)

    def _load_collection(
        self,
        name: str,
        store: dict[str, dict[str, Any]],
        model_class: Any,
    ) -> None:
        """Load versioned collection from JSON"""
        if self.storage_path is None:
            return
        file_path = self.storage_path / f"{name}.json"
        if not file_path.exists():
            return

        with open(file_path) as f:
            data: dict[str, dict[str, Any]] = json.load(f)

        for item_id, versions in data.items():
            store[item_id] = {}
            for version, obj_data in versions.items():
                store[item_id][version] = model_class(**obj_data)

    def _load_single(
        self,
        name: str,
        store: dict[str, Any],
        model_class: Any,
    ) -> None:
        """Load non-versioned collection from JSON"""
        if self.storage_path is None:
            return
        file_path = self.storage_path / f"{name}.json"
        if not file_path.exists():
            return

        with open(file_path) as f:
            data: dict[str, Any] = json.load(f)

        for item_id, obj_data in data.items():
            store[item_id] = model_class(**obj_data)

    def clear_all(self) -> None:
        """Clear all data (for testing)"""
        self._sops.clear()
        self._prompt_genomes.clear()
        self._meta_configs.clear()
        self._tools.clear()
        self._metrics.clear()
        self._task_sets.clear()
        self._domain_packages.clear()
