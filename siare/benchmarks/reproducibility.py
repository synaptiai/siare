"""Reproducibility tracking for benchmark runs.

Provides environment capture, dataset hashing, and manifest generation
to ensure benchmark results can be independently reproduced.
"""

import hashlib
import json
import logging
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


logger = logging.getLogger(__name__)


def _empty_str_dict() -> dict[str, str]:
    """Factory for empty string dict."""
    return {}


@dataclass
class EnvironmentSnapshot:
    """Complete environment snapshot for reproducibility.

    Captures all relevant system and dependency information needed
    to reproduce benchmark results.

    Attributes:
        timestamp: ISO format timestamp of capture
        git_commit: Current git commit hash
        git_branch: Current git branch name
        git_dirty: Whether working directory has uncommitted changes
        python_version: Python interpreter version
        dependencies: Installed package versions {package: version}
        hardware: Hardware information {cpu, cores, memory, platform}
    """

    timestamp: str
    git_commit: str
    git_branch: str
    git_dirty: bool
    python_version: str
    dependencies: dict[str, str] = field(default_factory=_empty_str_dict)
    hardware: dict[str, str] = field(default_factory=_empty_str_dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class ReproducibilityTracker:
    """Tracks reproducibility information for benchmark runs.

    Captures environment state, computes dataset hashes, and generates
    manifests that allow independent verification of results.

    Example:
        >>> tracker = ReproducibilityTracker()
        >>> env = tracker.capture_environment()
        >>> dataset_hash = tracker.compute_dataset_hash("data/frames.json")
        >>> tracker.save_manifest(results, config, "results/manifest.json")
    """

    @staticmethod
    def capture_environment() -> EnvironmentSnapshot:
        """Capture current environment state.

        Returns:
            EnvironmentSnapshot with git, Python, and hardware info
        """
        return EnvironmentSnapshot(
            timestamp=datetime.now(timezone.utc).isoformat(),
            git_commit=ReproducibilityTracker._get_git_commit(),
            git_branch=ReproducibilityTracker._get_git_branch(),
            git_dirty=ReproducibilityTracker._is_git_dirty(),
            python_version=sys.version,
            dependencies=ReproducibilityTracker._get_dependencies(),
            hardware=ReproducibilityTracker._get_hardware_info(),
        )

    @staticmethod
    def _get_git_commit() -> str:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            logger.debug("Failed to get git commit")
        return "unknown"

    @staticmethod
    def _get_git_branch() -> str:
        """Get current git branch name."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            logger.debug("Failed to get git branch")
        return "unknown"

    @staticmethod
    def _is_git_dirty() -> bool:
        """Check if git working directory has uncommitted changes."""
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if result.returncode == 0:
                return bool(result.stdout.strip())
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            logger.debug("Failed to check git dirty status")
        return False

    @staticmethod
    def _get_dependencies() -> dict[str, str]:
        """Get installed package versions.

        Returns:
            Dictionary mapping package names to versions
        """
        dependencies: dict[str, str] = {}

        try:
            # Use pip freeze for accurate versions
            result = subprocess.run(
                [sys.executable, "-m", "pip", "freeze"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if "==" in line:
                        name, version = line.split("==", 1)
                        dependencies[name.strip()] = version.strip()
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            logger.debug("Failed to get pip dependencies")

        return dependencies

    @staticmethod
    def _get_hardware_info() -> dict[str, str]:
        """Get hardware and platform information.

        Returns:
            Dictionary with CPU, memory, and platform details
        """
        import os

        info: dict[str, str] = {
            "platform": platform.platform(),
            "processor": platform.processor() or "unknown",
            "cpu_count": str(os.cpu_count() or "unknown"),
            "machine": platform.machine(),
            "system": platform.system(),
        }

        # Try to get memory info (platform-specific)
        try:
            if platform.system() == "Darwin":
                result = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
                if result.returncode == 0:
                    mem_bytes = int(result.stdout.strip())
                    info["memory_gb"] = f"{mem_bytes / (1024**3):.1f}"
            elif platform.system() == "Linux":
                with Path("/proc/meminfo").open() as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            mem_kb = int(line.split()[1])
                            info["memory_gb"] = f"{mem_kb / (1024**2):.1f}"
                            break
        except (subprocess.SubprocessError, FileNotFoundError, OSError, ValueError):
            logger.debug("Failed to get memory info")

        return info

    @staticmethod
    def compute_dataset_hash(path: str, algorithm: str = "sha256") -> str:
        """Compute cryptographic hash of dataset file.

        Uses streaming to handle large files efficiently.

        Args:
            path: Path to dataset file
            algorithm: Hash algorithm (sha256, md5, etc.)

        Returns:
            Hexadecimal hash string

        Raises:
            FileNotFoundError: If path does not exist
            ValueError: If algorithm is not supported
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        try:
            hasher = hashlib.new(algorithm)
        except ValueError as e:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}") from e

        # Stream file in chunks to handle large datasets
        chunk_size = 8192  # 8KB chunks
        with file_path.open("rb") as f:
            while chunk := f.read(chunk_size):
                hasher.update(chunk)

        return hasher.hexdigest()

    @staticmethod
    def compute_directory_hash(
        path: str,
        algorithm: str = "sha256",
        include_patterns: Optional[list[str]] = None,
    ) -> str:
        """Compute hash of all files in a directory.

        Creates a deterministic hash by sorting files and combining their hashes.

        Args:
            path: Path to directory
            algorithm: Hash algorithm (sha256, md5, etc.)
            include_patterns: Optional list of glob patterns to include

        Returns:
            Hexadecimal hash string representing entire directory

        Raises:
            FileNotFoundError: If path does not exist
            NotADirectoryError: If path is not a directory
        """
        dir_path = Path(path)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {path}")
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {path}")

        # Collect all files
        if include_patterns:
            files: list[Path] = []
            for pattern in include_patterns:
                files.extend(dir_path.glob(pattern))
        else:
            files = [f for f in dir_path.rglob("*") if f.is_file()]

        # Sort for deterministic ordering
        files.sort()

        # Combine hashes of all files
        combined_hasher = hashlib.new(algorithm)
        for file_path in files:
            file_hash = ReproducibilityTracker.compute_dataset_hash(
                str(file_path), algorithm
            )
            # Include relative path in hash for structure awareness
            rel_path = file_path.relative_to(dir_path)
            combined_hasher.update(f"{rel_path}:{file_hash}\n".encode())

        return combined_hasher.hexdigest()

    @staticmethod
    def save_manifest(
        results: dict[str, Any],
        config: dict[str, Any],
        output_path: str,
        dataset_path: Optional[str] = None,
        environment: Optional[EnvironmentSnapshot] = None,
    ) -> None:
        """Save reproducibility manifest to JSON file.

        The manifest contains all information needed to reproduce
        benchmark results:
        - Environment snapshot (git, Python, dependencies, hardware)
        - Dataset hash for verification
        - Configuration used
        - Results obtained

        Args:
            results: Benchmark results dictionary
            config: Benchmark configuration dictionary
            output_path: Path to save manifest JSON
            dataset_path: Optional path to dataset for hash computation
            environment: Optional pre-captured environment (captures if None)
        """
        # Capture environment if not provided
        if environment is None:
            environment = ReproducibilityTracker.capture_environment()

        manifest: dict[str, Any] = {
            "manifest_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "environment": environment.to_dict(),
            "config": config,
            "results": results,
        }

        # Add dataset hash if path provided
        if dataset_path:
            try:
                manifest["dataset_hash"] = ReproducibilityTracker.compute_dataset_hash(
                    dataset_path
                )
                manifest["dataset_path"] = dataset_path
            except (FileNotFoundError, OSError) as e:
                logger.warning(f"Could not hash dataset: {e}")
                manifest["dataset_hash"] = "unavailable"
                manifest["dataset_path"] = dataset_path

        # Ensure output directory exists
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Write manifest with pretty formatting
        with output_file.open("w") as f:
            json.dump(manifest, f, indent=2, default=str)

        logger.info(f"Saved reproducibility manifest to {output_path}")

    @staticmethod
    def load_manifest(path: str) -> dict[str, Any]:
        """Load reproducibility manifest from JSON file.

        Args:
            path: Path to manifest JSON file

        Returns:
            Manifest dictionary

        Raises:
            FileNotFoundError: If manifest file not found
            json.JSONDecodeError: If manifest is not valid JSON
        """
        manifest_path = Path(path)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {path}")

        with manifest_path.open() as f:
            return json.load(f)

    @staticmethod
    def verify_dataset(manifest_path: str, dataset_path: str) -> bool:
        """Verify dataset matches manifest hash.

        Args:
            manifest_path: Path to reproducibility manifest
            dataset_path: Path to dataset file to verify

        Returns:
            True if dataset hash matches manifest, False otherwise

        Raises:
            FileNotFoundError: If manifest or dataset not found
            KeyError: If manifest does not contain dataset_hash
        """
        manifest = ReproducibilityTracker.load_manifest(manifest_path)

        if "dataset_hash" not in manifest:
            raise KeyError("Manifest does not contain dataset_hash")

        expected_hash = manifest["dataset_hash"]
        if expected_hash == "unavailable":
            logger.warning("Manifest dataset hash is unavailable, cannot verify")
            return False

        actual_hash = ReproducibilityTracker.compute_dataset_hash(dataset_path)
        matches = expected_hash == actual_hash

        if not matches:
            logger.warning(
                f"Dataset hash mismatch: expected {expected_hash[:16]}..., "
                f"got {actual_hash[:16]}..."
            )

        return matches

    @staticmethod
    def compare_environments(
        env1: EnvironmentSnapshot,
        env2: EnvironmentSnapshot,
    ) -> dict[str, Any]:
        """Compare two environment snapshots for reproducibility assessment.

        Args:
            env1: First environment snapshot
            env2: Second environment snapshot

        Returns:
            Dictionary with comparison results:
            - matches: Whether environments are identical
            - git_matches: Whether git state matches
            - python_matches: Whether Python version matches
            - dependency_diff: Packages with different versions
            - hardware_diff: Hardware differences
        """
        result: dict[str, Any] = {
            "matches": True,
            "git_matches": (
                env1.git_commit == env2.git_commit
                and env1.git_branch == env2.git_branch
                and env1.git_dirty == env2.git_dirty
            ),
            "python_matches": env1.python_version == env2.python_version,
            "dependency_diff": {},
            "hardware_diff": {},
        }

        # Compare dependencies
        all_deps = set(env1.dependencies.keys()) | set(env2.dependencies.keys())
        for dep in all_deps:
            v1 = env1.dependencies.get(dep)
            v2 = env2.dependencies.get(dep)
            if v1 != v2:
                result["dependency_diff"][dep] = {"env1": v1, "env2": v2}

        # Compare hardware
        all_hw = set(env1.hardware.keys()) | set(env2.hardware.keys())
        for key in all_hw:
            v1 = env1.hardware.get(key)
            v2 = env2.hardware.get(key)
            if v1 != v2:
                result["hardware_diff"][key] = {"env1": v1, "env2": v2}

        # Overall match check
        result["matches"] = (
            result["git_matches"]
            and result["python_matches"]
            and not result["dependency_diff"]
        )

        return result
