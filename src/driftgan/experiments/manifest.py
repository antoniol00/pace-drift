"""Run manifests for traceability: git state, input hash, config, versions."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from importlib import metadata
from pathlib import Path
from typing import Any

from driftgan.config import ExperimentConfig

_TRACKED_PACKAGES = ("river", "torch", "numpy", "pandas", "scipy", "matplotlib", "pyarrow")


def _git_hash(repo_root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() or None
    except OSError:
        return None


def file_sha256(path: str | Path, cache_path: str | Path | None = None) -> str:
    """SHA-256 of a file, cached by (path, size, mtime) to avoid re-hashing
    the 1.5GB CSV on every run."""
    p = Path(path)
    stat = p.stat()
    cache_key = f"{p}|{stat.st_size}|{stat.st_mtime_ns}"
    cache: dict[str, str] = {}
    if cache_path is not None and Path(cache_path).exists():
        cache = json.loads(Path(cache_path).read_text())
        if cache_key in cache:
            return cache[cache_key]
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    digest = h.hexdigest()
    if cache_path is not None:
        cache[cache_key] = digest
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cache_path).write_text(json.dumps(cache, indent=2))
    return digest


def build_manifest(config: ExperimentConfig, phase: str, seed: int) -> dict[str, Any]:
    """Assemble the manifest for a (config, phase, seed) run."""
    from driftgan.config import REPO_ROOT

    versions: dict[str, str] = {}
    for pkg in _TRACKED_PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = "not-installed"

    csv_hash: str | None
    try:
        csv_hash = file_sha256(
            config.csv_path, cache_path=Path(config.results_dir) / ".csv_hash_cache.json"
        )
    except OSError:
        csv_hash = None

    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "phase": phase,
        "seed": seed,
        "git_hash": _git_hash(REPO_ROOT),
        "csv_path": config.csv_path,
        "csv_sha256": csv_hash,
        "config": config.resolved_dict(),
        "config_hash": config.config_hash(),
        "package_versions": versions,
    }


def save_manifest(manifest: dict[str, Any], run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "manifest.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return path
