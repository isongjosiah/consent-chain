"""Environment fingerprinting for reproducibility (CLAUDE.md §8)."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode("ascii").strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def device_label() -> str:
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.get_device_name(0)}"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def fingerprint(seed: int) -> dict:
    try:
        import flwr  # type: ignore

        flwr_version = flwr.__version__
    except ImportError:
        flwr_version = "uninstalled"
    return {
        "commit_hash": _git_commit(),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "flower_version": flwr_version,
        "os": platform.platform(),
        "gpu": device_label(),
        "random_seed": seed,
        "wall_start": datetime.now(timezone.utc).isoformat(),
    }


def write(fp_dir: Path, fingerprint_dict: dict) -> Path:
    fp_dir.mkdir(parents=True, exist_ok=True)
    path = fp_dir / "env.json"
    path.write_text(json.dumps(fingerprint_dict, indent=2) + "\n")
    return path


def stamp_end(fp_dir: Path) -> None:
    """Add wall_end + total_seconds to an existing env.json."""
    path = fp_dir / "env.json"
    if not path.exists():
        return
    data = json.loads(path.read_text())
    end = datetime.now(timezone.utc)
    data["wall_end"] = end.isoformat()
    try:
        start = datetime.fromisoformat(data["wall_start"])
        data["total_seconds"] = (end - start).total_seconds()
    except Exception:  # noqa: BLE001
        pass
    path.write_text(json.dumps(data, indent=2) + "\n")
