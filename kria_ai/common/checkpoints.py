from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LoadResult:
    missing_keys: tuple[str, ...] = ()
    unexpected_keys: tuple[str, ...] = ()


def resolve_path(path: str | Path, project_root: str | Path | None = None):
    resolved = Path(path).expanduser()
    if not resolved.is_absolute() and project_root is not None:
        resolved = Path(project_root) / resolved
    return resolved.resolve()


def extract_state_dict(checkpoint: Any):
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model_value = checkpoint["model"]
        return model_value.state_dict() if hasattr(model_value, "state_dict") else model_value
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    if hasattr(checkpoint, "state_dict"):
        return checkpoint.state_dict()
    raise TypeError(f"Unsupported checkpoint payload: {type(checkpoint).__name__}")


def load_checkpoint(path: str | Path, map_location="cpu"):
    import torch

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location=map_location)


def load_weights(model, path: str | Path, map_location="cpu", strict=False):
    state_dict = extract_state_dict(load_checkpoint(path, map_location=map_location))
    incompatible = model.load_state_dict(state_dict, strict=strict)
    return model, incompatible
