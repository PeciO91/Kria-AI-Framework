from pathlib import Path

from kria_ai.common.checkpoints import extract_state_dict, load_checkpoint, load_weights, resolve_path
from kria_ai.yolov26.export import apply_export_patch


def normalize_path(path):
    return str(resolve_path(path, Path(__file__).resolve().parents[2])) if path else path


def derive_weight_path(model_path, suffix):
    path = Path(model_path)
    return str(path.with_name(f"{path.stem}{suffix}{path.suffix}"))


def prepare_model(*args, **kwargs):
    raise RuntimeError(
        "prepare_model is no longer universal; use kria_ai.classification.models.build_model "
        "or kria_ai.yolov26.models.build_model"
    )


__all__ = [
    "apply_export_patch",
    "derive_weight_path",
    "extract_state_dict",
    "load_checkpoint",
    "load_weights",
    "normalize_path",
    "prepare_model",
]
