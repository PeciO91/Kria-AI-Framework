"""YOLOv26 model loading and head validation.

The loader in this module is deliberately family-specific. It imports
Ultralytics from the repository configured for the model, validates the DPU
architecture path, and loads an explicitly selected local checkpoint. It does
not route through the legacy common task loader or allow Ultralytics to resolve
a checkpoint name by downloading it.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

from kria_ai.common.checkpoints import extract_state_dict, load_checkpoint


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ULTRALYTICS_MODULE_ROOTS = ("ultralytics",)


def resolve_project_path(path: str | Path, *, project_root: str | Path = PROJECT_ROOT) -> Path:
    """Resolve ``path`` relative to the project, without requiring it to exist."""

    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = Path(project_root).expanduser() / resolved
    return resolved.resolve()


def clear_model_modules(root_names: tuple[str, ...] = _ULTRALYTICS_MODULE_ROOTS) -> None:
    """Remove cached model-package imports before selecting a local repository.

    A Python process may have imported a pip-installed Ultralytics package, or a
    different local checkout, before loading this model.  Removing the complete
    module tree ensures that checkpoint pickle classes and the architecture are
    both resolved from the configured checkout.
    """

    for module_name in tuple(sys.modules):
        if any(module_name == root or module_name.startswith(f"{root}.") for root in root_names):
            del sys.modules[module_name]
    importlib.invalidate_caches()


def add_repository_to_import_path(repository_path: str | Path) -> Path:
    """Put a local Ultralytics checkout first on ``sys.path`` and return it."""

    repository = resolve_project_path(repository_path)
    package_init = repository / "ultralytics" / "__init__.py"
    if not repository.is_dir() or not package_init.is_file():
        raise FileNotFoundError(
            f"Local Ultralytics repository not found at {repository} "
            f"(expected {package_init})"
        )

    # Remove equivalent spellings (relative paths, symlinks, and trailing
    # slashes) before inserting the canonical path at the front.
    retained_paths = []
    for entry in sys.path:
        try:
            is_repository = Path(entry or ".").expanduser().resolve() == repository
        except (OSError, RuntimeError):
            is_repository = False
        if not is_repository:
            retained_paths.append(entry)
    sys.path[:] = [str(repository), *retained_paths]
    return repository


def import_local_ultralytics(repository_path: str | Path):
    """Import and return Ultralytics from exactly ``repository_path``."""

    repository = add_repository_to_import_path(repository_path)
    clear_model_modules()
    module = importlib.import_module("ultralytics")

    module_file = getattr(module, "__file__", None)
    if module_file is None:
        raise ImportError(f"Imported ultralytics package from {repository} has no __file__")
    imported_file = Path(module_file).resolve()
    try:
        imported_file.relative_to(repository)
    except ValueError as error:
        raise ImportError(
            f"Expected ultralytics from {repository}, but imported {imported_file}"
        ) from error
    return module


def get_yolov26_head(model: Any) -> Any:
    """Return the terminal Detect/Segment head from an Ultralytics model."""

    layers = getattr(model, "model", None)
    if layers is None:
        raise TypeError(f"{type(model).__name__} has no Ultralytics 'model' layer sequence")
    try:
        head = layers[-1]
    except (IndexError, KeyError, TypeError) as error:
        raise TypeError("Ultralytics model has no terminal YOLO head") from error

    class_names = {cls.__name__ for cls in type(head).__mro__}
    if "Detect" not in class_names:
        raise TypeError(
            f"Expected an Ultralytics YOLOv26 Detect or Segment head, got {type(head).__name__}"
        )
    return head


def _integer_attribute(instance: Any, name: str) -> int:
    if not hasattr(instance, name):
        raise ValueError(f"YOLOv26 head is missing required attribute {name!r}")
    value = getattr(instance, name)
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"YOLOv26 head attribute {name!r} is not an integer: {value!r}") from error


def _stride_values(head: Any) -> tuple[float, ...]:
    if not hasattr(head, "stride"):
        raise ValueError("YOLOv26 head is missing required attribute 'stride'")
    values = head.stride
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "tolist"):
        values = values.tolist()
    if not isinstance(values, (tuple, list)):
        values = (values,)
    try:
        return tuple(float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"YOLOv26 head has invalid stride metadata: {values!r}") from error


def _validate_branch(head: Any, name: str, expected_levels: int) -> None:
    if not hasattr(head, name):
        raise ValueError(f"YOLOv26 head is missing required one2one branch {name!r}")
    branch = getattr(head, name)
    try:
        branch_levels = len(branch)
    except TypeError as error:
        raise ValueError(f"YOLOv26 head branch {name!r} is not a level sequence") from error
    if branch_levels != expected_levels:
        raise ValueError(
            f"YOLOv26 head branch {name!r} has {branch_levels} levels; "
            f"configured metadata requires {expected_levels}"
        )


def validate_yolov26_head(model: Any, config: Any) -> Any:
    """Validate the model's terminal head against a typed YOLOv26 config.

    Detection configurations require a concrete ``Detect`` head.  Segmentation
    configurations (identified by ``num_masks``/``prototype_channels``) require
    a ``Segment`` subclass and validate all mask/prototype attributes needed by
    raw-output export.
    """

    head = get_yolov26_head(model)
    class_names = {cls.__name__ for cls in type(head).__mro__}
    is_segmentation = hasattr(config, "num_masks") or hasattr(config, "prototype_channels")
    is_segment_head = "Segment" in class_names

    if is_segmentation and not is_segment_head:
        raise TypeError(
            f"Configuration {getattr(config, 'id', '<unknown>')!r} requires a Segment head, "
            f"got {type(head).__name__}"
        )
    if not is_segmentation and is_segment_head:
        raise TypeError(
            f"Detection configuration {getattr(config, 'id', '<unknown>')!r} requires a Detect head, "
            f"got {type(head).__name__}"
        )

    configured_classes = int(config.num_classes)
    if configured_classes != 80:
        raise ValueError(
            f"YOLOv26 deployment requires the configured COCO nc=80, got {configured_classes}"
        )
    head_classes = _integer_attribute(head, "nc")
    if head_classes != configured_classes:
        raise ValueError(f"YOLOv26 head nc={head_classes}; configured nc={configured_classes}")

    configured_strides = tuple(float(value) for value in config.strides)
    configured_levels = len(configured_strides)
    head_levels = _integer_attribute(head, "nl")
    if head_levels != configured_levels:
        raise ValueError(
            f"YOLOv26 head nl={head_levels}; configured strides define {configured_levels} levels"
        )
    head_strides = _stride_values(head)
    if head_strides != configured_strides:
        raise ValueError(
            f"YOLOv26 head strides={head_strides}; configured strides={configured_strides}"
        )

    head_reg_max = _integer_attribute(head, "reg_max")
    configured_reg_max = int(config.reg_max)
    if head_reg_max != configured_reg_max:
        raise ValueError(
            f"YOLOv26 head reg_max={head_reg_max}; configured reg_max={configured_reg_max}"
        )

    _validate_branch(head, "one2one_cv2", configured_levels)
    _validate_branch(head, "one2one_cv3", configured_levels)

    if is_segmentation:
        head_masks = _integer_attribute(head, "nm")
        configured_masks = int(config.num_masks)
        if head_masks != configured_masks:
            raise ValueError(f"YOLOv26 Segment nm={head_masks}; configured num_masks={configured_masks}")

        # Ultralytics width-scales the YAML's prototype_channels value before
        # assigning ``head.npr`` (e.g. base 256 becomes 64 for an n model).
        # Validate both values as positive metadata rather than comparing the
        # scaled runtime value with the unscaled architecture configuration.
        head_prototypes = _integer_attribute(head, "npr")
        configured_prototypes = int(config.prototype_channels)
        if head_prototypes < 1 or configured_prototypes < 1:
            raise ValueError(
                f"YOLOv26 Segment has invalid prototype metadata: npr={head_prototypes}, "
                f"configured prototype_channels={configured_prototypes}"
            )
        for attribute in ("proto", "cv4"):
            if not hasattr(head, attribute) or getattr(head, attribute) is None:
                raise ValueError(f"YOLOv26 Segment head is missing required mask attribute {attribute!r}")
        _validate_branch(head, "one2one_cv4", configured_levels)

    return head


def build_model(
    config: Any,
    device: Any = "cpu",
    checkpoint_path: str | Path | None = None,
) -> Any:
    """Load and validate a local checkpoint trained from the configured DPU architecture.

    ``checkpoint_path`` is an explicit override; otherwise the typed config's
    explicit ``checkpoint_path`` is used. Every configured path is resolved
    relative to the project root and checked before Ultralytics is imported.
    Loading the checkpoint architecture first makes class/head mismatches fail
    instead of silently leaving a newly constructed output head uninitialized.
    """

    repository = resolve_project_path(config.repository_path)
    architecture = resolve_project_path(config.architecture_path)
    selected_checkpoint = config.checkpoint_path if checkpoint_path is None else checkpoint_path
    checkpoint = resolve_project_path(selected_checkpoint)

    if not architecture.is_file():
        raise FileNotFoundError(f"YOLOv26 architecture not found: {architecture}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"YOLOv26 checkpoint not found: {checkpoint}")

    ultralytics = import_local_ultralytics(repository)
    task = "segment" if hasattr(config, "num_masks") else "detect"
    checkpoint_payload = load_checkpoint(checkpoint, map_location="cpu")
    optimization = checkpoint_payload.get("optimization") if isinstance(checkpoint_payload, dict) else None
    if optimization:
        representation = optimization.get("checkpoint_representation", optimization.get("representation"))
        if representation != "slim":
            raise ValueError(
                f"Optimized checkpoint {checkpoint} has representation {representation!r}; a slim checkpoint is required"
            )
        try:
            from pytorch_nndct.utils import slim
        except ImportError as error:
            raise ImportError("Loading an optimized YOLOv26 checkpoint requires the Vitis AI environment") from error
        wrapper = ultralytics.YOLO(str(architecture), task=task)
        model = slim.load_state_dict(wrapper.model, extract_state_dict(checkpoint_payload))
    else:
        wrapper = ultralytics.YOLO(str(checkpoint), task=task)
        model = wrapper.model
    validate_yolov26_head(model, config)
    model.to(device)
    model.eval()
    return model


# A descriptive alias for callers that prefer load terminology.
load_model = build_model


__all__ = [
    "PROJECT_ROOT",
    "add_repository_to_import_path",
    "build_model",
    "clear_model_modules",
    "get_yolov26_head",
    "import_local_ultralytics",
    "load_model",
    "resolve_project_path",
    "validate_yolov26_head",
]
