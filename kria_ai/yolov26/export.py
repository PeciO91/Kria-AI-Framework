"""DPU-facing raw-output export support for YOLOv26.

Ultralytics' end-to-end heads normally decode boxes and execute ``topk`` in the
model graph.  Vitis AI instead receives the split one2one convolution tensors;
decode and selection remain CPU work.  The contract below makes that otherwise
implicit tensor ordering and shape agreement explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MethodType
from typing import Any, Sequence

from kria_ai.yolov26.models import get_yolov26_head, validate_yolov26_head


@dataclass(frozen=True)
class OutputTensorContract:
    """The name, semantic kind, and NCHW shape metadata for one output."""

    name: str
    kind: str
    channels: int
    stride: int | None


@dataclass(frozen=True)
class OutputContract:
    """Ordered raw-output interface emitted by a patched YOLOv26 model."""

    task: str
    input_size: tuple[int, int] | None
    num_classes: int
    reg_max: int
    strides: tuple[int, ...]
    tensors: tuple[OutputTensorContract, ...]
    num_masks: int | None = None
    prototype_channels: int | None = None

    @property
    def tensor_count(self) -> int:
        return len(self.tensors)

    @property
    def expected_count(self) -> int:
        """Compatibility spelling for code that reports expected outputs."""

        return self.tensor_count

    @property
    def output_names(self) -> tuple[str, ...]:
        return tuple(tensor.name for tensor in self.tensors)

    @property
    def output_channels(self) -> tuple[int, ...]:
        return tuple(tensor.channels for tensor in self.tensors)

    def validate(self, outputs: Any) -> tuple[Any, ...]:
        """Validate and return outputs in contract order.

        Validation intentionally uses only the public ``shape`` interface.  It
        therefore works for regular/quantized torch tensors and for lightweight
        tensor doubles without importing torch.
        """

        if not isinstance(outputs, (tuple, list)):
            raise TypeError(
                f"YOLOv26 raw export must return a tuple/list, got {type(outputs).__name__}"
            )
        values = tuple(outputs)
        if len(values) != self.tensor_count:
            raise ValueError(
                f"YOLOv26 {self.task} raw export returned {len(values)} tensors; "
                f"contract requires {self.tensor_count}: {self.output_names}"
            )

        batch_size = None
        level_spatial_shapes: dict[int, tuple[int, int]] = {}
        for index, (value, tensor_contract) in enumerate(zip(values, self.tensors)):
            shape_value = getattr(value, "shape", None)
            if shape_value is None:
                raise TypeError(
                    f"Output {index} ({tensor_contract.name}) has no tensor shape"
                )
            try:
                shape = tuple(int(dimension) for dimension in shape_value)
            except (TypeError, ValueError) as error:
                raise TypeError(
                    f"Output {index} ({tensor_contract.name}) has invalid shape {shape_value!r}"
                ) from error
            if len(shape) != 4:
                raise ValueError(
                    f"Output {index} ({tensor_contract.name}) must be NCHW rank 4, got {shape}"
                )
            if shape[1] != tensor_contract.channels:
                raise ValueError(
                    f"Output {index} ({tensor_contract.name}) has {shape[1]} channels; "
                    f"contract requires {tensor_contract.channels}"
                )
            if batch_size is None:
                batch_size = shape[0]
            elif shape[0] != batch_size:
                raise ValueError(
                    f"Output {index} ({tensor_contract.name}) has batch {shape[0]}; "
                    f"previous outputs use batch {batch_size}"
                )

            spatial_shape = shape[2:]
            if tensor_contract.stride is not None:
                previous_shape = level_spatial_shapes.setdefault(
                    tensor_contract.stride, spatial_shape
                )
                if previous_shape != spatial_shape:
                    raise ValueError(
                        f"Outputs at stride {tensor_contract.stride} disagree on spatial shape: "
                        f"{previous_shape} versus {spatial_shape}"
                    )
                if self.input_size is not None:
                    expected_shape = tuple(
                        dimension // tensor_contract.stride for dimension in self.input_size
                    )
                    if spatial_shape != expected_shape:
                        raise ValueError(
                            f"Output {index} ({tensor_contract.name}) has spatial shape "
                            f"{spatial_shape}; expected {expected_shape} for input "
                            f"{self.input_size} and stride {tensor_contract.stride}"
                        )
            elif self.input_size is not None and tensor_contract.kind == "prototypes":
                # Both Proto and Proto26 upsample the smallest-stride feature map
                # once, so prototypes have half that effective stride.
                prototype_stride = min(self.strides) // 2
                expected_shape = tuple(
                    dimension // prototype_stride for dimension in self.input_size
                )
                if spatial_shape != expected_shape:
                    raise ValueError(
                        f"Output {index} ({tensor_contract.name}) has spatial shape "
                        f"{spatial_shape}; expected {expected_shape}"
                    )
        return values


# More explicit spelling retained as an alias for callers and type annotations.
ExportOutputContract = OutputContract


def _normalise_strides(values: Any) -> tuple[int, ...]:
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "tolist"):
        values = values.tolist()
    if not isinstance(values, (tuple, list)):
        values = (values,)
    return tuple(int(value) for value in values)


def _make_contract(
    *,
    task: str,
    input_size: tuple[int, int] | None,
    num_classes: int,
    reg_max: int,
    strides: tuple[int, ...],
    num_masks: int | None = None,
    prototype_channels: int | None = None,
) -> OutputContract:
    tensors = []
    for stride in strides:
        tensors.extend(
            (
                OutputTensorContract(
                    name=f"boxes_stride_{stride}",
                    kind="boxes",
                    channels=4 * reg_max,
                    stride=stride,
                ),
                OutputTensorContract(
                    name=f"classes_stride_{stride}",
                    kind="classes",
                    channels=num_classes,
                    stride=stride,
                ),
            )
        )
        if task == "segmentation":
            if num_masks is None:
                raise ValueError("Segmentation output contract requires num_masks")
            tensors.append(
                OutputTensorContract(
                    name=f"masks_stride_{stride}",
                    kind="masks",
                    channels=num_masks,
                    stride=stride,
                )
            )
    if task == "segmentation":
        tensors.append(
            OutputTensorContract(
                name="prototypes",
                kind="prototypes",
                channels=int(num_masks),
                stride=None,
            )
        )

    contract = OutputContract(
        task=task,
        input_size=input_size,
        num_classes=num_classes,
        reg_max=reg_max,
        strides=strides,
        tensors=tuple(tensors),
        num_masks=num_masks,
        prototype_channels=prototype_channels,
    )
    expected_count = len(strides) * (3 if task == "segmentation" else 2)
    if task == "segmentation":
        expected_count += 1
    if contract.tensor_count != expected_count:
        raise AssertionError("Internal YOLOv26 output-contract construction error")
    return contract


def build_output_contract(config: Any) -> OutputContract:
    """Build the ordered raw-output contract from a typed model config."""

    is_segmentation = hasattr(config, "num_masks")
    return _make_contract(
        task="segmentation" if is_segmentation else "detection",
        input_size=tuple(int(value) for value in config.input_size),
        num_classes=int(config.num_classes),
        reg_max=int(config.reg_max),
        strides=tuple(int(value) for value in config.strides),
        num_masks=int(config.num_masks) if is_segmentation else None,
        prototype_channels=(
            int(config.prototype_channels) if is_segmentation else None
        ),
    )


def _contract_from_head(head: Any) -> OutputContract:
    """Build a contract when legacy callers do not supply a typed config."""

    class_names = {cls.__name__ for cls in type(head).__mro__}
    is_segmentation = "Segment" in class_names
    strides = _normalise_strides(getattr(head, "stride", ()))
    levels = int(getattr(head, "nl", -1))
    if levels < 1 or len(strides) != levels:
        raise ValueError(
            f"YOLOv26 head nl={levels} is inconsistent with strides={strides}"
        )
    num_classes = int(getattr(head, "nc", -1))
    if num_classes < 1:
        raise ValueError(f"YOLOv26 raw export requires positive nc, got {num_classes}")
    for branch_name in ("one2one_cv2", "one2one_cv3"):
        branch = getattr(head, branch_name, None)
        if branch is None or len(branch) != levels:
            raise ValueError(
                f"YOLOv26 head requires {branch_name} with {levels} levels"
            )
    if is_segmentation:
        for attribute in ("nm", "npr", "proto", "cv4", "one2one_cv4"):
            if not hasattr(head, attribute) or getattr(head, attribute) is None:
                raise ValueError(
                    f"YOLOv26 Segment head is missing required mask attribute {attribute!r}"
                )
        if len(head.one2one_cv4) != levels:
            raise ValueError(
                f"YOLOv26 Segment one2one_cv4 has {len(head.one2one_cv4)} levels; "
                f"expected {levels}"
            )
    return _make_contract(
        task="segmentation" if is_segmentation else "detection",
        input_size=None,
        num_classes=num_classes,
        reg_max=int(head.reg_max),
        strides=strides,
        num_masks=int(head.nm) if is_segmentation else None,
        prototype_channels=int(head.npr) if is_segmentation else None,
    )


def _raw_one2one_forward(head: Any, feature_maps: Sequence[Any]) -> tuple[Any, ...]:
    """Emit split one2one tensors while preserving their independent INT8 scales."""

    if len(feature_maps) != head.nl:
        raise ValueError(
            f"YOLOv26 head received {len(feature_maps)} feature maps; expected {head.nl}"
        )
    outputs = []
    emit_masks = hasattr(head, "one2one_cv4") and hasattr(head, "proto")
    for level in range(head.nl):
        outputs.append(head.one2one_cv2[level](feature_maps[level]))
        outputs.append(head.one2one_cv3[level](feature_maps[level]))
        if emit_masks:
            outputs.append(head.one2one_cv4[level](feature_maps[level]))

    if emit_masks:
        # Proto26 consumes all levels and has a training-only semantic output;
        # legacy Proto consumes P3 only.  Neither semantic output belongs in the
        # deployment contract.
        if hasattr(head.proto, "feat_refine"):
            outputs.append(head.proto(feature_maps, return_semseg=False))
        else:
            outputs.append(head.proto(feature_maps[0]))
    return tuple(outputs)


def apply_export_patch(model: Any, config: Any | None = None) -> OutputContract:
    """Patch the terminal YOLOv26 head and return its explicit output contract.

    Detection emits box/class per level (six tensors for three levels).
    Segmentation emits box/class/mask per level and prototypes (ten tensors for
    three levels).  Call :func:`validate_export_outputs` on a representative
    forward result before inspection or quantization.
    """

    if config is None:
        head = get_yolov26_head(model)
        contract = _contract_from_head(head)
    else:
        head = validate_yolov26_head(model, config)
        contract = build_output_contract(config)

    head.export = True
    head.end2end = False
    head.forward = MethodType(_raw_one2one_forward, head)
    # Keep the contract discoverable without changing the model's forward
    # result (Vitis must see tensors only).
    model._kria_output_contract = contract
    return contract


def validate_export_outputs(outputs: Any, contract: OutputContract) -> tuple[Any, ...]:
    """Validate outputs against ``contract`` and return them as a tuple."""

    if not isinstance(contract, OutputContract):
        raise TypeError(f"Expected OutputContract, got {type(contract).__name__}")
    return contract.validate(outputs)


# Descriptive alias for new code; ``apply_export_patch`` preserves the legacy
# helper name used by the existing inspector/quantizer implementation.
patch_one2one_export = apply_export_patch


__all__ = [
    "ExportOutputContract",
    "OutputContract",
    "OutputTensorContract",
    "apply_export_patch",
    "build_output_contract",
    "patch_one2one_export",
    "validate_export_outputs",
]
