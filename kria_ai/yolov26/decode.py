"""CPU decoder for raw YOLOv26 one-to-one detection heads.

The exported DPU graph emits the NMS-free ``one2one`` box and class branches
before Ultralytics' in-graph decode and top-k operations.  This module pairs
those tensors by feature-map size, decodes anchor-free distances, and leaves
selection and coordinate post-processing to ``detection.postprocess``.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence

import numpy as np


_LAYOUTS = {"NHWC", "NCHW"}
_ROLES = {"box", "class", "fused"}


def _profile_start(profiler):
    if profiler is None or not bool(getattr(profiler, "enabled", True)):
        return None
    if not callable(getattr(profiler, "add", None)):
        raise TypeError("profiler must provide an add(stage, elapsed) method")
    return time.perf_counter()


def _profile_end(profiler, stage, start):
    if start is not None:
        profiler.add(stage, time.perf_counter() - start)


class UltralyticsDecoderCache:
    """Cache anchor-point grids for one decoder/runner instance.

    ``strides`` must be ordered from the finest to the coarsest pyramid level,
    for example ``(8, 16, 32)``.  Cached grids are keyed by level and shape so
    the cache remains valid if a runner accepts more than one input size.
    """

    def __init__(self, strides):
        try:
            values = tuple(float(stride) for stride in strides)
        except (TypeError, ValueError) as error:
            raise TypeError("strides must be a non-empty sequence of positive numbers") from error
        if not values or any(not np.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("strides must be a non-empty sequence of positive finite numbers")
        self.strides = values
        self._anchor_cache = {}

    def anchors(self, level, ny, nx):
        if not isinstance(level, (int, np.integer)) or not 0 <= int(level) < len(self.strides):
            raise ValueError(
                f"level must be in [0, {len(self.strides) - 1}], got {level!r}"
            )
        if not isinstance(ny, (int, np.integer)) or int(ny) <= 0:
            raise ValueError(f"ny must be a positive integer, got {ny!r}")
        if not isinstance(nx, (int, np.integer)) or int(nx) <= 0:
            raise ValueError(f"nx must be a positive integer, got {nx!r}")

        key = (int(level), int(ny), int(nx))
        cached = self._anchor_cache.get(key)
        if cached is None:
            grid_y, grid_x = np.meshgrid(
                np.arange(int(ny), dtype=np.float32),
                np.arange(int(nx), dtype=np.float32),
                indexing="ij",
            )
            cached = np.stack((grid_x + 0.5, grid_y + 0.5), axis=-1).reshape(-1, 2)
            cached.setflags(write=False)
            self._anchor_cache[key] = cached
        return cached


# Shorter family-specific name for new callers; retain the reference runner's
# class name so the decoder can be adopted without an API translation layer.
YOLOv26DecoderCache = UltralyticsDecoderCache
DecoderCache = UltralyticsDecoderCache


def _normalize_layout(layout, *, argument="layout"):
    if layout is None:
        return None
    if not isinstance(layout, str):
        raise TypeError(f"{argument} must be 'NHWC', 'NCHW', or None")
    normalized = layout.upper()
    if normalized not in _LAYOUTS:
        raise ValueError(f"{argument} must be 'NHWC' or 'NCHW', got {layout!r}")
    return normalized


def _channel_candidates(expected_channels):
    if expected_channels is None:
        return None
    if isinstance(expected_channels, (int, np.integer)):
        candidates = (int(expected_channels),)
    else:
        try:
            candidates = tuple(int(value) for value in expected_channels)
        except (TypeError, ValueError) as error:
            raise TypeError("expected_channels must be an integer or sequence of integers") from error
    if not candidates or any(value <= 0 for value in candidates):
        raise ValueError("expected_channels must contain positive integers")
    return frozenset(candidates)


def _as_nhwc(pred_int8, expected_channels, layout=None, *, tensor_name="output"):
    """Return a rank-four tensor in NHWC order without dequantizing it.

    In automatic mode the channel axis is identified from ``expected_channels``.
    If both candidate axes match (for example ``(1, 80, 80, 80)``), NHWC is
    chosen to match VART's native output convention.  Pass ``layout='NCHW'``
    for an ambiguous NCHW tensor.
    """

    if not isinstance(pred_int8, np.ndarray):
        raise TypeError(f"{tensor_name} must be a numpy.ndarray, got {type(pred_int8).__name__}")
    if pred_int8.ndim != 4:
        raise ValueError(
            f"{tensor_name} must be rank 4 in NHWC or NCHW layout; got shape {pred_int8.shape}"
        )
    if any(dimension <= 0 for dimension in pred_int8.shape):
        raise ValueError(f"{tensor_name} has an empty dimension: shape {pred_int8.shape}")

    normalized_layout = _normalize_layout(layout)
    candidates = _channel_candidates(expected_channels)
    if normalized_layout is None:
        if candidates is None:
            normalized_layout = "NHWC"
        else:
            last_matches = pred_int8.shape[-1] in candidates
            first_matches = pred_int8.shape[1] in candidates
            if last_matches:
                # Prefer NHWC when both axes match.  Shape metadata cannot
                # distinguish the memory semantics in that case.
                normalized_layout = "NHWC"
            elif first_matches:
                normalized_layout = "NCHW"
            else:
                expected = ", ".join(str(value) for value in sorted(candidates))
                raise ValueError(
                    f"{tensor_name} shape {pred_int8.shape} has neither axis 1 nor axis -1 "
                    f"equal to an expected channel count ({expected}); check num_classes, "
                    "reg_max, output selection, and tensor layout"
                )

    channel_count = pred_int8.shape[-1] if normalized_layout == "NHWC" else pred_int8.shape[1]
    if candidates is not None and channel_count not in candidates:
        expected = ", ".join(str(value) for value in sorted(candidates))
        raise ValueError(
            f"{tensor_name} declared as {normalized_layout} has {channel_count} channels; "
            f"expected one of ({expected})"
        )
    if normalized_layout == "NCHW":
        return np.transpose(pred_int8, (0, 2, 3, 1))
    return pred_int8


def output_spatial_rank(dims, expected_channels=None, layout=None):
    """Return the spatial area of NHWC/NCHW output dimensions.

    ``expected_channels`` enables automatic layout detection.  Ambiguous shapes
    default to NHWC, as does an omitted channel count; explicitly pass NCHW
    when shape metadata alone cannot identify the layout.
    """

    try:
        shape = tuple(int(value) for value in dims)
    except (TypeError, ValueError) as error:
        raise TypeError("output dimensions must be a sequence of four integers") from error
    if len(shape) != 4 or any(value <= 0 for value in shape):
        raise ValueError(f"output dimensions must be four positive integers, got {shape}")

    normalized_layout = _normalize_layout(layout)
    candidates = _channel_candidates(expected_channels)
    if normalized_layout is None:
        if candidates is None or shape[-1] in candidates:
            normalized_layout = "NHWC"
        elif shape[1] in candidates:
            normalized_layout = "NCHW"
        elif shape[1] == shape[2]:
            # The reference runner passes the fused channel count while its
            # deployed graph emits split box/class tensors.  Square feature
            # dimensions still identify an NHWC split tensor unambiguously.
            normalized_layout = "NHWC"
        elif shape[2] == shape[3]:
            normalized_layout = "NCHW"
        else:
            expected = ", ".join(str(value) for value in sorted(candidates))
            raise ValueError(
                f"output shape {shape} does not expose an expected channel count "
                f"({expected}) and its layout cannot be inferred; pass layout explicitly"
            )
    if normalized_layout == "NHWC":
        return shape[1] * shape[2]
    return shape[2] * shape[3]


_output_spatial_rank = output_spatial_rank


def _value_for_output(specification, index, output_count, *, name):
    if specification is None or isinstance(specification, str):
        return specification
    if isinstance(specification, Mapping):
        return specification.get(index)
    if not isinstance(specification, Sequence):
        raise TypeError(f"{name} must be a string, mapping, sequence, or None")
    if len(specification) != output_count:
        raise ValueError(
            f"{name} has {len(specification)} entries but there are {output_count} outputs"
        )
    return specification[index]


def _normalize_role(role, *, index):
    if role is None:
        return None
    if not isinstance(role, str):
        raise TypeError(f"output role for tensor {index} must be a string or None")
    normalized = role.lower()
    if normalized == "cls":
        normalized = "class"
    if normalized not in _ROLES:
        raise ValueError(
            f"output role for tensor {index} must be 'box', 'class'/'cls', or 'fused'; "
            f"got {role!r}"
        )
    return normalized


def _validate_output_order(output_order, output_count):
    if output_order is None:
        return tuple(range(output_count))
    try:
        raw_order = tuple(output_order)
    except TypeError as error:
        raise TypeError("output_order must be a sequence of integer output indices") from error
    if any(
        isinstance(index, (bool, np.bool_))
        or not isinstance(index, (int, np.integer))
        for index in raw_order
    ):
        raise TypeError("output_order must contain only integer output indices")
    order = tuple(int(index) for index in raw_order)
    if not order:
        raise ValueError("output_order must select at least one detection output")
    if len(set(order)) != len(order):
        raise ValueError(f"output_order contains duplicate indices: {order}")
    invalid = [index for index in order if index < 0 or index >= output_count]
    if invalid:
        raise IndexError(
            f"output_order contains out-of-range indices {invalid}; model has {output_count} outputs"
        )
    return order


def resolve_output_order(output_dims, num_classes, reg_max, output_layouts=None):
    """Return detection-output indices ordered from finest to coarsest grid.

    This helper validates only shapes.  ``decode_ultralytics_output`` performs
    the stricter per-level box/class grouping validation.
    """

    num_classes, reg_max = _validate_head_metadata(num_classes, reg_max)
    box_channels = 4 * reg_max
    expected = (box_channels, num_classes, box_channels + num_classes)
    output_count = len(output_dims)
    ranked = []
    for index, dims in enumerate(output_dims):
        layout = _value_for_output(
            output_layouts, index, output_count, name="output_layouts"
        )
        area = output_spatial_rank(dims, expected, layout)
        ranked.append((index, area))
    return [index for index, _ in sorted(ranked, key=lambda item: (-item[1], item[0]))]


def _validate_head_metadata(num_classes, reg_max):
    if isinstance(num_classes, bool) or not isinstance(num_classes, (int, np.integer)):
        raise TypeError(f"num_classes must be a positive integer, got {num_classes!r}")
    if isinstance(reg_max, bool) or not isinstance(reg_max, (int, np.integer)):
        raise TypeError(f"reg_max must be a positive integer, got {reg_max!r}")
    num_classes = int(num_classes)
    reg_max = int(reg_max)
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}")
    if reg_max <= 0:
        raise ValueError(f"reg_max must be positive, got {reg_max}")
    return num_classes, reg_max


def _validate_confidence_threshold(confidence_threshold):
    if isinstance(confidence_threshold, bool):
        raise TypeError("conf_threshold must be a real number in [0, 1]")
    try:
        value = float(confidence_threshold)
    except (TypeError, ValueError) as error:
        raise TypeError("conf_threshold must be a real number in [0, 1]") from error
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"conf_threshold must be finite and in [0, 1], got {value}")
    return value


def _validate_scale(scale, index):
    array = np.asarray(scale)
    if array.ndim != 0:
        raise ValueError(
            f"dequant_scales[{index}] must be a scalar per-tensor scale; got shape {array.shape}"
        )
    try:
        value = float(array)
    except (TypeError, ValueError) as error:
        raise TypeError(f"dequant_scales[{index}] must be a positive scalar") from error
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(
            f"dequant_scales[{index}] must be positive and finite, got {value}"
        )
    return np.float32(value)


def _softmax_last(values):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 0 or values.shape[-1] == 0:
        raise ValueError("softmax input must have a non-empty final dimension")
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=-1, keepdims=True)


def _sigmoid(values):
    values = np.asarray(values, dtype=np.float32)
    result = np.empty_like(values, dtype=np.float32)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    negative_exp = np.exp(values[~positive])
    result[~positive] = negative_exp / (1.0 + negative_exp)
    return result


def _empty_decoded(return_keep_index):
    decoded = (
        np.empty((0, 4), dtype=np.float32),
        np.empty(0, dtype=np.float32),
        np.empty(0, dtype=np.int32),
    )
    if return_keep_index:
        return decoded + (np.empty((0, 2), dtype=np.int32),)
    return decoded


def _prepare_groups(
    int8_outputs,
    dequant_scales,
    output_order,
    num_classes,
    reg_max,
    output_layouts,
    output_roles,
):
    if isinstance(int8_outputs, (str, bytes)) or not all(
        hasattr(int8_outputs, attribute) for attribute in ("__len__", "__getitem__")
    ):
        raise TypeError("int8_outputs must be a sequence of numpy INT8 tensors")
    output_count = len(int8_outputs)
    if output_count == 0:
        raise ValueError("int8_outputs is empty; expected one fused or two split tensors per level")
    if isinstance(dequant_scales, (str, bytes)) or not all(
        hasattr(dequant_scales, attribute) for attribute in ("__len__", "__getitem__")
    ):
        raise TypeError("dequant_scales must be a sequence with one scale per output")
    if len(dequant_scales) != output_count:
        raise ValueError(
            f"received {output_count} outputs but {len(dequant_scales)} dequant scales; "
            "supply one scale for every runner output"
        )

    order = _validate_output_order(output_order, output_count)
    box_channels = 4 * reg_max
    fused_channels = box_channels + num_classes
    expected_channels = (box_channels, num_classes, fused_channels)
    groups = {}

    for index in order:
        tensor = int8_outputs[index]
        if not isinstance(tensor, np.ndarray):
            raise TypeError(
                f"output {index} must be a numpy.ndarray, got {type(tensor).__name__}"
            )
        if tensor.dtype != np.int8:
            raise TypeError(
                f"output {index} must have dtype int8 for lazy DPU dequantization; "
                f"got {tensor.dtype}"
            )
        layout = _value_for_output(
            output_layouts, index, output_count, name="output_layouts"
        )
        tensor_nhwc = _as_nhwc(
            tensor,
            expected_channels,
            layout,
            tensor_name=f"output {index}",
        )
        channels = tensor_nhwc.shape[-1]
        declared_role = _normalize_role(
            _value_for_output(output_roles, index, output_count, name="output_roles"),
            index=index,
        )
        possible_roles = []
        if channels == fused_channels:
            possible_roles.append("fused")
        if channels == box_channels:
            possible_roles.append("box")
        if channels == num_classes:
            possible_roles.append("class")

        if declared_role is not None:
            if declared_role not in possible_roles:
                raise ValueError(
                    f"output {index} was declared as {declared_role!r}, but shape "
                    f"{tensor.shape} exposes {channels} channels; expected "
                    f"box={box_channels}, class={num_classes}, fused={fused_channels}"
                )
            role = declared_role
        elif len(possible_roles) == 1:
            role = possible_roles[0]
        elif len(possible_roles) > 1:
            raise ValueError(
                f"output {index} has ambiguous channel count {channels} because "
                f"4 * reg_max equals num_classes; provide output_roles mapping this "
                "tensor to 'box' or 'class'"
            )
        else:  # Defensive: _as_nhwc already rejects this case.
            raise ValueError(f"output {index} has unsupported channel count {channels}")

        batch, height, width, _ = tensor_nhwc.shape
        key = (batch, height, width)
        groups.setdefault(key, []).append(
            {
                "index": index,
                "role": role,
                "tensor": tensor_nhwc,
                "scale": _validate_scale(dequant_scales[index], index),
            }
        )

    validated = []
    for (batch, height, width), members in groups.items():
        role_counts = {role: 0 for role in _ROLES}
        for member in members:
            role_counts[member["role"]] += 1
        is_fused = len(members) == 1 and role_counts["fused"] == 1
        is_split = (
            len(members) == 2
            and role_counts["box"] == 1
            and role_counts["class"] == 1
        )
        if not (is_fused or is_split):
            description = ", ".join(
                f"output {member['index']} ({member['role']}, "
                f"{member['tensor'].shape[-1]} channels)"
                for member in members
            )
            raise ValueError(
                f"malformed YOLOv26 one2one group at batch/spatial shape "
                f"({batch}, {height}, {width}): {description}. Expected exactly one "
                f"fused [{fused_channels}-channel] tensor or one {box_channels}-channel "
                f"box tensor plus one {num_classes}-channel class tensor"
            )
        validated.append(
            {
                "batch": batch,
                "height": height,
                "width": width,
                "members": members,
                "spatial_area": height * width,
            }
        )

    # Feature-map area determines the stride association independently of the
    # runner's output permutation.  Equal areas would make that association
    # ambiguous (especially for differently shaped rectangular maps).
    areas = [group["spatial_area"] for group in validated]
    if len(set(areas)) != len(areas):
        raise ValueError(
            f"detection levels have duplicate spatial areas {areas}; cannot assign "
            "strides unambiguously. Check output_order/output_layouts"
        )
    validated.sort(key=lambda group: group["spatial_area"], reverse=True)
    return validated


def decode_ultralytics_output(
    int8_outputs,
    dequant_scales,
    conf_threshold,
    cache,
    output_order,
    num_classes,
    reg_max,
    profiler=None,
    return_keep_index=False,
    output_layouts=None,
    output_roles=None,
):
    """Decode raw INT8 YOLOv26 one-to-one box/class outputs.

    Outputs may be fused (``4 * reg_max + num_classes`` channels) or split into
    one box and one class tensor per level.  NHWC and NCHW layouts may be mixed;
    pass ``output_layouts`` as a global string, per-output sequence, or index
    mapping when automatic channel-axis detection is ambiguous.

    Classification logits are thresholded in quantized space.  Only surviving
    class scores and box rows are converted to float, preserving the lazy INT8
    behavior used by the board runner.

    Returns ``(boxes_xywh, scores, class_ids)``.  Boxes are top-left ``xywh`` in
    model-input pixels.  If ``return_keep_index`` is true, a fourth ``(N, 2)``
    array contains ``(level_index, flat_row_index)`` for gathering matching
    segmentation coefficients.  The flat index includes the batch offset.
    """

    num_classes, reg_max = _validate_head_metadata(num_classes, reg_max)
    confidence_threshold = _validate_confidence_threshold(conf_threshold)
    if not isinstance(return_keep_index, (bool, np.bool_)):
        raise TypeError(
            f"return_keep_index must be boolean, got {return_keep_index!r}"
        )
    if not hasattr(cache, "strides") or not callable(getattr(cache, "anchors", None)):
        raise TypeError("cache must provide ordered strides and anchors(level, ny, nx)")
    try:
        strides = tuple(float(stride) for stride in cache.strides)
    except (TypeError, ValueError) as error:
        raise TypeError("cache.strides must be a sequence of positive numbers") from error
    if not strides or any(not np.isfinite(stride) or stride <= 0.0 for stride in strides):
        raise ValueError("cache.strides must contain positive finite values")

    stage_start = _profile_start(profiler)
    groups = _prepare_groups(
        int8_outputs,
        dequant_scales,
        output_order,
        num_classes,
        reg_max,
        output_layouts,
        output_roles,
    )
    _profile_end(profiler, "decode_ultra_layout", stage_start)
    if len(groups) != len(strides):
        shapes = [f"{group['height']}x{group['width']}" for group in groups]
        raise ValueError(
            f"decoded {len(groups)} valid pyramid level(s) {shapes}, but cache has "
            f"{len(strides)} stride(s) {list(strides)}; select exactly one fused or "
            "one box/class pair for every stride"
        )

    if confidence_threshold <= 0.0:
        logit_threshold = -np.inf
    elif confidence_threshold >= 1.0:
        logit_threshold = np.inf
    else:
        logit_threshold = float(
            np.log(confidence_threshold / (1.0 - confidence_threshold))
        )

    all_boxes = []
    all_scores = []
    all_class_ids = []
    all_keep = []
    box_channels = 4 * reg_max
    projection = np.arange(reg_max, dtype=np.float32)

    for level, group in enumerate(groups):
        members = group["members"]
        if members[0]["role"] == "fused":
            member = members[0]
            rows = member["tensor"].reshape(-1, box_channels + num_classes)
            box_int8 = rows[:, :box_channels]
            class_int8 = rows[:, box_channels:]
            box_scale = member["scale"]
            class_scale = member["scale"]
        else:
            by_role = {member["role"]: member for member in members}
            box_member = by_role["box"]
            class_member = by_role["class"]
            box_int8 = box_member["tensor"].reshape(-1, box_channels)
            class_int8 = class_member["tensor"].reshape(-1, num_classes)
            box_scale = box_member["scale"]
            class_scale = class_member["scale"]

        if box_int8.shape[0] != class_int8.shape[0]:
            # Grouping by full batch/spatial shape should make this impossible,
            # but keep the error local and actionable if the grouping evolves.
            raise ValueError(
                f"level {level} box/class row mismatch: {box_int8.shape[0]} versus "
                f"{class_int8.shape[0]}"
            )

        stage_start = _profile_start(profiler)
        best_quantized = np.max(class_int8, axis=1)
        if np.isneginf(logit_threshold):
            survivor_mask = np.ones(best_quantized.shape, dtype=bool)
        elif np.isposinf(logit_threshold):
            survivor_mask = np.zeros(best_quantized.shape, dtype=bool)
        else:
            # q * scale > threshold, evaluated without allocating a dequantized
            # logit array.  This is exact for Vitis' symmetric per-tensor INT8.
            survivor_mask = best_quantized > (logit_threshold / float(class_scale))
        _profile_end(profiler, "decode_ultra_threshold", stage_start)
        if not np.any(survivor_mask):
            continue

        stage_start = _profile_start(profiler)
        surviving_classes = class_int8[survivor_mask]
        box_raw = box_int8[survivor_mask].astype(np.float32) * box_scale
        _profile_end(profiler, "decode_ultra_dequant", stage_start)

        stage_start = _profile_start(profiler)
        class_ids = np.argmax(surviving_classes, axis=1).astype(np.int32)
        best_surviving = np.take_along_axis(
            surviving_classes, class_ids[:, None], axis=1
        ).reshape(-1)
        best_logits = best_surviving.astype(np.float32) * class_scale
        scores = _sigmoid(best_logits)
        _profile_end(profiler, "decode_ultra_class_score", stage_start)

        stage_start = _profile_start(profiler)
        if reg_max > 1:
            distributions = _softmax_last(box_raw.reshape(-1, 4, reg_max))
            distances = np.sum(distributions * projection, axis=-1)
        else:
            distances = box_raw.reshape(-1, 4)

        base_anchors = np.asarray(
            cache.anchors(level, group["height"], group["width"]),
            dtype=np.float32,
        )
        expected_anchor_shape = (group["height"] * group["width"], 2)
        if base_anchors.shape != expected_anchor_shape:
            raise ValueError(
                f"cache.anchors({level}, {group['height']}, {group['width']}) returned "
                f"shape {base_anchors.shape}; expected {expected_anchor_shape}"
            )
        if group["batch"] > 1:
            base_anchors = np.tile(base_anchors, (group["batch"], 1))
        anchors = base_anchors[survivor_mask]
        stride = np.float32(strides[level])

        corners = np.empty((distances.shape[0], 4), dtype=np.float32)
        corners[:, 0] = (anchors[:, 0] - distances[:, 0]) * stride
        corners[:, 1] = (anchors[:, 1] - distances[:, 1]) * stride
        corners[:, 2] = (anchors[:, 0] + distances[:, 2]) * stride
        corners[:, 3] = (anchors[:, 1] + distances[:, 3]) * stride

        boxes_xywh = np.empty_like(corners)
        boxes_xywh[:, :2] = corners[:, :2]
        boxes_xywh[:, 2:] = corners[:, 2:] - corners[:, :2]
        _profile_end(profiler, "decode_ultra_box_decode", stage_start)

        all_boxes.append(boxes_xywh)
        all_scores.append(scores.astype(np.float32, copy=False))
        all_class_ids.append(class_ids)
        if return_keep_index:
            flat_indices = np.flatnonzero(survivor_mask).astype(np.int32)
            keep = np.empty((flat_indices.size, 2), dtype=np.int32)
            keep[:, 0] = level
            keep[:, 1] = flat_indices
            all_keep.append(keep)

    stage_start = _profile_start(profiler)
    if not all_boxes:
        result = _empty_decoded(return_keep_index)
    else:
        result = (
            np.concatenate(all_boxes, axis=0),
            np.concatenate(all_scores, axis=0),
            np.concatenate(all_class_ids, axis=0),
        )
        if return_keep_index:
            result += (np.concatenate(all_keep, axis=0),)
    _profile_end(profiler, "decode_ultra_concat", stage_start)
    return result


# Family-specific and concise aliases.  There is intentionally no YOLOv5
# anchor decoder in this module.
decode_yolov26_output = decode_ultralytics_output
decode_outputs = decode_ultralytics_output


__all__ = [
    "DecoderCache",
    "UltralyticsDecoderCache",
    "YOLOv26DecoderCache",
    "decode_outputs",
    "decode_ultralytics_output",
    "decode_yolov26_output",
    "output_spatial_rank",
    "resolve_output_order",
]
