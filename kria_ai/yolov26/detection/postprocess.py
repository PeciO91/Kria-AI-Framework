"""Post-processing for NMS-free YOLOv26 one-to-one detections."""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np


def _profile_start(profiler):
    if profiler is None or not bool(getattr(profiler, "enabled", True)):
        return None
    if not callable(getattr(profiler, "add", None)):
        raise TypeError("profiler must provide an add(stage, elapsed) method")
    return time.perf_counter()


def _profile_end(profiler, stage, start):
    if start is not None:
        profiler.add(stage, time.perf_counter() - start)


def _as_boxes(boxes, *, name, copy=False):
    try:
        if copy:
            array = np.array(boxes, dtype=np.float32, copy=True)
        else:
            array = np.asarray(boxes, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a numeric array with shape (N, 4)") from error
    if array.ndim != 2 or array.shape[1] != 4:
        raise ValueError(f"{name} must have shape (N, 4), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite coordinates")
    return array


def _as_scores(scores, *, expected_length=None):
    try:
        array = np.asarray(scores, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise TypeError("scores must be a one-dimensional numeric array") from error
    if array.ndim != 1:
        raise ValueError(f"scores must be one-dimensional, got shape {array.shape}")
    if expected_length is not None and array.size != expected_length:
        raise ValueError(
            f"scores has {array.size} rows but boxes has {expected_length} rows"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("scores contains NaN or infinite values")
    if np.any((array < 0.0) | (array > 1.0)):
        raise ValueError("scores must contain probabilities in [0, 1]")
    return array


def _as_class_ids(class_ids, *, expected_length=None):
    array = np.asarray(class_ids)
    if array.ndim != 1:
        raise ValueError(f"class_ids must be one-dimensional, got shape {array.shape}")
    if expected_length is not None and array.size != expected_length:
        raise ValueError(
            f"class_ids has {array.size} rows but boxes has {expected_length} rows"
        )
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"class_ids must have an integer dtype, got {array.dtype}")
    if np.any(array < 0):
        raise ValueError("class_ids must not contain negative values")
    if np.any(array > np.iinfo(np.int32).max):
        raise ValueError("class_ids contains a value too large for int32")
    return array.astype(np.int32, copy=False)


def _validate_max_detections(max_detections):
    if isinstance(max_detections, bool) or not isinstance(
        max_detections, (int, np.integer)
    ):
        raise TypeError(
            f"max_detections must be a non-negative integer, got {max_detections!r}"
        )
    value = int(max_detections)
    if value < 0:
        raise ValueError(f"max_detections must be non-negative, got {value}")
    return value


def _image_shape(shape, *, name):
    if isinstance(shape, np.ndarray):
        values = shape.tolist() if shape.ndim == 1 else shape.shape
    else:
        values = shape
    try:
        values = tuple(values)
    except TypeError as error:
        raise TypeError(f"{name} must provide at least (height, width)") from error
    if len(values) < 2:
        raise ValueError(f"{name} must provide at least (height, width), got {values}")
    try:
        height = float(values[0])
        width = float(values[1])
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} height and width must be numeric") from error
    if not np.isfinite(height) or not np.isfinite(width) or height <= 0 or width <= 0:
        raise ValueError(
            f"{name} height and width must be positive and finite, got {(height, width)}"
        )
    return height, width


def top_k_indices(scores, max_detections):
    """Return score-descending indices for an end-to-end one-to-one head.

    YOLOv26's one-to-one branch is trained to produce duplicate-free outputs,
    so post-processing uses top-k and deliberately does not run NMS.  A stable
    sort gives deterministic behavior for equal quantized scores.
    """

    score_array = _as_scores(scores)
    maximum = _validate_max_detections(max_detections)
    if maximum == 0 or score_array.size == 0:
        return np.empty(0, dtype=np.int32)
    order = np.argsort(-score_array, kind="stable")
    return order[: min(maximum, score_array.size)].astype(np.int32, copy=False)


def xywh_to_xyxy(boxes_xywh):
    """Convert top-left ``[x, y, width, height]`` boxes to ``xyxy``."""

    boxes = _as_boxes(boxes_xywh, name="boxes_xywh", copy=True)
    boxes[:, 2] += boxes[:, 0]
    boxes[:, 3] += boxes[:, 1]
    return boxes


def scale_to_original(
    boxes_xyxy,
    input_shape,
    original_shape,
    *,
    ratio_pad=None,
    clip=True,
):
    """Map letterboxed model-input boxes back to the original image.

    Shapes are ``(height, width)`` (additional image dimensions are ignored).
    ``ratio_pad`` may be supplied as ``((gain_x, gain_y), (pad_x, pad_y))`` to
    reuse preprocessing metadata.  When omitted, the standard centered
    aspect-preserving letterbox gain and padding are reconstructed.
    """

    boxes = _as_boxes(boxes_xyxy, name="boxes_xyxy", copy=True)
    input_height, input_width = _image_shape(input_shape, name="input_shape")
    original_height, original_width = _image_shape(
        original_shape, name="original_shape"
    )

    if ratio_pad is None:
        gain = min(
            input_height / original_height,
            input_width / original_width,
        )
        gain_x = gain_y = gain
        pad_x = (input_width - original_width * gain) / 2.0
        pad_y = (input_height - original_height * gain) / 2.0
    else:
        try:
            gains, padding = ratio_pad
            if np.isscalar(gains):
                gain_x = gain_y = float(gains)
            else:
                gain_x, gain_y = (float(value) for value in gains)
            pad_x, pad_y = (float(value) for value in padding)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "ratio_pad must be ((gain_x, gain_y), (pad_x, pad_y))"
            ) from error
        metadata = (gain_x, gain_y, pad_x, pad_y)
        if any(not np.isfinite(value) for value in metadata):
            raise ValueError("ratio_pad values must be finite")
        if gain_x <= 0.0 or gain_y <= 0.0:
            raise ValueError("ratio_pad gains must be positive")

    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / gain_x
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / gain_y
    if clip:
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0.0, original_width)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0.0, original_height)
    return boxes


def _validate_keep_indices(keep_indices, expected_length):
    if keep_indices is None:
        return None
    array = np.asarray(keep_indices)
    if array.ndim == 0:
        raise ValueError("keep_indices must have a leading detection dimension")
    if array.shape[0] != expected_length:
        raise ValueError(
            f"keep_indices has {array.shape[0]} rows but boxes has {expected_length} rows"
        )
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"keep_indices must have an integer dtype, got {array.dtype}")
    if np.any(array < 0):
        raise ValueError("keep_indices must not contain negative values")
    return array


def postprocess_detections(
    boxes_xywh,
    scores,
    class_ids,
    input_shape,
    original_shape,
    max_detections=300,
    *,
    keep_indices=None,
    ratio_pad=None,
    profiler=None,
    return_selection=False,
):
    """Top-k, convert, and rescale decoded YOLOv26 detections.

    The returned tuple is ``(boxes_xyxy, scores, class_ids)``.  If decoder
    survivor ``keep_indices`` are supplied, their selected rows are returned as
    a fourth item for segmentation.  Set ``return_selection`` to append the
    selected indices relative to the decoder arrays.
    """

    boxes = _as_boxes(boxes_xywh, name="boxes_xywh")
    score_array = _as_scores(scores, expected_length=boxes.shape[0])
    class_array = _as_class_ids(class_ids, expected_length=boxes.shape[0])
    keep_array = _validate_keep_indices(keep_indices, boxes.shape[0])

    stage_start = _profile_start(profiler)
    selected = top_k_indices(score_array, max_detections)
    _profile_end(profiler, "nms_or_topk", stage_start)

    stage_start = _profile_start(profiler)
    selected_boxes = xywh_to_xyxy(boxes[selected])
    selected_boxes = scale_to_original(
        selected_boxes,
        input_shape,
        original_shape,
        ratio_pad=ratio_pad,
    )
    _profile_end(profiler, "coord_scale", stage_start)

    result = (
        selected_boxes,
        score_array[selected].astype(np.float32, copy=False),
        class_array[selected].astype(np.int32, copy=False),
    )
    if keep_array is not None:
        result += (keep_array[selected],)
    if return_selection:
        result += (selected,)
    return result


def _color_for_class(color, class_id):
    if callable(color):
        value = color(class_id)
    elif (
        isinstance(color, Sequence)
        and len(color) > 0
        and isinstance(color[0], Sequence)
        and not isinstance(color[0], (str, bytes))
    ):
        value = color[class_id % len(color)]
    else:
        value = color
    try:
        result = tuple(int(channel) for channel in value)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "color must be a BGR triplet, sequence of triplets, or callable"
        ) from error
    if len(result) != 3 or any(channel < 0 or channel > 255 for channel in result):
        raise ValueError(f"drawing color must be a BGR triplet in [0, 255], got {result}")
    return result


def draw_detections(
    image,
    boxes_xyxy,
    scores,
    class_ids,
    class_names=None,
    *,
    color=(0, 255, 0),
    thickness=2,
    font_scale=0.5,
    copy=False,
    profiler=None,
):
    """Draw boxes and confidence labels on an OpenCV BGR image."""

    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be a numpy.ndarray, got {type(image).__name__}")
    if image.ndim not in (2, 3) or image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError(f"image must be a non-empty OpenCV image, got shape {image.shape}")
    if not isinstance(copy, (bool, np.bool_)):
        raise TypeError(f"copy must be boolean, got {copy!r}")
    if isinstance(thickness, bool) or not isinstance(thickness, (int, np.integer)):
        raise TypeError(f"thickness must be a positive integer, got {thickness!r}")
    thickness = int(thickness)
    if thickness <= 0:
        raise ValueError(f"thickness must be positive, got {thickness}")
    try:
        font_scale = float(font_scale)
    except (TypeError, ValueError) as error:
        raise TypeError("font_scale must be a positive number") from error
    if not np.isfinite(font_scale) or font_scale <= 0.0:
        raise ValueError(f"font_scale must be positive and finite, got {font_scale}")
    if class_names is not None and (
        not isinstance(class_names, Sequence) or isinstance(class_names, (str, bytes))
    ):
        raise TypeError("class_names must be a sequence of labels or None")

    boxes = _as_boxes(boxes_xyxy, name="boxes_xyxy")
    score_array = _as_scores(scores, expected_length=boxes.shape[0])
    class_array = _as_class_ids(class_ids, expected_length=boxes.shape[0])
    canvas = image.copy() if copy else image

    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("draw_detections requires OpenCV (cv2)") from error

    stage_start = _profile_start(profiler)
    for box, score, class_id_value in zip(boxes, score_array, class_array):
        class_id = int(class_id_value)
        box_color = _color_for_class(color, class_id)
        x1, y1, x2, y2 = (int(round(float(value))) for value in box)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, thickness)
        if class_names is not None and class_id < len(class_names):
            name = str(class_names[class_id])
        else:
            name = f"Class {class_id}"
        cv2.putText(
            canvas,
            f"{name}: {float(score):.2f}",
            (x1, max(15, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            box_color,
            thickness,
        )
    _profile_end(profiler, "draw_boxes", stage_start)
    return canvas


def scale_coords(input_shape, coords, original_shape):
    """Compatibility wrapper that scales a floating ``coords`` array in place."""

    if not isinstance(coords, np.ndarray):
        raise TypeError("coords must be a numpy.ndarray for in-place scaling")
    if not np.issubdtype(coords.dtype, np.floating):
        raise TypeError("coords must have a floating dtype for in-place scaling")
    scaled = scale_to_original(coords, input_shape, original_shape)
    coords[...] = scaled
    return coords


# Descriptive aliases for callers migrating from different runner naming.
top_k = top_k_indices
select_top_k = top_k_indices
xywh2xyxy = xywh_to_xyxy
scale_boxes_to_original = scale_to_original
draw_boxes = draw_detections
postprocess = postprocess_detections


__all__ = [
    "draw_boxes",
    "draw_detections",
    "postprocess",
    "postprocess_detections",
    "scale_boxes_to_original",
    "scale_coords",
    "scale_to_original",
    "select_top_k",
    "top_k",
    "top_k_indices",
    "xywh2xyxy",
    "xywh_to_xyxy",
]
