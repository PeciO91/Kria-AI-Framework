"""
Detection-specific post-processing helpers shared by the on-board runner.

  - scale_coords:          map xyxy boxes from the letterboxed image space
                           back to the original image.
  - non_max_suppression:   thin wrapper over cv2.dnn.NMSBoxes with optional
                           per-class suppression via the class-offset trick.
  - UltralyticsDecoderCache / decode_ultralytics_output: anchor-free INT8
                           decoder shared by the detection runner and the
                           instance-segmentation runner (run_instance_seg.py).

``letterbox`` lives in ``scripts/common/dataset_utils.py`` (single source of
truth, since it is used by both calibration preprocessing and detection
preprocessing). It is re-exported here for backward-compatible board-side
imports such as ``from detection_utils import letterbox``.
"""
import time

import cv2
import numpy as np

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """
    Aspect-preserving resize with constant padding (YOLO-style letterbox).
    """
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    # Add border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return img, ratio, (dw, dh)


def scale_coords(img1_shape, coords, img0_shape):
    """
    Rescale xyxy boxes from `img1_shape` (the letterboxed model input, e.g.
    640x640) back to `img0_shape` (the original image), accounting for the
    aspect-preserving padding inserted by `letterbox`.

    NOTE: Mutates `coords` in place. Pass a copy if you need to preserve
    the original array.
    """
    # Calculate scale and padding
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding

    # Apply padding and gain to coordinates
    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain

    # Clip coordinates to bounds of original image
    coords[:, 0].clip(0, img0_shape[1], out=coords[:, 0])  # x1
    coords[:, 1].clip(0, img0_shape[0], out=coords[:, 1])  # y1
    coords[:, 2].clip(0, img0_shape[1], out=coords[:, 2])  # x2
    coords[:, 3].clip(0, img0_shape[0], out=coords[:, 3])  # y2
    
    return coords

def non_max_suppression(boxes, scores, conf_threshold, iou_threshold, class_ids=None,
                        class_offset=4096):
    """
    OpenCV NMS wrapper with optional per-class suppression.

    Parameters
    ----------
    boxes : array-like
        Sequence (or ndarray) of [x, y, w, h] boxes.
    scores : array-like
        One score per box.
    class_ids : array-like, optional
        When provided, applies the standard "class offset" trick: boxes from
        different classes are shifted apart in coordinate space so they
        cannot suppress each other. This matches YOLOv5's per-class NMS
        semantics and works on every OpenCV version that exposes NMSBoxes.
    class_offset : int
        Spatial shift per class id; must exceed the input image size.

    Returns
    -------
    indices : ndarray
        1-D int array of kept box indices (relative to the input order).
    """
    if len(boxes) == 0:
        return np.empty(0, dtype=np.int32)

    if class_ids is None:
        nms_boxes = boxes if isinstance(boxes, list) else np.asarray(boxes).tolist()
    else:
        # Vectorized class-offset shift (no Python loop).
        # IMPORTANT: copy so we never mutate the caller's boxes array.
        boxes_arr = np.array(boxes, dtype=np.float32, copy=True)
        offsets = np.asarray(class_ids, dtype=np.float32) * float(class_offset)
        boxes_arr[:, 0] += offsets
        boxes_arr[:, 1] += offsets
        nms_boxes = boxes_arr.tolist()

    indices = cv2.dnn.NMSBoxes(nms_boxes, scores, conf_threshold, iou_threshold)
    if len(indices) > 0:
        return indices.flatten()
    return np.empty(0, dtype=np.int32)


# =============================================================
# PROFILER HELPERS (shared, duck-typed on StageProfiler)
# =============================================================
def _profile_start(profiler):
    if profiler is not None and profiler.enabled:
        return time.perf_counter()
    return None


def _profile_end(profiler, stage, start):
    if start is not None:
        profiler.add(stage, time.perf_counter() - start)


def _profile_add(profiler, stage, elapsed):
    if profiler is not None and profiler.enabled and elapsed is not None:
        profiler.add(stage, elapsed)


# =============================================================
# ANCHOR-FREE (ULTRALYTICS) DECODER  -- shared by detection +
# instance segmentation runners.
# =============================================================
class UltralyticsDecoderCache:
    def __init__(self, strides_cfg):
        self.strides = strides_cfg
        self._anchor_cache = {}

    def anchors(self, level, ny, nx):
        key = (level, ny, nx)
        cached = self._anchor_cache.get(key)
        if cached is None:
            grid_y, grid_x = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
            cached = np.stack((grid_x + 0.5, grid_y + 0.5), axis=-1).astype(np.float32)
            self._anchor_cache[key] = cached.reshape(-1, 2)
        return self._anchor_cache[key]


def _softmax_last(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def _as_nhwc(pred_int8, expected_channels):
    if pred_int8.shape[-1] == expected_channels:
        return pred_int8
    if pred_int8.ndim == 4 and pred_int8.shape[1] == expected_channels:
        return np.transpose(pred_int8, (0, 2, 3, 1))
    return pred_int8


def _output_spatial_rank(dims, expected_channels=None):
    if expected_channels is not None and len(dims) == 4:
        if dims[-1] == expected_channels:
            return dims[1] * dims[2]
        if dims[1] == expected_channels:
            return dims[2] * dims[3]
    return dims[1]


def decode_ultralytics_output(int8_outputs, dequant_scales, conf_threshold,
                              cache, output_order, num_classes, reg_max,
                              profiler=None, return_keep_index=False):
    """
    Decode Ultralytics anchor-free outputs directly from INT8 buffers.

    Lazy dequantization: the per-level tensor holds (4*reg_max + nc)
    channels; we find the max class in INT8 space and threshold against
    a pre-computed INT8 boundary. Only anchors surviving the threshold
    are cast to float32, so for typical conf=0.1 only ~1% of cells
    go through the float path (80x80 + 40x40 + 20x20 = 8400 anchors).

    When ``return_keep_index`` is True the function additionally returns a
    per-survivor ``(level_idx, flat_anchor_idx)`` array so the instance-seg
    runner can gather the matching mask coefficients for each kept box.
    """
    if conf_threshold <= 0.0:
        logit_thresh = -np.inf
    elif conf_threshold >= 1.0:
        logit_thresh = np.inf
    else:
        logit_thresh = float(np.log(conf_threshold / (1.0 - conf_threshold)))

    all_boxes = []
    all_scores = []
    all_class_ids = []
    all_keep = []  # (level_idx, flat_anchor_idx) per survivor
    expected_channels = (4 * reg_max) + num_classes

    # Group tensors by spatial size (ny*nx)
    spatial_groups = {}
    for src_idx in output_order:
        tensor = int8_outputs[src_idx]
        ny, nx = tensor.shape[1:3] if tensor.ndim == 4 else (tensor.shape[0], tensor.shape[1])
        spatial_size = ny * nx
        if spatial_size not in spatial_groups:
            spatial_groups[spatial_size] = []
        spatial_groups[spatial_size].append(src_idx)

    sorted_sizes = sorted(spatial_groups.keys(), reverse=True)

    # Process each spatial group (level)
    for spatial_size, indices in spatial_groups.items():
        if len(indices) == 1:
            # Fused output (channels == 4*reg_max + nc)
            src_idx = indices[0]
            pred_int8 = _as_nhwc(int8_outputs[src_idx], expected_channels)
            bs, ny, nx, channels = pred_int8.shape
            if channels != expected_channels:
                continue
            scale = dequant_scales[src_idx]
            pred_int8_2d = pred_int8.reshape(-1, channels)

            box_int8 = pred_int8_2d[:, :4 * reg_max]
            cls_int8 = pred_int8_2d[:, 4 * reg_max:]
            box_scale = scale
            cls_scale = scale

        elif len(indices) == 2:
            # Split outputs (box_head and cls_head)
            idx1, idx2 = indices
            t1 = _as_nhwc(int8_outputs[idx1], None)
            t2 = _as_nhwc(int8_outputs[idx2], None)

            # Identify which is box (4*reg_max channels) and which is cls (num_classes channels)
            if t1.shape[-1] == 4 * reg_max and t2.shape[-1] == num_classes:
                box_idx, cls_idx = idx1, idx2
                box_t, cls_t = t1, t2
            elif t2.shape[-1] == 4 * reg_max and t1.shape[-1] == num_classes:
                box_idx, cls_idx = idx2, idx1
                box_t, cls_t = t2, t1
            else:
                continue

            bs, ny, nx, _ = box_t.shape
            box_scale = dequant_scales[box_idx]
            cls_scale = dequant_scales[cls_idx]

            box_int8 = box_t.reshape(-1, 4 * reg_max)
            cls_int8 = cls_t.reshape(-1, num_classes)
        else:
            continue

        # 1. Threshold in INT8 space using the cls tensor
        #    logit > logit_thresh  <=>  int8 > ceil(logit_thresh / cls_scale).
        stage_start = _profile_start(profiler)
        if np.isinf(logit_thresh):
            int8_thresh = -129 if logit_thresh < 0 else 127
        else:
            int8_thresh = int(np.ceil(logit_thresh / cls_scale))
        int8_thresh = max(-129, min(127, int8_thresh))

        best_int8 = cls_int8.max(axis=1)
        mask = best_int8 > int8_thresh
        _profile_end(profiler, "decode_ultra_threshold", stage_start)
        if not mask.any():
            continue

        # 2. Full dequant only for survivors.
        stage_start = _profile_start(profiler)
        cls_logits = cls_int8[mask].astype(np.float32) * cls_scale
        _profile_end(profiler, "decode_ultra_dequant", stage_start)

        # 3. argmax on logits is equivalent to argmax on probs; compute
        #    sigmoid only on the single best class per survivor (not 80).
        stage_start = _profile_start(profiler)
        cls_id = np.argmax(cls_logits, axis=1).astype(np.int32)
        best_logits = np.take_along_axis(cls_logits, cls_id[:, None], axis=1).flatten()
        scores = (1.0 / (1.0 + np.exp(-best_logits))).astype(np.float32)
        _profile_end(profiler, "decode_ultra_class_score", stage_start)

        stage_start = _profile_start(profiler)
        box_raw = box_int8[mask].astype(np.float32) * box_scale
        if reg_max > 1:
            box_dist = (_softmax_last(box_raw.reshape(-1, 4, reg_max)) *
                        np.arange(reg_max, dtype=np.float32)).sum(axis=-1)
        else:
            box_dist = box_raw.reshape(-1, 4)

        # 4. Indexed anchor lookup. With bs=1 the mask length matches the
        #    flat anchor grid (ny*nx) directly; no tile required.
        # Map this spatial group to a pyramid level without assuming a fixed
        # input resolution: the largest feature map corresponds to the smallest
        # stride (cache.strides is ordered ascending, e.g. [8, 16, 32]).
        level_idx = sorted_sizes.index(spatial_size)
        if level_idx >= len(cache.strides):
            # Fallback (should not happen): clamp to the coarsest level.
            level_idx = len(cache.strides) - 1

        base_anchors = cache.anchors(level_idx, ny, nx)  # (ny*nx, 2)
        if bs == 1:
            anchors = base_anchors[mask]
        else:
            anchors = np.tile(base_anchors, (bs, 1))[mask]
        stride = cache.strides[level_idx]

        x1 = (anchors[:, 0] - box_dist[:, 0]) * stride
        y1 = (anchors[:, 1] - box_dist[:, 1]) * stride
        x2 = (anchors[:, 0] + box_dist[:, 2]) * stride
        y2 = (anchors[:, 1] + box_dist[:, 3]) * stride

        level_boxes = np.empty((box_dist.shape[0], 4), dtype=np.float32)
        level_boxes[:, 0] = x1
        level_boxes[:, 1] = y1
        level_boxes[:, 2] = x2 - x1
        level_boxes[:, 3] = y2 - y1
        _profile_end(profiler, "decode_ultra_box_decode", stage_start)

        all_boxes.append(level_boxes)
        all_scores.append(scores)
        all_class_ids.append(cls_id)

        if return_keep_index:
            # Flat anchor indices of survivors within this level's grid, paired
            # with the level so the runner can index that level's mask-coeff
            # tensor (which shares the same flat anchor ordering).
            flat_idx = np.nonzero(mask)[0].astype(np.int32)
            keep = np.empty((flat_idx.shape[0], 2), dtype=np.int32)
            keep[:, 0] = level_idx
            keep[:, 1] = flat_idx
            all_keep.append(keep)

    stage_start = _profile_start(profiler)
    if not all_boxes:
        _profile_end(profiler, "decode_ultra_concat", stage_start)
        empty = (np.empty((0, 4), dtype=np.float32),
                 np.empty(0, dtype=np.float32),
                 np.empty(0, dtype=np.int32))
        if return_keep_index:
            return empty + (np.empty((0, 2), dtype=np.int32),)
        return empty

    decoded = (np.concatenate(all_boxes, axis=0),
               np.concatenate(all_scores, axis=0),
               np.concatenate(all_class_ids, axis=0))
    _profile_end(profiler, "decode_ultra_concat", stage_start)
    if return_keep_index:
        return decoded + (np.concatenate(all_keep, axis=0),)
    return decoded