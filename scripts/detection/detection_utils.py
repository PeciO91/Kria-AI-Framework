"""
Detection-specific post-processing helpers shared by the on-board runner.

  - scale_coords:          map xyxy boxes from the letterboxed image space
                           back to the original image.
  - non_max_suppression:   thin wrapper over cv2.dnn.NMSBoxes with optional
                           per-class suppression via the class-offset trick.

``letterbox`` lives in ``scripts/common/dataset_utils.py`` (single source of
truth, since it is used by both calibration preprocessing and detection
preprocessing). It is re-exported here for backward-compatible board-side
imports such as ``from detection_utils import letterbox``.
"""
import cv2
import numpy as np

try:
    # Host-side: dataset_utils sits next to this file on sys.path.
    from dataset_utils import letterbox  # noqa: F401
except ImportError:  # pragma: no cover - board-side fallback
    letterbox = None  # type: ignore


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