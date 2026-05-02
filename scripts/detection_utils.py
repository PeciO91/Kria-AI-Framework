import cv2
import numpy as np

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """
    Resize image to a 32-pixel-multiple rectangle, maintaining aspect ratio.
    Pads the remaining space with a solid color.
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
    
    # Return image, the scale ratio, and the padding used
    return img, ratio, (dw, dh)

def scale_coords(img1_shape, coords, img0_shape):
    """
    Rescale coordinates (xyxy) from img1_shape (e.g. 640x640) 
    back to img0_shape (original image size).
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

    boxes  : list of [x, y, w, h]
    scores : list of float
    class_ids : list of int, optional
        When provided, applies the standard "class offset" trick: boxes from
        different classes are shifted apart in coordinate space so they
        cannot suppress each other. This matches YOLOv5's per-class NMS
        semantics and works on every OpenCV version that exposes NMSBoxes.
    class_offset : int
        Per-class spatial offset; only needs to exceed the input image size.

    Returns the indices of kept boxes (relative to the input lists).
    """
    if len(boxes) == 0:
        return []

    if class_ids is None:
        nms_boxes = boxes
    else:
        nms_boxes = [
            [b[0] + cid * class_offset, b[1] + cid * class_offset, b[2], b[3]]
            for b, cid in zip(boxes, class_ids)
        ]

    indices = cv2.dnn.NMSBoxes(nms_boxes, scores, conf_threshold, iou_threshold)
    if len(indices) > 0:
        return indices.flatten()
    return []