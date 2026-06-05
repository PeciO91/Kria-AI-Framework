import os
import cv2
import numpy as np

def crop_mask(masks, boxes):
    """
    "Crop" predicted masks by zeroing out everything not in the predicted bbox.
    Vectorized.
    masks: [n, h, w]
    boxes: [n, 4] (x1, y1, x2, y2) in mask coordinates
    """
    n, h, w = masks.shape
    x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, 1)  # x1 shape(n,1,1)
    r = np.arange(w, dtype=x1.dtype)[None, None, :]  # shape(1,1,w)
    c = np.arange(h, dtype=x1.dtype)[None, :, None]  # shape(1,h,1)

    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

def process_mask(protos, masks_in, bboxes, shape, upsample=False):
    """
    Assemble masks using predicted coefficients and the prototype tensor.
    protos: [nm, mh, mw]
    masks_in: [n, nm]
    bboxes: [n, 4] (x1, y1, x2, y2) in the letterboxed image coordinates (dpu_shape)
    shape: (ih, iw)  (dpu_shape)
    """
    c, mh, mw = protos.shape
    ih, iw = shape
    
    # masks = sigmoid(masks_in @ protos)
    masks = (masks_in @ protos.reshape(c, -1)) # [n, mh*mw]
    masks = 1.0 / (1.0 + np.exp(-masks)) # sigmoid
    masks = masks.reshape(-1, mh, mw)

    # Scale boxes to mask resolution
    downsampled_bboxes = bboxes.copy()
    downsampled_bboxes[:, 0] *= mw / iw
    downsampled_bboxes[:, 1] *= mh / ih
    downsampled_bboxes[:, 2] *= mw / iw
    downsampled_bboxes[:, 3] *= mh / ih

    masks = crop_mask(masks, downsampled_bboxes)  # [n, mh, mw]
    
    if upsample:
        masks = cv2.resize(masks.transpose(1, 2, 0), (iw, ih), interpolation=cv2.INTER_LINEAR)
        if len(masks.shape) == 2:
            masks = masks[:, :, None]
        masks = masks.transpose(2, 0, 1)  # [n, ih, iw]
    return masks

def scale_image_masks(masks, img1_shape, img0_shape):
    """
    Rescale masks from letterboxed image shape back to original image shape.
    masks: [n, h1, w1] where h1,w1 = img1_shape
    img0_shape: (h0, w0) of original image
    """
    if len(masks) == 0:
        return np.empty((0, img0_shape[0], img0_shape[1]), dtype=masks.dtype)

    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    
    top, left = int(round(pad[1] - 0.1)), int(round(pad[0] - 0.1))
    bottom, right = int(round(img1_shape[0] - pad[1] + 0.1)), int(round(img1_shape[1] - pad[0] + 0.1))
    
    # Crop the padding
    masks = masks[:, top:bottom, left:right]
    
    # Resize to original shape
    masks = cv2.resize(masks.transpose(1, 2, 0), (img0_shape[1], img0_shape[0]), interpolation=cv2.INTER_LINEAR)
    if len(masks.shape) == 2:
        masks = masks[:, :, None]
    masks = masks.transpose(2, 0, 1)
    
    return masks

def load_yolo_seg_labels(label_path, img_shape):
    """
    Load YOLO segmentation labels (class x1 y1 ... xn yn) and rasterize
    them to binary masks at the original image resolution.
    
    img_shape: (H, W)
    Returns:
        classes: list of int
        masks: boolean numpy array of shape (N, H, W)
    """
    if not os.path.exists(label_path):
        return [], np.empty((0, img_shape[0], img_shape[1]), dtype=bool)

    classes = []
    masks = []
    h, w = img_shape
    
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            cid = int(parts[0])
            coords = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
            
            # Unnormalize coordinates
            coords[:, 0] *= w
            coords[:, 1] *= h
            
            # Rasterize
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [coords.astype(np.int32)], 1)
            
            classes.append(cid)
            masks.append(mask.astype(bool))
            
    if len(masks) == 0:
        return [], np.empty((0, h, w), dtype=bool)
        
    return classes, np.stack(masks, axis=0)

def mask_iou_matrix(pred_masks, gt_masks):
    """
    Compute IoU between predicted and GT masks.
    pred_masks: (N, H, W) boolean array
    gt_masks: (M, H, W) boolean array
    Returns:
        iou: (N, M) float32 array
    """
    n = pred_masks.shape[0]
    m = gt_masks.shape[0]
    
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=np.float32)
        
    pred_flat = pred_masks.reshape(n, -1).astype(np.float32)
    gt_flat = gt_masks.reshape(m, -1).astype(np.float32)
    
    intersection = np.dot(pred_flat, gt_flat.T)
    
    area_pred = pred_flat.sum(axis=1)[:, None]
    area_gt = gt_flat.sum(axis=1)[None, :]
    
    union = area_pred + area_gt - intersection
    
    iou = np.zeros_like(intersection, dtype=np.float32)
    valid = union > 0
    iou[valid] = intersection[valid] / union[valid]
    
    return iou

def compute_ap(recall, precision):
    """
    Compute Average Precision given precision and recall curves.
    Standard 101-point interpolation used in COCO evaluation.
    """
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        
    i = np.where(mrec[1:] != mrec[:-1])[0]
    
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
