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
    Optimized process_mask: crops prototypes first.
    """
    c, mh, mw = protos.shape
    ih, iw = shape
    n = len(bboxes)
    
    if n == 0:
        return np.empty((0, mh, mw) if not upsample else (0, ih, iw), dtype=np.float32)
        
    downsampled_bboxes = bboxes.copy()
    downsampled_bboxes[:, 0] *= mw / iw
    downsampled_bboxes[:, 1] *= mh / ih
    downsampled_bboxes[:, 2] *= mw / iw
    downsampled_bboxes[:, 3] *= mh / ih
    
    masks = np.zeros((n, mh, mw), dtype=np.float32)
    for i in range(n):
        bx1, by1, bx2, by2 = downsampled_bboxes[i]
        px1, py1 = max(0, int(np.floor(bx1))), max(0, int(np.floor(by1)))
        px2, py2 = min(mw, int(np.ceil(bx2))), min(mh, int(np.ceil(by2)))
        
        if px2 <= px1 or py2 <= py1:
            continue
            
        proto_crop = np.ascontiguousarray(protos[:, py1:py2, px1:px2]).reshape(c, -1)
        mask_logits = masks_in[i] @ proto_crop
        mask_crop = 1.0 / (1.0 + np.exp(-mask_logits))
        mask_crop = mask_crop.reshape(py2 - py1, px2 - px1)
        
        r = np.arange(px1, px2, dtype=np.float32)
        c_idx = np.arange(py1, py2, dtype=np.float32)
        mask_crop = mask_crop * ((r[None, :] >= bx1) * (r[None, :] < bx2) * 
                                 (c_idx[:, None] >= by1) * (c_idx[:, None] < by2))
        masks[i, py1:py2, px1:px2] = mask_crop
        
    if upsample:
        masks = cv2.resize(masks.transpose(1, 2, 0), (iw, ih), interpolation=cv2.INTER_LINEAR)
        if len(masks.shape) == 2:
            masks = masks[:, :, None]
        masks = masks.transpose(2, 0, 1)
        
    return masks

def scale_image_masks(masks, bboxes, img1_shape, img0_shape):
    """
    masks: (n, sh, sw) float32 sigmoid probabilities. The source resolution
        (sh, sw) may equal the padded network input (img1_shape) OR a lower
        prototype resolution (e.g. 160x160). The img1->source scale is folded
        into the affine below, so we can warp straight from prototype space and
        skip an intermediate full-input upsample. When sh/sw == img1_shape the
        transform reduces EXACTLY to the previous full-input version.
    bboxes: (n, 4) x1,y1,x2,y2 in img1_shape (padded network input) space
    """
    n = masks.shape[0]
    if n == 0:
        return np.empty((0, img0_shape[0], img0_shape[1]), dtype=masks.dtype)

    # Ratio of the actual mask source resolution to the network input space in
    # which bboxes/pad/gain are expressed. rx == ry == 1 for full-input masks.
    src_h, src_w = masks.shape[1], masks.shape[2]
    rx = src_w / img1_shape[1]
    ry = src_h / img1_shape[0]

    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    pad_w = (img1_shape[1] - img0_shape[1] * gain) / 2
    pad_h = (img1_shape[0] - img0_shape[0] * gain) / 2

    top, left = int(round(pad_h - 0.1)), int(round(pad_w - 0.1))
    bottom, right = int(round(img1_shape[0] - pad_h + 0.1)), int(round(img1_shape[1] - pad_w + 0.1))
    Hc, Wc = bottom - top, right - left

    # match EXACTLY the scale the reference single full-canvas resize would use
    scale_y = Hc / img0_shape[0]
    scale_x = Wc / img0_shape[1]

    margin = 2  # destination pixels; covers the sigmoid's soft falloff near the box edge
    scaled_masks = np.zeros((n, img0_shape[0], img0_shape[1]), dtype=masks.dtype)

    for i in range(n):
        x1, y1, x2, y2 = bboxes[i]
        dx1 = (x1 - pad_w) / gain - margin
        dy1 = (y1 - pad_h) / gain - margin
        dx2 = (x2 - pad_w) / gain + margin
        dy2 = (y2 - pad_h) / gain + margin

        odx1, ody1 = max(0, int(np.floor(dx1))), max(0, int(np.floor(dy1)))
        odx2, ody2 = min(img0_shape[1], int(np.ceil(dx2))), min(img0_shape[0], int(np.ceil(dy2)))
        dst_w, dst_h = odx2 - odx1, ody2 - ody1
        if dst_w <= 0 or dst_h <= 0:
            continue

        # dst->src affine (WARP_INVERSE_MAP). rx/ry fold the img1->source scale
        # so we sample directly from the (possibly prototype-resolution) mask.
        # Derivation: src_input = scale*local + (edge + (o+0.5)*scale - 0.5),
        # then src_proto = (src_input + 0.5)*r - 0.5, which simplifies to the
        # coefficients below. Reduces to the full-input form when rx == ry == 1.
        M = np.array([
            [scale_x * rx, 0, (left + (odx1 + 0.5) * scale_x) * rx - 0.5],
            [0, scale_y * ry, (top + (ody1 + 0.5) * scale_y) * ry - 0.5],
        ], dtype=np.float32)

        scaled_masks[i, ody1:ody2, odx1:odx2] = cv2.warpAffine(
            masks[i], M, (dst_w, dst_h),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        )

    return scaled_masks

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
