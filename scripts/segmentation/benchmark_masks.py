import time
import numpy as np
import cv2

def scale_image_masks_baseline(masks, img1_shape, img0_shape):
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

def scale_image_masks_optimized(masks, bboxes, img1_shape, img0_shape):
    n = masks.shape[0]
    if n == 0:
        return np.empty((0, img0_shape[0], img0_shape[1]), dtype=masks.dtype)

    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    pad_w = (img1_shape[1] - img0_shape[1] * gain) / 2
    pad_h = (img1_shape[0] - img0_shape[0] * gain) / 2

    top, left = int(round(pad_h - 0.1)), int(round(pad_w - 0.1))
    bottom, right = int(round(img1_shape[0] - pad_h + 0.1)), int(round(img1_shape[1] - pad_w + 0.1))
    Hc, Wc = bottom - top, right - left

    scale_y = Hc / img0_shape[0]
    scale_x = Wc / img0_shape[1]

    margin = 2
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

        M = np.array([
            [scale_x, 0, left + (odx1 + 0.5) * scale_x - 0.5],
            [0, scale_y, top + (ody1 + 0.5) * scale_y - 0.5],
        ], dtype=np.float32)

        scaled_masks[i, ody1:ody2, odx1:odx2] = cv2.warpAffine(
            masks[i], M, (dst_w, dst_h),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        )

    return scaled_masks

def test_machine_precision():
    print("--- Running Machine Precision Validation ---")
    n = 2
    ih, iw = 640, 640
    orig_h, orig_w = 1080, 1920
    
    bboxes = np.array([
        [300, 300, 350, 350],
        [500, 500, 640, 640]
    ], dtype=np.float32)
    
    masks = np.zeros((n, ih, iw), dtype=np.float32)
    
    for i in range(n):
        x1, y1, x2, y2 = map(int, bboxes[i])
        y_grid, x_grid = np.ogrid[y1:y2, x1:x2]
        cy, cx = (y1+y2)/2, (x1+x2)/2
        masks[i, y1:y2, x1:x2] = np.exp(-((y_grid - cy)**2 + (x_grid - cx)**2) / 1000.0)

    baseline = scale_image_masks_baseline(masks, (ih, iw), (orig_h, orig_w))
    optimized = scale_image_masks_optimized(masks, bboxes, (ih, iw), (orig_h, orig_w))
    
    diff = np.abs(baseline - optimized)
    print(f"Max Pixel Difference (Float32): {diff.max():.8e}")
    
    bin_base = baseline > 0.5
    bin_opt = optimized > 0.5
    intersection = (bin_base & bin_opt).sum(axis=(1, 2))
    union = (bin_base | bin_opt).sum(axis=(1, 2))
    iou = intersection / np.maximum(union, 1)
    
    print(f"mIoU against baseline: {iou.mean():.6f}\n")

def benchmark(n_objects, runs=10):
    np.random.seed(42)
    ih, iw = 640, 640
    orig_h, orig_w = 1080, 1920
    masks = np.random.rand(n_objects, ih, iw).astype(np.float32)
    
    bboxes = []
    for _ in range(n_objects):
        bw = np.random.randint(20, iw//2)
        bh = np.random.randint(20, ih//2)
        bx1 = np.random.randint(0, iw - bw)
        by1 = np.random.randint(0, ih - bh)
        bboxes.append([bx1, by1, bx1+bw, by1+bh])
    bboxes = np.array(bboxes, dtype=np.float32)

    # Benchmark Baseline
    t0 = time.time()
    for _ in range(runs):
        _ = scale_image_masks_baseline(masks, (ih, iw), (orig_h, orig_w))
    t_base = (time.time() - t0) / runs * 1000
    
    # Benchmark Optimized
    t0 = time.time()
    for _ in range(runs):
        _ = scale_image_masks_optimized(masks, bboxes, (ih, iw), (orig_h, orig_w))
    t_opt = (time.time() - t0) / runs * 1000
    
    print(f"--- N = {n_objects} Detections ---")
    print(f"Baseline Time:  {t_base:.2f} ms")
    print(f"Optimized Time: {t_opt:.2f} ms")
    print(f"Speedup:        {t_base/t_opt:.2f}x")

if __name__ == "__main__":
    test_machine_precision()
    print("Running Mask Scaling Benchmarks...\n")
    for n in [5, 20, 50, 100, 200, 300]:
        benchmark(n)
