"""
Diagnostic script for YOLOv26s DPU output.

Runs a handful of images through the compiled xmodel and prints:
  - Per-image top-5 class predictions (before conf threshold).
  - Global class histogram over all surviving detections
    (threshold applied).
  - Raw value ranges of the box / classification channels.

This helps diagnose whether the quantized model is actually
predicting all 80 classes, or whether only "person" has enough
signal to pass the confidence filter.

Usage on the board:
    python3 debug_yolov26s.py --num-images 20 --conf 0.05
"""
import argparse
import os
import sys

import numpy as np
import cv2
import vart
import xir


CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]


def setup_dpu(model_path):
    graph = xir.Graph.deserialize(model_path)
    root = graph.get_root_subgraph()
    subgraph = [s for s in root.toposort_child_subgraph()
                if s.has_attr("device") and s.get_attr("device") == "DPU"][0]
    runner = vart.Runner.create_runner(subgraph, "run")
    in_tensors = runner.get_input_tensors()
    out_tensors = runner.get_output_tensors()
    dpu_shape = tuple(in_tensors[0].dims)
    fix_pos_in = in_tensors[0].get_attr("fix_point")
    fix_pos_outs = [t.get_attr("fix_point") for t in out_tensors]
    return subgraph, runner, dpu_shape, fix_pos_in, fix_pos_outs


def letterbox(img, new_shape=(640, 640)):
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = (new_shape[1] - new_unpad[0]) // 2
    dh = (new_shape[0] - new_unpad[1]) // 2
    img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    img_padded = np.full((new_shape[0], new_shape[1], 3), 114, dtype=np.uint8)
    img_padded[dh:dh + new_unpad[1], dw:dw + new_unpad[0]] = img_resized
    return img_padded


def preprocess(img_path, dpu_shape, fix_pos_in):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_lb = letterbox(img_rgb, new_shape=(dpu_shape[1], dpu_shape[2]))
    # Normalization: YOLO uses 0..1 scaling (mean=0, std=1).
    img_float = img_lb.astype(np.float32) / 255.0
    # Quantize to int8 with the DPU input fix_pos.
    scale = 2 ** fix_pos_in
    img_int8 = np.clip(np.round(img_float * scale), -128, 127).astype(np.int8)
    return np.expand_dims(img_int8, axis=0)


def decode_level(tensor_int8, scale, num_classes=80, reg_max=1):
    """Return (cls_probs [N, num_classes], box_dist [N, 4])."""
    bs, ny, nx, ch = tensor_int8.shape
    flat = tensor_int8.reshape(-1, ch).astype(np.float32) * scale
    box = flat[:, :4 * reg_max]
    cls_logits = flat[:, 4 * reg_max:]
    cls_probs = 1.0 / (1.0 + np.exp(-cls_logits))
    return cls_probs, box


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xmodel", default="yolov26s_kria.xmodel")
    parser.add_argument("--dataset", default="datasets/coco")
    parser.add_argument("--num-images", type=int, default=20)
    parser.add_argument("--conf", type=float, default=0.05)
    args = parser.parse_args()

    subgraph, runner, dpu_shape, fix_pos_in, fix_pos_outs = setup_dpu(args.xmodel)
    out_tensors = runner.get_output_tensors()
    output_data = [np.empty(tuple(t.dims), dtype=np.int8) for t in out_tensors]
    dequant_scales = [2.0 ** -fp for fp in fix_pos_outs]

    print(f"[INFO] DPU input shape:  {dpu_shape} fix_pos={fix_pos_in}")
    print(f"[INFO] DPU output shapes:")
    for t, fp in zip(out_tensors, fix_pos_outs):
        print(f"          {t.name}: {tuple(t.dims)} fix_pos={fp}")

    images = sorted([f for f in os.listdir(args.dataset)
                     if f.lower().endswith((".jpg", ".jpeg", ".png"))])[:args.num_images]
    if not images:
        print(f"[ERROR] No images found in {args.dataset}")
        sys.exit(1)

    global_hist = np.zeros(80, dtype=np.int64)
    box_min, box_max = float("inf"), float("-inf")
    cls_logit_min, cls_logit_max = float("inf"), float("-inf")

    for k, img_name in enumerate(images):
        img_int8 = preprocess(os.path.join(args.dataset, img_name),
                              dpu_shape, fix_pos_in)
        if img_int8 is None:
            continue

        jid = runner.execute_async([img_int8], output_data)
        runner.wait(jid)

        all_probs = []
        for lvl, (tensor, scale) in enumerate(zip(output_data, dequant_scales)):
            cls_probs, box = decode_level(tensor, scale)
            all_probs.append(cls_probs)
            box_min = min(box_min, float(box.min()))
            box_max = max(box_max, float(box.max()))
            cls_logit_min = min(cls_logit_min, float((tensor.astype(np.float32) * scale)[:, :, :, 4:].min()))
            cls_logit_max = max(cls_logit_max, float((tensor.astype(np.float32) * scale)[:, :, :, 4:].max()))

        flat_probs = np.concatenate(all_probs, axis=0)

        # Per-image top-5 class predictions (using best class per anchor).
        best_cls = np.argmax(flat_probs, axis=1)
        best_score = np.max(flat_probs, axis=1)
        surviving = best_score >= args.conf
        if surviving.any():
            cls_ids = best_cls[surviving]
            for cid in cls_ids:
                global_hist[cid] += 1

        # Print per-image summary of top classes by MAX score.
        top_classes_by_max = []
        for c in range(80):
            top_classes_by_max.append((c, float(flat_probs[:, c].max())))
        top_classes_by_max.sort(key=lambda x: -x[1])
        print(f"\n[IMG {k+1}/{len(images)}] {img_name}")
        for cid, sc in top_classes_by_max[:5]:
            print(f"    {CLASS_NAMES[cid]:<20s} max_prob={sc:.3f}")

    print("\n" + "=" * 60)
    print("GLOBAL HISTOGRAM (detections passing conf threshold):")
    print("=" * 60)
    total = int(global_hist.sum())
    for cid in np.argsort(-global_hist)[:20]:
        if global_hist[cid] == 0:
            break
        print(f"    {CLASS_NAMES[cid]:<20s} {int(global_hist[cid]):6d}  ({100*global_hist[cid]/max(total,1):.2f}%)")
    print(f"\nTotal surviving anchors: {total}")
    print(f"Box value range:       [{box_min:.2f}, {box_max:.2f}]")
    print(f"Cls logit range:       [{cls_logit_min:.2f}, {cls_logit_max:.2f}]")


if __name__ == "__main__":
    main()
