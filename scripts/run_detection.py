"""
Board-side YOLO object detection runner.

Three-stage pipeline that decouples CPU-bound preprocessing, DPU inference,
and disk I/O so each thread can run at its own rate:

  Producers  -> letterbox, LUT-normalize, push to img_queue
  Consumers  -> DPU execute, decode YOLO grids on int8 (lazy dequant),
                per-class NMS, draw, push (path, image) to write_queue
  Writer     -> single thread that flushes JPEG/PNG output

CPU optimizations layered on top of the multi-threaded skeleton:

  - Producer normalization is a per-channel LUT instead of a
    `(img * scale - shift).astype(int8)` float multiply.
  - The decoder only dequantizes the objectness logit channel before the
    confidence threshold; the full 5+num_classes row is dequantized for
    surviving cells only.
  - The confidence threshold is applied in logit space, so the sigmoid
    runs only on the handful of survivors per frame.
  - Per-level meshgrids and reshaped anchor tensors are cached in
    `DecoderCache` so the 80x80 grid is built once per consumer instead
    of once per frame.
  - Box / score / class lists are assembled with numpy concatenations
    rather than per-survivor Python appends.
"""
import os
import sys
import time
import threading
import queue
import argparse

import numpy as np
import cv2
import vart

from model_config import get_active_model
from dataset_config import get_active_dataset
from board_config import ACTIVE_THREADS, DPU_PEAK_GOPS, get_power_mw
from board_utils import (
    PowerMonitor, ProgressCounter, setup_dpu,
    build_norm_lut, apply_norm_lut, format_report,
)
from detection_utils import letterbox, scale_coords, non_max_suppression


# =============================================================
# DECODER CACHE
# =============================================================
class DecoderCache:
    """
    Per-thread cache of meshgrids and anchor tensors keyed by spatial size.

    Each consumer owns one cache; every frame at the same level reuses the
    same `(1, ny, nx, 1, 2)` grid and `(1, 1, 1, 3, 2)` anchor table.
    """
    def __init__(self, anchors_cfg, strides_cfg):
        self.anchors = [np.array(a, dtype=np.float32).reshape(1, 1, 1, 3, 2)
                        for a in anchors_cfg]
        self.strides = strides_cfg
        self._grid_cache = {}  # (level, ny, nx) -> grid (1, ny, nx, 1, 2)

    def grid(self, level, ny, nx):
        key = (level, ny, nx)
        g = self._grid_cache.get(key)
        if g is None:
            grid_y, grid_x = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
            g = np.stack((grid_x, grid_y), axis=-1).astype(np.float32)
            g = g.reshape(1, ny, nx, 1, 2)
            self._grid_cache[key] = g
        return g


# =============================================================
# YOLO DECODER (lazy dequant + vectorized assembly)
# =============================================================
def decode_yolo_output(int8_outputs, dequant_scales, conf_threshold,
                       cache, output_order):
    """
    Decode raw YOLOv5 P3/P4/P5 tensors directly from the DPU's INT8
    buffers.

    For every detection level we:
      1. Dequantize only the objectness logit channel and threshold it
         in logit space (no sigmoid on rejected cells).
      2. Dequantize the surviving rows in full and apply sigmoid.
      3. Decode xy / wh against the cached grid and anchor tensors.
      4. Append per-level numpy arrays.

    All per-level results are concatenated once at the end.
    """
    # sigmoid(x) > t  <=>  x > logit(t)
    if conf_threshold <= 0.0:
        logit_thresh = -np.inf
    elif conf_threshold >= 1.0:
        logit_thresh = np.inf
    else:
        logit_thresh = float(np.log(conf_threshold / (1.0 - conf_threshold)))

    all_boxes = []
    all_scores = []
    all_class_ids = []

    for level, src_idx in enumerate(output_order):
        pred_int8 = int8_outputs[src_idx]
        bs, ny, nx, channels = pred_int8.shape
        num_classes = (channels // 3) - 5
        scale = dequant_scales[src_idx]

        # Reshape int8 buffer in place to (1, ny, nx, 3, 5+num_classes).
        pred_int8 = pred_int8.reshape(1, ny, nx, 3, 5 + num_classes)

        # 1. Threshold using only the objectness logit channel.
        obj_logits = pred_int8[..., 4].astype(np.float32) * scale
        mask = obj_logits > logit_thresh
        if not mask.any():
            continue

        # 2. Full dequant + sigmoid only for survivors.
        v_pred = pred_int8[mask].astype(np.float32) * scale
        v_pred[..., :5] = 1.0 / (1.0 + np.exp(-v_pred[..., :5]))
        v_pred[..., 5:] = 1.0 / (1.0 + np.exp(-v_pred[..., 5:]))
        v_obj_conf = v_pred[..., 4]

        # 3. Cached grid + anchor lookups (broadcast then index by mask).
        grid = cache.grid(level, ny, nx)                     # (1, ny, nx, 1, 2)
        v_grid = np.broadcast_to(grid, (1, ny, nx, 3, 2))[mask]
        v_anchors = np.broadcast_to(cache.anchors[level],
                                    (1, ny, nx, 3, 2))[mask]
        stride = cache.strides[level]

        v_xy = (v_pred[..., 0:2] * 2.0 - 0.5 + v_grid) * stride
        v_wh = (v_pred[..., 2:4] * 2.0) ** 2 * v_anchors

        cls_probs = v_pred[..., 5:]
        cls_id = np.argmax(cls_probs, axis=-1).astype(np.int32)
        cls_conf = np.take_along_axis(cls_probs, cls_id[..., None], axis=-1).flatten()
        total_conf = (v_obj_conf * cls_conf).astype(np.float32)

        # 4. Build [xtl, ytl, w, h] without a Python loop.
        n = v_xy.shape[0]
        level_boxes = np.empty((n, 4), dtype=np.float32)
        level_boxes[:, 0] = v_xy[:, 0] - v_wh[:, 0] * 0.5
        level_boxes[:, 1] = v_xy[:, 1] - v_wh[:, 1] * 0.5
        level_boxes[:, 2] = v_wh[:, 0]
        level_boxes[:, 3] = v_wh[:, 1]

        all_boxes.append(level_boxes)
        all_scores.append(total_conf)
        all_class_ids.append(cls_id)

    if not all_boxes:
        return (np.empty((0, 4), dtype=np.float32),
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.int32))

    return (np.concatenate(all_boxes, axis=0),
            np.concatenate(all_scores, axis=0),
            np.concatenate(all_class_ids, axis=0))


# =============================================================
# PRODUCER: letterbox + LUT normalization
# =============================================================
def producer_worker(image_chunk, input_queue, dpu_shape, norm_mean, norm_std, fix_pos):
    dpu_h, dpu_w = dpu_shape[1], dpu_shape[2]
    lut = build_norm_lut(norm_mean, norm_std, fix_pos)

    for img_path in image_chunk:
        orig_img = cv2.imread(img_path)
        if orig_img is None:
            continue
        orig_shape = orig_img.shape[:2]
        img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        img_resized, _, _ = letterbox(img_rgb, new_shape=(dpu_h, dpu_w))
        img_int8 = np.expand_dims(apply_norm_lut(img_resized, lut), axis=0)
        input_queue.put((img_int8, orig_img, orig_shape, os.path.basename(img_path)))


# =============================================================
# WRITER: async image saver (decouples cv2.imwrite from consumer)
# =============================================================
def writer_worker(write_queue):
    while True:
        item = write_queue.get()
        if item is None:
            write_queue.task_done()
            break
        out_path, img = item
        cv2.imwrite(out_path, img)
        write_queue.task_done()


# =============================================================
# CONSUMER: DPU + decode + per-class NMS + draw
# =============================================================
def consumer_worker(thread_id, input_queue, write_queue, dpu_subgraph,
                    out_dir, m_cfg, d_cfg, fix_pos_outs, output_order,
                    progress, results):
    runner = vart.Runner.create_runner(dpu_subgraph, "run")
    output_tensors = runner.get_output_tensors()
    output_data = [np.empty(tuple(t.dims), dtype=np.int8) for t in output_tensors]

    # Pre-compute output dequant scales once (one float per output tensor).
    dequant_scales = [np.float32(2 ** -fp) for fp in fix_pos_outs]

    conf_thresh = m_cfg.get('conf_threshold', 0.25)
    iou_thresh = m_cfg.get('iou_threshold', 0.45)
    dpu_shape = tuple(runner.get_input_tensors()[0].dims)[1:3]  # H, W

    cache = DecoderCache(m_cfg['anchors'], m_cfg['strides'])
    class_names = d_cfg.get('classes')

    local_total = 0
    local_dpu_time = 0.0

    while True:
        item = input_queue.get()
        if item is None:
            input_queue.task_done()
            break

        img_int8, orig_img, orig_shape, file_name = item

        # 1. DPU execution
        t_start = time.perf_counter()
        jid = runner.execute_async([img_int8], output_data)
        runner.wait(jid)
        local_dpu_time += time.perf_counter() - t_start

        # 2. Decode directly off the int8 buffers (lazy dequant).
        boxes, scores, class_ids = decode_yolo_output(
            output_data, dequant_scales, conf_thresh, cache, output_order)

        # 3. Per-class NMS (operates on numpy arrays directly).
        if boxes.shape[0] > 0:
            indices = non_max_suppression(
                boxes, scores, conf_thresh, iou_thresh, class_ids=class_ids)
            if len(indices) > 0:
                final_boxes = boxes[indices]
                final_class_ids = class_ids[indices]
                final_scores = scores[indices]

                # xywh -> xyxy on the surviving rows only.
                xyxy = final_boxes.copy()
                xyxy[:, 2] = final_boxes[:, 0] + final_boxes[:, 2]
                xyxy[:, 3] = final_boxes[:, 1] + final_boxes[:, 3]
                xyxy = scale_coords(dpu_shape, xyxy, orig_shape)

                for j in range(xyxy.shape[0]):
                    x1, y1, x2, y2 = map(int, xyxy[j, :4])
                    cid = int(final_class_ids[j])
                    conf = float(final_scores[j])
                    cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    name = class_names[cid] if class_names and cid < len(class_names) else f"Class {cid}"
                    cv2.putText(orig_img, f"{name}: {conf:.2f}",
                                (x1, max(15, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 4. Hand off to writer thread (consumer never blocks on disk I/O).
        write_queue.put((os.path.join(out_dir, file_name), orig_img))

        local_total += 1
        progress.increment()
        input_queue.task_done()

    results[thread_id] = (local_total, local_dpu_time)
    del runner


# =============================================================
# MAIN
# =============================================================
def run_detection(model_id, dataset_id, thread_override):
    m_cfg = get_active_model(model_id)
    d_cfg = get_active_dataset(dataset_id)

    if m_cfg['type'] != 'detection':
        print(f"[ERROR] Model {model_id} is a {m_cfg['type']} model. "
              f"Use run_inference.py instead.")
        sys.exit(1)
    if 'anchors' not in m_cfg or 'strides' not in m_cfg:
        print(f"[ERROR] Model {model_id} is missing 'anchors' / 'strides' in model_config.py.")
        sys.exit(1)

    # Defaults: 3 consumers (KV260 supports up to 4) + 4 producers to keep
    # them fed now that LUT normalization removes the producer bottleneck.
    num_consumers = thread_override if thread_override else max(ACTIVE_THREADS, 3)
    num_producers = 4

    model_path = f"{model_id}_kria.xmodel"
    dataset_path = os.path.join("datasets", d_cfg['folder_name'])
    out_dir = f"outputs_{model_id}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n[INFO] Starting YOLO Detection Pipeline")
    print(f"       Model:    {m_cfg['name']}")
    print(f"       Dataset:  {d_cfg['name']}")
    print(f"       Threads:  {num_consumers} consumers, {num_producers} producers")
    print(f"       Output:   {out_dir}/")

    try:
        subgraph, dpu_shape, fix_pos_in, fix_pos_outs = setup_dpu(model_path)
    except Exception as e:
        print(f"[ERROR] Failed to load model {model_path}: {e}")
        return

    # Output-tensor permutation is fixed for a given xmodel: largest spatial
    # grid first (P3, P4, P5). Compute once.
    runner_tmp = vart.Runner.create_runner(subgraph, "run")
    out_dims = [tuple(t.dims) for t in runner_tmp.get_output_tensors()]
    del runner_tmp
    output_order = sorted(range(len(out_dims)), key=lambda i: out_dims[i][1], reverse=True)

    all_images = [os.path.join(dataset_path, f)
                  for f in os.listdir(dataset_path)
                  if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if not all_images:
        print(f"[ERROR] No images found in {dataset_path}")
        return

    img_queue = queue.Queue(maxsize=40)
    write_queue = queue.Queue(maxsize=128)
    progress = ProgressCounter()
    results = [None] * num_consumers
    total_imgs = len(all_images)

    chunk_size = (total_imgs + num_producers - 1) // num_producers
    chunks = [all_images[i:i + chunk_size] for i in range(0, total_imgs, chunk_size)]

    monitor = PowerMonitor()
    monitor.start()
    end_wall = time.time()
    start_wall = end_wall

    try:
        idle_p = float(np.mean([get_power_mw() / 1000.0 for _ in range(5)]))
        start_wall = time.time()

        # Async writer thread.
        w_thread = threading.Thread(target=writer_worker, args=(write_queue,), daemon=True)
        w_thread.start()

        c_threads = []
        for i in range(num_consumers):
            t = threading.Thread(target=consumer_worker, args=(
                i, img_queue, write_queue, subgraph, out_dir, m_cfg, d_cfg,
                fix_pos_outs, output_order, progress, results))
            t.start()
            c_threads.append(t)

        p_threads = []
        for i in range(num_producers):
            if i >= len(chunks):
                break
            t = threading.Thread(target=producer_worker, args=(
                chunks[i], img_queue, dpu_shape,
                d_cfg['normalization']['mean'], d_cfg['normalization']['std'], fix_pos_in))
            t.start()
            p_threads.append(t)

        for t in p_threads:
            t.join()
        for _ in range(num_consumers):
            img_queue.put(None)

        print(f"[INFO] DPU Processing & NMS started...")
        while any(t.is_alive() for t in c_threads):
            sys.stdout.write(f"\r[INFO] Progress: {progress.value}/{total_imgs} "
                             f"({(progress.value/total_imgs)*100:.1f}%) ")
            sys.stdout.flush()
            time.sleep(0.5)

        write_queue.put(None)
        w_thread.join()

        sys.stdout.write(f"\r[INFO] Progress: {total_imgs}/{total_imgs} (100.0%) Done!\n")
        end_wall = time.time()
    finally:
        monitor.stop()

    # ---- Report ----
    total_wall_time = end_wall - start_wall
    total_images = sum(r[0] for r in results if r)
    total_dpu_time = sum(r[1] for r in results if r)

    fps_app = total_images / total_wall_time if total_wall_time > 0 else 0.0
    avg_dpu_latency = total_dpu_time / total_images if total_images > 0 else 0.0

    avg_load_pwr = monitor.average(fallback=idle_p)
    energy_per_frame = (avg_load_pwr / fps_app) * 1000 if fps_app > 0 else 0.0
    duty_cycle = (total_dpu_time / (total_wall_time * num_consumers)) * 100 if total_wall_time > 0 else 0.0
    compute_eff = (fps_app * m_cfg['gops'] / DPU_PEAK_GOPS) * 100

    report = format_report(
        f"DETECTION REPORT: {m_cfg['name'].upper()} | DPU THREADS: {num_consumers}",
        [
            ("Images Processed:", f"{total_images}"),
            ("---", None),
            ("Application FPS:", f"{fps_app:.2f} img/s"),
            ("DPU Latency (avg):", f"{avg_dpu_latency*1000:.2f} ms"),
            ("---", None),
            ("Power (Load):", f"{avg_load_pwr:.2f} W"),
            ("Energy per frame:", f"{energy_per_frame:.2f} mJ/img"),
            ("---", None),
            ("DPU Duty Cycle:", f"{min(duty_cycle, 100.0):.2f} %"),
            ("DPU Compute Eff.:", f"{compute_eff:.2f} %"),
            ("---", None),
            ("Output Images:", f"./{out_dir}/"),
        ],
    )
    print("\n" + report)

    with open(f"results_{model_id}_t{num_consumers}.txt", "w") as f:
        f.write(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--threads', type=int)
    args = parser.parse_args()

    run_detection(args.model, args.dataset, args.threads)
