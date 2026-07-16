"""
Board-side YOLO instance segmentation runner.
"""
import os
import sys
import time
import threading
import queue
import argparse
import json
import traceback

import numpy as np
import cv2
# --- GStreamer-safe frame sequence writer wrapper ---
class _FrameSeqWriter:
    def __init__(self, filename, fourcc, fps, framesize):
        self.dir = filename + "_frames"
        os.makedirs(self.dir, exist_ok=True)
        self.cnt = 0
        self._opened = True
    def isOpened(self): return self._opened
    def write(self, frame):
        cv2.imwrite(os.path.join(self.dir, f"frame_{self.cnt:08d}.jpg"), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        self.cnt += 1
    def release(self): pass
cv2.VideoWriter = _FrameSeqWriter

import vart

from model_config import get_active_model
from dataset_config import get_active_dataset
from board_config import ACTIVE_THREADS, DPU_PEAK_GOPS, get_power_mw
from board_utils import (
    PowerMonitor, ProgressCounter, setup_dpu,
    build_norm_lut, apply_norm_lut, format_report,
    StageProfiler, merge_stage_profilers, format_profile_report,
)
from detection_utils import (
    letterbox, scale_coords, non_max_suppression,
    UltralyticsDecoderCache, decode_ultralytics_output, _output_spatial_rank,
    _softmax_last, _as_nhwc,
)
from seg_utils import process_mask, scale_image_masks, load_yolo_seg_labels, mask_iou_matrix, compute_ap

SEGMENTATION_PROFILE_GROUPS = [
    ("Setup", [
        "consumer_runner_create",
        "consumer_output_alloc",
        "consumer_dequant_setup",
    ]),
    ("Pipeline Totals", [
        "producer_total",
        "consumer_total",
        "writer_total",
    ]),
    ("Per-frame Latency", [
        "latency_preprocess_ready",
        "latency_input_queue",
        "latency_result_ready",
        "latency_write_queue",
        "latency_full_output",
    ]),
    ("Preprocessing", [
        "image_read",
        "bgr_to_rgb",
        "letterbox_total",
        "norm_lut",
        "expand_dims",
    ]),
    ("DPU / VART", [
        "dpu_api_total",
        "dpu_submit",
        "dpu_wait",
    ]),
    ("Decode", [
        "decode_total",
        "decode_ultra_threshold",
        "decode_ultra_dequant",
        "decode_ultra_class_score",
        "decode_ultra_box_decode",
        "decode_ultra_concat",
    ]),
    ("Post-processing", [
        "nms_or_topk",
        "coord_scale",
        "mask_assembly",
        "mask_scale",
        "draw_overlays",
    ]),
    ("Queue / Write", [
        "consumer_dequeue_wait",
        "producer_enqueue_wait",
        "consumer_enqueue_write_wait",
        "writer_dequeue_wait",
        "image_write",
    ]),
]

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
# OUTPUT ORDER RESOLUTION
# =============================================================
def resolve_segmentation_outputs(out_dims, num_classes, num_masks, reg_max):
    """
    Given the 10 output tensors of an instance-seg head, return:
      - det_order: order of the 6 box/cls tensors (largest spatial first)
      - mask_order: order of the 3 mask-coeff tensors (largest spatial first)
      - proto_idx: index of the 1 prototype tensor (largest spatial among num_masks channel tensors)
    """
    proto_candidates = [i for i, d in enumerate(out_dims) if d[-1] == num_masks]
    proto_idx = max(proto_candidates, key=lambda i: out_dims[i][1] * out_dims[i][2])
    
    expected_box = 4 * reg_max
    det_order_candidates = [i for i, d in enumerate(out_dims) if d[-1] in (expected_box, num_classes)]
    det_order = sorted(det_order_candidates, key=lambda i: out_dims[i][1] * out_dims[i][2], reverse=True)
    
    mask_order_candidates = [i for i in proto_candidates if i != proto_idx]
    mask_order = sorted(mask_order_candidates, key=lambda i: out_dims[i][1] * out_dims[i][2], reverse=True)
    
    return det_order, mask_order, proto_idx

# =============================================================
# PRODUCER: letterbox + LUT normalization
# =============================================================
def producer_worker(image_chunk, input_queue, dpu_shape, lut, profiler=None):
    dpu_h, dpu_w = dpu_shape[1], dpu_shape[2]
    for img_path in image_chunk:
        trace = (
            {"start": time.perf_counter()}
            if profiler is not None and profiler.enabled else None
        )
        total_start = _profile_start(profiler)
        stage_start = _profile_start(profiler)
        orig_img = cv2.imread(img_path)
        _profile_end(profiler, "image_read", stage_start)
        if orig_img is None:
            _profile_end(profiler, "producer_total", total_start)
            continue
        orig_shape = orig_img.shape[:2]

        stage_start = _profile_start(profiler)
        img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        _profile_end(profiler, "bgr_to_rgb", stage_start)

        stage_start = _profile_start(profiler)
        img_resized, _, _ = letterbox(img_rgb, new_shape=(dpu_h, dpu_w))
        _profile_end(profiler, "letterbox_total", stage_start)

        stage_start = _profile_start(profiler)
        img_norm = apply_norm_lut(img_resized, lut)
        _profile_end(profiler, "norm_lut", stage_start)

        stage_start = _profile_start(profiler)
        img_int8 = np.expand_dims(img_norm, axis=0)
        _profile_end(profiler, "expand_dims", stage_start)

        stage_start = _profile_start(profiler)
        if trace is not None:
            trace["producer_done"] = time.perf_counter()
            _profile_add(profiler, "latency_preprocess_ready",
                         trace["producer_done"] - trace["start"])
        input_queue.put((img_int8, orig_img, orig_shape,
                         os.path.basename(img_path), trace))
        _profile_end(profiler, "producer_enqueue_wait", stage_start)
        _profile_end(profiler, "producer_total", total_start)


# =============================================================
# WRITER: async image saver (decouples cv2.imwrite from consumer)
# =============================================================
def writer_worker(write_queue, profiler=None):
    while True:
        stage_start = _profile_start(profiler)
        item = write_queue.get()
        _profile_end(profiler, "writer_dequeue_wait", stage_start)
        if item is None:
            write_queue.task_done()
            break
        out_path, img, trace = item
        if trace is not None:
            trace["writer_start"] = time.perf_counter()
            _profile_add(profiler, "latency_write_queue",
                         trace["writer_start"] - trace["consumer_done"])
        total_start = _profile_start(profiler)
        stage_start = _profile_start(profiler)
        cv2.imwrite(out_path, img)
        _profile_end(profiler, "image_write", stage_start)
        if trace is not None:
            trace["writer_done"] = time.perf_counter()
            _profile_add(profiler, "latency_full_output",
                         trace["writer_done"] - trace["start"])
        _profile_end(profiler, "writer_total", total_start)
        write_queue.task_done()

# =============================================================
# CONSUMER: DPU + decode + per-class NMS + mask assembly + draw
# =============================================================
def consumer_worker(thread_id, input_queue, write_queue, dpu_subgraph,
                    out_dir, m_cfg, d_cfg, fix_pos_outs, det_order, mask_order,
                    proto_idx, progress, results, profiler=None, draw_outputs=True,
                    save_outputs=True, evaluate_accuracy=False, labels_dir=None):
    stage_start = _profile_start(profiler)
    runner = vart.Runner.create_runner(dpu_subgraph, "run")
    _profile_end(profiler, "consumer_runner_create", stage_start)
    print(f"[DEBUG] Consumer {thread_id}: DPU runner created, entering loop.")

    stage_start = _profile_start(profiler)
    output_tensors = runner.get_output_tensors()
    output_data = [np.empty(tuple(t.dims), dtype=np.int8) for t in output_tensors]
    _profile_end(profiler, "consumer_output_alloc", stage_start)

    stage_start = _profile_start(profiler)
    dequant_scales = [np.float32(2 ** -fp) for fp in fix_pos_outs]
    _profile_end(profiler, "consumer_dequant_setup", stage_start)

    conf_thresh = m_cfg.get('conf_threshold', 0.25)
    iou_thresh = m_cfg.get('iou_threshold', 0.45)
    mask_thresh = m_cfg.get('mask_threshold', 0.5)
    max_det = m_cfg.get('max_det', 300)
    dpu_shape = tuple(runner.get_input_tensors()[0].dims)[1:3]  # H, W

    cache = UltralyticsDecoderCache(m_cfg['strides'])
    num_classes = m_cfg.get('num_classes', len(d_cfg.get('classes', [])))
    reg_max = m_cfg.get('reg_max', 1)
    num_masks = m_cfg.get('num_masks', 32)
    class_names = d_cfg.get('classes')

    np.random.seed(42)
    class_colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(num_classes)]

    local_total = 0
    local_dpu_time = 0.0
    local_class_hist = {}
    local_eval_records = []
    local_gt_counts = {cid: 0 for cid in range(num_classes)}

    while True:
        stage_start = _profile_start(profiler)
        item = input_queue.get()
        _profile_end(profiler, "consumer_dequeue_wait", stage_start)
        if item is None:
            input_queue.task_done()
            break

        img_int8, orig_img, orig_shape, file_name, trace = item
        if trace is not None:
            trace["consumer_start"] = time.perf_counter()
            _profile_add(profiler, "latency_input_queue",
                         trace["consumer_start"] - trace["producer_done"])

        # 1. DPU execution
        consumer_total_start = _profile_start(profiler)
        dpu_total_start = _profile_start(profiler)
        t_start = time.perf_counter()
        stage_start = _profile_start(profiler)
        jid = runner.execute_async([img_int8], output_data)
        _profile_end(profiler, "dpu_submit", stage_start)

        stage_start = _profile_start(profiler)
        runner.wait(jid)
        _profile_end(profiler, "dpu_wait", stage_start)
        local_dpu_time += time.perf_counter() - t_start
        _profile_end(profiler, "dpu_api_total", dpu_total_start)

        # 2. Decode detection heads
        stage_start = _profile_start(profiler)
        boxes, scores, class_ids, keep_indices = decode_ultralytics_output(
            output_data, dequant_scales, conf_thresh, cache, det_order,
            num_classes, reg_max, profiler, return_keep_index=True)
        _profile_end(profiler, "decode_total", stage_start)

        result_ready_recorded = False
        
        # 3. Post-process
        if boxes.shape[0] > 0:
            stage_start = _profile_start(profiler)
            if m_cfg.get('decoder') == 'ultralytics_anchor_free':
                if scores.shape[0] > max_det:
                    indices = np.argpartition(-scores, max_det)[:max_det]
                else:
                    indices = np.arange(scores.shape[0])
            else:
                indices = non_max_suppression(
                    boxes, scores, conf_thresh, iou_thresh, class_ids=class_ids)
            _profile_end(profiler, "nms_or_topk", stage_start)

            if len(indices) > 0:
                final_boxes = boxes[indices]
                final_class_ids = class_ids[indices]
                final_scores = scores[indices]
                final_keep = keep_indices[indices]
                
                for cid in final_class_ids:
                    cid = int(cid)
                    local_class_hist[cid] = local_class_hist.get(cid, 0) + 1

                # Gather mask coefficients
                stage_start = _profile_start(profiler)
                mask_coeffs = np.empty((len(final_keep), num_masks), dtype=np.float32)
                for j in range(len(final_keep)):
                    level_idx = final_keep[j, 0]
                    flat_idx = final_keep[j, 1]
                    mask_idx = mask_order[level_idx]
                    scale = dequant_scales[mask_idx]
                    mask_int8 = output_data[mask_idx].reshape(-1, num_masks)[flat_idx]
                    mask_coeffs[j] = mask_int8.astype(np.float32) * scale
                
                proto_scale = dequant_scales[proto_idx]
                proto_tensor = output_data[proto_idx][0].astype(np.float32) * proto_scale
                proto = proto_tensor.transpose(2, 0, 1) # (nm, H, W)
                
                # xywh -> xyxy for boxes (in DPU space)
                stage_start = _profile_start(profiler)
                xyxy_dpu = final_boxes.copy()
                xyxy_dpu[:, 2] = final_boxes[:, 0] + final_boxes[:, 2]
                xyxy_dpu[:, 3] = final_boxes[:, 1] + final_boxes[:, 3]

                # Assemble masks at prototype resolution; scale_image_masks
                # warps straight to original image space (no 160->640 upsample).
                masks = process_mask(proto, mask_coeffs, xyxy_dpu, dpu_shape, upsample=False)
                _profile_end(profiler, "mask_assembly", stage_start)

                stage_start = _profile_start(profiler)
                xyxy = scale_coords(dpu_shape, xyxy_dpu.copy(), orig_shape)
                _profile_end(profiler, "coord_scale", stage_start)
                
                # Scale masks back to original image shape
                stage_start = _profile_start(profiler)
                masks = scale_image_masks(masks, xyxy_dpu, dpu_shape, orig_shape)
                binary_masks = masks > mask_thresh
                _profile_end(profiler, "mask_scale", stage_start)

                if evaluate_accuracy and labels_dir:
                    label_path = os.path.join(labels_dir, os.path.splitext(file_name)[0] + ".txt")
                    gt_classes, gt_masks = load_yolo_seg_labels(label_path, orig_shape)
                    for cid in gt_classes:
                        local_gt_counts[cid] += 1
                        
                    mask_iou = mask_iou_matrix(binary_masks, gt_masks)
                    gt_classes_arr = np.array(gt_classes)
                    tp = np.zeros(len(final_class_ids), dtype=bool)
                    
                    if len(gt_classes) > 0 and len(final_class_ids) > 0:
                        sort_idx = np.argsort(-final_scores)
                        pred_classes_sorted = final_class_ids[sort_idx]
                        mask_iou_sorted = mask_iou[sort_idx]
                        
                        gt_matched = np.zeros(len(gt_classes), dtype=bool)
                        
                        for i, p_cls in enumerate(pred_classes_sorted):
                            match_idx = np.where((gt_classes_arr == p_cls) & ~gt_matched)[0]
                            if len(match_idx) > 0:
                                ious = mask_iou_sorted[i, match_idx]
                                best_idx = match_idx[np.argmax(ious)]
                                if mask_iou_sorted[i, best_idx] > 0.5:
                                    tp[sort_idx[i]] = True
                                    gt_matched[best_idx] = True

                    for j in range(len(final_class_ids)):
                        local_eval_records.append((final_class_ids[j], final_scores[j], tp[j]))
                elif evaluate_accuracy:
                    for j in range(len(final_class_ids)):
                        local_eval_records.append((final_class_ids[j], final_scores[j], False))

                if trace is not None:
                    trace["result_ready"] = time.perf_counter()
                    _profile_add(profiler, "latency_result_ready",
                                 trace["result_ready"] - trace["start"])
                    result_ready_recorded = True

                if draw_outputs:
                    stage_start = _profile_start(profiler)
                    overlay = orig_img.copy()
                    alpha = 0.5
                    
                    for j in range(xyxy.shape[0]):
                        x1, y1, x2, y2 = map(int, xyxy[j, :4])
                        cid = int(final_class_ids[j])
                        conf = float(final_scores[j])
                        color = class_colors[cid]
                        
                        mask = binary_masks[j]
                        overlay[mask] = np.array(color) * alpha + overlay[mask] * (1 - alpha)
                        
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                        name = class_names[cid] if class_names and cid < len(class_names) else f"Class {cid}"
                        cv2.putText(overlay, f"{name}: {conf:.2f}",
                                    (x1, max(15, y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    cv2.addWeighted(overlay, 1.0, orig_img, 0.0, 0, orig_img)
                    _profile_end(profiler, "draw_overlays", stage_start)

        if trace is not None and not result_ready_recorded:
            trace["result_ready"] = time.perf_counter()
            _profile_add(profiler, "latency_result_ready",
                         trace["result_ready"] - trace["start"])

        if save_outputs:
            stage_start = _profile_start(profiler)
            if trace is not None:
                trace["consumer_done"] = time.perf_counter()
            write_queue.put((os.path.join(out_dir, file_name), orig_img, trace))
            _profile_end(profiler, "consumer_enqueue_write_wait", stage_start)

        local_total += 1
        progress.increment()
        input_queue.task_done()
        _profile_end(profiler, "consumer_total", consumer_total_start)

    results[thread_id] = (local_total, local_dpu_time, local_class_hist, profiler, local_eval_records, local_gt_counts)
    del runner


# =============================================================
# THREAD WRAPPERS: surface exceptions (default threads swallow them)
# =============================================================
def _safe_producer(worker_args):
    """Run producer_worker, printing any exception instead of dying silently."""
    try:
        producer_worker(*worker_args)
    except Exception:
        print(f"\n[ERROR] Producer thread crashed:")
        traceback.print_exc()
        sys.stdout.flush()


def _safe_consumer(worker_args):
    """Run consumer_worker, printing any exception instead of dying silently.

    On crash, keep draining the input queue so the producers do not block
    forever on a full queue (which would hang the main thread at ``t.join()``).
    """
    thread_id = worker_args[0]
    input_queue = worker_args[1]
    try:
        consumer_worker(*worker_args)
    except Exception:
        print(f"\n[ERROR] Consumer {thread_id} crashed:")
        traceback.print_exc()
        sys.stdout.flush()
        # Drain so producers unblock; re-inject the sentinel for any siblings.
        while True:
            item = input_queue.get()
            input_queue.task_done()
            if item is None:
                try:
                    input_queue.put_nowait(None)
                except queue.Full:
                    pass
                break


def _compute_efficiency(m_cfg, fps_app):
    gops = m_cfg.get('gops')
    if not gops or fps_app <= 0:
        return "N/A"
    return f"{(float(gops) * fps_app / DPU_PEAK_GOPS) * 100.0:.2f} %"


def _validate_worker_counts(thread_override, producers_override, video_path):
    max_consumers = 4
    if video_path:
        if thread_override is not None and thread_override != 1:
            raise ValueError("Video mode requires --threads 1 to preserve frame order and measure per-frame latency.")
        if producers_override is not None and producers_override != 1:
            raise ValueError("Video mode requires --producers 1 to preserve frame order and measure per-frame latency.")
        return 1, 1

    num_consumers = thread_override if thread_override is not None else ACTIVE_THREADS
    num_producers = producers_override if producers_override is not None else 4
    if not 1 <= num_consumers <= max_consumers:
        raise ValueError(
            f"--threads must be between 1 and {max_consumers} for the KV260. Got: {num_consumers}.")
    if num_producers < 1:
        raise ValueError(f"--producers must be at least 1. Got: {num_producers}.")
    return num_consumers, num_producers


def run_video_instance_seg(model_id, m_cfg, d_cfg, subgraph, dpu_shape, lut,
                           fix_pos_outs, det_order, mask_order, proto_idx,
                           video_path, output_video, draw_outputs, save_outputs,
                           profile_enabled, queue_size):
    if not os.path.isfile(video_path):
        print(f"[ERROR] Video file not found: {video_path}")
        return

    capture = None
    for backend, name in ((cv2.CAP_FFMPEG, "FFMPEG"),
                          (cv2.CAP_GSTREAMER, "GStreamer"),
                          (cv2.CAP_ANY, "ANY")):
        try:
            capture = cv2.VideoCapture(video_path, backend)
        except Exception as e:
            print(f"[WARN] VideoCapture backend {name} raised: {e}")
            continue
        if capture.isOpened():
            print(f"[INFO] Opened video with backend: {name}")
            break
    if capture is None or not capture.isOpened():
        print(f"[ERROR] Failed to open video: {video_path}")
        return

    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    source_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if source_width <= 0 or source_height <= 0:
        capture.release()
        print(f"[ERROR] Video has invalid dimensions: {source_width}x{source_height}")
        return
    if source_fps <= 0 or not np.isfinite(source_fps):
        source_fps = 30.0
        print("[WARN] Video FPS metadata is invalid; using 30.0 FPS for output encoding.")

    if save_outputs:
        if output_video is None:
            output_video = f"outputs_{model_id}.mp4"
        if not output_video.lower().endswith(('.mp4', '.avi')):
            capture.release()
            print(f"[ERROR] --output-video must use .mp4 or .avi extension: {output_video}")
            return
        output_dir = os.path.dirname(output_video)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        writer = cv2.VideoWriter(
            output_video, cv2.VideoWriter_fourcc(*'mp4v'), source_fps,
            (source_width, source_height))
        if not writer.isOpened():
            print(f"[WARN] MP4V writer failed; falling back to MJPG/AVI.")
            base, _ = os.path.splitext(output_video)
            output_video = f"{base}.avi"
            writer = cv2.VideoWriter(
                output_video, cv2.VideoWriter_fourcc(*'MJPG'), source_fps,
                (source_width, source_height))
            if not writer.isOpened():
                capture.release()
                print(f"[ERROR] Failed to open video writer: {output_video}")
                return
    else:
        writer = None
        output_video = None

    profiler = StageProfiler(enabled=profile_enabled)
    stage_start = _profile_start(profiler)
    runner = vart.Runner.create_runner(subgraph, "run")
    _profile_end(profiler, "consumer_runner_create", stage_start)
    stage_start = _profile_start(profiler)
    output_data = [np.empty(tuple(t.dims), dtype=np.int8)
                   for t in runner.get_output_tensors()]
    _profile_end(profiler, "consumer_output_alloc", stage_start)
    stage_start = _profile_start(profiler)
    dequant_scales = [np.float32(2 ** -fp) for fp in fix_pos_outs]
    _profile_end(profiler, "consumer_dequant_setup", stage_start)

    num_classes = m_cfg.get('num_classes', len(d_cfg.get('classes', [])))
    num_masks = m_cfg.get('num_masks', 32)
    reg_max = m_cfg.get('reg_max', 1)
    conf_thresh = m_cfg.get('conf_threshold', 0.25)
    iou_thresh = m_cfg.get('iou_threshold', 0.45)
    mask_thresh = m_cfg.get('mask_threshold', 0.5)
    max_det = m_cfg.get('max_det', 300)
    class_names = d_cfg.get('classes')
    cache = UltralyticsDecoderCache(m_cfg['strides'])
    color_rng = np.random.default_rng(42)
    class_colors = [tuple(color_rng.integers(0, 255, 3).tolist())
                    for _ in range(num_classes)]

    print(f"\n[INFO] Starting YOLO Instance Segmentation Video Pipeline")
    print(f"       Model:    {m_cfg['name']}")
    print(f"       Dataset:  {d_cfg['name']}")
    print(f"       Video:    {video_path}")
    print(f"       Source:   {source_width}x{source_height} @ {source_fps:.3f} FPS")
    print(f"       Frames:   {source_frame_count if source_frame_count > 0 else 'unknown'}")
    print(f"       Threads:  1 consumer, 1 producer")
    print(f"       Queue:    input maxsize {queue_size}")
    print(f"       Draw:     {'enabled' if draw_outputs else 'disabled'}")
    print(f"       Save:     {'enabled' if save_outputs else 'disabled'}")
    print(f"       Output:   {output_video if output_video else 'disabled'}")
    if profile_enabled:
        print("       Profile:  enabled")

    monitor = PowerMonitor()
    monitor.start()
    idle_power = 0.0
    frames_processed = 0
    total_dpu_time = 0.0
    class_hist = {}
    start_wall = time.time()
    end_wall = start_wall

    try:
        idle_power = float(np.mean([get_power_mw() / 1000.0 for _ in range(5)]))
        start_wall = time.time()
        while True:
            frame_start = time.perf_counter()
            producer_total_start = _profile_start(profiler)
            stage_start = _profile_start(profiler)
            ok, orig_img = capture.read()
            _profile_end(profiler, "image_read", stage_start)
            if not ok:
                break

            orig_shape = orig_img.shape[:2]
            stage_start = _profile_start(profiler)
            img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            _profile_end(profiler, "bgr_to_rgb", stage_start)
            stage_start = _profile_start(profiler)
            img_resized, _, _ = letterbox(
                img_rgb, new_shape=(dpu_shape[1], dpu_shape[2]))
            _profile_end(profiler, "letterbox_total", stage_start)
            stage_start = _profile_start(profiler)
            img_int8 = np.expand_dims(apply_norm_lut(img_resized, lut), axis=0)
            _profile_end(profiler, "norm_lut", stage_start)
            _profile_add(profiler, "expand_dims", 0.0)
            _profile_end(profiler, "producer_total", producer_total_start)
            _profile_add(profiler, "latency_preprocess_ready",
                         time.perf_counter() - frame_start)

            consumer_total_start = _profile_start(profiler)
            dpu_total_start = _profile_start(profiler)
            dpu_start = time.perf_counter()
            stage_start = _profile_start(profiler)
            job_id = runner.execute_async([img_int8], output_data)
            _profile_end(profiler, "dpu_submit", stage_start)
            stage_start = _profile_start(profiler)
            runner.wait(job_id)
            _profile_end(profiler, "dpu_wait", stage_start)
            total_dpu_time += time.perf_counter() - dpu_start
            _profile_end(profiler, "dpu_api_total", dpu_total_start)

            stage_start = _profile_start(profiler)
            boxes, scores, class_ids, keep_indices = decode_ultralytics_output(
                output_data, dequant_scales, conf_thresh, cache, det_order,
                num_classes, reg_max, profiler, return_keep_index=True)
            _profile_end(profiler, "decode_total", stage_start)

            result_ready_recorded = False
            if boxes.shape[0] > 0:
                stage_start = _profile_start(profiler)
                if m_cfg.get('decoder') == 'ultralytics_anchor_free':
                    if scores.shape[0] > max_det:
                        indices = np.argpartition(-scores, max_det)[:max_det]
                    else:
                        indices = np.arange(scores.shape[0])
                else:
                    indices = non_max_suppression(
                        boxes, scores, conf_thresh, iou_thresh, class_ids=class_ids)
                _profile_end(profiler, "nms_or_topk", stage_start)

                if len(indices) > 0:
                    final_boxes = boxes[indices]
                    final_class_ids = class_ids[indices]
                    final_scores = scores[indices]
                    final_keep = keep_indices[indices]
                    for class_id in final_class_ids:
                        class_id = int(class_id)
                        class_hist[class_id] = class_hist.get(class_id, 0) + 1

                    stage_start = _profile_start(profiler)
                    mask_coeffs = np.empty((len(final_keep), num_masks), dtype=np.float32)
                    for index, (level_idx, flat_idx) in enumerate(final_keep):
                        mask_idx = mask_order[level_idx]
                        scale = dequant_scales[mask_idx]
                        mask_int8 = output_data[mask_idx].reshape(-1, num_masks)[flat_idx]
                        mask_coeffs[index] = mask_int8.astype(np.float32) * scale
                    proto = (output_data[proto_idx][0].astype(np.float32) *
                             dequant_scales[proto_idx]).transpose(2, 0, 1)
                    xyxy_dpu = final_boxes.copy()
                    xyxy_dpu[:, 2] = final_boxes[:, 0] + final_boxes[:, 2]
                    xyxy_dpu[:, 3] = final_boxes[:, 1] + final_boxes[:, 3]
                    masks = process_mask(
                        proto, mask_coeffs, xyxy_dpu, dpu_shape[1:3], upsample=False)
                    _profile_end(profiler, "mask_assembly", stage_start)

                    stage_start = _profile_start(profiler)
                    xyxy = scale_coords(dpu_shape[1:3], xyxy_dpu.copy(), orig_shape)
                    _profile_end(profiler, "coord_scale", stage_start)
                    stage_start = _profile_start(profiler)
                    binary_masks = scale_image_masks(
                        masks, xyxy_dpu, dpu_shape[1:3], orig_shape) > mask_thresh
                    _profile_end(profiler, "mask_scale", stage_start)

                    _profile_add(profiler, "latency_result_ready",
                                 time.perf_counter() - frame_start)
                    result_ready_recorded = True

                    if draw_outputs:
                        stage_start = _profile_start(profiler)
                        overlay = orig_img.copy()
                        for index, box in enumerate(xyxy):
                            x1, y1, x2, y2 = map(int, box[:4])
                            class_id = int(final_class_ids[index])
                            confidence = float(final_scores[index])
                            color = class_colors[class_id]
                            mask = binary_masks[index]
                            overlay[mask] = np.array(color) * 0.5 + overlay[mask] * 0.5
                            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                            name = (class_names[class_id] if class_names and
                                    class_id < len(class_names) else f"Class {class_id}")
                            cv2.putText(overlay, f"{name}: {confidence:.2f}",
                                        (x1, max(15, y1 - 10)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        cv2.addWeighted(overlay, 1.0, orig_img, 0.0, 0, orig_img)
                        _profile_end(profiler, "draw_overlays", stage_start)

            if not result_ready_recorded:
                _profile_add(profiler, "latency_result_ready",
                             time.perf_counter() - frame_start)
            if writer is not None:
                writer_total_start = _profile_start(profiler)
                stage_start = _profile_start(profiler)
                writer.write(orig_img)
                _profile_end(profiler, "image_write", stage_start)
                _profile_end(profiler, "writer_total", writer_total_start)
                _profile_add(profiler, "latency_full_output",
                             time.perf_counter() - frame_start)
            _profile_end(profiler, "consumer_total", consumer_total_start)

            frames_processed += 1
            if frames_processed % 10 == 0:
                total_label = source_frame_count if source_frame_count > 0 else '?'
                print(f"\r[INFO] Progress: {frames_processed}/{total_label}", end='', flush=True)
        if frames_processed:
            print()
        end_wall = time.time()
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        monitor.stop()
        del runner

    total_wall_time = end_wall - start_wall
    fps_app = frames_processed / total_wall_time if total_wall_time > 0 else 0.0
    avg_dpu_latency = total_dpu_time / frames_processed if frames_processed > 0 else 0.0
    avg_load_power = monitor.average(fallback=idle_power)
    energy_per_frame = (avg_load_power / fps_app) * 1000.0 if fps_app > 0 else 0.0
    duty_cycle = (total_dpu_time / total_wall_time) * 100.0 if total_wall_time > 0 else 0.0
    compute_efficiency = _compute_efficiency(m_cfg, fps_app)

    report = format_report(
        f"VIDEO SEGMENTATION REPORT: {m_cfg['name'].upper()} | DPU THREADS: 1",
        [
            ("Frames Processed:", f"{frames_processed}"),
            ("Source FPS:", f"{source_fps:.3f}"),
            ("Application FPS:", f"{fps_app:.2f} img/s"),
            ("DPU Latency (avg):", f"{avg_dpu_latency * 1000.0:.2f} ms"),
            ("---", None),
            ("Power (Load):", f"{avg_load_power:.2f} W"),
            ("Energy per frame:", f"{energy_per_frame:.2f} mJ/img"),
            ("---", None),
            ("DPU Duty Cycle:", f"{min(duty_cycle, 100.0):.2f} %"),
            ("DPU Compute Eff.:", compute_efficiency),
            ("---", None),
            ("Input Video:", video_path),
            ("Output Video:", output_video if output_video else "disabled"),
        ],
    )
    print("\n" + report)
    full_report = report

    if class_hist:
        total_detections = sum(class_hist.values())
        print(f"\nDETECTION CLASS HISTOGRAM (total {total_detections} detections):")
        print("-" * 60)
        for class_id, count in sorted(class_hist.items(), key=lambda item: -item[1])[:20]:
            name = (class_names[class_id] if class_names and
                    class_id < len(class_names) else f"Class {class_id}")
            print(f"    {name:<20s} {count:6d}  ({100 * count / total_detections:.2f}%)")
        print()

    merged_profiler = None
    if profile_enabled:
        merged_profiler = merge_stage_profilers([profiler])
        profile_report = format_profile_report(
            "DETAILED PERFORMANCE PROFILE", merged_profiler,
            total_wall_time, SEGMENTATION_PROFILE_GROUPS)
        profile_note = (
            "Memory-transfer note: with the current NumPy-based VART Python API, "
            "explicit cache/DMA sync time is not separated. Use dpu_submit, "
            "dpu_wait, and dpu_api_total as the observable VART timing breakdown.\n"
        )
        print(profile_report + profile_note)
        full_report += "\n" + profile_report + profile_note

    result_path = f"results_{model_id}_video.txt"
    with open(result_path, "w") as result_file:
        result_file.write(full_report)

    if profile_enabled and merged_profiler is not None:
        profile_path = f"results_{model_id}_video_profile.json"
        payload = {
            "model": model_id,
            "dataset": d_cfg['name'],
            "video_path": video_path,
            "output_video": output_video,
            "source_fps": source_fps,
            "source_width": source_width,
            "source_height": source_height,
            "source_frame_count": source_frame_count,
            "threads": 1,
            "producers": 1,
            "queue_size": queue_size,
            "draw_outputs": draw_outputs,
            "save_outputs": save_outputs,
            "frames_processed": frames_processed,
            "wall_time_s": total_wall_time,
            "fps_app": fps_app,
            "avg_dpu_latency_ms": avg_dpu_latency * 1000.0,
            "stages": merged_profiler.summary(total_wall_time),
            "memory_transfer_note": (
                "Current NumPy-based VART Python API does not expose separate "
                "cache/DMA sync timing; dpu_submit/dpu_wait/dpu_api_total are "
                "the observable VART timing stages."
            ),
        }
        with open(profile_path, "w") as profile_file:
            json.dump(payload, profile_file, indent=2)


# =============================================================
# MAIN
# =============================================================
def run_instance_seg(model_id, dataset_id, thread_override, profile=False,
                     profile_json=False, queue_size=40, draw_outputs=True,
                     save_outputs=True, producers_override=None,
                     evaluate_accuracy=False, labels_dir=None, video_path=None,
                     output_video=None):
    m_cfg = get_active_model(model_id)
    d_cfg = get_active_dataset(dataset_id)
    profile_enabled = profile or profile_json
    try:
        queue_size = int(queue_size)
        if queue_size < 1:
            raise ValueError(f"--queue-size must be at least 1. Got: {queue_size}.")
        num_consumers, num_producers = _validate_worker_counts(
            thread_override, producers_override, video_path)
    except ValueError as error:
        print(f"[ERROR] {error}")
        return

    if output_video and not video_path:
        print("[ERROR] --output-video requires --video.")
        return

    if evaluate_accuracy and video_path:
        print("[ERROR] --accuracy is not supported with --video.")
        return

    if m_cfg.get('type') != 'segmentation' or not m_cfg.get('seg_instance', False):
        print(f"[ERROR] Model {model_id} is not an instance segmentation model. "
              f"Use run_detection.py or run_inference.py instead.")
        sys.exit(1)
        
    decoder = m_cfg.get('decoder', 'yolov5_anchor')
    if decoder != 'ultralytics_anchor_free':
        print(f"[ERROR] Instance segmentation requires 'ultralytics_anchor_free' decoder.")
        sys.exit(1)

    model_path = f"{model_id}_kria.xmodel"
    dataset_path = os.path.join("datasets", d_cfg['folder_name'])
    out_dir = f"outputs_{model_id}"
    if save_outputs:
        os.makedirs(out_dir, exist_ok=True)

    try:
        subgraph, dpu_shape, fix_pos_in, fix_pos_outs = setup_dpu(model_path)
    except Exception as e:
        print(f"[ERROR] Failed to load model {model_path}: {e}")
        return

    lut = build_norm_lut(d_cfg['normalization']['mean'], d_cfg['normalization']['std'], fix_pos_in)

    print(f"\n[INFO] Starting YOLO Instance Segmentation Pipeline")
    print(f"       Model:    {m_cfg['name']}")
    print(f"       Dataset:  {d_cfg['name']}")
    print(f"       Threads:  {num_consumers} consumers, {num_producers} producers")
    print(f"       Queue:    input maxsize {queue_size}")
    print(f"       Draw:     {'enabled' if draw_outputs else 'disabled'}")
    print(f"       Save:     {'enabled' if save_outputs else 'disabled'}")
    if evaluate_accuracy:
        active_labels = labels_dir if labels_dir else d_cfg.get('board_labels', 'UNKNOWN')
        print(f"       Accuracy: enabled (labels: {active_labels})")
    else:
        print(f"       Accuracy: disabled")
    print(f"       Output:   {out_dir}/" if save_outputs else "       Output:   disabled")
    if profile_enabled:
        print(f"       Profile:  enabled")

    runner_tmp = vart.Runner.create_runner(subgraph, "run")
    out_dims = [tuple(t.dims) for t in runner_tmp.get_output_tensors()]
    del runner_tmp
    
    num_classes = m_cfg.get('num_classes', len(d_cfg.get('classes', [])))
    num_masks = m_cfg.get('num_masks', 32)
    reg_max = m_cfg.get('reg_max', 1)
    
    det_order, mask_order, proto_idx = resolve_segmentation_outputs(
        out_dims, num_classes, num_masks, reg_max)

    print(f"[DEBUG] out_dims={out_dims}")
    print(f"[DEBUG] num_classes={num_classes} num_masks={num_masks} "
          f"reg_max={reg_max}")
    print(f"[DEBUG] det_order={det_order} mask_order={mask_order} "
          f"proto_idx={proto_idx} fix_pos_outs={fix_pos_outs}")

    if video_path:
        run_video_instance_seg(
            model_id, m_cfg, d_cfg, subgraph, dpu_shape, lut, fix_pos_outs,
            det_order, mask_order, proto_idx, video_path, output_video,
            draw_outputs, save_outputs, profile_enabled, queue_size)
        return

    all_images = [os.path.join(dataset_path, f)
                  for f in os.listdir(dataset_path)
                  if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if not all_images:
        print(f"[ERROR] No images found in {dataset_path}")
        return

    img_queue = queue.Queue(maxsize=queue_size)
    write_queue = queue.Queue(maxsize=128) if save_outputs else None
    progress = ProgressCounter()
    results = [None] * num_consumers
    total_imgs = len(all_images)
    producer_profilers = []
    writer_profiler = StageProfiler(enabled=True) if profile_enabled and save_outputs else None

    chunk_size = (total_imgs + num_producers - 1) // num_producers
    chunks = [all_images[i:i + chunk_size] for i in range(0, total_imgs, chunk_size)]

    monitor = PowerMonitor()
    monitor.start()
    end_wall = time.time()
    start_wall = end_wall

    try:
        idle_p = float(np.mean([get_power_mw() / 1000.0 for _ in range(5)]))
        start_wall = time.time()

        w_thread = None
        if save_outputs:
            w_thread = threading.Thread(
                target=writer_worker, args=(write_queue, writer_profiler), daemon=True)
            w_thread.start()

        c_threads = []
        for i in range(num_consumers):
            consumer_profiler = StageProfiler(enabled=True) if profile_enabled else None
            t = threading.Thread(target=_safe_consumer, args=((
                i, img_queue, write_queue, subgraph, out_dir, m_cfg, d_cfg,
                fix_pos_outs, det_order, mask_order, proto_idx, progress, results,
                consumer_profiler, draw_outputs, save_outputs, evaluate_accuracy,
                labels_dir if labels_dir else d_cfg.get('board_labels')),))
            t.start()
            c_threads.append(t)

        p_threads = []
        for i in range(num_producers):
            if i >= len(chunks):
                break
            producer_profiler = StageProfiler(enabled=True) if profile_enabled else None
            producer_profilers.append(producer_profiler)
            t = threading.Thread(target=_safe_producer, args=((
                chunks[i], img_queue, dpu_shape, lut, producer_profiler),))
            t.start()
            p_threads.append(t)

        # Wait for producers, but watch for consumer death to avoid a silent
        # deadlock: if every consumer exits while the queue is full, producers
        # block forever on put(). Surface that instead of hanging at join().
        while any(t.is_alive() for t in p_threads):
            if not any(t.is_alive() for t in c_threads):
                print("\n[ERROR] All consumer threads exited before producers "
                      "finished. Producers were likely blocked on a full queue. "
                      "See the consumer traceback above.")
                break
            sys.stdout.write(
                f"\r[DEBUG] producers alive {sum(t.is_alive() for t in p_threads)}"
                f"/{len(p_threads)}  consumers alive "
                f"{sum(t.is_alive() for t in c_threads)}/{len(c_threads)}  "
                f"queue {img_queue.qsize()}  progress {progress.value}/{total_imgs}   ")
            sys.stdout.flush()
            time.sleep(0.5)
        for _ in range(num_consumers):
            try:
                img_queue.put(None, timeout=5)
            except queue.Full:
                break

        print(f"[INFO] DPU Processing & Post-processing started...")
        while any(t.is_alive() for t in c_threads):
            sys.stdout.write(f"\r[INFO] Progress: {progress.value}/{total_imgs} "
                             f"({(progress.value/total_imgs)*100:.1f}%) ")
            sys.stdout.flush()
            time.sleep(0.5)

        if save_outputs:
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

    class_hist = {}
    for r in results:
        if r and len(r) > 2:
            for cid, count in r[2].items():
                class_hist[cid] = class_hist.get(cid, 0) + count

    fps_app = total_images / total_wall_time if total_wall_time > 0 else 0.0
    avg_dpu_latency = total_dpu_time / total_images if total_images > 0 else 0.0

    avg_load_pwr = monitor.average(fallback=idle_p)
    energy_per_frame = (avg_load_pwr / fps_app) * 1000 if fps_app > 0 else 0.0
    duty_cycle = (total_dpu_time / (total_wall_time * num_consumers)) * 100 if total_wall_time > 0 else 0.0
    compute_efficiency = _compute_efficiency(m_cfg, fps_app)

    report = format_report(
        f"SEGMENTATION REPORT: {m_cfg['name'].upper()} | DPU THREADS: {num_consumers}",
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
            ("DPU Compute Eff.:", compute_efficiency),
            ("---", None),
            ("Output Images:", f"./{out_dir}/" if save_outputs else "disabled"),
        ],
    )
    print("\n" + report)
    full_report = report

    merged_profiler = None
    profile_report = ""
    if profile_enabled:
        profilers = list(producer_profilers)
        profilers.extend(r[3] for r in results if r and len(r) > 3)
        if writer_profiler is not None:
            profilers.append(writer_profiler)
        merged_profiler = merge_stage_profilers(profilers)
        profile_report = format_profile_report(
            "DETAILED PERFORMANCE PROFILE", merged_profiler,
            total_wall_time, SEGMENTATION_PROFILE_GROUPS)
        profile_note = (
            "Memory-transfer note: with the current NumPy-based VART Python API, "
            "explicit cache/DMA sync time is not separated. Use dpu_submit, "
            "dpu_wait, and dpu_api_total as the observable VART timing breakdown.\n"
        )
        print(profile_report + profile_note)
        full_report += "\n" + profile_report + profile_note

    if class_hist:
        class_names_list = d_cfg.get('classes', [])
        total_dets = sum(class_hist.values())
        print(f"\nDETECTION CLASS HISTOGRAM (total {total_dets} detections):")
        print("-" * 60)
        sorted_items = sorted(class_hist.items(), key=lambda x: -x[1])[:20]
        for cid, count in sorted_items:
            name = class_names_list[cid] if cid < len(class_names_list) else f"Class {cid}"
            print(f"    {name:<20s} {count:6d}  ({100*count/total_dets:.2f}%)")
        print()

    if evaluate_accuracy:
        print(f"\n{'='*60}\nACCURACY REPORT (Mask mAP@0.5)\n{'='*60}")
        all_eval_records = []
        gt_counts = {cid: 0 for cid in range(m_cfg.get('num_classes', len(d_cfg.get('classes', []))))}
        for r in results:
            if r and len(r) > 5:
                all_eval_records.extend(r[4])
                for cid, count in r[5].items():
                    gt_counts[cid] += count
                    
        total_gts = sum(gt_counts.values())
        if total_gts == 0:
            acc_report = "No Ground Truth labels found in labels_dir. Cannot compute mAP.\n"
            print(acc_report)
            full_report += "\n" + acc_report
        else:
            ap_per_class = []
            p_per_class = []
            r_per_class = []
            for cid in range(m_cfg.get('num_classes', len(d_cfg.get('classes', [])))):
                if gt_counts.get(cid, 0) == 0:
                    continue
                class_records = [rec for rec in all_eval_records if rec[0] == cid]
                class_records.sort(key=lambda x: x[1], reverse=True)
                
                tps = np.array([1 if rec[2] else 0 for rec in class_records])
                fps = np.array([0 if rec[2] else 1 for rec in class_records])
                
                tp_sum = np.cumsum(tps)
                fp_sum = np.cumsum(fps)
                
                recalls = tp_sum / gt_counts[cid]
                precisions = tp_sum / (tp_sum + fp_sum + 1e-16)
                
                ap = compute_ap(recalls, precisions)
                ap_per_class.append(ap)
                p_per_class.append(precisions[-1] if len(precisions) > 0 else 0.0)
                r_per_class.append(recalls[-1] if len(recalls) > 0 else 0.0)
                
            mAP = np.mean(ap_per_class) if ap_per_class else 0.0
            mean_p = np.mean(p_per_class) if p_per_class else 0.0
            mean_r = np.mean(r_per_class) if r_per_class else 0.0
            f1 = 2 * (mean_p * mean_r) / (mean_p + mean_r + 1e-16)
            
            acc_report = (
                f"Mask mAP@0.5 : {mAP:.4f}\n"
                f"Precision    : {mean_p:.4f}\n"
                f"Recall       : {mean_r:.4f}\n"
                f"F1 Score     : {f1:.4f}\n"
            )
            print(acc_report)
            full_report += "\n" + acc_report

    result_path = f"results_{model_id}_t{num_consumers}.txt"
    with open(result_path, "w") as f:
        f.write(full_report)

    if profile_json and merged_profiler is not None:
        profile_path = f"results_{model_id}_t{num_consumers}_profile.json"
        payload = {
            "model": model_id,
            "dataset": dataset_id,
            "threads": num_consumers,
            "producers": num_producers,
            "queue_size": queue_size,
            "draw_outputs": draw_outputs,
            "save_outputs": save_outputs,
            "images_processed": total_images,
            "wall_time_s": total_wall_time,
            "fps_app": fps_app,
            "avg_dpu_latency_ms": avg_dpu_latency * 1000.0,
            "stages": merged_profiler.summary(total_wall_time),
            "memory_transfer_note": (
                "Current NumPy-based VART Python API does not expose separate "
                "cache/DMA sync timing; dpu_submit/dpu_wait/dpu_api_total are "
                "the observable VART timing stages."
            ),
        }
        with open(profile_path, "w") as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        help='Model ID. Falls back to ACTIVE_MODEL_ID '
                             'in model_config.py when omitted.')
    parser.add_argument('--dataset', type=str,
                        help='Dataset ID. Falls back to ACTIVE_DATASET_ID '
                             'in dataset_config.py when omitted.')
    parser.add_argument('--threads', type=int)
    parser.add_argument('--profile', action='store_true',
                        help='Enable detailed per-stage performance profiling')
    parser.add_argument('--profile-json', action='store_true',
                        help='Write detailed profile data to JSON')
    parser.add_argument('--queue-size', type=int, default=40,
                        help='Input queue maxsize; use 1 for low-latency mode')
    parser.add_argument('--no-draw', action='store_true',
                        help='Skip drawing boxes and labels on output images')
    parser.add_argument('--no-save', action='store_true',
                        help='Skip writing output images')
    parser.add_argument('--producers', type=int, default=None,
                        help='Number of producer threads (default: 4)')
    parser.add_argument('--accuracy', action='store_true',
                        help='Compute mask mAP@0.5 and P/R/F1 using ground truth labels')
    parser.add_argument('--labels-dir', type=str, default=None,
                        help='Override path to GT labels for accuracy evaluation')
    parser.add_argument('--video', type=str, default=None,
                        help='Input video path; uses one ordered DPU inference stream')
    parser.add_argument('--output-video', type=str, default=None,
                        help='Annotated .mp4 path; defaults to outputs_<model>.mp4')
    args = parser.parse_args()

    run_instance_seg(args.model, args.dataset, args.threads,
                     profile=args.profile, profile_json=args.profile_json,
                     queue_size=args.queue_size,
                     draw_outputs=not args.no_draw,
                     save_outputs=not args.no_save,
                     producers_override=args.producers,
                     evaluate_accuracy=args.accuracy,
                     labels_dir=args.labels_dir,
                     video_path=args.video,
                     output_video=args.output_video)
