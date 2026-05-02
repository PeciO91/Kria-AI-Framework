"""
Board-side segmentation runner (work in progress).

Multi-threaded producer/consumer pipeline that runs a semantic segmentation
xmodel on the Kria DPU and saves a colorized overlay per image. Argmax is
performed directly on the raw INT8 DPU output to avoid an extra dequantize
pass on the CPU.

Status: functional skeleton. The CITYSCAPES_COLORS table is a placeholder;
final dataset palettes and accuracy/mIoU evaluation are pending.
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
    compute_norm_constants, preprocess_image, format_report,
)


# Cityscapes-style 20-class palette (BGR order for OpenCV).
CITYSCAPES_COLORS = np.array([
    [0, 0, 0],       [128, 64, 128],  [244, 35, 232],  [70, 70, 70],
    [102, 102, 156], [190, 153, 153], [153, 153, 153], [250, 170, 30],
    [220, 220, 0],   [107, 142, 35],  [152, 251, 152], [70, 130, 180],
    [220, 20, 60],   [255, 0, 0],     [0, 0, 142],     [0, 0, 70],
    [0, 60, 100],    [0, 80, 100],    [0, 0, 230],     [119, 11, 32],
], dtype=np.uint8)


# =============================================================
# PRODUCER: standard resize + INT8 normalization
# =============================================================
def producer_worker(image_chunk, input_queue, dpu_shape, norm_mean, norm_std, fix_pos):
    math_scale, math_shift = compute_norm_constants(norm_mean, norm_std, fix_pos)
    for img_path in image_chunk:
        orig_img = cv2.imread(img_path)
        if orig_img is None:
            continue
        orig_shape = orig_img.shape[:2]
        img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        img_int8 = preprocess_image(img_rgb, dpu_shape, math_scale, math_shift)
        input_queue.put((img_int8, orig_img, orig_shape, os.path.basename(img_path)))


# =============================================================
# WRITER: async overlay saver
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
# CONSUMER: DPU inference + argmax + colorized overlay
# =============================================================
def consumer_worker(thread_id, input_queue, write_queue, dpu_subgraph,
                    out_dir, progress, results):
    runner = vart.Runner.create_runner(dpu_subgraph, "run")
    output_tensors = runner.get_output_tensors()
    output_data = [np.empty(tuple(output_tensors[0].dims), dtype=np.int8)]

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

        # 2. Argmax on raw INT8 logits (monotonic, dequantize is unnecessary).
        mask_indices = np.argmax(output_data[0][0], axis=-1)

        # 3. Colorize and resize back to the original image.
        colored_mask = CITYSCAPES_COLORS[mask_indices % len(CITYSCAPES_COLORS)]
        colored_mask = cv2.resize(
            colored_mask, (orig_shape[1], orig_shape[0]),
            interpolation=cv2.INTER_NEAREST)

        # 4. Blend overlay (60% original / 40% mask) and hand off to writer.
        blended = cv2.addWeighted(orig_img, 0.6, colored_mask, 0.4, 0)
        write_queue.put((os.path.join(out_dir, file_name), blended))

        local_total += 1
        progress.increment()
        input_queue.task_done()

    results[thread_id] = (local_total, local_dpu_time)
    del runner


# =============================================================
# MAIN
# =============================================================
def run_segmentation(model_id, dataset_id, thread_override):
    m_cfg = get_active_model(model_id)
    d_cfg = get_active_dataset(dataset_id)

    if m_cfg['type'] != 'segmentation':
        print(f"[ERROR] Model {model_id} is a {m_cfg['type']} model. "
              f"Use run_inference.py or run_detection.py instead.")
        sys.exit(1)

    num_consumers = thread_override if thread_override else ACTIVE_THREADS
    num_producers = 2

    model_path = f"{model_id}_kria.xmodel"
    dataset_path = d_cfg['calib_path']
    out_dir = f"outputs_{model_id}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n[INFO] Starting Segmentation Pipeline")
    print(f"       Model:    {m_cfg['name']}")
    print(f"       Dataset:  {d_cfg['name']}")
    print(f"       Threads:  {num_consumers} consumers, {num_producers} producers")
    print(f"       Output:   {out_dir}/")

    try:
        subgraph, dpu_shape, fix_pos_in, _ = setup_dpu(model_path)
    except Exception as e:
        print(f"[ERROR] Failed to load model {model_path}: {e}")
        return

    all_images = [os.path.join(dataset_path, f)
                  for f in os.listdir(dataset_path)
                  if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if not all_images:
        print(f"[ERROR] No images found in {dataset_path}")
        return

    img_queue = queue.Queue(maxsize=20)
    write_queue = queue.Queue(maxsize=64)
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

        w_thread = threading.Thread(target=writer_worker, args=(write_queue,), daemon=True)
        w_thread.start()

        c_threads = []
        for i in range(num_consumers):
            t = threading.Thread(target=consumer_worker, args=(
                i, img_queue, write_queue, subgraph, out_dir, progress, results))
            t.start()
            c_threads.append(t)

        p_threads = []
        for i in range(num_producers):
            t = threading.Thread(target=producer_worker, args=(
                chunks[i], img_queue, dpu_shape,
                d_cfg['normalization']['mean'], d_cfg['normalization']['std'], fix_pos_in))
            t.start()
            p_threads.append(t)

        for t in p_threads:
            t.join()
        for _ in range(num_consumers):
            img_queue.put(None)

        print(f"[INFO] DPU Processing & Mask Overlay started...")
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

    run_segmentation(args.model, args.dataset, args.threads)
