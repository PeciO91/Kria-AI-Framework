"""
Board-side classification runner.

Producer/consumer pipeline that resizes and INT8-normalizes every image on
CPU and runs the classification xmodel on the DPU. Top-1 / Top-5 accuracy
is computed against the directory-encoded ground truth (one folder per
class). Power, energy-per-frame and DPU duty cycle are sampled in the
background and emitted in the final analytical report.
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


# =============================================================
# PRODUCER: CPU preprocessing
# =============================================================
def producer_worker(image_chunk, input_queue, dpu_shape, norm_mean, norm_std, fix_pos):
    math_scale, math_shift = compute_norm_constants(norm_mean, norm_std, fix_pos)
    for img_path, class_idx in image_chunk:
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_int8 = preprocess_image(img_rgb, dpu_shape, math_scale, math_shift)
        input_queue.put((img_int8, class_idx))


# =============================================================
# CONSUMER: DPU inference + Top-1/Top-5 accuracy
# =============================================================
def consumer_worker(thread_id, input_queue, dpu_subgraph, progress, results):
    runner = vart.Runner.create_runner(dpu_subgraph, "run")
    output_tensors = runner.get_output_tensors()
    output_ndim = tuple(output_tensors[0].dims)
    output_data = [np.empty(output_ndim, dtype=np.int8)]

    local_correct_t1 = 0
    local_correct_t5 = 0
    local_total = 0
    local_dpu_time = 0.0

    while True:
        item = input_queue.get()
        if item is None:
            input_queue.task_done()
            break

        img_int8, class_idx = item
        target = int(class_idx)

        t_start = time.perf_counter()
        jid = runner.execute_async([img_int8], output_data)
        runner.wait(jid)
        local_dpu_time += time.perf_counter() - t_start

        # Top-1/Top-5 via argpartition (O(N) vs argsort's O(N log N)).
        logits = output_data[0][0]
        top5_idx = np.argpartition(logits, -5)[-5:]
        top1 = top5_idx[np.argmax(logits[top5_idx])]

        if target == top1:
            local_correct_t1 += 1
        if target in top5_idx:
            local_correct_t5 += 1

        local_total += 1
        progress.increment()
        input_queue.task_done()

    results[thread_id] = (local_correct_t1, local_correct_t5, local_total, local_dpu_time)
    del runner


# =============================================================
# MAIN
# =============================================================
def run_inference(model_id, dataset_id, thread_override):
    m_cfg = get_active_model(model_id)
    d_cfg = get_active_dataset(dataset_id)

    num_consumers = thread_override if thread_override else ACTIVE_THREADS
    num_producers = 4

    model_path = f"{model_id}_kria.xmodel"
    dataset_path = os.path.join("datasets", d_cfg['folder_name'], "train_data")

    print(f"\n[INFO] Starting Pipeline")
    print(f"       Model:    {m_cfg['name']}")
    print(f"       Dataset:  {d_cfg['name']}")
    print(f"       Threads:  {num_consumers} consumers, {num_producers} producers")

    try:
        subgraph, dpu_shape, fix_pos_in, _ = setup_dpu(model_path)
    except Exception as e:
        print(f"[ERROR] Failed to load model {model_path}: {e}")
        return

    all_images = []
    for c_idx, c_name in enumerate(d_cfg['classes']):
        c_dir = os.path.join(dataset_path, c_name)
        if not os.path.isdir(c_dir):
            continue
        for f in os.listdir(c_dir):
            if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                all_images.append((os.path.join(c_dir, f), c_idx))

    if not all_images:
        print(f"[ERROR] No images found in {dataset_path}")
        return

    img_queue = queue.Queue(maxsize=50)
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

        c_threads = []
        for i in range(num_consumers):
            t = threading.Thread(target=consumer_worker,
                                 args=(i, img_queue, subgraph, progress, results))
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

        print(f"[INFO] DPU Processing started...")
        while any(t.is_alive() for t in c_threads):
            sys.stdout.write(f"\r[INFO] Progress: {progress.value}/{total_imgs} "
                             f"({(progress.value/total_imgs)*100:.1f}%) ")
            sys.stdout.flush()
            time.sleep(0.5)

        sys.stdout.write(f"\r[INFO] Progress: {total_imgs}/{total_imgs} (100.0%) Done!\n")
        end_wall = time.time()
    finally:
        monitor.stop()

    # ---- Report ----
    total_wall_time = end_wall - start_wall
    total_t1 = sum(r[0] for r in results if r)
    total_t5 = sum(r[1] for r in results if r)
    total_imgs_done = sum(r[2] for r in results if r)
    total_dpu_busy = sum(r[3] for r in results if r)

    fps_app = total_imgs_done / total_wall_time if total_wall_time > 0 else 0.0
    avg_dpu_latency = total_dpu_busy / total_imgs_done if total_imgs_done > 0 else 0.0

    avg_load_pwr = monitor.average(fallback=idle_p)
    energy_per_frame = (avg_load_pwr / fps_app) * 1000 if fps_app > 0 else 0.0
    duty_cycle = (total_dpu_busy / (total_wall_time * num_consumers)) * 100 if total_wall_time > 0 else 0.0
    compute_eff = (fps_app * m_cfg['gops'] / DPU_PEAK_GOPS) * 100

    report = format_report(
        f"ANALYTICAL REPORT: {m_cfg['name'].upper()} | DPU THREADS: {num_consumers}",
        [
            ("Images Processed:", f"{total_imgs_done}"),
            ("Top-1 Accuracy:", f"{(total_t1/total_imgs_done)*100:.2f} %"),
            ("Top-5 Accuracy:", f"{(total_t5/total_imgs_done)*100:.2f} %"),
            ("---", None),
            ("Application FPS:", f"{fps_app:.2f} img/s"),
            ("DPU Latency (avg):", f"{avg_dpu_latency*1000:.2f} ms"),
            ("---", None),
            ("Power (Load):", f"{avg_load_pwr:.2f} W"),
            ("Energy per frame:", f"{energy_per_frame:.2f} mJ/img"),
            ("---", None),
            ("DPU Duty Cycle:", f"{min(duty_cycle, 100.0):.2f} %"),
            ("DPU Compute Eff.:", f"{compute_eff:.2f} %"),
        ],
    )
    print("\n" + report)

    with open(f"results_{model_id}_t{num_consumers}.txt", "w") as f:
        f.write(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Model ID')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset ID')
    parser.add_argument('--threads', type=int, help='Override DPU thread count')
    args = parser.parse_args()

    run_inference(args.model, args.dataset, args.threads)
