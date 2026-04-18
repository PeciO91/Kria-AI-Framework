import numpy as np
import cv2
import vart
import xir
import os
import time
import threading
import queue
import argparse
import sys

from model_config import get_active_model
from dataset_config import get_active_dataset
from board_config import ACTIVE_THREADS, DPU_PEAK_GOPS, get_power_mw

# Progress Tracking Globals
progress_cnt = 0
progress_lock = threading.Lock()

# =============================================================
# POWER MONITORING
# =============================================================
class PowerMonitor(threading.Thread):
    def __init__(self, interval=0.2):
        super().__init__()
        self.interval = interval
        self.samples = []
        self.stop_evt = threading.Event()
        self.daemon = True 
        
    def run(self):
        while not self.stop_evt.is_set():
            p = get_power_mw() / 1000.0
            if p > 0: self.samples.append(p)
            time.sleep(self.interval)

# =============================================================
# PRODUCER: CPU Preprocessing (Optimized Math)
# =============================================================
def producer_worker(image_chunk, input_queue, dpu_shape, norm_mean, norm_std, fix_pos):
    dpu_height, dpu_width = dpu_shape[1], dpu_shape[2]
    
    mean_np = np.array(norm_mean, dtype=np.float32)
    std_np = np.array(norm_std, dtype=np.float32)
    f_scale = np.float32(2 ** fix_pos)
    
    math_scale = np.float32(f_scale / (255.0 * std_np))
    math_shift = np.float32((mean_np * f_scale) / std_np)

    for img_path, class_idx in image_chunk:
        img = cv2.imread(img_path)
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (dpu_width, dpu_height))
        
        img_int8 = (img.astype(np.float32) * math_scale - math_shift).astype(np.int8)
        img_int8 = np.expand_dims(img_int8, axis=0)
        
        input_queue.put((img_int8, class_idx))

# =============================================================
# CONSUMER: DPU Inference
# =============================================================
def consumer_worker(thread_id, input_queue, dpu_subgraph, results):
    global progress_cnt
    runner = vart.Runner.create_runner(dpu_subgraph, "run")
    output_tensors = runner.get_output_tensors()
    output_ndim = tuple(output_tensors[0].dims)
    
    local_correct = 0
    local_total = 0
    local_dpu_time = 0
    output_data = [np.empty(output_ndim, dtype=np.int8)]

    while True:
        item = input_queue.get()
        if item is None:
            input_queue.task_done()
            break
            
        img_int8, class_idx = item
        input_data = [img_int8]

        t_start = time.perf_counter()
        jid = runner.execute_async(input_data, output_data)
        runner.wait(jid)
        local_dpu_time += (time.perf_counter() - t_start)

        if np.argmax(output_data[0][0]) == int(class_idx):
            local_correct += 1
        local_total += 1
        
        # Update progress counter safely
        with progress_lock:
            progress_cnt += 1
            
        input_queue.task_done()

    results[thread_id] = (local_correct, local_total, local_dpu_time)
    del runner

# =============================================================
# MAIN LOGIC
# =============================================================
def run_inference(model_id, dataset_id, thread_override):
    global progress_cnt
    # Load requested configurations
    m_cfg = get_active_model(model_id)
    d_cfg = get_active_dataset(dataset_id)
    
    # Set thread counts
    num_consumers = thread_override if thread_override else ACTIVE_THREADS
    num_producers = 4 
    
    model_name_lower = m_cfg['name'].lower()
    model_path = f"{model_name_lower}_kria.xmodel"
    dataset_path = os.path.join("datasets", d_cfg['folder_name'], "train_data")
    
    print(f"\n[INFO] Starting Pipeline")
    print(f"       Model:    {m_cfg['name']}")
    print(f"       Dataset:  {d_cfg['name']}")
    print(f"       Threads:  {num_consumers} Consumers, {num_producers} Producers")
    
    try:
        graph = xir.Graph.deserialize(model_path)
    except Exception:
        print(f"Error: Model {model_path} not found.")
        return

    subgraph = [s for s in graph.get_root_subgraph().get_children() 
                if s.get_attr("device").upper() == "DPU"][0]

    dummy_runner = vart.Runner.create_runner(subgraph, "run")
    input_tensors = dummy_runner.get_input_tensors()
    dpu_shape = tuple(input_tensors[0].dims)
    fix_pos = input_tensors[0].get_attr("fix_point")
    del dummy_runner

    all_images = []
    for c_idx, c_name in enumerate(d_cfg['classes']):
        c_dir = os.path.join(dataset_path, c_name)
        if not os.path.isdir(c_dir): continue
        for f in os.listdir(c_dir):
            if f.lower().endswith(('.jpg','.png')):
                all_images.append((os.path.join(c_dir, f), c_idx))

    if not all_images:
        print(f"Error: No images found in {dataset_path}")
        return

    img_queue = queue.Queue(maxsize=50) 
    results = [None] * num_consumers
    total_imgs = len(all_images)
    
    chunk_size = (total_imgs + num_producers - 1) // num_producers
    chunks = [all_images[i:i + chunk_size] for i in range(0, total_imgs, chunk_size)]

    monitor = PowerMonitor()
    monitor.start()
    idle_p = np.mean([get_power_mw() / 1000.0 for _ in range(5)])
    
    start_wall = time.time()

    # Launch Consumer Threads
    c_threads = []
    for i in range(num_consumers):
        t = threading.Thread(target=consumer_worker, args=(i, img_queue, subgraph, results))
        t.start()
        c_threads.append(t)

    # Launch Producer Threads
    p_threads = []
    for i in range(num_producers):
        t = threading.Thread(target=producer_worker, args=(
            chunks[i], img_queue, dpu_shape, 
            d_cfg['normalization']['mean'], d_cfg['normalization']['std'], fix_pos
        ))
        t.start()
        p_threads.append(t)

    # 1. Wait for Producers to finish filling the queue while showing progress
    for t in p_threads:
        while t.is_alive():
            with progress_lock:
                curr = progress_cnt
            percent = (curr / total_imgs) * 100
            sys.stdout.write(f"\r[INFO] Progress: {curr}/{total_imgs} ({percent:.1f}%) ")
            sys.stdout.flush()
            t.join(0.1)

    # 2. All images preprocessed. Send "None" stop signals to Consumers
    for _ in range(num_consumers):
        img_queue.put(None)

    # 3. Wait for Consumers to finish remaining tasks while showing progress
    for t in c_threads:
        while t.is_alive():
            with progress_lock:
                curr = progress_cnt
            percent = (curr / total_imgs) * 100
            sys.stdout.write(f"\r[INFO] Progress: {curr}/{total_imgs} ({percent:.1f}%) ")
            sys.stdout.flush()
            t.join(0.1)

    sys.stdout.write(f"\r[INFO] Progress: {total_imgs}/{total_imgs} (100.0%) Done!\n")

    end_wall = time.time()
    monitor.stop_evt.set()
    
    # REPORT GENERATION
    total_wall_time = end_wall - start_wall
    total_correct = sum(r[0] for r in results)
    total_images_processed = sum(r[1] for r in results)
    total_dpu_busy_time = sum(r[2] for r in results)

    fps_app = total_images_processed / total_wall_time
    avg_dpu_latency = total_dpu_busy_time / total_images_processed
    fps_dpu_theoretical = 1.0 / avg_dpu_latency

    avg_load_pwr = np.mean(monitor.samples) if monitor.samples else idle_p
    energy_per_frame = (avg_load_pwr / fps_app) * 1000
    duty_cycle = (total_dpu_busy_time / (total_wall_time * num_consumers)) * 100
    compute_eff = (fps_app * m_cfg['gops'] / DPU_PEAK_GOPS) * 100

    report = f"""
{"="*60}
  ANALYTICAL REPORT: {m_cfg['name'].upper()} | DPU THREADS: {num_consumers}
{"="*60}
System:             {total_images_processed} images
Overall Accuracy:   {(total_correct/total_images_processed)*100:.2f} %
{"-"*60}
Application FPS:    {fps_app:.2f} img/s
DPU Latency (avg):  {avg_dpu_latency*1000:.2f} ms
{"-"*60}
Power (Load):       {avg_load_pwr:.2f} W
Energy per frame:   {energy_per_frame:.2f} mJ/img
{"-"*60}
DPU Duty Cycle:     {min(duty_cycle, 100.0):.2f} %
DPU Compute Eff.:   {compute_eff:.2f} %
{"="*60}
"""
    print(report)
    
    with open(f"results_{model_name_lower}_t{num_consumers}.txt", "w") as f:
        f.write(report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Model ID from model_config.py')
    parser.add_argument('--dataset', type=str, help='Dataset ID from dataset_config.py')
    parser.add_argument('--threads', type=int, help='Override DPU thread count')
    args = parser.parse_args()

    run_inference(args.model, args.dataset, args.threads)