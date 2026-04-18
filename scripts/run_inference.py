import numpy as np
import cv2
import vart
import xir
import os
import time
import threading

from model_config import get_active_model
from dataset_config import get_active_dataset
from board_config import ACTIVE_THREADS, DPU_PEAK_GOPS, get_power_mw

# =============================================================
# POWER MONITORING
# =============================================================
class PowerMonitor(threading.Thread):
    def __init__(self, interval=0.2):
        super().__init__()
        self.interval = interval
        self.samples = []
        self.stop_evt = threading.Event()
        
    def run(self):
        while not self.stop_evt.is_set():
            p = get_power_mw() / 1000.0  # Convert mW to W
            if p > 0: self.samples.append(p)
            time.sleep(self.interval)

# =============================================================
# THREAD WORKER FUNCTION
# =============================================================
def worker(thread_id, image_chunk, dpu_subgraph, results, norm_mean, norm_std):
    """Each thread creates its own runner and processes its chunk of data."""
    runner = vart.Runner.create_runner(dpu_subgraph, "run")
    
    input_tensors = runner.get_input_tensors()
    output_tensors = runner.get_output_tensors()
    input_ndim = tuple(input_tensors[0].dims) # Expected NHWC (Batch, Height, Width, Channels)
    output_ndim = tuple(output_tensors[0].dims)
    fix_pos = input_tensors[0].get_attr("fix_point")

    local_correct = 0
    local_total = 0
    local_dpu_time = 0

    # Prepare buffers for this specific thread
    input_data = [np.empty(input_ndim, dtype=np.int8)]
    output_data = [np.empty(output_ndim, dtype=np.int8)]

    for img_path, class_idx in image_chunk:
        # 1. Preprocess (CPU)
        img = cv2.imread(img_path)
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Override any config - strictly follow what DPU hardware demands
        dpu_width = input_ndim[2]
        dpu_height = input_ndim[1]
        img = cv2.resize(img, (dpu_width, dpu_height))
        
        # Normalization
        img = (img.astype(np.float32) / 255.0 - norm_mean) / norm_std
        input_data[0][0,...] = (img * (2**fix_pos)).astype(np.int8)

        # 2. Inference (DPU)
        t_start = time.perf_counter()
        jid = runner.execute_async(input_data, output_data)
        runner.wait(jid)
        local_dpu_time += (time.perf_counter() - t_start)

        # 3. Postprocess (CPU) - Classification
        # We explicitly cast class_idx to int to prevent string vs int comparison bugs
        if np.argmax(output_data[0][0]) == int(class_idx):
            local_correct += 1
        local_total += 1

    results[thread_id] = (local_correct, local_total, local_dpu_time)
    del runner  # Clean up VART memory

# =============================================================
# MAIN LOGIC
# =============================================================
def run_inference():
    # Load configurations
    m_cfg = get_active_model()
    d_cfg = get_active_dataset()
    
    # Expected model path in the same directory
    model_name = m_cfg['name'].lower()
    model_path = f"{model_name}_kria.xmodel"
    
    # Standardized dataset path: datasets/<folder_name>/train_data
    dataset_path = os.path.join("datasets", d_cfg['folder_name'], "train_data")
    
    print(f"Preparing data and DPU for model: {m_cfg['name']} (Threads: {ACTIVE_THREADS})...")
    print(f"Target dataset path: {dataset_path}")
    
    try:
        graph = xir.Graph.deserialize(model_path)
    except Exception as e:
        print(f"Error: Model {model_path} not found. Did you copy it to the Kria board?")
        return

    subgraph = [s for s in graph.get_root_subgraph().get_children() 
                if s.get_attr("device").upper() == "DPU"][0]

    # Gather all image paths and their respective labels
    all_images = []
    classes = d_cfg['classes']
    
    for c_idx, c_name in enumerate(classes):
        c_dir = os.path.join(dataset_path, c_name)
        if not os.path.isdir(c_dir): continue
        for f in os.listdir(c_dir):
            if f.lower().endswith(('.jpg','.png')):
                all_images.append((os.path.join(c_dir, f), c_idx))

    if not all_images:
        print(f"Error: No images found in {dataset_path} directory.")
        return

    # Split data equally among threads using pure Python to preserve data types (int stays int)
    chunk_size = (len(all_images) + ACTIVE_THREADS - 1) // ACTIVE_THREADS
    chunks = [all_images[i:i + chunk_size] for i in range(0, len(all_images), chunk_size)]

    print("Measuring idle power consumption...")
    idle_p = np.mean([get_power_mw() / 1000.0 for _ in range(5)])

    print("Starting benchmark...")
    monitor = PowerMonitor()
    monitor.start()

    start_wall = time.time()
    threads = []
    results = [None] * ACTIVE_THREADS

    for i in range(ACTIVE_THREADS):
        # We pass only the chunk to the worker, it pulls DPU dims from the model itself
        t = threading.Thread(target=worker, args=(
            i, chunks[i], subgraph, results, 
            d_cfg['normalization']['mean'], d_cfg['normalization']['std']
        ))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    end_wall = time.time()
    monitor.stop_evt.set()
    monitor.join()

    # =============================================================
    # AGGREGATION AND REPORT CALCULATION
    # =============================================================
    total_wall_time = end_wall - start_wall
    total_correct = sum(r[0] for r in results)
    total_images = sum(r[1] for r in results)
    total_dpu_busy_time = sum(r[2] for r in results)

    fps_app = total_images / total_wall_time
    avg_dpu_latency = total_dpu_busy_time / total_images
    fps_dpu_theoretical = 1.0 / avg_dpu_latency

    avg_load_pwr = np.mean(monitor.samples) if monitor.samples else idle_p
    energy_per_frame = (avg_load_pwr / fps_app) * 1000

    # DPU Efficiency Metrics
    duty_cycle = (total_dpu_busy_time / (total_wall_time * ACTIVE_THREADS)) * 100
    compute_eff = (fps_app * m_cfg['gops'] / DPU_PEAK_GOPS) * 100

    # Build the report string
    report_text = f"""
{"="*60}
  ANALYTICAL REPORT: {m_cfg['name'].upper()} | THREADS: {ACTIVE_THREADS}
{"="*60}
Date and Time:      {time.strftime("%Y-%m-%d %H:%M:%S")}
Model Path:         {model_path}
Dataset:            {total_images} images
Overall Accuracy:   {(total_correct/total_images)*100:.2f} %
{"-"*60}
Application FPS:    {fps_app:.2f} img/s (System total)
Theoretical DPU FPS:{fps_dpu_theoretical:.2f} img/s (Compute only)
DPU Latency (avg):  {avg_dpu_latency*1000:.2f} ms
{"-"*60}
Power (Idle):       {idle_p:.2f} W
Power (Load):       {avg_load_pwr:.2f} W
Energy per frame:   {energy_per_frame:.2f} mJ/img
{"-"*60}
DPU Duty Cycle:     {min(duty_cycle, 100.0):.2f} %  (Time utilization of cores)
DPU Compute Eff.:   {compute_eff:.2f} %  (MAC utilization during compute)
{"="*60}
"""

    print(report_text)

    # Write results to file
    filename = f"results_{model_name}_t{ACTIVE_THREADS}.txt"
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"Report successfully saved to file: {filename}")
    except Exception as e:
        print(f"Error writing to file: {e}")

if __name__ == "__main__":
    run_inference()