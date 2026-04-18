# --- START OF FILE run_inference.py ---

import numpy as np
import cv2
import vart
import xir
import os
import time
import threading
import queue

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
        self.daemon = True # Ensures thread dies if main script crashes
        
    def run(self):
        while not self.stop_evt.is_set():
            p = get_power_mw() / 1000.0  # Convert mW to W
            if p > 0: self.samples.append(p)
            time.sleep(self.interval)

# =============================================================
# PRODUCER: CPU Preprocessing (OPTIMIZED MATH)
# =============================================================
def producer_worker(image_chunk, input_queue, dpu_shape, norm_mean, norm_std, fix_pos):
    """Reads, resizes, normalizes (with optimized math), and pushes to queue."""
    dpu_height, dpu_width = dpu_shape[1], dpu_shape[2]
    
    # ---------------------------------------------------------
    # PRE-CALCULATE CONSTANTS ONCE PER THREAD
    # ---------------------------------------------------------
    mean_np = np.array(norm_mean, dtype=np.float32)
    std_np = np.array(norm_std, dtype=np.float32)
    f_scale = np.float32(2 ** fix_pos)
    
    # Algebraically simplified normalization and quantization
    # scale = (2^fix_pos) / (255.0 * std)
    # shift = (mean * 2^fix_pos) / std
    math_scale = np.float32(f_scale / (255.0 * std_np))
    math_shift = np.float32((mean_np * f_scale) / std_np)
    # ---------------------------------------------------------

    for img_path, class_idx in image_chunk:
        img = cv2.imread(img_path)
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # You can also add interpolation=cv2.INTER_NEAREST here later if you want even more speed
        img = cv2.resize(img, (dpu_width, dpu_height))
        
        # ---------------------------------------------------------
        # OPTIMIZED PIXEL MATH (Vectorized via NumPy broadcasting)
        # ---------------------------------------------------------
        # 1. Cast to float32
        # 2. Multiply by pre-calculated scale
        # 3. Subtract pre-calculated shift
        # 4. Cast directly to int8
        img_int8 = (img.astype(np.float32) * math_scale - math_shift).astype(np.int8)
        
        # Wrap in expected DPU shape: (1, H, W, C)
        img_int8 = np.expand_dims(img_int8, axis=0)
        
        # Push to queue
        input_queue.put((img_int8, class_idx))

# =============================================================
# CONSUMER: DPU Inference
# =============================================================
def consumer_worker(thread_id, input_queue, dpu_subgraph, results):
    """Pulls preprocessed arrays from the queue and feeds the DPU."""
    runner = vart.Runner.create_runner(dpu_subgraph, "run")
    output_tensors = runner.get_output_tensors()
    output_ndim = tuple(output_tensors[0].dims)
    
    local_correct = 0
    local_total = 0
    local_dpu_time = 0

    # Pre-allocate output buffer for this thread
    output_data = [np.empty(output_ndim, dtype=np.int8)]

    while True:
        # Pull from queue
        item = input_queue.get()
        
        # Check for the kill signal (None)
        if item is None:
            input_queue.task_done()
            break
            
        img_int8, class_idx = item
        input_data = [img_int8]

        # Execute on DPU
        t_start = time.perf_counter()
        jid = runner.execute_async(input_data, output_data)
        runner.wait(jid)
        local_dpu_time += (time.perf_counter() - t_start)

        # Postprocess (Argmax is fast enough to do here)
        if np.argmax(output_data[0][0]) == int(class_idx):
            local_correct += 1
        local_total += 1
        
        input_queue.task_done()

    results[thread_id] = (local_correct, local_total, local_dpu_time)
    del runner  # Clean up VART memory

# =============================================================
# MAIN LOGIC
# =============================================================
def run_inference():
    m_cfg = get_active_model()
    d_cfg = get_active_dataset()
    
    model_name = m_cfg['name'].lower()
    model_path = f"{model_name}_kria.xmodel"
    dataset_path = os.path.join("datasets", d_cfg['folder_name'], "train_data")
    
    print(f"Preparing Producer-Consumer pipeline for: {m_cfg['name']}")
    
    try:
        graph = xir.Graph.deserialize(model_path)
    except Exception as e:
        print(f"Error: Model {model_path} not found.")
        return

    subgraph = [s for s in graph.get_root_subgraph().get_children() 
                if s.get_attr("device").upper() == "DPU"][0]

    # Extract DPU requirements to pass to Producers
    dummy_runner = vart.Runner.create_runner(subgraph, "run")
    input_tensors = dummy_runner.get_input_tensors()
    dpu_shape = tuple(input_tensors[0].dims)
    fix_pos = input_tensors[0].get_attr("fix_point")
    del dummy_runner

    # Gather images
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

    print("Measuring idle power consumption...")
    idle_p = np.mean([get_power_mw() / 1000.0 for _ in range(5)])

    # --- PIPELINE SETUP ---
    # Maxsize prevents the CPU from eating all RAM if it's faster than the DPU
    img_queue = queue.Queue(maxsize=50) 
    
    NUM_PRODUCERS = 4  # 4 CPU threads dedicated to OpenCV/Numpy
    NUM_CONSUMERS = ACTIVE_THREADS # 4 DPU threads
    
    results = [None] * NUM_CONSUMERS
    producer_threads = []
    consumer_threads = []

    # Split data for producers
    chunk_size = (len(all_images) + NUM_PRODUCERS - 1) // NUM_PRODUCERS
    chunks = [all_images[i:i + chunk_size] for i in range(0, len(all_images), chunk_size)]

    print(f"Starting benchmark... ({NUM_PRODUCERS} Producers, {NUM_CONSUMERS} Consumers)")
    monitor = PowerMonitor()
    monitor.start()
    
    start_wall = time.time()

    # 1. Start Consumers (They will block, waiting for data in the queue)
    for i in range(NUM_CONSUMERS):
        t = threading.Thread(target=consumer_worker, args=(i, img_queue, subgraph, results))
        t.start()
        consumer_threads.append(t)

    # 2. Start Producers (They will start filling the queue)
    for i in range(NUM_PRODUCERS):
        t = threading.Thread(target=producer_worker, args=(
            chunks[i], img_queue, dpu_shape, 
            d_cfg['normalization']['mean'], d_cfg['normalization']['std'], fix_pos
        ))
        t.start()
        producer_threads.append(t)

    # 3. Wait for all Producers to finish processing images
    for t in producer_threads:
        t.join()

    # 4. Send "Kill Signals" to Consumers (one for each consumer)
    for _ in range(NUM_CONSUMERS):
        img_queue.put(None)

    # 5. Wait for Consumers to finish their final inferences
    for t in consumer_threads:
        t.join()

    end_wall = time.time()
    monitor.stop_evt.set()
    
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

    duty_cycle = (total_dpu_busy_time / (total_wall_time * ACTIVE_THREADS)) * 100
    compute_eff = (fps_app * m_cfg['gops'] / DPU_PEAK_GOPS) * 100

    report_text = f"""
{"="*60}
  ANALYTICAL REPORT: {m_cfg['name'].upper()} | DPU THREADS: {ACTIVE_THREADS}
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

    filename = f"results_{model_name}_t{ACTIVE_THREADS}.txt"
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(report_text)
    except Exception as e:
        print(f"Error writing to file: {e}")

if __name__ == "__main__":
    run_inference()