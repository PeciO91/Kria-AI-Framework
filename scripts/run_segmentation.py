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
from board_config import ACTIVE_THREADS, get_power_mw

progress_cnt = 0
progress_lock = threading.Lock()

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

# Generate a color map for up to 30 classes (BGR format for OpenCV)
CITYSCAPES_COLORS = np.array([
    [0, 0, 0], [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35],
    [152, 251, 152], [70, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142],
    [0, 0, 70], [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]
], dtype=np.uint8)

# =============================================================
# PRODUCER: Preprocessing
# =============================================================
def producer_worker(image_chunk, input_queue, dpu_shape, norm_mean, norm_std, fix_pos):
    dpu_height, dpu_width = dpu_shape[1], dpu_shape[2]
    
    mean_np = np.array(norm_mean, dtype=np.float32)
    std_np = np.array(norm_std, dtype=np.float32)
    f_scale = np.float32(2 ** fix_pos)
    
    math_scale = np.float32(f_scale / (255.0 * std_np))
    math_shift = np.float32((mean_np * f_scale) / std_np)

    for img_path in image_chunk:
        orig_img = cv2.imread(img_path)
        if orig_img is None: continue
        
        orig_shape = orig_img.shape[:2] # (H, W)
        img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        
        # Standard resize for segmentation
        img_resized = cv2.resize(img_rgb, (dpu_width, dpu_height), interpolation=cv2.INTER_LINEAR)
        
        img_int8 = (img_resized.astype(np.float32) * math_scale - math_shift).astype(np.int8)
        img_int8 = np.expand_dims(img_int8, axis=0)
        
        input_queue.put((img_int8, orig_img, orig_shape, os.path.basename(img_path)))

# =============================================================
# CONSUMER: DPU Inference & Mask Overlay
# =============================================================
def consumer_worker(thread_id, input_queue, dpu_subgraph, out_dir):
    global progress_cnt
    runner = vart.Runner.create_runner(dpu_subgraph, "run")
    output_tensors = runner.get_output_tensors()
    output_ndim = tuple(output_tensors[0].dims)
    
    local_total = 0
    local_dpu_time = 0
    output_data = [np.empty(output_ndim, dtype=np.int8)]

    while True:
        item = input_queue.get()
        if item is None:
            input_queue.task_done()
            break
            
        img_int8, orig_img, orig_shape, file_name = item
        input_data = [img_int8]

        # 1. DPU EXECUTION
        t_start = time.perf_counter()
        jid = runner.execute_async(input_data, output_data)
        runner.wait(jid)
        local_dpu_time += (time.perf_counter() - t_start)

        # 2. GET WINNING CLASS PER PIXEL
        # output_data[0] shape is usually (1, H, W, Classes)
        # We run argmax on the raw INT8 data to save CPU time!
        mask_indices = np.argmax(output_data[0][0], axis=-1)

        # 3. COLORIZE MASK
        # Map the 2D array of class IDs to a 3D array of BGR colors
        colored_mask_dpu_size = CITYSCAPES_COLORS[mask_indices % len(CITYSCAPES_COLORS)]

        # 4. RESIZE MASK BACK TO ORIGINAL IMAGE SIZE
        # Use INTER_NEAREST to avoid interpolating colors into weird in-between shades
        colored_mask_orig_size = cv2.resize(
            colored_mask_dpu_size, 
            (orig_shape[1], orig_shape[0]), 
            interpolation=cv2.INTER_NEAREST
        )

        # 5. BLEND WITH ORIGINAL IMAGE
        # 0.6 = 60% original image, 0.4 = 40% mask opacity
        blended_img = cv2.addWeighted(orig_img, 0.6, colored_mask_orig_size, 0.4, 0)

        # Save output image
        cv2.imwrite(os.path.join(out_dir, file_name), blended_img)

        local_total += 1
        with progress_lock:
            progress_cnt += 1
        input_queue.task_done()

    results[thread_id] = (local_total, local_dpu_time)
    del runner

# =============================================================
# MAIN LOGIC
# =============================================================
def run_segmentation(model_id, dataset_id, thread_override):
    global progress_cnt
    m_cfg = get_active_model(model_id)
    d_cfg = get_active_dataset(dataset_id)
    
    if m_cfg['type'] != 'segmentation':
        print(f"[ERROR] Model {model_id} is a {m_cfg['type']} model. Use run_inference.py or run_detection.py instead.")
        sys.exit(1)
    
    num_consumers = thread_override if thread_override else ACTIVE_THREADS
    num_producers = 2
    
    model_name_lower = m_cfg['name'].lower()
    model_path = f"{model_name_lower}_kria.xmodel"
    dataset_path = d_cfg['calib_path'] 
    
    out_dir = f"outputs_{model_name_lower}"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"\n[INFO] Starting Segmentation Pipeline")
    print(f"       Saving overlay images to: {out_dir}/")
    
    try:
        graph = xir.Graph.deserialize(model_path)
    except Exception:
        print(f"Error: Model {model_path} not found.")
        return

    subgraph = [s for s in graph.get_root_subgraph().get_children() if s.get_attr("device").upper() == "DPU"][0]

    dummy_runner = vart.Runner.create_runner(subgraph, "run")
    in_tensors = dummy_runner.get_input_tensors()
    dpu_shape = tuple(in_tensors[0].dims)
    fix_pos_in = in_tensors[0].get_attr("fix_point")
    del dummy_runner

    all_images = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith(('.jpg','.png'))]
    if not all_images:
        print(f"Error: No images found in {dataset_path}")
        return

    img_queue = queue.Queue(maxsize=20) 
    global results
    results = [None] * num_consumers
    total_imgs = len(all_images)
    
    chunk_size = (total_imgs + num_producers - 1) // num_producers
    chunks = [all_images[i:i + chunk_size] for i in range(0, total_imgs, chunk_size)]

    monitor = PowerMonitor()
    monitor.start()
    
    start_wall = time.time()
    end_wall = time.time()
    
    try:
        idle_p = np.mean([get_power_mw() / 1000.0 for _ in range(5)])
        start_wall = time.time()

        c_threads = []
        for i in range(num_consumers):
            t = threading.Thread(target=consumer_worker, args=(i, img_queue, subgraph, out_dir))
            t.start()
            c_threads.append(t)

        p_threads = []
        for i in range(num_producers):
            t = threading.Thread(target=producer_worker, args=(
                chunks[i], img_queue, dpu_shape, 
                d_cfg['normalization']['mean'], d_cfg['normalization']['std'], fix_pos_in
            ))
            t.start()
            p_threads.append(t)

        for t in p_threads: t.join() 
        for _ in range(num_consumers): img_queue.put(None)

        print(f"[INFO] DPU Processing & Mask Overlay started...")
        while any(t.is_alive() for t in c_threads):
            with progress_lock: curr = progress_cnt
            sys.stdout.write(f"\r[INFO] Progress: {curr}/{total_imgs} ({(curr/total_imgs)*100:.1f}%) ")
            sys.stdout.flush()
            time.sleep(0.5)

        sys.stdout.write(f"\r[INFO] Progress: {total_imgs}/{total_imgs} (100.0%) Done!\n")
        end_wall = time.time()

    finally:
        monitor.stop_evt.set()
        monitor.join(timeout=1.0)
    
    total_wall_time = end_wall - start_wall
    total_images = sum(r[0] for r in results)
    total_dpu_time = sum(r[1] for r in results)

    fps_app = total_images / total_wall_time
    avg_dpu_latency = total_dpu_time / total_images

    print(f"\n{'='*60}\n  SEGMENTATION REPORT: {m_cfg['name'].upper()} \n{'='*60}")
    print(f"Images Processed:   {total_images}")
    print(f"Application FPS:    {fps_app:.2f} img/s")
    print(f"DPU Latency (avg):  {avg_dpu_latency*1000:.2f} ms")
    print(f"Output Images:      Saved to ./{out_dir}/\n{'='*60}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--threads', type=int)
    args = parser.parse_args()

    run_segmentation(args.model, args.dataset, args.threads)