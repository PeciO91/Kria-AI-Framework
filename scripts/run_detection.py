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
from detection_utils import letterbox, scale_coords, non_max_suppression

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

# =============================================================
# PRODUCER: Letterbox Preprocessing
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
        
        orig_shape = orig_img.shape[:2]
        img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        
        # USE LETTERBOX INSTEAD OF RESIZE
        img_resized, _, _ = letterbox(img_rgb, new_shape=(dpu_height, dpu_width))
        
        img_int8 = (img_resized.astype(np.float32) * math_scale - math_shift).astype(np.int8)
        img_int8 = np.expand_dims(img_int8, axis=0)
        
        # Pass the original image and shape so we can draw on it later
        input_queue.put((img_int8, orig_img, orig_shape, os.path.basename(img_path)))
    return boxes, scores, class_ids

# =============================================================
# CONSUMER: DPU Inference & NMS
# =============================================================
def consumer_worker(thread_id, input_queue, dpu_subgraph, out_dir, m_cfg, fix_pos_outs):
    global progress_cnt
    runner = vart.Runner.create_runner(dpu_subgraph, "run")
    output_tensors = runner.get_output_tensors()
    
    local_total = 0
    local_dpu_time = 0
    
    # CRITICAL FIX: Create empty arrays for ALL 3 output tensors
    output_data = [np.empty(tuple(t.dims), dtype=np.int8) for t in output_tensors]
    
    conf_thresh = m_cfg.get('conf_threshold', 0.25)
    iou_thresh = m_cfg.get('iou_threshold', 0.45)
    dpu_shape = tuple(runner.get_input_tensors()[0].dims)[1:3] # H, W

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

        # 2. DEQUANTIZE ALL 3 TENSORS
        # Convert INT8 back to Float32 using the unique fix_point for each tensor
        float_outs = [np.array(out) * (2 ** -fix_pos) for out, fix_pos in zip(output_data, fix_pos_outs)]

        # 3. DECODE & FILTER (Pass the list of 3 tensors)
        boxes, scores, class_ids = decode_yolo_output(float_outs, conf_thresh)
        
        # 4. NON-MAXIMUM SUPPRESSION (NMS)
        if len(boxes) > 0:
            indices = non_max_suppression(boxes, scores, conf_thresh, iou_thresh)
            
            if len(indices) > 0:
                final_boxes = np.array([boxes[i] for i in indices])
                
                # Convert xywh back to xyxy for scaling and drawing
                xyxy_boxes = np.zeros_like(final_boxes)
                xyxy_boxes[:, 0] = final_boxes[:, 0]
                xyxy_boxes[:, 1] = final_boxes[:, 1]
                xyxy_boxes[:, 2] = final_boxes[:, 0] + final_boxes[:, 2]
                xyxy_boxes[:, 3] = final_boxes[:, 1] + final_boxes[:, 3]
                
                # 5. SCALE COORDS BACK TO ORIGINAL IMAGE
                scaled_boxes = scale_coords(dpu_shape, xyxy_boxes, orig_shape)
                
                # 6. DRAW BOXES
                for i, box in enumerate(scaled_boxes):
                    x1, y1, x2, y2 = map(int, box[:4])
                    cls_id = class_ids[indices[i]]
                    conf = scores[indices[i]]
                    
                    cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"Class {cls_id}: {conf:.2f}"
                    cv2.putText(orig_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save output image
        cv2.imwrite(os.path.join(out_dir, file_name), orig_img)

        local_total += 1
        with progress_lock:
            progress_cnt += 1
        input_queue.task_done()

    results[thread_id] = (local_total, local_dpu_time)
    del runner

def decode_yolo_output(dpu_outputs, conf_threshold):
    """
    Decodes 3 raw tensors [P3, P4, P5] from the DPU.
    P3: 80x80, P4: 40x40, P5: 20x20
    """
    # Standard YOLOv5n anchors
    anchors = [ [[10,13], [16,30], [33,23]], [[30,61], [62,45], [59,119]], [[116,90], [156,198], [373,326]] ]
    strides = [8, 16, 32]
    
    boxes, scores, class_ids = [], [], []

    for i, pred in enumerate(dpu_outputs):
        # pred is a numpy array (1, H, W, 3*(5+classes))
        # Convert to float and de-quantize happens in consumer_worker
        bs, ny, nx, _ = pred.shape
        num_classes = (pred.shape[-1] // 3) - 5
        pred = pred.reshape(1, 3, ny, nx, 5 + num_classes)
        
        # Sigmoid activation (CPU side)
        pred = 1 / (1 + np.exp(-pred)) 
        
        # Build Grid
        grid_y, grid_x = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
        grid = np.stack((grid_x, grid_y), axis=-1).reshape(1, 1, ny, nx, 2)
        
        # Center XY and WH decoding
        xy = (pred[..., 0:2] * 2. - 0.5 + grid) * strides[i]
        wh = (pred[..., 2:4] * 2) ** 2 * np.array(anchors[i]).reshape(1, 3, 1, 1, 2)
        
        # Extract Confidence and Class
        obj_conf = pred[..., 4]
        cls_conf = np.max(pred[..., 5:], axis=-1)
        total_conf = obj_conf * cls_conf
        
        mask = total_conf > conf_threshold
        if mask.any():
            # Apply mask and extract valid boxes
            v_xy = xy[mask]
            v_wh = wh[mask]
            v_conf = total_conf[mask]
            v_cls = np.argmax(pred[..., 5:][mask], axis=-1)
            
            for box, sc, cl in zip(np.concatenate([v_xy, v_wh], axis=-1), v_conf, v_cls):
                boxes.append([box[0]-box[2]/2, box[1]-box[3]/2, box[2], box[3]])
                scores.append(float(sc))
                class_ids.append(int(cl))
                
    return boxes, scores, class_ids

# =============================================================
# MAIN LOGIC
# =============================================================
def run_detection(model_id, dataset_id, thread_override):
    global progress_cnt
    m_cfg = get_active_model(model_id)
    d_cfg = get_active_dataset(dataset_id)
    
    if m_cfg['type'] != 'detection':
        print(f"[ERROR] Model {model_id} is a {m_cfg['type']} model. Use run_inference.py instead.")
        sys.exit(1)
    
    num_consumers = thread_override if thread_override else ACTIVE_THREADS
    num_producers = 2  # Detection images are usually large, 2 is safer for RAM
    
    model_name_lower = m_cfg['name'].lower()
    model_path = f"{model_name_lower}_kria.xmodel"
    dataset_path = "datasets/coco/" # Using the flat folder for demo
    
    out_dir = f"outputs_{model_name_lower}"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"\n[INFO] Starting YOLO Detection Pipeline")
    print(f"       Saving drawn images to: {out_dir}/")
    
    try:
        graph = xir.Graph.deserialize(model_path)
    except Exception:
        print(f"Error: Model {model_path} not found.")
        return

    subgraph = [s for s in graph.get_root_subgraph().get_children() if s.get_attr("device").upper() == "DPU"][0]

    dummy_runner = vart.Runner.create_runner(subgraph, "run")
    in_tensors = dummy_runner.get_input_tensors()
    out_tensors = dummy_runner.get_output_tensors()
    dpu_shape = tuple(in_tensors[0].dims)
    fix_pos_in = in_tensors[0].get_attr("fix_point")
    # CRITICAL FIX: Get fix_points for all 3 output tensors
    fix_pos_outs = [t.get_attr("fix_point") for t in out_tensors] 
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
            t = threading.Thread(target=consumer_worker, args=(i, img_queue, subgraph, out_dir, m_cfg, fix_pos_outs))
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

        print(f"[INFO] DPU Processing & NMS started...")
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

    print(f"\n{'='*60}\n  DETECTION REPORT: {m_cfg['name'].upper()} \n{'='*60}")
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

    run_detection(args.model, args.dataset, args.threads)