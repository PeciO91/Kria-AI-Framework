import numpy as np
import cv2
import vart
import xir
import os
import time
import threading
import subprocess
import re

# =============================================================
# KONFIGURACE (Tady můžeš měnit počet vláken)
# =============================================================
NUM_THREADS = 4  # Pro KV260 jsou 3-4 vlákna obvykle ideální strop
MODEL_PATH = "output_kria/resnet18_intel_kria.xmodel"
DATASET_PATH = "seg_test/seg_test"
CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

# Teoretické parametry pro výpočty
DPU_PEAK_GOPS = 2457.0  
MODEL_GOPS_PER_IMG = 3.64 

# =============================================================
# MONITOROVÁNÍ SPOTŘEBY (Vzorkování 5x za sekundu)
# =============================================================
def get_current_power():
    try:
        out = subprocess.check_output(["xmutil", "xlnx_platformstats", "-p"], 
                                      stderr=subprocess.STDOUT, encoding='utf-8')
        match = re.search(r"SOM total power\s+:\s+(\d+)\s+mW", out)
        if match: return float(match.group(1)) / 1000.0
        return 0.0
    except: return 0.0

class PowerMonitor(threading.Thread):
    def __init__(self, interval=0.2):
        super().__init__()
        self.interval = interval
        self.samples = []
        self.stop_evt = threading.Event()
    def run(self):
        while not self.stop_evt.is_set():
            p = get_current_power()
            if p > 0: self.samples.append(p)
            time.sleep(self.interval)

# =============================================================
# WORKER FUNKCE PRO VLÁKNA
# =============================================================
def worker(thread_id, image_chunk, dpu_subgraph, results):
    """Každé vlákno si vytvoří vlastní runner a zpracuje svůj kus dat."""
    runner = vart.Runner.create_runner(dpu_subgraph, "run")
    
    input_tensors = runner.get_input_tensors()
    output_tensors = runner.get_output_tensors()
    input_ndim = tuple(input_tensors[0].dims)
    output_ndim = tuple(output_tensors[0].dims)
    fix_pos = input_tensors[0].get_attr("fix_point")

    local_correct = 0
    local_total = 0
    local_dpu_time = 0

    # Příprava bufferů pro toto vlákno
    input_data = [np.empty(input_ndim, dtype=np.int8)]
    output_data = [np.empty(output_ndim, dtype=np.int8)]

    for img_path, class_idx in image_chunk:
        # 1. Preprocess (CPU)
        img = cv2.imread(img_path)
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (input_ndim[2], input_ndim[1]))
        img = (img.astype(np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        input_data[0][0,...] = (img * (2**fix_pos)).astype(np.int8)

        # 2. Inference (DPU)
        t_start = time.perf_counter()
        jid = runner.execute_async(input_data, output_data)
        runner.wait(jid)
        local_dpu_time += (time.perf_counter() - t_start)

        # 3. Postprocess (CPU)
        if np.argmax(output_data[0][0]) == class_idx:
            local_correct += 1
        local_total += 1

    results[thread_id] = (local_correct, local_total, local_dpu_time)
    del runner # Vyčištění paměti

# =============================================================
# HLAVNÍ LOGIKA
# =============================================================
print(f"Příprava dat a DPU (Vláken: {NUM_THREADS})...")
graph = xir.Graph.deserialize(MODEL_PATH)
subgraph = [s for s in graph.get_root_subgraph().get_children() 
            if s.get_attr("device").upper() == "DPU"][0]

# Shromáždění všech cest k obrázkům
all_images = []
for c_idx, c_name in enumerate(CLASSES):
    c_dir = os.path.join(DATASET_PATH, c_name)
    if not os.path.isdir(c_dir): continue
    for f in os.listdir(c_dir):
        if f.lower().endswith(('.jpg','.png')):
            all_images.append((os.path.join(c_dir, f), c_idx))

# Rozdělení dat pro vlákna
chunks = np.array_split(all_images, NUM_THREADS)

print("Měřím klidovou spotřebu (Idle)...")
idle_p = np.mean([get_current_power() for _ in range(5)])

print("Startuji benchmark...")
monitor = PowerMonitor()
monitor.start()

start_wall = time.time()
threads = []
results = [None] * NUM_THREADS

for i in range(NUM_THREADS):
    t = threading.Thread(target=worker, args=(i, chunks[i], subgraph, results))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

end_wall = time.time()
monitor.stop_evt.set()
monitor.join()

# =============================================================
# AGREGACE A VÝPOČET REPROTU
# =============================================================
total_wall_time = end_wall - start_wall
total_correct = sum(r[0] for r in results)
total_images = sum(r[1] for r in results)
total_dpu_busy_time = sum(r[2] for r in results)

fps_app = total_images / total_wall_time
avg_dpu_latency = total_dpu_busy_time / total_images
fps_dpu_theoretical = 1.0 / avg_dpu_latency

avg_load_pwr = np.mean(monitor.samples) if monitor.samples else 0
energy_per_frame = (avg_load_pwr / fps_app) * 1000

# DPU Efficiency
duty_cycle = (total_dpu_busy_time / total_wall_time) * 100
compute_eff = (fps_app * MODEL_GOPS_PER_IMG / DPU_PEAK_GOPS) * 100

# =============================================================
# FINÁLNÍ REPORT (Identický formát jako předtím)
# =============================================================
print("\n" + "="*60)
print(f"  ANALYTICKÝ REPORT: RESNET18 | VLÁKEN: {NUM_THREADS}")
print("="*60)
print(f"Dataset:            {total_images} obrázků")
print(f"Celková přesnost:   {(total_correct/total_images)*100:.2f} %")
print("-"*60)
print(f"Aplikace FPS:       {fps_app:.2f} obr./s (Celý systém)")
print(f"Teoretické DPU FPS: {fps_dpu_theoretical:.2f} obr./s (Pouze výpočet)")
print(f"Latence DPU (avg):  {avg_dpu_latency*1000:.2f} ms")
print("-"*60)
print(f"Spotřeba (Idle):    {idle_p:.2f} W")
print(f"Spotřeba (Load):    {avg_load_pwr:.2f} W")
print(f"Energie na snímek:  {energy_per_frame:.2f} mJ/obr.")
print("-"*60)
print(f"DPU Duty Cycle:     {min(duty_cycle, 100.0):.2f} %  (Časové vytížení)")
print(f"DPU Compute Eff.:   {compute_eff:.2f} %  (Využití MAC při výpočtu)")
print("="*60)