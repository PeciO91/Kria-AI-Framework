import numpy as np
import cv2
import vart
import xir
import os
import time

# --- KONFIGURACE CEST ---
MODEL_PATH = "output_kria/resnet18_intel_kria.xmodel"
# Tady je tvoje specifická cesta
DATASET_PATH = "seg_test/seg_test"
CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

# --- INICIALIZACE DPU ---
graph = xir.Graph.deserialize(MODEL_PATH)
subgraph = [s for s in graph.get_root_subgraph().get_children() if s.get_attr("device").upper() == "DPU"][0]
dpu_runner = vart.Runner.create_runner(subgraph, "run")

input_tensors = dpu_runner.get_input_tensors()
output_tensors = dpu_runner.get_output_tensors()
input_ndim = tuple(input_tensors[0].dims)
output_ndim = tuple(output_tensors[0].dims)
fix_pos = input_tensors[0].get_attr("fix_point")

def preprocess(img_path):
    img = cv2.imread(img_path)
    if img is None: return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_ndim[2], input_ndim[1]))
    img = (img.astype(np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    img = img * (2**fix_pos)
    return img.astype(np.int8)

# --- BENCHMARK ---
print(f"Startuji benchmark. Cesta: {DATASET_PATH}")
total_images = 0
correct_predictions = 0
inference_times = []

# Buffery pro DPU
input_data = [np.empty(input_ndim, dtype=np.int8)]
output_data = [np.empty(output_ndim, dtype=np.int8)]

for class_idx, class_name in enumerate(CLASSES):
    class_dir = os.path.join(DATASET_PATH, class_name)
    if not os.path.isdir(class_dir):
        print(f"Varování: Složka {class_dir} nebyla nalezena, přeskakuji.")
        continue
    
    images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Třída {class_name:10}: nalezeno {len(images)} obrázků.")
    
    for img_name in images:
        img_path = os.path.join(class_dir, img_name)
        processed = preprocess(img_path)
        if processed is None: continue
        
        input_data[0][0,...] = processed
        
        # Měříme jen čas strávený v DPU
        start_t = time.perf_counter()
        job_id = dpu_runner.execute_async(input_data, output_data)
        dpu_runner.wait(job_id)
        end_t = time.perf_counter()
        
        inference_times.append(end_t - start_t)
        
        if np.argmax(output_data[0][0]) == class_idx:
            correct_predictions += 1
        total_images += 1
        
        if total_images % 100 == 0:
            print(f" Zpracováno {total_images} obrázků...")

# --- FINÁLNÍ VYHODNOCENÍ ---
if total_images > 0:
    avg_latency = np.mean(inference_times) * 1000 # v ms
    fps = 1.0 / np.mean(inference_times)
    accuracy = (correct_predictions / total_images) * 100

    print("\n" + "="*45)
    print(f" VÝSLEDKY VALIDACE NA KRIA KV260")
    print("="*45)
    print(f"Zpracováno celkem:  {total_images} obrázků")
    print(f"Celková přesnost:   {accuracy:.2f} %")
    print(f"Latence DPU (avg):  {avg_latency:.2f} ms")
    print(f"Propustnost (FPS):  {fps:.2f} obr./s")
    print("="*45)
else:
    print("Nebyl nalezen žádný obrázek ke zpracování. Zkontroluj cesty!")