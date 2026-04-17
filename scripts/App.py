import numpy as np
import cv2
import vart
import xir
import os

# 1. Načtení modelu
model_path = "output_kria/resnet18_intel_kria.xmodel"
graph = xir.Graph.deserialize(model_path)
root_subgraph = graph.get_root_subgraph()
child_subgraphs = root_subgraph.get_children()
dpu_subgraphs = [s for s in child_subgraphs if s.get_attr("device").upper() == "DPU"]
dpu_runner = vart.Runner.create_runner(dpu_subgraphs[0], "run")

# 2. Získání informací o vstupech a výstupech
input_tensors = dpu_runner.get_input_tensors()
output_tensors = dpu_runner.get_output_tensors()
input_ndim = tuple(input_tensors[0].dims)
output_ndim = tuple(output_tensors[0].dims)

# 3. Příprava BUFFERŮ (Tohle je ta oprava!)
# Vytvoříme seznamy s numpy poli, kam DPU zapíše data
input_data = [np.empty(input_ndim, dtype=np.int8)]
output_data = [np.empty(output_ndim, dtype=np.int8)]

def preprocess_fn(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (input_ndim[2], input_ndim[1]))
    image = image.astype(np.float32)
    # Normalizace
    image = (image / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    
    # Získání fix-point měřítka (důležité pro přesnost na DPU)
    fix_pos = input_tensors[0].get_attr("fix_point")
    image = image * (2**fix_pos)
    return image.astype(np.int8)

# 4. Načtení obrázku do bufferu
image_path = "test_image.jpg"
if not os.path.exists(image_path):
    print(f"Chyba: {image_path} nenalezen!")
    exit()

input_data[0][0,...] = preprocess_fn(image_path)

# 5. Spuštění na FPGA
# Předáme naše připravené buffery
job_id = dpu_runner.execute_async(input_data, output_data)
dpu_runner.wait(job_id)

# 6. Zpracování výsledku
# Výsledek už čeká v output_data[0]
result = output_data[0][0] # vezmeme první (a jediný) batch

# Softmax pro lidsky čitelná procenta
# Pozor: result je int8, pro softmax ho převedeme na float
result_float = result.astype(np.float32)
exp_x = np.exp(result_float - np.max(result_float))
probs = exp_x / exp_x.sum()

predicted_class = np.argmax(probs)
classes = ["budovy", "ledovec", "hory", "les", "moře", "ulice"]

print(f"\n" + "="*30)
print(f" VÝSLEDEK Z FPGA (DPU)")
print(f"="*30)
print(f"Předpověď: {classes[predicted_class]}")
print(f"Jistota:   {probs[predicted_class]*100:.2f}%")
print(f"="*30 + "\n")