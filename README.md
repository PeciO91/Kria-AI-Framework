# Kria Vitis AI Universal Deployment Pipeline

This repository provides a modular, automated pipeline for deploying PyTorch models to the Xilinx Kria KV260 FPGA using Vitis AI 3.5. The project orchestrates the entire flow from high-level PyTorch weights to hardware-accelerated DPU instructions.

## Development Status

- **Classification:** Stable and hardware-verified (ResNet18/50, MobileNetV2/V3/V4, InceptionV3).
- **Object Detection:** Stable and hardware-verified end-to-end with YOLOv5n on COCO (~21.8 FPS, 11.3 ms DPU latency on KV260, 2 threads).
- **Semantic Segmentation:** [Work in Progress] Post-processing and masking scripts in development.
- **Optimizer/Pruning:** [Work in Progress] Structural pruning supported; automated fine-tuning loops for accuracy recovery are under construction.

---

## Pipeline Architecture

The pipeline is built on a "task-agnostic" core. A central configuration file manages metadata, allowing the same quantization and compilation scripts to handle different model architectures (ResNet, YOLO, UNet) by simply switching a task type flag.

### 1. Host-Side Orchestration (scripts/)
- **deploy.py**: The Master Orchestrator. A single command to execute the Inspector, Quantizer, Compiler, and automated SCP transfer to the target board.
- **run_quantizer.py**: Automated INT8 calibration using a task-aware data loader that supports both standard class-folders and flat-folder structures.
- **model_utils.py**: Dynamically prepares models for Vitis AI by handling dynamic layer replacements and stripping incompatible heads.

### 2. Board-Side Execution (board/)
- **Multi-threaded Runners**: Optimized Producer/Consumer architecture. Producers handle CPU-bound preprocessing (resizing, letterboxing, normalization), while multiple Consumer threads manage asynchronous DPU execution.
- **Power Monitoring**: Integrated system calls to track real-time power consumption (mW) and energy efficiency (mJ/frame) on the Kria SOM.

---

## Project Structure

```text
├── scripts/
│   ├── deploy.py            # Master automation pipeline
│   ├── run_inspector.py     # DPU compatibility checker
│   ├── run_optimizer.py     # Structural pruning (WIP)
│   ├── run_quantizer.py     # INT8 calibration
│   ├── run_compiler.py      # DPU xmodel generation
│   ├── model_utils.py       # Task-aware graph manipulator
│   └── detection_utils.py   # Math helpers (NMS, Letterboxing)
├── board/
│   ├── run_inference.py     # Optimized classification runner
│   ├── run_detection.py     # YOLO detection runner (WIP)
│   └── run_segmentation.py  # UNet segmentation runner (WIP)
├── config/
│   ├── model_config.py      # Central model registry
│   ├── dataset_config.py    # Dataset & Normalization registry
│   └── board_config.py      # Hardware-specific parameters
└── models/                  # PyTorch weights (.pt) and definitions (.py)
```

---

## Usage: Classification (Stable)

### 1. Configuration
Define your model metadata in `config/model_config.py` and dataset paths in `config/dataset_config.py`.

### 2. Full Deployment Chain
From your Vitis AI 3.5 Docker container, run:

```bash
python3 scripts/deploy.py --model resnet18 --dataset intel_images --subset 100
```

This command will:
1. Inspect the model for DPU compatibility.
2. Quantize the model to INT8.
3. Compile the model to an `.xmodel` file.
4. Transfer the model and all required board scripts to the Kria board.

### 3. Board Execution
SSH into the Kria board and run:

```bash
python3 run_inference.py --model resnet18 --dataset intel_images --threads 2
```

---

## Usage: Object Detection (Stable)

### 1. Configuration
The `coco_detection` dataset is registered in `dataset_config.py` with the 80 COCO class names. The YOLOv5n model is registered in `model_config.py` with input shape 640x640 and confidence/IOU thresholds.

### 2. DPU Compatibility Notes
YOLOv5 uses SiLU activations by default, which are not natively supported by the DPU. The model YAML (`models/yolov5n/models/yolov5n.yaml`) overrides the activation with `nn.LeakyReLU(26/256, inplace=True)` for full DPU compatibility. The `Detect` head is stripped to return raw conv outputs; anchor decoding and NMS run on the ARM CPU.

### 3. Full Deployment Chain
```bash
python3 scripts/run_quantizer.py --model yolov5n --dataset coco_detection --quant_mode calib --subset_len 50
python3 scripts/run_quantizer.py --model yolov5n --dataset coco_detection --quant_mode test
python3 scripts/run_compiler.py --model yolov5n
scp -O build/yolov5n/compiled/yolov5n_kria.xmodel root@<board-ip>:/home/root/
```

### 4. Board Execution
```bash
python3 run_detection.py --model yolov5n --dataset coco_detection --threads 2
```

Drawn images with bounding boxes and COCO class labels are saved to `outputs_yolov5n/`.

---

## Performance Reporting

Upon completion, the pipeline generates an analytical report:

```text
============================================================
  ANALYTICAL REPORT: RESNET18 | DPU THREADS: 2
============================================================
System:             200 images
Overall Accuracy:   
  -> Top-1:         94.20 %
  -> Top-5:         99.10 %
------------------------------------------------------------
Application FPS:    48.52 img/s
DPU Latency (avg):  20.61 ms
------------------------------------------------------------
Power (Load):       3.85 W
Energy per frame:   79.35 mJ/img
------------------------------------------------------------
DPU Duty Cycle:     85.12 %
DPU Compute Eff.:   82.15 %
============================================================
```

---

## Technical Strategy

* **YOLO Detection:** To ensure 100% DPU compatibility, the `Detect` head is stripped during the Host-side phase, returning the three raw P3/P4/P5 conv outputs. SiLU activations are replaced with DPU-friendly LeakyReLU(26/256). Anchor decoding, sigmoid post-processing, coordinate scaling, and NMS run on the Kria ARM CPU using optimized NumPy/OpenCV code in `run_detection.py`.
* **Segmentation (WIP):** Implementing pixel-wise `argmax` logic that operates directly on raw INT8 DPU outputs to minimize CPU overhead.
* **Structural Pruning (WIP):** The `run_optimizer.py` script leverages the Vitis AI Pruner to slim models. The current work focuses on integrating the retraining loop to recover accuracy loss post-pruning.
