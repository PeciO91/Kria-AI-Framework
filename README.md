# Kria Vitis AI Universal Deployment Pipeline

This repository provides a modular, automated pipeline for deploying PyTorch models to the Xilinx Kria KV260 FPGA using Vitis AI 3.0. The project orchestrates the entire flow from high-level PyTorch weights to hardware-accelerated DPU instructions.

## Development Status

- **Classification:** Stable and hardware-verified.
- **Object Detection:** [Work in Progress] Logic implemented; DPU-compatibility and post-processing verification ongoing.
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
From your Vitis AI 3.0 Docker container, run:

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

## WIP Features and Technical Strategy

* **YOLO Detection:** To ensure 100% DPU compatibility, the "Detect Head" is stripped during the Host-side phase. The detection logic (Anchors and NMS) is implemented in optimized NumPy/OpenCV code within `run_detection.py` to run on the Kria ARM CPU.
* **Segmentation:** Implementing pixel-wise `argmax` logic that operates directly on raw INT8 DPU outputs to minimize CPU overhead.
* **Structural Pruning:** The `run_optimizer.py` script leverages the Vitis AI Pruner to slim models. The current work focuses on integrating the retraining loop to recover accuracy loss post-pruning.
