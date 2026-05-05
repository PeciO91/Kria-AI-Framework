# Kria Vitis AI Universal Deployment Pipeline

This repository provides a modular, automated pipeline for deploying PyTorch models to the Xilinx Kria KV260 FPGA using Vitis AI 3.5. The project orchestrates the entire flow from high-level PyTorch weights to hardware-accelerated DPU instructions.

## Development Status

- **Classification:** Stable and hardware-verified (ResNet18/50, MobileNetV2/V3/V4, InceptionV3).
- **Object Detection:** Stable and hardware-verified end-to-end on COCO with two model families:
  - **YOLOv5n** (anchor-based, 4.5 GOPs): ~21.8 FPS, 11.3 ms DPU latency on KV260, 2 threads.
  - **YOLOv26s** (Ultralytics anchor-free, DFL-free, end2end one2one head, 22.8 GOPs): ~23 FPS, 40 ms DPU latency on KV260, 3 threads.
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
The `coco_detection` dataset is registered in `dataset_config.py` with the 80 COCO class names. Detection models in `model_config.py` declare their decoder family (`yolov5_anchor` or `ultralytics_anchor_free`), strides, conf/iou thresholds, and any model-specific flags (e.g. `end2end`, `reg_max`, `replace_leaky_relu`).

### 2. DPU Compatibility Notes

**YOLOv5n:** Uses SiLU by default. The YAML (`models/yolov5n/models/yolov5n.yaml`) overrides the activation with `nn.LeakyReLU(26/256, inplace=True)` for DPU compatibility. The `Detect` head is stripped to return raw P3/P4/P5 conv outputs; anchor decoding and per-class NMS run on the ARM CPU.

**YOLOv26s:** The stock checkpoint ships with `C2PSA`, `PSABlock`, and `Attention` modules whose `matmul / softmax / permute / reshape / strided_slice` ops are not DPU-supported and would cause large CPU-fallback subgraphs. We retrain from a DPU-friendly YAML (`configs/yolov26s_dpu.yaml`) that uses only `Conv`, `C3`, `SPPF`, `Concat`, `Upsample` and a vanilla `Detect` head. A thin wrapper (`scripts/ultralytics_vitis_wrapper.py`) loads the Ultralytics model, optionally rewrites any residual `LeakyReLU` modules into `ReLU` (needed because the DPU rejects `DepthwiseConv + LeakyReLU` fusions), and exposes only the `one2one` detection branch so no dead `one2many` weights end up in the compiled graph. The board decodes the anchor-free `(4·reg_max + nc)`-channel grids directly off the INT8 buffers; post-processing is top-k by score (no NMS, since the `one2one` head is trained with bipartite matching).

### 3. Full Deployment Chain
```bash
# YOLOv5n (anchor-based)
python3 scripts/run_quantizer.py --model yolov5n  --dataset coco_detection --quant_mode calib --subset_len 50
python3 scripts/run_quantizer.py --model yolov5n  --dataset coco_detection --quant_mode test
python3 scripts/run_compiler.py  --model yolov5n
scp -O build/yolov5n/compiled/yolov5n_kria.xmodel root@<board-ip>:/home/root/

# YOLOv26s (Ultralytics anchor-free, end2end)
python3 scripts/run_quantizer.py --model yolov26s --dataset coco_detection --quant_mode calib --subset_len 50
python3 scripts/run_quantizer.py --model yolov26s --dataset coco_detection --quant_mode test
python3 scripts/run_compiler.py  --model yolov26s
scp -O build/yolov26s/compiled/yolov26s_kria.xmodel root@<board-ip>:/home/root/
```

### 4. Board Execution
```bash
python3 run_detection.py --model yolov5n  --dataset coco_detection --threads 2
python3 run_detection.py --model yolov26s --dataset coco_detection --threads 3
```

Drawn images with bounding boxes and COCO class labels are saved to `outputs_<model>/`. A per-class detection histogram is printed at the end of each run.

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

* **YOLOv5n (anchor-based):** The `Detect` head is stripped during the Host-side phase, returning the three raw P3/P4/P5 conv outputs. SiLU activations are replaced with DPU-friendly `LeakyReLU(26/256)`. Anchor decoding, sigmoid post-processing, coordinate scaling, and per-class NMS run on the Kria ARM CPU using vectorized NumPy/OpenCV code in `run_detection.py`.
* **YOLOv26s (anchor-free, end2end):** Retrained from a DPU-friendly architecture that excludes `C2PSA / PSABlock / Attention` modules. The wrapper exposes only the `one2one` detection branch and replaces any residual `LeakyReLU` with `ReLU` to avoid the unsupported `DepthwiseConv + LeakyReLU` DPU fusion. On-board decoding is lazy: the decoder finds the max class in INT8 space, thresholds in INT8, and only dequantizes surviving anchors to float32 (typical survivor rate at conf=0.1 is ~1%). Because the one2one head is trained with bipartite matching, NMS is skipped and final detections are selected via top-k by score.
* **Segmentation (WIP):** Implementing pixel-wise `argmax` logic that operates directly on raw INT8 DPU outputs to minimize CPU overhead.
* **Structural Pruning (WIP):** The `run_optimizer.py` script leverages the Vitis AI Pruner to slim models. The current work focuses on integrating the retraining loop to recover accuracy loss post-pruning.
