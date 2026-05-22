# Kria Vitis AI Universal Deployment Pipeline

This repository provides a modular pipeline for deploying PyTorch models to the Xilinx Kria KV260 using Vitis AI 3.5. The flow is:

```text
PyTorch checkpoint
  -> Inspector
  -> optional Optimizer / Pruner
  -> INT8 Quantizer calibration + export
  -> Vitis AI Compiler
  -> optional transfer to KV260
  -> board-side inference runner
```

The project is organized by task while keeping the deployment core shared.

## Development status

| Area | Status | Notes |
|---|---|---|
| Classification | Stable | ResNet18/50, MobileNetV2/V3/V4, InceptionV3. |
| Detection | Stable | YOLOv5n and YOLOv26s on COCO/KV260. |
| Segmentation | WIP | UNet_ResNet18 registry and board runner skeleton exist; evaluation path still pending. |
| Optimizer / pruning | WIP | Vitis AI structural pruning scripts are present; accuracy recovery loops are still being refined. |

## Project structure

```text
Project/
├── model_config.py                  # Authoritative model registry
├── dataset_config.py                # Authoritative dataset / normalization registry
├── board_config.py                  # KV260 board, DPU, transfer, and power metadata
├── configs/                         # Model-specific architecture configs
├── data/                            # Datasets and calibration data
├── models/                          # PyTorch checkpoints and model source trees
├── build/                           # Generated inspector / quantizer / compiler artifacts
├── docs/                            # Diagrams and documentation assets
└── scripts/
    ├── common/                      # Shared host pipeline + shared board utilities
    │   ├── deploy.py                # End-to-end orchestrator
    │   ├── run_inspector.py         # DPU compatibility inspection
    │   ├── run_quantizer.py         # INT8 calibration/export
    │   ├── run_optimizer.py         # Structural pruning (WIP)
    │   ├── run_compiler.py          # Vitis AI compiler wrapper
    │   ├── model_utils.py           # Task-aware model preparation
    │   ├── dataset_utils.py         # Shared calibration datasets + letterbox
    │   ├── optimizer_utils.py       # Pruning/evaluation helpers
    │   ├── detection_profiles.py    # Detection loss/profile presets used by common stages
    │   └── board_utils.py           # Shared KV260/VART/preprocessing helpers
    ├── classification/
    │   ├── README.md
    │   └── run_inference.py
    ├── detection/
    │   ├── README.md
    │   ├── run_detection.py
    │   └── detection_utils.py
    └── segmentation/
        ├── README.md
        └── run_segmentation.py
```

## Task guides

- **Classification**: see [`scripts/classification/README.md`](scripts/classification/README.md)
- **Object detection**: see [`scripts/detection/README.md`](scripts/detection/README.md)
- **Semantic segmentation**: see [`scripts/segmentation/README.md`](scripts/segmentation/README.md)

## Core registries

The three root config files are the source of truth:

- **`model_config.py`**: model type, model name, source loader, input shape, checkpoint path, GOPs, and detection decoder metadata.
- **`dataset_config.py`**: calibration/image paths, folder names, ordered class labels, and normalization values.
- **`board_config.py`**: KV260 DPU arch path, board IP/user, DPU peak GOPS, active threads, and power telemetry helpers.

Avoid hardcoding model, dataset, or board constants in stage scripts. Add or modify supported assets through these registries.

## Host quickstart

Run from the project root inside the Vitis AI PyTorch Docker environment.

For this Docker environment, preload the conda libstdc++ if modern Python wheels fail after importing `torch`:

```bash
export LD_PRELOAD=/opt/vitis_ai/conda/envs/vitis-ai-pytorch/lib/libstdc++.so.6
```

Classification smoke test:

```bash
python scripts/common/deploy.py \
  --model resnet18 \
  --dataset intel_images \
  --subset 32 \
  --transfer none
```

Detection smoke test:

```bash
python scripts/common/deploy.py \
  --model yolov5n \
  --dataset coco_detection \
  --subset 32 \
  --transfer none
```

The reorganized layout has been validated with both commands. Expected outputs:

```text
build/resnet18/compiled/resnet18_kria.xmodel
build/yolov5n/compiled/yolov5n_kria.xmodel
```

## Full deployment to KV260

Use the same orchestrator without `--transfer none`:

```bash
python scripts/common/deploy.py \
  --model resnet18 \
  --dataset intel_images \
  --subset 100 \
  --transfer scp
```

`deploy.py` transfers only the files needed by the active task:

- **Classification**: `run_inference.py` + shared configs/helpers.
- **Detection**: `run_detection.py`, `detection_utils.py` + shared configs/helpers.
- **Segmentation**: `run_segmentation.py` + shared configs/helpers.

On the board, run the task-specific runner from `/home/root/` (or the configured board user home):

```bash
python3 run_inference.py --model resnet18 --dataset intel_images --threads 2
python3 run_detection.py --model yolov5n --dataset coco_detection --threads 2
python3 run_segmentation.py --model unet_res18 --dataset cityscapes_seg --threads 2
```

## Supported datasets

| Dataset ID | Task | Main paths |
|---|---|---|
| `intel_images` | Classification | `data/intel_images/calibration_data` |
| `intel_images_inception` | Classification | `data/intel_images/calibration_data` |
| `coco_detection` | Detection | `data/coco2017/train2017`, `data/coco2017/val2017` |
| `cityscapes_seg` | Segmentation | `data/cityscapes/calibration_data` |

COCO detection calibration uses images from `data/coco2017`. YOLO-format labels are optional for pure quantizer calibration but required for training or mAP evaluation.

## Detection notes

- **YOLOv5n** keeps raw P3/P4/P5 DPU outputs and performs anchor decode + NMS on the ARM CPU.
- **YOLOv26s** uses a DPU-friendly Ultralytics wrapper, one2one branch, anchor-free decode, and top-k selection without NMS.
- Board profiling flags include `--profile`, `--profile-json`, `--queue-size`, `--producers`, `--no-draw`, and `--no-save`.

See [`scripts/detection/README.md`](scripts/detection/README.md) for details.

## Generated artifacts

Generated artifacts live under `build/<model-name>/`:

```text
build/<model>/inspector_report/
build/<model>/quantize_result/
build/<model>/compiled/
```

Do not hand-edit generated quantizer, compiler, `.xmodel`, weight, dataset, or calibration artifacts unless intentionally debugging the generated output.

## Documentation

Additional diagrams live under `docs/`.
