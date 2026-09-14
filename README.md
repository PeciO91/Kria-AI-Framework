# Kria Vitis AI Experimental Deployment Platform

This repository deploys PyTorch image-classification and YOLOv26 models to the AMD/Xilinx Kria KV260 with Vitis AI 3.5.

```text
prepared PyTorch model
  -> Inspector
  -> optional family-specific pruning/fine-tuning
  -> INT8 calibration and XMODEL export
  -> Vitis AI compiler
  -> manual board deployment
  -> KV260 benchmark
```

The code is organized by model family rather than a universal task abstraction.

## Supported scope

| Family | Models/tasks | Status |
|---|---|---|
| Classification | ResNet18/50, MobileNetV2, MobileNetV3-Large | Deployment path migrated; hardware parity validation required |
| YOLOv26 detection | YOLOv26s, COCO-80, one2one anchor-free output | Deployment path migrated; hardware parity validation required |
| YOLOv26 instance segmentation | YOLOv26n-Seg, COCO-80, CPU mask assembly | Deployment path migrated; hardware parity validation required |
| Optimizer/pruning | Family-specific Vitis iterative and one-step flows | Experimental; run only in Vitis AI Docker |

YOLOv5 and semantic segmentation are not part of the current architecture.

## Architecture

```text
kria_ai/
├── common/
│   ├── artifacts.py
│   ├── checkpoints.py
│   ├── model_metrics.py
│   ├── vitis/
│   │   ├── inspector.py
│   │   ├── quantizer.py
│   │   └── compiler.py
│   └── board/
│       ├── config.py
│       ├── runtime.py
│       ├── input_quantization.py
│       ├── power.py
│       └── profiling.py
├── classification/
│   ├── config.py
│   ├── models.py
│   ├── data.py
│   ├── preprocess.py
│   ├── evaluate.py
│   ├── optimize.py
│   ├── quantize.py
│   └── benchmark.py
└── yolov26/
    ├── models.py
    ├── data.py
    ├── preprocess.py
    ├── export.py
    ├── decode.py
    ├── quantize.py
    ├── detection/
    └── segmentation/
```

`common` never builds torchvision/Ultralytics models, parses labels, creates task losses, letterboxes images, decodes boxes, or assembles masks. Family modules prepare those objects and pass them to the small common Vitis/runtime APIs.

The root `model_config.py`, `dataset_config.py`, `board_config.py`, and files under `scripts/` are temporary compatibility adapters.

## Host commands

Run commands from the repository root. Inspection, quantization, optimization, and compilation require the Vitis AI 3.5 PyTorch Docker environment.

### Classification

```bash
python -m kria_ai classification evaluate --model resnet18 --dataset intel_images --subset-len 32
python -m kria_ai classification inspect --model resnet18
python -m kria_ai classification quantize --model resnet18 --dataset intel_images --mode calib --subset-len 100
python -m kria_ai classification quantize --model resnet18 --dataset intel_images --mode test
python -m kria_ai classification compile --model resnet18
```

### YOLOv26 detection

```bash
python -m kria_ai yolov26 detection inspect --model yolov26s
python -m kria_ai yolov26 detection quantize --model yolov26s --dataset coco --mode calib --subset-len 100
python -m kria_ai yolov26 detection quantize --model yolov26s --dataset coco --mode test
python -m kria_ai yolov26 detection compile --model yolov26s
```

### YOLOv26 instance segmentation

```bash
python -m kria_ai yolov26 segmentation inspect --model yolov26n_seg
python -m kria_ai yolov26 segmentation quantize --model yolov26n_seg --dataset coco --mode calib --subset-len 100
python -m kria_ai yolov26 segmentation quantize --model yolov26n_seg --dataset coco --mode test
python -m kria_ai yolov26 segmentation compile --model yolov26n_seg
```

YOLOv26 model loading validates the checkpoint head against the configured COCO-80 contract. An 8-class checkpoint is rejected rather than decoded with incorrect metadata.

AdaQuant is available for classification. YOLOv26 `--fast-ft` is intentionally disabled until a callback is validated against the patched raw-output graph.

## Optimizer

Optimization is family-owned and emits an explicit checkpoint plus manifest under `build/<model-id>/optimizer_report/`.

```bash
python -m kria_ai classification optimize --model resnet18 --dataset intel_images --method iterative --mode all --ratio 0.2
python -m kria_ai yolov26 detection optimize --model yolov26s --dataset coco --method one_step --mode all --ratio 0.2
python -m kria_ai yolov26 segmentation optimize --model yolov26n_seg --dataset coco --method one_step --mode all --ratio 0.2
```

Pass the result explicitly to a later stage:

```bash
python -m kria_ai classification quantize --model resnet18 --dataset intel_images --checkpoint build/resnet18/optimizer_report/resnet18_one_step_r0.2_optimized.pt --mode calib
```

Sparse optimizer checkpoints are rejected by quantization; use the materialized slim checkpoint.

## Generated artifacts

Canonical artifacts use model IDs:

```text
build/<model-id>/inspector_report/
build/<model-id>/optimizer_report/
build/<model-id>/quantize_result/<model-id>_int.xmodel
build/<model-id>/compiled/<model-id>_kria.xmodel
```

Do not hand-edit generated artifacts, checkpoints, datasets, or XMODEL files.

## KV260 benchmark

Transfer automation is intentionally deferred. Manually copy the `kria_ai` package, required XMODEL, and dataset to the board, then run:

```bash
python3 -m kria_ai classification benchmark --model resnet18 --dataset intel_images --xmodel resnet18_kria.xmodel --threads 2 --profile
python3 -m kria_ai yolov26 detection benchmark --model yolov26s --dataset coco --xmodel yolov26s_kria.xmodel --threads 2 --profile
python3 -m kria_ai yolov26 segmentation benchmark --model yolov26n_seg --dataset coco --xmodel yolov26n_seg_kria.xmodel --threads 2 --profile
```

Segmentation supports `--accuracy --labels-dir ...` for the current mask mAP@0.5 metric and `--video ... --threads 1 --producers 1` for ordered file-video inference.

Board code imports XIR/VART lazily, validates the KV260 1-4 runner limit, and reports unavailable power/compute metrics without crashing.

## Configuration

- `kria_ai/classification/config.py`: classification models and datasets.
- `kria_ai/yolov26/detection/config.py`: YOLOv26 detection assets and thresholds.
- `kria_ai/yolov26/segmentation/config.py`: YOLOv26 segmentation assets and mask metadata.
- `kria_ai/common/board/config.py`: KV260/DPU metadata.
- `configs/yolov26/`: DPU-friendly Ultralytics architectures.

Static registries use frozen Python dataclasses. Architecture YAML remains limited to Ultralytics model construction.

## Validation

Dependency-light checks:

```bash
python -m unittest discover -s tests -v
python -m compileall -q kria_ai tests
python -m kria_ai --help
```

Full behavior validation requires the Vitis AI Docker environment and KV260. Long pruning, compilation, transfer, and hardware commands should be run explicitly, not as part of a normal unit-test pass.
