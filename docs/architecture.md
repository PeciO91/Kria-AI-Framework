# Architecture

This document describes how the repository is organized, where CPU and DPU responsibilities split, and what the deployment boundary looks like. For measured results see [experiments.md](experiments.md); for commands see the [classification](../scripts/classification/README.md), [detection](../scripts/detection/README.md), and [segmentation](../scripts/segmentation/README.md) guides.

## Design Overview

The code is organized by model family rather than a universal task abstraction. Every family exposes the same pipeline stages — `inspect`, `evaluate`/`optimize`, `quantize`, `compile`, `benchmark` — routed through `python -m kria_ai`, but each family owns its task semantics.

```text
kria_ai/
├── common/
│   ├── artifacts.py            # canonical per-model build layout
│   ├── checkpoints.py          # checkpoint validation/loading helpers
│   ├── model_metrics.py
│   ├── vitis/
│   │   ├── inspector.py
│   │   ├── quantizer.py        # calibration + test-mode XMODEL export
│   │   └── compiler.py
│   └── board/
│       ├── config.py           # KV260 / DPU metadata
│       ├── runtime.py          # XIR/VART runner management
│       ├── input_quantization.py
│       ├── power.py            # SOM power sampling
│       └── profiling.py        # per-stage timing/reporting
├── classification/
│   ├── config.py               # models + datasets registry
│   ├── models.py  data.py  preprocess.py  evaluate.py
│   ├── optimize.py  quantize.py  benchmark.py
└── yolov26/
    ├── config.py               # shared COCO classes / normalization
    ├── models.py  data.py  preprocess.py
    ├── export.py               # raw-output graph export
    ├── decode.py               # anchor-free decode (top-k lives in detection/postprocess.py)
    ├── quantize.py
    ├── detection/              # configs, evaluation, benchmark
    └── segmentation/           # configs, masks, evaluation, benchmark
```

## Family / Common Boundary

`kria_ai/common/` never builds torchvision/Ultralytics models, parses labels, creates task losses, letterboxes images, decodes boxes, or assembles masks. Family modules construct those objects and pass them to the small common Vitis/runtime APIs. This keeps shared stages model-agnostic while family code stays the single place where model- and dataset-specific decisions live.

## Configuration And Compatibility Views

Authoritative registries are typed frozen dataclasses:

- `kria_ai/classification/config.py` — classification models and the `intel_images` dataset.
- `kria_ai/yolov26/detection/config.py` — `yolov26s` and `yolov26n_FaceHuman` checkpoints, decoder metadata (`strides`, `reg_max`, thresholds, `max_detections`), dataset image/label roots, and prune-exclude lists.
- `kria_ai/yolov26/segmentation/config.py` — `yolov26n_seg` and mask metadata (32 mask coefficients, mask threshold).
- `kria_ai/common/board/config.py` — KV260 board and `DPUCZDX8G_ISA1_B4096` at 300 MHz, default 2 runners, valid range 1–4.
- `configs/yolov26/` — DPU-targeted Ultralytics architecture YAML used only for model construction.

The root `model_config.py`, `dataset_config.py`, `board_config.py`, and files under `scripts/` are public compatibility views over these registries — they describe the same metadata and are not independent sources of truth.

Configured clocks and runner limits are settings, not measurements of hardware utilization.

## Model Preparation And Graph Policies

YOLOv26 loading validates the checkpoint head against configured class/head metadata exactly; a checkpoint whose head does not match its configured `nc` is rejected rather than decoded with wrong metadata.

The FaceHuman detector applies two project-specific adaptations for the DPUCZDX8G:

- **Activation policy** (`leaky_13_128_conv_relu_dw`): LeakyReLU(13/128) on regular Conv layers and ReLU on depthwise Conv layers, because DPUCZDX8G does not support fused depthwise LeakyReLU.
- **Graph policy** (`split_c3k2_cv1`): a weight-equivalent rewrite of the C3k2 runtime channel chunk into two Conv branches, avoiding `nndct_strided_slice`.

These graph-level adaptations are the project's contribution; it does not claim invention of the YOLO architecture or of the Vitis AI quantization algorithms. Head-level layers listed in `prune_excludes` are excluded from optimizer channel removal.

## CPU / DPU Output Contract

- **Detection**: the compiled graph exports six split one2one tensors. Anchor-free decode and top-k selection run on the ARM CPU; there is no NMS (end-to-end one2one matching).
- **Instance segmentation**: the graph exports box, class, and mask-coefficient tensors for three pyramid levels plus mask prototypes; CPU post-processing assembles masks.
- **Classification**: the DPU output is classified on the CPU.

Board code imports XIR/VART lazily so CLI parsing and configuration do not crash on a host machine, and unavailable power/compute metrics are reported as unavailable. This is graceful degradation only: actual inference still requires the board and its runtime. The current runtime requires exactly one DPU subgraph per compiled model — it is not a general heterogeneous CPU-subgraph executor.

## Artifact Layout

Canonical artifacts are keyed by model ID:

```text
build/<model-id>/inspector_report/
build/<model-id>/optimizer_report/
build/<model-id>/quantize_result/<model-id>_int.xmodel
build/<model-id>/compiled/<model-id>_kria.xmodel
```

For the measured detector: `build_fast_ft/yolov26n_facehuman/compiled/yolov26n_facehuman_kria.xmodel` (note the lowercase artifact ID vs the mixed-case CLI key `yolov26n_FaceHuman`). `--build-root` redirects the whole tree; keep one build root per pipeline run. Do not hand-edit generated artifacts, checkpoints, datasets, or XMODEL files.

## Optional Optimization Handoff

The optimizer is family-owned and experimental. It writes an explicit slim checkpoint plus manifest under `build/<model-id>/optimizer_report/`; nothing is selected automatically — pass the checkpoint explicitly to a later stage via `--checkpoint`. Sparse optimizer checkpoints are rejected by quantization; use the materialized slim checkpoint. Classification, detection, and segmentation all expose `--method {iterative,one_step}` (default `one_step`), supported experimentally; YOLO families keep their own `prune_excludes` lists. `--fast-ft` (YOLO Fast Finetune) is experimental and must be passed to both the calibration and the test/export quantize commands when used.

## Deployment Boundary

Transfer is manual and intentionally not automated. Copy to the KV260:

1. the `kria_ai/` package,
2. the compiled `<model-id>_kria.xmodel`,
3. test images (and labels if running `--accuracy`).

Board inference does not require PyTorch checkpoints, the Ultralytics source checkout, or the Vitis AI toolchain — only the XMODEL, `kria_ai`, NumPy, OpenCV, and a compatible XIR/VART runtime. On the board, producers feed a bounded queue to DPU consumers (1–4 runners), with optional writers; the pipeline reports per-stage timing and SOM power/energy (see [experiments.md](experiments.md) for definitions).

## Host Validation

Dependency-dependent checks (require the optional packages they import — NumPy, OpenCV, Ultralytics, Vitis AI — and are not claimed to pass in every environment):

```bash
python -m unittest discover -s tests -v
python -m compileall -q kria_ai tests
```

The narrow contract checks used for this documentation — `test_family_configs.py`, `test_benchmark_cli.py`, and the CLI `--help` surface — are listed in the [README](../README.md#reproducibility-and-documentation).
