# Project Map - Kria Vitis AI Experimental Deployment Platform

## Project purpose
Deploy PyTorch image-classification and YOLOv26 detection/instance-segmentation models to the AMD/Xilinx Kria KV260 using Vitis AI 3.5. The family-owned flow covers inspection, optional experimental pruning/fine-tuning, INT8 calibration/test, XMODEL compilation, manual deployment and board benchmarking; hardware parity validation is still required.

## Stack
- Python and PyTorch; torchvision classification and a local Ultralytics checkout for YOLOv26.
- Vitis AI 3.5 PyTorch Docker for host inspection, optimization, quantization and compilation.
- XIR/VART on KV260 for compiled DPU execution; NumPy/OpenCV CPU preprocessing and postprocessing.
- Frozen Python dataclasses for family/board configuration; Ultralytics YAML only for model architecture.
- Standard-library unittest for dependency-light contract tests; no root dependency manifest is currently provided.

## Entry points
- `python -m kria_ai classification <command>`: evaluate, inspect, optimize, quantize, compile, benchmark.
- `python -m kria_ai yolov26 detection <command>` and `python -m kria_ai yolov26 segmentation <command>`: family task workflows.
- `scripts/` and root registries remain compatibility adapters, not separate pipeline implementations.

## Structure
- `README.md` - current scope, architecture, commands and validation status.
- `DATASET_PREPARATION.md` - detection box labels, segmentation polygons and calibration semantics.
- `.windsurfrules` - deployment constraints; some legacy architecture descriptions predate kria_ai.
- `AGENTS.md` and `CLAUDE.md` - projectmem session-start and audit-trail bridges.
- `model_config.py`, `dataset_config.py`, `board_config.py` - public compatibility registry views backed by family/board dataclasses.
- `kria_ai/` - current model-family implementation.
  - `kria_ai/__main__.py`, `kria_ai/cli.py` - module entry point and lazy family dispatch.
  - `kria_ai/classification/` - classification models, data and deployment stages.
    - `kria_ai/classification/config.py` - model/dataset dataclasses and ordered classes/normalization.
    - `kria_ai/classification/models.py` - torchvision model construction and checkpoint preparation.
    - `kria_ai/classification/data.py`, `kria_ai/classification/preprocess.py` - host datasets and shared preprocessing.
    - `kria_ai/classification/cli.py` - stage dispatch.
    - `kria_ai/classification/evaluate.py`, `kria_ai/classification/optimize.py` - FP32 evaluation and experimental pruning/fine-tuning.
    - `kria_ai/classification/inspect.py`, `kria_ai/classification/quantize.py`, `kria_ai/classification/compile.py` - family host-stage orchestration.
    - `kria_ai/classification/benchmark.py` - board classification inference and reporting.
  - `kria_ai/yolov26/` - shared YOLOv26 graph, data and deployment preparation.
    - `kria_ai/yolov26/cli.py`, `kria_ai/yolov26/config.py` - task/stage routing and shared class/normalization constants.
    - `kria_ai/yolov26/models.py`, `kria_ai/yolov26/dpu.py`, `kria_ai/yolov26/export.py` - model loading, DPU graph patches and raw-output wrappers.
    - `kria_ai/yolov26/data.py`, `kria_ai/yolov26/preprocess.py` - image-only calibration/subsets and shared letterboxing.
    - `kria_ai/yolov26/inspect.py`, `kria_ai/yolov26/quantize.py`, `kria_ai/yolov26/compile.py` - shared task-aware host stages.
    - `kria_ai/yolov26/decode.py`, `kria_ai/yolov26/pruning.py` - anchor-free output contracts/decoding and pruning support.
    - `kria_ai/yolov26/detection/config.py`, `kria_ai/yolov26/detection/cli.py` - detection assets, head metadata, thresholds and routing.
    - `kria_ai/yolov26/detection/data.py`, `kria_ai/yolov26/detection/loss.py` - box-label parsing and task loss.
    - `kria_ai/yolov26/detection/evaluate.py`, `kria_ai/yolov26/detection/optimize.py` - detection evaluation and experimental optimization.
    - `kria_ai/yolov26/detection/postprocess.py`, `kria_ai/yolov26/detection/benchmark.py` - CPU top-k/coordinate transforms and board inference.
    - `kria_ai/yolov26/segmentation/config.py`, `kria_ai/yolov26/segmentation/cli.py` - instance-segmentation assets, mask metadata and routing.
    - `kria_ai/yolov26/segmentation/data.py`, `kria_ai/yolov26/segmentation/loss.py` - polygon/COCO data and task loss.
    - `kria_ai/yolov26/segmentation/evaluate.py`, `kria_ai/yolov26/segmentation/optimize.py` - mask evaluation and experimental optimization.
    - `kria_ai/yolov26/segmentation/masks.py`, `kria_ai/yolov26/segmentation/benchmark.py` - CPU mask assembly and board image/video inference.
  - `kria_ai/common/` - task-independent shared primitives.
    - `kria_ai/common/artifacts.py`, `kria_ai/common/checkpoints.py`, `kria_ai/common/model_metrics.py` - canonical artifact paths, checkpoint handling and model metrics.
    - `kria_ai/common/vitis/inspector.py`, `kria_ai/common/vitis/quantizer.py`, `kria_ai/common/vitis/compiler.py` - common Vitis tool adapters.
    - `kria_ai/common/board/config.py` - KV260 DPU/compiler metadata and runner limits.
    - `kria_ai/common/board/runtime.py`, `kria_ai/common/board/input_quantization.py` - XIR/VART runners and quantized input handling.
    - `kria_ai/common/board/power.py`, `kria_ai/common/board/profiling.py` - graceful power access and stage profiling.
- `configs/` - Ultralytics architecture YAMLs.
  - `configs/yolov26/detection/yolov26s_dpu.yaml`, `configs/yolov26/segmentation/yolov26n_seg_dpu.yaml` - current COCO DPU architectures.
  - `configs/yolo26n_dpu_leaky_dwrelu.yaml` - 2-class FaceHuman architecture.
- `scripts/` - deprecated forwarding entry points and compatibility utilities.
  - `scripts/common/run_inspector.py`, `scripts/common/run_optimizer.py`, `scripts/common/run_quantizer.py`, `scripts/common/run_compiler.py` - legacy host command adapters.
  - `scripts/common/model_utils.py`, `scripts/common/dataset_utils.py`, `scripts/common/board_utils.py` - compatibility helpers.
  - `scripts/classification/run_inference.py`, `scripts/detection/run_detection.py`, `scripts/segmentation/run_instance_seg.py` - legacy board benchmark adapters.
- `tests/` - host-side unit/contract tests.
  - `tests/test_family_configs.py`, `tests/test_common_primitives.py` - registries, artifacts and checkpoints.
  - `tests/test_board_runtime.py`, `tests/test_benchmark_cli.py` - board runtime and CLI contracts.
  - `tests/test_output_contracts.py`, `tests/test_detection_utils.py`, `tests/test_seg_utils.py` - output layout, box decode and masks.
  - `tests/test_yolo_models.py`, `tests/test_yolo_preprocessing.py`, `tests/test_yolo_quantize.py` - YOLO preparation and calibration contracts.
  - `tests/test_quantizer.py`, `tests/test_pruning_api.py`, `tests/test_vitis_compiler.py` - Vitis integration seams.
- `models/` - local checkpoints and third-party model sources; do not hand-edit generated weights.
- `data/` - datasets and deterministic subset caches; do not edit calibration artifacts manually.
- `build/`, `build_fast_ft/` - generated deployment artifacts and isolated Fast Finetune experiments.
- `results/` - benchmark/evaluation output.
- `docs/` - supplementary documentation; verify legacy examples against current README and CLI.

## Relationships
- `kria_ai/__main__.py` calls `kria_ai/cli.py`, which dispatches to `kria_ai/classification/cli.py` or `kria_ai/yolov26/cli.py`.
- `kria_ai/yolov26/cli.py` dispatches task CLIs; detection/segmentation CLIs forward inspect/quantize/compile to shared YOLO modules with a task flag.
- Family stage modules read family config/dataclasses and prepare models/data before calling `kria_ai/common/vitis/` adapters.
- `model_config.py`, `dataset_config.py` and `board_config.py` expose family/board configuration to `scripts/` compatibility callers.
- `kria_ai/common/artifacts.py` names inspector, optimizer, quantizer and compiler outputs by canonical model ID under the selected build root.
- `kria_ai/yolov26/models.py` uses DPU patches/export wrappers with YAML architectures and local model assets; raw one2one outputs are decoded on CPU.
- Host calibration and board YOLO benchmarks share `kria_ai/yolov26/preprocess.py`; decoder/mask metadata comes from task config.
- Family benchmark modules use `kria_ai/common/board/runtime.py`, quantization, power and profiling primitives while retaining task-specific decode/evaluation.
- `scripts/common/run_*.py` forward to current host CLIs; legacy board runners forward to family benchmark modules.
- `tests/` checks contracts across registries, preprocessing, outputs, runtime and Vitis seams without establishing hardware parity.

## Validation and execution boundaries
- From the repository root: `python -m unittest discover -s tests -v`, `python -m compileall -q kria_ai tests`, `python -m kria_ai --help`.
- Run the narrowest relevant check first; Vitis stages require the Docker environment and benchmarks require KV260.
- Pruning and YOLO Fast Finetune are experimental. Do not claim stable hardware parity from host tests alone.
- Transfer is manual. Long pruning/compilation, SSH/SCP, hardware execution, network access and installs require explicit user approval.
- Never hand-edit generated XMODEL, compiler, calibration, dataset or weight artifacts.

## Memory Maintenance
- PROJECT_MAP.md and plan.md are authored files and may be edited directly after precheck_file, preserving unrelated content.
- summary.md and events.jsonl are maintained through Projectmem tools and must not be edited manually.
