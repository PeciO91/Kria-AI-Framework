# Object Detection Deployment

Object detection is the stable YOLO deployment path for Kria KV260. It covers models whose `model_config.py` entry has `type: "detection"`.

Host-side stages are shared in `scripts/common/`; board-side execution uses `scripts/detection/run_detection.py` plus `scripts/detection/detection_utils.py`.

## Supported models

| Model ID | Decoder family | Input | Notes |
|---|---|---:|---|
| `yolov5n` | Anchor-based YOLOv5 | `640x640` | Raw P3/P4/P5 DPU outputs; CPU-side anchor decode + per-class NMS. |
| `yolov26s` | Ultralytics anchor-free end-to-end | `640x640` | DPU-friendly wrapper exposes one2one branch; top-k selection without NMS. |

Model-specific decoder metadata is centralized in `../../model_config.py`: thresholds, anchors, strides, `reg_max`, `max_det`, and decoder type.

## Dataset

The detection dataset is `coco_detection` in `../../dataset_config.py`.

Current host image paths:

```text
../../data/coco2017/train2017/
../../data/coco2017/val2017/
```

Subset cache path:

```text
../../data/coco2017/.subsets/
```

YOLO-format label paths are registered as:

```text
../../data/coco2017/labels/train2017/
../../data/coco2017/labels/val2017/
```

For quantizer calibration, labels are optional because the forward pass only needs images. For training, mAP evaluation, or label-aware fine-tuning, convert the COCO JSON annotations under `data/coco2017/annotations/` into YOLO `.txt` labels.

## Host environment notes

Inside the Vitis AI PyTorch Docker environment, some modern Python wheels may need the conda libstdc++ to be preloaded after `torch` is imported:

```bash
export LD_PRELOAD=/opt/vitis_ai/conda/envs/vitis-ai-pytorch/lib/libstdc++.so.6
```

Verify the library contains the required symbol:

```bash
strings /opt/vitis_ai/conda/envs/vitis-ai-pytorch/lib/libstdc++.so.6 | grep GLIBCXX_3.4.29
```

The YOLOv5 path may also need plotting/import dependencies such as `seaborn` in the Docker environment because the vendored YOLOv5 package imports plotting modules at import time.

## Host deployment

Run from the project root inside Vitis AI Docker.

### YOLOv5n

```bash
python scripts/common/deploy.py \
  --model yolov5n \
  --dataset coco_detection \
  --subset 100 \
  --transfer none
```

Validated smoke test:

```bash
python scripts/common/deploy.py \
  --model yolov5n \
  --dataset coco_detection \
  --subset 32 \
  --transfer none
```

Expected calibration evidence:

```text
Initialized YoloDataset with subset of 32 images.
Progress: 32/32 (100.0%)
Final Model: build/yolov5n/compiled/yolov5n_kria.xmodel
```

### YOLOv26s

```bash
python scripts/common/deploy.py \
  --model yolov26s \
  --dataset coco_detection \
  --subset 100 \
  --transfer none
```

The DPU-friendly architecture config is `../../configs/yolov26s_dpu.yaml`.

## Board execution

After transfer, run on the KV260:

```bash
python3 run_detection.py \
  --model yolov5n \
  --dataset coco_detection \
  --threads 2
```

```bash
python3 run_detection.py \
  --model yolov26s \
  --dataset coco_detection \
  --threads 2
```

Outputs:

- **Annotated images** in `outputs_<model>/` unless disabled.
- **Text report** in `results_<model>_t<threads>.txt`.
- **Class histogram** printed at the end of the run.
- **Performance metrics**: FPS, DPU latency, power, energy per frame, DPU duty cycle, compute efficiency.

## Profiling options

`run_detection.py` supports detailed pipeline profiling:

```bash
python3 run_detection.py \
  --model yolov26s \
  --dataset coco_detection \
  --threads 2 \
  --queue-size 1 \
  --no-draw \
  --no-save \
  --profile \
  --profile-json
```

Useful flags:

| Flag | Purpose |
|---|---|
| `--threads N` | Number of DPU consumer threads. |
| `--producers N` | Number of preprocessing producer threads. |
| `--queue-size N` | Input queue size; use `1` for low-latency experiments. |
| `--no-draw` | Skip drawing boxes/labels. |
| `--no-save` | Skip writing output images. |
| `--profile` | Print stage-level profiling. |
| `--profile-json` | Save profile data as JSON. |

Recent profiling result for YOLOv26s with `--queue-size 1 --no-draw --no-save` showed 2 DPU threads as the best throughput/average-latency point, while 1 thread had better tail latency.

## Decoder notes

### YOLOv5n

- The DPU graph returns raw P3/P4/P5 tensors.
- Anchor decoding, sigmoid activation, coordinate scaling, and per-class NMS run on the ARM CPU.
- Anchors and strides come from `model_config.py`.
- `detection_utils.py` provides `scale_coords` and `non_max_suppression`; `letterbox` is re-exported from `scripts/common/dataset_utils.py` so calibration and board preprocessing share one implementation.

### YOLOv26s

- Uses the Ultralytics loader path and DPU-friendly architecture.
- Avoids unsupported attention/PSA blocks in the compiled graph.
- Exposes the one2one branch for end-to-end matching.
- Uses anchor-free decode and top-k selection instead of NMS.

## Transfer behavior

`deploy.py` now transfers only files needed for the active task. For detection this includes:

```text
yolov*_kria.xmodel
run_detection.py
detection_utils.py
board_utils.py
model_config.py
dataset_config.py
board_config.py
```
