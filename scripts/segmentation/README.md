# Segmentation Deployment

Segmentation support is currently **work in progress**. This section covers models whose `model_config.py` entry has `type: "segmentation"`. Two distinct paths exist:

- **Semantic segmentation** (per-pixel class map): `unet_res18`, board runner `run_segmentation.py`.
- **Instance segmentation** (per-object masks): `yolov26n_seg` (`seg_instance: True`), board runner `run_instance_seg.py` + `seg_utils.py`.

Host-side stages are shared in `scripts/common/`; the board-side runner is selected automatically by `deploy.py` based on the `seg_instance` flag in the model registry.

## Supported models

| Model ID | Model name | Input | Path | Status |
|---|---|---:|---|---|
| `unet_res18` | UNet_ResNet18 | `512x512` | Semantic | Registered, but model file/checkpoint and evaluation flow are WIP. |
| `yolov26n_seg` | YOLOv26n-Seg | `640x640` | Instance | DPU-friendly Ultralytics wrapper; LUT-accelerated preprocessing + anchor-free decode + CPU mask assembly. Native mask mAP@0.5 support. Stable. |

The `unet_res18` registry entry expects:

```text
models/unet.py
models/unet.pt
```

The `yolov26n_seg` registry entry expects:

```text
models/yolo26n-seg.pt
configs/yolo26-seg_dpu.yaml
models/ultralytics-main      # Ultralytics loader repo
```

Key `yolov26n_seg` decoder metadata in `../../model_config.py`: `decoder: ultralytics_anchor_free`, `num_classes: 80`, `reg_max: 1`, `end2end: True`, `max_det: 300`, `num_masks: 32`, `num_protos: 256`, `mask_threshold: 0.5`, `conf_threshold: 0.1`, `iou_threshold: 0.45`.

## Datasets

### Semantic: `cityscapes_seg`

Expected host calibration path:

```text
../../data/cityscapes/calibration_data/
```

The `run_segmentation.py` runner reads images from the dataset `calib_path` and writes visual overlays. It does not yet compute mIoU or pixel accuracy.

### Instance: `coco_instance_seg`

Reuses the COCO 2017 images and YOLO-seg polygon labels:

```text
../../data/coco2017/train2017/
../../data/coco2017/val2017/
../../data/coco2017/labels/train2017/   # "class x1 y1 ... xn yn" normalized polygons
../../data/coco2017/labels/val2017/
```

On the board, polygon labels for `--accuracy` runs default to `datasets/coco2017/labels/val2017` (`board_labels` in `dataset_config.py`); override with `--labels-dir`. Labels are optional for pure quantizer calibration (forward pass only).

## Host deployment

Run from the project root inside the Vitis AI PyTorch Docker environment. The common pipeline runs Inspector -> Quantizer calib -> Quantizer test -> Compiler -> optional transfer.

### Semantic (UNet)

```bash
python scripts/common/deploy.py \
  --model unet_res18 \
  --dataset cityscapes_seg \
  --subset 100 \
  --transfer none
```

Compiled model output:

```text
build/unet_resnet18/compiled/unet_res18_kria.xmodel
```

### Instance (YOLOv26n-Seg)

The model must be trained with a DPU-friendly YAML (`configs/yolo26-seg_dpu.yaml`) that replaces C2PSA/Attention with standard convolutions and uses ReLU activations.

```bash
python scripts/common/deploy.py \
  --model yolov26n_seg \
  --dataset coco_instance_seg \
  --subset 100 \
  --transfer none
```

Compiled model output:

```text
build/yolov26n-seg/compiled/yolov26n_seg_kria.xmodel
```

## Board execution

After transfer, run on the KV260.

### Semantic (`run_segmentation.py`)

```bash
python3 run_segmentation.py \
  --model unet_res18 \
  --dataset cityscapes_seg \
  --threads 2
```

The runner:

- Loads `<model_id>_kria.xmodel`.
- Reads images from `d_cfg['calib_path']`.
- Applies LUT-based input normalization through `board_utils.py`.
- Runs the DPU with multiple consumer threads.
- Applies `argmax` directly on raw INT8 output logits.
- Saves colorized overlays to `outputs_<model_id>/`.
- Reports FPS, DPU latency, power, and energy per frame.

### Instance (`run_instance_seg.py`)

```bash
python3 run_instance_seg.py \
  --model yolov26n_seg \
  --dataset coco_instance_seg \
  --threads 2
```

The runner:

- Letterboxes inputs and applies LUT-based normalization (shared with detection).
- Decodes boxes/classes with the anchor-free Ultralytics decoder (`detection_utils.py`), no NMS.
- Assembles per-object masks on the ARM CPU from mask coefficients (`num_masks`) and prototypes (`num_protos`) via `seg_utils.process_mask`, then rescales with `scale_image_masks`.
- Binarizes masks at `mask_threshold` and saves overlays unless `--no-save`.
- Reports FPS, DPU latency, power, energy per frame, and stage profiling.

Producer/consumer and profiling flags mirror the detection runner: `--threads`, `--producers`, `--queue-size`, `--no-draw`, `--no-save`, `--profile`, `--profile-json`.

#### Mask mAP evaluation

With ground-truth YOLO-seg polygon labels on the board, compute mask mAP@0.5 and P/R/F1:

```bash
python3 run_instance_seg.py \
  --model yolov26n_seg \
  --dataset coco_instance_seg \
  --threads 2 \
  --accuracy \
  --labels-dir datasets/coco2017/labels/val2017
```

`seg_utils.py` provides `load_yolo_seg_labels` (polygon rasterization), `mask_iou_matrix`, and `compute_ap` (101-point interpolation) for this path. Omit `--labels-dir` to use the `board_labels` default from `dataset_config.py`.

## Important Notes & Limitations

### Semantic (`unet_res18`)

- **Palette placeholder**: `CITYSCAPES_COLORS` is a Cityscapes-style placeholder table.
- **No mIoU yet**: ground-truth mask loading and metric reporting are not implemented.
- **Dataset path semantics**: the runner currently uses `calib_path` for board-side images; this should be separated into explicit calibration/evaluation paths before treating segmentation as stable.
- **Model assets WIP**: `models/unet.py` and `models/unet.pt` must exist and match the registry entry.

### Instance (`yolov26n_seg`)

- **Training requirement**: must be trained from the DPU-friendly `configs/yolo26-seg_dpu.yaml` (no C2PSA/Attention; ReLU activations).
- **Mask quality vs. proto resolution**: masks are assembled at prototype resolution then upsampled; fine boundaries are limited by `num_protos` spatial size.
- **Evaluation gating**: `--accuracy` needs YOLO-seg polygon labels present on the board.

## Recommended next steps

1. Add a real UNet model file and checkpoint for the semantic path.
2. Add explicit segmentation image and mask paths to `dataset_config.py`.
3. Keep host preprocessing and board preprocessing numerically aligned.
4. Add mIoU / pixel-accuracy evaluation for semantic segmentation.
5. Validate compiled graphs with the Vitis AI Inspector before board deployment.

## Transfer behavior

`deploy.py` transfers only files needed for the active task. The runner and helper set depend on the `seg_instance` flag.

Semantic segmentation (`unet_res18`):

```text
unet_res18_kria.xmodel
run_segmentation.py
board_utils.py
model_config.py
dataset_config.py
board_config.py
```

Instance segmentation (`yolov26n_seg`):

```text
yolov26n_seg_kria.xmodel
run_instance_seg.py
seg_utils.py
detection_utils.py
board_utils.py
model_config.py
dataset_config.py
board_config.py
```
