# Semantic Segmentation Deployment

Semantic segmentation support is currently **work in progress**. This section covers models whose `model_config.py` entry has `type: "segmentation"`.

Host-side stages are shared in `scripts/common/`; board-side execution uses `scripts/segmentation/run_segmentation.py`.

## Current model

| Model ID | Model name | Input | Status |
|---|---|---:|---|
| `unet_res18` | UNet_ResNet18 | `512x512` | Registered, but model file/checkpoint and evaluation flow are WIP. |

The registry entry is in `../../model_config.py` and currently expects:

```text
models/unet.py
models/unet.pt
```

## Dataset

The current segmentation dataset entry is `cityscapes_seg` in `../../dataset_config.py`.

Expected host calibration path:

```text
../../data/cityscapes/calibration_data/
```

The current runner reads images from the dataset `calib_path` and writes visual overlays. It does not yet compute mIoU or pixel accuracy.

## Host deployment

Run from the project root inside the Vitis AI PyTorch Docker environment:

```bash
python scripts/common/deploy.py \
  --model unet_res18 \
  --dataset cityscapes_seg \
  --subset 100 \
  --transfer none
```

When the model file, checkpoint, and calibration images are available, the common pipeline will run:

1. Inspector
2. Quantizer calibration
3. Quantizer export
4. Compiler
5. Optional transfer to the KV260

Compiled model output:

```text
build/unet_resnet18/compiled/unet_res18_kria.xmodel
```

## Board execution

After transfer, run on the KV260:

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

## Current limitations

- **Palette placeholder**: `CITYSCAPES_COLORS` is a Cityscapes-style placeholder table.
- **No mIoU yet**: ground-truth mask loading and metric reporting are not implemented.
- **Dataset path semantics**: the runner currently uses `calib_path` for board-side images; this should be separated into explicit calibration/evaluation paths before treating segmentation as stable.
- **Model assets WIP**: `models/unet.py` and `models/unet.pt` must exist and match the registry entry.

## Recommended next steps

1. Add a real UNet model file and checkpoint.
2. Add explicit segmentation image and mask paths to `dataset_config.py`.
3. Keep host preprocessing and board preprocessing numerically aligned.
4. Add mIoU / pixel-accuracy evaluation.
5. Validate the compiled graph with the Vitis AI Inspector before board deployment.

## Transfer behavior

`deploy.py` now transfers only files needed for the active task. For segmentation this includes:

```text
unet_res18_kria.xmodel
run_segmentation.py
board_utils.py
model_config.py
dataset_config.py
board_config.py
```
