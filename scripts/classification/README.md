# Classification Deployment

Classification is the stable image-classification path for the Kria Vitis AI Universal Deployment Pipeline.

This section covers models whose `model_config.py` entry has `type: "classification"`. Host-side deployment uses the shared scripts in `scripts/common/`; board-side execution uses `scripts/classification/run_inference.py`.

## Supported models

| Model ID | Model name | Input | Notes |
|---|---:|---:|---|
| `resnet18` | ResNet18 | `224x224` | Default active model; validated in the reorganized layout. |
| `resnet50` | ResNet50 | `224x224` | Torchvision backbone. |
| `mobilenet_v2` | MobileNetV2 | `224x224` | Uses `classifier` as final layer. |
| `mobilenet_v3` | MobileNetV3-Large | `224x224` | Torchvision backbone. |
| `mobilenet_v4_hybrid` | MobileNetV4_Hybrid | `384x384` | Custom model file in `models/mobilenet_v4_hybrid.py`. |
| `inception_v3` | InceptionV3 | `299x299` | Uses Inception input resolution. |

The central model registry is `../../model_config.py`. Keep model metadata there instead of hardcoding model-specific behavior in scripts.

## Dataset

The standard classification dataset is `intel_images` from `../../dataset_config.py`.

Expected host calibration path:

```text
../../data/intel_images/calibration_data/
```

Expected board path:

```text
/home/root/datasets/intel_images/train_data/<class-name>/*.jpg
```

Classes are ordered in `dataset_config.py`:

```text
buildings, forest, glacier, mountain, sea, street
```

The model checkpoint and dataset must agree on class count. For example, the current `resnet18.pt` checkpoint has 6 output classes and must be deployed with `--dataset intel_images`, not the default `coco_detection` dataset.

## Host deployment

Run from the project root inside the Vitis AI PyTorch Docker environment:

```bash
python scripts/common/deploy.py \
  --model resnet18 \
  --dataset intel_images \
  --subset 100 \
  --transfer none
```

Use `--transfer none` for local validation. To copy the compiled xmodel and required board files to the Kria board, omit `--transfer none` or select a transfer method:

```bash
python scripts/common/deploy.py \
  --model resnet18 \
  --dataset intel_images \
  --subset 100 \
  --transfer scp
```

The deployment pipeline runs:

1. `scripts/common/run_inspector.py`
2. `scripts/common/run_quantizer.py --quant_mode calib`
3. `scripts/common/run_quantizer.py --quant_mode test`
4. `scripts/common/run_compiler.py`
5. Optional transfer to the board

Compiled model output:

```text
build/resnet18/compiled/resnet18_kria.xmodel
```

## Board execution

After transfer, run on the KV260:

```bash
python3 run_inference.py \
  --model resnet18 \
  --dataset intel_images \
  --threads 2
```

The runner computes:

- **Top-1 / Top-5 accuracy** against class-folder ground truth.
- **Application FPS**.
- **Average DPU latency**.
- **Power and energy per frame** where board power telemetry is available.
- **DPU duty cycle and compute efficiency** using `board_config.py` metadata.

## Validated smoke test

The reorganized layout was validated with:

```bash
python scripts/common/deploy.py \
  --model resnet18 \
  --dataset intel_images \
  --subset 32 \
  --transfer none
```

Result:

```text
PIPELINE COMPLETE
Final Model: build/resnet18/compiled/resnet18_kria.xmodel
```

## Notes

- Host-side common scripts live in `../common/`.
- Board-side shared helpers are copied from `../common/board_utils.py`.
- Keep normalization in `dataset_config.py` aligned with training preprocessing and board preprocessing.
