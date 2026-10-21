# Classification

Classification code lives under `kria_ai/classification/`; the files in this `scripts/` directory are temporary compatibility wrappers. See [docs/architecture.md](../../docs/architecture.md) for the family/common split and [docs/experiments.md](../../docs/experiments.md) for what is and is not measured.

## Registered Assets

Models (registered in `kria_ai/classification/config.py`):

| CLI key | Backbone | Input |
|---|---|---|
| `resnet18` | ResNet-18 | 224x224 |
| `resnet50` | ResNet-50 | 224x224 |
| `mobilenet_v2` | MobileNetV2 | 224x224 |
| `mobilenet_v3` | MobileNetV3-Large | 224x224 |

Dataset: `intel_images` — 6 ordered classes `buildings, forest, glacier, mountain, sea, street`; host images under `data/intel_images/calibration_data`, board images expected under `datasets/intel_images/train_data` on the KV260.

Status: the deployment path is implemented end to end; comparative hardware validation/results are not supplied, and measured host-vs-board preprocessing parity is not claimed.

## Host Stages

Run from the repository root inside the Vitis AI 3.5 PyTorch environment. `evaluate` reports model loss/metrics on the host; it is not a board benchmark.

```bash
python -m kria_ai classification evaluate --model resnet18 --dataset intel_images --subset-len 32
python -m kria_ai classification inspect --model resnet18
python -m kria_ai classification quantize --model resnet18 --dataset intel_images --mode calib --subset-len 100
python -m kria_ai classification quantize --model resnet18 --dataset intel_images --mode test
python -m kria_ai classification compile --model resnet18
```

Useful options: `--checkpoint` (source checkpoint override), `--subset-len`, `--batch-size`, `--seed`, `--device`, `--build-root`; `--quant-mode` is an alias for `--mode`.

## Optimization (Experimental)

Classification supports `--method {iterative,one_step}` with `--mode {search,prune,finetune,all}` and `--ratio`. Keep the method consistent between the optimize command and the checkpoint you pass on — the output checkpoint is named after the method:

```bash
python -m kria_ai classification optimize \
  --model resnet18 --dataset intel_images --method iterative --mode all --ratio 0.2

python -m kria_ai classification quantize \
  --model resnet18 --dataset intel_images --mode calib \
  --checkpoint build/resnet18/optimizer_report/resnet18_iterative_r0.2_optimized.pt
```

Optimizer output lands in `build/<model-id>/optimizer_report/` (or `<build-root>/...`). Sparse checkpoints are rejected by quantization; pass the materialized slim checkpoint. AdaQuant is available for classification.

## Artifacts

```text
build/<model-id>/inspector_report/
build/<model-id>/optimizer_report/
build/<model-id>/quantize_result/<model-id>_int.xmodel
build/<model-id>/compiled/<model-id>_kria.xmodel
```

## Board Benchmark

Manual transfer only: copy the `kria_ai/` package, the compiled `<model-id>_kria.xmodel`, and the test images to the KV260, then run from the directory containing them:

```bash
python3 -m kria_ai classification benchmark \
  --model resnet18 --dataset intel_images \
  --xmodel resnet18_kria.xmodel \
  --dataset-root datasets/intel_images/train_data \
  --threads 2 --profile
```

Runner counts are validated against the KV260 limit (1–4). Power and compute metrics report as unavailable rather than crashing when the runtime lacks them.
