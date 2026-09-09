# Classification

Classification code now lives under `kria_ai/classification/`. The files in this `scripts/` directory are temporary compatibility wrappers.

```bash
python -m kria_ai classification evaluate --model resnet18 --dataset intel_images
python -m kria_ai classification quantize --model resnet18 --dataset intel_images --mode calib
python -m kria_ai classification quantize --model resnet18 --dataset intel_images --mode test
python -m kria_ai classification compile --model resnet18
python3 -m kria_ai classification benchmark --model resnet18 --dataset intel_images --threads 2
```

Models and datasets are registered in `kria_ai/classification/config.py`. Host transforms, board preprocessing, and class order must remain aligned.
