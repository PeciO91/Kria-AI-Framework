# YOLOv26 Instance Segmentation

Segmentation code lives under `kria_ai/yolov26/segmentation/`; the files in this `scripts/` directory are temporary compatibility wrappers. Semantic segmentation is outside the current architecture. See [docs/architecture.md](../../docs/architecture.md) and [docs/experiments.md](../../docs/experiments.md) for context.

**Status: experimental / work in progress.** The instance-mask path is implemented — the compiled graph exports box, class, and mask-coefficient tensors for three pyramid levels plus mask prototypes, and CPU post-processing assembles masks — but a current validated board benchmark is absent.

## Registered Assets

- Model `yolov26n_seg`: configured checkpoint `models/yolov26n-seg.pt`, architecture `configs/yolov26/segmentation/yolov26n_seg_dpu.yaml`, COCO-80 classes, 640x640 input, 32 mask coefficients, mask threshold 0.5.
- Dataset `coco`: host images `data/coco2017/{train2017,val2017}`, YOLO polygon labels `data/coco2017/labels/{train2017,val2017}`; the loader can also read COCO polygon JSON under `data/coco2017/annotations/`. RLE-only annotations are not handled. Board roots: `datasets/coco2017` with labels at `datasets/coco2017/labels/val2017`.

The segmentation registry is separate from the detection registry so box and polygon label semantics cannot be mixed accidentally. Polygon rows need at least three points; they are not reinterpreted as boxes.

## Host Stages

Inside the Vitis AI 3.5 PyTorch environment, from the repository root:

```bash
python -m kria_ai yolov26 segmentation inspect  --model yolov26n_seg
python -m kria_ai yolov26 segmentation quantize --model yolov26n_seg --dataset coco --mode calib --subset-len 100
python -m kria_ai yolov26 segmentation quantize --model yolov26n_seg --dataset coco --mode test
python -m kria_ai yolov26 segmentation compile  --model yolov26n_seg
```

As with detection, quantizer `--mode test` is a one-image forward/export, not a test-set accuracy run. Optional optimization:

```bash
python -m kria_ai yolov26 segmentation optimize --model yolov26n_seg --dataset coco --method one_step --mode all --ratio 0.2
```

## Board Benchmark

Manual transfer only: copy the `kria_ai/` package, the compiled `yolov26n_seg_kria.xmodel`, and test data to the KV260.

```bash
python3 -m kria_ai yolov26 segmentation benchmark \
  --model yolov26n_seg --dataset coco \
  --xmodel yolov26n_seg_kria.xmodel \
  --dataset-root datasets/coco2017/val2017 \
  --threads 2 --producers 2 --profile
```

The explicit `--dataset-root datasets/coco2017/val2017` above overrides the registered board default `datasets/coco2017` with the user-assembled flat `val2017` image directory; matching polygon labels stay at `datasets/coco2017/labels/val2017` (used with `--accuracy --labels-dir`).

- `--accuracy --labels-dir ...` computes the current greedy **mask mAP@0.5** metric from YOLO polygon labels — a project metric, not official COCO mask AP.
- `--video <file> [--output-video <file>]` runs ordered file-video inference and requires `--threads 1 --producers 1`; image mode uses the producer/runner/writer pipeline.
- Runner counts are validated against the KV260 range 1–4.
