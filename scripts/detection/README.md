# YOLOv26 Object Detection

Detection code lives under `kria_ai/yolov26/`; the files in this `scripts/` directory are temporary compatibility wrappers. YOLOv5 is outside the current architecture. See [docs/architecture.md](../../docs/architecture.md) for module responsibilities and [docs/experiments.md](../../docs/experiments.md) for measured results and metric definitions.

## Registered Models

| CLI key | Artifact ID | Configured checkpoint | Classes | Input | Status |
|---|---|---|---|---|---|
| `yolov26n_FaceHuman` | `yolov26n_facehuman` | `models/yolov26n_2classes.pt` | 2 (`Person`, `Human face`) | 640x640 | Hardware-demonstrated reference; measured INT8 Fast Finetune artifact. |
| `yolov26s` | `yolov26s` | `models/yolo26s.pt` | 80 (COCO) | 640x640 | Implemented; hardware validation pending. |

Checkpoints, the Ultralytics checkout at `models/ultralytics-main`, and datasets are user-supplied. The configured FaceHuman checkpoint path is `models/yolov26n_2classes.pt`; if your checkpoint has a different filename, pass it explicitly with `--checkpoint` — do not rename or assume a similarly named local file is equivalent. `--checkpoint` may be supplied consistently to `inspect` and both `quantize` stages.

The CLI key is mixed-case (`--model yolov26n_FaceHuman`) while generated artifact paths use the lowercase ID (`build*/yolov26n_facehuman/`, `yolov26n_facehuman_kria.xmodel`).

## Datasets

| CLI key | Host roots | Labels | Classes |
|---|---|---|---|
| `openimagesv7` | `data/openimagesv7/train/images`, `data/openimagesv7/validation/images` | `data/openimagesv7/train/labels`, `data/openimagesv7/validation/labels` | `Person`, `Human face` |
| `coco` | see caveat | `data/coco2017/labels/{train2017,val2017}` | COCO-80 |

**COCO path caveat.** The current `coco` detection registry points its *image* roots at `data/openimagesv7/train2017` and `data/openimagesv7/val2017` while its *label* roots point at `data/coco2017/labels/...`. Those configured image roots are not verified as a real COCO layout and are absent locally. Check and correct the family dataset metadata in `kria_ai/yolov26/detection/config.py` before a real COCO deployment; the COCO commands below are parser examples, not a verified end-to-end recipe.

## Graph And Output Contract

The compiled graph exports six split one2one tensors; anchor-free decode and top-k selection run on the ARM CPU with no NMS. Loading validates the checkpoint head against configured metadata exactly (COCO model stays nc=80; FaceHuman is nc=2).

For `yolov26n_FaceHuman`, two project adaptations target the DPUCZDX8G: LeakyReLU(13/128) on regular Conv with ReLU on depthwise Conv (fused depthwise LeakyReLU is unsupported), and a weight-equivalent C3k2 channel-split rewrite that avoids `nndct_strided_slice`. Its head layers are listed in `prune_excludes` for the optimizer.

## Host Stages

Inside the Vitis AI 3.5 PyTorch environment, from the repository root:

```bash
python -m kria_ai yolov26 detection inspect  --model yolov26n_FaceHuman
python -m kria_ai yolov26 detection quantize --model yolov26n_FaceHuman --dataset openimagesv7 --mode calib --subset-len 100
python -m kria_ai yolov26 detection quantize --model yolov26n_FaceHuman --dataset openimagesv7 --mode test
python -m kria_ai yolov26 detection compile  --model yolov26n_FaceHuman
```

Fast Finetune is experimental and must appear in both calibration and export when used; keep a dedicated build root so baseline artifacts under `build/` are preserved:

```bash
python -m kria_ai yolov26 detection quantize --model yolov26n_FaceHuman --dataset openimagesv7 --mode calib --fast-ft --build-root build_fast_ft
python -m kria_ai yolov26 detection quantize --model yolov26n_FaceHuman --dataset openimagesv7 --mode test --fast-ft --build-root build_fast_ft
python -m kria_ai yolov26 detection compile  --model yolov26n_FaceHuman --build-root build_fast_ft
```

For `yolov26s`, substitute `--model yolov26s --dataset coco` (subject to the COCO caveat above).

**Three different "evaluations"** — do not conflate them:

- Host `python -m kria_ai yolov26 detection evaluate` reports **loss**, not mAP.
- Quantizer `--mode test` performs a **one-image forward/export** of the INT model — it is not a test-set accuracy run.
- Board `--accuracy` runs the **mAP evaluator** described below.

## Board Performance Benchmark

Manual transfer only: copy the `kria_ai/` package, the compiled XMODEL, and test data to the KV260. `--dataset-root` is a **flat** image directory — it is not scanned recursively — so pass the `images/test` path explicitly; for accuracy also pass `--labels-dir` ending in `labels/test`.

Measured 2-runner / 2-producer performance run (from a directory containing the package and XMODEL, with data at `datasets/openimagesv7/fpga_test_images/{images/test,labels/test}`):

```bash
python3 -m kria_ai yolov26 detection benchmark \
  --model yolov26n_FaceHuman --dataset openimagesv7 \
  --xmodel yolov26n_facehuman_kria.xmodel \
  --dataset-root datasets/openimagesv7/fpga_test_images/images/test \
  --threads 2 --producers 2 --queue-size 40 --confidence-threshold 0.3 \
  --no-draw --no-save --profile --profile-json \
  --output-report reports/facehuman_performance_t2_p2.txt
```

The 1/1 variant uses `--threads 1 --producers 1` and a distinct report path such as `reports/facehuman_performance_t1_p1.txt`. `--profile-json` writes a sibling `<report-stem>_profile.json` next to the `--output-report` text report.

## Board Accuracy Runs

Accuracy evaluation needs labels and uses a different confidence floor per operating point. The runs below mirror the recorded experiment's settings (2 runners / 2 producers, queue 40, drawing and saving disabled, profiling on) — note the board defaults differ (`--producers 4`, draw/save enabled), so do not rely on defaults:

```bash
# Operating point (confidence 0.3): precision / recall / F1
python3 -m kria_ai yolov26 detection benchmark \
  --model yolov26n_FaceHuman --dataset openimagesv7 \
  --xmodel yolov26n_facehuman_kria.xmodel \
  --dataset-root datasets/openimagesv7/fpga_test_images/images/test \
  --accuracy --labels-dir datasets/openimagesv7/fpga_test_images/labels/test \
  --confidence-threshold 0.3 \
  --threads 2 --producers 2 --queue-size 40 \
  --no-draw --no-save --profile --profile-json \
  --output-report reports/facehuman_accuracy_conf030.txt

# AP evaluation (confidence 0.001): mAP@0.5 and mAP@0.5:0.95
python3 -m kria_ai yolov26 detection benchmark \
  --model yolov26n_FaceHuman --dataset openimagesv7 \
  --xmodel yolov26n_facehuman_kria.xmodel \
  --dataset-root datasets/openimagesv7/fpga_test_images/images/test \
  --accuracy --labels-dir datasets/openimagesv7/fpga_test_images/labels/test \
  --confidence-threshold 0.001 \
  --threads 2 --producers 2 --queue-size 40 \
  --no-draw --no-save --profile --profile-json \
  --output-report reports/facehuman_accuracy_conf0001.txt
```

Accuracy mode defaults to a 0.001 confidence threshold so AP is evaluated across the score curve; the two runs above use distinct report paths.

These commands now execute the current versioned NumPy scorer `ultralytics-8.4.150-numpy-v1` (NumPy only, no new dependency). AP uses upstream class-aware validator matching/deduplication with 101-point interpolated trapezoidal integration; fixed-confidence P/R uses a separate class-agnostic geometric confusion-matrix matcher (strict score > run floor, IoU > 0.5, class errors counted FP/FN); F1 is the harmonic mean of macro P/R across all configured classes; board runs aggregate per-worker confusion matrices. The text report prints per-class precision/recall/AP and the evaluator ID; the sibling `<report-stem>_profile.json` records F1 plus `evaluator`, `precision_recall_source`, and `confusion_matrix`. **Running these commands does not reproduce the historical KV260 accuracy values** in the README/experiments tables — those FPGA numbers were produced by the previous confidence-ordered class-aware greedy matcher (the GPU values are Ultralytics 8.4.150, not the legacy scorer) and remain historical until fresh accuracy runs are performed.

## Profiling Semantics

`--profile` prints aggregate and per-worker stage tables; `--profile-json` emits a sibling JSON retaining the original summary fields plus per-worker stages, producer/queue/output settings, board and artifact paths, and SOM power/energy with availability and idle-fallback information. Stages include image read/JPEG decode, RGB conversion plus preprocessing (`preprocess` excludes image reading), input/output queue waits, runner setup, CPU decode, top-k, coordinate scaling, drawing, and image writing.

Interpret carefully: parent/child stages overlap, totals accumulate elapsed across threads, and `wall_pct` may exceed 100% — it is not utilization or an exclusive bottleneck share. Decoder threshold/dequantization/scoring/box samples are per pyramid level, not per image. Application FPS includes worker startup, image loading, enabled accuracy/drawing/saving and pipeline drain; model/OpenCV initialization is excluded and no warmup images are discarded. `runner_setup` is reported separately; the `dpu` stage is VART submit/wait, not pure hardware compute or separate DMA/cache sync. Queue-wait counts include end-of-stream sentinels; per-image totals do not.

OpenCV is initialized and checked on the main thread before workers start. Producer, DPU-consumer, and writer exceptions cancel the pipeline and exit nonzero; no success report is written after a worker failure, and a run that processes zero images is an error.

## Optimization (Experimental)

```bash
python -m kria_ai yolov26 detection optimize --model yolov26s --dataset coco --method one_step --mode all --ratio 0.2
```

The optimizer writes a slim checkpoint plus manifest under `build/<model-id>/optimizer_report/`; pass it to a later stage explicitly via `--checkpoint`. Sparse checkpoints are rejected by quantization.
