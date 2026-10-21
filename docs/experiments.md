# Experiments And Measurement Methodology

This document defines every reported metric, states the conditions under which the recorded measurements were taken, links the curated evidence, gives the T4 GPU context, lists what is and is not validated, and outlines planned experiments. Evidence files are curated excerpts of local run exports — fingerprints inside them identify the frozen experiment; they do not attest reproducible builds.

## Evidence Files

| File | Contents |
|---|---|
| [evidence/kv260-facehuman-2026-10-21.json](evidence/kv260-facehuman-2026-10-21.json) | Curated KV260 experiment excerpt: model/dataset/board/environment, performance and accuracy records (original metric keys and precision), run settings, source-report fingerprints, limitations. |
| [evidence/t4-facehuman-context.json](evidence/t4-facehuman-context.json) | Contextual T4 GPU exports of two-class FaceHuman YOLO26n and YOLO26s FP32 checkpoints (reports keyed by model scale): execution metadata, settings, accuracy, performance/power. |

The full raw run directories remain local-only (under ignored `results/`); only these sanitized excerpts are committed. No measurement was regenerated for this documentation.

## KV260 Experiment

Experiment `full_test_20261021T092551Z_DPXti26q`, completed 2026-10-21:

- **Model**: `yolov26n_FaceHuman` — YOLO26n, 2 ordered classes `Person`, `Human face`, 640x640 input, compiled as INT8 with the Fast Finetune quantizer path (artifact `yolov26n_facehuman_kria.xmodel`).
- **Board**: KV260, `DPUCZDX8G_ISA1_B4096` at 300 MHz (configured metadata). Recorded environment: Python 3.9.9, NumPy 1.21.2, OpenCV 4.5.2, aarch64.
- **Dataset**: 5,000 Open Images V7 test images, 14,847 ground-truth boxes (9,804 Person / 5,043 Human face).
- **Performance runs**: `performance_t1_p1_conf030` and `performance_t2_p2_conf030` — confidence threshold 0.3, queue size 40, accuracy/drawing/saving disabled, profiling enabled, 5,000 DPU jobs, runtime power sampling, no warmup exclusion.
- **Accuracy runs**: an operating-point evaluation at confidence 0.3 and a separate AP evaluation at confidence 0.001.

### Results

| DPU runners / producers | Application FPS | Mean VART submit/wait (ms/image) | SOM-total power (W) | Energy (mJ/image) |
|---|---:|---:|---:|---:|
| 1 / 1 | 21.69 | 12.25 | 6.62 | 305.19 |
| 2 / 2 | 35.83 | 14.08 | 6.98 | 194.69 |

| Evaluation | Confidence floor | Precision (%) | Recall (%) | F1 (%) | mAP50 (%) | mAP50:95 (%) |
|---|---:|---:|---:|---:|---:|---:|
| Operating point | 0.3 | 46.62 | 32.63 | 38.39 | -- | -- |
| AP evaluation | 0.001 | -- | -- | -- | 38.79 | 25.99 |

`--` means intentionally not shown in this table, not zero: each evaluation is reported at the confidence floor that defines it. The full source metrics — including the confidence-0.3 truncated AP — are retained in the evidence file; the truncated AP is not published as the mAP result.

## Metric Definitions And Caveats

- **Application FPS**: images per second across the whole run, including image loading, worker startup, and pipeline drain; model loading and main-thread OpenCV initialization are excluded; no warmup images are discarded.
- **VART submit/wait latency**: mean `execute_async` + `wait` per image. It is not end-to-end latency, not pure DPU kernel time, and not reciprocal throughput.
- **SOM-total power**: sampled power of the KV260 SOM. It is neither carrier-board/wall-socket power nor DPU-only power.
- **Energy per image**: sampled average power divided by application FPS.
- **Profiling stages**: parent/child/worker stages overlap and totals accumulate elapsed time across threads, so `wall_pct` can exceed 100%; it is not CPU or DPU utilization or an exclusive bottleneck share. The `dpu` profile stage is the VART submit/wait measure, not separate DMA/cache synchronization. Queue-wait counts include end-of-stream sentinels; per-image totals do not.
- **Accuracy evaluator**: a custom class-aware greedy matcher — IoU 0.5 for precision/recall, 101-recall-level AP at IoU 0.50:0.05:0.95, F1 as the harmonic mean of macro precision and macro recall. It is not the official COCO or Ultralytics evaluator.
- **Run statistics**: all numbers are single sequential passes — no repeated-run means or confidence intervals. Caches and thermal conditions were not reset between runs; thermal telemetry is absent.
- **Accuracy-run FPS**: online label matching runs in the pipeline during `--accuracy`, so its FPS includes matching overhead but excludes final AP aggregation; it is not used as a deployment-performance figure.
- **Exit status**: all four board processes completed successfully; per-run image/GT counts are recorded in the evidence file.

## Provenance Limits

- The recorded experiment's complete calibration/Fast Finetune recipe is not established by the available export — the README quick start is an example workflow, not that recipe.
- The exact checkpoint, Ultralytics source revision, and test assets used historically are not bundled; source fingerprints in the evidence identify the frozen exports but do not substitute for the missing raw archive, and do not attest reproducible builds.
- The measured-XMODEL fingerprint matches the current local compiled artifact. A remote-to-host checksum audit of the exported report files was not performed.

## T4 GPU Context (Not Comparable)

Two-class FaceHuman YOLO26n and YOLO26s FP32 checkpoints were exported and measured on an NVIDIA T4 for calibration/benchmark context. Note the GPU `s` scale is **not** the COCO-80 `yolov26s` registry entry — these are two-class FaceHuman checkpoints; whether they share identical original weights with the Kria artifact is not established, and the full build recipe is not recorded. These results are **contextual only**: they use FP32 (not INT8), a different evaluator, the one2many head with NMS (the Kria path uses one2one/top-k without NMS), GPU-only power (not SOM-total), different batch sizes, 5,000-image accuracy evaluation but only 500 timed performance images with 50 warmups, and one GPU despite two being visible. Do not derive quantization loss, speedup, or efficiency claims from these rows.

**FP32 accuracy** (5,000 images, same ordered classes):

| Model scale | mAP50 (%) | mAP50:95 (%) |
|---|---:|---:|
| YOLO26n | 43.92 | 30.73 |
| YOLO26s | 46.67 | 33.11 |

**FP32 performance** (500 timed images, 50 warmups; power is GPU-only; energy is compute-window per image):

| Model scale | Batch | FPS excl. disk I/O | FPS incl. disk I/O | Mean inference (ms) | E2E excl. I/O (ms/image) | GPU power (W) | Energy (J/image) |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLO26n | 1 | 64.474 | 43.334 | 11.5503 | 15.5100 | 37.551 | 0.5824 |
| YOLO26n | 16 | 177.442 | 69.179 | -- | 5.6357 (amortized) | 65.525 | 0.3693 |
| YOLO26s | 1 | 75.208 | 48.512 | 9.4596 | 13.2965 | 66.879 | 0.8893 |
| YOLO26s | 16 | 92.586 | 52.570 | -- | 10.8008 (amortized) | 64.526 | 0.6969 |

"Inference" is model-forward timing within the compute window; "E2E excl. I/O" adds CPU-side work excluding disk reads; the batch-16 E2E is amortized per image. CUDA synchronization and raw NVML sample alignment are not independently proven from the exported JSON — see `limitations` in the evidence file.

## Validation Matrix

| Claim | Status |
|---|---|
| `yolov26n_FaceHuman` INT8 Fast Finetune on KV260 | Reported: measured artifact and board run described above. |
| `yolov26s` (COCO-80) on KV260 | Implemented; hardware validation pending. Also affected by the COCO dataset image-root caveat in the [dataset guide](../DATASET_PREPARATION.md). |
| Classification models on KV260 | Implemented deployment path; comparative hardware validation/results not supplied. Host preprocessing vs board preprocessing parity is not promised as measured. |
| `yolov26n_seg` instance masks | CPU mask assembly implemented; current validated benchmark absent; experimental/WIP. |
| Pruning / optimizer flows | Implemented, requires the Vitis AI environment, experimental; no measured quality/throughput trade-off published. |
| Documentation commands | Parser/static-checked on the host only; not a fresh end-to-end board rerun. |

## Planned Experiments

Framed as future work, not current claims:

1. Controlled FP32-vs-INT8 comparison for `yolov26n_FaceHuman` using the same head, evaluator, and image set.
2. Repeated thermal- and cache-aware trials to produce means/confidence intervals for FPS, latency, power, and energy.
3. Baseline-vs-Fast-Finetune ablation quantifying what the flag contributes.
4. Pruning trade-off study (ratio vs accuracy vs measured FPS) once optimizer artifacts are validated on hardware.
5. Cross-task hardware validation: classification and instance-segmentation runs with the same reporting rigor.
6. Versioned releases of model/source/environment so experiments can be reproduced against pinned assets.
