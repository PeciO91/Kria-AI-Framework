# Experiments And Measurement Methodology

This document defines every reported metric, states the conditions under which the recorded measurements were taken, links the curated evidence, compares the KV260 runs against the T4 GPU baseline on the matched test set, lists what is and is not validated, and outlines planned experiments. Evidence files are curated excerpts of local run exports — fingerprints inside them identify the frozen experiment; they do not attest reproducible builds.

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
- **Historical KV260 accuracy evaluator** (produced the published KV260 accuracy numbers): a custom class-aware greedy matcher — IoU 0.5 for precision/recall, 101-recall-level AP at IoU 0.50:0.05:0.95, F1 as the harmonic mean of macro precision and macro recall. It was not the official COCO or Ultralytics evaluator, and it is not the current scorer described below.
- **Run statistics**: all numbers are single sequential passes — no repeated-run means or confidence intervals. Caches and thermal conditions were not reset between runs; thermal telemetry is absent.
- **Accuracy-run FPS**: online label matching runs in the pipeline during `--accuracy`, so its FPS includes matching overhead but excludes final AP aggregation; it is not used as a deployment-performance figure.
- **Exit status**: all four board processes completed successfully; per-run image/GT counts are recorded in the evidence file.

### Current Detection Evaluator

The current `--accuracy` code path uses a versioned NumPy scorer identified in reports as `ultralytics-8.4.150-numpy-v1` (no new runtime dependency — NumPy only). Its protocol:

- **AP**: upstream class-aware validator matching/deduplication semantics with a 101-point interpolated trapezoidal integration, averaged over classes that have ground truth. Box IoU uses a 1e-7 denominator epsilon.
- **Fixed-confidence P/R**: a separate class-agnostic geometric confusion-matrix matcher — strict score > configured run confidence floor and IoU > 0.5, with class errors counted as both FP and FN.
- **F1**: harmonic mean of macro precision and macro recall across all configured classes.
- **Determinism**: canonical filename/prediction-index ordering precedes the upstream global score sort, so per-worker results merge deterministically; the board aggregates per-worker confusion matrices. Equal-score tie ordering intentionally follows NumPy's default argsort and can differ across NumPy versions; exact tie reproduction also requires a matching NumPy version.
- **Reporting**: the text report prints the evaluator ID; the accuracy JSON records `evaluator`, `precision_recall_source`, `confusion_matrix`, `precision_recall_confidence_threshold` (the actual run floor — 0.3 for the operating-point run, 0.001 for the AP run), `precision_recall_matching_iou` (0.5), and `numpy_version` fields.
- The previous no-matrix aggregate API remains available and is explicitly labeled `ap_matches_at_input_confidence` — it is not the notebook confusion-matrix metric.

Reference: the parity fixture was generated from the exact downloaded Ultralytics 8.4.150 wheel (SHA256 `1af11709f0e525e72d8b1b71916c0081de3e63c49ab9ffdbffa995ae6a8c69de`); upstream reference functions were AST-isolated for CPU tests without installing the package. Parity tests cover matching, confusion-matrix counting, and AP integration, plus a 64-scene exact-source oracle — this establishes scorer parity with the pinned reference for axis-aligned detection boxes under the default non-Hungarian matcher, **not** full-model or hardware parity.

No fresh hardware accuracy runs have been performed with this scorer. Published KV260 accuracy retains legacy scoring; GPU accuracy uses Ultralytics 8.4.150. All historical numeric tables and evidence files are unchanged.

## Provenance Limits

- The recorded experiment's complete calibration/Fast Finetune recipe is not established by the available export — the README quick start is an example workflow, not that recipe. The user confirms 1,000 Fast Finetune images (2026-10-22), but neither the exact image manifest nor the full recipe is recovered from the export.
- The exact checkpoint, Ultralytics source revision, and test assets used historically are not bundled; source fingerprints in the evidence identify the frozen exports but do not substitute for the missing raw archive, and do not attest reproducible builds.
- The measured-XMODEL fingerprint matches the current local compiled artifact. A remote-to-host checksum audit of the exported report files was not performed.

## GPU–FPGA Comparison: Matched Test Set

Two-class FaceHuman YOLO26n and YOLO26s FP32 checkpoints — the corresponding unquantized baseline models — were exported and measured on an NVIDIA T4 and evaluated on the same matched 5,000-image FaceHuman test set used on the KV260. The GPU `s` scale is still **not** the COCO-80 `yolov26s` registry entry (two-class FaceHuman checkpoint), and no FPGA `s`-scale profile exists, so only the `n` scale is compared against the board below.

The comparison is descriptive, not a controlled isolation of hardware or quantization effects:

- GPU runs are **FP32**; the board artifact is **INT8 Fast Finetune**. GPU INT8 results are not provided yet (see Planned Experiments).
- Source metadata reports the GPU path using the **one2many head with NMS** while the Kria path exports **raw one2one tensors with top-k and no NMS**, and the published results were produced by different evaluators.
- GPU performance timing covers **500 images after 50 warmups**; FPGA runs cover all **5,000 images with no warmup exclusion**. Batch sizes and measurement windows differ.
- GPU power is **GPU-only**; board power is **SOM-total**.
- GPU preprocessing/transfer boundaries are not proven identical to the FPGA stage boundaries, and GPU "inference" includes a different head scope than raw DPU outputs — no per-stage speedup claims are made.

Within those limits, matched accuracy and profiling-stage tables follow.

### Accuracy Comparison (YOLO26n)

| YOLO26n configuration | Fast Finetune | Precision (%) | Recall (%) | F1 (%) | mAP50 (%) | mAP50:95 (%) | Status |
|---|---|---:|---:|---:|---:|---:|---|
| GPU (T4), FP32 | Not applied | 40.87 | 57.15 | 47.66 | 43.92 | 30.73 | Measured |
| GPU, INT8 | Not specified yet | Pending | Pending | Pending | Pending | Pending | Pending |
| FPGA (KV260), INT8 | With, 1,000 images | 46.62 | 32.63 | 38.39 | 38.79 | 25.99 | Measured |
| FPGA (KV260), INT8 | Without | Pending | Pending | Pending | Pending | Pending | Pending |

For each measured configuration, precision/recall/F1 are reported at confidence 0.3 and matching IoU 0.5; mAP comes from a separate confidence-0.001 evaluation. **Pending** means no verified measurement exists — not zero. GPU F1 is the harmonic mean of macro precision and macro recall, matching the board evaluator's definition. The published GPU and FPGA results used different evaluators and detection heads, so this is a descriptive deployment comparison, not an isolated quantization-loss measurement.

The published FPGA accuracy values were produced by the project’s legacy scorer (confidence-ordered class-aware matching with mean 101-point AP); the GPU values use Ultralytics 8.4.150. All numbers remain unchanged and do not retroactively change under the current evaluator described in [Current Detection Evaluator](#current-detection-evaluator). Fresh FPGA accuracy runs at confidence 0.3 and 0.001 — not yet performed — are needed before replacing the legacy FPGA rows.

Fast Finetune is the ablation between the two FPGA rows — both are INT8 quantization-calibrated; "Without" is not uncalibrated. The 1,000 Fast Finetune images are user-confirmed metadata (2026-10-22), not a count verified from the export.

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

### Profiling Stage Distribution

Broad stage-role comparison of mean/p50/p95 per sample — GPU batch 1 versus FPGA 1-runner and 2-runner performance runs. Stage names are normalized roles, not identical boundaries: `profiler`-merged raw samples make the pooled percentiles valid as stored, but parent/child stages overlap, thread totals are not wall-time shares or utilization, and the FPGA has no per-image end-to-end latency measurement (it cannot be recovered by summing overlapping means or as 1/FPS).

| Stage role | T4 FP32 batch 1: mean / p50 / p95 (ms/sample) | KV260 1 / 1: mean / p50 / p95 (ms/sample) | KV260 2 / 2: mean / p50 / p95 (ms/sample) |
|---|---|---|---|
| Image read / decode | 7.567 / 6.982 / 13.286 | 34.244 / 30.931 / 54.000 | 36.362 / 32.836 / 56.414 |
| Preprocessing | 1.902 / 1.857 / 2.405 | 11.544 / 11.493 / 15.178 | 18.817 / 17.852 / 29.323 |
| Model / accelerator execution | 11.550 / 9.405 / 12.322 | 12.254 / 12.190 / 12.466 | 14.082 / 12.631 / 21.940 |
| Postprocessing | 1.131 / 1.073 / 1.818 | 6.522 / 6.342 / 8.187 | 8.214 / 6.984 / 16.537 |

Stages with no GPU counterpart — input/output queue waits, runner setup, decoder sub-timers, top-k — are *not exported* rather than zero. Conversely, `draw_boxes`, `write`, and the writer queue waits were disabled in the recorded FPGA runs, so they are not measured (not zeros). FPGA `nms_or_topk` is top-k only here. FPGA sample-unit caveats: `runner_setup` is per runner, `consumer_dequeue_wait` counts include end-of-stream sentinels, `decode_ultra_threshold` is sampled per pyramid level, and `decode_ultra_dequant`/`class_score`/`box_decode` occur only for surviving levels.

Interpretation:

- CPU/data-path work matters on KV260: in the 1/1 run, image reading averages 34.24 ms and preprocessing 11.54 ms, compared with 12.25 ms for VART submit/wait. Image I/O and CPU stages are optimization candidates; these overlapping timers do not establish exclusive bottleneck shares.
- The 2/2 pipeline reaches 35.83 FPS versus 21.69 FPS for 1/1, even though mean VART submit/wait rises from 12.25 to 14.08 ms. Pipeline concurrency improves observed throughput; it does not make an individual DPU call faster.
- T4 FP32 batch-1 inference and KV260 submit/wait have means of 11.55 ms and 12.25/14.08 ms respectively, but cover different graph/runtime boundaries. Do not interpret similar means as equivalent hardware execution.
- FPGA per-image end-to-end latency was not measured. Neither a sum of overlapping means nor reciprocal application FPS reconstructs it.

### GPU Performance Stage Detail

All six GPU performance stages: batch-1 statistics, separately labeled batch-16 amortized-per-image statistics, and separate full-batch statistics over the 31 complete batches (the 4-image partial batch is excluded there).

| Reported GPU stage | Batch 1: mean / p50 / p95 (ms/image) | Batch 16 amortized: mean / p50 / p95 (ms/image) |
|---|---|---|
| Preprocessing | 1.9017 / 1.8569 / 2.4047 | 1.8450 / 1.8279 / 1.9270 |
| Model inference | 11.5503 / 9.4054 / 12.3215 | 2.9922 / 2.8837 / 3.0695 |
| Postprocessing | 1.1307 / 1.0730 / 1.8182 | 0.7028 / 0.7069 / 0.7811 |
| End-to-end excluding disk I/O | 15.5100 / 13.4748 / 16.4710 | 5.6357 / 5.5106 / 6.0562 |
| Disk / image load | 7.5666 / 6.9816 / 13.2855 | 8.8195 / 8.4912 / 10.4663 |
| End-to-end including disk I/O | 23.0767 / 20.7252 / 30.3421 | 14.4552 / 14.0754 / 16.1815 |

| Reported GPU stage | Full batch of 16: mean / p50 / p95 (ms/batch) |
|---|---|
| Preprocessing | 29.5003 / 29.2467 / 30.7777 |
| Model inference | 46.5755 / 46.1396 / 49.0719 |
| Postprocessing | 11.2414 / 11.3100 / 12.4020 |
| End-to-end excluding disk I/O | 88.8114 / 88.1694 / 94.3744 |
| Disk / image load | 140.8944 / 135.8587 / 167.2579 |
| End-to-end including disk I/O | 229.7058 / 225.2063 / 257.6020 |

### FPGA Profiling Stage Inventory

The full inventory of 20 distinct profiled stages across the four FPGA runs (performance 1/1, performance 2/2, accuracy at confidence 0.3, accuracy at confidence 0.001), with means and exact sample-unit labels — per image, per runner, per dequeue (including sentinels), or per pyramid level / surviving level. Sub-stage means are not summed into parent means, and sample units are not mixed. `P1`/`P2` are the two performance runs; `A0.3`/`A0.001` are the two accuracy runs.

| Recorded FPGA stage | One timing sample represents | Counts: P1 / P2 / A0.3 / A0.001 | P1 mean ms | P2 mean ms | A0.3 mean ms | A0.001 mean ms |
|---|---|---|---|---|---|---|
| `image_read` | image | 5000 / 5000 / 5000 / 5000 | 34.244 | 36.362 | 38.606 | 92.664 |
| `preprocess` | image | 5000 / 5000 / 5000 / 5000 | 11.544 | 18.817 | 21.133 | 84.525 |
| `input_enqueue_wait` | image enqueue | 5000 / 5000 / 5000 / 5000 | 0.161 | 0.173 | 0.161 | 0.161 |
| `producer_total` | image | 5000 / 5000 / 5000 / 5000 | 46.089 | 55.494 | 60.039 | 177.520 |
| `runner_setup` | runner | 1 / 2 / 2 / 2 | 217.040 | 273.295 | 280.535 | 183.270 |
| `consumer_dequeue_wait` | dequeue, including sentinel | 5001 / 5002 / 5002 / 5002 | 26.862 | 32.969 | 27.646 | 7.397 |
| `dpu` | image | 5000 / 5000 / 5000 / 5000 | 12.254 | 14.082 | 15.528 | 26.636 |
| `decode_ultra_layout` | image | 5000 / 5000 / 5000 / 5000 | 0.873 | 0.901 | 0.899 | 0.925 |
| `decode_ultra_threshold` | pyramid level | 15000 / 15000 / 15000 / 15000 | 0.632 | 0.849 | 1.004 | 6.458 |
| `decode_ultra_dequant` | surviving level | 5686 / 5686 / 5686 / 13428 | 0.163 | 0.245 | 0.317 | 3.199 |
| `decode_ultra_class_score` | surviving level | 5686 / 5686 / 5686 / 13428 | 0.416 | 0.504 | 0.521 | 1.189 |
| `decode_ultra_box_decode` | surviving level | 5686 / 5686 / 5686 / 13428 | 0.307 | 0.407 | 0.439 | 2.063 |
| `decode_ultra_concat` | image | 5000 / 5000 / 5000 / 5000 | 0.077 | 0.111 | 0.117 | 1.669 |
| `decode_total` | image | 5000 / 5000 / 5000 / 5000 | 4.603 | 5.793 | 6.456 | 41.108 |
| `nms_or_topk` | image; top-k only | 5000 / 5000 / 5000 / 5000 | 0.231 | 0.262 | 0.276 | 1.181 |
| `coord_scale` | image | 5000 / 5000 / 5000 / 5000 | 1.039 | 1.423 | 1.485 | 4.581 |
| `detection_postprocess` | image | 5000 / 5000 / 5000 / 5000 | 1.863 | 2.355 | 2.434 | 7.368 |
| `postprocess` | image | 5000 / 5000 / 5000 / 5000 | 6.522 | 8.214 | 8.967 | 48.581 |
| `accuracy` | image; labels + matching | — / — / 5000 / 5000 | — | — | 7.240 | 94.294 |
| `consumer_total` | image | 5000 / 5000 / 5000 / 5000 | 18.908 | 22.445 | 31.938 | 169.974 |

p50/p95 tails for the two performance runs:

| Recorded FPGA stage | P1 p50 / p95 (ms/sample) | P2 p50 / p95 (ms/sample) |
|---|---|---|
| `image_read` | 30.931 / 54.000 | 32.836 / 56.414 |
| `preprocess` | 11.493 / 15.178 | 17.852 / 29.323 |
| `input_enqueue_wait` | 0.155 / 0.200 | 0.164 / 0.236 |
| `producer_total` | 42.754 / 66.843 | 52.131 / 78.265 |
| `runner_setup` | 217.040 / 217.040 | 273.295 / 359.799 |
| `consumer_dequeue_wait` | 23.679 / 47.833 | 31.500 / 60.843 |
| `dpu` | 12.190 / 12.466 | 12.631 / 21.940 |
| `decode_ultra_layout` | 0.855 / 0.971 | 0.868 / 1.059 |
| `decode_ultra_threshold` | 0.419 / 1.174 | 0.447 / 1.595 |
| `decode_ultra_dequant` | 0.118 / 0.282 | 0.138 / 0.343 |
| `decode_ultra_class_score` | 0.408 / 0.520 | 0.441 / 0.690 |
| `decode_ultra_box_decode` | 0.285 / 0.389 | 0.306 / 0.496 |
| `decode_ultra_concat` | 0.080 / 0.114 | 0.089 / 0.147 |
| `decode_total` | 4.413 / 6.185 | 4.935 / 12.590 |
| `nms_or_topk` | 0.234 / 0.289 | 0.243 / 0.317 |
| `coord_scale` | 1.014 / 1.181 | 1.061 / 3.277 |
| `detection_postprocess` | 1.831 / 2.089 | 1.913 / 4.825 |
| `postprocess` | 6.342 / 8.187 | 6.984 / 16.537 |
| `consumer_total` | 18.700 / 20.688 | 20.178 / 34.803 |

### GPU Validator Stages

The GPU validator reports four stages from its own accuracy run at batch 16, confirmed by `yolo-eval.ipynb` — this is not the batch-1 performance experiment.

| Reported validator stage | Reported ms/image (separate accuracy run) |
|---|---|
| preprocess | 0.537518 |
| inference | 3.627588 |
| loss | 0.000470 |
| postprocess | 0.777018 |

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
5. GPU INT8 results for the same matched FaceHuman test set once T4 INT8 exports exist — this enables a precision-matched configuration but does not alone align backend, head, or evaluator.
6. Cross-task hardware validation: classification and instance-segmentation runs with the same reporting rigor.
7. Versioned releases of model/source/environment so experiments can be reproduced against pinned assets.
