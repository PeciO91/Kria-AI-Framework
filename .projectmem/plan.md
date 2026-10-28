# Project — plan

> Editable **intent** file: ideas + plans — what we *mean to do*.
> This is NOT the event log. `events.jsonl` -> `summary.md` records what
> *happened*; this file records what we *intend*. The AI reads it at
> session start and edits it directly (like `PROJECT_MAP.md`): add ideas
> and plans, check items off, move done work down to Shipped. Plans are
> never logged as events.

## Ideas
_Loose thoughts, not yet committed to._

## Active plans
_What we're working toward now. Use `- [ ]` / `- [x]` checklists._

## Next
_Queued, but not started._

- [ ] COCO detection image-root mismatch (#0008): documented, not fixed.
  Any registry correction and deployment validation need a separately approved
  code/hardware scope; the README redesign did not change dataset behavior.

- [ ] Deferred export-integrity check: the user manually copied the full Kria
  results; no SHA256SUMS is present. Local completion/profile validation passed,
  but remote-to-host file checksum comparison was not performed. Do this only
  if requested; do not restart tests or continuous agent monitoring.

## Someday / maybe

## Shipped
_Move completed plans here so the top stays about the future._

### Ultralytics 8.4.150 evaluation compatibility (#0011)

- [x] Audit notebook protocol and existing evaluator; identify usable
  existing NumPy/Torch host interpreters.
- [x] Acquire exact Ultralytics 8.4.150 reference: wheel downloaded to /tmp
  scratch (SHA256 1af11709f0e525e72d8b1b71916c0081de3e63c49ab9ffdbffa995ae6a8c69de),
  extracted read-only; no installs or environment changes.
- [x] Lead: author matching, confusion-matrix counting and AP parity tests
  first, port evaluation math to NumPy, and verify against pinned reference;
  keep historical reports unchanged. Scorer ultralytics-8.4.150-numpy-v1:
  12 parity tests green, 64-scene exact-source oracle green.
- [x] Verify narrow host regressions and one final suite, review full diff,
  label new evaluator in current documentation/report metadata; no hardware
  run, install, commit or push.
  Outcome: full suite 98 total = 90 pass, 6 skip, 2 known missing-cv2 errors
  (test_leaky_conv_relu_dw_policy and
  test_split_c3k2_cv1_is_numerically_equivalent); compileall, benchmark
  --help, links/anchors and diff checks pass. Accuracy metadata now records
  actual run confidence floor, matching IoU 0.5 and NumPy version; published
  KV260 accuracy stays labeled historical legacy scoring. No hardware run,
  install, commit or push performed.

### Python / C++ DPU latency investigation (#0010)

- [x] User confirmed the camera runs approximately 11 ms; the earlier 3 ms
  observation was mistaken. Controlled C++/Python API means were 10.836/11.589 ms;
  no production fix is needed.
- [x] Deleted the local `tmp/` tree and diagnostic host log at the user's
  request. Investigation history remains in Projectmem; board files unchanged.

### Evidence-led README redesign

- [x] Audit implementation, registries, guides, ignored assets and frozen
  KV260/T4 exports; approve balanced research-engineering presentation.
- [x] Lead: curate sanitized evidence; all eight source hashes and four board
  run copy invariants match, with original precision and no new measurements.
- [x] Delegate: rewrite README, preserve useful details in corrected task,
  dataset, architecture and experiment guides; allowlist only intended assets.
- [x] Validate 17 CLI help surfaces, 11 host contract tests, JSON, links,
  flags, 16 Bash blocks, ignore rules and diff whitespace.
- [x] Lead: verify all 14 numeric table rows, review complete diff and
  consolidated corrections; independent final review and scoped re-review pass.
- [x] Close documentation issue #0007 and prepare the requested single local
  documentation commit. No source/config changes, installs, network, hardware
  rerun, push or PR. Mermaid rendering was not observed; only syntax/structure
  checked. Exact historical assets/build recipe and broader experiments remain
  explicitly limited in the public documentation.

### YOLO26n accuracy and profiling comparison documentation

- [x] Approved scope (user-settled revision): README "Experimental Results"
  section with four-configuration Accuracy Comparison (YOLO26n) first, then
  Profiling Comparison with frozen stage-role table and interpretation;
  throughput/energy moved after the comparison tables.
- [x] docs/experiments.md: ACCURACY_TABLE before GPU n/s contextual tables;
  ms/sample units added to broad-distribution and GPU amortized headings;
  interpretation bullets, retained stage inventory/percentiles/validator.
- [x] Fast Finetune = 1,000 images recorded as user-confirmed (2026-10-22)
  metadata in docs/evidence/kv260-facehuman-2026-10-21.json model object;
  both FPGA rows are INT8-calibrated (no-FastFT is the ablation, not
  uncalibrated); Pending = no verified measurement, not zero.
- [x] Validation: links/anchors, JSON parse, diff whitespace, doc visibility;
  log /tmp/yolo26n-comparison-doc-checks.log; patch
  /tmp/yolo26n-comparison-implementation.patch.

### Full Kria Fast Finetune test and export

- [x] Lead: verify current source equals smoke-tested board package, unchanged
  XMODEL, available resources, no active benchmark, and all 5000 images/labels.
  Image-list hash and class counts (9804 Person, 5043 Human face) match Kaggle.
- [x] Lead: author the bounded full-run harness and artifact validation.
  Execute sequential full-dataset runs: performance at confidence 0.3 with
  runners/producers 1/1 and 2/2; accuracy at 0.3 and mAP at 0.001 using 2/2.
  Disable drawing/saving throughout and accuracy during performance runs.
  No warmup exclusion or repeated-run statistics; preserve code/model/data.
- [x] Delegate: mechanically launch the exact harness on the approved board,
  retain progress/logs, and transfer only validated reports/metadata to a new
  results/kria subfolder. No installs, firmware operations, source edits,
  overwritten results or annotated-image transfer.
  Monitoring was interrupted at the user's request; the detached board job
  completed, and the user performed the final SCP transfer.
- [x] Lead: inspect exported completion records, all four raw profile counts,
  process exit codes and validation markers, then record the actual outcome.
  File checksum verification is explicitly deferred above, not claimed done.
  Reuse frozen results; do not rerun derivations for presentation.
  Completed 2026-10-21T09:43:13Z: all four runs processed 5000 images, exit 0.
  Export: results/kria/export_P40HUict/full_test_20261021T092551Z_DPXti26q/
  (31 files, including summary.json; no images transferred).

### Detection startup and worker failure fix (#0006)

- [x] Delegate: add bounded host regressions, observe RED, then initialize
  OpenCV in main after validation/before runtime or thread startup.
- [x] Delegate: port segmentation-style stop-event/error handling into
  detection, including cancellation-aware queue operations, worker cleanup,
  failed imwrite checks and rejecting zero processed images. Preserve all
  lead-authored timing scopes, stage names, detection and accuracy algorithms.
- [x] Lead: review the full patch and measurement boundaries; delegate narrow
  host checks and one final suite gate. No installs or unrelated refactors.
  Lifecycle 9/9, profiling 9/9 and CLI 6/6 pass; full host suite has 78 pass,
  6 known OpenCV skips and 2 known missing-OpenCV Ultralytics errors.
  The successful-frame consumer_total boundary is preserved by a new
  deterministic RED-to-GREEN progress-update regression. Independent review
  found no blocking issues; native-call hangs remain outside queue cancellation.
- [x] Lead: author the exact remote recipe; delegate a plain-CLI smoke on
  the frozen 10-image subset with the existing Fast Finetune XMODEL.
  Use new board scratch, preserve earlier evidence and existing deployments;
  no preload wrapper, full dataset benchmark, firmware or artifact return.
- [x] Lead: inspect remote evidence, record confirmation and close #0006 only
  after host error-path checks and the unmodified CLI smoke pass.
  Normal CLI passed both 1runner/1producer and 2runner/2producer smokes:
  each 10 frames, 10 DPU jobs, 15 detections and 10 valid annotated outputs.
  Evidence stays on the board at
  /home/root/yolov26n_fastft_test/smoke_fix_NLVUtRtE/{single_runner,two_runners}/.
  Fixed package is staged there; original deployment and old evidence remain
  untouched. No result files copied back; #0006 is closed.

### Detection profiling parity

Goal: expose detection pipeline bottlenecks at segmentation's profiling
granularity without changing preprocessing, decoding, thresholds or accuracy.
The user approved this scope after the profiling audit (#0003).

- [x] Lead: author deterministic timing regression tests before implementation.
  Pin image-read/preprocess separation, input/output queue waits, existing
  decoder/top-k/coordinate/drawing hooks, setup and per-frame totals, disabled
  profiling, skipped images, and unchanged accuracy records.
- [x] Delegate: execute the narrow regression tests and retain the expected
  pre-fix failures; use an existing host interpreter, with no installs/network.
- [x] Lead: author the measurement changes in detection/benchmark.py using
  StageProfiler.time, preserving dpu/postprocess/accuracy/write stage names.
  Add nested decode_total/detection_postprocess stages, queue waits and
  producer_total/consumer_total; exclude sentinel bookkeeping from frame totals.
  Record runner_setup separately, but preserve application FPS scope including
  startup and pipeline drain. Do not claim excluded warmup or hardware duty.
- [x] Lead: add per-worker JSON summaries and additive run metadata:
  requested/active producers, queue capacity, board, draw/save/accuracy flags,
  paths, power/energy and availability/source. Explain overlapping thread
  times, per-level decoder samples, no warmup exclusion, and VART timing scope.
  Keep shared StageProfiler and other model families unchanged.
- [x] Delegate: insert lead-authored README usage/semantics and run targeted
  tests/help, then one final host suite and compileall pass. No board, Vitis,
  transfer, SSH, dependency installation, commits or pushes.
  Profiling: 8/8 pass; CLI: 6/6 pass; compileall and diff-check pass.
  Full suite: 67 pass, 6 skips, 2 Ultralytics import errors (missing OpenCV).
  Logs are /tmp/detection-profiling-{red,green,suite,static}.log.
- [x] Lead: review the full diff and timing assertions, record the outcome,
  and close #0003 only after passing evidence.
  Independent review found no blocking regressions. The idle-power text
  fallback was additionally tested RED-to-GREEN and fixed (#0004).
  Both issues are closed; real-board validation remains unperformed.
  Existing worker exception handling is unchanged; no warmup exclusion,
  hardware duty-cycle or per-layer tracing was added.
