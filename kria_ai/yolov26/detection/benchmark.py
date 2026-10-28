from __future__ import annotations

import argparse
import json
import queue
import threading
import time
import traceback
from collections.abc import Sequence
from pathlib import Path

from kria_ai.common.artifacts import ArtifactPaths
from kria_ai.common.board.config import get_board
from kria_ai.common.board.input_quantization import build_normalization_lut
from kria_ai.common.board.power import PowerMonitor, read_power_mw
from kria_ai.common.board.profiling import (
    ProgressCounter,
    StageProfiler,
    format_metrics,
    merge_stage_profilers,
)
from kria_ai.common.board.runtime import (
    allocate_output_buffers,
    create_runner,
    load_dpu_model,
    output_dequantization_scales,
)
from kria_ai.yolov26.detection.config import get_dataset, get_model
from kria_ai.yolov26.preprocess import preprocess_board_image


_TIMING_NOTES = (
    "Application FPS includes worker startup, image loading, enabled accuracy/"
    "drawing/saving, and pipeline drain; model loading and main-thread OpenCV "
    "initialization are excluded. No warmup "
    "images are excluded. runner_setup records runner/buffer/cache creation.",
    "Stage totals sum elapsed time across workers. Parent/child stages overlap; "
    "wall_pct is thread elapsed time / application wall time and can exceed "
    "100%. It is not CPU or DPU utilization or an exclusive bottleneck share.",
    "Decoder layout/concat stages are per image; threshold/dequant/class-score/"
    "box-decode stages are per pyramid level (survivor-dependent), not per image.",
    "Queue waits include dequeue of the end-of-stream sentinel; frame totals "
    "exclude sentinels. producer_total includes read/preprocess/enqueue; "
    "consumer_total excludes input wait but includes enabled accuracy/drawing/"
    "output enqueue. postprocess includes decode_total/detection_postprocess.",
    "The NumPy VART API does not expose separate cache/DMA synchronization "
    "timing; dpu measures execute_async plus wait, not pure hardware compute.",
)


def _queue_put(target_queue, item, stop_event):
    while not stop_event.is_set():
        try:
            target_queue.put(item, timeout=0.1)
            return True
        except queue.Full:
            continue
    return False


def _queue_get(source_queue, stop_event, profiler, stage):
    with profiler.time(stage):
        while not stop_event.is_set():
            try:
                return True, source_queue.get(timeout=0.1)
            except queue.Empty:
                continue
    return False, None


def _record_worker_error(label, errors, error_lock, stop_event):
    failure = traceback.format_exc()
    with error_lock:
        errors.append((label, failure))
    stop_event.set()


def _raise_worker_errors(errors):
    if not errors:
        return
    details = "\n\n".join(f"[{label}]\n{failure}" for label, failure in errors)
    raise RuntimeError(f"Detection pipeline worker failure:\n{details}")


def _initialize_opencv(draw, save):
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "OpenCV (cv2) is required for the detection benchmark"
        ) from error
    required = ["imread", "cvtColor", "resize", "copyMakeBorder"]
    if draw:
        required += ["rectangle", "putText"]
    if save:
        required += ["imwrite"]
    missing = [name for name in required if not callable(getattr(cv2, name, None))]
    if missing:
        location = getattr(cv2, "__file__", None) or "unknown"
        raise RuntimeError(
            "OpenCV module is missing required functions "
            f"{missing} (cv2 at {location})"
        )
    return cv2


def _producer(
    images,
    input_queue,
    input_shape,
    lut,
    profiler,
    stop_event,
    errors,
    error_lock,
):
    try:
        import cv2

        for image_path in images:
            if stop_event.is_set():
                break
            started = time.perf_counter() if profiler.enabled else None
            with profiler.time("image_read"):
                image = cv2.imread(str(image_path))
            if image is None:
                print(f"[WARN] Unable to read image; skipping: {image_path}")
                continue
            original_shape = image.shape[:2]
            with profiler.time("preprocess"):
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                input_data = preprocess_board_image(image_rgb, input_shape, lut)
            with profiler.time("input_enqueue_wait"):
                queued = _queue_put(
                    input_queue,
                    (input_data, image, original_shape, image_path.name),
                    stop_event,
                )
            if not queued:
                break
            if started is not None:
                profiler.add("producer_total", time.perf_counter() - started)
    except Exception:
        _record_worker_error("image producer", errors, error_lock, stop_event)


def _writer(output_queue, output_dir, profiler, stop_event, errors, error_lock):
    try:
        import cv2

        while not stop_event.is_set():
            found, item = _queue_get(
                output_queue, stop_event, profiler, "writer_dequeue_wait"
            )
            if not found:
                break
            if item is None:
                output_queue.task_done()
                break
            try:
                filename, image = item
                with profiler.time("write"):
                    output_path = output_dir / filename
                    if not cv2.imwrite(str(output_path), image):
                        raise OSError(
                            f"Failed to write detection image: {output_path}"
                        )
            finally:
                output_queue.task_done()
    except Exception:
        _record_worker_error("image writer", errors, error_lock, stop_event)


def _consumer(
    index,
    input_queue,
    output_queue,
    dpu_model,
    input_shape,
    model_config,
    dataset_config,
    output_order,
    progress,
    results,
    profiler,
    draw,
    save,
    accuracy,
    labels_dir,
    confidence_threshold,
    iou_thresholds,
    stop_event,
    errors,
    error_lock,
):
    runner = None
    total = 0
    dpu_time = 0.0
    class_histogram = {}
    evaluation_records = []
    ground_truth_counts = {}
    labeled_images = 0
    confusion_matrix = None
    try:
        from kria_ai.yolov26.decode import UltralyticsDecoderCache, decode_ultralytics_output
        from kria_ai.yolov26.detection.evaluate import (
            detection_confusion_matrix,
            load_yolo_detection_labels,
            match_detection_predictions,
        )
        from kria_ai.yolov26.detection.postprocess import draw_detections, postprocess_detections

        if accuracy:
            import numpy as np

            confusion_matrix = np.zeros(
                (model_config.num_classes + 1, model_config.num_classes + 1),
                dtype=np.int64,
            )
        with profiler.time("runner_setup"):
            runner = create_runner(dpu_model)
            output_buffers = allocate_output_buffers(runner)
            scales = output_dequantization_scales(dpu_model.outputs)
            cache = UltralyticsDecoderCache(model_config.strides)
        while not stop_event.is_set():
            found, item = _queue_get(
                input_queue, stop_event, profiler, "consumer_dequeue_wait"
            )
            if not found:
                break
            if item is None:
                input_queue.task_done()
                break
            try:
                input_data, original_image, original_shape, filename = item
                consumer_started = (
                    time.perf_counter() if profiler.enabled else None
                )
                started = time.perf_counter()
                job_id = runner.execute_async([input_data], output_buffers)
                runner.wait(job_id)
                elapsed = time.perf_counter() - started
                dpu_time += elapsed
                profiler.add("dpu", elapsed)

                with profiler.time("postprocess"):
                    with profiler.time("decode_total"):
                        boxes, scores, class_ids = decode_ultralytics_output(
                            output_buffers,
                            scales,
                            confidence_threshold,
                            cache,
                            output_order,
                            model_config.num_classes,
                            model_config.reg_max,
                            profiler=profiler,
                        )
                    with profiler.time("detection_postprocess"):
                        boxes, scores, class_ids = postprocess_detections(
                            boxes,
                            scores,
                            class_ids,
                            input_shape,
                            original_shape,
                            model_config.max_detections,
                            profiler=profiler,
                        )
                for class_id in class_ids:
                    value = int(class_id)
                    class_histogram[value] = class_histogram.get(value, 0) + 1
                if accuracy:
                    with profiler.time("accuracy"):
                        label_path = labels_dir / f"{Path(filename).stem}.txt"
                        labeled_images += int(label_path.is_file())
                        ground_truth_classes, ground_truth_boxes = load_yolo_detection_labels(
                            label_path,
                            original_shape,
                            model_config.num_classes,
                        )
                        for class_id in ground_truth_classes:
                            value = int(class_id)
                            ground_truth_counts[value] = ground_truth_counts.get(value, 0) + 1
                        matches = match_detection_predictions(
                            class_ids,
                            scores,
                            boxes,
                            ground_truth_classes,
                            ground_truth_boxes,
                            iou_thresholds,
                        )
                        confusion_matrix += detection_confusion_matrix(
                            class_ids, scores, boxes, ground_truth_classes,
                            ground_truth_boxes, model_config.num_classes,
                            confidence_threshold=confidence_threshold,
                        )
                        evaluation_records.extend(
                            (int(class_id), float(score), matched.copy(), filename, prediction_index)
                            for prediction_index, (class_id, score, matched) in enumerate(
                                zip(class_ids, scores, matches)
                            )
                        )
                if draw:
                    original_image = draw_detections(
                        original_image,
                        boxes,
                        scores,
                        class_ids,
                        dataset_config.classes,
                        profiler=profiler,
                    )
                if save:
                    with profiler.time("writer_enqueue_wait"):
                        queued = _queue_put(
                            output_queue,
                            (filename, original_image),
                            stop_event,
                        )
                    if not queued:
                        break
                total += 1
                progress.increment()
                if consumer_started is not None:
                    profiler.add(
                        "consumer_total",
                        time.perf_counter() - consumer_started,
                    )
            finally:
                input_queue.task_done()
    except Exception:
        _record_worker_error(
            f"DPU consumer {index}",
            errors,
            error_lock,
            stop_event,
        )
    finally:
        results[index] = (
            total,
            dpu_time,
            class_histogram,
            profiler,
            evaluation_records,
            ground_truth_counts,
            labeled_images,
            confusion_matrix,
        )
        if runner is not None:
            del runner


def _image_paths(root):
    return sorted(
        path for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def _xmodel_path(model_id, explicit, build_root):
    if explicit:
        return Path(explicit)
    local = Path(f"{model_id}_kria.xmodel")
    return local if local.is_file() else ArtifactPaths(model_id, build_root).compiled_xmodel


def _accuracy_metrics(results, num_classes, iou_thresholds, *, confidence_threshold=None):
    from kria_ai.yolov26.detection.evaluate import aggregate_detection_metrics

    evaluation_records = []
    ground_truth_counts = {}
    labeled_images = 0
    confusion_matrix = None
    for result in results:
        if not result:
            continue
        if len(result) < 8 or result[7] is None:
            raise RuntimeError("Incomplete worker accuracy results: confusion matrix is missing")
        evaluation_records.extend(result[4])
        for class_id, count in result[5].items():
            ground_truth_counts[class_id] = ground_truth_counts.get(class_id, 0) + count
        labeled_images += result[6]
        confusion_matrix = (
            result[7].copy() if confusion_matrix is None else confusion_matrix + result[7]
        )
    metrics = aggregate_detection_metrics(
        evaluation_records,
        ground_truth_counts,
        num_classes,
        iou_thresholds,
        confusion_matrix=confusion_matrix,
    )
    metrics["ground_truths"] = sum(ground_truth_counts.values())
    metrics["labeled_images"] = labeled_images
    metrics["precision_recall_confidence_threshold"] = confidence_threshold
    metrics["precision_recall_matching_iou"] = 0.5
    return metrics


def _worker_profiles(producers, consumers, writer, results, wall_time, save):
    workers = []
    groups = [
        ("producer", producers, "producer_total"),
        ("consumer", consumers, "consumer_total"),
    ]
    if save:
        groups.append(("writer", [writer], "write"))
    for role, profilers, count_stage in groups:
        for index, profiler in enumerate(profilers):
            stages = profiler.summary(wall_time)
            count = next(
                (row["count"] for row in stages if row["stage"] == count_stage),
                0,
            )
            if role == "consumer":
                count = results[index][0] if results[index] is not None else 0
            workers.append({
                "role": role,
                "index": index,
                "images_processed": count,
                "stages": stages,
            })
    return workers


def _format_profile(profile, wall_time, workers):
    sections = [("ALL WORKERS", profile.summary(wall_time))]
    sections.extend(
        (
            f"{worker['role'].upper()} {worker['index']} "
            f"({worker['images_processed']} images)",
            worker["stages"],
        )
        for worker in workers
    )
    lines = ["DETAILED PERFORMANCE PROFILE"]
    for title, rows in sections:
        lines.extend((
            title,
            "-" * 103,
            f"{'Stage':<32}{'Count':>8}{'Total (s)':>13}"
            f"{'Avg (ms)':>13}{'P50 (ms)':>13}{'P95 (ms)':>13}"
            f"{'Wall %':>11}",
        ))
        for row in rows:
            lines.append(
                f"{row['stage']:<32}{row['count']:>8d}{row['total_s']:>13.4f}"
                f"{row['avg_ms']:>13.3f}{row['p50_ms']:>13.3f}"
                f"{row['p95_ms']:>13.3f}{row['wall_pct']:>10.2f}%"
            )
    lines.extend(("", *_TIMING_NOTES))
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(prog="python -m kria_ai yolov26 detection benchmark")
    parser.add_argument("--model")
    parser.add_argument("--dataset")
    parser.add_argument("--board")
    parser.add_argument("--xmodel")
    parser.add_argument("--dataset-root")
    parser.add_argument("--build-root", default="build")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--producers", type=int, default=4)
    parser.add_argument("--queue-size", type=int, default=40)
    parser.add_argument("--accuracy", action="store_true")
    parser.add_argument("--labels-dir")
    parser.add_argument("--confidence-threshold", type=float)
    parser.add_argument("--no-draw", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-json", action="store_true")
    parser.add_argument("--output-dir")
    parser.add_argument("--output-report")
    args = parser.parse_args(argv)

    model_config = get_model(args.model)
    dataset_config = get_dataset(args.dataset)
    board_config = get_board(args.board)
    requested_runners = board_config.default_runners if args.threads is None else args.threads
    try:
        runner_count = board_config.validate_runner_count(requested_runners)
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    if args.producers < 1 or args.queue_size < 1:
        parser.error("--producers and --queue-size must be positive")
    if args.accuracy and not args.labels_dir:
        parser.error("--accuracy requires --labels-dir")
    if args.confidence_threshold is not None and not 0.0 <= args.confidence_threshold <= 1.0:
        parser.error("--confidence-threshold must be in [0, 1]")
    labels_dir = Path(args.labels_dir) if args.labels_dir else None
    if labels_dir is not None and not labels_dir.is_dir():
        parser.error(f"Label directory not found: {labels_dir}")
    confidence_threshold = args.confidence_threshold
    if confidence_threshold is None:
        confidence_threshold = 0.001 if args.accuracy else model_config.confidence_threshold

    _initialize_opencv(not args.no_draw, not args.no_save)

    from kria_ai.yolov26.decode import validate_detection_output_contract
    from kria_ai.yolov26.detection.evaluate import DETECTION_IOU_THRESHOLDS

    iou_thresholds = DETECTION_IOU_THRESHOLDS

    xmodel_path = _xmodel_path(model_config.id, args.xmodel, args.build_root)
    dpu_model = load_dpu_model(xmodel_path)
    input_metadata = dpu_model.inputs[0]
    if (
        len(input_metadata.shape) != 4
        or input_metadata.shape[0] != 1
        or input_metadata.shape[-1] != 3
    ):
        raise RuntimeError(
            f"Expected a batch-one NHWC RGB input tensor, got {input_metadata.shape}"
        )
    input_shape = tuple(input_metadata.shape[1:3])
    if input_shape != tuple(model_config.input_size):
        raise RuntimeError(
            f"XMODEL input {input_shape} does not match configured input "
            f"{model_config.input_size}"
        )
    output_order = validate_detection_output_contract(
        [metadata.shape for metadata in dpu_model.outputs],
        model_config.num_classes,
        model_config.reg_max,
        len(model_config.strides),
    )
    lut = build_normalization_lut(dataset_config.mean, dataset_config.std, input_metadata.fixed_point)
    dataset_root = Path(args.dataset_root) if args.dataset_root else dataset_config.board_images
    images = _image_paths(dataset_root)
    if not images:
        raise FileNotFoundError(f"No detection images found under {dataset_root}")
    output_dir = Path(args.output_dir or f"outputs_{model_config.id}")
    save = not args.no_save
    draw = not args.no_draw
    if save:
        output_dir.mkdir(parents=True, exist_ok=True)

    input_queue = queue.Queue(maxsize=args.queue_size)
    output_queue = queue.Queue(maxsize=128) if save else None
    progress = ProgressCounter()
    stop_event = threading.Event()
    errors = []
    error_lock = threading.Lock()
    results = [None] * runner_count
    profile_enabled = args.profile or args.profile_json
    writer_profiler = StageProfiler(profile_enabled)
    writer_thread = None
    if save:
        writer_thread = threading.Thread(
            target=_writer,
            args=(
                output_queue, output_dir, writer_profiler,
                stop_event, errors, error_lock,
            ),
            name="detection-writer",
        )
    consumers = []
    consumer_profilers = [StageProfiler(profile_enabled) for _ in range(runner_count)]
    for index, profiler in enumerate(consumer_profilers):
        thread = threading.Thread(
            target=_consumer,
            args=(
                index, input_queue, output_queue, dpu_model, input_shape,
                model_config, dataset_config, output_order, progress, results,
                profiler, draw, save, args.accuracy, labels_dir,
                confidence_threshold, iou_thresholds,
                stop_event, errors, error_lock,
            ),
            name=f"detection-consumer-{index}",
        )
        consumers.append(thread)
    chunk_size = (len(images) + args.producers - 1) // args.producers
    chunks = [images[index:index + chunk_size] for index in range(0, len(images), chunk_size)]
    producer_profilers = [StageProfiler(profile_enabled) for _ in chunks]
    producers = [
        threading.Thread(
            target=_producer,
            args=(
                chunk, input_queue, input_shape, lut, profiler,
                stop_event, errors, error_lock,
            ),
            name=f"detection-producer-{index}",
        )
        for index, (chunk, profiler) in enumerate(
            zip(chunks, producer_profilers)
        )
    ]

    idle_values = [read_power_mw(board_config.power_command) / 1000.0 for _ in range(5)]
    idle_power = sum(idle_values) / len(idle_values)
    monitor = PowerMonitor(sample=lambda: read_power_mw(board_config.power_command))
    monitor.start()
    started = time.perf_counter()
    all_threads = [*producers, *consumers]
    if writer_thread is not None:
        all_threads.append(writer_thread)
    try:
        if writer_thread is not None:
            writer_thread.start()
        for thread in consumers:
            thread.start()
        for thread in producers:
            thread.start()
        for thread in producers:
            thread.join()
        if not stop_event.is_set():
            for _ in consumers:
                if not _queue_put(input_queue, None, stop_event):
                    break
        for thread in consumers:
            thread.join()
        if writer_thread is not None and not stop_event.is_set():
            _queue_put(output_queue, None, stop_event)
        if writer_thread is not None:
            writer_thread.join()
    finally:
        wall_time = time.perf_counter() - started
        stop_event.set()
        for thread in all_threads:
            if thread.is_alive():
                thread.join(timeout=1.0)
        monitor.stop()

    _raise_worker_errors(errors)
    total = sum(result[0] for result in results if result)
    if total == 0 and images:
        raise RuntimeError(
            f"No images were processed of {len(images)} discovered under "
            f"{dataset_root}"
        )
    if args.accuracy and total != len(images):
        raise RuntimeError(
            f"Accuracy evaluation processed {total} of {len(images)} images"
        )
    dpu_time = sum(result[1] for result in results if result)
    class_histogram = {}
    for result in results:
        if result:
            for class_id, count in result[2].items():
                class_histogram[class_id] = class_histogram.get(class_id, 0) + count
    detection_count = sum(class_histogram.values())
    fps = total / wall_time if wall_time else 0.0
    power = monitor.average(idle_power)
    power_available = power > 0.0
    energy_mj = power / fps * 1000.0 if power_available and fps else None
    power_source = (
        "runtime_samples" if monitor.samples
        else "idle_fallback" if power_available else "unavailable"
    )
    power_note = " (idle fallback)" if power_source == "idle_fallback" else ""
    accuracy_metrics = (
        _accuracy_metrics(
            results, model_config.num_classes, iou_thresholds,
            confidence_threshold=confidence_threshold,
        )
        if args.accuracy
        else None
    )
    if accuracy_metrics is not None and accuracy_metrics["ground_truths"] == 0:
        raise RuntimeError(f"No ground-truth boxes found under {labels_dir}")
    report_metrics = [
        ("Images processed:", total),
        ("Detections:", detection_count),
        ("Application FPS:", f"{fps:.2f}"),
        ("DPU latency:", f"{dpu_time / total * 1000.0:.2f} ms" if total else "N/A"),
        ("Power:", f"{power:.2f} W{power_note}" if power_available else "N/A"),
        (
            "Energy/image:",
            f"{energy_mj:.2f} mJ{power_note}" if energy_mj is not None else "N/A",
        ),
        ("Compute efficiency:", "N/A"),
    ]
    if accuracy_metrics is not None:
        report_metrics.extend([
            ("Evaluator:", accuracy_metrics["evaluator"]),
            ("Labeled images:", accuracy_metrics["labeled_images"]),
            ("Ground-truth boxes:", accuracy_metrics["ground_truths"]),
            ("Score threshold:", f"{confidence_threshold:.4f}"),
            ("Precision@0.5:", f"{accuracy_metrics['precision']:.4f}"),
            ("Recall@0.5:", f"{accuracy_metrics['recall']:.4f}"),
            ("mAP@0.5:", f"{accuracy_metrics['map50']:.4f}"),
            ("mAP@0.5:0.95:", f"{accuracy_metrics['map50_95']:.4f}"),
        ])
        for class_id, class_name in enumerate(dataset_config.classes):
            class_metrics = accuracy_metrics["per_class"].get(class_id)
            if class_metrics is None:
                continue
            report_metrics.append((
                f"{class_name}:",
                f"P={class_metrics['precision']:.4f} "
                f"R={class_metrics['recall']:.4f} "
                f"AP50={class_metrics['map50']:.4f} "
                f"AP50-95={class_metrics['map50_95']:.4f}",
            ))
    report = format_metrics(
        f"YOLOV26 DETECTION: {model_config.name} | DPU RUNNERS: {runner_count}",
        report_metrics,
    )
    profilers = [*producer_profilers, *consumer_profilers]
    if save:
        profilers.append(writer_profiler)
    profile = merge_stage_profilers(profilers) if profile_enabled else None
    workers = (
        _worker_profiles(
            producer_profilers, consumer_profilers, writer_profiler,
            results, wall_time, save,
        )
        if profile_enabled else []
    )
    if args.profile and profile is not None:
        report += "\n" + _format_profile(profile, wall_time, workers)
    print(report)
    report_path = Path(args.output_report or f"results_{model_config.id}_t{runner_count}.txt")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    if args.profile_json and profile is not None:
        payload = {
            "profile_schema_version": 2,
            "model": model_config.id,
            "dataset": dataset_config.id,
            "board": board_config.id,
            "xmodel": str(xmodel_path),
            "dataset_root": str(dataset_root),
            "labels_dir": str(labels_dir) if labels_dir is not None else None,
            "threads": runner_count,
            "producers": args.producers,
            "active_producers": len(producers),
            "queue_size": args.queue_size,
            "writer_queue_size": output_queue.maxsize if save else 0,
            "draw_outputs": draw,
            "save_outputs": save,
            "accuracy_enabled": args.accuracy,
            "items_discovered": len(images),
            "images_processed": total,
            "detections": detection_count,
            "class_histogram": class_histogram,
            "wall_time_s": wall_time,
            "fps": fps,
            "dpu_latency_ms": dpu_time / total * 1000.0 if total else None,
            "power_available": power_available,
            "power_w": power if power_available else None,
            "power_scope": "SOM total power",
            "power_samples": len(monitor.samples),
            "power_source": power_source,
            "energy_mj": energy_mj,
            "confidence_threshold": confidence_threshold,
            "accuracy": accuracy_metrics,
            "warmup_images": 0,
            "timing_notes": _TIMING_NOTES,
            "stages": profile.summary(wall_time),
            "workers": workers,
        }
        report_path.with_name(f"{report_path.stem}_profile.json").write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
