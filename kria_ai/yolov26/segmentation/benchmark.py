"""KV260 board benchmark for YOLOv26 instance segmentation.

Heavy board and image-processing dependencies are intentionally imported only
once command-line parsing has completed so that ``--help`` works on a host
without NumPy, OpenCV, XIR, or VART installed.
"""

import argparse
import json
import queue
import random
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
from kria_ai.yolov26.preprocess import preprocess_board_image
from kria_ai.yolov26.segmentation.config import get_dataset, get_model


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
_MEMORY_TRANSFER_NOTE = (
    "The NumPy VART API does not expose separate cache/DMA synchronization "
    "timing; DPU submit/wait is reported as one observable DPU stage."
)


class _FrameSequenceWriter:
    """Small VideoWriter-compatible fallback that writes ordered JPEG frames."""

    def __init__(self, cv2_module, directory):
        self._cv2 = cv2_module
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.frame_index = 0

    def write(self, frame):
        frame_path = self.directory / f"frame_{self.frame_index:08d}.jpg"
        if not self._cv2.imwrite(
            str(frame_path),
            frame,
            [self._cv2.IMWRITE_JPEG_QUALITY, 90],
        ):
            raise OSError(f"Failed to write fallback video frame: {frame_path}")
        self.frame_index += 1

    def release(self):
        return None


def _profile_start(profiler):
    return time.perf_counter() if profiler.enabled else None


def _profile_end(profiler, stage, started):
    if started is not None:
        profiler.add(stage, time.perf_counter() - started)


def _xmodel_path(model_id, explicit, build_root):
    if explicit:
        return Path(explicit)
    local = Path(f"{model_id}_kria.xmodel")
    return local if local.is_file() else ArtifactPaths(model_id, build_root).compiled_xmodel


def _image_paths(root):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Segmentation image directory not found: {root}")
    return sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
    )


def _class_colors(class_count):
    generator = random.Random(42)
    return tuple(
        tuple(generator.randrange(32, 256) for _ in range(3))
        for _ in range(class_count)
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
    started = _profile_start(profiler)
    while not stop_event.is_set():
        try:
            item = source_queue.get(timeout=0.1)
            _profile_end(profiler, stage, started)
            return True, item
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
    raise RuntimeError(f"Segmentation pipeline worker failure:\n{details}")


def _producer(
    image_paths,
    input_queue,
    input_shape,
    normalization_lut,
    profiler,
    stop_event,
    errors,
    error_lock,
):
    try:
        import cv2

        for image_path in image_paths:
            if stop_event.is_set():
                break
            total_started = _profile_start(profiler)
            started = _profile_start(profiler)
            image = cv2.imread(str(image_path))
            _profile_end(profiler, "image_read", started)
            if image is None:
                print(f"[WARN] Unable to read image; skipping: {image_path}")
                continue

            original_shape = image.shape[:2]
            started = _profile_start(profiler)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            input_data = preprocess_board_image(
                image_rgb,
                input_shape,
                normalization_lut,
            )
            _profile_end(profiler, "preprocess", started)

            started = _profile_start(profiler)
            queued = _queue_put(
                input_queue,
                (input_data, image, original_shape, image_path),
                stop_event,
            )
            _profile_end(profiler, "input_enqueue_wait", started)
            _profile_end(profiler, "producer_total", total_started)
            if not queued:
                break
    except Exception:
        _record_worker_error("image producer", errors, error_lock, stop_event)


def _process_segmentation_outputs(
    output_buffers,
    dequantization_scales,
    decoder_cache,
    detection_order,
    mask_order,
    prototype_index,
    input_shape,
    original_shape,
    model_config,
    profiler,
):
    """Decode one2one detections, select top-k, and construct image masks."""
    import numpy as np

    from kria_ai.yolov26.decode import decode_ultralytics_output
    from kria_ai.yolov26.detection.postprocess import (
        postprocess_detections,
        xywh_to_xyxy,
    )
    from kria_ai.yolov26.segmentation.masks import (
        extract_mask_prototypes,
        gather_mask_coefficients,
        process_mask,
        scale_image_masks,
    )

    started = _profile_start(profiler)
    boxes_xywh, scores, class_ids, keep_indices = decode_ultralytics_output(
        output_buffers,
        dequantization_scales,
        model_config.confidence_threshold,
        decoder_cache,
        detection_order,
        model_config.num_classes,
        model_config.reg_max,
        profiler=profiler,
        return_keep_index=True,
    )
    _profile_end(profiler, "decode_total", started)

    started = _profile_start(profiler)
    (
        boxes_original,
        selected_scores,
        selected_classes,
        selected_keep,
        selection,
    ) = postprocess_detections(
        boxes_xywh,
        scores,
        class_ids,
        input_shape,
        original_shape,
        model_config.max_detections,
        keep_indices=keep_indices,
        profiler=profiler,
        return_selection=True,
    )
    _profile_end(profiler, "detection_postprocess", started)

    if selection.size == 0:
        binary_masks = np.empty(
            (0, original_shape[0], original_shape[1]),
            dtype=bool,
        )
        return boxes_original, selected_scores, selected_classes, binary_masks

    boxes_input = xywh_to_xyxy(boxes_xywh[selection])
    started = _profile_start(profiler)
    mask_coefficients = gather_mask_coefficients(
        output_buffers,
        dequantization_scales,
        mask_order,
        selected_keep,
        model_config.num_masks,
    )
    prototypes = extract_mask_prototypes(
        output_buffers,
        dequantization_scales,
        prototype_index,
    )
    masks = process_mask(
        prototypes,
        mask_coefficients,
        boxes_input,
        input_shape,
        upsample=False,
    )
    _profile_end(profiler, "mask_assembly", started)

    started = _profile_start(profiler)
    masks = scale_image_masks(
        masks,
        boxes_input,
        input_shape,
        original_shape,
    )
    binary_masks = masks > model_config.mask_threshold
    _profile_end(profiler, "mask_scale_threshold", started)
    return boxes_original, selected_scores, selected_classes, binary_masks


def _draw_instances(
    image,
    boxes,
    scores,
    class_ids,
    binary_masks,
    class_names,
    colors,
    profiler,
):
    import cv2
    import numpy as np

    from kria_ai.yolov26.detection.postprocess import draw_detections

    started = _profile_start(profiler)
    if len(binary_masks):
        overlay = image.copy()
        for mask, class_id in zip(binary_masks, class_ids):
            overlay[mask] = np.asarray(colors[int(class_id)], dtype=np.uint8)
        cv2.addWeighted(overlay, 0.5, image, 0.5, 0.0, dst=image)
    image = draw_detections(
        image,
        boxes,
        scores,
        class_ids,
        class_names,
        color=colors,
    )
    _profile_end(profiler, "draw_overlays", started)
    return image


def _evaluate_image(
    image_path,
    original_shape,
    class_ids,
    scores,
    binary_masks,
    labels_dir,
    eval_records,
    gt_counts,
    profiler,
):
    from kria_ai.yolov26.segmentation.evaluate import (
        load_yolo_seg_labels,
        match_mask_predictions,
    )

    started = _profile_start(profiler)
    label_path = labels_dir / f"{image_path.stem}.txt"
    gt_classes, gt_masks = load_yolo_seg_labels(label_path, original_shape)
    for class_id in gt_classes:
        if class_id not in gt_counts:
            raise ValueError(
                f"Ground-truth class {class_id} in {label_path} is outside the "
                f"configured range [0, {len(gt_counts) - 1}]"
            )
        gt_counts[class_id] += 1

    true_positive = match_mask_predictions(
        class_ids,
        scores,
        binary_masks,
        gt_classes,
        gt_masks,
        iou_threshold=0.5,
    )
    eval_records.extend(
        (int(class_id), float(score), bool(is_true_positive))
        for class_id, score, is_true_positive in zip(
            class_ids,
            scores,
            true_positive,
        )
    )
    _profile_end(profiler, "mask_map50_evaluation", started)


def _consumer(
    index,
    input_queue,
    output_queue,
    dpu_model,
    input_shape,
    model_config,
    dataset_config,
    detection_order,
    mask_order,
    prototype_index,
    labels_dir,
    draw,
    save,
    colors,
    progress,
    results,
    profiler,
    stop_event,
    errors,
    error_lock,
):
    runner = None
    total = 0
    dpu_time = 0.0
    class_histogram = {}
    eval_records = []
    gt_counts = {class_id: 0 for class_id in range(model_config.num_classes)}
    try:
        from kria_ai.yolov26.decode import UltralyticsDecoderCache

        started = _profile_start(profiler)
        runner = create_runner(dpu_model)
        output_buffers = allocate_output_buffers(runner)
        dequantization_scales = output_dequantization_scales(dpu_model.outputs)
        decoder_cache = UltralyticsDecoderCache(model_config.strides)
        _profile_end(profiler, "runner_setup", started)

        while not stop_event.is_set():
            found, item = _queue_get(
                input_queue,
                stop_event,
                profiler,
                "consumer_dequeue_wait",
            )
            if not found:
                break
            if item is None:
                input_queue.task_done()
                break

            try:
                input_data, image, original_shape, image_path = item
                consumer_started = _profile_start(profiler)
                started = time.perf_counter()
                job_id = runner.execute_async([input_data], output_buffers)
                runner.wait(job_id)
                elapsed = time.perf_counter() - started
                dpu_time += elapsed
                profiler.add("dpu", elapsed)

                started = _profile_start(profiler)
                boxes, scores, class_ids, binary_masks = (
                    _process_segmentation_outputs(
                        output_buffers,
                        dequantization_scales,
                        decoder_cache,
                        detection_order,
                        mask_order,
                        prototype_index,
                        input_shape,
                        original_shape,
                        model_config,
                        profiler,
                    )
                )
                _profile_end(profiler, "postprocess_total", started)

                for class_id in class_ids:
                    value = int(class_id)
                    class_histogram[value] = class_histogram.get(value, 0) + 1

                if labels_dir is not None:
                    _evaluate_image(
                        image_path,
                        original_shape,
                        class_ids,
                        scores,
                        binary_masks,
                        labels_dir,
                        eval_records,
                        gt_counts,
                        profiler,
                    )

                if draw:
                    image = _draw_instances(
                        image,
                        boxes,
                        scores,
                        class_ids,
                        binary_masks,
                        dataset_config.classes,
                        colors,
                        profiler,
                    )
                if save:
                    started = _profile_start(profiler)
                    queued = _queue_put(
                        output_queue,
                        (image_path.name, image),
                        stop_event,
                    )
                    _profile_end(profiler, "writer_enqueue_wait", started)
                    if not queued:
                        break

                total += 1
                progress.increment()
                _profile_end(profiler, "consumer_total", consumer_started)
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
        results[index] = {
            "total": total,
            "dpu_time": dpu_time,
            "class_histogram": class_histogram,
            "eval_records": eval_records,
            "gt_counts": gt_counts,
            "profiler": profiler,
        }
        if runner is not None:
            del runner


def _writer(
    output_queue,
    output_dir,
    profiler,
    stop_event,
    errors,
    error_lock,
):
    try:
        import cv2

        while not stop_event.is_set():
            found, item = _queue_get(
                output_queue,
                stop_event,
                profiler,
                "writer_dequeue_wait",
            )
            if not found:
                break
            if item is None:
                output_queue.task_done()
                break
            try:
                filename, image = item
                started = _profile_start(profiler)
                output_path = output_dir / filename
                if not cv2.imwrite(str(output_path), image):
                    raise OSError(f"Failed to write segmentation image: {output_path}")
                _profile_end(profiler, "image_write", started)
            finally:
                output_queue.task_done()
    except Exception:
        _record_worker_error("image writer", errors, error_lock, stop_event)


def _idle_power(board_config):
    samples = [
        read_power_mw(board_config.power_command) / 1000.0
        for _ in range(5)
    ]
    return sum(samples) / len(samples)


def _run_image_pipeline(
    images,
    output_dir,
    labels_dir,
    dpu_model,
    input_shape,
    normalization_lut,
    model_config,
    dataset_config,
    board_config,
    runner_count,
    producer_count,
    queue_size,
    draw,
    save,
    profile_enabled,
    detection_order,
    mask_order,
    prototype_index,
):
    input_queue = queue.Queue(maxsize=queue_size)
    output_queue = queue.Queue(maxsize=128) if save else None
    progress = ProgressCounter()
    stop_event = threading.Event()
    errors = []
    error_lock = threading.Lock()
    results = [None] * runner_count
    colors = _class_colors(model_config.num_classes)

    writer_profiler = StageProfiler(profile_enabled)
    writer_thread = None
    if save:
        writer_thread = threading.Thread(
            target=_writer,
            args=(
                output_queue,
                output_dir,
                writer_profiler,
                stop_event,
                errors,
                error_lock,
            ),
            name="segmentation-writer",
        )

    consumer_profilers = [
        StageProfiler(profile_enabled) for _ in range(runner_count)
    ]
    consumers = [
        threading.Thread(
            target=_consumer,
            args=(
                index,
                input_queue,
                output_queue,
                dpu_model,
                input_shape,
                model_config,
                dataset_config,
                detection_order,
                mask_order,
                prototype_index,
                labels_dir,
                draw,
                save,
                colors,
                progress,
                results,
                consumer_profilers[index],
                stop_event,
                errors,
                error_lock,
            ),
            name=f"segmentation-consumer-{index}",
        )
        for index in range(runner_count)
    ]

    chunk_size = (len(images) + producer_count - 1) // producer_count
    chunks = [
        images[index:index + chunk_size]
        for index in range(0, len(images), chunk_size)
    ]
    producer_profilers = [StageProfiler(profile_enabled) for _ in chunks]
    producers = [
        threading.Thread(
            target=_producer,
            args=(
                chunk,
                input_queue,
                input_shape,
                normalization_lut,
                profiler,
                stop_event,
                errors,
                error_lock,
            ),
            name=f"segmentation-producer-{index}",
        )
        for index, (chunk, profiler) in enumerate(
            zip(chunks, producer_profilers)
        )
    ]

    idle_power = _idle_power(board_config)
    monitor = PowerMonitor(
        sample=lambda: read_power_mw(board_config.power_command)
    )
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
    completed_results = [result for result in results if result is not None]
    total = sum(result["total"] for result in completed_results)
    dpu_time = sum(result["dpu_time"] for result in completed_results)
    class_histogram = {}
    eval_records = []
    gt_counts = {class_id: 0 for class_id in range(model_config.num_classes)}
    for result in completed_results:
        for class_id, count in result["class_histogram"].items():
            class_histogram[class_id] = class_histogram.get(class_id, 0) + count
        eval_records.extend(result["eval_records"])
        for class_id, count in result["gt_counts"].items():
            gt_counts[class_id] += count

    profile = None
    if profile_enabled:
        profilers = [*producer_profilers, *consumer_profilers]
        if save:
            profilers.append(writer_profiler)
        profile = merge_stage_profilers(profilers)

    fps = total / wall_time if wall_time > 0.0 else 0.0
    power = monitor.average(idle_power)
    return {
        "mode": "images",
        "items_discovered": len(images),
        "items_processed": total,
        "wall_time_s": wall_time,
        "fps": fps,
        "dpu_time_s": dpu_time,
        "dpu_latency_ms": dpu_time / total * 1000.0 if total else None,
        "dpu_duty_pct": (
            min(100.0, dpu_time / (wall_time * runner_count) * 100.0)
            if wall_time > 0.0
            else 0.0
        ),
        "power_w": power,
        "energy_mj": power / fps * 1000.0 if fps > 0.0 else None,
        "class_histogram": class_histogram,
        "eval_records": eval_records,
        "gt_counts": gt_counts,
        "profile": profile,
        "output": str(output_dir) if save else None,
        "video_source": None,
    }


def _open_video_capture(cv2_module, video_path):
    attempts = []
    seen = set()
    for attribute, name in (
        ("CAP_FFMPEG", "FFMPEG"),
        ("CAP_GSTREAMER", "GStreamer"),
        ("CAP_ANY", "automatic"),
    ):
        backend = getattr(cv2_module, attribute, None)
        if backend is None or backend in seen:
            continue
        seen.add(backend)
        try:
            capture = cv2_module.VideoCapture(str(video_path), backend)
        except Exception as error:
            attempts.append(f"{name}: {error}")
            continue
        if capture.isOpened():
            return capture, name
        capture.release()
        attempts.append(f"{name}: not opened")
    detail = "; ".join(attempts) if attempts else "no OpenCV backend available"
    raise RuntimeError(f"Unable to open input video {video_path}: {detail}")


def _open_video_writer(cv2_module, output_path, fps, frame_size):
    """Open an explicit OpenCV writer, falling back to an ordered JPEG set."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    codec = "MJPG" if output_path.suffix.lower() == ".avi" else "mp4v"
    writer = None
    try:
        writer = cv2_module.VideoWriter(
            str(output_path),
            cv2_module.VideoWriter_fourcc(*codec),
            fps,
            frame_size,
        )
        if writer.isOpened():
            return writer, str(output_path), "video"
    except Exception as error:
        print(f"[WARN] OpenCV VideoWriter failed: {error}")
    if writer is not None:
        writer.release()

    frame_directory = Path(f"{output_path}_frames")
    print(
        "[WARN] OpenCV VideoWriter did not open; writing an ordered JPEG "
        f"frame sequence to {frame_directory}"
    )
    return (
        _FrameSequenceWriter(cv2_module, frame_directory),
        str(frame_directory),
        "frame_sequence",
    )


def _run_video(
    video_path,
    output_video,
    dpu_model,
    input_shape,
    normalization_lut,
    model_config,
    dataset_config,
    board_config,
    draw,
    save,
    profile_enabled,
    detection_order,
    mask_order,
    prototype_index,
):
    import cv2
    import numpy as np

    from kria_ai.yolov26.decode import UltralyticsDecoderCache

    capture, backend_name = _open_video_capture(cv2, video_path)
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    source_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if source_width <= 0 or source_height <= 0:
        capture.release()
        raise RuntimeError(
            f"Video has invalid dimensions: {source_width}x{source_height}"
        )
    if source_fps <= 0.0 or not np.isfinite(source_fps):
        source_fps = 30.0
        print("[WARN] Invalid video FPS metadata; using 30 FPS for output")

    video_writer = None
    output_destination = None
    output_kind = None
    if save:
        video_writer, output_destination, output_kind = _open_video_writer(
            cv2,
            output_video,
            source_fps,
            (source_width, source_height),
        )

    profiler = StageProfiler(profile_enabled)
    runner = None
    monitor = None
    idle_power = 0.0
    frames_processed = 0
    dpu_time = 0.0
    class_histogram = {}
    colors = _class_colors(model_config.num_classes)
    wall_time = 0.0
    try:
        started = _profile_start(profiler)
        runner = create_runner(dpu_model)
        output_buffers = allocate_output_buffers(runner)
        dequantization_scales = output_dequantization_scales(dpu_model.outputs)
        decoder_cache = UltralyticsDecoderCache(model_config.strides)
        _profile_end(profiler, "runner_setup", started)

        idle_power = _idle_power(board_config)
        monitor = PowerMonitor(
            sample=lambda: read_power_mw(board_config.power_command)
        )
        monitor.start()
        wall_started = time.perf_counter()
        while True:
            frame_started = _profile_start(profiler)
            started = _profile_start(profiler)
            ok, image = capture.read()
            _profile_end(profiler, "video_read", started)
            if not ok:
                break

            original_shape = image.shape[:2]
            started = _profile_start(profiler)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            input_data = preprocess_board_image(
                image_rgb,
                input_shape,
                normalization_lut,
            )
            _profile_end(profiler, "preprocess", started)

            started = time.perf_counter()
            job_id = runner.execute_async([input_data], output_buffers)
            runner.wait(job_id)
            elapsed = time.perf_counter() - started
            dpu_time += elapsed
            profiler.add("dpu", elapsed)

            boxes, scores, class_ids, binary_masks = (
                _process_segmentation_outputs(
                    output_buffers,
                    dequantization_scales,
                    decoder_cache,
                    detection_order,
                    mask_order,
                    prototype_index,
                    input_shape,
                    original_shape,
                    model_config,
                    profiler,
                )
            )
            for class_id in class_ids:
                value = int(class_id)
                class_histogram[value] = class_histogram.get(value, 0) + 1

            if draw:
                image = _draw_instances(
                    image,
                    boxes,
                    scores,
                    class_ids,
                    binary_masks,
                    dataset_config.classes,
                    colors,
                    profiler,
                )
            if video_writer is not None:
                started = _profile_start(profiler)
                video_writer.write(image)
                _profile_end(profiler, "video_write", started)

            frames_processed += 1
            _profile_end(profiler, "frame_total", frame_started)
            if frames_processed % 10 == 0:
                total_label = source_frame_count if source_frame_count > 0 else "?"
                print(
                    f"\r[INFO] Video frames: {frames_processed}/{total_label}",
                    end="",
                    flush=True,
                )
        wall_time = time.perf_counter() - wall_started
        if frames_processed:
            print()
    finally:
        capture.release()
        if video_writer is not None:
            video_writer.release()
        if monitor is not None:
            monitor.stop()
        if runner is not None:
            del runner

    fps = frames_processed / wall_time if wall_time > 0.0 else 0.0
    power = monitor.average(idle_power) if monitor is not None else idle_power
    return {
        "mode": "video",
        "items_discovered": source_frame_count if source_frame_count > 0 else None,
        "items_processed": frames_processed,
        "wall_time_s": wall_time,
        "fps": fps,
        "dpu_time_s": dpu_time,
        "dpu_latency_ms": (
            dpu_time / frames_processed * 1000.0
            if frames_processed
            else None
        ),
        "dpu_duty_pct": (
            min(100.0, dpu_time / wall_time * 100.0)
            if wall_time > 0.0
            else 0.0
        ),
        "power_w": power,
        "energy_mj": power / fps * 1000.0 if fps > 0.0 else None,
        "class_histogram": class_histogram,
        "eval_records": [],
        "gt_counts": {},
        "profile": profiler if profile_enabled else None,
        "output": output_destination,
        "output_kind": output_kind,
        "video_source": {
            "path": str(video_path),
            "backend": backend_name,
            "width": source_width,
            "height": source_height,
            "fps": source_fps,
            "frame_count": (
                source_frame_count if source_frame_count > 0 else None
            ),
        },
    }


def _accuracy_metrics(stats, model_config):
    if not stats["gt_counts"]:
        return None
    total_ground_truths = sum(stats["gt_counts"].values())
    if total_ground_truths == 0:
        return {"ground_truths": 0, "metrics": None}

    from kria_ai.yolov26.segmentation.evaluate import aggregate_mask_map50

    current = aggregate_mask_map50(
        stats["eval_records"],
        stats["gt_counts"],
        model_config.num_classes,
    )
    return {
        "ground_truths": total_ground_truths,
        "metrics": {
            "map50": float(current["map50"]),
            "precision": float(current["precision"]),
            "recall": float(current["recall"]),
            "f1": float(current["f1"]),
        },
    }


def _format_class_histogram(class_histogram, class_names):
    if not class_histogram:
        return ""
    total = sum(class_histogram.values())
    lines = [
        "DETECTION CLASS HISTOGRAM",
        "-" * 60,
    ]
    for class_id, count in sorted(
        class_histogram.items(),
        key=lambda item: (-item[1], item[0]),
    )[:20]:
        name = (
            class_names[class_id]
            if 0 <= class_id < len(class_names)
            else f"Class {class_id}"
        )
        lines.append(
            f"{name:<24}{count:>8d}  ({count / total * 100.0:6.2f}%)"
        )
    return "\n".join(lines) + "\n"


def _format_profile(profile, wall_time):
    if profile is None:
        return ""
    lines = [
        "DETAILED PERFORMANCE PROFILE",
        "-" * 92,
        f"{'Stage':<32}{'Count':>8}{'Total (s)':>14}"
        f"{'Avg (ms)':>14}{'P95 (ms)':>14}{'Wall %':>10}",
    ]
    for row in profile.summary(wall_time):
        lines.append(
            f"{row['stage']:<32}{row['count']:>8d}{row['total_s']:>14.4f}"
            f"{row['avg_ms']:>14.3f}{row['p95_ms']:>14.3f}"
            f"{row['wall_pct']:>9.2f}%"
        )
    lines.extend(("", _MEMORY_TRANSFER_NOTE))
    return "\n".join(lines) + "\n"


def _build_report(
    stats,
    model_config,
    dataset_config,
    board_config,
    runner_count,
    producer_count,
    accuracy,
    include_profile,
):
    item_name = "Frames" if stats["mode"] == "video" else "Images"
    model_gops = getattr(model_config, "gops", None)
    compute_efficiency = (
        model_gops * stats["fps"] / board_config.dpu_peak_gops * 100.0
        if model_gops and stats["fps"] > 0.0
        else None
    )
    metrics = [
        ("Board:", board_config.name),
        ("Dataset:", dataset_config.name),
        ("Mode:", stats["mode"]),
        ("DPU runners:", runner_count),
        ("Producer threads:", producer_count),
        (f"{item_name} processed:", stats["items_processed"]),
        ("---", None),
        ("Application FPS:", f"{stats['fps']:.2f} img/s"),
        (
            "DPU latency:",
            f"{stats['dpu_latency_ms']:.2f} ms"
            if stats["dpu_latency_ms"] is not None
            else "N/A",
        ),
        ("DPU duty cycle:", f"{stats['dpu_duty_pct']:.2f}%"),
        (
            "Compute efficiency:",
            f"{compute_efficiency:.2f}%"
            if compute_efficiency is not None
            else "N/A (model GOPs unavailable)",
        ),
        ("Power:", f"{stats['power_w']:.2f} W"),
        (
            "Energy/item:",
            f"{stats['energy_mj']:.2f} mJ"
            if stats["energy_mj"] is not None
            else "N/A",
        ),
        ("---", None),
        ("Output:", stats["output"] or "disabled"),
    ]
    if stats["video_source"] is not None:
        source = stats["video_source"]
        metrics[3:3] = [
            ("Input video:", source["path"]),
            (
                "Source video:",
                f"{source['width']}x{source['height']} @ {source['fps']:.3f} FPS",
            ),
        ]
    if accuracy is not None:
        metrics.append(("Ground-truth masks:", accuracy["ground_truths"]))
        if accuracy["metrics"] is None:
            metrics.append(("Mask mAP@0.5:", "N/A (no labels found)"))
        else:
            metrics.extend(
                (
                    ("Mask mAP@0.5:", f"{accuracy['metrics']['map50']:.4f}"),
                    ("Mask precision:", f"{accuracy['metrics']['precision']:.4f}"),
                    ("Mask recall:", f"{accuracy['metrics']['recall']:.4f}"),
                    ("Mask F1:", f"{accuracy['metrics']['f1']:.4f}"),
                )
            )

    report = format_metrics(
        f"YOLOV26 SEGMENTATION: {model_config.name} | DPU RUNNERS: {runner_count}",
        metrics,
    )
    histogram = _format_class_histogram(
        stats["class_histogram"],
        dataset_config.classes,
    )
    if histogram:
        report += "\n" + histogram
    if include_profile:
        report += "\n" + _format_profile(stats["profile"], stats["wall_time_s"])
    return report


def _profile_payload(
    stats,
    args,
    model_config,
    dataset_config,
    board_config,
    runner_count,
    producer_count,
    queue_size,
    accuracy,
    output_indices,
):
    return {
        "model": model_config.id,
        "dataset": dataset_config.id,
        "board": board_config.id,
        "mode": stats["mode"],
        "threads": runner_count,
        "producers": producer_count,
        "queue_size": queue_size,
        "draw_outputs": not args.no_draw,
        "save_outputs": not args.no_save,
        "items_discovered": stats["items_discovered"],
        "items_processed": stats["items_processed"],
        "wall_time_s": stats["wall_time_s"],
        "fps": stats["fps"],
        "dpu_latency_ms": stats["dpu_latency_ms"],
        "dpu_duty_pct": stats["dpu_duty_pct"],
        "power_w": stats["power_w"],
        "energy_mj": stats["energy_mj"],
        "class_histogram": stats["class_histogram"],
        "accuracy": accuracy,
        "output": stats["output"],
        "output_kind": stats.get("output_kind"),
        "video_source": stats["video_source"],
        "output_indices": output_indices,
        "stages": stats["profile"].summary(stats["wall_time_s"]),
        "memory_transfer_note": _MEMORY_TRANSFER_NOTE,
    }


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m kria_ai yolov26 segmentation benchmark",
        description=(
            "Run the raw one2one YOLOv26 instance-segmentation graph on a "
            "Kria board. Image directories use a producer/runner/writer "
            "pipeline; file video uses one ordered runner."
        ),
    )
    parser.add_argument("--model", help="segmentation model registry ID")
    parser.add_argument("--dataset", help="segmentation dataset registry ID")
    parser.add_argument("--board", help="board registry ID (default: kv260)")
    parser.add_argument("--xmodel", help="explicit compiled XMODEL path")
    parser.add_argument(
        "--build-root",
        default="build",
        help="artifact root used when --xmodel and a local XMODEL are absent",
    )
    parser.add_argument(
        "--dataset-root",
        help="override the board image directory for image mode",
    )
    parser.add_argument(
        "--labels-dir",
        help="override YOLO polygon labels used with --accuracy",
    )
    parser.add_argument(
        "--threads",
        type=int,
        help="DPU runner count (KV260 range: 1-4; video requires 1)",
    )
    parser.add_argument(
        "--producers",
        type=int,
        help="image producer count (default: 4; video requires 1)",
    )
    parser.add_argument(
        "--queue-size",
        type=int,
        default=40,
        help="maximum number of preprocessed images waiting for a runner",
    )
    parser.add_argument(
        "--accuracy",
        action="store_true",
        help="compute the current greedy mask mAP@0.5 metric from YOLO labels",
    )
    parser.add_argument("--no-draw", action="store_true", help="do not draw masks and boxes")
    parser.add_argument("--no-save", action="store_true", help="do not save annotated output")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="append detailed per-stage timing to the text report",
    )
    parser.add_argument(
        "--profile-json",
        action="store_true",
        help="write per-stage timing and benchmark metadata as JSON",
    )
    parser.add_argument("--output-dir", help="annotated image output directory")
    parser.add_argument("--output-report", help="text report path")
    parser.add_argument(
        "--video",
        help="input video file; processed in order with exactly one DPU runner",
    )
    parser.add_argument(
        "--output-video",
        help="annotated .mp4 or .avi path (falls back to a JPEG frame directory)",
    )
    return parser


def main(argv: Sequence[str] | None = None):
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        model_config = get_model(args.model)
        dataset_config = get_dataset(args.dataset)
        board_config = get_board(args.board)
    except ValueError as error:
        parser.error(str(error))

    if args.queue_size < 1:
        parser.error("--queue-size must be positive")
    if args.video:
        requested_runners = args.threads if args.threads is not None else 1
        requested_producers = args.producers if args.producers is not None else 1
        if requested_runners != 1:
            parser.error("video mode requires --threads 1 to preserve frame order")
        if requested_producers != 1:
            parser.error("video mode requires --producers 1")
        if args.accuracy:
            parser.error("--accuracy is available only in image mode")
        if args.output_video and args.no_save:
            parser.error("--output-video cannot be used with --no-save")
        if args.output_video and Path(args.output_video).suffix.lower() not in {
            ".mp4",
            ".avi",
        }:
            parser.error("--output-video must end in .mp4 or .avi")
    else:
        requested_runners = (
            args.threads
            if args.threads is not None
            else board_config.default_runners
        )
        requested_producers = args.producers if args.producers is not None else 4
        if args.output_video:
            parser.error("--output-video requires --video")
    if requested_producers < 1:
        parser.error("--producers must be positive")
    try:
        runner_count = board_config.validate_runner_count(requested_runners)
    except (TypeError, ValueError) as error:
        parser.error(str(error))

    # Keep host-side help usable without the target runtime or vision stack.
    try:
        __import__("numpy")
        __import__("cv2")
        from kria_ai.yolov26.segmentation.masks import (
            resolve_segmentation_outputs,
        )
    except ImportError as error:
        parser.error(
            "NumPy and OpenCV are required to run the board segmentation "
            f"benchmark: {error}"
        )

    dpu_model = load_dpu_model(
        _xmodel_path(model_config.id, args.xmodel, args.build_root)
    )
    input_metadata = dpu_model.inputs[0]
    if (
        len(input_metadata.shape) != 4
        or input_metadata.shape[0] != 1
        or input_metadata.shape[-1] != 3
    ):
        raise RuntimeError(
            "Expected a batch-one NHWC RGB input tensor, got "
            f"{input_metadata.shape}"
        )
    input_shape = tuple(input_metadata.shape[1:3])
    if input_shape != tuple(model_config.input_size):
        raise RuntimeError(
            f"XMODEL input {input_shape} does not match configured input "
            f"{model_config.input_size}"
        )

    output_dimensions = [metadata.shape for metadata in dpu_model.outputs]
    detection_order, mask_order, prototype_index = (
        resolve_segmentation_outputs(
            output_dimensions,
            model_config.num_classes,
            model_config.num_masks,
            model_config.reg_max,
        )
    )
    output_indices = {
        "detection": detection_order,
        "mask_coefficients": mask_order,
        "prototype": prototype_index,
    }
    normalization_lut = build_normalization_lut(
        dataset_config.mean,
        dataset_config.std,
        input_metadata.fixed_point,
    )

    draw = not args.no_draw
    save = not args.no_save
    profile_enabled = args.profile or args.profile_json
    print(f"[INFO] Model: {model_config.name}")
    print(f"[INFO] Dataset: {dataset_config.name}")
    print(f"[INFO] Board: {board_config.name}")
    print(f"[INFO] Input: {input_metadata.shape}")
    print(
        "[INFO] Resolved outputs: "
        f"detection={detection_order}, masks={mask_order}, "
        f"prototype={prototype_index}"
    )

    if args.video:
        video_path = Path(args.video)
        if not video_path.is_file():
            raise FileNotFoundError(f"Input video not found: {video_path}")
        output_video = Path(
            args.output_video or f"outputs_{model_config.id}.mp4"
        )
        stats = _run_video(
            video_path,
            output_video,
            dpu_model,
            input_shape,
            normalization_lut,
            model_config,
            dataset_config,
            board_config,
            draw,
            save,
            profile_enabled,
            detection_order,
            mask_order,
            prototype_index,
        )
        labels_dir = None
    else:
        dataset_root = (
            Path(args.dataset_root)
            if args.dataset_root
            else dataset_config.board_images
        )
        images = _image_paths(dataset_root)
        if not images:
            raise FileNotFoundError(
                f"No segmentation images found under {dataset_root}"
            )
        labels_dir = None
        if args.accuracy:
            labels_dir = (
                Path(args.labels_dir)
                if args.labels_dir
                else dataset_config.board_labels
            )
            if not labels_dir.is_dir():
                raise FileNotFoundError(
                    f"Segmentation labels directory not found: {labels_dir}"
                )
        output_dir = Path(
            args.output_dir or f"outputs_{model_config.id}"
        )
        if save:
            output_dir.mkdir(parents=True, exist_ok=True)
        stats = _run_image_pipeline(
            images,
            output_dir,
            labels_dir,
            dpu_model,
            input_shape,
            normalization_lut,
            model_config,
            dataset_config,
            board_config,
            runner_count,
            requested_producers,
            args.queue_size,
            draw,
            save,
            profile_enabled,
            detection_order,
            mask_order,
            prototype_index,
        )

    accuracy = _accuracy_metrics(stats, model_config) if args.accuracy else None
    report = _build_report(
        stats,
        model_config,
        dataset_config,
        board_config,
        runner_count,
        requested_producers,
        accuracy,
        include_profile=args.profile,
    )
    print("\n" + report)

    default_report = (
        f"results_{model_config.id}_video.txt"
        if args.video
        else f"results_{model_config.id}_t{runner_count}.txt"
    )
    report_path = Path(args.output_report or default_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"[INFO] Report written to {report_path}")

    if args.profile_json:
        payload = _profile_payload(
            stats,
            args,
            model_config,
            dataset_config,
            board_config,
            runner_count,
            requested_producers,
            args.queue_size,
            accuracy,
            output_indices,
        )
        profile_path = report_path.with_name(
            f"{report_path.stem}_profile.json"
        )
        profile_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"[INFO] Profile written to {profile_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
