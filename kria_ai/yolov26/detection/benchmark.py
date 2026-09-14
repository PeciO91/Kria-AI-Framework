import argparse
import json
import queue
import threading
import time
from collections.abc import Sequence
from pathlib import Path

from kria_ai.common.artifacts import ArtifactPaths
from kria_ai.common.board.config import get_board
from kria_ai.common.board.input_quantization import build_normalization_lut
from kria_ai.common.board.power import PowerMonitor, read_power_mw
from kria_ai.common.board.profiling import ProgressCounter, StageProfiler, format_metrics, merge_stage_profilers
from kria_ai.common.board.runtime import allocate_output_buffers, create_runner, load_dpu_model, output_dequantization_scales
from kria_ai.yolov26.detection.config import get_dataset, get_model
from kria_ai.yolov26.preprocess import preprocess_board_image


def _producer(images, input_queue, input_shape, lut, profiler):
    import cv2

    for image_path in images:
        started = time.perf_counter()
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        original_shape = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_data = preprocess_board_image(image_rgb, input_shape, lut)
        profiler.add("preprocess", time.perf_counter() - started)
        input_queue.put((input_data, image, original_shape, image_path.name))


def _writer(output_queue, output_dir, profiler):
    import cv2

    while True:
        item = output_queue.get()
        if item is None:
            output_queue.task_done()
            break
        filename, image = item
        started = time.perf_counter()
        cv2.imwrite(str(output_dir / filename), image)
        profiler.add("write", time.perf_counter() - started)
        output_queue.task_done()


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
):
    from kria_ai.yolov26.decode import UltralyticsDecoderCache, decode_ultralytics_output
    from kria_ai.yolov26.detection.postprocess import draw_detections, postprocess_detections

    runner = create_runner(dpu_model)
    output_buffers = allocate_output_buffers(runner)
    scales = output_dequantization_scales(dpu_model.outputs)
    cache = UltralyticsDecoderCache(model_config.strides)
    total = 0
    dpu_time = 0.0
    class_histogram = {}
    while True:
        item = input_queue.get()
        if item is None:
            input_queue.task_done()
            break
        input_data, original_image, original_shape, filename = item
        started = time.perf_counter()
        job_id = runner.execute_async([input_data], output_buffers)
        runner.wait(job_id)
        elapsed = time.perf_counter() - started
        dpu_time += elapsed
        profiler.add("dpu", elapsed)

        started = time.perf_counter()
        boxes, scores, class_ids = decode_ultralytics_output(
            output_buffers,
            scales,
            model_config.confidence_threshold,
            cache,
            output_order,
            model_config.num_classes,
            model_config.reg_max,
        )
        boxes, scores, class_ids = postprocess_detections(
            boxes,
            scores,
            class_ids,
            input_shape,
            original_shape,
            model_config.max_detections,
        )
        profiler.add("postprocess", time.perf_counter() - started)
        for class_id in class_ids:
            value = int(class_id)
            class_histogram[value] = class_histogram.get(value, 0) + 1
        if draw:
            original_image = draw_detections(
                original_image,
                boxes,
                scores,
                class_ids,
                dataset_config.classes,
            )
        if save:
            output_queue.put((filename, original_image))
        total += 1
        progress.increment()
        input_queue.task_done()
    results[index] = (total, dpu_time, class_histogram, profiler)
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

    from kria_ai.yolov26.decode import validate_detection_output_contract

    dpu_model = load_dpu_model(_xmodel_path(model_config.id, args.xmodel, args.build_root))
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
    results = [None] * runner_count
    profile_enabled = args.profile or args.profile_json
    writer_profiler = StageProfiler(profile_enabled)
    writer_thread = None
    if save:
        writer_thread = threading.Thread(target=_writer, args=(output_queue, output_dir, writer_profiler))
        writer_thread.start()
    consumers = []
    for index in range(runner_count):
        profiler = StageProfiler(profile_enabled)
        thread = threading.Thread(
            target=_consumer,
            args=(
                index, input_queue, output_queue, dpu_model, input_shape,
                model_config, dataset_config, output_order, progress, results,
                profiler, draw, save,
            ),
        )
        thread.start()
        consumers.append(thread)
    chunk_size = (len(images) + args.producers - 1) // args.producers
    chunks = [images[index:index + chunk_size] for index in range(0, len(images), chunk_size)]
    producer_profilers = [StageProfiler(profile_enabled) for _ in chunks]
    producers = [
        threading.Thread(target=_producer, args=(chunk, input_queue, input_shape, lut, profiler))
        for chunk, profiler in zip(chunks, producer_profilers)
    ]

    idle_values = [read_power_mw(board_config.power_command) / 1000.0 for _ in range(5)]
    idle_power = sum(idle_values) / len(idle_values)
    monitor = PowerMonitor(sample=lambda: read_power_mw(board_config.power_command))
    monitor.start()
    started = time.perf_counter()
    try:
        for thread in producers:
            thread.start()
        for thread in producers:
            thread.join()
        for _ in consumers:
            input_queue.put(None)
        for thread in consumers:
            thread.join()
        if save:
            output_queue.put(None)
            writer_thread.join()
    finally:
        wall_time = time.perf_counter() - started
        monitor.stop()

    total = sum(result[0] for result in results if result)
    dpu_time = sum(result[1] for result in results if result)
    class_histogram = {}
    for result in results:
        if result:
            for class_id, count in result[2].items():
                class_histogram[class_id] = class_histogram.get(class_id, 0) + count
    detection_count = sum(class_histogram.values())
    fps = total / wall_time if wall_time else 0.0
    power = monitor.average(idle_power)
    report = format_metrics(
        f"YOLOV26 DETECTION: {model_config.name} | DPU RUNNERS: {runner_count}",
        [
            ("Images processed:", total),
            ("Detections:", detection_count),
            ("Application FPS:", f"{fps:.2f}"),
            ("DPU latency:", f"{dpu_time / total * 1000.0:.2f} ms" if total else "N/A"),
            ("Power:", f"{power:.2f} W"),
            ("Energy/image:", f"{power / fps * 1000.0:.2f} mJ" if fps else "N/A"),
            ("Compute efficiency:", "N/A"),
        ],
    )
    profilers = [*producer_profilers, *(result[3] for result in results if result), writer_profiler]
    profile = merge_stage_profilers(profilers) if profile_enabled else None
    if args.profile and profile is not None:
        lines = ["DETAILED PROFILE"]
        for row in profile.summary(wall_time):
            lines.append(
                f"{row['stage']}: avg={row['avg_ms']:.2f} ms "
                f"p95={row['p95_ms']:.2f} ms total={row['total_s']:.3f} s"
            )
        report += "\n" + "\n".join(lines) + "\n"
    print(report)
    report_path = Path(args.output_report or f"results_{model_config.id}_t{runner_count}.txt")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    if args.profile_json and profile is not None:
        payload = {
            "model": model_config.id,
            "dataset": dataset_config.id,
            "threads": runner_count,
            "images_processed": total,
            "detections": detection_count,
            "class_histogram": class_histogram,
            "wall_time_s": wall_time,
            "fps": fps,
            "dpu_latency_ms": dpu_time / total * 1000.0 if total else None,
            "stages": profile.summary(wall_time),
        }
        report_path.with_name(f"{report_path.stem}_profile.json").write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
