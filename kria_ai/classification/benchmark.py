import argparse
import queue
import threading
import time
from collections.abc import Sequence
from pathlib import Path

from kria_ai.classification.config import get_dataset, get_model, validate_model_dataset
from kria_ai.classification.preprocess import preprocess_board_image
from kria_ai.common.artifacts import ArtifactPaths
from kria_ai.common.board.config import get_board
from kria_ai.common.board.input_quantization import build_normalization_lut
from kria_ai.common.board.power import PowerMonitor, read_power_mw
from kria_ai.common.board.profiling import ProgressCounter, format_metrics
from kria_ai.common.board.runtime import allocate_output_buffers, create_runner, load_dpu_model


def _producer(images, input_queue, input_shape, lut):
    import cv2

    for image_path, class_index in images:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_queue.put((preprocess_board_image(image_rgb, input_shape, lut), class_index))


def _consumer(index, input_queue, dpu_model, progress, results):
    import numpy as np

    runner = create_runner(dpu_model)
    output_buffers = allocate_output_buffers(runner)
    top1_correct = 0
    top5_correct = 0
    total = 0
    dpu_time = 0.0
    while True:
        item = input_queue.get()
        if item is None:
            input_queue.task_done()
            break
        image, target = item
        started = time.perf_counter()
        job_id = runner.execute_async([image], output_buffers)
        runner.wait(job_id)
        dpu_time += time.perf_counter() - started
        logits = output_buffers[0][0]
        count = min(5, logits.size)
        top_indices = np.argpartition(logits, -count)[-count:]
        top1 = top_indices[np.argmax(logits[top_indices])]
        top1_correct += int(target == top1)
        top5_correct += int(target in top_indices)
        total += 1
        progress.increment()
        input_queue.task_done()
    results[index] = (top1_correct, top5_correct, total, dpu_time)
    del runner


def _images(dataset_root, classes):
    images = []
    for class_index, class_name in enumerate(classes):
        class_dir = dataset_root / class_name
        if not class_dir.is_dir():
            continue
        images.extend(
            (path, class_index)
            for path in sorted(class_dir.iterdir())
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
    return images


def _xmodel_path(model_id, explicit, build_root):
    if explicit:
        return Path(explicit)
    local = Path(f"{model_id}_kria.xmodel")
    return local if local.is_file() else ArtifactPaths(model_id, build_root).compiled_xmodel


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(prog="python -m kria_ai classification benchmark")
    parser.add_argument("--model")
    parser.add_argument("--dataset")
    parser.add_argument("--board")
    parser.add_argument("--xmodel")
    parser.add_argument("--dataset-root")
    parser.add_argument("--build-root", default="build")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--producers", type=int, default=4)
    parser.add_argument("--queue-size", type=int, default=80)
    parser.add_argument("--output")
    args = parser.parse_args(argv)

    model_config = get_model(args.model)
    dataset_config = get_dataset(args.dataset)
    validate_model_dataset(model_config, dataset_config)
    board_config = get_board(args.board)
    runner_count = board_config.validate_runner_count(args.threads or board_config.default_runners)
    if args.producers < 1 or args.queue_size < 1:
        parser.error("--producers and --queue-size must be positive")

    dpu_model = load_dpu_model(_xmodel_path(model_config.id, args.xmodel, args.build_root))
    input_metadata = dpu_model.inputs[0]
    if len(input_metadata.shape) != 4 or input_metadata.shape[-1] != 3:
        raise RuntimeError(f"Expected NHWC RGB input, got {input_metadata.shape}")
    input_shape = input_metadata.shape[1:3]
    lut = build_normalization_lut(dataset_config.mean, dataset_config.std, input_metadata.fixed_point)
    dataset_root = Path(args.dataset_root) if args.dataset_root else dataset_config.board_root
    images = _images(dataset_root, dataset_config.classes)
    if not images:
        raise FileNotFoundError(f"No classification images found under {dataset_root}")

    input_queue = queue.Queue(maxsize=args.queue_size)
    progress = ProgressCounter()
    results = [None] * runner_count
    chunk_size = (len(images) + args.producers - 1) // args.producers
    chunks = [images[index:index + chunk_size] for index in range(0, len(images), chunk_size)]
    idle_samples = [read_power_mw(board_config.power_command) / 1000.0 for _ in range(5)]
    idle_power = sum(idle_samples) / len(idle_samples)
    monitor = PowerMonitor(sample=lambda: read_power_mw(board_config.power_command))
    monitor.start()
    started = time.perf_counter()
    try:
        consumers = [
            threading.Thread(target=_consumer, args=(index, input_queue, dpu_model, progress, results))
            for index in range(runner_count)
        ]
        for thread in consumers:
            thread.start()
        producers = [
            threading.Thread(target=_producer, args=(chunk, input_queue, input_shape, lut))
            for chunk in chunks
        ]
        for thread in producers:
            thread.start()
        for thread in producers:
            thread.join()
        for _ in consumers:
            input_queue.put(None)
        for thread in consumers:
            thread.join()
    finally:
        elapsed = time.perf_counter() - started
        monitor.stop()

    top1 = sum(result[0] for result in results if result)
    top5 = sum(result[1] for result in results if result)
    total = sum(result[2] for result in results if result)
    dpu_time = sum(result[3] for result in results if result)
    fps = total / elapsed if elapsed else 0.0
    power = monitor.average(idle_power)
    report = format_metrics(
        f"CLASSIFICATION: {model_config.name} | DPU RUNNERS: {runner_count}",
        [
            ("Images processed:", total),
            ("Top-1 accuracy:", f"{top1 / total * 100.0:.2f}%" if total else "N/A"),
            ("Top-5 accuracy:", f"{top5 / total * 100.0:.2f}%" if total else "N/A"),
            ("---", None),
            ("Application FPS:", f"{fps:.2f}"),
            ("DPU latency:", f"{dpu_time / total * 1000.0:.2f} ms" if total else "N/A"),
            ("Power:", f"{power:.2f} W"),
            ("Energy/image:", f"{power / fps * 1000.0:.2f} mJ" if fps else "N/A"),
        ],
    )
    print(report)
    output_path = Path(args.output or f"results_{model_config.id}_t{runner_count}.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
