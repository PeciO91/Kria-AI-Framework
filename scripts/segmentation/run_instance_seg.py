import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kria_ai.yolov26.segmentation.benchmark import main as run_benchmark


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov26n_seg")
    parser.add_argument("--dataset", default="coco")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--producers", type=int)
    parser.add_argument("--queue-size", type=int, default=40)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-json", action="store_true")
    parser.add_argument("--no-draw", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--accuracy", action="store_true")
    parser.add_argument("--labels-dir")
    parser.add_argument("--video")
    parser.add_argument("--output-video")
    parser.add_argument("--xmodel")
    parser.add_argument("--dataset-root")
    args = parser.parse_args(argv)
    command = [
        "--model", args.model,
        "--dataset", args.dataset,
        "--queue-size", str(args.queue_size),
    ]
    for value, flag in (
        (args.threads, "--threads"),
        (args.producers, "--producers"),
    ):
        if value is not None:
            command.extend([flag, str(value)])
    for enabled, flag in (
        (args.profile, "--profile"),
        (args.profile_json, "--profile-json"),
        (args.no_draw, "--no-draw"),
        (args.no_save, "--no-save"),
        (args.accuracy, "--accuracy"),
    ):
        if enabled:
            command.append(flag)
    for value, flag in (
        (args.labels_dir, "--labels-dir"),
        (args.video, "--video"),
        (args.output_video, "--output-video"),
        (args.xmodel, "--xmodel"),
        (args.dataset_root, "--dataset-root"),
    ):
        if value:
            command.extend([flag, value])
    print("[DEPRECATED] Use: python -m kria_ai yolov26 segmentation benchmark " + " ".join(command))
    return run_benchmark(command)


if __name__ == "__main__":
    raise SystemExit(main())
