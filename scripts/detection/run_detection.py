import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kria_ai.yolov26.detection.benchmark import main as run_benchmark


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov26s")
    parser.add_argument("--dataset", default="coco")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--producers", type=int, default=4)
    parser.add_argument("--queue-size", type=int, default=40)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-json", action="store_true")
    parser.add_argument("--no-draw", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--xmodel")
    parser.add_argument("--dataset-root")
    args = parser.parse_args(argv)
    if args.model != "yolov26s":
        parser.error("The refactored detection path supports YOLOv26 only")
    command = [
        "--model", args.model,
        "--dataset", args.dataset,
        "--producers", str(args.producers),
        "--queue-size", str(args.queue_size),
    ]
    if args.threads is not None:
        command.extend(["--threads", str(args.threads)])
    for enabled, flag in (
        (args.profile, "--profile"),
        (args.profile_json, "--profile-json"),
        (args.no_draw, "--no-draw"),
        (args.no_save, "--no-save"),
    ):
        if enabled:
            command.append(flag)
    if args.xmodel:
        command.extend(["--xmodel", args.xmodel])
    if args.dataset_root:
        command.extend(["--dataset-root", args.dataset_root])
    print("[DEPRECATED] Use: python -m kria_ai yolov26 detection benchmark " + " ".join(command))
    return run_benchmark(command)


if __name__ == "__main__":
    raise SystemExit(main())
