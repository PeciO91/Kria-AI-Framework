import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kria_ai.classification.benchmark import main as run_benchmark


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--dataset", default="intel_images")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--producers", type=int, default=4)
    parser.add_argument("--queue-size", type=int, default=80)
    parser.add_argument("--xmodel")
    parser.add_argument("--dataset-root")
    args = parser.parse_args(argv)
    command = [
        "--model", args.model,
        "--dataset", args.dataset,
        "--producers", str(args.producers),
        "--queue-size", str(args.queue_size),
    ]
    if args.threads is not None:
        command.extend(["--threads", str(args.threads)])
    if args.xmodel:
        command.extend(["--xmodel", args.xmodel])
    if args.dataset_root:
        command.extend(["--dataset-root", args.dataset_root])
    print("[DEPRECATED] Use: python -m kria_ai classification benchmark " + " ".join(command))
    return run_benchmark(command)


if __name__ == "__main__":
    raise SystemExit(main())
