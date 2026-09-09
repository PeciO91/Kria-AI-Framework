import argparse

try:
    from ._bootstrap import PROJECT_ROOT
except ImportError:
    from _bootstrap import PROJECT_ROOT
from model_config import ACTIVE_MODEL_ID, get_active_model
from kria_ai.cli import main as run_cli


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--dataset")
    parser.add_argument("--checkpoint")
    parser.add_argument("--method", choices=("iterative", "one_step", "onestep"), default="one_step")
    parser.add_argument("--mode", choices=("search", "prune", "finetune", "all"), default="all")
    parser.add_argument("--ratio", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--subset_len", "--subset-len", type=int, default=200)
    parser.add_argument("--batch_size", "--batch-size", type=int, default=4)
    parser.add_argument("--num_subnet", "--num-subnets", type=int, default=1000)
    parser.add_argument("--channel-divisible", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--build-root", default="build")
    parser.add_argument("--output-checkpoint")
    args = parser.parse_args(argv)
    model_id = args.model or ACTIVE_MODEL_ID
    model_type = get_active_model(model_id)["type"]
    if model_type == "classification":
        command = ["classification", "optimize"]
    else:
        command = ["yolov26", model_type, "optimize"]
    command.extend([
        "--model", model_id,
        "--method", "one_step" if args.method == "onestep" else args.method,
        "--mode", args.mode,
        "--ratio", str(args.ratio),
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--subset-len", str(args.subset_len),
        "--batch-size", str(args.batch_size),
        "--channel-divisible", str(args.channel_divisible),
        "--device", args.device,
        "--build-root", args.build_root,
    ])
    command.extend(["--num-subnets", str(args.num_subnet)])
    if args.dataset:
        command.extend(["--dataset", args.dataset])
    if args.checkpoint:
        command.extend(["--checkpoint", args.checkpoint])
    if args.output_checkpoint:
        command.extend(["--output-checkpoint", args.output_checkpoint])
    print("[DEPRECATED] Use: python -m kria_ai " + " ".join(command))
    return run_cli(command)


if __name__ == "__main__":
    raise SystemExit(main())
