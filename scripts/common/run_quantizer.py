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
    parser.add_argument("--quant_mode", "--quant-mode", choices=("calib", "test"), default="calib")
    parser.add_argument("--subset_len", "--subset-len", type=int, default=100)
    parser.add_argument("--batch_size", "--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--build-root", default="build")
    parser.add_argument("--fast_ft", "--fast-ft", action="store_true")
    args = parser.parse_args(argv)
    model_id = args.model or ACTIVE_MODEL_ID
    model_type = get_active_model(model_id)["type"]
    if model_type == "classification":
        command = ["classification", "quantize"]
    else:
        command = ["yolov26", model_type, "quantize"]
    command.extend([
        "--model", model_id,
        "--mode", args.quant_mode,
        "--subset-len", str(args.subset_len),
        "--batch-size", str(args.batch_size),
        "--device", args.device,
        "--build-root", args.build_root,
    ])
    if args.dataset:
        command.extend(["--dataset", args.dataset])
    if args.checkpoint:
        command.extend(["--checkpoint", args.checkpoint])
    if args.fast_ft:
        command.append("--fast-ft")
    print("[DEPRECATED] Use: python -m kria_ai " + " ".join(command))
    return run_cli(command)


if __name__ == "__main__":
    raise SystemExit(main())
