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
    parser.add_argument("--build-root", default="build")
    args = parser.parse_args(argv)
    model_id = args.model or ACTIVE_MODEL_ID
    model_type = get_active_model(model_id)["type"]
    if model_type == "classification":
        command = ["classification", "compile"]
    else:
        command = ["yolov26", model_type, "compile"]
    command.extend(["--model", model_id, "--build-root", args.build_root])
    print("[DEPRECATED] Use: python -m kria_ai " + " ".join(command))
    return run_cli(command)


if __name__ == "__main__":
    raise SystemExit(main())
