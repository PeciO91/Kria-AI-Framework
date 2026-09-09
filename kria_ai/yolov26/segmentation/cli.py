import argparse
import importlib
import sys
from collections.abc import Sequence


COMMANDS = {
    "inspect": ("kria_ai.yolov26.inspect", True),
    "quantize": ("kria_ai.yolov26.quantize", True),
    "compile": ("kria_ai.yolov26.compile", True),
    "evaluate": ("kria_ai.yolov26.segmentation.evaluate", False),
    "optimize": ("kria_ai.yolov26.segmentation.optimize", False),
    "benchmark": ("kria_ai.yolov26.segmentation.benchmark", False),
}


def main(argv: Sequence[str] | None = None):
    values = list(argv) if argv is not None else sys.argv[1:]
    if values and values[0] in COMMANDS:
        module_name, shared = COMMANDS[values[0]]
        module = importlib.import_module(module_name)
        forwarded = ["--task", "segmentation", *values[1:]] if shared else values[1:]
        return module.main(forwarded)
    parser = argparse.ArgumentParser(prog="python -m kria_ai yolov26 segmentation")
    parser.add_argument("command", nargs="?", choices=tuple(COMMANDS))
    args = parser.parse_args(values)
    if args.command is None:
        parser.print_help()
    return 0
