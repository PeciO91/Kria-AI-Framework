from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Sequence


COMMAND_MODULES = {
    "detection": "kria_ai.yolov26.detection.cli",
    "segmentation": "kria_ai.yolov26.segmentation.cli",
    "inspect": "kria_ai.yolov26.inspect",
    "quantize": "kria_ai.yolov26.quantize",
    "compile": "kria_ai.yolov26.compile",
}


def main(argv: Sequence[str] | None = None):
    values = list(argv) if argv is not None else sys.argv[1:]
    if values and values[0] in COMMAND_MODULES:
        module = importlib.import_module(COMMAND_MODULES[values[0]])
        return module.main(values[1:])
    parser = argparse.ArgumentParser(prog="python -m kria_ai yolov26")
    parser.add_argument("command", nargs="?", choices=tuple(COMMAND_MODULES))
    args = parser.parse_args(values)
    if args.command is None:
        parser.print_help()
    return 0
