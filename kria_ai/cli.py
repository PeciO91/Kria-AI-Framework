import argparse
import importlib
import sys
from collections.abc import Sequence


FAMILY_MODULES = {
    "classification": "kria_ai.classification.cli",
    "yolov26": "kria_ai.yolov26.cli",
}


def build_parser():
    parser = argparse.ArgumentParser(prog="python -m kria_ai")
    parser.add_argument("family", nargs="?", choices=tuple(FAMILY_MODULES))
    return parser


def main(argv: Sequence[str] | None = None):
    values = list(argv) if argv is not None else sys.argv[1:]
    if values and values[0] in FAMILY_MODULES:
        module = importlib.import_module(FAMILY_MODULES[values[0]])
        return module.main(values[1:])
    parser = build_parser()
    args = parser.parse_args(values)
    if args.family is None:
        parser.print_help()
        return 0
    return 0
