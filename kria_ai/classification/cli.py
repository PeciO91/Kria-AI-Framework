import argparse
import importlib
import sys
from collections.abc import Sequence


COMMAND_MODULES = {
    "evaluate": "kria_ai.classification.evaluate",
    "inspect": "kria_ai.classification.inspect",
    "optimize": "kria_ai.classification.optimize",
    "quantize": "kria_ai.classification.quantize",
    "compile": "kria_ai.classification.compile",
    "benchmark": "kria_ai.classification.benchmark",
}


def main(argv: Sequence[str] | None = None):
    values = list(argv) if argv is not None else sys.argv[1:]
    if values and values[0] in COMMAND_MODULES:
        module = importlib.import_module(COMMAND_MODULES[values[0]])
        return module.main(values[1:])
    parser = argparse.ArgumentParser(prog="python -m kria_ai classification")
    parser.add_argument("command", nargs="?", choices=tuple(COMMAND_MODULES))
    args = parser.parse_args(values)
    if args.command is None:
        parser.print_help()
    return 0
