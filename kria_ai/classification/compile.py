import argparse
from collections.abc import Sequence

from kria_ai.classification.config import get_model
from kria_ai.common.artifacts import ArtifactPaths
from kria_ai.common.board.config import get_board
from kria_ai.common.vitis.compiler import compile_xmodel


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(prog="python -m kria_ai classification compile")
    parser.add_argument("--model")
    parser.add_argument("--board")
    parser.add_argument("--build-root", default="build")
    args = parser.parse_args(argv)

    model_config = get_model(args.model)
    board_config = get_board(args.board)
    artifacts = ArtifactPaths(model_config.id, args.build_root)
    output_path = compile_xmodel(
        input_xmodel=artifacts.quantized_xmodel,
        arch_path=board_config.compiler_arch_path,
        output_dir=artifacts.compiled_dir,
        net_name=f"{model_config.id}_kria",
    )
    print(f"Compiled XMODEL: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
