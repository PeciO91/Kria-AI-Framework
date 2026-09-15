from __future__ import annotations

import argparse
from collections.abc import Sequence

from kria_ai.common.artifacts import ArtifactPaths
from kria_ai.common.board.config import get_board
from kria_ai.common.vitis.compiler import compile_xmodel


def _model_config(task, model_id):
    if task == "detection":
        from kria_ai.yolov26.detection.config import get_model
    else:
        from kria_ai.yolov26.segmentation.config import get_model
    return get_model(model_id)


def compile_yolov26(
    task: str,
    *,
    model_id: str | None = None,
    board: str | None = None,
    build_root: str = "build",
):
    """Run Vitis AI compiler for YOLOv26 detection or segmentation models."""
    if task not in ("detection", "segmentation"):
        raise ValueError(f"task must be 'detection' or 'segmentation', got {task!r}")
    model_config = _model_config(task, model_id)
    board_config = get_board(board)
    artifacts = ArtifactPaths(model_config.id, build_root)
    return compile_xmodel(
        input_xmodel=artifacts.quantized_xmodel,
        arch_path=board_config.compiler_arch_path,
        output_dir=artifacts.compiled_dir,
        net_name=f"{model_config.id}_kria",
    )


def compile_detection(**kwargs):
    return compile_yolov26("detection", **kwargs)


def compile_segmentation(**kwargs):
    return compile_yolov26("segmentation", **kwargs)


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(prog="python -m kria_ai yolov26 <task> compile")
    parser.add_argument("--task", required=True, choices=("detection", "segmentation"))
    parser.add_argument("--model")
    parser.add_argument("--board")
    parser.add_argument("--build-root", default="build")
    args = parser.parse_args(argv)

    output_path = compile_yolov26(
        args.task,
        model_id=args.model,
        board=args.board,
        build_root=args.build_root,
    )
    print(f"Compiled XMODEL: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
