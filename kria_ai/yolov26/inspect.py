import argparse
from collections.abc import Sequence

from kria_ai.common.artifacts import ArtifactPaths
from kria_ai.common.board.config import get_board
from kria_ai.common.vitis.inspector import inspect_model
from kria_ai.yolov26.export import apply_export_patch, validate_export_outputs
from kria_ai.yolov26.models import build_model


def _model_config(task, model_id):
    if task == "detection":
        from kria_ai.yolov26.detection.config import get_model
    else:
        from kria_ai.yolov26.segmentation.config import get_model
    return get_model(model_id)


def _device(name):
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def inspect_yolov26(
    task: str,
    *,
    model_id: str | None = None,
    checkpoint: str | None = None,
    board: str | None = None,
    build_root: str = "build",
    device: str = "auto",
):
    """Run Vitis AI Inspector for YOLOv26 detection or segmentation models."""
    if task not in ("detection", "segmentation"):
        raise ValueError(f"task must be 'detection' or 'segmentation', got {task!r}")

    import torch

    model_config = _model_config(task, model_id)
    board_config = get_board(board)
    torch_device = _device(device)
    model = build_model(model_config, device=torch_device, checkpoint_path=checkpoint)
    contract = apply_export_patch(model, model_config)
    height, width = model_config.input_size
    example_input = torch.randn(1, 3, height, width, device=torch_device)
    with torch.no_grad():
        validate_export_outputs(model(example_input), contract)
    output_dir = ArtifactPaths(model_config.id, build_root).inspector_dir
    return inspect_model(
        model=model,
        example_inputs=(example_input,),
        target=board_config.dpu_fingerprint,
        device=torch_device,
        output_dir=output_dir,
    )


def inspect_detection(**kwargs):
    return inspect_yolov26("detection", **kwargs)


def inspect_segmentation(**kwargs):
    return inspect_yolov26("segmentation", **kwargs)


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(prog="python -m kria_ai yolov26 <task> inspect")
    parser.add_argument("--task", required=True, choices=("detection", "segmentation"))
    parser.add_argument("--model")
    parser.add_argument("--checkpoint")
    parser.add_argument("--board")
    parser.add_argument("--build-root", default="build")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    output_dir = inspect_yolov26(
        args.task,
        model_id=args.model,
        checkpoint=args.checkpoint,
        board=args.board,
        build_root=args.build_root,
        device=args.device,
    )
    print(f"Inspector report: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
