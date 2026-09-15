from __future__ import annotations

import argparse
from collections.abc import Sequence

from kria_ai.common.artifacts import ArtifactPaths
from kria_ai.common.board.config import get_board
from kria_ai.common.vitis.quantizer import run_quantization
from kria_ai.yolov26.data import build_calibration_loader
from kria_ai.yolov26.export import apply_export_patch, validate_export_outputs
from kria_ai.yolov26.models import build_model


def _configs(task, model_id, dataset_id):
    if task == "detection":
        from kria_ai.yolov26.detection.config import get_dataset, get_model
    else:
        from kria_ai.yolov26.segmentation.config import get_dataset, get_model
    return get_model(model_id), get_dataset(dataset_id)


def _device(name):
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _adapt_batch(batch, device):
    images = batch if hasattr(batch, "to") else batch[0]
    images = images.to(device)
    return (images,), {}, images.size(0)


def quantize_yolov26(
    task: str,
    *,
    model_id: str | None = None,
    dataset_id: str | None = None,
    checkpoint: str | None = None,
    mode: str = "calib",
    subset_len: int = 100,
    batch_size: int = 4,
    seed: int = 42,
    device: str = "auto",
    board: str | None = None,
    build_root: str = "build",
    fast_ft: bool = False,
):
    """Run Vitis AI quantization for YOLOv26 detection or segmentation models."""
    if task not in ("detection", "segmentation"):
        raise ValueError(f"task must be 'detection' or 'segmentation', got {task!r}")
    if fast_ft:
        raise ValueError("YOLOv26 AdaQuant is disabled until a raw-export-graph callback is validated")

    import torch

    model_config, dataset_config = _configs(task, model_id, dataset_id)
    board_config = get_board(board)
    torch_device = _device(device)
    torch.manual_seed(seed)
    model = build_model(model_config, device=torch_device, checkpoint_path=checkpoint)
    contract = apply_export_patch(model, model_config)
    height, width = model_config.input_size
    example_input = torch.randn(1, 3, height, width, device=torch_device)
    with torch.no_grad():
        validate_export_outputs(model(example_input), contract)

    sample_count = 1 if mode == "test" else subset_len
    batch_sz = 1 if mode == "test" else batch_size
    loader = build_calibration_loader(
        model_config,
        dataset_config,
        subset_len=sample_count,
        batch_size=batch_sz,
        seed=seed,
    )
    artifacts = ArtifactPaths(model_config.id, build_root)
    return run_quantization(
        mode=mode,
        model=model,
        example_inputs=(example_input,),
        batches=loader,
        adapt_batch=_adapt_batch,
        device=torch_device,
        output_dir=artifacts.quantize_dir,
        xmodel_filename=artifacts.quantized_xmodel.name,
        target=board_config.dpu_fingerprint,
        max_samples=sample_count,
    )


def quantize_detection(**kwargs):
    return quantize_yolov26("detection", **kwargs)


def quantize_segmentation(**kwargs):
    return quantize_yolov26("segmentation", **kwargs)


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(prog="python -m kria_ai yolov26 <task> quantize")
    parser.add_argument("--task", required=True, choices=("detection", "segmentation"))
    parser.add_argument("--model")
    parser.add_argument("--dataset")
    parser.add_argument("--checkpoint")
    parser.add_argument("--mode", "--quant-mode", choices=("calib", "test"), default="calib")
    parser.add_argument("--subset-len", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--board")
    parser.add_argument("--build-root", default="build")
    parser.add_argument("--fast-ft", action="store_true")
    args = parser.parse_args(argv)

    if args.fast_ft:
        parser.error("YOLOv26 AdaQuant is disabled until a raw-export-graph callback is validated")

    result = quantize_yolov26(
        args.task,
        model_id=args.model,
        dataset_id=args.dataset,
        checkpoint=args.checkpoint,
        mode=args.mode,
        subset_len=args.subset_len,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        board=args.board,
        build_root=args.build_root,
        fast_ft=args.fast_ft,
    )
    output = result.quant_config_path or result.xmodel_path
    print(f"YOLOv26 {args.task} quantization {args.mode} complete: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
