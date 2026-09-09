"""Segmentation-only Vitis AI structured pruning for YOLOv26.

PyTorch, Ultralytics, OpenCV, and Vitis imports are intentionally deferred until
after argument parsing so ``--help`` remains available outside the Vitis AI
container.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any


_MODES_WITH_SEARCH = frozenset(("search", "all"))
_MODES_WITH_PRUNING = frozenset(("prune", "finetune", "all"))
_MODES_WITH_FINETUNING = frozenset(("finetune", "all"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m kria_ai yolov26 segmentation optimize",
        description="Vitis AI 3.5 structured pruning for YOLOv26 segmentation",
    )
    parser.add_argument("--model", help="Segmentation model ID")
    parser.add_argument("--dataset", help="Segmentation dataset ID")
    parser.add_argument(
        "--method",
        choices=("iterative", "one_step"),
        default="one_step",
        help="Vitis coarse-grained pruning method",
    )
    parser.add_argument(
        "--mode",
        choices=("search", "prune", "finetune", "all"),
        default="all",
        help="Run search, pruning, fine-tuning, or the complete pipeline",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.2,
        help="Requested MAC removal ratio (strictly between zero and one)",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Fine-tuning epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Fine-tuning learning rate")
    parser.add_argument(
        "--subset-len",
        "--subset",
        dest="subset_len",
        type=int,
        default=200,
        help="Number of train/validation samples used by optimizer callbacks",
    )
    parser.add_argument(
        "--batch-size",
        "--batch",
        dest="batch_size",
        type=int,
        default=4,
        help="Segmentation data-loader batch size",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--channel-divisible",
        type=int,
        default=2,
        help="Require remaining pruned channels to be divisible by this value",
    )
    parser.add_argument("--device", default="auto", help="PyTorch device (auto, cpu, cuda, cuda:N)")
    parser.add_argument(
        "--checkpoint",
        help="Local source YOLO segmentation checkpoint (defaults to the registry entry)",
    )
    parser.add_argument(
        "--output-checkpoint",
        help="Output checkpoint path (defaults under the model optimizer report)",
    )
    parser.add_argument("--num-subnets", type=int, default=200, help="One-step candidate count")
    parser.add_argument("--build-root", default="build", help="Build artifact root")
    return parser


def _validate_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not 0.0 < args.ratio < 1.0:
        parser.error("--ratio must be strictly between 0 and 1")
    if args.epochs < 0:
        parser.error("--epochs must be non-negative")
    if args.mode in _MODES_WITH_FINETUNING and args.epochs < 1:
        parser.error("--epochs must be positive for finetune and all modes")
    if args.lr <= 0.0:
        parser.error("--lr must be positive")
    if args.subset_len < 1:
        parser.error("--subset-len must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.channel_divisible < 1:
        parser.error("--channel-divisible must be positive")
    if args.num_subnets < 1:
        parser.error("--num-subnets must be positive")


def _resolve_device(torch_module: Any, name: str):
    if name == "auto":
        name = "cuda" if torch_module.cuda.is_available() else "cpu"
    device = torch_module.device(name)
    if device.type == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {name}")
    return device


def _seed_everything(torch_module: Any, seed: int) -> None:
    import random

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)


def _install_vitis_deepcopy_workaround(torch_module: Any) -> None:
    """Work around Vitis' traced-tensor ``new_empty`` deepcopy failure."""

    current = torch_module.Tensor.__deepcopy__
    if getattr(current, "_kria_vitis_safe_deepcopy", False):
        return

    def safe_deepcopy(tensor, memo):
        try:
            return current(tensor, memo)
        except RuntimeError as error:
            if "new_empty" not in str(error):
                raise
            if isinstance(tensor, torch_module.nn.Parameter):
                copied = torch_module.nn.Parameter(
                    tensor.clone(), requires_grad=tensor.requires_grad
                )
            else:
                copied = tensor.clone()
            memo[id(tensor)] = copied
            return copied

    safe_deepcopy._kria_vitis_safe_deepcopy = True
    safe_deepcopy._kria_vitis_original_deepcopy = current
    torch_module.Tensor.__deepcopy__ = safe_deepcopy


def _move_targets(targets: dict[str, Any], device: Any, torch_module: Any):
    return {
        name: value.to(device) if isinstance(value, torch_module.Tensor) else value
        for name, value in targets.items()
    }


def segmentation_objective(model: Any, dataloader: Any) -> float:
    """Return negative YOLOv26 E2E mask loss for Vitis' maximize API."""

    from kria_ai.yolov26.segmentation.evaluate import evaluate_loss

    return -float(evaluate_loss(model, dataloader))


def calibrate_batch_norm(model: Any, dataloader: Any) -> None:
    """Refresh BN statistics through the segmentation training/E2E forward."""

    import torch

    from kria_ai.yolov26.segmentation.loss import forward_for_loss, prepare_loss_model

    prepare_loss_model(model)
    device = next(model.parameters()).device
    was_training = model.training
    model.train()
    try:
        with torch.no_grad():
            for images, _targets in dataloader:
                forward_for_loss(model, images.to(device))
    finally:
        if not was_training:
            model.eval()


def finetune_segmentation(
    model: Any,
    dataloader: Any,
    *,
    epochs: int,
    lr: float,
) -> list[float]:
    """Fine-tune a sparse or slim model with YOLOv26 E2E mask loss."""

    import torch

    from kria_ai.yolov26.segmentation.loss import create_loss, forward_for_loss, reduce_loss

    for parameter in model.parameters():
        parameter.requires_grad = True
    loss_fn = create_loss(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    device = next(model.parameters()).device
    epoch_losses: list[float] = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        batches = 0
        for images, targets in dataloader:
            images = images.to(device)
            targets = _move_targets(targets, device, torch)
            optimizer.zero_grad(set_to_none=True)
            outputs = forward_for_loss(model, images)
            loss = reduce_loss(loss_fn(outputs, targets))
            if not isinstance(loss, torch.Tensor):
                raise TypeError(
                    "YOLOv26 segmentation loss must reduce to a tensor, "
                    f"got {type(loss).__name__}"
                )
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
            batches += 1
        if not batches:
            raise RuntimeError("The segmentation fine-tuning loader produced no batches")
        scheduler.step()
        average_loss = total_loss / batches
        epoch_losses.append(average_loss)
        print(f"epoch={epoch + 1}/{epochs} segmentation_loss={average_loss:.6f}")

    model.eval()
    return epoch_losses


def _vitis_gpus(torch_module: Any, device: Any) -> list[str]:
    if device.type != "cuda":
        return []
    index = device.index
    if index is None:
        index = torch_module.cuda.current_device()
    return [str(index)]


def _run_search(
    runner: Any,
    *,
    method: str,
    train_loader: Any,
    validation_loader: Any,
    ratio: float,
    excludes: list[Any],
    gpus: list[str],
    num_subnets: int,
) -> None:
    if method == "iterative":
        runner.ana(
            segmentation_objective,
            args=(validation_loader,),
            gpus=gpus or None,
            excludes=excludes,
        )
        return

    runner.search(
        gpus=gpus,
        calibration_fn=calibrate_batch_norm,
        calib_args=(train_loader,),
        removal_ratio=ratio,
        excludes=excludes,
        eval_fn=segmentation_objective,
        eval_args=(validation_loader,),
        num_subnet=num_subnets,
    )


def _prune_model(
    runner: Any,
    *,
    method: str,
    ratio: float,
    excludes: list[Any],
    channel_divisible: int,
    pruning_info_path: Path,
):
    common = {
        "removal_ratio": ratio,
        "channel_divisible": channel_divisible,
        "pruning_info_path": str(pruning_info_path),
    }
    if method == "iterative":
        return runner.prune(mode="sparse", excludes=excludes, **common)
    return runner.prune(mode="slim", index=None, **common)


def _materialize_iterative_slim_model(
    runner: Any,
    sparse_model: Any,
    *,
    ratio: float,
    excludes: list[Any],
    channel_divisible: int,
    pruning_info_path: Path,
):
    """Transfer fine-tuned sparse weights into the reduced slim topology."""

    if not hasattr(sparse_model, "slim_state_dict"):
        raise RuntimeError("Vitis iterative sparse model does not expose slim_state_dict()")
    slim_state_dict = sparse_model.slim_state_dict()
    slim_model = runner.prune(
        removal_ratio=ratio,
        excludes=excludes,
        mode="slim",
        pruning_info_path=str(pruning_info_path),
        channel_divisible=channel_divisible,
    )
    slim_model.load_state_dict(slim_state_dict, strict=True)
    slim_model.eval()
    return slim_model, slim_state_dict


@contextmanager
def _working_directory(path: Path):
    """Keep Vitis' implicit ``.vai`` cache inside the optimizer artifact dir."""

    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _portable_state_dict(model: Any):
    state_dict = model.state_dict()
    portable = state_dict.__class__()
    for name, value in state_dict.items():
        portable[name] = value.detach().cpu() if hasattr(value, "detach") else value
    if hasattr(state_dict, "_metadata"):
        portable._metadata = state_dict._metadata
    return portable


def _metric_summary(metrics: dict[str, Any] | None):
    if metrics is None:
        return None
    return {
        "params": metrics["params"],
        "trainable_params": metrics["trainable_params"],
        "size_mb": metrics["size_mb"],
        "gflops": metrics["gflops"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_arguments(parser, args)

    import torch

    from kria_ai.common.artifacts import ArtifactPaths, write_manifest
    from kria_ai.common.model_metrics import (
        collect_model_metrics,
        metrics_from_slim_state_dict,
        save_metrics_report,
    )
    from kria_ai.yolov26.models import build_model, resolve_project_path
    from kria_ai.yolov26.pruning import create_pruning_runner, resolve_prune_excludes
    from kria_ai.yolov26.segmentation.config import get_dataset, get_model
    from kria_ai.yolov26.segmentation.data import build_loader

    _seed_everything(torch, args.seed)
    _install_vitis_deepcopy_workaround(torch)
    device = _resolve_device(torch, args.device)
    if args.method == "iterative" and args.mode in _MODES_WITH_SEARCH and device.type != "cuda":
        raise RuntimeError("Vitis AI 3.5 iterative analysis requires a CUDA device")
    model_config = get_model(args.model)
    dataset_config = get_dataset(args.dataset)

    artifacts = ArtifactPaths(model_config.id, Path(args.build_root).expanduser())
    output_dir = artifacts.ensure(artifacts.optimizer_dir).resolve()
    manifest_path = output_dir / "manifest.json"
    metrics_path = output_dir / "metrics.json"
    default_checkpoint = output_dir / (
        f"{model_config.id}_{args.method}_r{args.ratio:g}_optimized.pt"
    )
    output_checkpoint = (
        Path(args.output_checkpoint).expanduser().resolve()
        if args.output_checkpoint
        else default_checkpoint
    )

    source_checkpoint = resolve_project_path(
        args.checkpoint if args.checkpoint is not None else model_config.checkpoint_path
    )
    model = build_model(
        model_config,
        device=device,
        checkpoint_path=args.checkpoint,
    )
    height, width = model_config.input_size
    input_shape = (height, width)

    before_metrics = None
    if args.mode in _MODES_WITH_PRUNING:
        before_metrics = collect_model_metrics(model, input_shape)

    example_input = torch.randn(1, 3, height, width, device=device)
    excludes = resolve_prune_excludes(model, model_config)
    print(
        f"model={model_config.id} task=segmentation method={args.method} "
        f"mode={args.mode} device={device} excludes={len(excludes)}"
    )

    search_requested = args.mode in _MODES_WITH_SEARCH
    finetune_requested = args.mode in _MODES_WITH_FINETUNING
    if args.method == "one_step" and args.mode in _MODES_WITH_PRUNING and not search_requested:
        search_files = list((output_dir / ".vai").glob(f"*_ratio_{args.ratio}.search"))
        if not search_files:
            raise RuntimeError(
                "One-step prune/finetune mode requires a cached search result; run --mode search or --mode all first"
            )
    train_loader = None
    validation_loader = None
    if finetune_requested or (search_requested and args.method == "one_step"):
        train_loader = build_loader(
            model_config,
            dataset_config,
            split="train",
            subset_len=args.subset_len,
            batch_size=args.batch_size,
            seed=args.seed,
            shuffle=finetune_requested,
            augment=finetune_requested,
        )
    if search_requested:
        validation_loader = build_loader(
            model_config,
            dataset_config,
            split="validation",
            subset_len=args.subset_len,
            batch_size=args.batch_size,
            seed=args.seed,
            shuffle=False,
            augment=False,
        )

    pruned_model = None
    final_model = None
    iterative_slim_state_dict = None
    training_losses: list[float] = []
    with _working_directory(output_dir):
        runner = create_pruning_runner(model, example_input, method=args.method)
        if search_requested:
            _run_search(
                runner,
                method=args.method,
                train_loader=train_loader,
                validation_loader=validation_loader,
                ratio=args.ratio,
                excludes=excludes,
                gpus=_vitis_gpus(torch, device),
                num_subnets=args.num_subnets,
            )

        if args.mode in _MODES_WITH_PRUNING:
            pruned_model = _prune_model(
                runner,
                method=args.method,
                ratio=args.ratio,
                excludes=excludes,
                channel_divisible=args.channel_divisible,
                pruning_info_path=output_dir / f"pruning_info_{args.method}.json",
            )
            pruned_model.to(device)
            if finetune_requested:
                training_losses = finetune_segmentation(
                    pruned_model,
                    train_loader,
                    epochs=args.epochs,
                    lr=args.lr,
                )

            if args.method == "iterative":
                final_model, iterative_slim_state_dict = _materialize_iterative_slim_model(
                    runner,
                    pruned_model,
                    ratio=args.ratio,
                    excludes=excludes,
                    channel_divisible=args.channel_divisible,
                    pruning_info_path=output_dir / "pruning_info_iterative_slim.json",
                )
            else:
                final_model = pruned_model
                final_model.eval()

    after_metrics = None
    if final_model is not None:
        if iterative_slim_state_dict is not None:
            after_metrics = metrics_from_slim_state_dict(
                pruned_model,
                iterative_slim_state_dict,
                input_shape,
                before_metrics,
            )
        else:
            after_metrics = collect_model_metrics(final_model, input_shape)
        save_metrics_report(
            metrics_path,
            model_config.id,
            before_metrics,
            after_metrics,
        )

    checkpoint_representation = "slim" if final_model is not None else None
    optimization = {
        "format_version": 1,
        "task": "segmentation",
        "model_id": model_config.id,
        "dataset_id": dataset_config.id,
        "method": args.method,
        "mode": args.mode,
        "ratio": args.ratio,
        "channel_divisible": args.channel_divisible,
        "epochs": args.epochs if finetune_requested else 0,
        "lr": args.lr,
        "subset_len": args.subset_len,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "num_subnets": args.num_subnets,
        "source_checkpoint": str(source_checkpoint),
        "checkpoint_representation": checkpoint_representation,
        "pruning_lifecycle": (
            "sparse_to_slim"
            if final_model is not None and args.method == "iterative"
            else "slim"
            if final_model is not None
            else "search_only"
        ),
        "output_head_excludes": list(model_config.prune_excludes),
        "training_loss": training_losses,
        "metrics": {
            "before": _metric_summary(before_metrics),
            "after": _metric_summary(after_metrics),
        },
        "vitis_cache": str(output_dir / ".vai"),
        "manifest": str(manifest_path),
    }

    saved_checkpoint = None
    if final_model is not None:
        output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": _portable_state_dict(final_model),
                "optimization": optimization,
            },
            output_checkpoint,
        )
        saved_checkpoint = output_checkpoint
        print(f"Optimized segmentation checkpoint: {output_checkpoint}")

    manifest = {
        "stage": "optimizer",
        "task": "segmentation",
        "artifact_type": "yolov26_segmentation_optimizer",
        "model_id": model_config.id,
        "dataset_id": dataset_config.id,
        "checkpoint": str(saved_checkpoint) if saved_checkpoint is not None else None,
        "metrics_report": str(metrics_path) if after_metrics is not None else None,
        "vitis_cache": str((output_dir / ".vai").resolve()),
        "optimization": optimization,
    }
    write_manifest(manifest_path, manifest)
    print(f"Optimizer manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
