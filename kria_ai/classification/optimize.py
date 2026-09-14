"""Classification-only Vitis AI structured-pruning pipeline.

Heavy framework imports intentionally live behind :func:`main` so command help
remains available outside the PyTorch/Vitis AI environment.
"""

from __future__ import annotations

import argparse
import importlib
import os
import random
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any


METHODS = ("iterative", "one_step")
MODES = ("search", "prune", "finetune", "all")
DEFAULT_NUM_SUBNETS = 1000
DEFAULT_BN_CALIBRATION_BATCHES = 100
WEIGHT_DECAY = 5e-4


def _ratio(value: str) -> float:
    ratio = float(value)
    if not 0.0 < ratio < 1.0:
        raise argparse.ArgumentTypeError("ratio must be greater than 0 and less than 1")
    return ratio


def _positive_float(value: str) -> float:
    number = float(value)
    if not number > 0.0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return number


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return number


def _nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m kria_ai classification optimize",
        description="Vitis AI structured pruning for classification models",
    )
    parser.add_argument("--model", help="Classification model ID")
    parser.add_argument("--dataset", help="Classification dataset ID")
    parser.add_argument(
        "--method",
        choices=METHODS,
        default="one_step",
        help="Vitis AI pruning method (default: %(default)s)",
    )
    parser.add_argument(
        "--mode",
        choices=MODES,
        default="all",
        help="Optimization stage to run (default: %(default)s)",
    )
    parser.add_argument(
        "--ratio",
        type=_ratio,
        default=0.2,
        help="Requested MAC/channel removal ratio (default: %(default)s)",
    )
    parser.add_argument(
        "--epochs",
        type=_positive_int,
        default=5,
        help="Cross-entropy fine-tuning epochs (default: %(default)s)",
    )
    parser.add_argument(
        "--lr",
        type=_positive_float,
        default=1e-3,
        help="Fine-tuning learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--subset-len",
        type=_positive_int,
        default=200,
        help="Maximum samples used by search and fine-tuning (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=_positive_int,
        default=4,
        help="Data-loader batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=_nonnegative_int,
        default=42,
        help="Random seed (default: %(default)s)",
    )
    parser.add_argument(
        "--channel-divisible",
        type=_positive_int,
        default=2,
        help="Required divisor for remaining channels (default: %(default)s)",
    )
    parser.add_argument(
        "--num-subnets",
        type=_positive_int,
        default=DEFAULT_NUM_SUBNETS,
        help="Candidate count for one-step search (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch device, for example auto, cpu, cuda, or cuda:0",
    )
    parser.add_argument("--checkpoint", help="Source model checkpoint override")
    parser.add_argument(
        "--output-checkpoint",
        help="Optimized checkpoint path (default: canonical optimizer directory)",
    )
    parser.add_argument(
        "--build-root",
        default="build",
        help="Artifact build root (default: %(default)s)",
    )
    return parser


def _device(name: str, torch: Any) -> Any:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _runner_gpus(device: Any, torch: Any) -> list[str]:
    if device.type != "cuda":
        return []
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    return [str(index)]


def _load_pruning_runner_factory() -> Any:
    try:
        pytorch_nndct = importlib.import_module("pytorch_nndct")
    except ImportError as error:
        raise ImportError(
            "Classification optimization requires pytorch_nndct from the "
            "Vitis AI 3.5 PyTorch environment"
        ) from error
    try:
        return pytorch_nndct.get_pruning_runner
    except AttributeError as error:
        raise ImportError(
            "Installed pytorch_nndct does not expose get_pruning_runner"
        ) from error


def _install_tensor_deepcopy_workaround(torch: Any) -> None:
    """Work around the Vitis one-step ``new_empty`` deepcopy failure."""

    current = torch.Tensor.__deepcopy__
    if getattr(current, "_kria_vitis_safe_deepcopy", False):
        return

    def safe_deepcopy(tensor: Any, memo: dict[int, Any]) -> Any:
        try:
            return current(tensor, memo)
        except RuntimeError as error:
            if "new_empty" not in str(error):
                raise
            clone = tensor.clone()
            if isinstance(tensor, torch.nn.Parameter):
                clone = torch.nn.Parameter(clone, requires_grad=tensor.requires_grad)
            memo[id(tensor)] = clone
            return clone

    safe_deepcopy._kria_vitis_safe_deepcopy = True
    torch.Tensor.__deepcopy__ = safe_deepcopy


def _top1_evaluator(model: Any, dataloader: Any) -> float:
    """Return classification top-1 accuracy for Vitis' maximize contract."""

    from kria_ai.classification.evaluate import evaluate

    device = next(model.parameters()).device
    return float(evaluate(model, dataloader, device=device)["top1"])


def _calibrate_batch_norm(
    model: Any,
    dataloader: Any,
    maximum_batches: int = DEFAULT_BN_CALIBRATION_BATCHES,
) -> None:
    """Refresh BatchNorm statistics for one-step candidate evaluation."""

    import torch

    model.train()
    device = next(model.parameters()).device
    with torch.no_grad():
        for batch_index, (images, _) in enumerate(dataloader):
            if batch_index >= maximum_batches:
                break
            model(images.to(device))


def _snapshot_state_dict(model: Any) -> dict[str, Any]:
    snapshot = {}
    for name, value in model.state_dict().items():
        snapshot[name] = value.detach().cpu().clone() if hasattr(value, "detach") else value
    return snapshot


def _finetune_classification(
    model: Any,
    train_loader: Any,
    evaluation_loader: Any,
    device: Any,
    epochs: int,
    learning_rate: float,
) -> tuple[Any, list[dict[str, float]]]:
    """Fine-tune with cross-entropy and restore the best top-1 epoch."""

    import torch

    from kria_ai.classification.evaluate import evaluate

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_top1 = float("-inf")
    best_state = None
    history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            batch_size = images.size(0)
            total_loss += loss.detach().item() * batch_size
            total_samples += batch_size
        scheduler.step()

        metrics = evaluate(model, evaluation_loader, device=device, loss_fn=criterion)
        train_loss = total_loss / max(total_samples, 1)
        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "loss": float(metrics["loss"]),
            "top1": float(metrics["top1"]),
            "top5": float(metrics["top5"]),
        }
        history.append(epoch_metrics)
        print(
            f"Epoch {epoch + 1}/{epochs}: train_loss={train_loss:.6f} "
            f"val_loss={metrics['loss']:.6f} top1={metrics['top1']:.2f}% "
            f"top5={metrics['top5']:.2f}%"
        )
        if epoch_metrics["top1"] > best_top1:
            best_top1 = epoch_metrics["top1"]
            best_state = _snapshot_state_dict(model)

    if best_state is not None:
        # Sparse Vitis models own mask buffers that their state-dict hook omits.
        model.load_state_dict(best_state, strict=False)
    model.eval()
    return model, history


def _prune(
    runner: Any,
    method: str,
    ratio: float,
    channel_divisible: int,
) -> tuple[Any, str]:
    if method == "iterative":
        model = runner.prune(
            removal_ratio=ratio,
            mode="sparse",
            channel_divisible=channel_divisible,
        )
        return model, "sparse"
    model = runner.prune(
        removal_ratio=ratio,
        mode="slim",
        index=None,
        channel_divisible=channel_divisible,
    )
    return model, "slim"


def _convert_sparse_to_slim(
    sparse_model: Any,
    runner: Any,
    ratio: float,
    channel_divisible: int,
) -> Any:
    """Materialize iterative sparse weights in the final slim architecture."""

    if not hasattr(sparse_model, "slim_state_dict"):
        raise RuntimeError(
            "Vitis iterative sparse model does not expose slim_state_dict()"
        )
    slim_state_dict = sparse_model.slim_state_dict()
    slim_model = runner.prune(
        removal_ratio=ratio,
        mode="slim",
        channel_divisible=channel_divisible,
    )
    slim_model.load_state_dict(slim_state_dict, strict=True)
    slim_model.eval()
    return slim_model


def _compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "params": int(metrics["params"]),
        "trainable_params": int(metrics["trainable_params"]),
        "size_mb": float(metrics["size_mb"]),
        "gflops": float(metrics["gflops"]),
    }


@contextmanager
def _working_directory(path: Path):
    """Keep Vitis' implicit ``.vai`` cache inside the optimizer artifact dir."""

    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _resolved_source_checkpoint(model_config: Any, checkpoint: str | None) -> Path:
    source = Path(checkpoint) if checkpoint else Path(model_config.checkpoint_path)
    if not source.is_absolute():
        source = Path(__file__).resolve().parents[2] / source
    return source.expanduser().resolve()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # These imports are deliberately after argument parsing so --help has no
    # dependency on PyTorch, torchvision, or Vitis AI.
    import torch

    from kria_ai.classification.config import (
        get_dataset,
        get_model,
        validate_model_dataset,
    )
    from kria_ai.classification.data import build_loader
    from kria_ai.classification.models import build_model
    from kria_ai.common.artifacts import ArtifactPaths, write_manifest
    from kria_ai.common.model_metrics import (
        collect_model_metrics,
        metrics_from_slim_state_dict,
        save_metrics_report,
    )

    model_config = get_model(args.model)
    dataset_config = get_dataset(args.dataset)
    validate_model_dataset(model_config, dataset_config)
    device = _device(args.device, torch)
    if args.method == "iterative" and args.mode in {"search", "all"} and device.type != "cuda":
        raise RuntimeError("Vitis AI 3.5 iterative analysis requires a CUDA device")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    artifacts = ArtifactPaths(model_config.id, args.build_root)
    optimizer_dir = artifacts.ensure(artifacts.optimizer_dir).resolve()
    output_checkpoint = (
        Path(args.output_checkpoint).expanduser().resolve()
        if args.output_checkpoint
        else optimizer_dir / f"{model_config.id}_{args.method}_r{args.ratio:g}_optimized.pt"
    )
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    source_checkpoint = _resolved_source_checkpoint(model_config, args.checkpoint)

    model, incompatible = build_model(
        model_config,
        device=device,
        checkpoint_path=source_checkpoint,
        strict=False,
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(
            f"[WARN] Missing keys: {incompatible.missing_keys}; "
            f"unexpected keys: {incompatible.unexpected_keys}"
        )

    before_metrics = collect_model_metrics(model, model_config.input_size)
    height, width = model_config.input_size
    example_input = torch.randn(1, 3, height, width, device=device)

    needs_search = args.mode in {"search", "all"}
    needs_pruned_model = args.mode in {"prune", "finetune", "all"}
    needs_finetune = args.mode in {"finetune", "all"}
    if args.method == "one_step" and needs_pruned_model and not needs_search:
        search_files = list((optimizer_dir / ".vai").glob(f"*_ratio_{args.ratio}.search"))
        if not search_files:
            raise RuntimeError(
                "One-step prune/finetune mode requires a cached search result; run --mode search or --mode all first"
            )
    if needs_pruned_model and model_config.constructor.startswith("mobilenet"):
        raise RuntimeError(
            "Vitis AI 3.5 slim checkpoint reconstruction does not support MobileNet depthwise groups; "
            "classification pruning is currently limited to ResNet models"
        )
    needs_train_loader = needs_finetune or (needs_search and args.method == "one_step")
    needs_evaluation_loader = needs_search or needs_finetune

    train_loader = None
    if needs_train_loader:
        train_loader = build_loader(
            model_config,
            dataset_config,
            split="calibration",
            subset_len=args.subset_len,
            batch_size=args.batch_size,
            seed=args.seed,
            shuffle=True,
        )
    evaluation_loader = None
    if needs_evaluation_loader:
        evaluation_loader = build_loader(
            model_config,
            dataset_config,
            split="evaluation",
            subset_len=args.subset_len,
            batch_size=args.batch_size,
            seed=args.seed,
            shuffle=False,
        )

    pruning_runner_factory = _load_pruning_runner_factory()
    _install_tensor_deepcopy_workaround(torch)
    current_model = model
    representation = "baseline"
    finetune_history: list[dict[str, float]] = []

    # Vitis AI 3.5 stores sensitivity/search/spec files below a process-relative
    # .vai directory. Run the lifecycle from the canonical optimizer directory
    # so separate search, prune, and finetune invocations share those results.
    with _working_directory(optimizer_dir):
        runner = pruning_runner_factory(model, example_input, args.method)
        gpus = _runner_gpus(device, torch)

        if needs_search:
            if args.method == "iterative":
                runner.ana(
                    _top1_evaluator,
                    args=(evaluation_loader,),
                    gpus=gpus,
                )
            else:
                runner.search(
                    gpus=gpus,
                    calibration_fn=_calibrate_batch_norm,
                    calib_args=(train_loader,),
                    eval_fn=_top1_evaluator,
                    eval_args=(evaluation_loader,),
                    num_subnet=args.num_subnets,
                    removal_ratio=args.ratio,
                )

        if needs_pruned_model:
            current_model, representation = _prune(
                runner,
                method=args.method,
                ratio=args.ratio,
                channel_divisible=args.channel_divisible,
            )
            current_model.to(device)

        if needs_finetune:
            current_model, finetune_history = _finetune_classification(
                current_model,
                train_loader,
                evaluation_loader,
                device=device,
                epochs=args.epochs,
                learning_rate=args.lr,
            )
        if needs_pruned_model and args.method == "iterative":
            current_model = _convert_sparse_to_slim(
                current_model,
                runner,
                ratio=args.ratio,
                channel_divisible=args.channel_divisible,
            )
            current_model.to(device)
            representation = "slim"

    if representation == "sparse" and hasattr(current_model, "slim_state_dict"):
        after_metrics = metrics_from_slim_state_dict(
            current_model,
            current_model.slim_state_dict(),
            model_config.input_size,
            before_metrics,
        )
    else:
        after_metrics = collect_model_metrics(current_model, model_config.input_size)

    metrics_path = save_metrics_report(
        optimizer_dir / "model_metrics.json",
        model_config.id,
        before_metrics,
        after_metrics,
    )
    optimization_metadata = {
        "task": "classification",
        "model_id": model_config.id,
        "dataset_id": dataset_config.id,
        "method": args.method,
        "mode": args.mode,
        "ratio": args.ratio,
        "epochs": args.epochs if needs_finetune else 0,
        "learning_rate": args.lr,
        "subset_len": args.subset_len,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "channel_divisible": args.channel_divisible,
        "device": str(device),
        "representation": representation,
        "search_completed": needs_search,
        "prune_completed": needs_pruned_model,
        "finetune_completed": needs_finetune,
        "source_checkpoint": str(source_checkpoint),
        "metrics": {
            "before": _compact_metrics(before_metrics),
            "after": _compact_metrics(after_metrics),
        },
        "finetune_history": finetune_history,
    }
    saved_checkpoint = None
    if needs_pruned_model:
        checkpoint_payload = {
            "state_dict": _snapshot_state_dict(current_model),
            "optimization": optimization_metadata,
        }
        torch.save(checkpoint_payload, output_checkpoint)
        saved_checkpoint = output_checkpoint

    manifest = {
        "stage": "optimizer",
        "task": "classification",
        "model_id": model_config.id,
        "dataset_id": dataset_config.id,
        "checkpoint": str(saved_checkpoint) if saved_checkpoint else None,
        "metrics_report": str(metrics_path.resolve()),
        "vitis_cache": str((optimizer_dir / ".vai").resolve()),
        "optimization": optimization_metadata,
    }
    manifest_path = write_manifest(optimizer_dir / "manifest.json", manifest)

    if saved_checkpoint:
        print(f"Optimized checkpoint: {saved_checkpoint}")
    print(f"Optimizer manifest: {manifest_path}")
    print(f"Model metrics: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
