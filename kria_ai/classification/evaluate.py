import argparse
from collections.abc import Sequence

from kria_ai.classification.config import get_dataset, get_model, validate_model_dataset
from kria_ai.classification.data import build_loader
from kria_ai.classification.models import build_model


def accuracy(outputs, targets, topk=(1, 5)):
    import torch

    with torch.no_grad():
        maximum = min(max(topk), outputs.shape[1])
        predictions = outputs.topk(maximum, dim=1, largest=True, sorted=True).indices.t()
        correct = predictions.eq(targets.reshape(1, -1).expand_as(predictions))
        return [
            correct[:min(k, maximum)].reshape(-1).float().sum().mul_(100.0 / targets.size(0))
            for k in topk
        ]


def evaluate(model, dataloader, device=None, loss_fn=None):
    import torch

    device = device or next(model.parameters()).device
    loss_fn = loss_fn or torch.nn.CrossEntropyLoss()
    model.eval()
    total = 0
    total_loss = 0.0
    top1_sum = 0.0
    top5_sum = 0.0
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            batch_size = images.size(0)
            top1, top5 = accuracy(outputs, targets)
            total += batch_size
            total_loss += float(loss_fn(outputs, targets)) * batch_size
            top1_sum += float(top1) * batch_size
            top5_sum += float(top5) * batch_size
    if total == 0:
        return {"samples": 0, "loss": 0.0, "top1": 0.0, "top5": 0.0}
    return {
        "samples": total,
        "loss": total_loss / total,
        "top1": top1_sum / total,
        "top5": top5_sum / total,
    }


def evaluate_loss(model, dataloader, loss_fn=None):
    return evaluate(model, dataloader, loss_fn=loss_fn)["loss"]


def _device(name):
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(prog="python -m kria_ai classification evaluate")
    parser.add_argument("--model")
    parser.add_argument("--dataset")
    parser.add_argument("--checkpoint")
    parser.add_argument("--subset-len", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    model_config = get_model(args.model)
    dataset_config = get_dataset(args.dataset)
    validate_model_dataset(model_config, dataset_config)
    device = _device(args.device)
    model, incompatible = build_model(model_config, device=device, checkpoint_path=args.checkpoint)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(f"[WARN] Missing keys: {incompatible.missing_keys}; unexpected keys: {incompatible.unexpected_keys}")
    loader = build_loader(
        model_config,
        dataset_config,
        split="evaluation",
        subset_len=args.subset_len,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    metrics = evaluate(model, loader, device=device)
    print(
        f"samples={metrics['samples']} loss={metrics['loss']:.6f} "
        f"top1={metrics['top1']:.2f}% top5={metrics['top5']:.2f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
