import argparse
from collections.abc import Sequence


def evaluate_loss(model, dataloader, loss_fn=None, device=None):
    import torch

    from kria_ai.yolov26.detection.loss import create_loss, forward_for_loss, reduce_loss

    device = device or next(model.parameters()).device
    loss_fn = loss_fn or create_loss(model)
    total = 0.0
    batches = 0
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = {
                name: value.to(device) if isinstance(value, torch.Tensor) else value
                for name, value in targets.items()
            }
            total += float(reduce_loss(loss_fn(forward_for_loss(model, images), targets)))
            batches += 1
    return total / batches if batches else 0.0


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(prog="python -m kria_ai yolov26 detection evaluate")
    parser.add_argument("--model")
    parser.add_argument("--dataset")
    parser.add_argument("--checkpoint")
    parser.add_argument("--subset-len", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    import torch

    from kria_ai.yolov26.detection.config import get_dataset, get_model
    from kria_ai.yolov26.detection.data import build_loader
    from kria_ai.yolov26.models import build_model

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    model_config = get_model(args.model)
    dataset_config = get_dataset(args.dataset)
    model = build_model(model_config, device=device, checkpoint_path=args.checkpoint)
    loader = build_loader(
        model_config,
        dataset_config,
        split="validation",
        subset_len=args.subset_len,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    loss = evaluate_loss(model, loader, device=device)
    print(f"samples={len(loader.dataset)} detection_loss={loss:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
