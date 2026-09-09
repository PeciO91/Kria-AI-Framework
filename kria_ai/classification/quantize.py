import argparse
from collections.abc import Sequence

from kria_ai.classification.config import get_dataset, get_model, validate_model_dataset
from kria_ai.classification.data import build_loader
from kria_ai.classification.evaluate import evaluate_loss
from kria_ai.classification.models import build_model
from kria_ai.common.artifacts import ArtifactPaths
from kria_ai.common.vitis.quantizer import run_quantization


def _device(name):
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _adapt_batch(batch, device):
    images = batch[0].to(device)
    return (images,), {}, images.size(0)


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(prog="python -m kria_ai classification quantize")
    parser.add_argument("--model")
    parser.add_argument("--dataset")
    parser.add_argument("--checkpoint")
    parser.add_argument("--mode", "--quant-mode", choices=("calib", "test"), default="calib")
    parser.add_argument("--subset-len", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--build-root", default="build")
    parser.add_argument("--fast-ft", action="store_true")
    args = parser.parse_args(argv)

    import torch

    model_config = get_model(args.model)
    dataset_config = get_dataset(args.dataset)
    validate_model_dataset(model_config, dataset_config)
    device = _device(args.device)
    torch.manual_seed(args.seed)
    model, incompatible = build_model(model_config, device=device, checkpoint_path=args.checkpoint)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(f"[WARN] Missing keys: {incompatible.missing_keys}; unexpected keys: {incompatible.unexpected_keys}")

    subset_len = 1 if args.mode == "test" else args.subset_len
    batch_size = 1 if args.mode == "test" else args.batch_size
    loader = build_loader(
        model_config,
        dataset_config,
        split="calibration",
        subset_len=subset_len,
        batch_size=batch_size,
        seed=args.seed,
    )
    fast_finetune = None
    if args.fast_ft and args.mode == "calib":
        fast_finetune = lambda quant_model: evaluate_loss(quant_model, loader)

    height, width = model_config.input_size
    example_input = torch.randn(1, 3, height, width, device=device)
    artifacts = ArtifactPaths(model_config.id, args.build_root)
    result = run_quantization(
        mode=args.mode,
        model=model,
        example_inputs=(example_input,),
        batches=loader,
        adapt_batch=_adapt_batch,
        device=device,
        output_dir=artifacts.quantize_dir,
        xmodel_filename=artifacts.quantized_xmodel.name,
        fast_finetune=fast_finetune,
        load_fast_finetune=args.fast_ft and args.mode == "test",
        max_samples=subset_len,
    )
    output = result.quant_config_path or result.xmodel_path
    print(f"Quantization {args.mode} complete: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
