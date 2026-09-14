import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class QuantizationResult:
    quant_model: object
    output_dir: Path
    processed_samples: int
    quant_config_path: Path | None
    xmodel_path: Path | None
    manifest_path: Path


def _resolve_exported_xmodel(output_dir, model, requested_filename, previous_timestamp=None):
    requested_path = output_dir / requested_filename
    generated_path = output_dir / f"{model.__class__.__name__}_int.xmodel"
    if generated_path != requested_path and generated_path.is_file():
        generated_path.replace(requested_path)
        return requested_path
    if requested_path.is_file():
        current_timestamp = requested_path.stat().st_mtime_ns
        if previous_timestamp is None or current_timestamp != previous_timestamp:
            return requested_path
    candidates = [path for path in output_dir.glob("*_int.xmodel") if path != requested_path]
    if len(candidates) != 1:
        names = ", ".join(path.name for path in candidates) or "none"
        raise RuntimeError(f"Unable to identify exported XMODEL in {output_dir}; candidates: {names}")
    candidates[0].replace(requested_path)
    return requested_path


def run_quantization(
    *,
    mode,
    model,
    example_inputs,
    batches,
    adapt_batch,
    device,
    output_dir,
    xmodel_filename=None,
    fast_finetune=None,
    load_fast_finetune=False,
    target=None,
    max_samples=None,
    quantizer_factory=None,
):
    import torch

    if mode not in {"calib", "test"}:
        raise ValueError(f"Unsupported quantization mode: {mode!r}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    example_inputs = tuple(example_inputs)
    if mode == "test" and example_inputs and hasattr(example_inputs[0], "shape"):
        if example_inputs[0].shape[0] != 1:
            raise ValueError("XMODEL export requires a batch-one example input")
    if quantizer_factory is None:
        from pytorch_nndct.apis import torch_quantizer

        quantizer_factory = torch_quantizer
    quantizer = quantizer_factory(
        mode,
        model,
        example_inputs,
        device=device,
        output_dir=str(output_dir),
        target=target,
    )
    quant_model = quantizer.quant_model
    quant_model.eval()
    if mode == "calib" and fast_finetune is not None:
        quantizer.fast_finetune(fast_finetune, (quant_model,))
    elif mode == "test" and load_fast_finetune:
        quantizer.load_ft_param()

    processed_samples = 0
    with torch.no_grad():
        for batch in batches:
            positional, keyword, sample_count = adapt_batch(batch, device)
            quant_model(*positional, **keyword)
            processed_samples += sample_count
            if max_samples is not None and processed_samples >= max_samples:
                break
    if processed_samples == 0:
        raise RuntimeError("Quantization requires at least one successful forward batch")

    quant_config_path = None
    xmodel_path = None
    if mode == "calib":
        quantizer.export_quant_config()
        quant_config_path = output_dir / "quant_info.json"
        if not quant_config_path.is_file():
            raise RuntimeError(f"Quantizer did not produce {quant_config_path}")
    else:
        if not xmodel_filename:
            raise ValueError("xmodel_filename is required in test mode")
        requested_path = output_dir / xmodel_filename
        previous_timestamp = requested_path.stat().st_mtime_ns if requested_path.is_file() else None
        quantizer.export_xmodel(output_dir=str(output_dir), deploy_check=False)
        xmodel_path = _resolve_exported_xmodel(
            output_dir,
            model,
            xmodel_filename,
            previous_timestamp=previous_timestamp,
        )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "stage": "quantization",
                "mode": mode,
                "processed_samples": processed_samples,
                "target": target,
                "quant_config": str(quant_config_path) if quant_config_path else None,
                "xmodel": str(xmodel_path) if xmodel_path else None,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return QuantizationResult(
        quant_model=quant_model,
        output_dir=output_dir,
        processed_samples=processed_samples,
        quant_config_path=quant_config_path,
        xmodel_path=xmodel_path,
        manifest_path=manifest_path,
    )
