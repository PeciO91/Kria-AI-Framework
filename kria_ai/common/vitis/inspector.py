from pathlib import Path


def inspect_model(*, model, example_inputs, target, device, output_dir):
    from pytorch_nndct.apis import Inspector

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    inspector = Inspector(target)
    inspector.inspect(model, tuple(example_inputs), device=device, output_dir=str(destination))
    return destination
