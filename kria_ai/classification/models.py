from pathlib import Path

from kria_ai.common.checkpoints import LoadResult, extract_state_dict, load_checkpoint


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _replace_head(model, attribute, num_classes):
    import torch.nn as nn

    try:
        head = getattr(model, attribute)
    except AttributeError as error:
        raise AttributeError(f"{type(model).__name__} has no classifier head {attribute!r}") from error
    if isinstance(head, nn.Sequential):
        linear_indices = [index for index, layer in enumerate(head) if isinstance(layer, nn.Linear)]
        if not linear_indices:
            raise TypeError(f"Sequential head {attribute!r} contains no Linear layer")
        index = linear_indices[-1]
        head[index] = nn.Linear(head[index].in_features, num_classes)
    elif isinstance(head, nn.Linear):
        setattr(model, attribute, nn.Linear(head.in_features, num_classes))
    else:
        raise TypeError(f"Unsupported classifier head type for {attribute!r}: {type(head).__name__}")


def build_model(config, device="cpu", checkpoint_path=None, strict=False):
    import torchvision.models as torchvision_models

    try:
        constructor = getattr(torchvision_models, config.constructor)
    except AttributeError as error:
        raise ValueError(f"Torchvision has no model constructor {config.constructor!r}") from error
    try:
        model = constructor(weights=None)
    except TypeError:
        model = constructor(pretrained=False)
    _replace_head(model, config.head_attribute, config.num_classes)
    selected_checkpoint = Path(checkpoint_path) if checkpoint_path else config.checkpoint_path
    if not selected_checkpoint.is_absolute():
        selected_checkpoint = PROJECT_ROOT / selected_checkpoint
    checkpoint = load_checkpoint(selected_checkpoint, map_location=device)
    state_dict = extract_state_dict(checkpoint)
    optimization = checkpoint.get("optimization") if isinstance(checkpoint, dict) else None
    representation = None
    if optimization:
        representation = optimization.get("checkpoint_representation", optimization.get("representation"))
    if representation == "sparse":
        raise ValueError(
            f"Checkpoint {selected_checkpoint} is sparse and must be materialized as a slim model before quantization"
        )
    if representation == "slim":
        try:
            from pytorch_nndct.utils import slim
        except ImportError as error:
            raise ImportError("Loading an optimized slim checkpoint requires the Vitis AI environment") from error
        model = slim.load_state_dict(model, state_dict)
        incompatible = LoadResult()
    else:
        incompatible = model.load_state_dict(state_dict, strict=strict)
    model.to(device)
    model.eval()
    return model, incompatible
