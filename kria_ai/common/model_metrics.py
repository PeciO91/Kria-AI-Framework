import json
from pathlib import Path


def count_parameters(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def model_size_mb(model):
    size_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    return size_bytes / (1024 * 1024)


def estimate_gflops(model, input_shape, fallback_gops=None):
    try:
        import torch
        from thop import profile

        device = next(model.parameters()).device
        dummy_input = torch.randn(1, 3, input_shape[0], input_shape[1], device=device)
        flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
        return flops / 1e9
    except ImportError:
        return float(fallback_gops or 0.0)
    except Exception:
        return float(fallback_gops or 0.0)
    finally:
        for module in model.modules():
            module._forward_hooks.clear()
            module._forward_pre_hooks.clear()


def per_layer_channel_summary(model):
    import torch.nn as nn

    convolution_types = (nn.Conv1d, nn.Conv2d, nn.Conv3d)
    deconvolution_types = (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)
    batch_norm_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
    summary = []
    for name, module in model.named_modules():
        parameters = sum(parameter.numel() for parameter in module.parameters())
        if isinstance(module, convolution_types + deconvolution_types):
            summary.append((name, module.in_channels, module.out_channels, parameters, type(module).__name__))
        elif isinstance(module, nn.Linear):
            summary.append((name, module.in_features, module.out_features, parameters, "Linear"))
        elif isinstance(module, batch_norm_types):
            summary.append((name, module.num_features, module.num_features, parameters, type(module).__name__))
    return summary


def collect_model_metrics(model, input_shape, fallback_gops=None):
    total, trainable = count_parameters(model)
    return {
        "params": total,
        "trainable_params": trainable,
        "size_mb": model_size_mb(model),
        "gflops": estimate_gflops(model, input_shape, fallback_gops),
        "layer_summary": per_layer_channel_summary(model),
    }


def metrics_from_slim_state_dict(pruned_model, slim_state_dict, input_shape, reference_metrics):
    slim_params = sum(value.numel() for value in slim_state_dict.values() if hasattr(value, "numel"))
    wrapper_metrics = collect_model_metrics(pruned_model, input_shape, reference_metrics.get("gflops"))
    wrapper_is_slim = slim_params > 0 and abs(wrapper_metrics["params"] - slim_params) / slim_params < 0.05
    if wrapper_is_slim:
        gflops = wrapper_metrics["gflops"]
        layer_summary = wrapper_metrics["layer_summary"]
    else:
        reference_params = reference_metrics.get("params", 0)
        reference_gflops = reference_metrics.get("gflops", 0)
        gflops = reference_gflops * slim_params / reference_params if reference_params else reference_gflops
        layer_summary = []
        for name, in_channels, out_channels, parameters, layer_type in reference_metrics.get("layer_summary", []):
            prefix = f"{name}."
            matches = [value for key, value in slim_state_dict.items() if key.startswith(prefix)]
            slim_layer_params = sum(value.numel() for value in matches) if matches else parameters
            layer_summary.append((name, in_channels, out_channels, slim_layer_params, layer_type))
    return {
        "params": slim_params,
        "trainable_params": slim_params,
        "size_mb": slim_params * 4 / (1024 * 1024),
        "gflops": gflops,
        "layer_summary": layer_summary,
    }


def save_metrics_report(path, model_id, before, after):
    payload = {
        "model_id": model_id,
        "before": before,
        "after": after,
        "deltas": {
            "parameters": after["params"] - before["params"],
            "size_mb": after["size_mb"] - before["size_mb"],
            "gflops": after["gflops"] - before["gflops"],
        },
    }
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, default=list) + "\n", encoding="utf-8")
    return report_path
