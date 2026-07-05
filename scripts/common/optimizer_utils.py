"""
Utility functions for model analysis and reporting in the Vitis AI optimizer pipeline.

Provides helpers for:
- Parameter counting (total and trainable)
- Model size estimation in MB
- GFLOPs estimation (using thop.profile if available, otherwise fallback to config)
- Per-layer channel summary for Conv2d and Linear layers
- Formatting before/after reports with deltas and percentages
"""
import os
import json
import torch
import torch.nn as nn


def count_parameters(model):
    """
    Count total and trainable parameters in a model.

    Returns:
        tuple: (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def model_size_mb(model):
    """
    Estimate model size in MB based on parameter storage.

    Returns:
        float: Model size in megabytes
    """
    size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return size_bytes / (1024 * 1024)


def estimate_gflops(model, input_shape, fallback_gops=None):
    """
    Estimate GFLOPs using thop.profile if available, otherwise use fallback.

    Args:
        model: PyTorch model
        input_shape: Tuple (H, W) of input dimensions
        fallback_gops: Fallback GOPs value from model_config if thop unavailable

    Returns:
        float: Estimated GFLOPs
    """
    try:
        from thop import profile
        device = next(model.parameters()).device
        dummy_input = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        return flops / 1e9  # Convert to GFLOPs
    except ImportError:
        if fallback_gops is not None:
            return fallback_gops
        print("[WARN] thop not installed and no fallback GOPs provided. Returning 0.0")
        return 0.0
    except Exception as e:
        print(f"[WARN] thop.profile failed: {e}. Using fallback GOPs if available.")
        if fallback_gops is not None:
            return fallback_gops
        return 0.0
    finally:
        # thop does NOT clear its hooks if it crashes! Leftover hooks will
        # crash the VAIQ pruner with "OrderedDict mutated during iteration".
        for m in model.modules():
            m._forward_hooks.clear()
            m._forward_pre_hooks.clear()


_CONV_TYPES = (nn.Conv1d, nn.Conv2d, nn.Conv3d)
_DECONV_TYPES = (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)
_BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


def per_layer_channel_summary(model):
    """
    Extract per-layer channel information for Conv (1d/2d/3d), ConvTranspose,
    Linear, and BatchNorm layers.

    For BatchNorm, in_channels == out_channels == num_features (BN does not
    change channel count, but we surface it because pruning can resize it
    along with its paired Conv).

    Args:
        model: PyTorch model

    Returns:
        list: List of tuples (name, in_channels, out_channels, param_count, layer_type)
    """
    summary = []
    for name, module in model.named_modules():
        if isinstance(module, _CONV_TYPES) or isinstance(module, _DECONV_TYPES):
            in_ch = module.in_channels
            out_ch = module.out_channels
            params = sum(p.numel() for p in module.parameters())
            summary.append((name, in_ch, out_ch, params, type(module).__name__))
        elif isinstance(module, nn.Linear):
            in_ch = module.in_features
            out_ch = module.out_features
            params = sum(p.numel() for p in module.parameters())
            summary.append((name, in_ch, out_ch, params, "Linear"))
        elif isinstance(module, _BN_TYPES):
            in_ch = module.num_features
            out_ch = module.num_features
            params = sum(p.numel() for p in module.parameters())
            summary.append((name, in_ch, out_ch, params, type(module).__name__))
    return summary


def format_report(before_metrics, after_metrics, model_name):
    """
    Format a pretty console table showing before/after metrics with deltas.

    Args:
        before_metrics: Dict with keys: params, size_mb, gflops, layer_summary
        after_metrics: Dict with same structure
        model_name: Name of the model for the report header

    Returns:
        str: Formatted report string
    """
    lines = []
    lines.append("=" * 80)
    lines.append(f"  OPTIMIZER REPORT: {model_name}")
    lines.append("=" * 80)

    # Helper to format delta
    def format_delta(before, after, fmt):
        delta = after - before
        pct = (delta / before * 100) if before != 0 else 0
        sign = "+" if delta > 0 else ""
        return f"{sign}{delta:{fmt}} ({sign}{pct:.1f}%)"

    # Parameters
    before_params = before_metrics['params']
    after_params = after_metrics['params']
    lines.append(f"\nParameters:")
    lines.append(f"  Before:  {before_params:,.0f}")
    lines.append(f"  After:   {after_params:,.0f}")
    lines.append(f"  Delta:   {format_delta(before_params, after_params, ',.0f')}")

    # Model size
    before_size = before_metrics['size_mb']
    after_size = after_metrics['size_mb']
    lines.append(f"\nModel Size:")
    lines.append(f"  Before:  {before_size:.2f} MB")
    lines.append(f"  After:   {after_size:.2f} MB")
    lines.append(f"  Delta:   {format_delta(before_size, after_size, '.2f')}")

    # GFLOPs
    before_gflops = before_metrics['gflops']
    after_gflops = after_metrics['gflops']
    lines.append(f"\nEstimated GFLOPs:")
    lines.append(f"  Before:  {before_gflops:.2f}")
    lines.append(f"  After:   {after_gflops:.2f}")
    lines.append(f"  Delta:   {format_delta(before_gflops, after_gflops, '.2f')}")

    # Layer summary (show top 10 largest layers by parameter count)
    lines.append(f"\nTop 10 Largest Layers (by parameter count):")
    before_layers = sorted(before_metrics['layer_summary'], key=lambda x: x[3], reverse=True)[:10]
    after_layers = {name: (in_ch, out_ch, params) for name, in_ch, out_ch, params, _ in after_metrics['layer_summary']}

    lines.append(f"  {'Layer Name':<40} {'Before':>12} {'After':>12} {'Delta':>12}")
    lines.append(f"  {'-'*40} {'-'*12} {'-'*12} {'-'*12}")
    for name, in_ch, out_ch, params, layer_type in before_layers:
        before_str = f"{params:,}"
        if name in after_layers:
            after_in, after_out, after_params = after_layers[name]
            after_str = f"{after_params:,}"
            delta = after_params - params
            delta_pct = (delta / params * 100) if params > 0 else 0
            delta_str = f"{delta:,} ({delta_pct:+.1f}%)"
        else:
            after_str = "REMOVED"
            delta_str = "-100%"
        lines.append(f"  {name:<40} {before_str:>12} {after_str:>12} {delta_str:>12}")
        
    # Top 10 Most Pruned Layers
    lines.append(f"\nTop 10 Most Pruned Layers:")
    pruned_layers = []
    for name, in_ch, out_ch, params, layer_type in before_metrics['layer_summary']:
        if name in after_layers:
            _, _, after_params = after_layers[name]
            delta = params - after_params
            if delta > 0:
                pruned_layers.append((name, params, after_params, delta))
        else:
            pruned_layers.append((name, params, 0, params))
            
    pruned_layers = sorted(pruned_layers, key=lambda x: x[3], reverse=True)[:10]
    
    lines.append(f"  {'Layer Name':<40} {'Before':>12} {'After':>12} {'Reduction':>12}")
    lines.append(f"  {'-'*40} {'-'*12} {'-'*12} {'-'*12}")
    for name, b_params, a_params, delta in pruned_layers:
        delta_pct = (-delta / b_params * 100) if b_params > 0 else 0
        lines.append(f"  {name:<40} {b_params:>12,} {a_params:>12,} {-delta:>12,} ({delta_pct:+.1f}%)")

    lines.append("\n" + "=" * 80)
    return "\n".join(lines)


def save_report_json(before_metrics, after_metrics, model_name, output_dir):
    """
    Save detailed before/after metrics to a JSON file.

    Args:
        before_metrics: Dict with keys: params, size_mb, gflops, layer_summary
        after_metrics: Dict with same structure
        model_name: Name of the model
        output_dir: Directory to save the JSON report
    """
    os.makedirs(output_dir, exist_ok=True)
    
    report = {
        "model_name": model_name,
        "before": {
            "parameters": before_metrics['params'],
            "trainable_parameters": before_metrics['trainable_params'],
            "size_mb": before_metrics['size_mb'],
            "gflops": before_metrics['gflops'],
            "layer_summary": [
                {
                    "name": name,
                    "in_channels": in_ch,
                    "out_channels": out_ch,
                    "parameters": params,
                    "type": layer_type
                }
                for name, in_ch, out_ch, params, layer_type in before_metrics['layer_summary']
            ]
        },
        "after": {
            "parameters": after_metrics['params'],
            "trainable_parameters": after_metrics['trainable_params'],
            "size_mb": after_metrics['size_mb'],
            "gflops": after_metrics['gflops'],
            "layer_summary": [
                {
                    "name": name,
                    "in_channels": in_ch,
                    "out_channels": out_ch,
                    "parameters": params,
                    "type": layer_type
                }
                for name, in_ch, out_ch, params, layer_type in after_metrics['layer_summary']
            ]
        },
        "deltas": {
            "parameters": after_metrics['params'] - before_metrics['params'],
            "parameters_pct": ((after_metrics['params'] - before_metrics['params']) / before_metrics['params'] * 100) if before_metrics['params'] != 0 else 0,
            "size_mb": after_metrics['size_mb'] - before_metrics['size_mb'],
            "size_mb_pct": ((after_metrics['size_mb'] - before_metrics['size_mb']) / before_metrics['size_mb'] * 100) if before_metrics['size_mb'] != 0 else 0,
            "gflops": after_metrics['gflops'] - before_metrics['gflops'],
            "gflops_pct": ((after_metrics['gflops'] - before_metrics['gflops']) / before_metrics['gflops'] * 100) if before_metrics['gflops'] != 0 else 0,
        }
    }
    
    report_path = os.path.join(output_dir, "summary.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    return report_path


def evaluate_accuracy(model, dataloader, device=None):
    """Top-1 classification accuracy in percent.

    Shared evaluator used by the Vitis AI sensitivity analysis (`runner.ana`)
    and one-step subnet search (`runner.search`). Both call this with
    signature ``(model, dataloader)`` and expect a higher-is-better scalar.

    For non-classification tasks the dataloader's targets are typically dummy
    zeros (see scripts/common/dataset_utils.py) and the returned number is
    meaningless; callers should gate on ``m_cfg['type']`` before relying on
    the result.

    Args:
        model: PyTorch model (will be put in eval mode).
        dataloader: Yields ``(images, targets)`` batches.
        device: Optional device override; defaults to the model's device.

    Returns:
        float: Top-1 accuracy in percent, or 0.0 if no samples were seen.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100. * correct / total if total > 0 else 0.0


def evaluate_loss(model, dataloader, loss_fn=None, device=None):
    """Aggregate loss over a dataloader.

    Shared evaluator used by the Vitis AI quantizer's AdaQuant fast
    fine-tuning (signature ``(model, loader, loss_fn)``). Defaults to
    ``CrossEntropyLoss`` for classification when ``loss_fn`` is None.

    Args:
        model: PyTorch model (will be put in eval mode).
        dataloader: Yields ``(images, targets)`` batches.
        loss_fn: Loss callable; defaults to ``nn.CrossEntropyLoss()``.
        device: Optional device override; defaults to the model's device.

    Returns:
        float: Sum of per-batch losses (lower is better).
    """
    if device is None:
        device = next(model.parameters()).device
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            total_loss += loss_fn(outputs, targets).item()
    return total_loss


def collect_model_metrics(model, input_shape, fallback_gops=None):
    """
    Collect all metrics for a model in a single dictionary.

    Args:
        model: PyTorch model
        input_shape: Tuple (H, W) of input dimensions
        fallback_gops: Fallback GOPs value if thop unavailable

    Returns:
        dict: Dictionary containing params, trainable_params, size_mb, gflops, layer_summary
    """
    total_params, trainable_params = count_parameters(model)
    size_mb = model_size_mb(model)
    gflops = estimate_gflops(model, input_shape, fallback_gops)
    layer_summary = per_layer_channel_summary(model)
    
    return {
        'params': total_params,
        'trainable_params': trainable_params,
        'size_mb': size_mb,
        'gflops': gflops,
        'layer_summary': layer_summary
    }


def compute_metrics_for_pruned_model(pruned_model, slim_sd, input_shape,
                                     reference_metrics):
    """
    Collect metrics for a Vitis-AI-pruned model.

    Strategy:
      1. Params and size are taken from ``slim_state_dict`` (the compact
         deployment view, ground truth for the pruned model).
      2. GFLOPs and per-layer summary are computed by running the standard
         ``collect_model_metrics`` (thop + module walk) on the pruned model.
         This gives true slim values when ``IterativePruningRunner.prune``
         returns a structurally-slimmed graph.
      3. We sanity-check thop's reported parameter count against the slim
         count from step 1. If they match (within 5%), the wrapper is slim
         and the thop GFLOPs / module-derived layer summary are trustworthy.
         Otherwise the wrapper retains full shapes (older Vitis AI versions
         or depthwise corner cases) and we fall back to the original
         parameter-ratio scaling against ``reference_metrics``.

    Args:
        pruned_model: Model returned by ``IterativePruningRunner.prune``.
        slim_sd: Result of ``pruned_model.slim_state_dict()``.
        input_shape: Tuple ``(H, W)`` of the input resolution.
        reference_metrics: ``collect_model_metrics`` output for the un-pruned
            model. Used both as a fallback for GFLOPs scaling and as a layer
            summary fallback if the wrapper does not expose slim modules.

    Returns:
        dict with keys params, trainable_params, size_mb, gflops, layer_summary.
    """
    # 1. Ground-truth params and size from slim_state_dict.
    slim_params = sum(t.numel() for t in slim_sd.values() if hasattr(t, 'numel'))
    size_mb = slim_params * 4 / (1024 * 1024)

    # 2. Try thop + module walk on the pruned model directly.
    wrapper_metrics = None
    try:
        wrapper_metrics = collect_model_metrics(
            pruned_model, input_shape, reference_metrics.get('gflops')
        )
    except Exception as e:
        print(f"[WARN] Metric collection on pruned model failed: {e}")

    # 3. Decide whether the wrapper is slim (trust thop) or full (scale).
    use_wrapper = (
        wrapper_metrics is not None
        and slim_params > 0
        and abs(wrapper_metrics['params'] - slim_params) / slim_params < 0.05
    )

    if use_wrapper:
        gflops = wrapper_metrics['gflops']
        layer_summary = wrapper_metrics['layer_summary']
        print(f"[INFO] After-prune GFLOPs measured on pruned model: {gflops:.3f}")
    else:
        ref_params = reference_metrics.get('params', 0)
        ref_gflops = reference_metrics.get('gflops', 0)
        gflops = (ref_gflops * (slim_params / ref_params)
                  if ref_params > 0 else ref_gflops)
        # (channels at least changed even if shapes are masked); otherwise
        # fall back to the reference summary.
        layer_summary = []
        for name, in_ch, out_ch, params, l_type in reference_metrics.get('layer_summary', []):
            # Sum the number of elements in slim_sd for this layer's prefix
            prefix = name + "."
            slim_layer_params = sum(t.numel() for k, t in slim_sd.items() if k.startswith(prefix))
            # Fallback to the original param count if it's not found in slim_sd
            if slim_layer_params == 0 and not any(k.startswith(prefix) for k in slim_sd.keys()):
                slim_layer_params = params
                
            layer_summary.append((name, in_ch, out_ch, slim_layer_params, l_type))
            
        if wrapper_metrics is not None:
            print(f"[INFO] Pruner wrapper reports {wrapper_metrics['params']:,} params "
                  f"vs slim {slim_params:,}; using scaled GFLOPs estimate "
                  f"({gflops:.3f}).")

    return {
        'params': slim_params,
        'trainable_params': slim_params,
        'size_mb': size_mb,
        'gflops': gflops,
        'layer_summary': layer_summary,
    }
