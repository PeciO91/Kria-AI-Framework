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
import sys
import json
from collections import OrderedDict
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
        dummy_input = torch.randn(1, 3, input_shape[0], input_shape[1])
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


def per_layer_channel_summary(model):
    """
    Extract per-layer channel information for Conv2d and Linear layers.

    Args:
        model: PyTorch model

    Returns:
        list: List of tuples (name, in_channels, out_channels, param_count)
    """
    summary = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            in_ch = module.in_channels
            out_ch = module.out_channels
            params = sum(p.numel() for p in module.parameters())
            summary.append((name, in_ch, out_ch, params, "Conv2d"))
        elif isinstance(module, nn.Linear):
            in_ch = module.in_features
            out_ch = module.out_features
            params = sum(p.numel() for p in module.parameters())
            summary.append((name, in_ch, out_ch, params, "Linear"))
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
    def format_delta(before, after, metric_name):
        delta = after - before
        pct = (delta / before * 100) if before != 0 else 0
        sign = "+" if delta > 0 else ""
        return f"{sign}{delta:,.0f} ({sign}{pct:.1f}%)"

    # Parameters
    before_params = before_metrics['params']
    after_params = after_metrics['params']
    lines.append(f"\nParameters:")
    lines.append(f"  Before:  {before_params:,.0f}")
    lines.append(f"  After:   {after_params:,.0f}")
    lines.append(f"  Delta:   {format_delta(before_params, after_params, 'params')}")

    # Model size
    before_size = before_metrics['size_mb']
    after_size = after_metrics['size_mb']
    lines.append(f"\nModel Size:")
    lines.append(f"  Before:  {before_size:.2f} MB")
    lines.append(f"  After:   {after_size:.2f} MB")
    lines.append(f"  Delta:   {format_delta(before_size, after_size, 'size_mb')}")

    # GFLOPs
    before_gflops = before_metrics['gflops']
    after_gflops = after_metrics['gflops']
    lines.append(f"\nEstimated GFLOPs:")
    lines.append(f"  Before:  {before_gflops:.2f}")
    lines.append(f"  After:   {after_gflops:.2f}")
    lines.append(f"  Delta:   {format_delta(before_gflops, after_gflops, 'gflops')}")

    # Layer summary (show top 10 largest layers by parameter count)
    lines.append(f"\nTop 10 Largest Layers (by parameter count):")
    before_layers = sorted(before_metrics['layer_summary'], key=lambda x: x[3], reverse=True)[:10]
    after_layers = {name: (in_ch, out_ch, params) for name, in_ch, out_ch, params, _ in after_metrics['layer_summary']}

    lines.append(f"  {'Layer Name':<40} {'Before':>12} {'After':>12} {'Delta':>12}")
    lines.append(f"  {'-'*40} {'-'*12} {'-'*12} {'-'*12}")
    for name, in_ch, out_ch, params, layer_type in before_layers:
        before_str = f"{out_ch}ch"
        if name in after_layers:
            after_in, after_out, after_params = after_layers[name]
            after_str = f"{after_out}ch"
            delta = after_out - out_ch
            delta_str = f"{delta:+d}"
        else:
            after_str = "REMOVED"
            delta_str = "N/A"
        lines.append(f"  {name:<40} {before_str:>12} {after_str:>12} {delta_str:>12}")

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
