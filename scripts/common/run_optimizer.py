"""
Vitis AI structural pruning and optimization pipeline.

Implements the full Vitis AI pruning workflow:
  - ana: Sensitivity analysis (iterative method) or subnet search (onestep method)
  - prune: Structural channel pruning with configurable ratio
  - finetune: Fine-tuning to recover accuracy after pruning
  - all: Run the complete pipeline (ana -> prune -> finetune)

Two pruning algorithms are supported via --method:
  - iterative: IterativePruningRunner. Sensitivity analysis on each layer
    (.vai/<Model>.sens) is done once and reused for any pruning ratio.
  - onestep: OneStepPruningRunner (EagleEye-style). Random subnet sampling
    with adaptive-BN calibration; one .vai/<Model>_ratio_<R>.search per ratio.

The pipeline produces detailed before/after reports showing parameter count,
model size, FLOPs, and per-layer channel changes.
"""
import os
import sys
import argparse
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from pytorch_nndct import IterativePruningRunner
from pytorch_nndct import OneStepPruningRunner

# --- MONKEY PATCH FOR VAIQ DEEPCOPY ISSUE ---
# Vitis AI's __torch_function__ override breaks deepcopy for some tensors/parameters.
# This patch catches the "new_empty" error and falls back to clone().
_original_tensor_deepcopy = torch.Tensor.__deepcopy__
def _safe_tensor_deepcopy(self, memo):
    try:
        return _original_tensor_deepcopy(self, memo)
    except RuntimeError as e:
        if "new_empty" in str(e):
            if isinstance(self, torch.nn.Parameter):
                return torch.nn.Parameter(self.clone(), requires_grad=self.requires_grad)
            return self.clone()
        raise e
torch.Tensor.__deepcopy__ = _safe_tensor_deepcopy
# --- MONKEY PATCH FOR VAIQ THOP CRASH ---
# Vitis AI's internal profiler uses `thop` which crashes on some YOLO layers (e.g. CUDA mismatches).
# We intercept the VAIQ MACs counter, force it to run safely on CPU, and aggressively scrub hooks.
import pytorch_nndct.utils.profiler as nndct_profiler
_orig_model_complexity = nndct_profiler.model_complexity

def _safe_model_complexity(model, inputs, **kwargs):
    # Scrub existing hooks that might corrupt thop
    for m in model.modules():
        m._forward_hooks.clear()
        m._forward_pre_hooks.clear()
        
    device = next(model.parameters()).device
    model.cpu()
    
    if isinstance(inputs, torch.Tensor):
        cpu_inputs = inputs.cpu()
    elif isinstance(inputs, tuple):
        cpu_inputs = tuple(t.cpu() if isinstance(t, torch.Tensor) else t for t in inputs)
    elif isinstance(inputs, list):
        cpu_inputs = [t.cpu() if isinstance(t, torch.Tensor) else t for t in inputs]
    else:
        cpu_inputs = inputs
    
    try:
        macs, params = _orig_model_complexity(model, cpu_inputs, **kwargs)
    finally:
        model.to(device)
        # Aggressively scrub hooks again so VAIQ's graph compiler doesn't trip on them
        for m in model.modules():
            m._forward_hooks.clear()
            m._forward_pre_hooks.clear()
            
    return macs, params

nndct_profiler.model_complexity = _safe_model_complexity
# ----------------------------------------
# Project-root import path (PROJECT_ROOT + scripts/common/ added to sys.path).
from _bootstrap import PROJECT_ROOT  # noqa: F401

from model_config import get_active_model
from dataset_config import get_active_dataset
from model_utils import prepare_model, derive_weight_path
from optimizer_utils import (
    collect_model_metrics,
    compute_metrics_for_pruned_model,
    evaluate_accuracy,
    format_report,
    save_report_json
)
from dataset_utils import FlatImageDataset, YoloDataset, yolo_collate_fn, build_or_load_subset_indices
from detection_profiles import get_profile


def build_finetune_loader(m_cfg, d_cfg, subset_len=200, batch_size=32):
    """Build a data loader for fine-tuning based on model task type."""
    task_type = m_cfg.get('type', 'classification')
    
    if task_type == 'classification':
        # Use ImageFolder for classification with actual labels
        base_transform = transforms.Compose([
            transforms.Resize(m_cfg['input_shape']),
            transforms.ToTensor(),
            transforms.Normalize(d_cfg['normalization']['mean'],
                                 d_cfg['normalization']['std']),
        ])
        dataset = ImageFolder(root=d_cfg['calib_path'], transform=base_transform)
        # limit to subset_len with random sampling to avoid bias from
        # alphabetical ordering (ImageFolder sorts classes by name).
        if len(dataset) > subset_len:
            random.seed(42)  # Fixed seed for reproducibility
            indices = random.sample(range(len(dataset)), subset_len)
            dataset = Subset(dataset, indices)
    elif task_type == 'detection' or m_cfg.get('seg_instance'):
        yolo_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(d_cfg['normalization']['mean'],
                                 d_cfg['normalization']['std']),
        ])
        subset_indices = build_or_load_subset_indices(
            split="train",
            n=subset_len,
            cache_dir=d_cfg['subset_cache_dir']
        )
        dataset = YoloDataset(
            images_dir=d_cfg['images_train'],
            labels_dir=d_cfg['labels_train'],
            input_shape=m_cfg['input_shape'],
            normalization=d_cfg['normalization'],
            augment=True, # Apply augmentations during fine-tuning!
            indices=subset_indices
        )
    else:
        base_transform = transforms.Compose([
            transforms.Resize(m_cfg['input_shape']),
            transforms.ToTensor(),
            transforms.Normalize(d_cfg['normalization']['mean'],
                                 d_cfg['normalization']['std']),
        ])
        dataset = FlatImageDataset(root_dir=d_cfg['calib_path'], transform=base_transform)
        if len(dataset) > subset_len:
            indices = list(range(subset_len))
            dataset = Subset(dataset, indices)
    
    collate_fn = yolo_collate_fn if task_type == 'detection' or m_cfg.get('seg_instance') else None
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


# =============================================================
# OPTIMIZER PIPELINE
# =============================================================
def run_sensitivity_analysis(runner, m_cfg, d_cfg, device, subset_len=200):
    """Run Vitis AI sensitivity analysis using IterativePruningRunner."""
    print(f"\n{'='*70}")
    print("  MODE: SENSITIVITY ANALYSIS (ana)")
    print(f"{'='*70}")
    
    task_type = m_cfg.get('type', 'classification')
    
    # Build validation loader based on task type
    if task_type == 'detection' or m_cfg.get('seg_instance'):
        print("[INFO] Constructing COCO val set for detection sensitivity analysis.")
        subset_indices = build_or_load_subset_indices(
            split="val",
            n=subset_len,
            cache_dir=d_cfg['subset_cache_dir']
        )
        val_dataset = YoloDataset(
            images_dir=d_cfg['images_val'],
            labels_dir=d_cfg['labels_val'],
            input_shape=m_cfg['input_shape'],
            normalization=d_cfg['normalization'],
            augment=False,
            indices=subset_indices
        )
        loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=yolo_collate_fn)
    else:
        loader = build_finetune_loader(m_cfg, d_cfg, subset_len=subset_len, batch_size=4)

    print(f"[INFO] Running sensitivity analysis on {len(loader.dataset)} samples...")

    # Define evaluation function
    if task_type == 'detection' or m_cfg.get('seg_instance'):
        def detection_loss_eval(model, dataloader):
            device_local = next(model.parameters()).device
            profile = get_profile(m_cfg)
            loss_fn = profile.loss_fn(model)
            total_loss = 0.0
            count = 0
            was_training = model.training
            model.train() # Enable training output format for loss calculation
            try:
                with torch.no_grad():
                    for images, targets in dataloader:
                        images = images.to(device_local)
                        targets = {k: v.to(device_local) if isinstance(v, torch.Tensor) else v for k, v in targets.items()}
                        outputs = model(images)
                        loss_val = loss_fn(outputs, targets)
                        if isinstance(loss_val, dict):
                            loss_val = sum(l for l in loss_val.values())
                        elif isinstance(loss_val, (list, tuple)):
                            loss_val = loss_val[0]
                        
                        if hasattr(loss_val, 'sum'):
                            loss_val = loss_val.sum()
                        
                        total_loss += loss_val.item()
                        count += 1
            finally:
                if not was_training:
                    model.eval()
            return -(total_loss / max(1, count)) # Negative loss: higher (closer to 0) is better
        
        eval_fn = detection_loss_eval
    else:
        eval_fn = evaluate_accuracy
    # Retrieve excludes if applicable
    excludes = []
    if task_type == 'detection' or m_cfg.get('seg_instance'):
        profile = get_profile(m_cfg)
        excludes = profile.prune_excludes(None)

    import re
    max_retries = 30
    for attempt in range(max_retries):
        try:
            runner.ana(eval_fn, args=(loader,), excludes=excludes)
            print("[SUCCESS] Sensitivity analysis complete.")
            print("[INFO] Analysis results saved to .vai/ directory.")
            break
        except Exception as e:
            error_str = str(e)
            if "Must exclude node from pruning:" in error_str:
                node_part = error_str.split("Must exclude node from pruning:")[1].split("\n")[0].strip()
                if node_part.endswith("."):
                    node_part = node_part[:-1]
                
                node_name = node_part
                print(f"[WARN] VAIQ constraint hit (Attempt {attempt+1}). Auto-excluding: {node_name}")
                if node_name not in excludes:
                    excludes.append(node_name)
                    continue
            
            import traceback
            print(f"[ERROR] Sensitivity analysis failed: {e}")
            traceback.print_exc()
            print("[INFO] Continuing with pruning using specified ratio...")
            break
    
    return runner


def run_subnet_search(runner, m_cfg, d_cfg, device, ratio,
                      num_subnet=200, num_calib_forward=100, subset_len=200):
    """Run Vitis AI one-step subnet search using OneStepPruningRunner.

    Generates .vai/<Model>_ratio_<ratio>.search by randomly sampling
    `num_subnet` candidate subnets and scoring each via adaptive-BN
    calibration + evaluation. The best subnet is then realised by a
    subsequent runner.prune(removal_ratio=ratio) call.
    """
    print(f"\n{'='*70}")
    print(f"  MODE: ONE-STEP SUBNET SEARCH (ratio={ratio}, num_subnet={num_subnet})")
    print(f"{'='*70}")

    task_type = m_cfg.get('type', 'classification')
    if task_type != 'classification':
        print(f"[WARN] One-step search is only fully supported for classification in v1.")
        print(f"[INFO] Running search on {task_type} model - results may be limited.")

    loader = build_finetune_loader(m_cfg, d_cfg, subset_len=subset_len, batch_size=4)

    def calibration_fn(m, dataloader, number_forward=num_calib_forward):
        """Adaptive-BN calibration: re-estimate BN running stats for the subnet."""
        m.train()
        with torch.no_grad():
            for index, batch in enumerate(dataloader):
                images = batch[0].to(device)
                m(images)
                if index > number_forward:
                    break

    gpus = [str(torch.cuda.current_device())] if device.type == 'cuda' else []

    print(f"[INFO] Calibrating {num_subnet} candidate subnets "
          f"({len(loader.dataset)} samples, {num_calib_forward} BN-calib batches)...")

    # Define evaluation function and excludes for search
    if task_type == 'detection' or m_cfg.get('seg_instance'):
        profile = get_profile(m_cfg)
        excludes = profile.prune_excludes(None)
        def detection_loss_eval(model, dataloader):
            device_local = next(model.parameters()).device
            loss_fn = profile.loss_fn(model)
            total_loss = 0.0
            count = 0
            was_training = model.training
            model.train()
            try:
                with torch.no_grad():
                    for images, targets in dataloader:
                        images = images.to(device_local)
                        targets = {k: v.to(device_local) if isinstance(v, torch.Tensor) else v for k, v in targets.items()}
                        outputs = model(images)
                        loss_val = loss_fn(outputs, targets)
                        if isinstance(loss_val, dict):
                            loss_val = sum(l for l in loss_val.values())
                        elif isinstance(loss_val, (list, tuple)):
                            loss_val = sum(loss_val)
                        total_loss += loss_val.item()
                        count += 1
            finally:
                if not was_training:
                    model.eval()
            return -(total_loss / max(1, count))
        
        eval_fn = detection_loss_eval
    else:
        eval_fn = evaluate_accuracy
        excludes = []

    try:
        runner.search(
            gpus=gpus,
            calibration_fn=calibration_fn,
            calib_args=(loader,),
            eval_fn=eval_fn,
            eval_args=(loader,),
            num_subnet=num_subnet,
            removal_ratio=ratio,
            excludes=excludes,
        )
        print("[SUCCESS] Subnet search complete.")
        print(f"[INFO] Search result saved to .vai/ directory.")
    except Exception as e:
        import traceback
        print(f"[ERROR] Subnet search failed: {e}")
        traceback.print_exc()
        print("[INFO] Continuing with pruning using specified ratio...")

    return runner


def run_pruning(model, runner, m_cfg, device, ratio=0.2, before_metrics=None):
    """Run structural pruning using IterativePruningRunner and collect metrics."""
    print(f"\n{'='*70}")
    print(f"  MODE: PRUNING (ratio={ratio})")
    print(f"{'='*70}")
    
    # Collect before metrics if not provided
    if before_metrics is None:
        input_h, input_w = m_cfg['input_shape']
        before_metrics = collect_model_metrics(model, (input_h, input_w), m_cfg.get('gops'))
        print(f"[INFO] Before pruning:")
        print(f"  Parameters: {before_metrics['params']:,.0f}")
        print(f"  Model size: {before_metrics['size_mb']:.2f} MB")
        print(f"  GFLOPs: {before_metrics['gflops']:.2f}")
    
    print(f"[INFO] Applying structural pruning with removal_ratio {ratio}...")
    
    # Retrieve excludes if task type is detection or segmentation
    task_type = m_cfg.get('type', 'classification')
    excludes = []
    if task_type == 'detection' or m_cfg.get('seg_instance'):
        profile = get_profile(m_cfg)
        excludes = profile.prune_excludes(model)
        print(f"[INFO] Excluding layers from pruning: {excludes}")

        
    # Apply pruning using IterativePruningRunner with excludes
    pruned_model = runner.prune(removal_ratio=ratio, excludes=excludes)
    
    # Compute after-pruning metrics. Preferred path: read slim params/size
    # from slim_state_dict and run thop on the pruned model for real GFLOPs;
    # falls back to a scaled estimate if the wrapper retains full shapes.
    input_h, input_w = m_cfg['input_shape']
    if hasattr(pruned_model, 'slim_state_dict'):
        slim_sd = pruned_model.slim_state_dict()
        after_metrics = compute_metrics_for_pruned_model(
            pruned_model, slim_sd, (input_h, input_w), before_metrics
        )
        print(f"[INFO] After pruning (from pruned model + slim_state_dict):")
    else:
        # No slim view available: profile the pruned model as-is.
        after_metrics = collect_model_metrics(
            pruned_model, (input_h, input_w), m_cfg.get('gops')
        )
        print(f"[INFO] After pruning:")
    
    print(f"  Parameters: {after_metrics['params']:,.0f}")
    print(f"  Model size: {after_metrics['size_mb']:.2f} MB")
    print(f"  GFLOPs: {after_metrics['gflops']:.2f}")
    
    # Calculate and print deltas
    param_delta = after_metrics['params'] - before_metrics['params']
    param_pct = (param_delta / before_metrics['params'] * 100) if before_metrics['params'] > 0 else 0
    print(f"[INFO] Parameter reduction: {param_delta:,.0f} ({param_pct:.1f}%)")
    
    return pruned_model, before_metrics, after_metrics


def run_finetune(model, m_cfg, d_cfg, device, epochs=5, lr=1e-3, subset_len=200):
    """Run fine-tuning to recover accuracy after pruning."""
    print(f"\n{'='*70}")
    print(f"  MODE: FINE-TUNING ({epochs} epochs)")
    print(f"{'='*70}")
    
    task_type = m_cfg.get('type', 'classification')
    
    if task_type not in ['classification', 'detection', 'segmentation']:
        print(f"[WARN] Fine-tuning is only implemented for classification, detection, and segmentation.")
        print(f"[INFO] Skipping fine-tuning for {task_type} model.")
        print(f"[INFO] Model is pruned but weights are not retrained.")
        return model
    
    # Build train loader
    train_loader = build_finetune_loader(m_cfg, d_cfg, subset_len=subset_len, batch_size=4)
    
    # Ensure all parameters require gradients for fine-tuning
    for param in model.parameters():
        param.requires_grad = True
        
    # Define loss and optimizer based on task type
    if task_type == 'classification':
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        print(f"[INFO] Optimizer: SGD (lr={lr}, momentum=0.9, weight_decay=1e-4)")
    elif task_type == 'detection' or m_cfg.get('seg_instance'):
        profile = get_profile(m_cfg)
        profile.prepare_for_finetune(model)
        criterion = profile.loss_fn(model)
        # YOLO works much better with AdamW for fine-tuning
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        print(f"[INFO] Optimizer: AdamW (lr={lr}, weight_decay=1e-4)")
        
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    print(f"[INFO] Fine-tuning on {len(train_loader.dataset)} samples...")
    print(f"[INFO] Scheduler: CosineAnnealingLR")
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            
            optimizer.zero_grad()
            if task_type == 'classification':
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
            elif task_type == 'detection' or m_cfg.get('seg_instance'):
                targets = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in labels.items()}
                outputs = profile.forward_for_loss(model, images)
                loss = criterion(outputs, targets)
                
            if isinstance(loss, tuple) and len(loss) >= 2:
                # Ultralytics returns (total_loss, detached_loss_components)
                loss_val = loss[0]
            elif isinstance(loss, dict):
                loss_val = sum(l for l in loss.values())
            elif isinstance(loss, (list, tuple)):
                loss_val = sum(loss)
            else:
                loss_val = loss

            if isinstance(loss_val, torch.Tensor) and loss_val.dim() > 0:
                loss_val = loss_val.sum()
                
            loss_val.backward()
            optimizer.step()
            
            epoch_loss += loss_val.item()
            
            if task_type == 'classification':
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs} [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss_val.item():.4f}")
        
        avg_loss = epoch_loss / len(train_loader)
        scheduler.step()
        
        if task_type == 'classification' and total > 0:
            accuracy = 100. * correct / total
            print(f"  Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%")
        else:
            print(f"  Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
    
    print(f"[SUCCESS] Fine-tuning complete. Best loss: {best_loss:.4f}")
    model.eval()
    return model


def run_optimizer():
    parser = argparse.ArgumentParser(description="Vitis AI Optimizer/Pruning Pipeline")
    parser.add_argument('--model', type=str,
                        help='Model ID. Falls back to ACTIVE_MODEL_ID '
                             'in model_config.py when omitted.')
    parser.add_argument('--dataset', type=str,
                        help='Dataset ID. Falls back to ACTIVE_DATASET_ID '
                             'in dataset_config.py when omitted.')
    parser.add_argument('--mode', choices=['ana', 'prune', 'finetune', 'all'], default='all',
                        help='Optimizer mode: ana=sensitivity analysis, prune=structural pruning, '
                             'finetune=recover accuracy, all=full pipeline')
    parser.add_argument('--ratio', type=float, default=0.2,
                        help='Channel reduction ratio (0.1-0.5)')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Fine-tuning epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate for fine-tuning')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for fine-tuning')
    parser.add_argument('--subset_len', type=int, default=200,
                        help='Number of samples for ana/finetune')
    parser.add_argument('--device', type=str, choices=['auto', 'cuda', 'cpu'], default='auto',
                        help='Device to use')
    parser.add_argument('--method', choices=['iterative', 'onestep'], default='iterative',
                        help='Pruning algorithm. iterative=sensitivity analysis (.sens cache, '
                             'reusable for any ratio); onestep=EagleEye subnet search '
                             '(.search cache, per-ratio).')
    parser.add_argument('--num_subnet', type=int, default=200,
                        help='Number of candidate subnets for one-step search '
                             '(ignored for iterative)')
    parser.add_argument('--num_calib_forward', type=int, default=100,
                        help='Adaptive-BN calibration forward batches for one-step search')
    args = parser.parse_args()


    # Load configurations
    m_cfg = get_active_model(args.model)
    if args.dataset is None and m_cfg.get('type') == 'segmentation':
        d_cfg = get_active_dataset('coco_instance_seg')
    else:
        d_cfg = get_active_dataset(args.dataset)
    
    # Set device
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[INFO] Using device: {device}")
    
    # Prepare model
    model = prepare_model(m_cfg, d_cfg, device)
    
    # Clear stale Vitis AI sensitivity cache for this model class.
    # Different architectures (e.g. ResNet18 BasicBlock vs ResNet50 Bottleneck)
    # share the same class name and would otherwise reuse incompatible caches.
    model_class_name = model.__class__.__name__
    vai_cache_dir = ".vai"
    if os.path.isdir(vai_cache_dir) and args.mode in ['ana', 'all']:
        for fname in os.listdir(vai_cache_dir):
            if fname.startswith(f"{model_class_name}.") or fname.startswith(f"{model_class_name}_"):
                cache_path = os.path.join(vai_cache_dir, fname)
                try:
                    os.remove(cache_path)
                    print(f"[INFO] Removed stale cache: {cache_path}")
                except OSError as e:
                    print(f"[WARN] Could not remove {cache_path}: {e}")
    
    # Initialize the pruning runner.
    # Both runners share the same prune() API; only the pre-step differs
    # (ana for iterative, search for onestep).
    input_h, input_w = m_cfg['input_shape']
    dummy_input = torch.randn([1, 3, input_h, input_w], dtype=torch.float32).to(device)
    if args.method == 'onestep':
        print(f"[INFO] Pruning method: ONE-STEP (EagleEye subnet search)")
        runner = OneStepPruningRunner(model, dummy_input)
    else:
        print(f"[INFO] Pruning method: ITERATIVE (sensitivity analysis)")
        runner = IterativePruningRunner(model, dummy_input)
    
    # Output directory for reports
    output_dir = os.path.join("build", m_cfg['name'].lower(), "optimizer_report")
    os.makedirs(output_dir, exist_ok=True)
    
    # Track metrics
    before_metrics = None
    after_metrics = None
    current_model = model
    
    # Run based on mode. The "ana" stage maps to:
    #   - sensitivity analysis (iterative method)
    #   - subnet search (onestep method)
    if args.mode in ['ana', 'all']:
        if args.method == 'onestep':
            runner = run_subnet_search(
                runner, m_cfg, d_cfg, device,
                args.ratio, args.num_subnet, args.num_calib_forward, args.subset_len,
            )
        else:
            runner = run_sensitivity_analysis(
                runner, m_cfg, d_cfg, device, args.subset_len,
            )
    
    if args.mode in ['prune', 'all']:
        current_model, before_metrics, after_metrics = run_pruning(
            current_model, runner, m_cfg, device, args.ratio, before_metrics
        )
        
        # Save pruned model. The Vitis AI wrapper keeps full shapes in
        # nn.Module but exposes a slim_state_dict() for compact deployment.
        # We save the FULL state dict (so it can be reloaded into a freshly
        # pruned wrapper for inspection/quantization). The slim version is
        # saved separately in _slim.pt for standalone use.
        pruned_weight_path = derive_weight_path(m_cfg['model_path'], "_pruned")
        slim_weight_path = derive_weight_path(m_cfg['model_path'], "_slim")
        save_path = pruned_weight_path
        slim_path = slim_weight_path
        
        torch.save(current_model.state_dict(), save_path)
        print(f"[INFO] Pruned model (full-shape state_dict) saved to: {save_path}")
        if hasattr(current_model, 'slim_state_dict'):
            torch.save(current_model.slim_state_dict(), slim_path)
            print(f"[INFO] Slim state_dict saved to: {slim_path}")
    
    if args.mode in ['finetune', 'all']:
        # In finetune-only mode, we must first prune the model structurally to
        # match the saved slim_state_dict shape before loading pruned weights.
        # We use IterativePruningRunner here regardless of the original --method
        # because it can consume the .spec generated by either iterative or onestep.
        # This avoids the brittle per-ratio .search cache requirement of onestep.
        if args.mode == 'finetune':
            save_path = derive_weight_path(m_cfg['model_path'], "_pruned")
            if os.path.exists(save_path):
                print(f"[INFO] Re-applying pruning to match saved slim model structure...")
                input_h, input_w = m_cfg['input_shape']
                dummy_input = torch.randn([1, 3, input_h, input_w], dtype=torch.float32).to(device)
                model.to(device)
                re_slim_runner = IterativePruningRunner(model, dummy_input)
                current_model = re_slim_runner.prune(removal_ratio=args.ratio)
                print(f"[INFO] Loading pruned weights from: {save_path}")
                pruned_sd = torch.load(save_path, map_location=device)
                try:
                    current_model.load_state_dict(pruned_sd, strict=False)
                except Exception as e:
                    print(f"[WARN] Could not load pruned state dict: {e}")
                    print(f"[INFO] Continuing fine-tune with freshly pruned weights.")
            else:
                print(f"[WARN] Pruned weights not found at {save_path}. Fine-tuning original model.")
        
        current_model = run_finetune(current_model, m_cfg, d_cfg, device, args.epochs, args.lr, args.subset_len)
        
        finetuned_weight_path = derive_weight_path(m_cfg['model_path'], "_finetuned")
        finetuned_slim_weight_path = derive_weight_path(m_cfg['model_path'], "_finetuned_slim")
        save_path = finetuned_weight_path
        slim_path = finetuned_slim_weight_path
        
        torch.save(current_model.state_dict(), save_path)
        print(f"[INFO] Fine-tuned model (full-shape state_dict) saved to: {save_path}")
        if hasattr(current_model, 'slim_state_dict'):
            torch.save(current_model.slim_state_dict(), slim_path)
            print(f"[INFO] Fine-tuned slim state_dict saved to: {slim_path}")
        
        # Do NOT re-collect metrics after fine-tuning - preserve the estimated pruning metrics
        # since the model object structure doesn't reflect structural changes
    
    # Generate and save report
    if args.mode in ['prune', 'all'] and before_metrics and after_metrics:
        print("\n" + format_report(before_metrics, after_metrics, m_cfg['name']))
        report_path = save_report_json(before_metrics, after_metrics, m_cfg['name'], output_dir)
        print(f"[INFO] Detailed report saved to: {report_path}")
    
    print(f"\n{'='*70}")
    print("  OPTIMIZER PIPELINE COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    run_optimizer()
