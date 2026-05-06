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
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

# Try to import Vitis AI pruning runners (iterative + onestep)
HAS_VITIS_PRUNER = False
HAS_ONESTEP_PRUNER = False
try:
    from pytorch_nndct import IterativePruningRunner
    HAS_VITIS_PRUNER = True
    try:
        from pytorch_nndct import OneStepPruningRunner
        HAS_ONESTEP_PRUNER = True
    except ImportError:
        print("[WARN] OneStepPruningRunner not available in this Vitis AI build.")
except ImportError:
    print("[WARN] Vitis AI Optimizer (pytorch_nndct) not found.")
    print("[INFO] Install from: https://github.com/Xilinx/Vitis-AI/tree/master/src/vai_optimizer")
    print("[INFO] Or use: pip install pytorch_nndct from the vai_optimizer directory")

# Project-root import path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from model_config import get_active_model
from dataset_config import get_active_dataset
from model_utils import prepare_model, derive_weight_path
from optimizer_utils import (
    collect_model_metrics,
    format_report,
    save_report_json
)
from detection_utils import letterbox


# =============================================================
# DATASET LOADERS FOR FINE-TUNING
# =============================================================
class SimpleImageDataset(torch.utils.data.Dataset):
    """Flat-folder image dataset for fine-tuning."""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        from PIL import Image
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0  # Label unused for calibration-style fine-tuning


class YoloCalibrationDataset(torch.utils.data.Dataset):
    """Letterbox-based dataset for YOLO fine-tuning."""
    def __init__(self, root_dir, input_shape, transform=None):
        self.root_dir = root_dir
        self.input_shape = input_shape
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        import cv2
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img, _, _ = letterbox(img, new_shape=self.input_shape)
        if self.transform:
            img = self.transform(img)
        return img, 0


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
        # Limit to subset_len with random sampling to avoid bias from
        # alphabetical ordering (ImageFolder sorts classes by name).
        if len(dataset) > subset_len:
            from torch.utils.data import Subset
            random.seed(42)  # Fixed seed for reproducibility
            indices = random.sample(range(len(dataset)), subset_len)
            dataset = Subset(dataset, indices)
    elif task_type == 'detection':
        yolo_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(d_cfg['normalization']['mean'],
                                 d_cfg['normalization']['std']),
        ])
        dataset = YoloCalibrationDataset(
            root_dir=d_cfg['calib_path'],
            input_shape=m_cfg['input_shape'],
            transform=yolo_transform,
        )
        if len(dataset) > subset_len:
            from torch.utils.data import Subset
            indices = list(range(subset_len))
            dataset = Subset(dataset, indices)
    else:
        base_transform = transforms.Compose([
            transforms.Resize(m_cfg['input_shape']),
            transforms.ToTensor(),
            transforms.Normalize(d_cfg['normalization']['mean'],
                                 d_cfg['normalization']['std']),
        ])
        dataset = SimpleImageDataset(root_dir=d_cfg['calib_path'], transform=base_transform)
        if len(dataset) > subset_len:
            from torch.utils.data import Subset
            indices = list(range(subset_len))
            dataset = Subset(dataset, indices)
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# =============================================================
# METRICS HELPERS
# =============================================================
def compute_metrics_from_state_dict(state_dict, reference_metrics):
    """Compute parameters and size from a state_dict (e.g. slim_state_dict).

    GFLOPs cannot be measured without a traced model, so we scale the
    reference GFLOPs by the parameter reduction ratio as a best-effort
    approximation.
    """
    total_params = 0
    for key, tensor in state_dict.items():
        if hasattr(tensor, 'numel'):
            total_params += tensor.numel()

    # 4 bytes per FP32 param
    size_mb = total_params * 4 / (1024 * 1024)

    # Scale GFLOPs by the parameter ratio (approximation)
    ref_params = reference_metrics.get('params', 0)
    ref_gflops = reference_metrics.get('gflops', 0)
    if ref_params > 0:
        gflops = ref_gflops * (total_params / ref_params)
    else:
        gflops = ref_gflops

    return {
        'params': total_params,
        'trainable_params': total_params,
        'size_mb': size_mb,
        'gflops': gflops,
        'layer_summary': reference_metrics.get('layer_summary', []),
    }


# =============================================================
# OPTIMIZER PIPELINE
# =============================================================
def run_sensitivity_analysis(model, runner, m_cfg, d_cfg, device, subset_len=200):
    """Run Vitis AI sensitivity analysis using IterativePruningRunner."""
    print(f"\n{'='*70}")
    print("  MODE: SENSITIVITY ANALYSIS (ana)")
    print(f"{'='*70}")
    
    task_type = m_cfg.get('type', 'classification')
    
    if task_type != 'classification':
        print(f"[WARN] Sensitivity analysis is only fully supported for classification in v1.")
        print(f"[INFO] Running analysis on {task_type} model - results may be limited.")
    
    # Build validation loader
    loader = build_finetune_loader(m_cfg, d_cfg, subset_len=subset_len, batch_size=32)
    
    # Define evaluation function for sensitivity analysis
    def eval_fn(model, dataloader):
        """Evaluation function for Vitis AI sensitivity analysis."""
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, targets in dataloader:
                images = images.to(device)
                targets = targets.to(device)
                outputs = model(images)
                if task_type == 'classification':
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
        if total > 0:
            return 100. * correct / total
        return 0.0
    
    print(f"[INFO] Running sensitivity analysis on {len(loader.dataset)} samples...")
    
    try:
        # Run Vitis AI sensitivity analysis
        runner.ana(eval_fn, args=(loader,))
        print("[SUCCESS] Sensitivity analysis complete.")
        print("[INFO] Analysis results saved to .vai/ directory.")
    except Exception as e:
        print(f"[ERROR] Sensitivity analysis failed: {e}")
        print("[INFO] Continuing with pruning using specified ratio...")
    
    return runner


def run_subnet_search(model, runner, m_cfg, d_cfg, device, ratio,
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

    loader = build_finetune_loader(m_cfg, d_cfg, subset_len=subset_len, batch_size=32)

    def calibration_fn(m, dataloader, number_forward=num_calib_forward):
        """Adaptive-BN calibration: re-estimate BN running stats for the subnet."""
        m.train()
        with torch.no_grad():
            for index, batch in enumerate(dataloader):
                images = batch[0].to(device)
                m(images)
                if index > number_forward:
                    break

    def eval_fn(m, dataloader):
        """Evaluation function for subnet scoring."""
        m.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, targets in dataloader:
                images = images.to(device)
                targets = targets.to(device)
                outputs = m(images)
                if task_type == 'classification':
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
        return 100. * correct / total if total > 0 else 0.0

    gpus = [str(torch.cuda.current_device())] if device.type == 'cuda' else []

    print(f"[INFO] Calibrating {num_subnet} candidate subnets "
          f"({len(loader.dataset)} samples, {num_calib_forward} BN-calib batches)...")

    try:
        runner.search(
            gpus=gpus,
            calibration_fn=calibration_fn,
            calib_args=(loader,),
            eval_fn=eval_fn,
            eval_args=(loader,),
            num_subnet=num_subnet,
            removal_ratio=ratio,
            excludes=[],
        )
        print("[SUCCESS] Subnet search complete.")
        print(f"[INFO] Search result saved to .vai/ directory.")
    except Exception as e:
        print(f"[ERROR] Subnet search failed: {e}")
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
    
    # Apply pruning using IterativePruningRunner
    pruned_model = runner.prune(removal_ratio=ratio)
    
    # Compute REAL after metrics from slim_state_dict (Vitis AI's structurally pruned weights)
    if hasattr(pruned_model, 'slim_state_dict'):
        slim_sd = pruned_model.slim_state_dict()
        after_metrics = compute_metrics_from_state_dict(slim_sd, before_metrics)
        print(f"[INFO] After pruning (from slim_state_dict):")
    else:
        # Fallback to standard metric collection
        input_h, input_w = m_cfg['input_shape']
        after_metrics = collect_model_metrics(pruned_model, (input_h, input_w), m_cfg.get('gops'))
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
    
    if task_type != 'classification':
        print(f"[WARN] Fine-tuning is only implemented for classification in v1.")
        print(f"[INFO] Skipping fine-tuning for {task_type} model.")
        print(f"[INFO] Model is pruned but weights are not retrained.")
        return model
    
    # Build train loader
    train_loader = build_finetune_loader(m_cfg, d_cfg, subset_len=subset_len, batch_size=32)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    print(f"[INFO] Fine-tuning on {len(train_loader.dataset)} samples...")
    print(f"[INFO] Optimizer: SGD (lr={lr}, momentum=0.9, weight_decay=1e-4)")
    print(f"[INFO] Scheduler: CosineAnnealingLR")
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if task_type == 'classification':
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs} [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f}")
        
        avg_loss = epoch_loss / len(train_loader)
        scheduler.step()
        
        if total > 0:
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
    parser.add_argument('--model', type=str, required=True, help='Model ID')
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

    if not HAS_VITIS_PRUNER:
        print("[ERROR] Vitis AI Optimizer (pytorch_nndct) is required but not installed.")
        print("[INFO] Installation instructions:")
        print("[INFO]   cd Vitis-AI/src/vai_optimizer/pytorch_binding")
        print("[INFO]   python setup.py install")
        sys.exit(1)

    if args.method == 'onestep' and not HAS_ONESTEP_PRUNER:
        print("[ERROR] --method onestep requested but OneStepPruningRunner is not "
              "available in this Vitis AI build. Falling back to --method iterative.")
        args.method = 'iterative'

    # Load configurations
    m_cfg = get_active_model(args.model)
    d_cfg = get_active_dataset(args.dataset) if args.dataset else get_active_dataset()
    
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
                current_model, runner, m_cfg, d_cfg, device,
                args.ratio, args.num_subnet, args.num_calib_forward, args.subset_len,
            )
        else:
            runner = run_sensitivity_analysis(
                current_model, runner, m_cfg, d_cfg, device, args.subset_len,
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
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        save_path = os.path.join(project_root, pruned_weight_path)
        slim_path = os.path.join(project_root, slim_weight_path)
        
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
            pruned_weight_path = derive_weight_path(m_cfg['model_path'], "_pruned")
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            save_path = os.path.join(project_root, pruned_weight_path)
            if os.path.exists(save_path):
                print(f"[INFO] Re-applying pruning to match saved slim model structure...")
                input_h, input_w = m_cfg['input_shape']
                dummy_input = torch.randn([1, 3, input_h, input_w], dtype=torch.float32).to(device)
                model.to(device)
                from pytorch_nndct import IterativePruningRunner
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
        
        # Re-save fine-tuned model (full state_dict + optional slim).
        pruned_weight_path = derive_weight_path(m_cfg['model_path'], "_pruned")
        slim_weight_path = derive_weight_path(m_cfg['model_path'], "_slim")
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        save_path = os.path.join(project_root, pruned_weight_path)
        slim_path = os.path.join(project_root, slim_weight_path)
        
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
