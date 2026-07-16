"""
Vitis AI INT8 calibration / export stage.

Two-phase quantization driver around `pytorch_nndct.apis.torch_quantizer`:

  1. quant_mode='calib': runs a forward pass over the calibration set so the
     quantizer can record per-tensor statistics. Optionally enables AdaQuant
     fast fine-tuning. Exports a quant_info.json under
     build/<model>/quantize_result/.
  2. quant_mode='test': re-runs a single forward pass and exports the
     INT8 xmodel that the compiler will consume.

The script picks the appropriate dataset loader based on m_cfg['type']:
classification uses torchvision.ImageFolder; detection uses a Letterbox
loader that mirrors the on-board preprocessing; everything else falls back
to a flat-folder loader.
"""
import os
import sys
import argparse
import random

import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
import pytorch_nndct

# Project-root import path (PROJECT_ROOT + scripts/common/ added to sys.path).
from _bootstrap import PROJECT_ROOT  # noqa: F401

from model_config import get_active_model
from dataset_config import get_active_dataset
from model_utils import prepare_model, apply_export_patch
from dataset_utils import YoloDataset, yolo_collate_fn, build_or_load_subset_indices
from optimizer_utils import evaluate_loss
from detection_profiles import get_profile


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        help='Model ID. Falls back to ACTIVE_MODEL_ID '
                             'in model_config.py when omitted.')
    parser.add_argument('--dataset', type=str,
                        help='Dataset ID. Falls back to ACTIVE_DATASET_ID '
                             'in dataset_config.py when omitted.')
    parser.add_argument('--quant_mode', default='calib', choices=['calib', 'test'])
    parser.add_argument('--subset_len', default=100, type=int,
                        help='Number of calibration images')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--fast_ft', action='store_true',
                        help='Enable AdaQuant fast fine-tuning')
    return parser.parse_args()


def evaluate_detection_loss(model, dataloader, loss_fn, device=None):
    """
    Evaluates detection loss over a dataloader. Used for AdaQuant fast fine-tuning
    of detection models. Ensured model outputs training formats for correct loss evaluation.
    """
    if device is None:
        device = next(model.parameters()).device
    total_loss = 0.0
    was_training = model.training
    model.train() # Must be in training mode to output both branches (one2one/one2many) for loss
    try:
        with torch.no_grad():
            for images, targets in dataloader:
                images = images.to(device)
                targets = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in targets.items()}
                outputs = model(images)
                loss_val = loss_fn(outputs, targets)
                if isinstance(loss_val, dict):
                    loss_val = sum(l for l in loss_val.values())
                elif isinstance(loss_val, (list, tuple)):
                    loss_val = sum(loss_val)
                total_loss += loss_val.item()
    finally:
        if not was_training:
            model.eval()
    return total_loss


# =============================================================
# MAIN
# =============================================================
def run_quantization(args):
    m_cfg = get_active_model(args.model)
    d_cfg = get_active_dataset(args.dataset)  # None falls back to ACTIVE_DATASET_ID

    actual_subset_len = 1 if args.quant_mode == 'test' else args.subset_len
    output_dir = os.path.join("build", m_cfg['name'].lower(), "quantize_result")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n=== Starting Quantization: {m_cfg['name']} ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    model = prepare_model(m_cfg, device)

    # Disable Detect head postprocessing to bypass the XIR compiler crash on
    # aten::topk. The patched forward exports the one2one branches (NMS-free)
    # as 6 split tensors that run_detection.py merges, instead of running the
    # in-graph DFL decode + topk. The Inspector stage applies the SAME patch so
    # it reports on exactly the graph we quantize and compile.
    apply_export_patch(model)

    # Build the calibration loader matching the model's task type.
    curr_batch_size = 1 if args.quant_mode == 'test' else args.batch_size
    base_transform = transforms.Compose([
        transforms.Resize(m_cfg['input_shape']),
        transforms.ToTensor(),
        transforms.Normalize(d_cfg['normalization']['mean'],
                             d_cfg['normalization']['std']),
    ])

    task_type = m_cfg.get('type')
    if task_type == 'classification':
        dataset = ImageFolder(root=d_cfg['calib_path'], transform=base_transform)
        # Limit to actual_subset_len with random sampling to avoid bias from
        # alphabetical ordering (ImageFolder sorts classes by name).
        if len(dataset) > actual_subset_len:
            random.seed(42)  # Fixed seed for reproducibility
            indices = random.sample(range(len(dataset)), actual_subset_len)
            dataset = Subset(dataset, indices)
    elif task_type in ('detection', 'segmentation'):
        # Anchor-free detection AND Ultralytics instance segmentation share the
        # letterbox preprocessing of the board runner. Calibration is forward
        # only, so the parsed labels are unused here (instance-seg polygon
        # labels are tolerated and simply ignored).
        print("[INFO] Using letterbox YOLO dataset loader for "
              f"{'instance-seg' if task_type == 'segmentation' else 'detection'} "
              "calibration.")
        
        # Load cached or newly built subset indices
        subset_indices = build_or_load_subset_indices(
            split="calib",
            n=actual_subset_len,
            cache_dir=d_cfg['subset_cache_dir']
        )
        
        dataset = YoloDataset(
            images_dir=d_cfg['images_train'],
            labels_dir=d_cfg['labels_train'],
            input_shape=m_cfg['input_shape'],
            normalization=d_cfg['normalization'],
            augment=False,
            indices=subset_indices
        )


    collate_fn = yolo_collate_fn if task_type in ('detection', 'segmentation') else None
    loader = torch.utils.data.DataLoader(dataset, batch_size=curr_batch_size, shuffle=False, collate_fn=collate_fn)
    input_h, input_w = m_cfg['input_shape']
    dummy_input = torch.randn([1, 3, input_h, input_w]).to(device)

    quantizer = pytorch_nndct.apis.torch_quantizer(
        args.quant_mode, model, (dummy_input,), device=device, output_dir=output_dir)
    quant_model = quantizer.quant_model

    # Optional AdaQuant fast fine-tuning. Supported for classification and detection.
    if args.fast_ft:
        if task_type not in ['classification', 'detection']:
            print(f"[WARN] --fast_ft is currently only wired for classification and detection "
                  f"(task='{task_type}'). Skipping AdaQuant.")
        elif args.quant_mode == 'calib':
            print("[INFO] Phase 1: Running Fast Fine-Tuning (AdaQuant)...")
            if task_type == 'classification':
                loss_fn = torch.nn.CrossEntropyLoss()
                quantizer.fast_finetune(evaluate_loss, (quant_model, loader, loss_fn))
            elif task_type == 'detection':
                profile = get_profile(m_cfg)
                loss_fn = profile.loss_fn(quant_model)
                quantizer.fast_finetune(evaluate_detection_loss, (quant_model, loader, loss_fn))
        else:
            print("[INFO] Phase 2: Loading Fine-Tuned parameters...")
            quantizer.load_ft_param()

    # Forward pass over calibration data.
    print("[INFO] Processing forward pass...")
    processed_count = 0
    with torch.no_grad():
        for batch in loader:
            images = batch[0] if isinstance(batch, (tuple, list)) else batch
            images = images.to(device)
            quant_model(images)
            processed_count += images.size(0)

            display_num = min(processed_count, actual_subset_len)
            percent = (display_num / actual_subset_len) * 100
            sys.stdout.write(f"\r[INFO] Progress: {display_num}/{actual_subset_len} "
                             f"({percent:.1f}%) ")
            sys.stdout.flush()
            if processed_count >= actual_subset_len:
                break
    print("\n[INFO] Forward pass finished.")

    # Export per phase.
    if args.quant_mode == 'calib':
        quantizer.export_quant_config()
    else:
        quantizer.export_xmodel(deploy_check=False, output_dir=output_dir)
        # Rename xmodel from <ActualClass>_int.xmodel to <model_id>_int.xmodel.
        # We use args.model (the canonical model_id from model_config) rather
        # than m_cfg['name'].lower() so the filename matches what board-side
        # runners look for. The Python class name is only meaningful to the
        # Vitis AI quantizer and must be stripped here.
        actual_class_name = model.__class__.__name__
        old_name = f"{actual_class_name}_int.xmodel"
        new_name = f"{args.model}_int.xmodel"
        old_path = os.path.join(output_dir, old_name)
        new_path = os.path.join(output_dir, new_name)
        if os.path.exists(old_path):
            os.rename(old_path, new_path)
            print(f"[INFO] Renamed xmodel from {old_name} to {new_name}")
        else:
            print(f"[WARNING] Expected file {old_path} not found, checking for existing xmodel files...")
            # List all xmodel files in the directory for debugging
            for f in os.listdir(output_dir):
                if f.endswith('.xmodel'):
                    print(f"[INFO] Found xmodel: {f}")
        print("[INFO] Export finished.")


if __name__ == '__main__':
    run_quantization(parse_args())
