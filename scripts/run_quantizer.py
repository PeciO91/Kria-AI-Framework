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

# Project-root import path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from model_config import get_active_model
from dataset_config import get_active_dataset
from model_utils import prepare_model
from dataset_utils import FlatImageDataset
from optimizer_utils import evaluate_loss


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

    model = prepare_model(m_cfg, d_cfg, device)

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
    elif task_type == 'detection':
        print("[INFO] Using YOLO Letterbox loader for detection calibration.")
        # Letterbox handles the resize; only ToTensor + Normalize remain.
        yolo_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(d_cfg['normalization']['mean'],
                                 d_cfg['normalization']['std']),
        ])
        dataset = FlatImageDataset(
            root_dir=d_cfg['calib_path'],
            transform=yolo_transform,
            letterbox_shape=m_cfg['input_shape'],
        )
        # Random sampling for detection calibration as well.
        if len(dataset) > actual_subset_len:
            random.seed(42)
            indices = random.sample(range(len(dataset)), actual_subset_len)
            dataset = Subset(dataset, indices)
    else:
        print(f"[INFO] Using flat-folder image loader for {task_type} task.")
        dataset = FlatImageDataset(root_dir=d_cfg['calib_path'], transform=base_transform)
        # Random sampling for other task types as well.
        if len(dataset) > actual_subset_len:
            random.seed(42)
            indices = random.sample(range(len(dataset)), actual_subset_len)
            dataset = Subset(dataset, indices)

    loader = torch.utils.data.DataLoader(dataset, batch_size=curr_batch_size, shuffle=False)
    input_h, input_w = m_cfg['input_shape']
    dummy_input = torch.randn([1, 3, input_h, input_w]).to(device)

    quantizer = pytorch_nndct.apis.torch_quantizer(
        args.quant_mode, model, (dummy_input,), device=device, output_dir=output_dir)
    quant_model = quantizer.quant_model

    # Optional AdaQuant fast fine-tuning. Only available for classification.
    if args.fast_ft:
        if task_type != 'classification':
            print(f"[WARN] --fast_ft is currently only wired for classification "
                  f"(task='{task_type}'). Skipping AdaQuant.")
        elif args.quant_mode == 'calib':
            print("[INFO] Phase 1: Running Fast Fine-Tuning (AdaQuant)...")
            loss_fn = torch.nn.CrossEntropyLoss()
            quantizer.fast_finetune(evaluate_loss, (quant_model, loader, loss_fn))
        else:
            print("[INFO] Phase 2: Loading Fine-Tuned parameters...")
            quantizer.load_ft_param()

    # Forward pass over calibration data.
    print("[INFO] Processing forward pass...")
    processed_count = 0
    with torch.no_grad():
        for images, _ in loader:
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
