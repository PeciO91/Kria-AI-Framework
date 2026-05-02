"""
Vitis AI structural pruning stage (work in progress).

Wraps the Vitis AI NNDCT `Pruner` to slim a trained model along the channel
dimension. The output is a structurally pruned skeleton whose weights still
need to be recovered through fine-tuning before quantization.

Currently implemented:
  - prune mode: applies a single-shot pruning ratio and saves the skeleton.

Pending:
  - ana mode: per-layer sensitivity analysis (requires a validation loader).
  - integrated fine-tuning loop to recover accuracy after pruning.
"""
import os
import sys
import argparse

import torch
from pytorch_nndct.apis import Pruner

# Project-root import path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from model_config import get_active_model
from dataset_config import get_active_dataset
from model_utils import prepare_model


def run_optimizer():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--mode', choices=['ana', 'prune'], default='prune')
    parser.add_argument('--ratio', type=float, default=0.2,
                        help='Channel reduction ratio (0.1-0.5)')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Fine-tuning epochs (reserved for future use)')
    args = parser.parse_args()

    m_cfg = get_active_model(args.model)
    d_cfg = get_active_dataset()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = prepare_model(m_cfg, d_cfg, device)

    input_h, input_w = m_cfg['input_shape']
    dummy_input = torch.randn([1, 3, input_h, input_w]).to(device)
    pruner = Pruner(model, dummy_input)

    pruned_weight_path = m_cfg['model_path'].replace(".pt", "_pruned.pt")

    if args.mode == 'ana':
        # TODO: requires a validation loader and an evaluate() callback.
        print("[WARN] 'ana' mode is not yet implemented. Use --mode prune instead.")
        return

    print(f"=== Starting Pruning: {m_cfg['name']} ===")
    model = pruner.prune(threshold=args.ratio)

    print("[WARN] Model is structurally pruned but its weights are now untrained.")
    print("[INFO] You must run fine-tuning before quantizing this skeleton.")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    save_path = os.path.join(project_root, pruned_weight_path)
    torch.save(model.state_dict(), save_path)
    print(f"[SUCCESS] Pruned skeleton saved to: {save_path}")


if __name__ == '__main__':
    run_optimizer()
