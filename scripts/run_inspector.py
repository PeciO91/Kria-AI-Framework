import os
import sys
import torch
import argparse
from pytorch_nndct.apis import Inspector

# --- Path auto-fix ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from model_config import get_active_model
from dataset_config import get_active_dataset
from board_config import DPU_FINGERPRINT
from model_utils import prepare_model 

def run_model_inspector(model_id, dataset_id, prune_threshold):
    m_cfg = get_active_model(model_id)
    d_cfg = get_active_dataset(dataset_id)
    
    output_dir = os.path.join("build", m_cfg['name'].lower(), "inspector_report")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n=== Starting Inspector for: {m_cfg['name']} ===")
    
    # NEW: Automatic device detection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    # NEW: Pass the pruning threshold to the loader
    model = prepare_model(m_cfg, d_cfg, device, prune_threshold=prune_threshold)

    inspector = Inspector(DPU_FINGERPRINT)
    input_h, input_w = m_cfg['input_shape']
    dummy_input = torch.randn(1, 3, input_h, input_w).to(device)
    
    print(f"[INFO] Inspecting with input shape: {dummy_input.shape}...")
    inspector.inspect(model, (dummy_input,), device=device, output_dir=output_dir)
    print(f"=== Inspection Complete. Report saved to {output_dir} ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Model ID')
    parser.add_argument('--dataset', type=str, help='Dataset ID')
    # LOOPHOLE FIX: Added threshold argument
    parser.add_argument('--prune_threshold', type=float, help='Threshold used for pruning')
    args = parser.parse_args()

    run_model_inspector(args.model, args.dataset, args.prune_threshold)