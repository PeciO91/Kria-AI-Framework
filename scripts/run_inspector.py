import os
import sys
import torch
import argparse
from pytorch_nndct.apis import Inspector

# --- Path auto-fix to find configs in project root ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from model_config import get_active_model
from dataset_config import get_active_dataset
from board_config import DPU_FINGERPRINT
from model_utils import prepare_model # <--- Our new shared utility

def run_model_inspector(model_id, dataset_id):
    # 1. Load active configurations (Supports CLI override)
    m_cfg = get_active_model(model_id)
    d_cfg = get_active_dataset(dataset_id)
    
    # 2. Define and create output directory
    output_dir = os.path.join("build", m_cfg['name'].lower(), "inspector_report")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n=== Starting Inspector for: {m_cfg['name']} ===")
    print(f"=== Output Directory: {output_dir} ===")

    device = torch.device("cpu")
    
    # 3. Use the shared utility to load and adapt the model
    # This handles torchvision, timm, and custom .py files automatically
    model = prepare_model(m_cfg, d_cfg, device)

    # 4. Initialize Inspector with hardware fingerprint
    inspector = Inspector(DPU_FINGERPRINT)

    # 5. Run Inspection
    input_h, input_w = m_cfg['input_shape']
    dummy_input = torch.randn(1, 3, input_h, input_w)
    
    print(f"[INFO] Inspecting with input shape: {dummy_input.shape}...")
    
    # The inspector results will be placed in the specified directory
    inspector.inspect(model, (dummy_input,), device=device, output_dir=output_dir)
    
    print(f"=== Inspection Complete. Report saved to {output_dir} ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Model ID from model_config.py')
    parser.add_argument('--dataset', type=str, help='Dataset ID from dataset_config.py')
    args = parser.parse_args()

    run_model_inspector(args.model, args.dataset)