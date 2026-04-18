import os
import sys
import torch
import torchvision.models as models
from pytorch_nndct.apis import Inspector
import argparse

# --- Path auto-fix to find configs in project root ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from model_config import get_active_model
from dataset_config import get_active_dataset
from board_config import DPU_FINGERPRINT

def run_model_inspector():
    # 1. Load active configurations
    m_cfg = get_active_model()
    d_cfg = get_active_dataset()
    
    # 2. Define and create a specific output directory
    # Resulting path: build/resnet18/inspector_report/
    output_dir = os.path.join("build", m_cfg['name'].lower(), "inspector_report")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"=== Starting Inspector for: {m_cfg['name']} ===")
    print(f"=== Output Directory: {output_dir} ===")

    device = torch.device("cpu")
    
    # 3. Dynamic Model Instantiation
    try:
        model_fn = getattr(models, m_cfg['model_class'])
        model = model_fn()
    except AttributeError:
        print(f"Error: {m_cfg['model_class']} not found in torchvision.models")
        return

    # Adjust final layer to match dataset classes
    num_classes = len(d_cfg['classes'])
    last_layer_name = m_cfg.get('last_layer_name', 'fc')
    
    try:
        # Dynamické načtení poslední vrstvy (např. 'fc' nebo 'classifier')
        last_layer = getattr(model, last_layer_name)
        
        # Zjištění in_features (MobileNet má Sequential blok, ResNet má přímo Linear)
        if isinstance(last_layer, torch.nn.Sequential):
            in_features = last_layer[-1].in_features
        else:
            in_features = last_layer.in_features
            
        # Dynamické nahrazení vrstvy novou hlavičkou pro náš počet tříd
        setattr(model, last_layer_name, torch.nn.Linear(in_features, num_classes))
        
    except AttributeError:
        print(f"Chyba: Model nemá vrstvu s názvem '{last_layer_name}'. Zkontroluj model_config.py.")
        exit(1)

    # 4. Load weights
    try:
        checkpoint = torch.load(m_cfg['model_path'], map_location=device)
        if isinstance(checkpoint, torch.nn.Module):
            model = checkpoint
        else:
            model.load_state_dict(checkpoint)
    except FileNotFoundError:
        print(f"Error: Model weights not found at {m_cfg['model_path']}")
        return

    model.eval()

    # 5. Initialize Inspector with hardware fingerprint
    # We pass the output_dir to the Inspector to keep root clean
    inspector = Inspector(DPU_FINGERPRINT)

    # 6. Run Inspection
    input_h, input_w = m_cfg['input_shape']
    dummy_input = torch.randn(1, 3, input_h, input_w)
    
    print(f"Inspecting with input shape: {dummy_input.shape}...")
    
    # The inspector results will be placed in the specified directory
    # Note: NNDCT might still create some temp files, but the core report 
    # will be managed here.
    inspector.inspect(model, (dummy_input,), device=device, output_dir=output_dir)
    
    print(f"=== Inspection Complete. Report saved to {output_dir} ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Model ID from model_config.py')
    parser.add_argument('--dataset', type=str, help='Dataset ID from dataset_config.py')
    args = parser.parse_args()

    # Pass the CLI arguments to the config getters
    m_cfg = get_active_model(args.model)
    d_cfg = get_active_dataset(args.dataset)
    run_model_inspector()