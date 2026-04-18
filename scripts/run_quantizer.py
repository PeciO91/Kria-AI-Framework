import os
import sys
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import pytorch_nndct

# --- Path auto-fix to find configs in project root ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from model_config import get_active_model
from dataset_config import get_active_dataset

# 1. Argument Parser for switching modes
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='Model ID from model_config.py')
parser.add_argument('--dataset', type=str, help='Dataset ID from dataset_config.py')
parser.add_argument('--quant_mode', default='calib', choices=['calib', 'test'], 
                    help='Quantization mode: calib (calibration) or test (evaluation/export)')
parser.add_argument('--subset_len', default=100, type=int, 
                    help='Number of images to use for quantization')
parser.add_argument('--batch_size', default=32, type=int, 
                    help='Batch size for calibration')
args = parser.parse_args()

def run_quantization():
    # Load modular configurations
    m_cfg = get_active_model(args.model)
    d_cfg = get_active_dataset(args.dataset)
    
    # ---------------------------------------------------------
    # AUTOMATIC SUBSET OPTIMIZATION
    # For 'test' mode, we only need 1 image for export.
    # For 'calib' mode, we use the value from command line arguments.
    actual_subset_len = 1 if args.quant_mode == 'test' else args.subset_len
    # ---------------------------------------------------------
    
    # Define and create output directory
    output_dir = os.path.join("build", m_cfg['name'].lower(), "quantize_result")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"=== Starting Quantization: {m_cfg['name']} ===")
    print(f"=== Mode: {args.quant_mode} | Subset Length: {actual_subset_len} ===")
    device = torch.device("cpu") # Quantization usually runs on CPU
    
    # 2. Dynamic Model Loading
    try:
        # Load weights from configured path
        checkpoint = torch.load(m_cfg['model_path'], map_location=device)
        
        if isinstance(checkpoint, torch.nn.Module):
            model = checkpoint
        else:
            # Reconstruct model skeleton from torchvision if only state_dict is provided
            model_fn = getattr(torchvision.models, m_cfg['model_class'])
            model = model_fn()
            num_classes = len(d_cfg['classes'])
            last_layer_name = m_cfg.get('last_layer_name', 'fc')
            
            last_layer = getattr(model, last_layer_name)
            if isinstance(last_layer, torch.nn.Sequential):
                in_features = last_layer[-1].in_features
            else:
                in_features = last_layer.in_features
                
            setattr(model, last_layer_name, torch.nn.Linear(in_features, num_classes))
            model.load_state_dict(checkpoint)
            
        model.eval()
        print(f"Successfully loaded model weights from {m_cfg['model_path']}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 3. Data Preparation using dataset_config values
    curr_batch_size = 1 if args.quant_mode == 'test' else args.batch_size
    
    # Preprocessing pipeline driven by dataset_config
    transform = transforms.Compose([
        transforms.Resize(m_cfg['input_shape']),
        transforms.ToTensor(),
        transforms.Normalize(d_cfg['normalization']['mean'], 
                             d_cfg['normalization']['std'])
    ])
    
    try:
        # Load calibration data from the path defined in dataset_config
        dataset = ImageFolder(root=d_cfg['calib_path'], transform=transform)
        loader = torch.utils.data.DataLoader(dataset, batch_size=curr_batch_size, shuffle=False)
        print(f"Loaded {len(dataset)} calibration images from {d_cfg['calib_path']}")
    except Exception as e:
        print(f"Error loading calibration dataset: {e}")
        return

    # 4. Initialize Quantizer
    # Use input shape from dataset configuration
    input_h, input_w = m_cfg['input_shape']
    dummy_input = torch.randn([1, 3, input_h, input_w])
    
    quantizer = pytorch_nndct.apis.torch_quantizer(
        args.quant_mode, model, (dummy_input,), device=device, output_dir=output_dir)
    
    quant_model = quantizer.quant_model

# 5. Run Execution Loop (Forward Pass)
    print(f"Running forward pass for {actual_subset_len} images...")
    processed_count = 0
    with torch.no_grad():
        for images, _ in loader:
            quant_model(images)
            processed_count += images.size(0)
            if processed_count >= actual_subset_len:
                break

    # 6. Export Final Results
    if args.quant_mode == 'calib':
        quantizer.export_quant_config()
        print(f"Calibration finished. Config saved to: {output_dir}")
    else:
        # Exporting XMODEL for the FPGA (Kria)
        # deploy_check=False skips on-CPU verification to save time
        quantizer.export_xmodel(deploy_check=False, output_dir=output_dir)
        print(f"Export finished. XMODEL generated in: {output_dir}")

if __name__ == '__main__':
    run_quantization()