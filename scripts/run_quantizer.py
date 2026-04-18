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
    actual_subset_len = 1 if args.quant_mode == 'test' else args.subset_len
    # ---------------------------------------------------------
    
    # Define and create output directory
    output_dir = os.path.join("build", m_cfg['name'].lower(), "quantize_result")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n=== Starting Quantization: {m_cfg['name']} ===")
    print(f"=== Mode: {args.quant_mode} | Target: {actual_subset_len} images ===")
    device = torch.device("cpu") 
    
    # 2. Dynamic Model Loading
    try:
        checkpoint = torch.load(m_cfg['model_path'], map_location=device)
        
        if isinstance(checkpoint, torch.nn.Module):
            model = checkpoint
        else:
            model_fn = getattr(torchvision.models, m_cfg['model_class'])
            model = model_fn()
            num_classes = len(d_cfg['classes'])
            
            # Dynamic layer replacement
            last_layer_name = m_cfg.get('last_layer_name', 'fc')
            last_layer = getattr(model, last_layer_name)
            if isinstance(last_layer, torch.nn.Sequential):
                in_features = last_layer[-1].in_features
            else:
                in_features = last_layer.in_features
                
            setattr(model, last_layer_name, torch.nn.Linear(in_features, num_classes))
            model.load_state_dict(checkpoint)
            
        model.eval()
        print(f"[INFO] Loaded model weights from {m_cfg['model_path']}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 3. Data Preparation
    curr_batch_size = 1 if args.quant_mode == 'test' else args.batch_size
    transform = transforms.Compose([
        transforms.Resize(m_cfg['input_shape']),
        transforms.ToTensor(),
        transforms.Normalize(d_cfg['normalization']['mean'], 
                             d_cfg['normalization']['std'])
    ])
    
    try:
        dataset = ImageFolder(root=d_cfg['calib_path'], transform=transform)
        loader = torch.utils.data.DataLoader(dataset, batch_size=curr_batch_size, shuffle=False)
        print(f"[INFO] Loaded {len(dataset)} images from {d_cfg['calib_path']}")
    except Exception as e:
        print(f"Error loading calibration dataset: {e}")
        return

    # 4. Initialize Quantizer
    input_h, input_w = m_cfg['input_shape']
    dummy_input = torch.randn([1, 3, input_h, input_w])
    
    quantizer = pytorch_nndct.apis.torch_quantizer(
        args.quant_mode, model, (dummy_input,), device=device, output_dir=output_dir)
    
    quant_model = quantizer.quant_model

    # 5. Run Execution Loop (Forward Pass)
    print(f"[INFO] Processing forward pass (Batch Size: {curr_batch_size})...")
    processed_count = 0
    with torch.no_grad():
        for images, _ in loader:
            quant_model(images)
            processed_count += images.size(0)
            
            # Progress display (caps at target length)
            display_num = min(processed_count, actual_subset_len)
            percent = (display_num / actual_subset_len) * 100
            sys.stdout.write(f"\r[INFO] Progress: {display_num}/{actual_subset_len} ({percent:.1f}%) ")
            sys.stdout.flush()
            
            if processed_count >= actual_subset_len:
                break
    
    sys.stdout.write(f"\r[INFO] Progress: {actual_subset_len}/{actual_subset_len} (100.0%) Done!\n")

    # 6. Export Final Results
    if args.quant_mode == 'calib':
        quantizer.export_quant_config()
        print(f"[INFO] Calibration finished. Config saved to: {output_dir}")
    else:
        quantizer.export_xmodel(deploy_check=False, output_dir=output_dir)
        print(f"[INFO] Export finished. XMODEL generated in: {output_dir}")

if __name__ == '__main__':
    run_quantization()