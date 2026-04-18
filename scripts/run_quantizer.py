import os
import sys
import argparse
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import pytorch_nndct

# --- Path auto-fix ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from model_config import get_active_model
from dataset_config import get_active_dataset
from model_utils import prepare_model

# 1. Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='Model ID')
parser.add_argument('--dataset', type=str, help='Dataset ID')
parser.add_argument('--quant_mode', default='calib', choices=['calib', 'test'])
parser.add_argument('--subset_len', default=100, type=int, help='Calib images')
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--fast_ft', action='store_true', help='Enable Fast Fine-Tuning')
args = parser.parse_args()

# The evaluation function required by Fast Fine-Tuning
def evaluate(model, loader, loss_fn):
    """AdaQuant needs to minimize loss, not track accuracy."""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
    return total_loss

def run_quantization():
    m_cfg = get_active_model(args.model)
    d_cfg = get_active_dataset(args.dataset)
    
    actual_subset_len = 1 if args.quant_mode == 'test' else args.subset_len
    output_dir = os.path.join("build", m_cfg['name'].lower(), "quantize_result")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n=== Starting Quantization: {m_cfg['name']} ===")
    device = torch.device("cpu") 
    
    # 2. Prepare Model
    model = prepare_model(m_cfg, d_cfg, device)

    # 3. Data Preparation
    curr_batch_size = 1 if args.quant_mode == 'test' else args.batch_size
    transform = transforms.Compose([
        transforms.Resize(m_cfg['input_shape']),
        transforms.ToTensor(),
        transforms.Normalize(d_cfg['normalization']['mean'], d_cfg['normalization']['std'])
    ])
    
    dataset = ImageFolder(root=d_cfg['calib_path'], transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=curr_batch_size, shuffle=False)

    # 4. Initialize Quantizer
    input_h, input_w = m_cfg['input_shape']
    dummy_input = torch.randn([1, 3, input_h, input_w])
    
    quantizer = pytorch_nndct.apis.torch_quantizer(
        args.quant_mode, model, (dummy_input,), device=device, output_dir=output_dir)
    
    quant_model = quantizer.quant_model

    # 5. Handle Fast Fine-Tuning Logic (The Official Flow)
    if args.fast_ft:
        if args.quant_mode == 'calib':
            print("[INFO] Phase 1: Running Fast Fine-Tuning (AdaQuant)...")
            # AdaQuant needs a loss function to optimize weight adjustments
            loss_fn = torch.nn.CrossEntropyLoss()
            quantizer.fast_finetune(evaluate, (quant_model, loader, loss_fn))
        
        elif args.quant_mode == 'test':
            print("[INFO] Phase 2: Loading Fine-Tuned parameters...")
            # CRITICAL: This loads the optimized weights before exporting the XMODEL
            quantizer.load_ft_param()

    # 6. Standard Forward Pass (Re-calibration)
    print(f"[INFO] Processing forward pass...")
    processed_count = 0
    with torch.no_grad():
        for images, _ in loader:
            quant_model(images)
            processed_count += images.size(0)
            
            display_num = min(processed_count, actual_subset_len)
            percent = (display_num / actual_subset_len) * 100
            sys.stdout.write(f"\r[INFO] Progress: {display_num}/{actual_subset_len} ({percent:.1f}%) ")
            sys.stdout.flush()
            
            if processed_count >= actual_subset_len:
                break
    
    sys.stdout.write(f"\r[INFO] Progress: {actual_subset_len}/{actual_subset_len} (100.0%) Done!\n")

    # 7. Export Final Results
    if args.quant_mode == 'calib':
        quantizer.export_quant_config()
        print(f"[INFO] Calibration finished.")
    else:
        # If fast_ft was used in calib, the XMODEL exported here will contain the optimized weights
        quantizer.export_xmodel(deploy_check=False, output_dir=output_dir)
        print(f"[INFO] Export finished. XMODEL is now optimized with Fast Fine-Tuning.")

if __name__ == '__main__':
    run_quantization()