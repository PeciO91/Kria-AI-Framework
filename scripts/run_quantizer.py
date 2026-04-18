import os
import sys
import argparse
import torch
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
from model_utils import prepare_model

# 1. Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='Model ID from model_config.py')
parser.add_argument('--dataset', type=str, help='Dataset ID from dataset_config.py')
parser.add_argument('--quant_mode', default='calib', choices=['calib', 'test'])
parser.add_argument('--subset_len', default=100, type=int, help='Images for calibration')
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--fast_ft', action='store_true', help='Enable Fast Fine-Tuning (AdaQuant)')
args = parser.parse_args()

# Evaluation function required by Fast Fine-Tuning
def evaluate(model, loader, device):
    """Simple evaluation loop to provide feedback during fine-tuning."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def run_quantization():
    m_cfg = get_active_model(args.model)
    d_cfg = get_active_dataset(args.dataset)
    
    actual_subset_len = 1 if args.quant_mode == 'test' else args.subset_len
    output_dir = os.path.join("build", m_cfg['name'].lower(), "quantize_result")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n=== Starting Quantization: {m_cfg['name']} ===")
    device = torch.device("cpu") # NNDCT quantization usually stays on CPU
    
    # 2. Load and prepare model
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

    # 5. Fast Fine-Tuning (Optional)
    # This must be run in 'calib' mode before the final forward pass
    if args.quant_mode == 'calib' and args.fast_ft:
        print("[INFO] Starting Fast Fine-Tuning (This may take several minutes)...")
        # fast_finetune uses a subset of the calibration data to optimize weights
        quantizer.fast_finetune(evaluate, (quant_model, loader, device))

    # 6. Standard Forward Pass (Calibration)
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
        print(f"[INFO] Calibration finished. Config saved to: {output_dir}")
    else:
        quantizer.export_xmodel(deploy_check=False, output_dir=output_dir)
        print(f"[INFO] Export finished. XMODEL generated in: {output_dir}")

if __name__ == '__main__':
    run_quantization()