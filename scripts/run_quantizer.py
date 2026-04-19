import os
import sys
import argparse
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from PIL import Image 
import pytorch_nndct

# --- Path auto-fix ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from model_config import get_active_model
from dataset_config import get_active_dataset
from model_utils import prepare_model

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='Model ID')
parser.add_argument('--dataset', type=str, help='Dataset ID')
parser.add_argument('--quant_mode', default='calib', choices=['calib', 'test'])
parser.add_argument('--subset_len', default=100, type=int, help='Calib images')
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--fast_ft', action='store_true', help='Enable Fast Fine-Tuning')
# LOOPHOLE FIX: Added threshold argument
parser.add_argument('--prune_threshold', type=float, help='Threshold used for pruning')
args = parser.parse_args()

class SimpleImageDataset(Dataset):
    """A generic dataset that loads images from a flat folder for calibration."""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0  # Return dummy label 0, quantizer ignores it anyway

def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss = 0
    device = next(model.parameters()).device
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
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
    
    # NEW: Automatic device detection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    # NEW: Pass pruning threshold to loader
    model = prepare_model(m_cfg, d_cfg, device, prune_threshold=args.prune_threshold)

    # Data Prep
    curr_batch_size = 1 if args.quant_mode == 'test' else args.batch_size
    transform = transforms.Compose([
        transforms.Resize(m_cfg['input_shape']),
        transforms.ToTensor(),
        transforms.Normalize(d_cfg['normalization']['mean'], d_cfg['normalization']['std'])
    ])
    
    # Choose dataset loader based on task type
    if m_cfg.get('type') == 'classification':
        # Verify dataset structure for ImageFolder (needs sub-folders)
        subdirs = [d for d in os.listdir(d_cfg['calib_path']) if os.path.isdir(os.path.join(d_cfg['calib_path'], d))]
        if not subdirs:
            raise RuntimeError(f"Quantization Error: {d_cfg['calib_path']} must contain at least one subdirectory (e.g. 'data/calib/images/1.jpg') for ImageFolder to work.")
        dataset = ImageFolder(root=d_cfg['calib_path'], transform=transform)
    else:
        # Detection and Segmentation just use flat folders
        print(f"[INFO] Using flat-folder image loader for {m_cfg.get('type')} task.")
        dataset = SimpleImageDataset(root_dir=d_cfg['calib_path'], transform=transform)
        if len(dataset) == 0:
            raise RuntimeError(f"Quantization Error: No images found in {d_cfg['calib_path']}.")

    loader = torch.utils.data.DataLoader(dataset, batch_size=curr_batch_size, shuffle=False)
    input_h, input_w = m_cfg['input_shape']
    dummy_input = torch.randn([1, 3, input_h, input_w]).to(device)
    
    quantizer = pytorch_nndct.apis.torch_quantizer(
        args.quant_mode, model, (dummy_input,), device=device, output_dir=output_dir)
    
    quant_model = quantizer.quant_model

    if args.fast_ft:
        if args.quant_mode == 'calib':
            print("[INFO] Phase 1: Running Fast Fine-Tuning (AdaQuant)...")
            loss_fn = torch.nn.CrossEntropyLoss()
            quantizer.fast_finetune(evaluate, (quant_model, loader, loss_fn))
        elif args.quant_mode == 'test':
            print("[INFO] Phase 2: Loading Fine-Tuned parameters...")
            quantizer.load_ft_param()

    # Forward Pass
    print(f"[INFO] Processing forward pass...")
    processed_count = 0
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            quant_model(images)
            processed_count += images.size(0)
            
            display_num = min(processed_count, actual_subset_len)
            percent = (display_num / actual_subset_len) * 100
            sys.stdout.write(f"\r[INFO] Progress: {display_num}/{actual_subset_len} ({percent:.1f}%) ")
            sys.stdout.flush()
            if processed_count >= actual_subset_len: break
    
    print(f"\n[INFO] Forward pass finished.")

    if args.quant_mode == 'calib':
        quantizer.export_quant_config()
    else:
        quantizer.export_xmodel(deploy_check=False, output_dir=output_dir)
        print(f"[INFO] Export finished.")

if __name__ == '__main__':
    run_quantization()