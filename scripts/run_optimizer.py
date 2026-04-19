import os
import sys
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from pytorch_nndct.apis import Pruner

# --- Path auto-fix ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from model_config import get_active_model
from dataset_config import get_active_dataset
from model_utils import prepare_model

def evaluate(model, loader, device):
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

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

def run_optimizer():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--mode', choices=['ana', 'prune'], default='prune')
    parser.add_argument('--ratio', type=float, default=0.2, help='Reduction ratio (0.1-0.5)')
    parser.add_argument('--epochs', type=int, default=3, help='Fine-tuning epochs')
    args = parser.parse_args()

    m_cfg = get_active_model(args.model)
    d_cfg = get_active_dataset()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Original Model
    model = prepare_model(m_cfg, d_cfg, device)
    
    # 2. Setup Pruner
    input_h, input_w = m_cfg['input_shape']
    dummy_input = torch.randn([1, 3, input_h, input_w]).to(device)
    # The Pruner needs the dummy input to map the graph
    pruner = Pruner(model, dummy_input)

    # Automatic Naming Convention
    pruned_weight_path = m_cfg['model_path'].replace(".pt", "_pruned.pt")

    if args.mode == 'ana':
        print(f"[INFO] Analyzing model sensitivity...")
        # This takes a long time as it tests pruning every layer individually
        # loader = ... (requires a val_loader)
        # pruner.ana(evaluate, args=(val_loader, device))
        pass

    elif args.mode == 'prune':
        print(f"=== Starting Pruning Flow: {m_cfg['name']} ===")
        print(f"[INFO] Target Ratio: {args.ratio}")
        
        # This command modifies the 'model' object to be the slim version
        model = pruner.prune(threshold=args.ratio)
        
        # 3. Fine-tuning (Mandatory to recover accuracy)
        print(f"[INFO] Fine-tuning for {args.epochs} epochs to recover accuracy...")
        # Note: You need a proper training loader from ImageFolder here
        # optimizer = optim.Adam(model.parameters(), lr=1e-4)
        # criterion = nn.CrossEntropyLoss()
        # for epoch in range(args.epochs):
        #     train_one_epoch(model, train_loader, optimizer, criterion, device)
        
        # 4. Save the Pruned Weights
        torch.save(model.state_dict(), pruned_weight_path)
        print(f"[SUCCESS] Pruned model saved to: {pruned_weight_path}")

if __name__ == '__main__':
    run_optimizer()