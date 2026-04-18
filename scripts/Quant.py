# DŮLEŽITÉ: NNDCT musí být první!
import pytorch_nndct
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import os
import argparse

# Nastavení argumentů podle dokumentace
parser = argparse.ArgumentParser()
parser.add_argument('--quant_mode', default='calib', choices=['calib', 'test'], help='quantization mode')
parser.add_argument('--subset_len', default=100, type=int, help='number of images for quantization')
parser.add_argument('--batch_size', default=32, type=int, help='batch size for calibration')
args = parser.parse_args()

def run_quantization():
    device = torch.device("cpu")
    quant_mode = args.quant_mode
    
    # 1. Načtení modelu (používáme tvůj "celý model" z předchozího kroku)
    checkpoint = torch.load('Resnet18/resnet18_intel.pt', map_location=device)
    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint
    else:
        # Pokud by to byl jen state_dict, musíme nejdřív vytvořit kostru
        model = torchvision.models.resnet18()
        model.fc = torch.nn.Linear(model.fc.in_features, 6)
        model.load_state_dict(checkpoint)
    
    model.eval()

    # 2. Příprava dat (Calibration Dataset)
    # Pro test/deploy mód s batch_size 1, jinak tvých 32
    curr_batch_size = 1 if quant_mode == 'test' else args.batch_size
    
    transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = ImageFolder(root='Resnet18/calibration_data', transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=curr_batch_size, shuffle=False)

    # 3. Inicializace Quantizeru
    # (batch, channels, height, width)
    input_shape = torch.randn([1, 3, 224, 224])
    quantizer = pytorch_nndct.apis.torch_quantizer(
        quant_mode, model, (input_shape,), device=device)
    
    quant_model = quantizer.quant_model

    # 4. Spuštění s daty
    print(f"Spouštím model v režimu: {quant_mode}")
    count = 0
    with torch.no_grad():
        for images, _ in loader:
            quant_model(images)
            count += images.size(0)
            if count >= args.subset_len:
                break

    # 5. Export výsledků podle manuálu
    if quant_mode == 'calib':
        quantizer.export_quant_config()
        print("Kalibrace hotova. Konfigurace uložena v quantize_result/")
    else:
        # Pro FPGA potřebujeme XMODEL
        quantizer.export_xmodel(deploy_check=False, output_dir="quantize_result")
        print("Export hotov. XMODEL najdeš v quantize_result/")

if __name__ == '__main__':
    run_quantization()