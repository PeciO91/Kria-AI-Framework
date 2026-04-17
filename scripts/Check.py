import torch
import torchvision.models as models
from pytorch_nndct.apis import Inspector

# 1. Načtení tvého modelu
device = torch.device("cpu")
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 6) # Intel dataset má 6 tříd
checkpoint = torch.load('Resnet18/resnet18_intel.pt', map_location=device)

if isinstance(checkpoint, torch.nn.Module):
    print("Detekován celý model, používám ho přímo.")
    model = checkpoint
else:
    print("Detekován state_dict, nahrávám váhy do kostry.")
    model.load_state_dict(checkpoint)

model.eval()

# 2. Inicializace Inspektora s tvým fingerprintem
# Použijeme tvůj reálný fingerprint z Krie
inspector = Inspector("0x101000056010407") 

# 3. Spuštění inspekce (potřebuje ukázkový vstup)
dummy_input = torch.randn(1, 3, 224, 224)
inspector.inspect(model, (dummy_input,), device=device)