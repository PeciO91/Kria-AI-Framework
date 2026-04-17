# Global selector for the active model
ACTIVE_MODEL_ID = "resnet18"

MODELS = {
    "resnet18": {
        "name": "ResNet18",
        "model_path": "models/resnet18.pt", # Path to your PyTorch weights
        "model_class": "resnet18",          # Used to instantiate the model in scripts
        "xmodel_path": "output_kria/resnet18_intel_kria.xmodel",
        "gops": 3.64
    },
    "resnet50": {
        "name": "ResNet50",
        "model_path": "models/resnet50.pt",
        "model_class": "resnet50",
        "xmodel_path": "output_kria/resnet50.xmodel",
        "gops": 7.71
    }
}

def get_active_model():
    """Returns configuration for the active model."""
    if ACTIVE_MODEL_ID not in MODELS:
        raise ValueError(f"Model ID '{ACTIVE_MODEL_ID}' not found in configuration.")
    return MODELS[ACTIVE_MODEL_ID]