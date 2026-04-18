# Global selector for the active model
ACTIVE_MODEL_ID = "mobilenet_v2"

MODELS = {
    "resnet18": {
        "type": "classification",
        "name": "ResNet18",
        "model_path": "models/resnet18.pt", # Path to your PyTorch weights
        "model_class": "resnet18",          # Used to instantiate the model in scripts
        "input_shape": (224, 224),
        "gops": 3.64
    },
    "resnet50": {
        "type": "classification",
        "name": "ResNet50",
        "model_path": "models/resnet50.pt",
        "model_class": "resnet50",
        "input_shape": (224, 224),
        "gops": 7.71
    },
    "mobilenet_v2": {
        "type": "classification",
        "name": "MobileNetV2",
        "model_class": "mobilenet_v2",
        "last_layer_name": "classifier",
        "input_shape": (224, 224),
        "model_path": "models/mobilenetv2.pt",
        "gops": 0.44
    },
    "mobilenet_v3": {
        "type": "classification",
        "name": "MobileNetV3-Large",
        "model_class": "mobilenet_v3",
        "input_shape": (224, 224),
        "model_path": "models/mobilenet_v3.pt",
        "gops": 0.44
    },
    "mobilenet_v4": {
        "type": "classification",
        "name": "MobileNetV4-Medium",
        "model_class": "mobilenet_v4",
        "input_shape": (224, 224),
        "model_path": "models/mobilenet_v4.pt",
        "gops": 0.92
    },
    "inception_v3": {
        "type": "classification",
        "name": "InceptionV3",
        "model_class": "inception_v3",
        "input_shape": (299, 299), # Note: Inception requires 299x299
        "model_path": "models/inception_v3.pt",
        "gops": 5.71
    }
}

def get_active_model():
    """Returns configuration for the active model."""
    if ACTIVE_MODEL_ID not in MODELS:
        raise ValueError(f"Model ID '{ACTIVE_MODEL_ID}' not found in configuration.")
    return MODELS[ACTIVE_MODEL_ID]