# Global selector for the active model
ACTIVE_MODEL_ID = "resnet18"

MODELS = {
    "resnet18": {
        "source": "torchvision",
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
        "model_path": "models/mobilenet_v2.pt",
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
    "mobilenet_v4_hybrid": {
        "source": "custom",
        "file_path": "models/mobilenet_v4_hybrid.py",
        "type": "classification",
        "name": "MobileNetV4_Hybrid",
        "model_class": "MobileNetV4HybridLarge",
        "last_layer_name": "head", # We will target the 'head' Sequential block
        "input_shape": (384, 384), # CRITICAL: Must be 384
        "model_path": "models/mobilenet_v4.pt", # Your weights from Colab
        "gops": 3.8 # Approximate for the Large version
    },
    "inception_v3": {
        "type": "classification",
        "name": "InceptionV3",
        "model_class": "inception_v3",
        "input_shape": (299, 299), # Note: Inception requires 299x299
        "model_path": "models/inception_v3.pt",
        "gops": 5.71
    },
    "yolov5s": {
        "source": "custom",
        "file_path": "models/yolov5s.py",      # You will need to provide this model file
        "type": "detection",
        "name": "YOLOv5-Small",
        "model_class": "YOLOv5",               # The main class inside yolov5s.py
        "input_shape": (640, 640),             # YOLO standard resolution
        "model_path": "models/yolov5s.pt",
        "gops": 15.8,
        "conf_threshold": 0.25,                # Minimum score to keep a bounding box
        "iou_threshold": 0.45                  # Overlap threshold for NMS
    },
    "unet_res18": {
        "source": "custom",
        "file_path": "models/unet.py",         # You will need to provide this model file
        "type": "segmentation",
        "name": "UNet_ResNet18",
        "model_class": "UNet",                 # The main class inside unet.py
        "input_shape": (512, 512),             # Typical segmentation resolution
        "model_path": "models/unet.pt",
        "gops": 25.0
    }
}

def get_active_model(model_id=None):
    """Returns configuration for the model, allowing CLI override."""
    # Use the provided model_id, otherwise fall back to the global ACTIVE_MODEL_ID
    target_id = model_id if model_id else ACTIVE_MODEL_ID
    
    if target_id not in MODELS:
        available = ", ".join(MODELS.keys())
        raise ValueError(f"Model ID '{target_id}' not found. Available: {available}")
    return MODELS[target_id]