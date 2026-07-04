"""
Central registry of model definitions consumed by every stage of the
pipeline (inspector, quantizer, compiler, board runners).

Each entry describes:

  - source         : 'torchvision' or 'custom' (loader strategy in model_utils).
  - loader         : Optional custom loader name ('yolo' or 'ultralytics').
  - type           : 'classification', 'detection', or 'segmentation'.
  - name           : Human-readable name; used to derive the build directory
                     and the compiled xmodel filename.
  - model_class    : Class or factory name to instantiate.
  - model_path     : Path to .pt weights, relative to the project root.
  - input_shape    : (H, W) input resolution.
  - gops           : Approximate compute cost; used for compute-efficiency
                     metrics in the analytical report.
  - last_layer_name: Optional, classification only. Override of the final
                     layer attribute name (default 'fc') when adapting class
                     count.
  - file_path      : Generic custom-source loaders only. Location of the model
                     definition file.
  - repo_path      : Local repository path for repo-backed loaders.
  - yaml_path      : Optional architecture YAML for YOLO-style loaders.

Detection models additionally carry conf_threshold, iou_threshold, anchors
and strides used by the on-board YOLO decoder.
"""

# Default model when no --model is passed.
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
        "source": "torchvision",
        "type": "classification",
        "name": "ResNet50",
        "model_path": "models/resnet50.pt",
        "model_class": "resnet50",
        "input_shape": (224, 224),
        "gops": 7.71
    },
    "mobilenet_v2": {
        "source": "torchvision",
        "type": "classification",
        "name": "MobileNetV2",
        "model_class": "mobilenet_v2",
        "last_layer_name": "classifier",
        "input_shape": (224, 224),
        "model_path": "models/mobilenet_v2.pt",
        "gops": 0.44
    },
    "mobilenet_v3": {
        "source": "torchvision",
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
        "source": "torchvision",
        "type": "classification",
        "name": "InceptionV3",
        "model_class": "inception_v3",
        "input_shape": (299, 299), # Note: Inception requires 299x299
        "model_path": "models/inception_v3.pt",
        "gops": 5.71
    },
    "yolov5n": {
        "source": "custom",
        "loader": "yolo",
        "type": "detection",
        "name": "YOLOv5n",
        "input_shape": (640, 640),
        "model_path": "models/yolov5n/yolov5n.pt",     # YOLOv5n weights
        "repo_path": "models/yolov5n",
        "yaml_path": "models/yolov5n/models/yolov5n.yaml",  # Architecture config
        "gops": 4.5,                           # YOLOv5n is ~4.5 GOPs
        "conf_threshold": 0.25,
        "iou_threshold": 0.45,
        # YOLOv5 anchors per detection level (P3, P4, P5)
        "anchors": [
            [[10, 13], [16, 30], [33, 23]],
            [[30, 61], [62, 45], [59, 119]],
            [[116, 90], [156, 198], [373, 326]]
        ],
        "strides": [8, 16, 32]
    },
    "yolov26s": {
        "source": "custom",
        "loader": "ultralytics",
        "type": "detection",
        "name": "YOLOv26s",
        "input_shape": (640, 640),
        "model_path": "models/yolo26s.pt",
        "repo_path": "models/ultralytics-main",
        "yaml_path": "configs/yolov26s_dpu.yaml",
        "gops": 22.8,
        "conf_threshold": 0.1,
        "iou_threshold": 0.45,
        "decoder": "ultralytics_anchor_free",
        "num_classes": 80,
        "reg_max": 1,
        "end2end": True,
        "max_det": 300,
        "strides": [8, 16, 32],
        # Output convs of the Detect head must keep their channel counts during
        # pruning. Detect is the last layer in configs/yolov26s_dpu.yaml
        # (layers 0-21 + Detect), so it lives at model.model[22].
        "prune_excludes": [
            "model.22.cv2.*.2",
            "model.22.cv3.*.2",
            "model.22.one2one_cv2.*.2",
            "model.22.one2one_cv3.*.2"
        ]
    },
    "yolov26n_seg": {
        "source": "custom",
        "loader": "ultralytics",
        "type": "segmentation",
        "name": "YOLOv26n-Seg",
        "input_shape": (640, 640),
        "model_path": "models/yolov26n-seg.pt",
        "repo_path": "models/ultralytics-main",
        "yaml_path": "configs/yolov26n-seg_dpu.yaml",
        "gops": 10.5,
        # Instance-segmentation: reuse the anchor-free detection decoder for
        # boxes/classes, then assemble per-object masks on the ARM CPU from the
        # exported mask coefficients + prototypes (see run_instance_seg.py).
        "seg_instance": True,
        "decoder": "ultralytics_anchor_free",
        "num_classes": 80,
        "reg_max": 1,
        "end2end": True,
        "max_det": 300,
        "strides": [8, 16, 32],
        # Mask head: 32 coefficients per anchor decoded against 256-channel
        # prototypes (Segment26 / Proto26). mask_threshold binarizes the final
        # per-instance mask after sigmoid.
        "num_masks": 32,
        "num_protos": 256,
        "conf_threshold": 0.1,
        "iou_threshold": 0.45,
        "mask_threshold": 0.5,
        # Output convs of the Segment26 head must keep their channel counts
        # during pruning. Segment26 is the last layer in
        # configs/yolo26-seg_dpu.yaml (layers 0-21 + Segment26), so it lives at
        # model.model[22]. Protect box (cv2), class (cv3) and mask (cv4) convs
        # of both the one2many and one2one branches.
        "prune_excludes": [
            "model.22.cv2.*.2",
            "model.22.cv3.*.2",
            "model.22.cv4.*.2",
            "model.22.one2one_cv2.*.2",
            "model.22.one2one_cv3.*.2",
            "model.22.one2one_cv4.*.2"
        ]
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
    """Return the configuration dict for `model_id`, falling back to ACTIVE_MODEL_ID."""
    target_id = model_id if model_id else ACTIVE_MODEL_ID
    if target_id not in MODELS:
        available = ", ".join(MODELS.keys())
        raise ValueError(f"Model ID '{target_id}' not found. Available: {available}")
    return MODELS[target_id]