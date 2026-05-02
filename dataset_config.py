"""
Central registry of dataset definitions used by both the host-side
calibration loaders and the board-side accuracy / labeling logic.

Each entry contains:

  - name          : Human-readable description used in the analytical report.
  - folder_name   : Subdirectory under datasets/ on the board (or under
                    data/ on the host) where the images live.
  - calib_path    : Host-side path to the calibration set used by the
                    quantizer.
  - classes       : Ordered list of class names. Index = class id.
                    Classification: defines train_data/<class>/ structure
                    that run_inference.py walks. Detection: maps decoded
                    class ids to human-readable labels for drawn output.
  - normalization : Mean / std used for INT8 input normalization. YOLO
                    models trained on 0..1 inputs use mean=0, std=1.
"""

# Default dataset when no --dataset is passed.
ACTIVE_DATASET_ID = "coco_detection"

DATASETS = {
    "intel_images": {
        "name": "Intel Image Classification",
        "folder_name": "intel_images",  # Used for standardized path generation
        "classes": ["buildings", "forest", "glacier", "mountain", "sea", "street"],
        "calib_path": "data/intel_images/calibration_data",  # Path for calibration images
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        }
    },
    "intel_images_inception": {
        "name": "Intel Images (Inception Size)",
        "folder_name": "intel_images",  # Shares the same data folder as standard Intel images
        "classes": ["buildings", "forest", "glacier", "mountain", "sea", "street"],
        "calib_path": "data/intel_images/calibration_data",  # Path for calibration images
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        }
    },
    "coco_detection": {
        "name": "COCO Detection Calibration",
        "folder_name": "coco",
        # Point this to a folder containing ~200 images from your detection dataset
        "calib_path": "data/coco/calibration_data",
        "classes": [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
            "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
            "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
            "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
            "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
            "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
            "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
            "toothbrush"
        ],
        "normalization": {
            "mean": [0.0, 0.0, 0.0],  # YOLO often uses 0-1 scaling (mean 0, std 1)
            "std": [1.0, 1.0, 1.0]    # check your specific YOLO training config!
        }
    },
    "cityscapes_seg": {
        "name": "Cityscapes Segmentation Calibration",
        "folder_name": "cityscapes",
        # Point this to a folder containing ~200 images from your segmentation dataset
        "calib_path": "data/cityscapes/calibration_data",
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        }
    }
}


def get_active_dataset(dataset_id=None):
    """Return the configuration dict for `dataset_id`, falling back to ACTIVE_DATASET_ID."""
    target_id = dataset_id if dataset_id else ACTIVE_DATASET_ID
    if target_id not in DATASETS:
        available = ", ".join(DATASETS.keys())
        raise ValueError(f"Dataset ID '{target_id}' not found. Available: {available}")
    return DATASETS[target_id]