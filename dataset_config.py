# =============================================================
# DATASET CONFIGURATION
# =============================================================
ACTIVE_DATASET_ID = "coco_detection"  # Default dataset; can be overridden via CLI

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
    """Returns configuration for the dataset, allowing CLI override."""
    target_id = dataset_id if dataset_id else ACTIVE_DATASET_ID
    
    if target_id not in DATASETS:
        available = ", ".join(DATASETS.keys())
        raise ValueError(f"Dataset ID '{target_id}' not found. Available: {available}")
    return DATASETS[target_id]