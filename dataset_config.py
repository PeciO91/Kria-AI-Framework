# =============================================================
# DATASET CONFIGURATION
# =============================================================
ACTIVE_DATASET_ID = "intel_images"

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
    }
}

def get_active_dataset(dataset_id=None):
    """Returns configuration for the dataset, allowing CLI override."""
    target_id = dataset_id if dataset_id else ACTIVE_DATASET_ID
    
    if target_id not in DATASETS:
        available = ", ".join(DATASETS.keys())
        raise ValueError(f"Dataset ID '{target_id}' not found. Available: {available}")
    return DATASETS[target_id]