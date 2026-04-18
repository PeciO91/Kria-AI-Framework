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

def get_active_dataset():
    """Returns the configuration of the currently active dataset."""
    if ACTIVE_DATASET_ID not in DATASETS:
        raise ValueError(f"Dataset ID '{ACTIVE_DATASET_ID}' not found.")
    return DATASETS[ACTIVE_DATASET_ID]