# Global selector for the active dataset
ACTIVE_DATASET_ID = "intel_images"

DATASETS = {
    "intel_images": {
        "name": "Intel Image Classification",
        "path": "data/intel_images/seg_test",      # Path for inference testing
        "calib_path": "data/intel_images/seg_calib", # Path for quantization calibration
        "classes": ["buildings", "forest", "glacier", "mountain", "sea", "street"],
        "input_shape": (150, 150), # Height, Width as defined by the dataset
        "normalization": {
            "mean": [0.485, 0.456, 0.406], # ImageNet mean for RGB
            "std": [0.229, 0.224, 0.225]   # ImageNet standard deviation
        }
    },
    "industrial": {
        "name": "Industrial Inspection",
        "path": "data/industrial/test",
        "calib_path": "data/industrial/calib",
        "classes": ["ok", "defect"],
        "input_shape": (224, 224),
        "normalization": {
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5]
        }
    }
}

def get_active_dataset():
    """Returns the configuration dictionary for the selected active dataset."""
    if ACTIVE_DATASET_ID not in DATASETS:
        raise ValueError(f"Dataset ID '{ACTIVE_DATASET_ID}' not found in configuration.")
    return DATASETS[ACTIVE_DATASET_ID]