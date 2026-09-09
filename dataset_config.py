from kria_ai.classification.config import DATASETS as CLASSIFICATION_DATASETS
from kria_ai.yolov26.detection.config import DATASETS as DETECTION_DATASETS
from kria_ai.yolov26.segmentation.config import DATASETS as SEGMENTATION_DATASETS


ACTIVE_DATASET_ID = "intel_images"


def _classification(config):
    return {
        "name": config.name,
        "folder_name": config.board_root.parent.name,
        "calib_path": str(config.calibration_root),
        "classes": list(config.classes),
        "normalization": {"mean": list(config.mean), "std": list(config.std)},
    }


def _yolo(config):
    values = {
        "name": config.name,
        "folder_name": config.board_images.name,
        "images_train": str(config.train_images),
        "images_val": str(config.validation_images),
        "labels_train": str(config.train_labels),
        "labels_val": str(config.validation_labels),
        "subset_cache_dir": str(config.subset_cache_dir),
        "classes": list(config.classes),
        "normalization": {"mean": list(config.mean), "std": list(config.std)},
    }
    if hasattr(config, "board_labels"):
        values["board_labels"] = str(config.board_labels)
    return values


DATASETS = {
    **{dataset_id: _classification(config) for dataset_id, config in CLASSIFICATION_DATASETS.items()},
    **{dataset_id: _yolo(config) for dataset_id, config in DETECTION_DATASETS.items()},
}
for dataset_id, config in SEGMENTATION_DATASETS.items():
    DATASETS.setdefault(dataset_id, _yolo(config)).update(
        {"board_labels": str(config.board_labels)}
    )


def get_active_dataset(dataset_id=None):
    resolved_id = dataset_id or ACTIVE_DATASET_ID
    try:
        return DATASETS[resolved_id]
    except KeyError as error:
        raise ValueError(f"Unknown dataset {resolved_id!r}; available: {', '.join(DATASETS)}") from error
