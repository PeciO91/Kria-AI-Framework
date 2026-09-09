from dataclasses import dataclass
from pathlib import Path

from kria_ai.yolov26.config import COCO_CLASSES, YOLO_MEAN, YOLO_STD


@dataclass(frozen=True)
class DetectionModelConfig:
    id: str
    name: str
    checkpoint_path: Path
    repository_path: Path
    architecture_path: Path
    input_size: tuple[int, int]
    num_classes: int
    reg_max: int
    strides: tuple[int, ...]
    confidence_threshold: float
    max_detections: int
    prune_excludes: tuple[str, ...]

    def __post_init__(self):
        if self.num_classes != len(COCO_CLASSES):
            raise ValueError(f"{self.id}: YOLOv26 detection must use {len(COCO_CLASSES)} COCO classes")
        if self.reg_max < 1 or not self.strides:
            raise ValueError(f"{self.id}: invalid head metadata")


@dataclass(frozen=True)
class DetectionDatasetConfig:
    id: str
    name: str
    train_images: Path
    validation_images: Path
    train_labels: Path
    validation_labels: Path
    subset_cache_dir: Path
    board_images: Path
    classes: tuple[str, ...]
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


MODELS = {
    "yolov26s": DetectionModelConfig(
        id="yolov26s",
        name="YOLOv26s",
        checkpoint_path=Path("models/yolo26s.pt"),
        repository_path=Path("models/ultralytics-main"),
        architecture_path=Path("configs/yolov26/detection/yolov26s_dpu.yaml"),
        input_size=(640, 640),
        num_classes=80,
        reg_max=1,
        strides=(8, 16, 32),
        confidence_threshold=0.1,
        max_detections=300,
        prune_excludes=(
            "model.22.cv2.*.2",
            "model.22.cv3.*.2",
            "model.22.one2one_cv2.*.2",
            "model.22.one2one_cv3.*.2",
        ),
    ),
}

DATASETS = {
    "coco": DetectionDatasetConfig(
        id="coco",
        name="COCO 2017 Detection",
        train_images=Path("data/coco2017/train2017"),
        validation_images=Path("data/coco2017/val2017"),
        train_labels=Path("data/coco2017/labels/train2017"),
        validation_labels=Path("data/coco2017/labels/val2017"),
        subset_cache_dir=Path("data/coco2017/.subsets/detection"),
        board_images=Path("datasets/coco2017"),
        classes=COCO_CLASSES,
        mean=YOLO_MEAN,
        std=YOLO_STD,
    ),
}

DEFAULT_MODEL_ID = "yolov26s"
DEFAULT_DATASET_ID = "coco"


def get_model(model_id=None):
    resolved_id = model_id or DEFAULT_MODEL_ID
    try:
        return MODELS[resolved_id]
    except KeyError as error:
        raise ValueError(f"Unknown YOLOv26 detection model {resolved_id!r}; available: {', '.join(MODELS)}") from error


def get_dataset(dataset_id=None):
    resolved_id = dataset_id or DEFAULT_DATASET_ID
    try:
        return DATASETS[resolved_id]
    except KeyError as error:
        raise ValueError(f"Unknown YOLOv26 detection dataset {resolved_id!r}; available: {', '.join(DATASETS)}") from error
