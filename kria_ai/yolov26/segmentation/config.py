from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kria_ai.yolov26.config import COCO_CLASSES, YOLO_MEAN, YOLO_STD


@dataclass(frozen=True)
class SegmentationModelConfig:
    id: str
    name: str
    checkpoint_path: Path
    repository_path: Path
    architecture_path: Path
    input_size: tuple[int, int]
    num_classes: int
    reg_max: int
    strides: tuple[int, ...]
    num_masks: int
    prototype_channels: int
    confidence_threshold: float
    mask_threshold: float
    max_detections: int
    prune_excludes: tuple[str, ...]

    def __post_init__(self):
        if self.num_classes != len(COCO_CLASSES):
            raise ValueError(f"{self.id}: YOLOv26 segmentation must use {len(COCO_CLASSES)} COCO classes")
        if self.reg_max < 1 or self.num_masks < 1 or not self.strides:
            raise ValueError(f"{self.id}: invalid head metadata")


@dataclass(frozen=True)
class SegmentationDatasetConfig:
    id: str
    name: str
    train_images: Path
    validation_images: Path
    train_labels: Path
    validation_labels: Path
    subset_cache_dir: Path
    board_images: Path
    board_labels: Path
    classes: tuple[str, ...]
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


MODELS = {
    "yolov26n_seg": SegmentationModelConfig(
        id="yolov26n_seg",
        name="YOLOv26n-Seg",
        checkpoint_path=Path("models/yolov26n-seg.pt"),
        repository_path=Path("models/ultralytics-main"),
        architecture_path=Path("configs/yolov26/segmentation/yolov26n_seg_dpu.yaml"),
        input_size=(640, 640),
        num_classes=80,
        reg_max=1,
        strides=(8, 16, 32),
        num_masks=32,
        prototype_channels=256,
        confidence_threshold=0.4,
        mask_threshold=0.5,
        max_detections=300,
        prune_excludes=(
            "model.22.cv2.*.2",
            "model.22.cv3.*.2",
            "model.22.cv4.*.2",
            "model.22.one2one_cv2.*.2",
            "model.22.one2one_cv3.*.2",
            "model.22.one2one_cv4.*.2",
            "model.22.proto.*",
            "SegmentationModel::SegmentationModel/C3k2[model]/C3k2[6]/C3k[m]/ModuleList[0]/Conv[cv3]/Conv2d[conv]/ret.117",
            "SegmentationModel::SegmentationModel/C3k2[model]/C3k2[8]/C3k[m]/ModuleList[0]/Conv[cv3]/Conv2d[conv]/ret.173",
            "SegmentationModel::SegmentationModel/C3k2[model]/C3k2[12]/C3k[m]/ModuleList[0]/Conv[cv3]/Conv2d[conv]/ret.241",
            "SegmentationModel::SegmentationModel/C3k2[model]/C3k2[15]/C3k[m]/ModuleList[0]/Conv[cv3]/Conv2d[conv]/ret.297",
            "SegmentationModel::SegmentationModel/C3k2[model]/C3k2[18]/C3k[m]/ModuleList[0]/Conv[cv3]/Conv2d[conv]/ret.355",
            "SegmentationModel::SegmentationModel/C3k2[model]/C3k2[21]/C3k[m]/ModuleList[0]/Conv[cv3]/Conv2d[conv]/ret.413",
        ),
    ),
}

DATASETS = {
    "coco": SegmentationDatasetConfig(
        id="coco",
        name="COCO 2017 Instance Segmentation",
        train_images=Path("data/coco2017/train2017"),
        validation_images=Path("data/coco2017/val2017"),
        train_labels=Path("data/coco2017/labels/train2017"),
        validation_labels=Path("data/coco2017/labels/val2017"),
        subset_cache_dir=Path("data/coco2017/.subsets/segmentation"),
        board_images=Path("datasets/coco2017"),
        board_labels=Path("datasets/coco2017/labels/val2017"),
        classes=COCO_CLASSES,
        mean=YOLO_MEAN,
        std=YOLO_STD,
    ),
}

DEFAULT_MODEL_ID = "yolov26n_seg"
DEFAULT_DATASET_ID = "coco"


def get_model(model_id=None):
    resolved_id = model_id or DEFAULT_MODEL_ID
    try:
        return MODELS[resolved_id]
    except KeyError as error:
        raise ValueError(f"Unknown YOLOv26 segmentation model {resolved_id!r}; available: {', '.join(MODELS)}") from error


def get_dataset(dataset_id=None):
    resolved_id = dataset_id or DEFAULT_DATASET_ID
    try:
        return DATASETS[resolved_id]
    except KeyError as error:
        raise ValueError(f"Unknown YOLOv26 segmentation dataset {resolved_id!r}; available: {', '.join(DATASETS)}") from error
