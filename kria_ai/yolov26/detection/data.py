from pathlib import Path

from kria_ai.yolov26.data import (
    CalibrationImageDataset,
    _resolve,
    _split_images,
    build_or_load_subset_indices,
)
from kria_ai.yolov26.preprocess import image_to_float_tensor, letterbox


def _clip_unit(value):
    return max(0.0, min(1.0, value))


def parse_detection_labels(
    label_path,
    original_shape,
    input_shape,
    ratio,
    padding,
    num_classes=80,
):
    """Parse only five-column YOLO detection rows into letterboxed xywh."""
    label_path = Path(label_path)
    if not label_path.is_file():
        return [], []

    original_height, original_width = original_shape
    target_height, target_width = input_shape
    scale = ratio[0]
    pad_width, pad_height = padding
    boxes = []
    classes = []

    with label_path.open("r", encoding="utf-8") as label_file:
        for line_number, line in enumerate(label_file, start=1):
            tokens = line.strip().split()
            if not tokens:
                continue
            if len(tokens) != 5:
                raise ValueError(f"{label_path}:{line_number}: detection rows require exactly five columns")
            class_id = int(tokens[0])
            if not 0 <= class_id < num_classes:
                raise ValueError(f"{label_path}:{line_number}: class ID {class_id} is outside [0, {num_classes - 1}]")
            center_x, center_y, box_width, box_height = map(float, tokens[1:])
            boxes.append([
                _clip_unit(
                    (center_x * original_width * scale + pad_width) / target_width
                ),
                _clip_unit(
                    (center_y * original_height * scale + pad_height) / target_height
                ),
                _clip_unit(box_width * original_width * scale / target_width),
                _clip_unit(box_height * original_height * scale / target_height),
            ])
            classes.append(class_id)
    return boxes, classes


def _augment_image_and_boxes(image, boxes):
    import random

    import cv2
    import numpy as np

    boxes_array = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if random.random() < 0.5:
        image = cv2.flip(image, 1)
        boxes_array[:, 0] = 1.0 - boxes_array[:, 0]

    if random.random() < 0.5:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 0] = np.clip(hsv[:, :, 0] * random.uniform(0.9, 1.1), 0, 179)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.7, 1.3), 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * random.uniform(0.7, 1.3), 0, 255)
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return image, boxes_array.tolist()


class DetectionDataset(CalibrationImageDataset):
    """YOLO detection dataset with strictly five-column box labels."""

    def __init__(
        self,
        images_dir,
        labels_dir,
        input_shape=(640, 640),
        normalization=None,
        augment=False,
        indices=None,
        num_classes=80,
    ):
        super().__init__(
            images_dir=images_dir,
            input_shape=input_shape,
            normalization=normalization,
            indices=indices,
        )
        self.labels_dir = _resolve(labels_dir)
        self.augment = augment
        self.num_classes = num_classes

    def __getitem__(self, index):
        import cv2
        import torch

        filename = self.image_files[index]
        image_path = self.images_dir / filename
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise FileNotFoundError(f"Unable to read detection image: {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        original_shape = image_rgb.shape[:2]
        resized, ratio, padding = letterbox(image_rgb, new_shape=self.input_shape)

        boxes, classes = parse_detection_labels(
            self.labels_dir / f"{Path(filename).stem}.txt",
            original_shape=original_shape,
            input_shape=self.input_shape,
            ratio=ratio,
            padding=padding,
            num_classes=self.num_classes,
        )
        if self.augment and boxes:
            resized, boxes = _augment_image_and_boxes(resized, boxes)

        image_tensor = image_to_float_tensor(resized, mean=self.mean, std=self.std)
        boxes_tensor = (
            torch.tensor(boxes, dtype=torch.float32)
            if boxes
            else torch.zeros((0, 4), dtype=torch.float32)
        )
        classes_tensor = (
            torch.tensor(classes, dtype=torch.int64)
            if classes
            else torch.zeros((0,), dtype=torch.int64)
        )
        return image_tensor, {"bboxes": boxes_tensor, "cls": classes_tensor}


def detection_collate_fn(batch):
    """Collate detection targets into the flat Ultralytics loss format."""
    import torch

    images = []
    batch_boxes = []
    batch_classes = []
    batch_indices = []
    for batch_index, (image, target) in enumerate(batch):
        images.append(image)
        count = target["bboxes"].shape[0]
        if count:
            batch_boxes.append(target["bboxes"])
            batch_classes.append(target["cls"])
            batch_indices.append(
                torch.full((count,), batch_index, dtype=torch.float32)
            )

    images_tensor = torch.stack(images, dim=0)
    if batch_boxes:
        boxes_tensor = torch.cat(batch_boxes, dim=0)
        classes_tensor = torch.cat(batch_classes, dim=0)
        batch_index_tensor = torch.cat(batch_indices, dim=0)
    else:
        boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
        classes_tensor = torch.zeros((0,), dtype=torch.int64)
        batch_index_tensor = torch.zeros((0,), dtype=torch.float32)
    return images_tensor, {
        "bboxes": boxes_tensor,
        "cls": classes_tensor,
        "batch_idx": batch_index_tensor,
    }


def _split_labels(dataset_config, split):
    roots = {
        "calibration": dataset_config.train_labels,
        "train": dataset_config.train_labels,
        "validation": dataset_config.validation_labels,
        "val": dataset_config.validation_labels,
        "evaluation": dataset_config.validation_labels,
    }
    try:
        return roots[split]
    except KeyError as error:
        raise ValueError(f"Unknown YOLOv26 detection split {split!r}") from error


def build_dataset(
    model_config,
    dataset_config,
    split="train",
    subset_len=None,
    seed=42,
    augment=False,
):
    normalization = {"mean": dataset_config.mean, "std": dataset_config.std}
    dataset = DetectionDataset(
        images_dir=_split_images(dataset_config, split),
        labels_dir=_split_labels(dataset_config, split),
        input_shape=model_config.input_size,
        normalization=normalization,
        augment=augment,
        num_classes=model_config.num_classes,
    )
    if subset_len is not None:
        indices = build_or_load_subset_indices(
            split=split,
            n=subset_len,
            seed=seed,
            cache_dir=dataset_config.subset_cache_dir,
            dataset_length=len(dataset),
        )
        dataset.image_files = [dataset.image_files[index] for index in indices]
    return dataset


def build_loader(
    model_config,
    dataset_config,
    split="train",
    subset_len=None,
    batch_size=4,
    seed=42,
    shuffle=False,
    num_workers=0,
    augment=False,
):
    from torch.utils.data import DataLoader

    dataset = build_dataset(
        model_config,
        dataset_config,
        split=split,
        subset_len=subset_len,
        seed=seed,
        augment=augment,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=detection_collate_fn,
    )


YoloDetectionDataset = DetectionDataset
yolo_collate_fn = detection_collate_fn
