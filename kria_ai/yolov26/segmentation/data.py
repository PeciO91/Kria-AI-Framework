from pathlib import Path

from kria_ai.yolov26.data import (
    CalibrationImageDataset,
    _resolve,
    _split_images,
    build_or_load_subset_indices,
)
from kria_ai.yolov26.preprocess import image_to_float_tensor, letterbox


COCO_CATEGORY_IDS = (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39,
    40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
    58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77,
    78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
)
COCO_CATEGORY_TO_CLASS = {
    category_id: class_id
    for class_id, category_id in enumerate(COCO_CATEGORY_IDS)
}


def _clip_unit(value):
    return max(0.0, min(1.0, value))


def _box_from_transformed_points(points, input_shape):
    import numpy as np

    target_height, target_width = input_shape
    x_values = np.clip(points[:, 0], 0.0, float(target_width))
    y_values = np.clip(points[:, 1], 0.0, float(target_height))
    x_min, x_max = float(x_values.min()), float(x_values.max())
    y_min, y_max = float(y_values.min()), float(y_values.max())
    return [
        _clip_unit(((x_min + x_max) / 2.0) / target_width),
        _clip_unit(((y_min + y_max) / 2.0) / target_height),
        _clip_unit((x_max - x_min) / target_width),
        _clip_unit((y_max - y_min) / target_height),
    ]


def parse_polygon_labels(
    label_path,
    original_shape,
    input_shape,
    ratio,
    padding,
    num_classes=80,
):
    """Parse normalized YOLO polygons into letterboxed boxes and masks."""
    import cv2
    import numpy as np

    label_path = Path(label_path)
    if not label_path.is_file():
        return [], [], []

    original_height, original_width = original_shape
    target_height, target_width = input_shape
    scale = ratio[0]
    pad_width, pad_height = padding
    boxes = []
    classes = []
    masks = []

    with label_path.open("r", encoding="utf-8") as label_file:
        for line_number, line in enumerate(label_file, start=1):
            tokens = line.strip().split()
            if not tokens:
                continue
            coordinate_count = len(tokens) - 1
            if coordinate_count < 6 or coordinate_count % 2:
                raise ValueError(f"{label_path}:{line_number}: segmentation rows require at least three coordinate pairs")
            class_id = int(tokens[0])
            if not 0 <= class_id < num_classes:
                raise ValueError(f"{label_path}:{line_number}: class ID {class_id} is outside [0, {num_classes - 1}]")
            points = np.asarray(tokens[1:], dtype=np.float32).reshape(-1, 2)
            points[:, 0] = points[:, 0] * original_width * scale + pad_width
            points[:, 1] = points[:, 1] * original_height * scale + pad_height

            mask = np.zeros((target_height, target_width), dtype=np.uint8)
            cv2.fillPoly(mask, [points.astype(np.int32)], 1)
            boxes.append(_box_from_transformed_points(points, input_shape))
            classes.append(class_id)
            masks.append(mask)
    return boxes, classes, masks


def parse_coco_annotations(
    annotations,
    original_shape,
    input_shape,
    ratio,
    padding,
    category_id_map=None,
):
    """Rasterize polygon-style COCO annotations in letterboxed image space."""
    import cv2
    import numpy as np

    del original_shape  # COCO polygon and bbox coordinates are absolute pixels.
    category_id_map = category_id_map or COCO_CATEGORY_TO_CLASS
    target_height, target_width = input_shape
    scale = ratio[0]
    pad_width, pad_height = padding
    boxes = []
    classes = []
    masks = []

    for annotation in annotations:
        class_id = category_id_map.get(annotation.get("category_id"))
        segments = annotation.get("segmentation")
        if class_id is None or not isinstance(segments, list):
            continue

        mask = np.zeros((target_height, target_width), dtype=np.uint8)
        transformed_parts = []
        for segment in segments:
            if (
                not isinstance(segment, (list, tuple))
                or len(segment) < 6
                or len(segment) % 2
            ):
                continue
            points = np.asarray(segment, dtype=np.float32).reshape(-1, 2)
            points[:, 0] = points[:, 0] * scale + pad_width
            points[:, 1] = points[:, 1] * scale + pad_height
            cv2.fillPoly(mask, [points.astype(np.int32)], 1)
            transformed_parts.append(points)
        if not transformed_parts:
            continue

        bbox = annotation.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            x_min, y_min, box_width, box_height = map(float, bbox)
            box = [
                _clip_unit(
                    ((x_min + box_width / 2.0) * scale + pad_width)
                    / target_width
                ),
                _clip_unit(
                    ((y_min + box_height / 2.0) * scale + pad_height)
                    / target_height
                ),
                _clip_unit(box_width * scale / target_width),
                _clip_unit(box_height * scale / target_height),
            ]
        else:
            box = _box_from_transformed_points(
                np.concatenate(transformed_parts, axis=0),
                input_shape,
            )
        boxes.append(box)
        classes.append(class_id)
        masks.append(mask)
    return boxes, classes, masks


def _augment_image_boxes_and_masks(image, boxes, masks):
    import random

    import cv2
    import numpy as np

    boxes_array = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if random.random() < 0.5:
        image = cv2.flip(image, 1)
        boxes_array[:, 0] = 1.0 - boxes_array[:, 0]
        masks = [cv2.flip(mask, 1) for mask in masks]

    if random.random() < 0.5:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 0] = np.clip(hsv[:, :, 0] * random.uniform(0.9, 1.1), 0, 179)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.7, 1.3), 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * random.uniform(0.7, 1.3), 0, 255)
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return image, boxes_array.tolist(), masks


def _default_annotations_path(images_dir):
    images_dir = _resolve(images_dir)
    candidate = images_dir.parent / "annotations" / f"instances_{images_dir.name}.json"
    return candidate if candidate.is_file() else None


def _load_coco_index(annotations_path, class_names=None):
    import json

    with annotations_path.open("r", encoding="utf-8") as annotation_file:
        data = json.load(annotation_file)

    image_ids = {
        image["file_name"]: image["id"]
        for image in data.get("images", ())
        if "file_name" in image and "id" in image
    }
    annotations_by_image = {image_id: [] for image_id in image_ids.values()}
    for annotation in data.get("annotations", ()):
        image_id = annotation.get("image_id")
        if image_id in annotations_by_image:
            annotations_by_image[image_id].append(annotation)

    category_id_map = dict(COCO_CATEGORY_TO_CLASS)
    if class_names is not None and data.get("categories"):
        class_lookup = {name: index for index, name in enumerate(class_names)}
        mapped_categories = {
            category["id"]: class_lookup[category["name"]]
            for category in data["categories"]
            if category.get("name") in class_lookup and "id" in category
        }
        if mapped_categories:
            category_id_map = mapped_categories
    return image_ids, annotations_by_image, category_id_map


class SegmentationDataset(CalibrationImageDataset):
    """YOLO polygon/COCO instance-segmentation dataset."""

    def __init__(
        self,
        images_dir,
        labels_dir,
        input_shape=(640, 640),
        normalization=None,
        augment=False,
        indices=None,
        annotations_path=None,
        classes=None,
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
        self.image_ids = None
        self.annotations_by_image = None
        self.category_id_map = None

        if annotations_path is None:
            annotations_path = _default_annotations_path(self.images_dir)
        else:
            annotations_path = _resolve(annotations_path)
            if not annotations_path.is_file():
                raise FileNotFoundError(
                    f"COCO annotations file not found: {annotations_path}"
                )
        if annotations_path is not None:
            (
                self.image_ids,
                self.annotations_by_image,
                self.category_id_map,
            ) = _load_coco_index(annotations_path, class_names=classes)

    def __getitem__(self, index):
        import cv2
        import numpy as np
        import torch

        filename = self.image_files[index]
        image_path = self.images_dir / filename
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise FileNotFoundError(f"Unable to read segmentation image: {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        original_shape = image_rgb.shape[:2]
        resized, ratio, padding = letterbox(image_rgb, new_shape=self.input_shape)

        if self.image_ids is not None and filename in self.image_ids:
            image_id = self.image_ids[filename]
            boxes, classes, masks = parse_coco_annotations(
                self.annotations_by_image.get(image_id, ()),
                original_shape=original_shape,
                input_shape=self.input_shape,
                ratio=ratio,
                padding=padding,
                category_id_map=self.category_id_map,
            )
        else:
            boxes, classes, masks = parse_polygon_labels(
                self.labels_dir / f"{Path(filename).stem}.txt",
                original_shape=original_shape,
                input_shape=self.input_shape,
                ratio=ratio,
                padding=padding,
                num_classes=self.num_classes,
            )

        if self.augment and boxes:
            resized, boxes, masks = _augment_image_boxes_and_masks(
                resized,
                boxes,
                masks,
            )

        target_height, target_width = self.input_shape
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
        masks_tensor = (
            torch.tensor(np.asarray(masks), dtype=torch.float32)
            if masks
            else torch.zeros((0, target_height, target_width), dtype=torch.float32)
        )
        return image_tensor, {
            "bboxes": boxes_tensor,
            "cls": classes_tensor,
            "masks": masks_tensor,
        }


def segmentation_collate_fn(batch):
    """Collate aligned boxes, classes, and instance masks for YOLO loss."""
    import torch

    images = []
    batch_boxes = []
    batch_classes = []
    batch_indices = []
    batch_masks = []
    for batch_index, (image, target) in enumerate(batch):
        images.append(image)
        count = target["bboxes"].shape[0]
        if target["masks"].shape[0] != count:
            raise ValueError("Segmentation boxes and masks must have matching counts")
        if count:
            batch_boxes.append(target["bboxes"])
            batch_classes.append(target["cls"])
            batch_indices.append(
                torch.full((count,), batch_index, dtype=torch.float32)
            )
            batch_masks.append(target["masks"])

    images_tensor = torch.stack(images, dim=0)
    if batch_boxes:
        boxes_tensor = torch.cat(batch_boxes, dim=0)
        classes_tensor = torch.cat(batch_classes, dim=0)
        batch_index_tensor = torch.cat(batch_indices, dim=0)
        masks_tensor = torch.cat(batch_masks, dim=0)
    else:
        boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
        classes_tensor = torch.zeros((0,), dtype=torch.int64)
        batch_index_tensor = torch.zeros((0,), dtype=torch.float32)
        masks_tensor = torch.zeros(
            (0, images_tensor.shape[2], images_tensor.shape[3]),
            dtype=torch.float32,
        )
    return images_tensor, {
        "bboxes": boxes_tensor,
        "cls": classes_tensor,
        "batch_idx": batch_index_tensor,
        "masks": masks_tensor,
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
        raise ValueError(f"Unknown YOLOv26 segmentation split {split!r}") from error


def build_dataset(
    model_config,
    dataset_config,
    split="train",
    subset_len=None,
    seed=42,
    augment=False,
    annotations_path=None,
):
    normalization = {"mean": dataset_config.mean, "std": dataset_config.std}
    dataset = SegmentationDataset(
        images_dir=_split_images(dataset_config, split),
        labels_dir=_split_labels(dataset_config, split),
        input_shape=model_config.input_size,
        normalization=normalization,
        augment=augment,
        annotations_path=annotations_path,
        classes=dataset_config.classes,
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
    annotations_path=None,
):
    from torch.utils.data import DataLoader

    dataset = build_dataset(
        model_config,
        dataset_config,
        split=split,
        subset_len=subset_len,
        seed=seed,
        augment=augment,
        annotations_path=annotations_path,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=segmentation_collate_fn,
    )


YoloSegmentationDataset = SegmentationDataset
parse_segmentation_labels = parse_polygon_labels
yolo_collate_fn = segmentation_collate_fn
