"""
Shared dataset utilities for the Vitis AI deployment pipeline.

Used by the quantizer (calibration) and the optimizer (fine-tuning) stages
so that host-side preprocessing stays consistent between them and matches
on-board preprocessing for detection models.
"""
import os
import random
import cv2
import numpy as np
from PIL import Image
import torch

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """
    Aspect-preserving resize with constant padding (YOLO-style letterbox).

    Lives in dataset_utils because it is used by both calibration-time
    preprocessing (this module's YOLO datasets) and on-board detection
    preprocessing (``scripts/detection/detection_utils.py`` re-exports it).
    Keep this implementation as the single source of truth.

    Returns the padded image, the (rw, rh) scale ratios, and the (dw, dh)
    padding applied to each side.
    """
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    # Add border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return img, ratio, (dw, dh)


class FlatImageDataset(torch.utils.data.Dataset):
    """Flat-folder image dataset for calibration / fine-tuning.

    Two loading modes:
      - ``letterbox_shape=None`` (default): PIL load; resizing is expected
        to be performed by the supplied ``transform`` (classification path).
      - ``letterbox_shape=(H, W)``: OpenCV load + letterbox resize for
        detection-style preprocessing (YOLO path). Mirrors the on-board
        preprocessing so the activation ranges seen by the quantizer match
        those produced at deployment.

    The label is always 0 since this dataset is used for calibration-style
    workflows where targets are unused.
    """

    def __init__(self, root_dir, transform=None, letterbox_shape=None):
        self.root_dir = root_dir
        self.transform = transform
        self.letterbox_shape = letterbox_shape  # (H, W) or None
        self.image_files = [f for f in os.listdir(root_dir)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        if self.letterbox_shape is not None:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img, _, _ = letterbox(img, new_shape=self.letterbox_shape)
        else:
            img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, 0


class YoloDataset(torch.utils.data.Dataset):
    """
    Dataset to load images and labels from a standard YOLO folder structure.
    Expects labels as space-separated standard .txt files containing:
    <class_id> <cx> <cy> <w> <h>
    """
    def __init__(self, images_dir, labels_dir, input_shape=(640, 640), normalization=None, augment=False, indices=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.input_shape = input_shape
        self.normalization = normalization or {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]}
        self.augment = augment

        # Scan for valid image files
        valid_exts = ('.png', '.jpg', '.jpeg', '.webp')
        all_files = sorted(os.listdir(images_dir)) if os.path.exists(images_dir) else []
        self.image_files = [f for f in all_files if f.lower().endswith(valid_exts)]

        if indices is not None:
            self.image_files = [self.image_files[i] for i in indices if i < len(self.image_files)]
            print(f"[INFO] Initialized YoloDataset with subset of {len(self.image_files)} images.")
        else:
            print(f"[INFO] Initialized YoloDataset with {len(self.image_files)} images.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        img_path = os.path.join(self.images_dir, filename)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h_orig, w_orig = img.shape[:2]
        target_h, target_w = self.input_shape

        # 1. Letterbox resizing to target_shape
        letterboxed, ratio, (dw, dh) = letterbox(img, new_shape=self.input_shape)
        r = ratio[0]  # scale ratio

        # 2. Parse labels from matching .txt file
        boxes = []
        classes = []
        
        stem, _ = os.path.splitext(filename)
        label_path = os.path.join(self.labels_dir, f"{stem}.txt")
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    tokens = line.split()
                    if len(tokens) < 5:
                        continue
                    class_id = int(tokens[0])
                    cx_orig, cy_orig, w_orig_norm, h_orig_norm = map(float, tokens[1:5])

                    # Shift and scale coordinates to match the letterbox pad/scale
                    # x_norm_target = (x_norm_orig * W_orig * r + dw) / W_target
                    cx_target = (cx_orig * w_orig * r + dw) / target_w
                    cy_target = (cy_orig * h_orig * r + dh) / target_h
                    w_target = (w_orig_norm * w_orig * r) / target_w
                    h_target = (h_orig_norm * h_orig * r) / target_h

                    # Clip to bounds [0, 1]
                    cx_target = max(0.0, min(1.0, cx_target))
                    cy_target = max(0.0, min(1.0, cy_target))
                    w_target = max(0.0, min(1.0, w_target))
                    h_target = max(0.0, min(1.0, h_target))

                    boxes.append([cx_target, cy_target, w_target, h_target])
                    classes.append(class_id)

        # 3. Apply augmentations if requested
        if self.augment and len(boxes) > 0:
            letterboxed, boxes = self._apply_augmentations(letterboxed, boxes)

        # 4. Standard Normalize and ToTensor
        img_tensor = torch.from_numpy(letterboxed).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor(self.normalization['mean']).view(3, 1, 1)
        std = torch.tensor(self.normalization['std']).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        # Wrap targets
        bboxes_tensor = torch.tensor(boxes, dtype=torch.float32) if len(boxes) > 0 else torch.zeros((0, 4), dtype=torch.float32)
        classes_tensor = torch.tensor(classes, dtype=torch.int64) if len(classes) > 0 else torch.zeros((0,), dtype=torch.int64)

        target = {
            "bboxes": bboxes_tensor,
            "cls": classes_tensor
        }

        return img_tensor, target

    def _apply_augmentations(self, img, boxes):
        boxes = np.array(boxes)
        # Horizontal Flip (50% probability)
        if random.random() < 0.5:
            img = cv2.flip(img, 1)
            boxes[:, 0] = 1.0 - boxes[:, 0]

        # HSV Jitter (50% probability)
        if random.random() < 0.5:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
            h_gain = random.uniform(0.9, 1.1)
            hsv[:, :, 0] = np.clip(hsv[:, :, 0] * h_gain, 0, 179)
            s_gain = random.uniform(0.7, 1.3)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s_gain, 0, 255)
            v_gain = random.uniform(0.7, 1.3)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * v_gain, 0, 255)
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        return img, list(boxes)


def yolo_collate_fn(batch):
    """
    Collate function to format batches according to Ultralytics YOLOv8/v26 loss input requirements.
    Stacks images into [B, 3, H, W] and returns target bounding boxes concatenated
    along a single tensor with an added column for the sample index.
    """
    images = []
    batch_bboxes = []
    batch_classes = []
    batch_indices = []

    for i, (img, target) in enumerate(batch):
        images.append(img)
        num_boxes = target['bboxes'].shape[0]
        if num_boxes > 0:
            batch_bboxes.append(target['bboxes'])
            batch_classes.append(target['cls'])
            batch_indices.append(torch.full((num_boxes,), i, dtype=torch.float32))

    images_tensor = torch.stack(images, dim=0)

    if len(batch_bboxes) > 0:
        bboxes_tensor = torch.cat(batch_bboxes, dim=0)
        classes_tensor = torch.cat(batch_classes, dim=0)
        batch_idx_tensor = torch.cat(batch_indices, dim=0)
    else:
        bboxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
        classes_tensor = torch.zeros((0,), dtype=torch.int64)
        batch_idx_tensor = torch.zeros((0,), dtype=torch.float32)

    targets_dict = {
        "bboxes": bboxes_tensor,
        "cls": classes_tensor,
        "batch_idx": batch_idx_tensor
    }

    # Provide dummy masks to satisfy v8SegmentationLoss during VAIQ sensitivity analysis.
    # Note: If fine-tuning a segmentation model is needed later, YoloDataset must be updated
    # to load real segmentation masks from COCO polygons instead of using these zeros!
    if bboxes_tensor.shape[0] > 0:
        H, W = images_tensor.shape[2], images_tensor.shape[3]
        targets_dict["masks"] = torch.zeros((bboxes_tensor.shape[0], H, W), dtype=torch.float32)
    else:
        targets_dict["masks"] = torch.zeros((0, images_tensor.shape[2], images_tensor.shape[3]), dtype=torch.float32)

    return images_tensor, targets_dict


def build_or_load_subset_indices(split, n, seed=42, cache_dir="data/coco/.subsets", total_count=1000):
    """
    Loads deterministic subset indices from a text file, or generates them on-the-fly,
    saves them to cache_dir, and returns them.
    """
    os.makedirs(cache_dir, exist_ok=True)
    filename = f"{split}_{n}_seed{seed}.txt"
    cache_path = os.path.join(cache_dir, filename)

    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            indices = [int(line.strip()) for line in f if line.strip()]
        return indices

    print(f"[INFO] Cache not found at {cache_path}. Generating deterministic subset...")
    rng = random.Random(seed)
    all_indices = list(range(total_count))
    rng.shuffle(all_indices)
    
    indices = all_indices[:n]
    
    with open(cache_path, 'w') as f:
        for idx in indices:
            f.write(f"{idx}\n")
    print(f"[INFO] Wrote subset indices to: {cache_path}")
    
    return indices
