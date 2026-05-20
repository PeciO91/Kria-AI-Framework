"""
Shared dataset utilities for the Vitis AI deployment pipeline.

Used by the quantizer (calibration) and the optimizer (fine-tuning) stages
so that host-side preprocessing stays consistent between them and matches
on-board preprocessing for detection models.
"""
import os

import cv2
from PIL import Image
import torch

from detection_utils import letterbox


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
