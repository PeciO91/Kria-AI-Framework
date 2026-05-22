# YOLO Dataset Preparation Guide

This guide describes how to format any custom object detection dataset (or the standard COCO 2017 dataset) into the standardized, lightweight YOLO TXT format expected by the Vitis AI Universal Deployment Pipeline.

Using this format ensures your dataset loader remains fast, completely generic, and highly portable.

---

## 1. Expected Directory Structure

Your dataset must live under the `data/` directory (e.g. `data/coco/` or `data/custom_defect_detection/`) and follow this exact layout:

```text
data/
└── <dataset_id>/                 # e.g., "coco" or "my_custom_dataset"
    ├── images/
    │   ├── train/                # Training images (.jpg, .png, etc.)
    │   └── val/                  # Validation images
    ├── labels/
    │   ├── train/                # Training annotation files (.txt)
    │   └── val/                  # Validation annotation files
    └── .subsets/                 # Automatically created by the pipeline for caching subsets
```

For every image in `images/train/` (e.g., `000001.jpg`), there must be a matching annotation file of the same name in `labels/train/` (e.g., `000001.txt`). Images with no objects (background samples) can simply omit the corresponding `.txt` file.

---

## 2. Label File Format

Each `.txt` label file contains one line per target object on the image. The format for each line is:

```text
<class_id> <x_center> <y_center> <width> <height>
```

- **`class_id`**: Zero-indexed integer class ID (e.g., `0` for person, `1` for car). Max value must be less than `num_classes`.
- **`x_center`**: Horizontal center coordinate of the bounding box, normalized relative to the image width. Values must be in range `[0.0, 1.0]`.
- **`y_center`**: Vertical center coordinate of the bounding box, normalized relative to the image height. Values must be in range `[0.0, 1.0]`.
- **`width`**: Normalized bounding box width. Values must be in range `[0.0, 1.0]`.
- **`height`**: Normalized bounding box height. Values must be in range `[0.0, 1.0]`.

### Example (`labels/train/000001.txt`):
```text
0 0.3015 0.4120 0.1560 0.3220
1 0.7230 0.6510 0.2800 0.4100
```
This denotes two objects:
- Object 1: Class `0` (e.g., "person"), centered at 30.15% width, 41.2% height, with a width of 15.6% and height of 32.2%.
- Object 2: Class `1` (e.g., "car"), centered at 72.3% width, 65.1% height, with a width of 28.0% and height of 41.0%.

---

## 3. How to Convert COCO JSON to YOLO TXT

If you have annotations in the official COCO JSON format, you can easily convert them. Here is a helper Python script you can save as `scripts/coco_to_yolo.py` and run:

```python
import os
import json
import shutil
from tqdm import tqdm

def convert_coco_json(json_path, images_src_dir, output_root, split="train"):
    """
    Converts a COCO JSON file to YOLO TXT directory structure.
    """
    # Create target folders
    target_images_dir = os.path.join(output_root, "images", split)
    target_labels_dir = os.path.join(output_root, "labels", split)
    os.makedirs(target_images_dir, exist_ok=True)
    os.makedirs(target_labels_dir, exist_ok=True)

    print(f"Loading COCO annotations from {json_path}...")
    with open(json_path, "r") as f:
        coco_data = json.load(f)

    # COCO non-contiguous IDs to YOLO 0-79 mapping
    coco_id_to_yolo_id = {
        1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9,
        11: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17,
        20: 18, 21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25,
        31: 26, 32: 27, 33: 28, 34: 29, 35: 30, 36: 31, 37: 32, 38: 33,
        39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39, 46: 40, 47: 41,
        48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49,
        56: 50, 57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57,
        64: 58, 65: 59, 67: 60, 70: 61, 72: 62, 73: 63, 74: 64, 75: 65,
        76: 66, 77: 67, 78: 68, 79: 69, 80: 70, 81: 71, 82: 72, 84: 73,
        85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79
    }

    # Index images by ID
    images = {img['id']: img for img in coco_data['images']}

    # Index annotations by image ID
    annotations_by_img = {}
    for ann in coco_data['annotations']:
        if ann.get('iscrowd', 0) == 1:
            continue
        cat_id = ann['category_id']
        if cat_id not in coco_id_to_yolo_id:
            continue
        
        img_id = ann['image_id']
        if img_id not in annotations_by_img:
            annotations_by_img[img_id] = []
        annotations_by_img[img_id].append(ann)

    print(f"Converting annotations and copying files for split: {split}...")
    for img_id, anns in tqdm(annotations_by_img.items()):
        if img_id not in images:
            continue
        
        img_info = images[img_id]
        filename = img_info['file_name']
        img_w = img_info['width']
        img_h = img_info['height']

        src_img_path = os.path.join(images_src_dir, filename)
        if not os.path.exists(src_img_path):
            continue

        # Copy image file to new target folder
        dest_img_path = os.path.join(target_images_dir, filename)
        shutil.copy(src_img_path, dest_img_path)

        # Write matching .txt file
        stem, _ = os.path.splitext(filename)
        dest_txt_path = os.path.join(target_labels_dir, f"{stem}.txt")

        with open(dest_txt_path, "w") as f:
            for ann in anns:
                cat_id = ann['category_id']
                yolo_class = coco_id_to_yolo_id[cat_id]

                # COCO box is [x_min, y_min, width, height] (pixels)
                x_min, y_min, bw, bh = ann['bbox']

                # Convert to normalized YOLO [cx, cy, w, h] format
                cx = (x_min + bw / 2.0) / img_w
                cy = (y_min + bh / 2.0) / img_h
                norm_w = bw / img_w
                norm_h = bh / img_h

                f.write(f"{yolo_class} {cx:.6f} {cy:.6f} {norm_w:.6f} {norm_h:.6f}\n")

    print(f"Successfully created standard YOLO TXT structure for split '{split}' in: {output_root}")

if __name__ == "__main__":
    # Example usage:
    convert_coco_json(
        json_path="data/coco2017/annotations/instances_val2017.json",
        images_src_dir="data/coco2017/val2017",
        output_root="data/coco",
        split="val"
    )
```

---

## 4. Registering a Custom Dataset

Once your dataset has been structured, register it in `dataset_config.py` under the `DATASETS` registry so that the deployment pipeline knows its exact paths:

```python
    "my_custom_dataset": {
        "name": "My Custom Industrial Defect Detection",
        "folder_name": "my_custom_dataset",
        "images_train": "data/my_custom_dataset/images/train",
        "images_val": "data/my_custom_dataset/images/val",
        "labels_train": "data/my_custom_dataset/labels/train",
        "labels_val": "data/my_custom_dataset/labels/val",
        "subset_cache_dir": "data/my_custom_dataset/.subsets",
        "classes": [
            "dent", "scratch", "corrosion"
        ],
        "normalization": {
            "mean": [0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0]
        }
    }
```

Now you can invoke the pipeline commands by passing your custom dataset ID:
```bash
python3 scripts/common/run_quantizer.py --model yolov26s --dataset my_custom_dataset --quant_mode calib
```
