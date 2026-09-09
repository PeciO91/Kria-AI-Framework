# YOLOv26 Dataset Preparation

YOLOv26 detection and instance segmentation own separate label parsers even when they share COCO image directories.

## Detection labels

Detection files use exactly five columns:

```text
<class_id> <x_center> <y_center> <width> <height>
```

Coordinates are normalized to `[0, 1]`. The expected layout is:

```text
data/<dataset>/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── .subsets/
```

An image with no objects may omit its TXT file. Polygon rows are not accepted by the detection parser and are not reinterpreted as boxes.

## Instance-segmentation labels

YOLO polygon rows use:

```text
<class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
```

Each point is normalized to `[0, 1]`. At least three points are required. The segmentation loader can also read standard COCO polygon annotations from:

```text
data/coco2017/annotations/instances_train2017.json
data/coco2017/annotations/instances_val2017.json
```

RLE-only annotations are not handled by the current loader.

## Registry fields

Detection datasets are registered in `kria_ai/yolov26/detection/config.py`. Segmentation datasets are registered separately in `kria_ai/yolov26/segmentation/config.py` so label semantics cannot be mixed accidentally.

Each registry specifies:

- training and validation image roots
- task-specific label roots
- deterministic subset cache directory
- board image/label roots
- ordered COCO-80 class names
- input normalization

## Calibration

Calibration uses the image-only loader and does not parse labels or masks:

```bash
python -m kria_ai yolov26 detection quantize \
  --model yolov26s --dataset coco --mode calib --subset-len 100

python -m kria_ai yolov26 segmentation quantize \
  --model yolov26n_seg --dataset coco --mode calib --subset-len 100
```

Subset indices are generated deterministically from the actual number of scanned images rather than a fixed assumed dataset size.

## Consistency requirements

- The configured model head and dataset class mapping must both be COCO-80.
- Host calibration and board inference use the same YOLO letterbox implementation in `kria_ai/yolov26/preprocess.py`.
- Detection boxes and segmentation masks are transformed with the same scale and padding as their image.
- Do not edit generated subset, quantizer, compiler, or XMODEL artifacts manually.
