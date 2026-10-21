# Dataset Preparation

Datasets are registered per task in typed configs, and each task owns its label parser. This guide lists the actual registered paths and label semantics; a generic `data/<dataset>/{images,labels}` layout is not authoritative.

## Registered Paths

| Task | Dataset key | Calibration/train images | Validation images | Labels | Classes |
|---|---|---|---|---|---|
| Classification | `intel_images` | `data/intel_images/calibration_data` | `data/intel_images/calibration_data` | directory-per-class | 6: `buildings, forest, glacier, mountain, sea, street` |
| Detection | `openimagesv7` | `data/openimagesv7/train/images` | `data/openimagesv7/validation/images` | `data/openimagesv7/{train,validation}/labels` | 2: `0: Person`, `1: Human face` |
| Detection | `coco` | see caveat below | see caveat below | `data/coco2017/labels/{train2017,val2017}` | COCO-80 |
| Segmentation | `coco` | `data/coco2017/train2017` | `data/coco2017/val2017` | `data/coco2017/labels/{train2017,val2017}` or COCO polygon JSON | COCO-80 |

Registries: `kria_ai/classification/config.py`, `kria_ai/yolov26/detection/config.py`, `kria_ai/yolov26/segmentation/config.py` (detection and segmentation are registered separately so box and polygon label semantics cannot be mixed accidentally). The YOLO registries carry deterministic subset caches (`data/<dataset>/.subsets/...`), task-specific label roots, and board image/label roots; the classification registry instead carries calibration, evaluation, and board roots with class-folder organization. Note that the classification calibration and evaluation roots both point at `data/intel_images/calibration_data` — it is not an independently held-out evaluation split. All registries carry ordered class names and input normalization.

## Detection Labels (YOLO Boxes)

Exactly five columns, normalized to `[0, 1]`:

```text
<class_id> <x_center> <y_center> <width> <height>
```

Expected OpenImagesV7 layout:

```text
data/openimagesv7/
├── train/{images,labels}/
├── validation/{images,labels}/
└── .subsets/detection/
```

An image with no objects may omit its TXT file. Polygon rows are not accepted by the detection parser and are not reinterpreted as boxes. Class IDs must match the registered order — `0: Person`, `1: Human face` for `openimagesv7`; COCO-80 order for `coco`.

## Instance-Segmentation Labels (Polygons)

YOLO polygon rows, all coordinates normalized to `[0, 1]`, at least three points:

```text
<class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
```

The loader can also read standard COCO polygon annotations from `data/coco2017/annotations/instances_{train,val}2017.json`. RLE-only annotations are not handled.

## Calibration And Subsets

For YOLO tasks, calibration uses the image-only loader and does not parse labels or masks; classification calibration iterates the class-folder dataset (labels come from the folder structure):

```bash
python -m kria_ai yolov26 detection quantize --model yolov26n_FaceHuman --dataset openimagesv7 --mode calib --subset-len 100
python -m kria_ai yolov26 detection quantize --model yolov26s --dataset coco --mode calib --subset-len 100
python -m kria_ai yolov26 segmentation quantize --model yolov26n_seg --dataset coco --mode calib --subset-len 100
python -m kria_ai classification quantize --model resnet18 --dataset intel_images --mode calib --subset-len 100
```

Subset indices are generated deterministically from the actual number of scanned images rather than a fixed assumed dataset size.

## Board Layout

Board dataset roots are configured separately (`datasets/...` on the KV260) and must be assembled manually. For detection and segmentation, the benchmark `--dataset-root` is **one flat directory** of images — it is not scanned recursively — so pass the `images/test`-style path explicitly, and `--labels-dir` the matching `labels/test`-style path. Classification keeps its class-folder structure on the board (`datasets/intel_images/train_data`). The recorded detector experiment used `datasets/openimagesv7/fpga_test_images/{images/test,labels/test}`; segmentation uses `datasets/coco2017` with labels `datasets/coco2017/labels/val2017`.

## Known Caveats

- **COCO detection image roots**: the `coco` detection registry currently points train/validation *images* at `data/openimagesv7/train2017` and `data/openimagesv7/val2017` while *labels* point at `data/coco2017/labels/...`. Those image roots do not match a real COCO layout and are absent locally. Check and correct the family dataset metadata in `kria_ai/yolov26/detection/config.py` before deploying a real COCO detection model; documented COCO commands are parser examples, not a verified end-to-end recipe.
- **Head/dataset class compatibility**: the configured model head and the dataset class mapping must agree — COCO-80 models with COCO-80 datasets, the two-class FaceHuman head with `openimagesv7`. Mismatched checkpoints are rejected at load time.
- Host calibration and board inference share the same YOLO letterbox in `kria_ai/yolov26/preprocess.py`; boxes and masks are transformed with the same scale and padding as their image.
- Do not edit generated subset, quantizer, compiler, or XMODEL artifacts manually.
