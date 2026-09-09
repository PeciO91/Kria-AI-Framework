# YOLOv26 Object Detection

The supported detection path is YOLOv26 COCO-80. YOLOv5 is outside the current refactor scope. Implementation lives under `kria_ai/yolov26/`; this directory contains temporary compatibility wrappers.

```bash
python -m kria_ai yolov26 detection inspect --model yolov26s
python -m kria_ai yolov26 detection quantize --model yolov26s --dataset coco --mode calib
python -m kria_ai yolov26 detection quantize --model yolov26s --dataset coco --mode test
python -m kria_ai yolov26 detection compile --model yolov26s
python3 -m kria_ai yolov26 detection benchmark --model yolov26s --dataset coco --threads 2 --profile
```

The DPU graph exports six split one2one tensors. Anchor-free decode and top-k selection run on the ARM CPU without NMS.
