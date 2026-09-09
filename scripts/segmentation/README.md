# YOLOv26 Instance Segmentation

The supported segmentation path is YOLOv26 COCO-80 instance segmentation. Semantic segmentation is outside the current architecture. Implementation lives under `kria_ai/yolov26/segmentation/`; this directory contains temporary compatibility wrappers.

```bash
python -m kria_ai yolov26 segmentation inspect --model yolov26n_seg
python -m kria_ai yolov26 segmentation quantize --model yolov26n_seg --dataset coco --mode calib
python -m kria_ai yolov26 segmentation quantize --model yolov26n_seg --dataset coco --mode test
python -m kria_ai yolov26 segmentation compile --model yolov26n_seg
python3 -m kria_ai yolov26 segmentation benchmark --model yolov26n_seg --dataset coco --threads 2 --profile
```

The DPU graph exports box, class, and mask-coefficient tensors for three levels plus prototypes. CPU postprocessing assembles masks. Use `--accuracy --labels-dir ...` for the current mask mAP@0.5 path or `--video ... --threads 1 --producers 1` for ordered file-video processing.
