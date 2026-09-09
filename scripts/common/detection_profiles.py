from kria_ai.yolov26.detection.loss import create_loss as create_detection_loss
from kria_ai.yolov26.pruning import resolve_prune_excludes
from kria_ai.yolov26.segmentation.loss import create_loss as create_segmentation_loss


def get_profile(*args, **kwargs):
    raise RuntimeError(
        "DetectionProfile was removed; use the explicit kria_ai.yolov26 detection or segmentation modules"
    )


__all__ = [
    "create_detection_loss",
    "create_segmentation_loss",
    "get_profile",
    "resolve_prune_excludes",
]
