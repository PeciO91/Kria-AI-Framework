from kria_ai.classification.config import MODELS as CLASSIFICATION_MODELS
from kria_ai.yolov26.detection.config import MODELS as DETECTION_MODELS
from kria_ai.yolov26.segmentation.config import MODELS as SEGMENTATION_MODELS


ACTIVE_MODEL_ID = "resnet18"


def _classification(config):
    return {
        "source": "torchvision",
        "type": "classification",
        "name": config.name,
        "model_path": str(config.checkpoint_path),
        "model_class": config.constructor,
        "input_shape": config.input_size,
        "num_classes": config.num_classes,
        "last_layer_name": config.head_attribute,
    }


def _detection(config):
    return {
        "source": "custom",
        "loader": "ultralytics",
        "type": "detection",
        "name": config.name,
        "model_path": str(config.checkpoint_path),
        "repo_path": str(config.repository_path),
        "yaml_path": str(config.architecture_path),
        "input_shape": config.input_size,
        "num_classes": config.num_classes,
        "reg_max": config.reg_max,
        "max_det": config.max_detections,
        "strides": list(config.strides),
        "conf_threshold": config.confidence_threshold,
        "decoder": "ultralytics_anchor_free",
        "prune_excludes": list(config.prune_excludes),
    }


def _segmentation(config):
    values = _detection(config)
    values.update({
        "type": "segmentation",
        "num_masks": config.num_masks,
        "num_protos": config.num_masks,
        "prototype_channels": config.prototype_channels,
        "mask_threshold": config.mask_threshold,
    })
    return values


MODELS = {
    **{model_id: _classification(config) for model_id, config in CLASSIFICATION_MODELS.items()},
    **{model_id: _detection(config) for model_id, config in DETECTION_MODELS.items()},
    **{model_id: _segmentation(config) for model_id, config in SEGMENTATION_MODELS.items()},
}


def get_active_model(model_id=None):
    resolved_id = model_id or ACTIVE_MODEL_ID
    try:
        return MODELS[resolved_id]
    except KeyError as error:
        raise ValueError(f"Unknown model {resolved_id!r}; available: {', '.join(MODELS)}") from error
