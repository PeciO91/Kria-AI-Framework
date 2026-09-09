from types import MethodType

from kria_ai.yolov26.detection.loss import (
    forward_for_loss,
    prepare_loss_model as prepare_detection_loss_model,
    reduce_loss,
)


def prepare_loss_model(model):
    prepare_detection_loss_model(model)
    proto = model.model[-1].proto
    if not hasattr(proto, "_kria_original_forward"):
        original_forward = proto.forward

        def forward_without_semantic_branch(self, features, return_semseg=False):
            return original_forward(features, return_semseg=False)

        proto._kria_original_forward = original_forward
        proto.forward = MethodType(forward_without_semantic_branch, proto)
    return model


def create_loss(model):
    from ultralytics.utils.loss import E2ELoss, v8SegmentationLoss

    prepare_loss_model(model)
    return E2ELoss(model, loss_fn=v8SegmentationLoss)


__all__ = ["create_loss", "forward_for_loss", "prepare_loss_model", "reduce_loss"]
