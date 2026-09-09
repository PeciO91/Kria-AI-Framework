from types import SimpleNamespace


def prepare_loss_model(model):
    defaults = {
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5,
        "reg_max": 16,
        "overlap_mask": False,
    }
    current = getattr(model, "args", None)
    if isinstance(current, dict):
        defaults.update(current)
        model.args = SimpleNamespace(**defaults)
    elif current is None:
        model.args = SimpleNamespace(**defaults)
    else:
        for name, value in defaults.items():
            if not hasattr(current, name):
                setattr(current, name, value)
    model.model[-1].end2end = True
    return model


def create_loss(model):
    from ultralytics.utils.loss import E2ELoss, v8DetectionLoss

    prepare_loss_model(model)
    return E2ELoss(model, loss_fn=v8DetectionLoss)


def forward_for_loss(model, images):
    was_training = model.training
    model.train()
    try:
        return model(images)
    finally:
        if not was_training:
            model.eval()


def reduce_loss(loss):
    if isinstance(loss, tuple) and len(loss) >= 2:
        value = loss[0]
    elif isinstance(loss, dict):
        value = sum(loss.values())
    elif isinstance(loss, (list, tuple)):
        value = sum(loss)
    else:
        value = loss
    return value.sum() if hasattr(value, "dim") and value.dim() > 0 else value
