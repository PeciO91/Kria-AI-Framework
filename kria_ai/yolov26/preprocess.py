def letterbox(image, new_shape=(640, 640), color=(114, 114, 114)):
    """Resize an RGB image with the pipeline's exact YOLO letterbox rules."""
    import cv2

    shape = image.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    ratio_value = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    ratio = ratio_value, ratio_value
    new_unpad = (
        int(round(shape[1] * ratio_value)),
        int(round(shape[0] * ratio_value)),
    )
    pad_width = new_shape[1] - new_unpad[0]
    pad_height = new_shape[0] - new_unpad[1]
    pad_width /= 2
    pad_height /= 2

    if shape[::-1] != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    top = int(round(pad_height - 0.1))
    bottom = int(round(pad_height + 0.1))
    left = int(round(pad_width - 0.1))
    right = int(round(pad_width + 0.1))
    image = cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=color,
    )
    return image, ratio, (left, top)


def image_to_float_tensor(image_rgb, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)):
    """Convert an HWC uint8 RGB image to a normalized CHW float tensor."""
    import torch

    image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
    mean_tensor = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    std_tensor = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
    return (image_tensor - mean_tensor) / std_tensor


def preprocess_host_image(
    image_rgb,
    input_shape,
    mean=(0.0, 0.0, 0.0),
    std=(1.0, 1.0, 1.0),
):
    """Letterbox an RGB image and return the host-side float input tensor."""
    resized, _, _ = letterbox(image_rgb, new_shape=input_shape)
    return image_to_float_tensor(resized, mean=mean, std=std)


def preprocess_board_image(image_rgb, input_shape, normalization_lut):
    """Letterbox and quantize one RGB image into a batched NHWC INT8 input."""
    import numpy as np

    from kria_ai.common.board.input_quantization import apply_normalization_lut

    resized, _, _ = letterbox(image_rgb, new_shape=input_shape)
    quantized = apply_normalization_lut(resized, normalization_lut)
    return np.expand_dims(quantized, axis=0)
