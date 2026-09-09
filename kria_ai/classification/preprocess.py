def build_transform(model_config, dataset_config):
    import torchvision.transforms as transforms

    return transforms.Compose([
        transforms.Resize(model_config.input_size),
        transforms.ToTensor(),
        transforms.Normalize(dataset_config.mean, dataset_config.std),
    ])


def preprocess_board_image(image_rgb, input_shape, normalization_lut):
    import cv2
    import numpy as np

    from kria_ai.common.board.input_quantization import apply_normalization_lut

    height, width = input_shape
    resized = cv2.resize(image_rgb, (width, height))
    quantized = apply_normalization_lut(resized, normalization_lut)
    return np.expand_dims(quantized, axis=0)
