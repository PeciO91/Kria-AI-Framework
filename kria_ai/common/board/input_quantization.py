_CV2_SIGNED_LUT_AVAILABLE = None


def normalization_constants(mean, std, fixed_point):
    import numpy as np

    mean_values = np.asarray(mean, dtype=np.float32)
    std_values = np.asarray(std, dtype=np.float32)
    if mean_values.shape != std_values.shape or np.any(std_values <= 0):
        raise ValueError("Mean and standard deviation must have matching positive channel values")
    fixed_scale = np.float32(2 ** fixed_point)
    return fixed_scale / (255.0 * std_values), mean_values * fixed_scale / std_values


def build_normalization_lut(mean, std, fixed_point):
    import numpy as np

    scale, shift = normalization_constants(mean, std, fixed_point)
    values = np.arange(256, dtype=np.float32)[:, None]
    table = np.clip(np.rint(values * scale - shift), -128, 127).astype(np.int8)
    if all(np.array_equal(table[:, 0], table[:, index]) for index in range(1, table.shape[1])):
        return np.ascontiguousarray(table[:, 0])
    return table


def apply_normalization_lut(image_uint8, lut):
    import numpy as np

    global _CV2_SIGNED_LUT_AVAILABLE
    if _CV2_SIGNED_LUT_AVAILABLE is not False:
        try:
            import cv2

            shaped_lut = lut if lut.ndim == 1 else lut.reshape(256, 1, lut.shape[1])
            output = cv2.LUT(image_uint8, shaped_lut)
            if output.dtype == np.int8:
                _CV2_SIGNED_LUT_AVAILABLE = True
                return output
            _CV2_SIGNED_LUT_AVAILABLE = False
        except Exception:
            _CV2_SIGNED_LUT_AVAILABLE = False
    if lut.ndim == 1:
        return np.take(lut, image_uint8)
    if lut.shape[1] != image_uint8.shape[-1]:
        raise ValueError(
            f"LUT has {lut.shape[1]} channels but image has {image_uint8.shape[-1]}"
        )
    return np.stack(
        [np.take(lut[:, channel], image_uint8[..., channel]) for channel in range(lut.shape[1])],
        axis=-1,
    )
