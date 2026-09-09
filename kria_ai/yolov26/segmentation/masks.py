"""NumPy/OpenCV instance-mask assembly and image-space scaling helpers."""

import numpy as np


__all__ = [
    "crop_mask",
    "extract_mask_prototypes",
    "gather_mask_coefficients",
    "process_mask",
    "resolve_segmentation_outputs",
    "scale_image_masks",
]


def gather_mask_coefficients(
    output_data,
    dequant_scales,
    mask_order,
    keep_indices,
    num_masks,
):
    """Gather and dequantize mask coefficients selected by detection decoding.

    ``keep_indices`` contains ``(feature_level, flattened_position)`` pairs.
    Mask heads are looked up through ``mask_order`` because VART output tensor
    order is not guaranteed to match feature-level order.
    """
    mask_coefficients = np.empty((len(keep_indices), num_masks), dtype=np.float32)
    for index, (level_index, flat_index) in enumerate(keep_indices):
        mask_index = mask_order[level_index]
        mask_int8 = output_data[mask_index].reshape(-1, num_masks)[flat_index]
        mask_coefficients[index] = (
            mask_int8.astype(np.float32) * dequant_scales[mask_index]
        )
    return mask_coefficients


def extract_mask_prototypes(output_data, dequant_scales, prototype_index):
    """Dequantize an NHWC prototype output and return it in CHW order."""
    prototype_tensor = (
        output_data[prototype_index][0].astype(np.float32)
        * dequant_scales[prototype_index]
    )
    return prototype_tensor.transpose(2, 0, 1)


def crop_mask(masks, boxes):
    """Zero mask values outside their corresponding ``xyxy`` boxes.

    Args:
        masks: Array shaped ``(N, H, W)``.
        boxes: Array shaped ``(N, 4)`` in mask coordinates. The lower bounds
            are inclusive and the upper bounds are exclusive.
    """
    _, height, width = masks.shape
    x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, axis=1)
    columns = np.arange(width, dtype=x1.dtype)[None, None, :]
    rows = np.arange(height, dtype=x1.dtype)[None, :, None]
    return masks * (
        (columns >= x1) * (columns < x2) * (rows >= y1) * (rows < y2)
    )


def process_mask(protos, masks_in, bboxes, shape, upsample=False):
    """Assemble sigmoid masks from coefficients and prototype channels.

    Prototype regions are cropped before the coefficient matrix product to
    preserve the optimized board-side behavior. ``bboxes`` are ``xyxy`` boxes
    in the padded input space described by ``shape``.
    """
    channels, mask_height, mask_width = protos.shape
    input_height, input_width = shape
    count = len(bboxes)

    if count == 0:
        output_shape = (
            (0, input_height, input_width)
            if upsample
            else (0, mask_height, mask_width)
        )
        return np.empty(output_shape, dtype=np.float32)

    downsampled_boxes = bboxes.copy()
    downsampled_boxes[:, 0] *= mask_width / input_width
    downsampled_boxes[:, 1] *= mask_height / input_height
    downsampled_boxes[:, 2] *= mask_width / input_width
    downsampled_boxes[:, 3] *= mask_height / input_height

    masks = np.zeros((count, mask_height, mask_width), dtype=np.float32)
    for index in range(count):
        box_x1, box_y1, box_x2, box_y2 = downsampled_boxes[index]
        pixel_x1 = max(0, int(np.floor(box_x1)))
        pixel_y1 = max(0, int(np.floor(box_y1)))
        pixel_x2 = min(mask_width, int(np.ceil(box_x2)))
        pixel_y2 = min(mask_height, int(np.ceil(box_y2)))

        if pixel_x2 <= pixel_x1 or pixel_y2 <= pixel_y1:
            continue

        prototype_crop = np.ascontiguousarray(
            protos[:, pixel_y1:pixel_y2, pixel_x1:pixel_x2]
        ).reshape(channels, -1)
        mask_logits = masks_in[index] @ prototype_crop
        mask_crop = 1.0 / (1.0 + np.exp(-mask_logits))
        mask_crop = mask_crop.reshape(pixel_y2 - pixel_y1, pixel_x2 - pixel_x1)

        columns = np.arange(pixel_x1, pixel_x2, dtype=np.float32)
        rows = np.arange(pixel_y1, pixel_y2, dtype=np.float32)
        mask_crop = mask_crop * (
            (columns[None, :] >= box_x1)
            * (columns[None, :] < box_x2)
            * (rows[:, None] >= box_y1)
            * (rows[:, None] < box_y2)
        )
        masks[index, pixel_y1:pixel_y2, pixel_x1:pixel_x2] = mask_crop

    if upsample:
        try:
            import cv2
        except ImportError as error:
            raise RuntimeError("process_mask with upsample=True requires OpenCV (cv2)") from error
        masks = cv2.resize(
            masks.transpose(1, 2, 0),
            (input_width, input_height),
            interpolation=cv2.INTER_LINEAR,
        )
        if masks.ndim == 2:
            masks = masks[:, :, None]
        masks = masks.transpose(2, 0, 1)

    return masks


def scale_image_masks(masks, bboxes, img1_shape, img0_shape):
    """Map letterboxed masks directly into the original image space.

    ``masks`` may be at padded-input resolution or at a lower prototype
    resolution. ``bboxes`` remain in padded-input coordinates; the source mask
    scale is folded into the inverse affine transform to avoid an intermediate
    full-input resize.
    """
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("scale_image_masks requires OpenCV (cv2)") from error
    count = masks.shape[0]
    if count == 0:
        return np.empty(
            (0, img0_shape[0], img0_shape[1]),
            dtype=masks.dtype,
        )

    source_height, source_width = masks.shape[1], masks.shape[2]
    source_x_scale = source_width / img1_shape[1]
    source_y_scale = source_height / img1_shape[0]

    gain = min(
        img1_shape[0] / img0_shape[0],
        img1_shape[1] / img0_shape[1],
    )
    pad_width = (img1_shape[1] - img0_shape[1] * gain) / 2
    pad_height = (img1_shape[0] - img0_shape[0] * gain) / 2

    top = int(round(pad_height - 0.1))
    left = int(round(pad_width - 0.1))
    bottom = int(round(img1_shape[0] - pad_height + 0.1))
    right = int(round(img1_shape[1] - pad_width + 0.1))
    crop_height = bottom - top
    crop_width = right - left

    scale_y = crop_height / img0_shape[0]
    scale_x = crop_width / img0_shape[1]

    margin = 2
    scaled_masks = np.zeros(
        (count, img0_shape[0], img0_shape[1]),
        dtype=masks.dtype,
    )

    for index in range(count):
        x1, y1, x2, y2 = bboxes[index]
        destination_x1 = (x1 - pad_width) / gain - margin
        destination_y1 = (y1 - pad_height) / gain - margin
        destination_x2 = (x2 - pad_width) / gain + margin
        destination_y2 = (y2 - pad_height) / gain + margin

        output_x1 = max(0, int(np.floor(destination_x1)))
        output_y1 = max(0, int(np.floor(destination_y1)))
        output_x2 = min(img0_shape[1], int(np.ceil(destination_x2)))
        output_y2 = min(img0_shape[0], int(np.ceil(destination_y2)))
        destination_width = output_x2 - output_x1
        destination_height = output_y2 - output_y1
        if destination_width <= 0 or destination_height <= 0:
            continue

        transform = np.array(
            [
                [
                    scale_x * source_x_scale,
                    0,
                    (left + (output_x1 + 0.5) * scale_x) * source_x_scale
                    - 0.5,
                ],
                [
                    0,
                    scale_y * source_y_scale,
                    (top + (output_y1 + 0.5) * scale_y) * source_y_scale
                    - 0.5,
                ],
            ],
            dtype=np.float32,
        )

        scaled_masks[index, output_y1:output_y2, output_x1:output_x2] = (
            cv2.warpAffine(
                masks[index],
                transform,
                (destination_width, destination_height),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            )
        )

    return scaled_masks


def resolve_segmentation_outputs(output_dims, num_classes, num_masks, reg_max, output_layouts=None):
    """Resolve 10 segmentation outputs into (detection_order, mask_order, prototype_index).

    Supports both NHWC and NCHW tensor dimensions. If output_layouts is not
    provided, layout is automatically inferred from matching channel counts.
    """
    dimensions = [tuple(int(value) for value in dims) for dims in output_dims]
    if len(dimensions) != 10 or any(len(dims) != 4 for dims in dimensions):
        raise ValueError(f"Expected ten rank-four segmentation outputs, got {dimensions}")

    box_channels = 4 * reg_max
    expected = {num_masks, box_channels, num_classes}

    if output_layouts is None:
        all_nhwc = all(dims[-1] in expected for dims in dimensions)
        all_nchw = all(dims[1] in expected for dims in dimensions)
        if all_nhwc and not all_nchw:
            layouts = ["NHWC"] * 10
        elif all_nchw and not all_nhwc:
            layouts = ["NCHW"] * 10
        else:
            layouts = [None] * 10
    elif isinstance(output_layouts, (list, tuple)):
        layouts = list(output_layouts)
    else:
        layouts = [output_layouts] * 10

    def _channel_and_area(index, dims):
        layout = layouts[index]
        if layout is not None:
            norm_layout = str(layout).upper()
            if norm_layout == "NHWC":
                return dims[-1], dims[1] * dims[2]
            elif norm_layout == "NCHW":
                return dims[1], dims[2] * dims[3]
            raise ValueError(f"Unsupported layout {layout!r} for output {index}")

        # Auto-detect layout based on expected channels
        last_matches = dims[-1] in expected
        first_matches = dims[1] in expected

        if last_matches and not first_matches:
            return dims[-1], dims[1] * dims[2]
        elif first_matches and not last_matches:
            return dims[1], dims[2] * dims[3]
        elif last_matches:
            return dims[-1], dims[1] * dims[2]
        elif first_matches:
            return dims[1], dims[2] * dims[3]
        else:
            raise ValueError(
                f"Output {index} shape {dims} has neither axis -1 nor axis 1 matching "
                f"expected channels {sorted(expected)}"
            )

    channels_and_areas = [_channel_and_area(i, d) for i, d in enumerate(dimensions)]

    prototype_candidates = [
        index for index, (ch, _) in enumerate(channels_and_areas)
        if ch == num_masks
    ]
    if len(prototype_candidates) != 4:
        raise ValueError(f"Expected three mask outputs and one prototype output; got {prototype_candidates}")

    prototype_index = max(
        prototype_candidates,
        key=lambda index: channels_and_areas[index][1],
    )
    mask_order = sorted(
        (index for index in prototype_candidates if index != prototype_index),
        key=lambda index: channels_and_areas[index][1],
        reverse=True,
    )
    detection_candidates = [
        index for index, (ch, _) in enumerate(channels_and_areas)
        if ch in (box_channels, num_classes)
    ]
    if len(detection_candidates) != 6:
        raise ValueError(f"Expected six box/class outputs, got {detection_candidates}")
    detection_order = sorted(
        detection_candidates,
        key=lambda index: (
            channels_and_areas[index][1],
            1 if channels_and_areas[index][0] == box_channels else 0,
        ),
        reverse=True,
    )
    return detection_order, mask_order, prototype_index
