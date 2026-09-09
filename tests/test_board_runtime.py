import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import numpy as np
    from kria_ai.common.board.input_quantization import (
        apply_normalization_lut,
        build_normalization_lut,
        normalization_constants,
    )
    from kria_ai.common.board.runtime import (
        TensorMetadata,
        allocate_output_buffers,
        output_dequantization_scales,
    )
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class FakeTensor:
    def __init__(self, name, dims, fix_point=7):
        self.name = name
        self.dims = list(dims)
        self.fix_point = fix_point

    def get_name(self):
        return self.name

    def get_attr(self, key):
        if key == "fix_point":
            return str(self.fix_point)
        raise KeyError(key)


class FakeRunner:
    def __init__(self, output_tensors):
        self._output_tensors = output_tensors

    def get_output_tensors(self):
        return self._output_tensors


@unittest.skipUnless(NUMPY_AVAILABLE, "NumPy is required")
class TestBoardRuntime(unittest.TestCase):
    def test_tensor_metadata_and_scales(self):
        meta1 = TensorMetadata(name="out_box", shape=(1, 20, 20, 4), fixed_point=7)
        meta2 = TensorMetadata(name="out_cls", shape=(1, 20, 20, 80), fixed_point=4)

        self.assertEqual(meta1.name, "out_box")
        self.assertEqual(meta1.shape, (1, 20, 20, 4))
        self.assertEqual(meta1.fixed_point, 7)

        scales = output_dequantization_scales([meta1, meta2])
        self.assertEqual(len(scales), 2)
        self.assertAlmostEqual(scales[0], 2.0 ** -7)
        self.assertAlmostEqual(scales[1], 2.0 ** -4)

    def test_allocate_output_buffers(self):
        tensors = [
            FakeTensor("tensor_a", (1, 80, 80, 4)),
            FakeTensor("tensor_b", (1, 80, 80, 80)),
        ]
        runner = FakeRunner(tensors)
        buffers = allocate_output_buffers(runner)

        self.assertEqual(len(buffers), 2)
        self.assertEqual(buffers[0].shape, (1, 80, 80, 4))
        self.assertEqual(buffers[0].dtype, np.int8)
        self.assertEqual(buffers[1].shape, (1, 80, 80, 80))
        self.assertEqual(buffers[1].dtype, np.int8)

    def test_normalization_constants(self):
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        fixed_point = 7

        scale, shift = normalization_constants(mean, std, fixed_point)
        self.assertEqual(scale.shape, (3,))
        self.assertEqual(shift.shape, (3,))

        # Scale = 2^7 / (255 * std)
        expected_scale = 128.0 / (255.0 * np.array(std, dtype=np.float32))
        expected_shift = np.array(mean, dtype=np.float32) * 128.0 / np.array(std, dtype=np.float32)
        np.testing.assert_allclose(scale, expected_scale, rtol=1e-5)
        np.testing.assert_allclose(shift, expected_shift, rtol=1e-5)

    def test_build_and_apply_normalization_lut_1d(self):
        # When all channels have identical mean and std (like YOLO: mean=0, std=1)
        mean = (0.0, 0.0, 0.0)
        std = (1.0, 1.0, 1.0)
        fixed_point = 7

        lut = build_normalization_lut(mean, std, fixed_point)
        # Should collapse to 1D table of shape (256,)
        self.assertEqual(lut.shape, (256,))
        self.assertEqual(lut.dtype, np.int8)

        # 0 -> 0 * 128/255 = 0
        self.assertEqual(lut[0], 0)
        # 255 -> 255 * 128/255 = 128 -> clipped to 127
        self.assertEqual(lut[255], 127)

        # Apply LUT to a dummy uint8 image
        image = np.array([[[0, 128, 255]]], dtype=np.uint8)
        quantized = apply_normalization_lut(image, lut)

        self.assertEqual(quantized.shape, (1, 1, 3))
        self.assertEqual(quantized.dtype, np.int8)
        self.assertEqual(quantized[0, 0, 0], 0)
        self.assertEqual(quantized[0, 0, 1], int(round(128.0 * 128.0 / 255.0)))
        self.assertEqual(quantized[0, 0, 2], 127)

    def test_build_and_apply_normalization_lut_3d(self):
        # Classification (ImageNet: different mean/std per channel)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        fixed_point = 7

        lut = build_normalization_lut(mean, std, fixed_point)
        self.assertEqual(lut.shape, (256, 3))
        self.assertEqual(lut.dtype, np.int8)

        # Apply to a synthetic RGB image
        image = np.full((10, 10, 3), 128, dtype=np.uint8)
        quantized = apply_normalization_lut(image, lut)

        self.assertEqual(quantized.shape, (10, 10, 3))
        self.assertEqual(quantized.dtype, np.int8)
        for ch in range(3):
            self.assertEqual(quantized[0, 0, ch], lut[128, ch])

    def test_channel_mismatch_raises(self):
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        lut = build_normalization_lut(mean, std, 7)  # (256, 3)

        image_4ch = np.zeros((4, 4, 4), dtype=np.uint8)
        with self.assertRaises(ValueError):
            apply_normalization_lut(image_4ch, lut)


if __name__ == "__main__":
    unittest.main()
