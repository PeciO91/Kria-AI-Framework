import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import numpy as np
    import cv2
    from kria_ai.yolov26.preprocess import letterbox
    from kria_ai.yolov26.detection.postprocess import scale_to_original
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import torch
    from kria_ai.yolov26.preprocess import image_to_float_tensor
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TestYOLOPreprocessing(unittest.TestCase):
    @unittest.skipUnless(CV2_AVAILABLE, "NumPy and OpenCV are required")
    def test_letterbox_landscape(self):
        # 400x800 image (H=400, W=800)
        image = np.ones((400, 800, 3), dtype=np.uint8) * 200
        new_shape = (640, 640)
        pad_color = (114, 114, 114)

        result, ratio, (pad_w, pad_h) = letterbox(image, new_shape=new_shape, color=pad_color)

        self.assertEqual(result.shape, (640, 640, 3))
        # Ratio should be 640/800 = 0.8
        self.assertAlmostEqual(ratio[0], 0.8)
        self.assertAlmostEqual(ratio[1], 0.8)
        # Unpadded width = 800 * 0.8 = 640, height = 400 * 0.8 = 320
        # pad_w = 0, pad_h = (640 - 320)/2 = 160
        self.assertAlmostEqual(pad_w, 0.0)
        self.assertAlmostEqual(pad_h, 160.0)

        # Padding area at top and bottom should have pad_color
        np.testing.assert_array_equal(result[0, 320], pad_color)
        np.testing.assert_array_equal(result[639, 320], pad_color)
        # Center should have image color
        np.testing.assert_array_equal(result[320, 320], [200, 200, 200])

    @unittest.skipUnless(CV2_AVAILABLE, "NumPy and OpenCV are required")
    def test_letterbox_portrait(self):
        # 800x400 image (H=800, W=400)
        image = np.ones((800, 400, 3), dtype=np.uint8) * 150
        new_shape = (640, 640)
        pad_color = (114, 114, 114)

        result, ratio, (pad_w, pad_h) = letterbox(image, new_shape=new_shape, color=pad_color)

        self.assertEqual(result.shape, (640, 640, 3))
        # Ratio should be 640/800 = 0.8
        self.assertAlmostEqual(ratio[0], 0.8)
        self.assertAlmostEqual(ratio[1], 0.8)
        # Unpadded width = 400 * 0.8 = 320, height = 800 * 0.8 = 640
        # pad_w = 160, pad_h = 0
        self.assertAlmostEqual(pad_w, 160.0)
        self.assertAlmostEqual(pad_h, 0.0)

        # Padding area at left and right should have pad_color
        np.testing.assert_array_equal(result[320, 0], pad_color)
        np.testing.assert_array_equal(result[320, 639], pad_color)
        # Center should have image color
        np.testing.assert_array_equal(result[320, 320], [150, 150, 150])

    @unittest.skipUnless(CV2_AVAILABLE, "NumPy and OpenCV are required")
    def test_letterbox_square_exact(self):
        # Already 640x640
        image = np.ones((640, 640, 3), dtype=np.uint8) * 100
        result, ratio, (pad_w, pad_h) = letterbox(image, new_shape=(640, 640))

        self.assertEqual(result.shape, (640, 640, 3))
        self.assertAlmostEqual(ratio[0], 1.0)
        self.assertAlmostEqual(ratio[1], 1.0)
        self.assertAlmostEqual(pad_w, 0.0)
        self.assertAlmostEqual(pad_h, 0.0)
        np.testing.assert_array_equal(result, image)

    @unittest.skipUnless(CV2_AVAILABLE, "NumPy and OpenCV are required")
    def test_letterbox_odd_padding(self):
        # 401x800 image (H=401, W=800): 400.8 resized height rounds to 321,
        # leaving 319 pixels of padding split 159 top / 160 bottom.
        image = np.ones((401, 800, 3), dtype=np.uint8) * 200

        result, ratio, (left, top) = letterbox(image, new_shape=(640, 640))

        self.assertEqual(result.shape, (640, 640, 3))
        self.assertAlmostEqual(ratio[0], 0.8)
        self.assertAlmostEqual(ratio[1], 0.8)
        self.assertEqual((left, top), (0, 159))
        np.testing.assert_array_equal(result[158, 320], (114, 114, 114))
        np.testing.assert_array_equal(result[159, 320], [200, 200, 200])

    @unittest.skipUnless(CV2_AVAILABLE, "NumPy and OpenCV are required")
    def test_scale_to_original_odd_padding_round_trip(self):
        # Inverse of the odd letterbox above: gain 0.8, actual top offset 159.
        original_box = np.array([[100.0, 50.0, 300.0, 200.0]], dtype=np.float32)
        input_box = original_box.copy()
        input_box[:, [0, 2]] = input_box[:, [0, 2]] * 0.8
        input_box[:, [1, 3]] = input_box[:, [1, 3]] * 0.8 + 159.0

        restored = scale_to_original(
            input_box,
            input_shape=(640, 640),
            original_shape=(401, 800),
        )
        np.testing.assert_allclose(restored, original_box, rtol=1e-6, atol=1e-5)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch and NumPy are required")
    def test_image_to_float_tensor(self):
        image = np.array([[[0, 128, 255]]], dtype=np.uint8)  # HWC: (1, 1, 3)
        tensor = image_to_float_tensor(image, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))

        self.assertEqual(tensor.shape, (3, 1, 1))
        self.assertAlmostEqual(float(tensor[0, 0, 0]), 0.0 / 255.0, places=5)
        self.assertAlmostEqual(float(tensor[1, 0, 0]), 128.0 / 255.0, places=5)
        self.assertAlmostEqual(float(tensor[2, 0, 0]), 255.0 / 255.0, places=5)


if __name__ == "__main__":
    unittest.main()
