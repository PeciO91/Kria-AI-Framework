import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import numpy as np
    from kria_ai.yolov26.decode import UltralyticsDecoderCache, decode_ultralytics_output
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


@unittest.skipUnless(NUMPY_AVAILABLE, "NumPy is required")
class TestDetectionUtils(unittest.TestCase):
    def test_decode_ultralytics_output(self):
        # We simulate the exact arguments required by decode_ultralytics_output
        box_tensor = np.zeros((1, 20, 20, 64), dtype=np.int8)
        cls_tensor = np.zeros((1, 20, 20, 2), dtype=np.int8)

        # Dequant scale: 0.25 -> an INT8 value of 40 becomes 10.0
        box_tensor[0, 10, 10, 5] = 40
        box_tensor[0, 10, 10, 16 + 5] = 40
        box_tensor[0, 10, 10, 32 + 5] = 40
        box_tensor[0, 10, 10, 48 + 5] = 40

        cls_tensor[0, 10, 10, 1] = 40

        int8_outputs = [box_tensor, cls_tensor]
        dequant_scales = [0.25, 0.25]
        output_order = [0, 1]

        # Strides config is [32] since it's 20x20 for 640x640 input
        cache = UltralyticsDecoderCache([32])

        boxes, scores, class_ids, keep_indices = decode_ultralytics_output(
            int8_outputs,
            dequant_scales,
            conf_threshold=0.5,
            cache=cache,
            output_order=output_order,
            num_classes=2,
            reg_max=16,
            return_keep_index=True,
        )

        self.assertEqual(len(boxes), 1)
        self.assertEqual(len(scores), 1)
        self.assertEqual(len(class_ids), 1)
        self.assertEqual(len(keep_indices), 1)

        # Grid 10, 10 with stride 32 => center is 336, 336
        # Dist is 10.0 (40 * 0.25)
        # 10.0 * stride 32 = 320
        # x1 = 336 - 160 = 176, y1 = 336 - 160 = 176
        # x2 = 336 + 160 = 496, y2 = 336 + 160 = 496
        # w = 320, h = 320
        np.testing.assert_allclose(boxes[0], [176, 176, 320, 320], rtol=1e-3)
        self.assertEqual(class_ids[0], 1)
        self.assertAlmostEqual(scores[0], 1.0, places=3)

        # Level 0, 10*20 + 10 = 210
        self.assertEqual(keep_indices[0][0], 0)
        self.assertEqual(keep_indices[0][1], 210)


if __name__ == "__main__":
    unittest.main()
