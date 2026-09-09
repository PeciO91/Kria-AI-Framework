import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import numpy as np
    from kria_ai.yolov26.segmentation.evaluate import compute_ap, mask_iou_matrix
    from kria_ai.yolov26.segmentation.masks import crop_mask, resolve_segmentation_outputs
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import cv2
    from kria_ai.yolov26.segmentation.masks import scale_image_masks
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


@unittest.skipUnless(NUMPY_AVAILABLE, "NumPy is required")
class TestSegUtils(unittest.TestCase):
    def test_crop_mask(self):
        # Create a 4x4 mask of all ones
        masks = np.ones((1, 4, 4), dtype=np.float32)
        # Bounding box covering only the bottom-right 2x2 area: x1=2, y1=2, x2=4, y2=4
        boxes = np.array([[2, 2, 4, 4]], dtype=np.float32)

        cropped = crop_mask(masks, boxes)

        expected = np.zeros((1, 4, 4), dtype=np.float32)
        expected[0, 2:4, 2:4] = 1.0

        np.testing.assert_array_equal(cropped, expected)

    def test_mask_iou_matrix(self):
        # Two predicted masks (2x2)
        pred = np.zeros((2, 2, 2), dtype=bool)
        pred[0, 0, 0] = True  # Top-left
        pred[1, 1, 1] = True  # Bottom-right

        # Two GT masks
        gt = np.zeros((2, 2, 2), dtype=bool)
        gt[0, 0, 0] = True  # Top-left (perfect match for pred 0)
        gt[1, 0, 0] = True
        gt[1, 1, 1] = True  # Top-left and bottom-right (overlap with pred 0 and 1)

        iou = mask_iou_matrix(pred, gt)

        self.assertEqual(iou.shape, (2, 2))
        self.assertAlmostEqual(iou[0, 0], 1.0)
        self.assertAlmostEqual(iou[0, 1], 0.5)
        self.assertAlmostEqual(iou[1, 0], 0.0)
        self.assertAlmostEqual(iou[1, 1], 0.5)

    def test_compute_ap(self):
        # Perfect PR curve
        recall = np.array([0.1, 0.5, 1.0])
        precision = np.array([1.0, 1.0, 1.0])
        ap = compute_ap(recall, precision)
        self.assertAlmostEqual(ap, 1.0)

        # Triangle PR curve
        recall = np.array([0.5, 1.0])
        precision = np.array([1.0, 0.5])
        ap = compute_ap(recall, precision)
        self.assertAlmostEqual(ap, 0.75)

    @unittest.skipUnless(CV2_AVAILABLE, "OpenCV (cv2) is required")
    def test_scale_image_masks(self):
        img1_shape = (640, 640)
        img0_shape = (320, 640)

        masks = np.zeros((1, 640, 640), dtype=np.float32)
        masks[0, 160:360, 200:400] = 1.0

        bboxes = np.array([[200, 160, 400, 360]], dtype=np.float32)
        scaled = scale_image_masks(masks, bboxes, img1_shape, img0_shape)

        self.assertEqual(scaled.shape, (1, 320, 640))
        self.assertEqual(scaled[0, 100, 300], 1.0)
        self.assertEqual(scaled[0, 250, 300], 0.0)

    def test_resolve_segmentation_outputs_nhwc(self):
        # NHWC layout (1, H, W, C)
        out_dims = [
            (1, 80, 80, 64),   # 0: P3 box
            (1, 80, 80, 80),   # 1: P3 cls
            (1, 80, 80, 32),   # 2: P3 mask
            (1, 40, 40, 64),   # 3: P4 box
            (1, 40, 40, 80),   # 4: P4 cls
            (1, 40, 40, 32),   # 5: P4 mask
            (1, 20, 20, 64),   # 6: P5 box
            (1, 20, 20, 80),   # 7: P5 cls
            (1, 20, 20, 32),   # 8: P5 mask
            (1, 160, 160, 32), # 9: Proto
        ]

        np.random.seed(42)
        order = np.random.permutation(10)
        shuffled_dims = [out_dims[i] for i in order]
        expected_proto_idx = int(np.where(order == 9)[0][0])

        det_indices = [int(np.where(order == i)[0][0]) for i in [0, 1, 3, 4, 6, 7]]
        mask_indices = [int(np.where(order == i)[0][0]) for i in [2, 5, 8]]

        det_order, mask_order, proto_idx = resolve_segmentation_outputs(
            shuffled_dims, num_classes=80, num_masks=32, reg_max=16
        )

        self.assertEqual(proto_idx, expected_proto_idx)
        self.assertEqual(det_order, det_indices)
        self.assertEqual(mask_order, mask_indices)

    def test_resolve_segmentation_outputs_nchw(self):
        # NCHW layout (1, C, H, W)
        out_dims = [
            (1, 64, 80, 80),   # 0: P3 box
            (1, 80, 80, 80),   # 1: P3 cls
            (1, 32, 80, 80),   # 2: P3 mask
            (1, 64, 40, 40),   # 3: P4 box
            (1, 80, 40, 40),   # 4: P4 cls
            (1, 32, 40, 40),   # 5: P4 mask
            (1, 64, 20, 20),   # 6: P5 box
            (1, 80, 20, 20),   # 7: P5 cls
            (1, 32, 20, 20),   # 8: P5 mask
            (1, 32, 160, 160), # 9: Proto
        ]

        np.random.seed(123)
        order = np.random.permutation(10)
        shuffled_dims = [out_dims[i] for i in order]
        expected_proto_idx = int(np.where(order == 9)[0][0])

        det_indices = [int(np.where(order == i)[0][0]) for i in [0, 1, 3, 4, 6, 7]]
        mask_indices = [int(np.where(order == i)[0][0]) for i in [2, 5, 8]]

        det_order, mask_order, proto_idx = resolve_segmentation_outputs(
            shuffled_dims, num_classes=80, num_masks=32, reg_max=16
        )

        self.assertEqual(proto_idx, expected_proto_idx)
        self.assertEqual(det_order, det_indices)
        self.assertEqual(mask_order, mask_indices)


if __name__ == '__main__':
    unittest.main()
