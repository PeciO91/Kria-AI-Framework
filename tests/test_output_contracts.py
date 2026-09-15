"""Dependency-free checks for the canonical XMODEL output contracts.

``OutputContract.validate`` consumes only the public ``.shape`` interface, so
tiny tensor doubles are sufficient and no NumPy/OpenCV/PyTorch is needed.
"""

import os
import sys
import types
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kria_ai.yolov26.detection.config import get_model as get_detection_model
from kria_ai.yolov26.export import build_output_contract
from kria_ai.yolov26.models import validate_yolov26_head
from kria_ai.yolov26.segmentation.config import get_model as get_segmentation_model


def _outputs(dims_list):
    return [types.SimpleNamespace(shape=dims) for dims in dims_list]


# Canonical 640x640 NCHW contracts: strides (8, 16, 32) -> grids 80, 40, 20.
DETECTION_DIMS = [
    (1, 4, 80, 80),
    (1, 80, 80, 80),
    (1, 4, 40, 40),
    (1, 80, 40, 40),
    (1, 4, 20, 20),
    (1, 80, 20, 20),
]

SEGMENTATION_DIMS = [
    (1, 4, 80, 80),     # P3 box
    (1, 80, 80, 80),    # P3 cls
    (1, 32, 80, 80),    # P3 mask
    (1, 4, 40, 40),     # P4 box
    (1, 80, 40, 40),    # P4 cls
    (1, 32, 40, 40),    # P4 mask
    (1, 4, 20, 20),     # P5 box
    (1, 80, 20, 20),    # P5 cls
    (1, 32, 20, 20),    # P5 mask
    (1, 32, 160, 160),  # prototypes
]


class TestDetectionOutputContract(unittest.TestCase):
    def test_canonical_six_outputs(self):
        contract = build_output_contract(get_detection_model())
        outputs = _outputs(DETECTION_DIMS)
        self.assertEqual(tuple(contract.validate(outputs)), tuple(outputs))

    def test_rejects_wrong_output_count(self):
        contract = build_output_contract(get_detection_model())
        with self.assertRaises(ValueError):
            contract.validate(_outputs(DETECTION_DIMS[:-1]))

    def test_rejects_wrong_channels(self):
        contract = build_output_contract(get_detection_model())
        dims = list(DETECTION_DIMS)
        dims[0] = (1, 5, 80, 80)
        with self.assertRaises(ValueError):
            contract.validate(_outputs(dims))

    def test_two_class_test_model_contract(self):
        contract = build_output_contract(get_detection_model("yolov26n_dpu_test"))
        dims = [
            (1, 4, 80, 80),
            (1, 2, 80, 80),
            (1, 4, 40, 40),
            (1, 2, 40, 40),
            (1, 4, 20, 20),
            (1, 2, 20, 20),
        ]
        outputs = _outputs(dims)
        self.assertEqual(tuple(contract.validate(outputs)), tuple(outputs))

    def test_two_class_head_validation(self):
        class Detect:
            nc = 2
            nl = 3
            stride = (8, 16, 32)
            reg_max = 1
            one2one_cv2 = [None, None, None]
            one2one_cv3 = [None, None, None]

        model = types.SimpleNamespace(model=[Detect()])
        head = validate_yolov26_head(model, get_detection_model("yolov26n_dpu_test"))
        self.assertIs(head, model.model[-1])


class TestSegmentationOutputContract(unittest.TestCase):
    def test_canonical_ten_outputs(self):
        contract = build_output_contract(get_segmentation_model())
        outputs = _outputs(SEGMENTATION_DIMS)
        self.assertEqual(tuple(contract.validate(outputs)), tuple(outputs))

    def test_rejects_wrong_output_count(self):
        contract = build_output_contract(get_segmentation_model())
        with self.assertRaises(ValueError):
            contract.validate(_outputs(SEGMENTATION_DIMS[:-1]))

    def test_rejects_wrong_channels(self):
        contract = build_output_contract(get_segmentation_model())
        dims = list(SEGMENTATION_DIMS)
        dims[8] = (1, 33, 20, 20)
        with self.assertRaises(ValueError):
            contract.validate(_outputs(dims))


if __name__ == "__main__":
    unittest.main()
