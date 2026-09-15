import unittest

from kria_ai.classification.config import get_dataset as get_classification_dataset
from kria_ai.classification.config import get_model as get_classification_model
from kria_ai.classification.config import validate_model_dataset
from kria_ai.common.board.config import KV260
from kria_ai.yolov26.config import COCO_CLASSES
from kria_ai.yolov26.detection.config import get_model as get_detection_model
from kria_ai.yolov26.segmentation.config import get_model as get_segmentation_model


class TestFamilyConfigs(unittest.TestCase):
    def test_family_defaults_are_compatible(self):
        validate_model_dataset(get_classification_model(), get_classification_dataset())

    def test_model_ids_are_preserved(self):
        self.assertEqual(get_classification_model().id, "resnet18")
        self.assertEqual(get_detection_model().id, "yolov26s")
        self.assertEqual(get_segmentation_model().id, "yolov26n_seg")

    def test_yolov26_is_coco_80(self):
        self.assertEqual(len(COCO_CLASSES), 80)
        self.assertEqual(get_detection_model().num_classes, 80)
        self.assertEqual(get_segmentation_model().num_classes, 80)

    def test_yolov26n_dpu_test_model(self):
        config = get_detection_model("yolov26n_dpu_test")
        self.assertEqual(config.id, "yolov26n_dpu_test")
        self.assertEqual(config.num_classes, 2)
        self.assertEqual(config.input_size, (640, 640))
        self.assertEqual(config.reg_max, 1)
        self.assertEqual(config.strides, (8, 16, 32))
        self.assertEqual(str(config.checkpoint_path), "models/yolo26n_dpu_leaky_dwrelu.pt")
        self.assertEqual(str(config.architecture_path), "configs/yolo26n_dpu_leaky_dwrelu.yaml")
        self.assertEqual(config.activation_policy, "leaky_13_128_conv_relu_dw")
        self.assertEqual(config.graph_policy, "split_c3k2_cv1")
        self.assertTrue(any(pattern.startswith("model.23.") for pattern in config.prune_excludes))

    def test_kv260_runner_limits(self):
        self.assertEqual(KV260.validate_runner_count(1), 1)
        self.assertEqual(KV260.validate_runner_count(4), 4)
        with self.assertRaises(ValueError):
            KV260.validate_runner_count(0)
        with self.assertRaises(ValueError):
            KV260.validate_runner_count(5)


if __name__ == "__main__":
    unittest.main()
