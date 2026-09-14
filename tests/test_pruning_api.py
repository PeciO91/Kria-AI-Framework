"""Pruning API contract tests against the frozen Vitis AI 3.5 signatures."""

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kria_ai.classification.optimize import _prune
from kria_ai.yolov26.detection.optimize import _prune_model as _detection_prune
from kria_ai.yolov26.segmentation.optimize import _prune_model as _segmentation_prune


class _RecordingRunner:
    def __init__(self):
        self.calls = []
        self.model = object()

    def prune(self, **kwargs):
        self.calls.append(kwargs)
        return self.model


class TestClassificationPrune(unittest.TestCase):
    def test_iterative_signature(self):
        runner = _RecordingRunner()
        model, representation = _prune(runner, "iterative", 0.2, 2)
        self.assertIs(model, runner.model)
        self.assertEqual(representation, "sparse")
        self.assertEqual(
            runner.calls,
            [{"removal_ratio": 0.2, "mode": "sparse", "channel_divisible": 2}],
        )

    def test_one_step_signature(self):
        runner = _RecordingRunner()
        model, representation = _prune(runner, "one_step", 0.2, 2)
        self.assertIs(model, runner.model)
        self.assertEqual(representation, "slim")
        self.assertEqual(
            runner.calls,
            [
                {
                    "removal_ratio": 0.2,
                    "mode": "slim",
                    "index": None,
                    "channel_divisible": 2,
                }
            ],
        )


class _YoloPruneCases:
    prune_model = None

    def test_iterative_signature(self):
        runner = _RecordingRunner()
        result = self.prune_model(
            runner,
            method="iterative",
            ratio=0.2,
            excludes=["excluded"],
            channel_divisible=2,
            pruning_info_path=Path("info.json"),
        )
        self.assertIs(result, runner.model)
        self.assertEqual(
            runner.calls,
            [
                {
                    "mode": "sparse",
                    "excludes": ["excluded"],
                    "removal_ratio": 0.2,
                    "channel_divisible": 2,
                    "pruning_info_path": "info.json",
                }
            ],
        )

    def test_one_step_signature_has_no_excludes(self):
        runner = _RecordingRunner()
        result = self.prune_model(
            runner,
            method="one_step",
            ratio=0.2,
            excludes=["excluded"],
            channel_divisible=2,
            pruning_info_path=Path("info.json"),
        )
        self.assertIs(result, runner.model)
        self.assertEqual(
            runner.calls,
            [
                {
                    "mode": "slim",
                    "index": None,
                    "removal_ratio": 0.2,
                    "channel_divisible": 2,
                    "pruning_info_path": "info.json",
                }
            ],
        )


class TestDetectionPrune(_YoloPruneCases, unittest.TestCase):
    prune_model = staticmethod(_detection_prune)


class TestSegmentationPrune(_YoloPruneCases, unittest.TestCase):
    prune_model = staticmethod(_segmentation_prune)


if __name__ == "__main__":
    unittest.main()
