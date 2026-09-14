"""CLI validation tests that must not require board runtime dependencies."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kria_ai.classification.benchmark import main as classification_main
from kria_ai.yolov26.detection.benchmark import main as detection_main


class TestBenchmarkRunnerValidation(unittest.TestCase):
    def _assert_rejects_threads(self, main, threads):
        # Must fail argument validation before importing the NumPy-dependent
        # decoder or the XIR/VART board runtime.
        with self.assertRaises(SystemExit) as context:
            main(["--threads", str(threads)])
        self.assertEqual(context.exception.code, 2)

    def test_classification_rejects_zero_threads(self):
        self._assert_rejects_threads(classification_main, 0)

    def test_classification_rejects_five_threads(self):
        self._assert_rejects_threads(classification_main, 5)

    def test_detection_rejects_zero_threads(self):
        self._assert_rejects_threads(detection_main, 0)

    def test_detection_rejects_five_threads(self):
        self._assert_rejects_threads(detection_main, 5)


if __name__ == "__main__":
    unittest.main()
