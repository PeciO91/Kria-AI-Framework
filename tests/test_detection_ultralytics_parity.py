import inspect
import json
import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None

from kria_ai.yolov26.detection import evaluate


@unittest.skipIf(np is None, "NumPy is required")
class TestUltralyticsDetectionParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = json.loads(
            Path(__file__).with_name("fixtures").joinpath(
                "detection_ultralytics_8_4_150.json"
            ).read_text()
        )
        cls.thresholds = np.asarray(cls.reference["iou_thresholds"], dtype=np.float32)

    def inputs(self, case):
        return (
            np.asarray(case["prediction_classes"], dtype=np.int32),
            np.asarray(case["prediction_scores"], dtype=np.float32),
            np.asarray(case["prediction_boxes"], dtype=np.float32).reshape(-1, 4),
            np.asarray(case["ground_truth_classes"], dtype=np.int32),
            np.asarray(case["ground_truth_boxes"], dtype=np.float32).reshape(-1, 4),
        )

    def records(self, cases):
        records = []
        counts = {}
        for case in cases:
            classes, scores, boxes, gt_classes, gt_boxes = self.inputs(case)
            matched = evaluate.match_detection_predictions(
                classes, scores, boxes, gt_classes, gt_boxes, self.thresholds
            )
            records.extend(
                (int(class_id), float(score), flags, case["name"], index)
                for index, (class_id, score, flags) in enumerate(zip(classes, scores, matched))
            )
            for class_id in gt_classes:
                counts[int(class_id)] = counts.get(int(class_id), 0) + 1
        return records, counts

    def test_reference_is_pinned(self):
        self.assertEqual(self.reference["reference_version"], "8.4.150")
        self.assertEqual(
            self.reference["wheel_sha256"],
            "1af11709f0e525e72d8b1b71916c0081de3e63c49ab9ffdbffa995ae6a8c69de",
        )
        np.testing.assert_array_equal(
            np.asarray(evaluate.DETECTION_IOU_THRESHOLDS, dtype=np.float32),
            self.thresholds,
        )

    def test_iou_matches_reference_including_tiny_and_degenerate_boxes(self):
        for case in self.reference["cases"]:
            with self.subTest(case=case["name"]):
                _, _, boxes, _, gt_boxes = self.inputs(case)
                actual = evaluate.box_iou_matrix(boxes, gt_boxes)
                expected = np.asarray(case["iou"], dtype=np.float32).reshape(
                    len(boxes), len(gt_boxes)
                )
                np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-9)

    def test_ap_matching_matches_reference_for_all_edge_cases(self):
        for case in self.reference["cases"]:
            with self.subTest(case=case["name"]):
                actual = evaluate.match_detection_predictions(*self.inputs(case), self.thresholds)
                expected = np.asarray(case["matches"], dtype=bool).reshape(-1, len(self.thresholds))
                np.testing.assert_array_equal(actual, expected)

    def test_confusion_matrix_matches_reference_at_both_confidence_floors(self):
        function = getattr(evaluate, "detection_confusion_matrix", None)
        self.assertTrue(callable(function), "separate Ultralytics confusion-matrix matching is required")
        for case in self.reference["cases"]:
            for confidence in (0.3, 0.001):
                with self.subTest(case=case["name"], confidence=confidence):
                    actual = function(
                        *self.inputs(case), num_classes=2,
                        confidence_threshold=confidence, iou_threshold=0.5,
                    )
                    np.testing.assert_array_equal(
                        actual, case["confusion_matrices"][str(confidence)]
                    )

    def test_ap_integration_matches_exact_reference(self):
        for curve in self.reference["ap_curves"]:
            with self.subTest(curve=curve):
                actual = evaluate._compute_ap(
                    np.asarray(curve["recall"]), np.asarray(curve["precision"])
                )
                self.assertAlmostEqual(actual, curve["expected"], places=12)

    def test_aggregate_matches_reference_ap_and_notebook_fixed_confidence_metrics(self):
        self.assertIn("confusion_matrix", inspect.signature(evaluate.aggregate_detection_metrics).parameters)
        records, counts = self.records(self.reference["cases"])
        for confidence in (0.3, 0.001):
            matrix = np.asarray(self.reference["aggregate"]["confusion_matrices"][str(confidence)])
            metrics = evaluate.aggregate_detection_metrics(
                records, counts, 2, self.thresholds, confusion_matrix=matrix
            )
            ap = np.asarray(self.reference["aggregate"]["ap"])
            self.assertAlmostEqual(metrics["map50"], ap[:, 0].mean(), places=12)
            self.assertAlmostEqual(metrics["map50_95"], ap.mean(), places=12)
            precisions, recalls = [], []
            for index, class_id in enumerate(self.reference["aggregate"]["ap_classes"]):
                tp = matrix[class_id, class_id]
                fp = matrix[class_id, :].sum() - tp
                fn = matrix[:, class_id].sum() - tp
                precision = tp / (tp + fp) if tp + fp else 0.0
                recall = tp / (tp + fn) if tp + fn else 0.0
                precisions.append(precision)
                recalls.append(recall)
                row = metrics["per_class"][class_id]
                self.assertEqual((row["tp"], row["fp"], row["fn"]), (tp, fp, fn))
                self.assertAlmostEqual(row["map50"], ap[index, 0], places=12)
                self.assertAlmostEqual(row["map50_95"], ap[index].mean(), places=12)
            precision, recall = np.mean(precisions), np.mean(recalls)
            self.assertAlmostEqual(metrics["precision"], precision, places=12)
            self.assertAlmostEqual(metrics["recall"], recall, places=12)
            self.assertAlmostEqual(metrics["f1"], 2 * precision * recall / (precision + recall), places=12)
            self.assertEqual(metrics["evaluator"], "ultralytics-8.4.150-numpy-v1")
            self.assertEqual(metrics["precision_recall_source"], "confusion_matrix")
            reversed_metrics = evaluate.aggregate_detection_metrics(
                list(reversed(records)), counts, 2, self.thresholds, confusion_matrix=matrix
            )
            self.assertEqual(metrics, reversed_metrics)

    def test_macro_pr_includes_configured_class_without_ground_truth(self):
        self.assertIn("confusion_matrix", inspect.signature(evaluate.aggregate_detection_metrics).parameters)
        case = next(
            row for row in self.reference["cases"] if row["name"] == "false_positive_class_without_gt"
        )
        records, counts = self.records([case])
        metrics = evaluate.aggregate_detection_metrics(
            records, counts, 2, self.thresholds,
            confusion_matrix=case["confusion_matrices"]["0.3"],
        )
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["f1"], 0.5)
        self.assertAlmostEqual(metrics["map50"], 0.995, places=12)
        self.assertEqual(metrics["per_class"][1]["fp"], 1)

    def test_confusion_matrix_validates_shape_and_class_ids(self):
        function = getattr(evaluate, "detection_confusion_matrix", None)
        self.assertTrue(callable(function))
        case = self.reference["cases"][0]
        args = list(self.inputs(case))
        args[0] = np.full(len(args[0]), 2)
        with self.assertRaises(ValueError):
            function(*args, num_classes=2, confidence_threshold=0.3)
        with self.assertRaises(ValueError):
            evaluate.aggregate_detection_metrics(
                [], {}, 2, self.thresholds, confusion_matrix=np.zeros((2, 2))
            )

    def test_board_aggregation_uses_worker_confusion_matrices(self):
        from kria_ai.yolov26.detection.benchmark import _accuracy_metrics

        case = next(
            row for row in self.reference["cases"] if row["name"] == "wrong_class_steals_cm_match"
        )
        records, counts = self.records([case])
        matrix = np.asarray(case["confusion_matrices"]["0.3"])
        worker = (1, 0.0, {}, None, records, counts, 1, matrix)
        metrics = _accuracy_metrics([worker], 2, self.thresholds)
        self.assertEqual(metrics["precision"], 0.0)
        self.assertEqual(metrics["recall"], 0.0)
        self.assertAlmostEqual(metrics["map50"], 0.995, places=12)
        self.assertEqual(metrics["confusion_matrix"], matrix.tolist())
        doubled = _accuracy_metrics([worker, worker], 2, self.thresholds)
        self.assertEqual(doubled["ground_truths"], 2)
        np.testing.assert_array_equal(doubled["confusion_matrix"], matrix * 2)

    def test_empty_dataset_has_zero_metrics_and_rejects_inconsistent_counts(self):
        metrics = evaluate.aggregate_detection_metrics(
            [], {}, 2, self.thresholds, confusion_matrix=np.zeros((3, 3))
        )
        for key in ("precision", "recall", "f1", "map50", "map50_95"):
            self.assertEqual(metrics[key], 0.0)
        with self.assertRaises(ValueError):
            evaluate.aggregate_detection_metrics(
                [], {0: 1}, 2, self.thresholds, confusion_matrix=np.zeros((3, 3))
            )

    def test_incomplete_worker_accuracy_results_fail_explicitly(self):
        from kria_ai.yolov26.detection.benchmark import _accuracy_metrics

        worker = (0, 0.0, {}, None, [], {}, 0, None)
        try:
            _accuracy_metrics([worker], 2, self.thresholds)
        except Exception as error:
            self.assertIsInstance(error, RuntimeError)
            self.assertIn("confusion matrix", str(error))
        else:
            self.fail("incomplete worker accuracy results must be rejected")

    def test_board_metric_metadata_records_effective_confidence_and_numpy(self):
        from kria_ai.yolov26.detection.benchmark import _accuracy_metrics

        self.assertIn("confidence_threshold", inspect.signature(_accuracy_metrics).parameters)
        case = next(row for row in self.reference["cases"] if row["name"] == "perfect")
        records, counts = self.records([case])
        for confidence in (0.3, 0.001):
            worker = (
                1, 0.0, {}, None, records, counts, 1,
                np.asarray(case["confusion_matrices"][str(confidence)]),
            )
            metrics = _accuracy_metrics(
                [worker], 2, self.thresholds, confidence_threshold=confidence
            )
            self.assertEqual(metrics["precision_recall_confidence_threshold"], confidence)
            self.assertEqual(metrics["precision_recall_matching_iou"], 0.5)
            self.assertEqual(metrics["numpy_version"], np.__version__)


if __name__ == "__main__":
    unittest.main()
