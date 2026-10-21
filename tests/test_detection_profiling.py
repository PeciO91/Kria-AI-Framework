"""Detection timing boundaries without images, VART or a Kria board."""

import contextlib
import io
import json
import queue
import tempfile
import threading
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from kria_ai.common.board.profiling import StageProfiler
from kria_ai.yolov26.detection import benchmark

try:
    import numpy as np
except ImportError:
    np = None


class Clock:
    def __init__(self):
        self.value = 0.0

    def now(self):
        return self.value

    def advance(self, seconds, result=None):
        self.value += seconds
        return result


class TimedQueue(queue.Queue):
    def __init__(self, clock, get_seconds=0.0, put_seconds=0.0):
        super().__init__()
        self.clock = clock
        self.get_seconds = get_seconds
        self.put_seconds = put_seconds

    def get(self, *args, **kwargs):
        self.clock.advance(self.get_seconds)
        return super().get(*args, **kwargs)

    def put(self, *args, **kwargs):
        self.clock.advance(self.put_seconds)
        return super().put(*args, **kwargs)


class TestDetectionProducerProfiling(unittest.TestCase):
    def test_read_preprocess_and_backpressure_are_separate(self):
        clock = Clock()
        image = types.SimpleNamespace(shape=(32, 32, 3))
        cv2 = types.ModuleType("cv2")
        cv2.COLOR_BGR2RGB = 1
        cv2.imread = lambda _: clock.advance(0.010, image)
        cv2.cvtColor = lambda *_: clock.advance(0.001, image)
        output = TimedQueue(clock, put_seconds=0.020)
        profiler = StageProfiler(True)
        with (
            patch.dict("sys.modules", {"cv2": cv2}),
            patch("time.perf_counter", clock.now),
            patch.object(
                benchmark, "preprocess_board_image",
                lambda *_: clock.advance(0.002, "input"),
            ),
        ):
            benchmark._producer(
                [Path("image.jpg")], output, (32, 32), None, profiler,
                threading.Event(), [], threading.Lock(),
            )
        self.assertAlmostEqual(profiler._samples["image_read"][0], 0.010)
        self.assertAlmostEqual(profiler._samples["preprocess"][0], 0.003)
        self.assertAlmostEqual(profiler._samples["input_enqueue_wait"][0], 0.020)
        self.assertAlmostEqual(profiler._samples["producer_total"][0], 0.033)
        self.assertEqual(output.get()[::3], ("input", "image.jpg"))

    def test_skipped_image_has_no_preprocessing_or_frame_total(self):
        clock = Clock()
        cv2 = types.ModuleType("cv2")
        cv2.imread = lambda _: clock.advance(0.010)
        profiler = StageProfiler(True)
        output = queue.Queue()
        with (
            patch.dict("sys.modules", {"cv2": cv2}),
            patch("time.perf_counter", clock.now),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            benchmark._producer(
                [Path("bad.jpg")], output, (32, 32), None, profiler,
                threading.Event(), [], threading.Lock(),
            )
        self.assertEqual(profiler._samples, {"image_read": [0.010]})
        self.assertTrue(output.empty())

    def test_writer_records_idle_wait_separately_from_writing(self):
        clock = Clock()
        cv2 = types.ModuleType("cv2")
        cv2.imwrite = lambda *_: clock.advance(0.007, True)
        source = TimedQueue(clock, get_seconds=0.005)
        source.put(("image.jpg", object()))
        source.put(None)
        profiler = StageProfiler(True)
        with (
            patch.dict("sys.modules", {"cv2": cv2}),
            patch("time.perf_counter", clock.now),
        ):
            benchmark._writer(
                source, Path("."), profiler,
                threading.Event(), [], threading.Lock(),
            )
        self.assertAlmostEqual(profiler._samples["write"][0], 0.007)
        self.assertEqual(len(profiler._samples["writer_dequeue_wait"]), 2)
        self.assertAlmostEqual(
            sum(profiler._samples["writer_dequeue_wait"]), 0.010
        )
        self.assertEqual(source.unfinished_tasks, 0)


@unittest.skipIf(np is None, "NumPy is required")
class TestDetectionConsumerProfiling(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((32, 32, 3), dtype=np.uint8)
        self.outputs = [
            np.ones((1, 1, 1, 4), dtype=np.int8),
            np.array([[[[1, -10]]]], dtype=np.int8),
        ]
        self.model = types.SimpleNamespace(
            strides=(8,), num_classes=2, reg_max=1, max_detections=300
        )

    def run_consumer(self, *, enabled, draw, save, accuracy, progress_seconds=0.0):
        clock = Clock()
        source = TimedQueue(clock, get_seconds=0.006)
        source.put(("input", self.image, (32, 32), "image.jpg"))
        source.put(None)
        output = TimedQueue(clock, put_seconds=0.020) if save else None
        profiler = StageProfiler(enabled)
        results = [None]
        cv2 = types.ModuleType("cv2")
        cv2.FONT_HERSHEY_SIMPLEX = 0
        cv2.rectangle = lambda *_: clock.advance(0.003)
        cv2.putText = lambda *_: clock.advance(0.005)
        runner = types.SimpleNamespace(
            execute_async=lambda *_: 1,
            wait=lambda _: clock.advance(0.004),
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict("sys.modules", {"cv2": cv2}),
            patch("time.perf_counter", clock.now),
            patch.object(
                benchmark, "create_runner",
                lambda _: clock.advance(0.001, runner),
            ),
            patch.object(
                benchmark, "allocate_output_buffers",
                lambda _: clock.advance(0.002, self.outputs),
            ),
            patch.object(benchmark, "output_dequantization_scales", lambda _: (1, 1)),
        ):
            Path(directory, "image.txt").write_text(
                "0 0.1875 0.1875 0.375 0.375\n", encoding="utf-8"
            )
            benchmark._consumer(
                0, source, output, types.SimpleNamespace(outputs=()),
                (32, 32), self.model,
                types.SimpleNamespace(classes=("Person", "Face")),
                [0, 1], types.SimpleNamespace(
                    increment=lambda: clock.advance(progress_seconds)
                ),
                results, profiler, draw, save, accuracy, Path(directory),
                0.3, (0.5, 0.75), threading.Event(), [], threading.Lock(),
            )
        self.assertEqual(source.unfinished_tasks, 0)
        return profiler, results[0], output

    def test_real_decode_draw_and_queue_stages_preserve_accuracy(self):
        profiler, result, output = self.run_consumer(
            enabled=True, draw=True, save=True, accuracy=True
        )
        samples = profiler._samples
        for stage in (
            "decode_total", "decode_ultra_layout", "decode_ultra_threshold",
            "decode_ultra_dequant", "decode_ultra_class_score",
            "decode_ultra_box_decode", "decode_ultra_concat",
            "detection_postprocess", "nms_or_topk", "coord_scale", "postprocess",
            "accuracy", "draw_boxes",
        ):
            self.assertEqual(len(samples[stage]), 1, stage)
        self.assertAlmostEqual(samples["runner_setup"][0], 0.003)
        self.assertAlmostEqual(samples["dpu"][0], 0.004)
        self.assertAlmostEqual(samples["draw_boxes"][0], 0.008)
        self.assertAlmostEqual(samples["writer_enqueue_wait"][0], 0.020)
        self.assertAlmostEqual(samples["consumer_total"][0], 0.032)
        self.assertEqual(len(samples["consumer_dequeue_wait"]), 2)
        self.assertEqual(result[0], 1)
        self.assertAlmostEqual(result[1], 0.004)
        self.assertEqual(result[2], {0: 1})
        self.assertEqual(result[5], {0: 1})
        self.assertEqual(result[6], 1)
        record = result[4][0]
        self.assertEqual(record[0], 0)
        self.assertAlmostEqual(record[1], 0.7310586, places=6)
        np.testing.assert_array_equal(record[2], [True, True])
        self.assertEqual(record[3:], ("image.jpg", 0))
        self.assertEqual(output.get()[0], "image.jpg")

    def test_consumer_total_includes_progress_update(self):
        profiler, result, _ = self.run_consumer(
            enabled=True, draw=False, save=False, accuracy=False,
            progress_seconds=0.007,
        )
        self.assertAlmostEqual(profiler._samples['consumer_total'][0], 0.011)
        self.assertAlmostEqual(result[1], 0.004)
        self.assertEqual(result[0], 1)

    def test_optional_work_is_absent_from_profile(self):
        profiler, result, output = self.run_consumer(
            enabled=True, draw=False, save=False, accuracy=False
        )
        self.assertFalse(
            {"accuracy", "draw_boxes", "writer_enqueue_wait"}
            & profiler._samples.keys()
        )
        self.assertAlmostEqual(profiler._samples["consumer_total"][0], 0.004)
        self.assertEqual(result[0], 1)
        self.assertIsNone(output)

    def test_disabled_profiling_keeps_inference_without_samples(self):
        profiler, result, _ = self.run_consumer(
            enabled=False, draw=False, save=False, accuracy=False
        )
        self.assertEqual(profiler._samples, {})
        self.assertEqual(result[0], 1)
        self.assertAlmostEqual(result[1], 0.004)


@unittest.skipIf(np is None, "NumPy is required")
class TestDetectionProfileOutput(unittest.TestCase):
    def test_json_preserves_per_worker_timings_and_run_scope(self):
        for power_available in (True, False):
            with self.subTest(power_available=power_available):
                self.check_output(power_available)

    def test_idle_power_fallback_is_explicit_in_text(self):
        self.check_output(True, sampled=False)

    def check_output(self, power_available, sampled=True):
        from kria_ai.yolov26.detection.config import get_dataset, get_model

        model = replace(
            get_model("yolov26n_FaceHuman"), input_size=(32, 32), strides=(8,)
        )
        dataset = get_dataset("openimagesv7")
        metadata = types.SimpleNamespace(
            inputs=(types.SimpleNamespace(shape=(1, 32, 32, 3), fixed_point=7),),
            outputs=(
                types.SimpleNamespace(shape=(1, 1, 1, 4), fixed_point=0),
                types.SimpleNamespace(shape=(1, 1, 1, 2), fixed_point=0),
            ),
        )
        outputs = [
            np.ones((1, 1, 1, 4), dtype=np.int8),
            np.array([[[[1, -10]]]], dtype=np.int8),
        ]
        cv2 = types.ModuleType("cv2")
        cv2.COLOR_BGR2RGB = 1
        cv2.imread = lambda _: np.zeros((32, 32, 3), dtype=np.uint8)
        cv2.cvtColor = lambda image, _: image
        cv2.resize = lambda image, *_: image
        cv2.copyMakeBorder = lambda image, *_: image
        runner = types.SimpleNamespace(
            execute_async=lambda *_: 1, wait=lambda _: None
        )

        class Monitor:
            def __init__(self, **_):
                self.samples = [5.0] if power_available and sampled else []

            def start(self):
                pass

            def stop(self):
                pass

            def average(self, fallback):
                return 5.0 if self.samples else fallback

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict("sys.modules", {"cv2": cv2}),
            patch.object(benchmark, "get_model", return_value=model),
            patch.object(benchmark, "get_dataset", return_value=dataset),
            patch.object(benchmark, "load_dpu_model", return_value=metadata),
            patch.object(benchmark, "create_runner", return_value=runner),
            patch.object(benchmark, "allocate_output_buffers", return_value=outputs),
            patch.object(benchmark, "build_normalization_lut", return_value=None),
            patch.object(
                benchmark, "preprocess_board_image",
                return_value=np.zeros((1, 32, 32, 3), dtype=np.int8),
            ),
            patch.object(benchmark, "PowerMonitor", Monitor),
            patch.object(
                benchmark, "read_power_mw",
                return_value=5000.0 if power_available else 0.0,
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            root = Path(directory)
            for name in ("a.jpg", "b.jpg"):
                (root / name).touch()
            report = root / "report.txt"
            self.assertEqual(benchmark.main([
                "--dataset-root", str(root), "--xmodel", "fake.xmodel",
                "--threads", "1", "--producers", "4", "--queue-size", "3",
                "--no-draw", "--no-save", "--confidence-threshold", "0.3",
                "--profile", "--profile-json", "--output-report", str(report),
            ]), 0)
            payload = json.loads((root / "report_profile.json").read_text())
            self.assertEqual(payload["images_processed"], 2)
            self.assertEqual(payload["detections"], 2)
            self.assertEqual(payload["class_histogram"], {"0": 2})
            self.assertGreater(payload["wall_time_s"], 0.0)
            self.assertGreater(payload["fps"], 0.0)
            self.assertEqual(payload["board"], "kv260")
            self.assertEqual(payload["producers"], 4)
            self.assertEqual(payload["active_producers"], 2)
            self.assertEqual(payload["queue_size"], 3)
            self.assertFalse(payload["draw_outputs"])
            self.assertFalse(payload["save_outputs"])
            self.assertFalse(payload["accuracy_enabled"])
            self.assertEqual(payload["items_discovered"], 2)
            self.assertEqual(payload["warmup_images"], 0)
            self.assertTrue(payload["timing_notes"])
            self.assertEqual(payload["xmodel"], "fake.xmodel")
            workers = payload["workers"]
            self.assertEqual(len(workers), 3)
            producers = [w for w in workers if w["role"] == "producer"]
            consumers = [w for w in workers if w["role"] == "consumer"]
            self.assertEqual(sum(w["images_processed"] for w in producers), 2)
            self.assertEqual(len(consumers), 1)
            self.assertEqual(consumers[0]["images_processed"], 2)
            stages = {row["stage"]: row for row in payload["stages"]}
            self.assertEqual(stages["image_read"]["count"], 2)
            self.assertEqual(stages["preprocess"]["count"], 2)
            self.assertEqual(stages["decode_ultra_threshold"]["count"], 2)
            self.assertNotIn("draw_boxes", stages)
            self.assertNotIn("write", stages)
            worker_stages = {
                row["stage"]: row for row in consumers[0]["stages"]
            }
            self.assertEqual(worker_stages["runner_setup"]["count"], 1)
            self.assertEqual(worker_stages["consumer_total"]["count"], 2)
            self.assertEqual(payload["power_available"], power_available)
            if power_available:
                self.assertEqual(payload["power_w"], 5.0)
                self.assertEqual(
                    payload["power_source"],
                    "runtime_samples" if sampled else "idle_fallback",
                )
                self.assertAlmostEqual(
                    payload["energy_mj"],
                    2500.0 * payload["wall_time_s"],
                )
                if not sampled:
                    self.assertIn("idle fallback", report.read_text())
            else:
                self.assertIsNone(payload["power_w"])
                self.assertIsNone(payload["energy_mj"])
                self.assertEqual(payload["power_source"], "unavailable")
                self.assertIn("N/A", report.read_text())


if __name__ == "__main__":
    unittest.main()
