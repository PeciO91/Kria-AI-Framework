"""Detection pipeline lifecycle: OpenCV startup ordering and worker
failure propagation, exercised through the real main() and worker
threads with only external cv2/VART/power faked."""

import contextlib
import io
import json
import multiprocessing
import queue
import sys
import tempfile
import threading
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from kria_ai.yolov26.detection import benchmark

try:
    import numpy as np
except ImportError:
    np = None


def _model_config():
    from kria_ai.yolov26.detection.config import get_model

    return replace(
        get_model("yolov26n_FaceHuman"), input_size=(32, 32), strides=(8,)
    )


def _dataset_config():
    from kria_ai.yolov26.detection.config import get_dataset

    return get_dataset("openimagesv7")


def _metadata():
    return types.SimpleNamespace(
        inputs=(types.SimpleNamespace(shape=(1, 32, 32, 3), fixed_point=7),),
        outputs=(
            types.SimpleNamespace(shape=(1, 1, 1, 4), fixed_point=0),
            types.SimpleNamespace(shape=(1, 1, 1, 2), fixed_point=0),
        ),
    )


def _outputs():
    return [
        np.ones((1, 1, 1, 4), dtype=np.int8),
        np.array([[[[1, -10]]]], dtype=np.int8),
    ]


def _make_cv2(**overrides):
    cv2 = types.ModuleType("cv2")
    cv2.COLOR_BGR2RGB = 1
    cv2.FONT_HERSHEY_SIMPLEX = 0
    cv2.imread = lambda _: np.zeros((32, 32, 3), dtype=np.uint8)
    cv2.cvtColor = lambda image, _: image
    cv2.resize = lambda image, *_: image
    cv2.copyMakeBorder = lambda image, *_: image
    cv2.rectangle = lambda *_, **__: None
    cv2.putText = lambda *_, **__: None
    cv2.imwrite = lambda *_: True
    for name, value in overrides.items():
        if value is None:
            delattr(cv2, name)
        else:
            setattr(cv2, name, value)
    return cv2


class _Monitor:
    stopped = False

    def __init__(self, **_):
        self.samples = [5.0]

    def start(self):
        pass

    def stop(self):
        type(self).stopped = True

    def average(self, fallback):
        return 5.0 if self.samples else fallback


def _argv(root, report, annotated, extra=()):
    return [
        "--dataset-root", str(root), "--xmodel", "fake.xmodel",
        "--threads", "1", "--producers", "1", "--queue-size", "2",
        "--confidence-threshold", "0.3",
        "--profile", "--profile-json",
        "--output-report", str(report), "--output-dir", str(annotated),
        *extra,
    ]


def _patches(cv2, monitor=_Monitor, runner=None, create_runner=None):
    if create_runner is None:
        if runner is None:
            runner = types.SimpleNamespace(
                execute_async=lambda *_: 1, wait=lambda _: None
            )
        create_runner = lambda _: runner
    real_queue = queue.Queue
    return [
        patch.dict("sys.modules", {"cv2": cv2}),
        patch.object(benchmark, "get_model", return_value=_model_config()),
        patch.object(benchmark, "get_dataset", return_value=_dataset_config()),
        patch.object(benchmark, "load_dpu_model", return_value=_metadata()),
        patch.object(benchmark, "create_runner", side_effect=create_runner),
        patch.object(
            benchmark, "allocate_output_buffers", return_value=_outputs()
        ),
        patch.object(
            benchmark, "output_dequantization_scales", return_value=(1, 1)
        ),
        patch.object(benchmark, "build_normalization_lut", return_value=None),
        patch.object(
            benchmark,
            "preprocess_board_image",
            return_value=np.zeros((1, 32, 32, 3), dtype=np.int8),
        ),
        patch.object(benchmark, "PowerMonitor", monitor),
        patch.object(benchmark, "read_power_mw", return_value=5000.0),
        patch.object(
            benchmark.queue, "Queue", lambda maxsize=0: real_queue(maxsize=1)
        ),
    ]


def _write_images(root, count=6):
    for index in range(count):
        (root / f"image_{index}.jpg").touch()


def _child_run(root, argv, fail, gated_cv2, result_queue):
    cv2_kwargs = {}
    if fail == "imread_raise":
        def _raise(_):
            raise RuntimeError("read failed")
        cv2_kwargs["imread"] = _raise
    elif fail == "imwrite_raise":
        def _raise_write(*_):
            raise RuntimeError("write failed")
        cv2_kwargs["imwrite"] = _raise_write
    elif fail == "imwrite_false":
        cv2_kwargs["imwrite"] = lambda *_: False

    if gated_cv2:
        impls = _make_cv2(**cv2_kwargs)

        class GatedCv2(types.ModuleType):
            def __init__(self):
                super().__init__("cv2")
                self._impls = {
                    name: getattr(impls, name)
                    for name in dir(impls)
                    if not name.startswith("__")
                }
                self._unlocked = False

            def __getattr__(self, name):
                impl = self._impls.get(name)
                if impl is None:
                    raise AttributeError(name)
                if not self._unlocked:
                    if threading.current_thread() is threading.main_thread():
                        self._unlocked = True
                    else:
                        raise AttributeError(name)
                return impl

        cv2 = GatedCv2()
    else:
        cv2 = _make_cv2(**cv2_kwargs)

    runner = None
    create_runner = None
    if fail == "runner_setup":
        def _raise_runner(_):
            raise RuntimeError("runner setup failed")
        create_runner = _raise_runner
    elif fail == "dpu_wait":
        def _raise_wait(_):
            raise RuntimeError("dpu failed")
        runner = types.SimpleNamespace(
            execute_async=lambda *_: 1, wait=_raise_wait
        )

    _Monitor.stopped = False
    patches = _patches(
        cv2, runner=runner, create_runner=create_runner
    )
    live = []
    try:
        with patches[0], patches[1], patches[2], patches[3], patches[4], \
                patches[5], patches[6], patches[7], patches[8], patches[9], \
                patches[10], patches[11]:
            with contextlib.redirect_stdout(io.StringIO()):
                code = benchmark.main(argv)
        live = [
            t.name for t in threading.enumerate()
            if t.name.startswith("detection-") and t.is_alive()
        ]
        result_queue.put({
            "outcome": "ok", "code": code,
            "live_threads": live, "monitor_stopped": _Monitor.stopped,
        })
    except BaseException as exc:
        live = [
            t.name for t in threading.enumerate()
            if t.name.startswith("detection-") and t.is_alive()
        ]
        result_queue.put({
            "outcome": "error",
            "detail": f"{type(exc).__name__}: {exc}",
            "live_threads": live, "monitor_stopped": _Monitor.stopped,
        })
        result_queue.close()
        result_queue.join_thread()
        raise SystemExit(1)
    result_queue.close()
    result_queue.join_thread()


def _run_in_child(root, argv, fail=None, gated_cv2=False, timeout=5.0):
    context = multiprocessing.get_context("fork")
    result_queue = context.Queue()
    process = context.Process(
        target=_child_run, args=(root, argv, fail, gated_cv2, result_queue)
    )
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join(2)
        raise AssertionError("benchmark.main hung in child process")
    result = result_queue.get() if not result_queue.empty() else {}
    result["exitcode"] = process.exitcode
    return result


@unittest.skipIf(np is None, "NumPy is required")
class TestOpenCvInitialization(unittest.TestCase):
    def test_opencv_initialized_on_main_thread_before_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_images(root)
            report = root / "report.txt"
            annotated = root / "annotated"
            result = _run_in_child(
                root, _argv(root, report, annotated), gated_cv2=True
            )
            self.assertEqual(
                result["outcome"], "ok", result.get("detail", result)
            )
            self.assertEqual(result["code"], 0)
            payload = json.loads(
                (root / "report_profile.json").read_text()
            )
            self.assertEqual(payload["images_processed"], 6)
            self.assertEqual(payload["items_discovered"], 6)
            self.assertGreater(payload["detections"], 0)

    def test_missing_imread_fails_before_model_load(self):
        cv2 = _make_cv2(imread=None)
        loader = patch.object(
            benchmark, "load_dpu_model", return_value=_metadata()
        )
        patches = _patches(cv2)
        patches[3] = loader
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_images(root)
            report = root / "report.txt"
            with patches[0], patches[1], patches[2], loader as mocked, \
                    patches[4], patches[5], patches[6], patches[7], \
                    patches[8], patches[9], patches[10], patches[11]:
                with self.assertRaises(RuntimeError) as caught:
                    benchmark.main(
                        _argv(root, report, root / "annotated")
                    )
            self.assertIn("imread", str(caught.exception))
            mocked.assert_not_called()
            self.assertFalse(report.exists())


@unittest.skipIf(np is None, "NumPy is required")
class TestWorkerFailurePropagation(unittest.TestCase):
    def _run_failure(self, fail):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_images(root)
            report = root / "report.txt"
            result = _run_in_child(
                root, _argv(root, report, root / "annotated"), fail=fail
            )
            report_exists = report.exists()
            profile_exists = (root / "report_profile.json").exists()
        return result, report_exists, profile_exists

    def _check_failure(self, fail, worker_label, cause):
        result, report_exists, profile_exists = self._run_failure(fail)
        self.assertEqual(
            result["outcome"], "error",
            f"expected worker failure, got {result}",
        )
        self.assertEqual(result["exitcode"], 1)
        detail = result["detail"]
        self.assertIn("RuntimeError", detail)
        self.assertIn(worker_label, detail)
        self.assertIn(cause, detail)
        self.assertFalse(report_exists, "success report written after failure")
        self.assertFalse(
            profile_exists, "profile JSON written after failure"
        )
        self.assertTrue(
            result["monitor_stopped"], "power monitor was not stopped"
        )
        self.assertEqual(result["live_threads"], [])

    def test_producer_imread_failure_propagates(self):
        self._check_failure("imread_raise", "image producer", "read failed")

    def test_consumer_runner_setup_failure_propagates(self):
        self._check_failure(
            "runner_setup", "DPU consumer", "runner setup failed"
        )

    def test_consumer_dpu_failure_propagates(self):
        self._check_failure("dpu_wait", "DPU consumer", "dpu failed")

    def test_writer_imwrite_failure_propagates(self):
        self._check_failure("imwrite_raise", "image writer", "write failed")

    def test_writer_imwrite_false_propagates(self):
        self._check_failure(
            "imwrite_false", "image writer", "Failed to write detection image"
        )


@unittest.skipIf(np is None, "NumPy is required")
class TestZeroAndPartialProcessing(unittest.TestCase):
    def test_zero_processed_images_is_error(self):
        cv2 = _make_cv2(imread=lambda _: None)
        patches = _patches(cv2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_images(root)
            report = root / "report.txt"
            with patches[0], patches[1], patches[2], patches[3], patches[4], \
                    patches[5], patches[6], patches[7], patches[8], \
                    patches[9], patches[10], patches[11]:
                with self.assertRaises(RuntimeError) as caught:
                    benchmark.main(_argv(root, report, root / "annotated"))
            self.assertIn("No images were processed", str(caught.exception))
            self.assertFalse(report.exists())

    def test_one_unreadable_image_still_reports(self):
        state = {"calls": 0}

        def _imread(_):
            state["calls"] += 1
            if state["calls"] == 2:
                return None
            return np.zeros((32, 32, 3), dtype=np.uint8)

        cv2 = _make_cv2(imread=_imread)
        patches = _patches(cv2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_images(root)
            report = root / "report.txt"
            with patches[0], patches[1], patches[2], patches[3], patches[4], \
                    patches[5], patches[6], patches[7], patches[8], \
                    patches[9], patches[10], patches[11]:
                with contextlib.redirect_stdout(io.StringIO()):
                    code = benchmark.main(
                        _argv(root, report, root / "annotated")
                    )
            self.assertEqual(code, 0)
            payload = json.loads(
                (root / "report_profile.json").read_text()
            )
            self.assertEqual(payload["images_processed"], 5)
            self.assertEqual(payload["items_discovered"], 6)


if __name__ == "__main__":
    unittest.main()
