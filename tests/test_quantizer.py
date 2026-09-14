"""Quantizer lifecycle tests using a fake torch module and quantizer factory."""

import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kria_ai.common.vitis.quantizer import run_quantization


class _NoGrad:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class _FakeQuantModel:
    def __init__(self):
        self.forwards = 0
        self.was_eval = False

    def eval(self):
        self.was_eval = True

    def __call__(self, *args, **kwargs):
        self.forwards += 1


class ToyModel:
    pass


class _FakeQuantizer:
    def __init__(self, model, output_dir):
        self.quant_model = _FakeQuantModel()
        self._model = model
        self._output_dir = Path(output_dir)

    def export_quant_config(self):
        (self._output_dir / "quant_info.json").write_text("{}")

    def export_xmodel(self, output_dir, deploy_check=False):
        Path(output_dir, f"{self._model.__class__.__name__}_int.xmodel").write_text("xmodel")


class _FakeTorch(types.ModuleType):
    def no_grad(self):
        return _NoGrad()


def _fake_torch():
    module = _FakeTorch("torch")
    return module


def _adapt_batch(batch, device):
    return (batch,), {}, 1


class TestQuantizerLifecycle(unittest.TestCase):
    def setUp(self):
        self._previous_torch = sys.modules.get("torch")
        sys.modules["torch"] = _fake_torch()
        self.addCleanup(self._restore_torch)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.output_dir = Path(self._tmp.name)

    def _restore_torch(self):
        if self._previous_torch is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = self._previous_torch

    def _factory(self, calls):
        def factory(mode, model, example_inputs, device=None, output_dir=None, target=None):
            calls.append(
                {
                    "mode": mode,
                    "model": model,
                    "example_inputs": example_inputs,
                    "device": device,
                    "output_dir": output_dir,
                    "target": target,
                }
            )
            return _FakeQuantizer(model, output_dir)

        return factory

    def test_calib_mode_writes_manifest(self):
        calls = []
        result = run_quantization(
            mode="calib",
            model=ToyModel(),
            example_inputs=(object(),),
            batches=["batch-1", "batch-2"],
            adapt_batch=_adapt_batch,
            device="cpu",
            output_dir=self.output_dir,
            target="fingerprint",
            quantizer_factory=self._factory(calls),
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["mode"], "calib")
        self.assertEqual(result.quant_model.forwards, 2)
        self.assertTrue(result.quant_model.was_eval)
        self.assertEqual(result.processed_samples, 2)
        self.assertEqual(result.quant_config_path, self.output_dir / "quant_info.json")
        self.assertTrue(result.quant_config_path.is_file())
        self.assertIsNone(result.xmodel_path)

        manifest_path = self.output_dir / "manifest.json"
        self.assertEqual(result.manifest_path, manifest_path)
        self.assertTrue(manifest_path.is_file())
        payload = json.loads(manifest_path.read_text())
        self.assertEqual(
            payload,
            {
                "stage": "quantization",
                "mode": "calib",
                "processed_samples": 2,
                "target": "fingerprint",
                "quant_config": str(self.output_dir / "quant_info.json"),
                "xmodel": None,
            },
        )

    def test_test_mode_renames_xmodel_and_writes_manifest(self):
        calls = []
        result = run_quantization(
            mode="test",
            model=ToyModel(),
            example_inputs=(object(),),
            batches=["batch-1"],
            adapt_batch=_adapt_batch,
            device="cpu",
            output_dir=self.output_dir,
            xmodel_filename="toy_int.xmodel",
            target="fingerprint",
            quantizer_factory=self._factory(calls),
        )

        self.assertEqual(result.quant_model.forwards, 1)
        self.assertEqual(result.processed_samples, 1)
        self.assertIsNone(result.quant_config_path)
        expected_xmodel = self.output_dir / "toy_int.xmodel"
        self.assertEqual(result.xmodel_path, expected_xmodel)
        self.assertTrue(expected_xmodel.is_file())
        self.assertFalse((self.output_dir / "ToyModel_int.xmodel").exists())

        payload = json.loads(result.manifest_path.read_text())
        self.assertEqual(
            payload,
            {
                "stage": "quantization",
                "mode": "test",
                "processed_samples": 1,
                "target": "fingerprint",
                "quant_config": None,
                "xmodel": str(expected_xmodel),
            },
        )


if __name__ == "__main__":
    unittest.main()
