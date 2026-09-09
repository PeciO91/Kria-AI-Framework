import tempfile
import unittest
from pathlib import Path

from kria_ai.common.artifacts import ArtifactPaths, write_manifest
from kria_ai.common.checkpoints import extract_state_dict, resolve_path


class FakeModel:
    def state_dict(self):
        return {"weight": 1}


class TestArtifactPaths(unittest.TestCase):
    def test_uses_canonical_model_id(self):
        paths = ArtifactPaths("mobilenet_v2", Path("output"))
        self.assertEqual(paths.quantized_xmodel, Path("output/mobilenet_v2/quantize_result/mobilenet_v2_int.xmodel"))
        self.assertEqual(paths.compiled_xmodel, Path("output/mobilenet_v2/compiled/mobilenet_v2_kria.xmodel"))

    def test_rejects_path_like_model_id(self):
        with self.assertRaises(ValueError):
            ArtifactPaths("../model")

    def test_writes_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_manifest(Path(directory) / "stage" / "manifest.json", {"model": "resnet18"})
            self.assertIn('"model": "resnet18"', path.read_text(encoding="utf-8"))


class TestCheckpoints(unittest.TestCase):
    def test_extracts_common_payloads(self):
        self.assertEqual(extract_state_dict({"weight": 1}), {"weight": 1})
        self.assertEqual(extract_state_dict({"state_dict": {"weight": 2}}), {"weight": 2})
        self.assertEqual(extract_state_dict({"model": FakeModel()}), {"weight": 1})

    def test_resolves_relative_path(self):
        self.assertEqual(resolve_path("models/model.pt", "/tmp/project"), Path("/tmp/project/models/model.pt"))


if __name__ == "__main__":
    unittest.main()
