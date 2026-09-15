"""PyTorch-gated tests for YOLOv26 quantization helpers."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import torch
    from torch import nn

    from kria_ai.yolov26.quantize import _forward_loop

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required")
class TestForwardLoop(unittest.TestCase):
    def test_forward_loop_is_eval_no_grad_and_counts_samples(self):
        records = []

        class Recorder(nn.Module):
            def forward(self, inputs):
                records.append((self.training, torch.is_grad_enabled(), inputs.size(0)))
                return inputs

        model = Recorder()
        batches = [torch.randn(2, 3, 8, 8), (torch.randn(1, 3, 8, 8), "labels")]

        processed = _forward_loop(model, batches, torch.device("cpu"))

        self.assertEqual(processed, 3)
        self.assertEqual(len(records), 2)
        self.assertTrue(all(not training for training, _, _ in records))
        self.assertTrue(all(not grad for _, grad, _ in records))
        self.assertEqual([size for _, _, size in records], [2, 1])


if __name__ == "__main__":
    unittest.main()
