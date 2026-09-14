"""PyTorch-gated tests for YOLOv26 slim-checkpoint reconstruction."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import torch
    from torch import nn

    from kria_ai.yolov26.models import load_slim_state_dict

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required")
class TestLoadSlimStateDict(unittest.TestCase):
    def test_shrinks_depthwise_conv_and_batchnorm(self):
        model = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, padding=1, groups=3),
            nn.BatchNorm2d(3),
        )
        state_dict = {
            "0.weight": torch.randn(2, 1, 3, 3),
            "0.bias": torch.randn(2),
            "1.weight": torch.randn(2),
            "1.bias": torch.randn(2),
            "1.running_mean": torch.zeros(2),
            "1.running_var": torch.ones(2),
            "1.num_batches_tracked": torch.tensor(0),
        }

        result = load_slim_state_dict(model, state_dict)

        conv = result[0]
        self.assertEqual(conv.groups, 2)
        self.assertEqual(conv.in_channels, 2)
        self.assertEqual(conv.out_channels, 2)
        self.assertEqual(result[1].num_features, 2)

        output = result(torch.randn(1, 2, 4, 4))
        self.assertEqual(tuple(output.shape), (1, 2, 4, 4))


if __name__ == "__main__":
    unittest.main()
