"""PyTorch-gated tests for YOLOv26 slim-checkpoint reconstruction."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import torch
    from torch import nn

    from kria_ai.yolov26.dpu import C3k2DPU
    from kria_ai.yolov26.models import (
        apply_yolov26_activation_policy,
        apply_yolov26_graph_policy,
        import_local_ultralytics,
        load_slim_state_dict,
    )

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


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required")
class TestActivationPolicy(unittest.TestCase):
    def test_leaky_conv_relu_dw_policy(self):
        import importlib

        import_local_ultralytics("models/ultralytics-main")
        ultralytics_modules = importlib.import_module("ultralytics.nn.modules")

        model = nn.Sequential(
            ultralytics_modules.Conv(3, 8, 3),
            ultralytics_modules.DWConv(8, 8, 3),
            ultralytics_modules.Conv(8, 8, 1, act=False),
        )

        result, conv_count, dwconv_count = apply_yolov26_activation_policy(
            model, "leaky_13_128_conv_relu_dw"
        )

        self.assertIs(result, model)
        self.assertEqual(conv_count, 1)
        self.assertEqual(dwconv_count, 1)
        self.assertIsInstance(model[0].act, nn.LeakyReLU)
        self.assertEqual(model[0].act.negative_slope, 13.0 / 128.0)
        self.assertIsInstance(model[1].act, nn.ReLU)
        self.assertIsInstance(model[2].act, nn.Identity)

    def test_none_policy_is_noop(self):
        model = nn.Sequential(nn.Conv2d(3, 8, 3))

        result, conv_count, dwconv_count = apply_yolov26_activation_policy(model, None)

        self.assertIs(result, model)
        self.assertEqual(conv_count, 0)
        self.assertEqual(dwconv_count, 0)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required")
class TestGraphPolicy(unittest.TestCase):
    def test_split_c3k2_cv1_is_numerically_equivalent(self):
        import importlib

        import_local_ultralytics("models/ultralytics-main")
        ultralytics_modules = importlib.import_module("ultralytics.nn.modules")

        torch.manual_seed(42)
        c3k2 = ultralytics_modules.C3k2(16, 16, n=2, c3k=True)
        model = nn.Sequential(c3k2)
        model.eval()

        inputs = torch.randn(1, 16, 20, 20)
        with torch.no_grad():
            expected = model(inputs)

        result, replacement_count = apply_yolov26_graph_policy(model, "split_c3k2_cv1")

        self.assertIs(result, model)
        self.assertEqual(replacement_count, 1)
        self.assertIsInstance(model[0], C3k2DPU)
        self.assertFalse(
            any(isinstance(module, ultralytics_modules.C3k2) for module in model.modules())
        )

        with torch.no_grad():
            actual = model(inputs)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

        traced = torch.jit.trace(model[0], inputs)
        graph = traced.inlined_graph
        for operator in ("aten::chunk", "aten::split", "aten::slice"):
            self.assertNotIn(operator, str(graph))

    def test_none_policy_is_noop(self):
        model = nn.Sequential(nn.Conv2d(3, 8, 3))

        result, replacement_count = apply_yolov26_graph_policy(model, None)

        self.assertIs(result, model)
        self.assertEqual(replacement_count, 0)


if __name__ == "__main__":
    unittest.main()
