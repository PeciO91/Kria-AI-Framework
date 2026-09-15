from __future__ import annotations

import copy
from typing import Any

import torch
from torch import nn


class C3k2DPU(nn.Module):
    def __init__(self, old: Any, conv_type: type[nn.Module]):
        super().__init__()
        self.c = old.c
        old_cv1 = old.cv1
        old_conv = old_cv1.conv
        old_bn = old_cv1.bn
        channels = int(old.c)

        if old_conv.out_channels != 2 * channels:
            raise ValueError(
                f"C3k2 cv1 has {old_conv.out_channels} outputs; expected {2 * channels}"
            )
        if old_conv.bias is not None:
            raise ValueError("C3k2DPU requires a bias-free cv1 convolution")
        if not isinstance(old_bn, nn.BatchNorm2d):
            raise TypeError("C3k2DPU requires an unfused BatchNorm2d cv1")

        def _scalar(value):
            if isinstance(value, (tuple, list)):
                return value[0]
            return value

        conv_kwargs = {
            "k": _scalar(old_conv.kernel_size),
            "s": _scalar(old_conv.stride),
            "p": _scalar(old_conv.padding),
            "g": _scalar(old_conv.groups),
            "d": _scalar(old_conv.dilation),
        }
        self.cv1a = conv_type(
            old_conv.in_channels,
            channels,
            act=copy.deepcopy(old_cv1.act),
            **conv_kwargs,
        )
        self.cv1b = conv_type(
            old_conv.in_channels,
            channels,
            act=copy.deepcopy(old_cv1.act),
            **conv_kwargs,
        )
        device = old_conv.weight.device
        dtype = old_conv.weight.dtype
        self.cv1a.to(device=device, dtype=dtype)
        self.cv1b.to(device=device, dtype=dtype)

        for branch in (self.cv1a, self.cv1b):
            branch.bn.eps = old_bn.eps
            branch.bn.momentum = old_bn.momentum

        with torch.no_grad():
            for branch, start in ((self.cv1a, 0), (self.cv1b, channels)):
                stop = start + channels
                branch.conv.weight.copy_(old_conv.weight[start:stop])
                branch.bn.weight.copy_(old_bn.weight[start:stop])
                branch.bn.bias.copy_(old_bn.bias[start:stop])
                branch.bn.running_mean.copy_(old_bn.running_mean[start:stop])
                branch.bn.running_var.copy_(old_bn.running_var[start:stop])
                branch.bn.num_batches_tracked.copy_(old_bn.num_batches_tracked)

        self.cv1a.conv.weight.requires_grad_(old_conv.weight.requires_grad)
        self.cv1b.conv.weight.requires_grad_(old_conv.weight.requires_grad)
        for branch in (self.cv1a, self.cv1b):
            branch.bn.weight.requires_grad_(old_bn.weight.requires_grad)
            branch.bn.bias.requires_grad_(old_bn.bias.requires_grad)

        self.m = old.m
        self.cv2 = old.cv2
        for attribute in ("i", "f", "type", "np"):
            if hasattr(old, attribute):
                setattr(self, attribute, getattr(old, attribute))
        self.train(old.training)

    def forward(self, inputs):
        values = [self.cv1a(inputs), self.cv1b(inputs)]
        for block in self.m:
            values.append(block(values[-1]))
        return self.cv2(torch.cat(values, dim=1))


__all__ = ["C3k2DPU"]
