import os
import sys

import torch
import torch.nn as nn


class UltralyticsVitisDetect(nn.Module):
    def __init__(self, weights_path, repo_root, head_variant="one2one",
                 replace_leaky_relu=False):
        super().__init__()
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from ultralytics import YOLO

        yolo = YOLO(weights_path)
        self.model = yolo.model
        self.head_variant = head_variant
        if replace_leaky_relu:
            self._replace_leaky_relu(self.model)
        self.names = getattr(self.model, "names", None)
        self.yaml = getattr(self.model, "yaml", None)
        self.stride = getattr(self.model, "stride", None)
        self.model.eval()

    def _replace_leaky_relu(self, module):
        for name, child in module.named_children():
            if isinstance(child, nn.LeakyReLU):
                setattr(module, name, nn.ReLU(inplace=getattr(child, "inplace", False)))
            else:
                self._replace_leaky_relu(child)

    def forward(self, x):
        y = []
        for m in self.model.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            if m is self.model.model[-1] and hasattr(m, "one2one") and hasattr(m, "one2many"):
                head = m.one2one if self.head_variant == "one2one" and m.end2end else m.one2many
                return tuple(
                    torch.cat((head["box_head"][i](x[i]), head["cls_head"][i](x[i])), dim=1)
                    for i in range(m.nl)
                )
            x = m(x)
            y.append(x if m.i in self.model.save else None)
        return x
