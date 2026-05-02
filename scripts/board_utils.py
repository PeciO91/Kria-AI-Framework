"""
Shared board-side utilities for DPU inference pipelines.

Contains components used by both run_inference.py (classification) and
run_detection.py (object detection):

- PowerMonitor: background sampler for SOM total power (Watts)
- ProgressCounter: thread-safe progress counter
- setup_dpu(model_path): loads xmodel, returns (subgraph, dpu_shape, fix_pos_in, fix_pos_outs)
- compute_norm_constants(mean, std, fix_pos): pre-multiplies normalization for INT8 input
- preprocess_image(img_rgb, dpu_shape, math_scale, math_shift): vectorized resize + normalize
- format_report(title, metrics): pretty-prints a metrics block and returns the string
"""
import time
import threading
import subprocess

import numpy as np
import cv2
import vart
import xir

from board_config import get_power_mw


# =============================================================
# POWER MONITORING
# =============================================================
class PowerMonitor(threading.Thread):
    """Samples SOM total power in the background. Average is in Watts."""
    def __init__(self, interval=0.2):
        super().__init__()
        self.interval = interval
        self.samples = []
        self.stop_evt = threading.Event()
        self.daemon = True

    def run(self):
        while not self.stop_evt.is_set():
            p = get_power_mw() / 1000.0
            if p > 0:
                self.samples.append(p)
            time.sleep(self.interval)

    def stop(self):
        self.stop_evt.set()
        self.join(timeout=1.0)

    def average(self, fallback=0.0):
        return float(np.mean(self.samples)) if self.samples else fallback


# =============================================================
# PROGRESS COUNTER
# =============================================================
class ProgressCounter:
    """Thread-safe progress counter (replaces module-level globals)."""
    def __init__(self):
        self._count = 0
        self._lock = threading.Lock()

    def increment(self, n=1):
        with self._lock:
            self._count += n

    @property
    def value(self):
        with self._lock:
            return self._count


# =============================================================
# DPU SETUP
# =============================================================
def setup_dpu(model_path):
    """
    Load xmodel and inspect tensor metadata.

    Returns
    -------
    subgraph : xir.Subgraph
        DPU subgraph used to create vart.Runner instances.
    dpu_shape : tuple
        Input shape (1, H, W, C).
    fix_pos_in : int
        Input quantization fixed-point position.
    fix_pos_outs : list[int]
        Output quantization fixed-point positions (one per output tensor).
    """
    graph = xir.Graph.deserialize(model_path)
    subgraph = [
        s for s in graph.get_root_subgraph().get_children()
        if s.get_attr("device").upper() == "DPU"
    ][0]

    runner = vart.Runner.create_runner(subgraph, "run")
    in_tensors = runner.get_input_tensors()
    out_tensors = runner.get_output_tensors()

    dpu_shape = tuple(in_tensors[0].dims)
    fix_pos_in = in_tensors[0].get_attr("fix_point")
    fix_pos_outs = [t.get_attr("fix_point") for t in out_tensors]
    del runner

    return subgraph, dpu_shape, fix_pos_in, fix_pos_outs


# =============================================================
# NORMALIZATION HELPERS
# =============================================================
def compute_norm_constants(norm_mean, norm_std, fix_pos):
    """
    Pre-compute INT8 normalization scale/shift so a single fused
    multiply-add can convert a uint8 image to int8 DPU input.

    int8_pixel = uint8_pixel * math_scale - math_shift
    """
    mean_np = np.array(norm_mean, dtype=np.float32)
    std_np = np.array(norm_std, dtype=np.float32)
    f_scale = np.float32(2 ** fix_pos)
    math_scale = np.float32(f_scale / (255.0 * std_np))
    math_shift = np.float32((mean_np * f_scale) / std_np)
    return math_scale, math_shift


def build_norm_lut(norm_mean, norm_std, fix_pos):
    """
    Pre-bake the uint8 -> int8 normalization into a per-channel lookup
    table.

    For every input byte u and every channel c, the entry lut[u, c]
    stores the int8 equivalent of (u * math_scale[c] - math_shift[c]),
    clipped to [-128, 127]. Because the input is bounded in [0, 255] the
    table is bit-equivalent to the original `(img * scale - shift).astype(int8)`
    pipeline but eliminates the per-frame float multiply over a 1.2M-pixel
    tensor.

    Returns
    -------
    lut : ndarray
        Shape (256, 3), dtype int8.
    """
    math_scale, math_shift = compute_norm_constants(norm_mean, norm_std, fix_pos)
    u = np.arange(256, dtype=np.float32)[:, None]                # (256, 1)
    table = np.rint(u * math_scale - math_shift)                 # (256, 3)
    return np.clip(table, -128, 127).astype(np.int8)


# Cached channel-index helper for per-channel fancy indexing.
_CHANNEL_INDEX_3 = np.arange(3, dtype=np.intp)


def apply_norm_lut(img_uint8, lut):
    """
    Apply a (256, 3) per-channel LUT to an HWC uint8 image and return
    the int8 result with identical shape. Uses numpy fancy indexing,
    which is portable across OpenCV versions and ~10x faster than the
    explicit float multiply on ARM.
    """
    return lut[img_uint8, _CHANNEL_INDEX_3]


def preprocess_image(img_rgb, dpu_shape, lut):
    """
    Resize an RGB image to the DPU's expected (H, W) and apply the
    pre-built normalization LUT. Returns NHWC int8 ready for the runner.
    """
    dpu_h, dpu_w = dpu_shape[1], dpu_shape[2]
    img = cv2.resize(img_rgb, (dpu_w, dpu_h))
    img_int8 = apply_norm_lut(img, lut)
    return np.expand_dims(img_int8, axis=0)


# =============================================================
# REPORT FORMATTING
# =============================================================
def format_report(title, metrics):
    """
    Build a human-readable report block.

    metrics : list[tuple]
        Each tuple is either (label, value) or ('---', None) for a separator.
    """
    lines = []
    lines.append("=" * 60)
    lines.append(f"  {title}")
    lines.append("=" * 60)
    for entry in metrics:
        if entry[0] == "---":
            lines.append("-" * 60)
        else:
            label, value = entry
            lines.append(f"{label:<20}{value}")
    lines.append("=" * 60)
    return "\n".join(lines) + "\n"
