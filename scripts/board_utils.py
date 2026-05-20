"""
Shared board-side utilities for DPU inference pipelines.

Contains components used by run_inference.py (classification),
run_detection.py (object detection), and run_segmentation.py:

- PowerMonitor: background sampler for SOM total power (Watts)
- ProgressCounter: thread-safe progress counter
- StageProfiler / merge_stage_profilers / format_profile_report: per-stage
  timing instrumentation for multi-threaded pipelines
- setup_dpu(model_path): loads xmodel, returns
  (subgraph, dpu_shape, fix_pos_in, fix_pos_outs)
- build_norm_lut(mean, std, fix_pos): builds a uint8 -> int8 normalization LUT
- apply_norm_lut(img_uint8, lut): applies the LUT to an HWC image
- preprocess_image(img_rgb, dpu_shape, lut): resize + normalize, returns NHWC int8
- format_report(title, metrics): pretty-prints a metrics block and returns the string
"""
import time
import threading

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


class StageProfiler:
    def __init__(self, enabled=False):
        self.enabled = enabled
        self._samples = {}

    def add(self, stage, elapsed):
        if not self.enabled:
            return
        self._samples.setdefault(stage, []).append(float(elapsed))

    def time(self, stage):
        return _StageTimer(self, stage)

    def merge(self, other):
        if not self.enabled or other is None:
            return
        for stage, values in other._samples.items():
            self._samples.setdefault(stage, []).extend(values)

    def summary(self, wall_time=None):
        rows = []
        for stage, values in self._samples.items():
            if not values:
                continue
            arr = np.asarray(values, dtype=np.float64)
            total = float(arr.sum())
            row = {
                "stage": stage,
                "count": int(arr.size),
                "total_s": total,
                "avg_ms": float(arr.mean() * 1000.0),
                "p50_ms": float(np.percentile(arr, 50) * 1000.0),
                "p95_ms": float(np.percentile(arr, 95) * 1000.0),
                "min_ms": float(arr.min() * 1000.0),
                "max_ms": float(arr.max() * 1000.0),
            }
            if wall_time and wall_time > 0:
                row["wall_pct"] = float((total / wall_time) * 100.0)
            else:
                row["wall_pct"] = 0.0
            rows.append(row)
        return rows


class _StageTimer:
    def __init__(self, profiler, stage):
        self.profiler = profiler
        self.stage = stage
        self.start = None

    def __enter__(self):
        if self.profiler.enabled:
            self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.profiler.enabled and self.start is not None:
            self.profiler.add(self.stage, time.perf_counter() - self.start)
        return False


def merge_stage_profilers(profilers):
    merged = StageProfiler(enabled=True)
    for profiler in profilers:
        if profiler is not None:
            merged.merge(profiler)
    return merged


def format_profile_report(title, profiler, wall_time, groups):
    summary = {row["stage"]: row for row in profiler.summary(wall_time)}
    lines = []
    lines.append("=" * 96)
    lines.append(f"  {title}")
    lines.append("=" * 96)
    lines.append("Times are accumulated across threads; Wall % can exceed 100% for parallel stages.")
    header = (
        f"{'Stage':<34}{'Count':>8}{'Avg ms':>10}{'P50 ms':>10}"
        f"{'P95 ms':>10}{'Max ms':>10}{'Total s':>10}{'Wall %':>9}"
    )
    lines.append(header)
    lines.append("-" * 96)

    for group_name, stages in groups:
        emitted = False
        group_lines = []
        for stage in stages:
            row = summary.get(stage)
            if row is None:
                continue
            emitted = True
            group_lines.append(
                f"{stage:<34}{row['count']:>8d}{row['avg_ms']:>10.2f}"
                f"{row['p50_ms']:>10.2f}{row['p95_ms']:>10.2f}"
                f"{row['max_ms']:>10.2f}{row['total_s']:>10.3f}"
                f"{row['wall_pct']:>9.1f}"
            )
        if emitted:
            lines.append(f"[{group_name}]")
            lines.extend(group_lines)

    lines.append("=" * 96)
    return "\n".join(lines) + "\n"


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
    Pre-bake the uint8 -> int8 normalization into a lookup table.

    For every input byte u and every channel c, the entry lut[u, c]
    stores the int8 equivalent of (u * math_scale[c] - math_shift[c]),
    clipped to [-128, 127]. Because the input is bounded in [0, 255] the
    table is bit-equivalent to the original `(img * scale - shift).astype(int8)`
    pipeline but eliminates the per-frame float multiply over a 1.2M-pixel
    tensor.

    If all three channels share identical mean/std (and therefore identical
    LUT columns), a flat 1D LUT of shape (256,) is returned so `apply_norm_lut`
    can dispatch to the faster `np.take` path without a per-frame check.

    Returns
    -------
    lut : ndarray
        Shape (256,) when channels are identical, else (256, 3). dtype int8.
    """
    math_scale, math_shift = compute_norm_constants(norm_mean, norm_std, fix_pos)
    u = np.arange(256, dtype=np.float32)[:, None]                # (256, 1)
    table = np.rint(u * math_scale - math_shift)                 # (256, 3)
    lut = np.clip(table, -128, 127).astype(np.int8)
    if (np.array_equal(lut[:, 0], lut[:, 1]) and
            np.array_equal(lut[:, 0], lut[:, 2])):
        return np.ascontiguousarray(lut[:, 0])
    return lut


# Cached channel-index helper for per-channel fancy indexing.
_CHANNEL_INDEX_3 = np.arange(3, dtype=np.intp)


def apply_norm_lut(img_uint8, lut):
    """
    Apply a normalization LUT to an HWC uint8 image and return the int8
    result with identical shape. Uses numpy fancy indexing, which is
    portable across OpenCV versions and ~10x faster than the explicit
    float multiply on ARM.

    A 1D LUT of shape (256,) means all channels share the same mapping
    and `np.take` is used. A 2D LUT of shape (256, 3) dispatches to
    per-channel fancy indexing.
    """
    if lut.ndim == 1:
        return np.take(lut, img_uint8)
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
