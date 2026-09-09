import threading
import time


class ProgressCounter:
    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()

    def increment(self, count=1):
        with self._lock:
            self._value += count

    @property
    def value(self):
        with self._lock:
            return self._value


class StageProfiler:
    def __init__(self, enabled=False):
        self.enabled = enabled
        self._samples = {}

    def add(self, stage, elapsed):
        if self.enabled:
            self._samples.setdefault(stage, []).append(float(elapsed))

    def time(self, stage):
        return StageTimer(self, stage)

    def merge(self, other):
        if not self.enabled or other is None:
            return
        for stage, values in other._samples.items():
            self._samples.setdefault(stage, []).extend(values)

    def summary(self, wall_time=None):
        import numpy as np

        rows = []
        for stage, values in self._samples.items():
            if not values:
                continue
            samples = np.asarray(values, dtype=np.float64)
            total = float(samples.sum())
            rows.append({
                "stage": stage,
                "count": int(samples.size),
                "total_s": total,
                "avg_ms": float(samples.mean() * 1000.0),
                "p50_ms": float(np.percentile(samples, 50) * 1000.0),
                "p95_ms": float(np.percentile(samples, 95) * 1000.0),
                "min_ms": float(samples.min() * 1000.0),
                "max_ms": float(samples.max() * 1000.0),
                "wall_pct": float(total / wall_time * 100.0) if wall_time and wall_time > 0 else 0.0,
            })
        return rows


class StageTimer:
    def __init__(self, profiler, stage):
        self.profiler = profiler
        self.stage = stage
        self.started = None

    def __enter__(self):
        if self.profiler.enabled:
            self.started = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.started is not None:
            self.profiler.add(self.stage, time.perf_counter() - self.started)
        return False


def merge_stage_profilers(profilers):
    merged = StageProfiler(enabled=True)
    for profiler in profilers:
        merged.merge(profiler)
    return merged


def format_metrics(title, metrics):
    lines = ["=" * 60, f"  {title}", "=" * 60]
    for label, value in metrics:
        lines.append("-" * 60 if label == "---" else f"{label:<24}{value}")
    lines.append("=" * 60)
    return "\n".join(lines) + "\n"
