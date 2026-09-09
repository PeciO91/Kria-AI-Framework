from kria_ai.classification.preprocess import preprocess_board_image
from kria_ai.common.board.input_quantization import (
    apply_normalization_lut,
    build_normalization_lut,
    normalization_constants,
)
from kria_ai.common.board.power import PowerMonitor
from kria_ai.common.board.profiling import (
    ProgressCounter,
    StageProfiler,
    format_metrics,
    merge_stage_profilers,
)
from kria_ai.common.board.runtime import load_dpu_model


compute_norm_constants = normalization_constants
build_norm_lut = build_normalization_lut
apply_norm_lut = apply_normalization_lut
format_report = format_metrics


def preprocess_image(image_rgb, dpu_shape, lut):
    input_shape = dpu_shape[1:3] if len(dpu_shape) == 4 else dpu_shape
    return preprocess_board_image(image_rgb, input_shape, lut)


def setup_dpu(model_path):
    dpu_model = load_dpu_model(model_path)
    return (
        dpu_model.subgraph,
        dpu_model.inputs[0].shape,
        dpu_model.inputs[0].fixed_point,
        [output.fixed_point for output in dpu_model.outputs],
    )


def format_profile_report(title, profiler, wall_time, groups):
    rows = {row["stage"]: row for row in profiler.summary(wall_time)}
    metrics = []
    for group, stages in groups:
        metrics.append((group, ""))
        for stage in stages:
            row = rows.get(stage)
            if row:
                metrics.append((stage, f"{row['avg_ms']:.2f} ms avg, {row['total_s']:.3f} s total"))
    return format_metrics(title, metrics)


__all__ = [
    "PowerMonitor",
    "ProgressCounter",
    "StageProfiler",
    "apply_norm_lut",
    "build_norm_lut",
    "compute_norm_constants",
    "format_profile_report",
    "format_report",
    "merge_stage_profilers",
    "preprocess_image",
    "setup_dpu",
]
