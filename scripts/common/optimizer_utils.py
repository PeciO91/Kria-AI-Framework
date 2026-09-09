from kria_ai.classification.evaluate import evaluate_loss
from kria_ai.common.model_metrics import (
    collect_model_metrics,
    count_parameters,
    estimate_gflops,
    metrics_from_slim_state_dict,
    model_size_mb,
    per_layer_channel_summary,
    save_metrics_report,
)


compute_metrics_for_pruned_model = metrics_from_slim_state_dict
save_report_json = save_metrics_report


__all__ = [
    "collect_model_metrics",
    "compute_metrics_for_pruned_model",
    "count_parameters",
    "estimate_gflops",
    "evaluate_loss",
    "model_size_mb",
    "per_layer_channel_summary",
    "save_report_json",
]
