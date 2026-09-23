from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None


DETECTION_IOU_THRESHOLDS = tuple(0.5 + 0.05 * index for index in range(10))


def _require_numpy():
    if np is None:
        raise RuntimeError("NumPy is required for detection accuracy evaluation")


def _box_array(boxes, name):
    _require_numpy()
    array = np.asarray(boxes, dtype=np.float32)
    if array.size == 0:
        return np.empty((0, 4), dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 4:
        raise ValueError(f"{name} must have shape (N, 4), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite coordinates")
    return array


def load_yolo_detection_labels(label_path, image_shape, num_classes):
    _require_numpy()
    label_path = Path(label_path)
    height, width = (int(value) for value in image_shape[:2])
    if height <= 0 or width <= 0:
        raise ValueError(f"image_shape must be positive, got {image_shape}")
    if not label_path.is_file():
        return np.empty(0, dtype=np.int32), np.empty((0, 4), dtype=np.float32)

    classes = []
    boxes = []
    with label_path.open("r", encoding="utf-8") as label_file:
        for line_number, line in enumerate(label_file, start=1):
            tokens = line.strip().split()
            if not tokens:
                continue
            if len(tokens) != 5:
                raise ValueError(
                    f"{label_path}:{line_number}: detection rows require exactly five columns"
                )
            try:
                class_id = int(tokens[0])
                center_x, center_y, box_width, box_height = map(float, tokens[1:])
            except ValueError as error:
                raise ValueError(
                    f"{label_path}:{line_number}: invalid YOLO detection row"
                ) from error
            if not 0 <= class_id < num_classes:
                raise ValueError(
                    f"{label_path}:{line_number}: class ID {class_id} is outside "
                    f"[0, {num_classes - 1}]"
                )
            coordinates = np.asarray(
                (center_x, center_y, box_width, box_height), dtype=np.float32
            )
            if not np.all(np.isfinite(coordinates)):
                raise ValueError(f"{label_path}:{line_number}: coordinates must be finite")
            if np.any(coordinates < 0.0) or np.any(coordinates > 1.0):
                raise ValueError(
                    f"{label_path}:{line_number}: normalized coordinates must be in [0, 1]"
                )
            if box_width <= 0.0 or box_height <= 0.0:
                raise ValueError(
                    f"{label_path}:{line_number}: box width and height must be positive"
                )
            x1 = (center_x - box_width / 2.0) * width
            y1 = (center_y - box_height / 2.0) * height
            x2 = (center_x + box_width / 2.0) * width
            y2 = (center_y + box_height / 2.0) * height
            classes.append(class_id)
            boxes.append(
                (
                    max(0.0, min(float(width), x1)),
                    max(0.0, min(float(height), y1)),
                    max(0.0, min(float(width), x2)),
                    max(0.0, min(float(height), y2)),
                )
            )
    return (
        np.asarray(classes, dtype=np.int32),
        np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
    )


def box_iou_matrix(prediction_boxes, ground_truth_boxes):
    predictions = _box_array(prediction_boxes, "prediction_boxes")
    ground_truths = _box_array(ground_truth_boxes, "ground_truth_boxes")
    if not len(predictions) or not len(ground_truths):
        return np.zeros((len(predictions), len(ground_truths)), dtype=np.float32)

    intersection_min = np.maximum(predictions[:, None, :2], ground_truths[None, :, :2])
    intersection_max = np.minimum(predictions[:, None, 2:], ground_truths[None, :, 2:])
    intersection_size = np.maximum(intersection_max - intersection_min, 0.0)
    intersection = intersection_size[..., 0] * intersection_size[..., 1]
    prediction_size = np.maximum(predictions[:, 2:] - predictions[:, :2], 0.0)
    ground_truth_size = np.maximum(ground_truths[:, 2:] - ground_truths[:, :2], 0.0)
    prediction_area = prediction_size[:, 0] * prediction_size[:, 1]
    ground_truth_area = ground_truth_size[:, 0] * ground_truth_size[:, 1]
    union = prediction_area[:, None] + ground_truth_area[None, :] - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=np.float32),
        where=union > 0.0,
    )


def match_detection_predictions(
    prediction_classes,
    prediction_scores,
    prediction_boxes,
    ground_truth_classes,
    ground_truth_boxes,
    iou_thresholds=DETECTION_IOU_THRESHOLDS,
):
    _require_numpy()
    prediction_classes = np.asarray(prediction_classes, dtype=np.int32)
    prediction_scores = np.asarray(prediction_scores, dtype=np.float32)
    predictions = _box_array(prediction_boxes, "prediction_boxes")
    ground_truth_classes = np.asarray(ground_truth_classes, dtype=np.int32)
    ground_truths = _box_array(ground_truth_boxes, "ground_truth_boxes")
    thresholds = np.asarray(iou_thresholds, dtype=np.float32)
    if prediction_classes.shape != prediction_scores.shape or prediction_classes.ndim != 1:
        raise ValueError("prediction classes and scores must be equally sized vectors")
    if len(predictions) != len(prediction_classes):
        raise ValueError("prediction boxes, classes, and scores must have equal lengths")
    if ground_truth_classes.ndim != 1 or len(ground_truths) != len(ground_truth_classes):
        raise ValueError("ground-truth boxes and classes must have equal lengths")
    if thresholds.ndim != 1 or not len(thresholds):
        raise ValueError("iou_thresholds must be a non-empty vector")
    if np.any((thresholds <= 0.0) | (thresholds > 1.0)):
        raise ValueError("IoU thresholds must be in (0, 1]")

    matches = np.zeros((len(predictions), len(thresholds)), dtype=bool)
    if not len(predictions) or not len(ground_truths):
        return matches
    iou = box_iou_matrix(predictions, ground_truths)
    prediction_order = np.argsort(-prediction_scores, kind="stable")
    for threshold_index, threshold in enumerate(thresholds):
        matched_ground_truths = np.zeros(len(ground_truths), dtype=bool)
        for prediction_index in prediction_order:
            candidates = np.where(
                (ground_truth_classes == prediction_classes[prediction_index])
                & ~matched_ground_truths
            )[0]
            if not len(candidates):
                continue
            candidate_iou = iou[prediction_index, candidates]
            best_ground_truth = candidates[int(np.argmax(candidate_iou))]
            if iou[prediction_index, best_ground_truth] >= threshold:
                matches[prediction_index, threshold_index] = True
                matched_ground_truths[best_ground_truth] = True
    return matches


def _compute_ap(recall, precision):
    recall_levels = np.linspace(0.0, 1.0, 101)
    interpolated = [
        np.max(precision[recall >= level]) if np.any(recall >= level) else 0.0
        for level in recall_levels
    ]
    return float(np.mean(interpolated))


def aggregate_detection_metrics(
    evaluation_records,
    ground_truth_counts,
    num_classes,
    iou_thresholds=DETECTION_IOU_THRESHOLDS,
):
    _require_numpy()
    thresholds = np.asarray(iou_thresholds, dtype=np.float32)
    records = list(evaluation_records)
    per_class = {}
    class_ap = []
    class_precision = []
    class_recall = []
    for class_id in range(num_classes):
        ground_truth_count = int(ground_truth_counts.get(class_id, 0))
        if ground_truth_count == 0:
            continue
        class_records = [record for record in records if int(record[0]) == class_id]
        class_records.sort(
            key=lambda record: (
                -float(record[1]),
                str(record[3]) if len(record) > 3 else "",
                int(record[4]) if len(record) > 4 else 0,
            )
        )
        if class_records:
            correct = np.stack(
                [np.asarray(record[2], dtype=bool) for record in class_records]
            )
            if correct.shape[1] != len(thresholds):
                raise ValueError("evaluation record IoU dimensions do not match thresholds")
            true_positives = np.cumsum(correct, axis=0, dtype=np.float64)
            false_positives = np.cumsum(~correct, axis=0, dtype=np.float64)
            recall = true_positives / ground_truth_count
            precision = true_positives / np.maximum(
                true_positives + false_positives, 1e-16
            )
            average_precision = np.asarray(
                [
                    _compute_ap(recall[:, index], precision[:, index])
                    for index in range(len(thresholds))
                ],
                dtype=np.float64,
            )
            precision50 = float(precision[-1, 0])
            recall50 = float(recall[-1, 0])
        else:
            average_precision = np.zeros(len(thresholds), dtype=np.float64)
            precision50 = 0.0
            recall50 = 0.0
        per_class[class_id] = {
            "ground_truths": ground_truth_count,
            "predictions": len(class_records),
            "precision": precision50,
            "recall": recall50,
            "map50": float(average_precision[0]),
            "map50_95": float(np.mean(average_precision)),
        }
        class_ap.append(average_precision)
        class_precision.append(precision50)
        class_recall.append(recall50)

    if class_ap:
        average_precision = np.stack(class_ap)
        map50 = float(np.mean(average_precision[:, 0]))
        map50_95 = float(np.mean(average_precision))
        precision = float(np.mean(class_precision))
        recall = float(np.mean(class_recall))
    else:
        map50 = map50_95 = precision = recall = 0.0
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-16)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map50": map50,
        "map50_95": map50_95,
        "per_class": per_class,
        "iou_thresholds": [float(value) for value in thresholds],
    }


def evaluate_loss(model, dataloader, loss_fn=None, device=None):
    import torch

    from kria_ai.yolov26.detection.loss import create_loss, forward_for_loss, reduce_loss

    device = device or next(model.parameters()).device
    loss_fn = loss_fn or create_loss(model)
    total = 0.0
    batches = 0
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = {
                name: value.to(device) if isinstance(value, torch.Tensor) else value
                for name, value in targets.items()
            }
            total += float(reduce_loss(loss_fn(forward_for_loss(model, images), targets)))
            batches += 1
    return total / batches if batches else 0.0


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(prog="python -m kria_ai yolov26 detection evaluate")
    parser.add_argument("--model")
    parser.add_argument("--dataset")
    parser.add_argument("--checkpoint")
    parser.add_argument("--subset-len", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    import torch

    from kria_ai.yolov26.detection.config import get_dataset, get_model
    from kria_ai.yolov26.detection.data import build_loader
    from kria_ai.yolov26.models import build_model

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    model_config = get_model(args.model)
    dataset_config = get_dataset(args.dataset)
    model = build_model(model_config, device=device, checkpoint_path=args.checkpoint)
    loader = build_loader(
        model_config,
        dataset_config,
        split="validation",
        subset_len=args.subset_len,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    loss = evaluate_loss(model, loader, device=device)
    print(f"samples={len(loader.dataset)} detection_loss={loss:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
