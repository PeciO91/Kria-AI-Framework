"""NumPy/OpenCV primitives for YOLO instance-mask evaluation."""

import os

try:
    import numpy as np
except ImportError:
    np = None

try:
    import cv2
except ImportError:
    cv2 = None


__all__ = [
    "aggregate_mask_map50",
    "compute_ap",
    "compute_mask_map50",
    "load_yolo_seg_labels",
    "mask_iou_matrix",
    "match_mask_predictions",
]


def load_yolo_seg_labels(label_path, img_shape):
    """Load YOLO polygon labels and rasterize them at image resolution.

    Label rows have the form ``class x1 y1 ... xn yn``, with normalized
    polygon coordinates. The return value is a class-id list and a boolean
    array shaped ``(N, H, W)``.
    """
    if not os.path.exists(label_path):
        return [], np.empty((0, img_shape[0], img_shape[1]), dtype=bool)

    classes = []
    masks = []
    height, width = img_shape

    with open(label_path, "r") as label_file:
        for line in label_file:
            parts = line.strip().split()
            if len(parts) < 3:
                continue

            class_id = int(parts[0])
            coordinates = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
            coordinates[:, 0] *= width
            coordinates[:, 1] *= height

            mask = np.zeros((height, width), dtype=np.uint8)
            if cv2 is None:
                raise RuntimeError("OpenCV (cv2) is required to rasterize polygon labels")
            cv2.fillPoly(mask, [coordinates.astype(np.int32)], 1)
            classes.append(class_id)
            masks.append(mask.astype(bool))

    if len(masks) == 0:
        return [], np.empty((0, height, width), dtype=bool)

    return classes, np.stack(masks, axis=0)


def mask_iou_matrix(pred_masks, gt_masks):
    """Compute every predicted/ground-truth mask intersection over union."""
    prediction_count = pred_masks.shape[0]
    ground_truth_count = gt_masks.shape[0]

    if prediction_count == 0 or ground_truth_count == 0:
        return np.zeros(
            (prediction_count, ground_truth_count),
            dtype=np.float32,
        )

    predictions = pred_masks.reshape(prediction_count, -1).astype(np.float32)
    ground_truth = gt_masks.reshape(ground_truth_count, -1).astype(np.float32)
    intersection = np.dot(predictions, ground_truth.T)
    prediction_area = predictions.sum(axis=1)[:, None]
    ground_truth_area = ground_truth.sum(axis=1)[None, :]
    union = prediction_area + ground_truth_area - intersection

    iou = np.zeros_like(intersection, dtype=np.float32)
    valid = union > 0
    iou[valid] = intersection[valid] / union[valid]
    return iou


def compute_ap(recall, precision):
    """Compute area under the precision envelope at recall transitions."""
    interpolated_recall = np.concatenate(([0.0], recall, [1.0]))
    interpolated_precision = np.concatenate(([1.0], precision, [0.0]))

    for index in range(interpolated_precision.size - 1, 0, -1):
        interpolated_precision[index - 1] = np.maximum(
            interpolated_precision[index - 1],
            interpolated_precision[index],
        )

    transitions = np.where(
        interpolated_recall[1:] != interpolated_recall[:-1]
    )[0]
    return np.sum(
        (interpolated_recall[transitions + 1] - interpolated_recall[transitions])
        * interpolated_precision[transitions + 1]
    )


def match_mask_predictions(
    pred_classes,
    pred_scores,
    pred_masks,
    gt_classes,
    gt_masks,
    iou_threshold=0.5,
):
    """Greedily match one image's masks using the current mAP@0.5 policy.

    Predictions are considered in descending confidence order. A prediction
    may only match an unmatched ground truth of the same class, and an IoU
    must be strictly greater than ``iou_threshold``. The returned flags are in
    the predictions' original order.
    """
    iou = mask_iou_matrix(pred_masks, gt_masks)
    ground_truth_classes = np.array(gt_classes)
    true_positive = np.zeros(len(pred_classes), dtype=bool)

    if len(gt_classes) > 0 and len(pred_classes) > 0:
        sorted_indices = np.argsort(-pred_scores)
        sorted_classes = pred_classes[sorted_indices]
        sorted_iou = iou[sorted_indices]
        ground_truth_matched = np.zeros(len(gt_classes), dtype=bool)

        for sorted_index, predicted_class in enumerate(sorted_classes):
            candidate_indices = np.where(
                (ground_truth_classes == predicted_class) & ~ground_truth_matched
            )[0]
            if len(candidate_indices) > 0:
                candidate_iou = sorted_iou[sorted_index, candidate_indices]
                best_index = candidate_indices[np.argmax(candidate_iou)]
                if sorted_iou[sorted_index, best_index] > iou_threshold:
                    true_positive[sorted_indices[sorted_index]] = True
                    ground_truth_matched[best_index] = True

    return true_positive


def aggregate_mask_map50(eval_records, gt_counts, num_classes=None):
    """Aggregate per-prediction records into the current mask mAP@0.5 metrics.

    Args:
        eval_records: Iterable of ``(class_id, confidence, is_true_positive)``.
        gt_counts: Per-class ground-truth counts, either a mapping or sequence.
        num_classes: Number of classes to inspect. If omitted, it is inferred
            from ``gt_counts``.

    Returns:
        A dictionary containing mean ``map50``, ``precision``, ``recall``, and
        ``f1`` plus per-class AP, precision, and recall dictionaries. As in the
        board runner, classes without ground truths do not contribute to means.
    """
    records = list(eval_records)
    if num_classes is None:
        if hasattr(gt_counts, "keys"):
            keys = list(gt_counts.keys())
            num_classes = max(keys) + 1 if keys else 0
        else:
            num_classes = len(gt_counts)

    average_precision = []
    final_precision = []
    final_recall = []
    ap_per_class = {}
    precision_per_class = {}
    recall_per_class = {}

    for class_id in range(num_classes):
        ground_truth_count = (
            gt_counts.get(class_id, 0)
            if hasattr(gt_counts, "get")
            else gt_counts[class_id]
        )
        if ground_truth_count == 0:
            continue

        class_records = [record for record in records if record[0] == class_id]
        class_records.sort(key=lambda record: record[1], reverse=True)

        true_positives = np.array(
            [1 if record[2] else 0 for record in class_records]
        )
        false_positives = np.array(
            [0 if record[2] else 1 for record in class_records]
        )
        true_positive_sum = np.cumsum(true_positives)
        false_positive_sum = np.cumsum(false_positives)
        recalls = true_positive_sum / ground_truth_count
        precisions = true_positive_sum / (
            true_positive_sum + false_positive_sum + 1e-16
        )

        class_ap = compute_ap(recalls, precisions)
        class_precision = precisions[-1] if len(precisions) > 0 else 0.0
        class_recall = recalls[-1] if len(recalls) > 0 else 0.0
        average_precision.append(class_ap)
        final_precision.append(class_precision)
        final_recall.append(class_recall)
        ap_per_class[class_id] = class_ap
        precision_per_class[class_id] = class_precision
        recall_per_class[class_id] = class_recall

    map50 = np.mean(average_precision) if average_precision else 0.0
    mean_precision = np.mean(final_precision) if final_precision else 0.0
    mean_recall = np.mean(final_recall) if final_recall else 0.0
    f1 = 2 * (mean_precision * mean_recall) / (
        mean_precision + mean_recall + 1e-16
    )

    return {
        "map50": map50,
        "precision": mean_precision,
        "recall": mean_recall,
        "f1": f1,
        "ap_per_class": ap_per_class,
        "precision_per_class": precision_per_class,
        "recall_per_class": recall_per_class,
    }


compute_mask_map50 = aggregate_mask_map50


def evaluate_loss(model, dataloader, loss_fn=None, device=None):
    import torch

    from kria_ai.yolov26.segmentation.loss import create_loss, forward_for_loss, reduce_loss

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


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(prog="python -m kria_ai yolov26 segmentation evaluate")
    parser.add_argument("--model")
    parser.add_argument("--dataset")
    parser.add_argument("--checkpoint")
    parser.add_argument("--subset-len", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    if np is None or cv2 is None:
        parser.error("NumPy and OpenCV are required for segmentation evaluation")

    import torch

    from kria_ai.yolov26.models import build_model
    from kria_ai.yolov26.segmentation.config import get_dataset, get_model
    from kria_ai.yolov26.segmentation.data import build_loader

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
    print(f"samples={len(loader.dataset)} segmentation_loss={loss:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
