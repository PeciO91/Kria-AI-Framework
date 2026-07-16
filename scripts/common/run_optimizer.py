"""Vitis AI PyTorch optimizer / pruning runner.

This module is being rebuilt from the UG1333 guide. It supports:
  - Coarse-grained pruning runner creation (iterative / one_step)
  - Classification and yolov26 detection/segmentation pruning flows
"""

import os
import argparse
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.models.resnet import resnet50, resnet18
from pytorch_nndct import get_pruning_runner

# --- VAIQ deepcopy compatibility patch ---
# Vitis AI's ``override_torch_function`` breaks ``Tensor.__deepcopy__`` for the
# traced graph tensors. The one-step search path calls
# ``copy.deepcopy(self._model)`` inside ``_generate_slim_model`` and raises
# a ``new_empty`` RuntimeError. We fall back to ``clone()`` for those tensors.
_original_tensor_deepcopy = torch.Tensor.__deepcopy__


def _safe_tensor_deepcopy(self, memo):
    try:
        return _original_tensor_deepcopy(self, memo)
    except RuntimeError as e:
        if "new_empty" in str(e):
            if isinstance(self, torch.nn.Parameter):
                return torch.nn.Parameter(self.clone(), requires_grad=self.requires_grad)
            return self.clone()
        raise


torch.Tensor.__deepcopy__ = _safe_tensor_deepcopy

# _bootstrap sets PROJECT_ROOT and sys.path before other project imports.
from _bootstrap import PROJECT_ROOT  # noqa: F401

from model_config import get_active_model
from dataset_config import get_active_dataset
from model_utils import prepare_model, derive_weight_path
from dataset_utils import FlatImageDataset, YoloDataset, yolo_collate_fn, build_or_load_subset_indices
from optimizer_utils import collect_model_metrics, compute_metrics_for_pruned_model, format_report, save_report_json
from detection_profiles import get_profile


# ---------------------------------------------------------------------------
# Coarse-grained pruning
# ---------------------------------------------------------------------------

def create_pruning_runner(model, inputs, method):
    """Create a coarse-grained pruning runner.

    Args:
        model: The model to be pruned.
        inputs: Input tensor with the same shape and dtype as the model input.
        method: Either 'iterative' or 'one_step'.
    """
    if method == "iterative":
        return get_pruning_runner(model, inputs, "iterative")
    elif method == "one_step":
        return get_pruning_runner(model, inputs, "one_step")
    raise ValueError(
        f"Unknown coarse-grained pruning method: {method}. "
        f"Expected 'iterative' or 'one_step'."
    )


# ---------------------------------------------------------------------------
# Model creation helper (from the guide examples)
# ---------------------------------------------------------------------------

def create_model(model_name="resnet50", pretrained=True):
    """Create a torchvision model for the pruning examples."""
    if model_name == "resnet50":
        return resnet50(pretrained=pretrained)
    elif model_name == "resnet18":
        return resnet18(pretrained=pretrained)
    raise ValueError(f"Unknown model_name: {model_name}")


# ---------------------------------------------------------------------------
# Iterative pruning
# ---------------------------------------------------------------------------

class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(outputs, targets, topk=(1, 5)):
    """Computes the accuracy over the k top predictions for the specified topk values."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.size(0)
        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def eval_fn(model, dataloader):
    """Classification evaluation function for iterative pruning.

    Returns top-1 accuracy. The runner will maximize this score.
    """
    top1 = AverageMeter('Acc@1', ':6.2f')
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for i, (images, targets) in enumerate(dataloader):
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            acc1, _ = accuracy(outputs, targets, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
    return top1.avg


def detection_eval_fn(model, dataloader, m_cfg):
    """Detection / segmentation evaluation for iterative ana and one-step search.

    Returns the negative average loss so that a higher score is better,
    matching the runner's maximize-score contract (same role as the top-1
    accuracy returned by the classification ``eval_fn``).
    """
    profile = get_profile(m_cfg)
    loss_fn = profile.loss_fn(model)
    device = next(model.parameters()).device
    total_loss = 0.0
    count = 0
    was_training = model.training
    model.train()  # training output format is required for the loss
    try:
        with torch.no_grad():
            for images, targets in dataloader:
                images = images.to(device)
                targets = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                           for k, v in targets.items()}
                outputs = model(images)
                loss_val = loss_fn(outputs, targets)
                if isinstance(loss_val, dict):
                    loss_val = sum(l for l in loss_val.values())
                elif isinstance(loss_val, (list, tuple)):
                    loss_val = sum(loss_val)
                if isinstance(loss_val, torch.Tensor):
                    loss_val = loss_val.sum()
                total_loss += float(loss_val)
                count += 1
    finally:
        if not was_training:
            model.eval()
    return -(total_loss / max(1, count))


def run_iterative_analysis(runner, dataloader, eval_fn=eval_fn, eval_args=None,
                           excludes=None):
    """Run the one-time sensitivity analysis for iterative pruning.

    The result is cached in .vai/<model>.sens and can be reused for multiple
    pruning ratios.
    """
    if eval_args is None:
        eval_args = (dataloader,)
    runner.ana(eval_fn, args=eval_args, excludes=excludes)


def run_iterative_prune(runner, removal_ratio, excludes=None):
    """Generate a pruned model from the cached analysis result."""
    return runner.prune(removal_ratio=removal_ratio, excludes=excludes)


# ---------------------------------------------------------------------------
# One-step pruning
# ---------------------------------------------------------------------------

def calibration_fn(model, dataloader, number_forward=100):
    """Calibrate BatchNorm statistics for adaptive-BN one-step search."""
    model.train()
    device = next(model.parameters()).device
    with torch.no_grad():
        for index, (images, target) in enumerate(dataloader):
            images = images.to(device)
            model(images)
            if index >= number_forward:
                break


def run_one_step_search(runner, dataloader, num_subnet=1000, removal_ratio=0.7,
                        gpus=['0'], calibration_fn=calibration_fn, eval_fn=eval_fn,
                        excludes=None, eval_args=None, calib_args=None):
    """Run adaptive-BN search for one-step pruning."""
    if eval_args is None:
        eval_args = (dataloader,)
    if calib_args is None:
        calib_args = (dataloader,)
    runner.search(
        gpus=gpus,
        calibration_fn=calibration_fn,
        calib_args=calib_args,
        eval_fn=eval_fn,
        eval_args=eval_args,
        num_subnet=num_subnet,
        removal_ratio=removal_ratio,
        excludes=excludes,
    )


def run_one_step_prune(runner, removal_ratio, index=None):
    """Generate the final pruned model from the one-step search result."""
    return runner.prune(removal_ratio=removal_ratio, index=index)


# ---------------------------------------------------------------------------
# Retraining
# ---------------------------------------------------------------------------

def train(train_loader, model, criterion, optimizer, epoch):
    """Train one epoch for classification."""
    model.train()
    device = next(model.parameters()).device
    for i, (images, target) in enumerate(train_loader):
        images = images.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


def evaluate(dataloader, model, criterion):
    """Evaluate classification top-1 and top-5 accuracy."""
    model.eval()
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    device = next(model.parameters()).device
    with torch.no_grad():
        for images, target in dataloader:
            images = images.to(device)
            target = target.to(device)
            output = model(images)
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
    return top1.avg, top5.avg


def run_retrain(model, train_loader, val_loader, criterion, epochs,
                lr=1e-3, weight_decay=5e-4, save_prefix='model'):
    """Fine-tune a pruned model and save the best checkpoint."""
    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=weight_decay)
    best_acc1 = 0
    for epoch in range(epochs):
        train(train_loader, model, criterion, optimizer, epoch)
        acc1, acc5 = evaluate(val_loader, model, criterion)
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        if is_best:
            torch.save(model.state_dict(), f'{save_prefix}_pruned.pth')
            if hasattr(model, 'slim_state_dict'):
                torch.save(model.slim_state_dict(), f'{save_prefix}_slim.pth')
    return model


def _reduce_loss(loss):
    """Reduce a YOLO loss (tuple / dict / list / tensor) to a scalar tensor."""
    if isinstance(loss, tuple) and len(loss) >= 2:
        # Ultralytics returns (total_loss, detached_loss_components).
        loss_val = loss[0]
    elif isinstance(loss, dict):
        loss_val = sum(l for l in loss.values())
    elif isinstance(loss, (list, tuple)):
        loss_val = sum(loss)
    else:
        loss_val = loss
    if isinstance(loss_val, torch.Tensor) and loss_val.dim() > 0:
        loss_val = loss_val.sum()
    return loss_val


def train_detection(train_loader, model, profile, criterion, optimizer, epoch):
    """Train one epoch for a yolov26 detection / segmentation model."""
    model.train()
    device = next(model.parameters()).device
    for i, (images, targets) in enumerate(train_loader):
        images = images.to(device)
        targets = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                   for k, v in targets.items()}
        optimizer.zero_grad()
        outputs = profile.forward_for_loss(model, images)
        loss_val = _reduce_loss(criterion(outputs, targets))
        loss_val.backward()
        optimizer.step()


def run_retrain_detection(model, m_cfg, train_loader, epochs,
                          lr=1e-3, weight_decay=1e-4, save_prefix='model'):
    """Fine-tune a pruned yolov26 detection / segmentation model.

    Uses the DetectionProfile to attach the correct (E2E) loss and to keep the
    Detect/Segment26 head in training mode. AdamW + CosineAnnealingLR mirror
    the working configuration from the previous pipeline.
    """
    profile = get_profile(m_cfg)
    profile.prepare_for_finetune(model)
    criterion = profile.loss_fn(model)

    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float('inf')
    device = next(model.parameters()).device
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for images, targets in train_loader:
            images = images.to(device)
            targets = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                       for k, v in targets.items()}
            optimizer.zero_grad()
            outputs = profile.forward_for_loss(model, images)
            loss_val = _reduce_loss(criterion(outputs, targets))
            loss_val.backward()
            optimizer.step()
            epoch_loss += loss_val.item()
        scheduler.step()

        avg_loss = epoch_loss / max(1, len(train_loader))
        print(f"  Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), f'{save_prefix}_pruned.pth')
            if hasattr(model, 'slim_state_dict'):
                torch.save(model.slim_state_dict(), f'{save_prefix}_slim.pth')

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Final pruned (slim) model generation
# ---------------------------------------------------------------------------

def generate_slim_model(model, input_signature, method, removal_ratio,
                        checkpoint_path='model_pruned.pth', index=None):
    """Generate a slim model by loading a retrained checkpoint into it."""
    runner = get_pruning_runner(model, input_signature, method)
    if method == 'one_step':
        slim_model = runner.prune(removal_ratio=removal_ratio, mode='slim', index=index)
    else:
        slim_model = runner.prune(removal_ratio=removal_ratio, mode='slim')
    slim_model.load_state_dict(torch.load(checkpoint_path))
    return slim_model


def generate_slim_model_without_api(model, checkpoint_path='model_pruned.pth'):
    """Generate a slim model without a pruning runner."""
    from pytorch_nndct.utils import slim
    return slim.load_state_dict(model, torch.load(checkpoint_path))


# ---------------------------------------------------------------------------
# Task-aware data loaders
# ---------------------------------------------------------------------------

def _is_detection_or_segmentation(m_cfg):
    return m_cfg.get('type') in ('detection', 'segmentation')


def build_finetune_loader(m_cfg, d_cfg, subset_len=200, batch_size=4):
    """Build a training loader for fine-tuning."""
    task_type = m_cfg.get('type', 'classification')

    if task_type == 'classification':
        transform = transforms.Compose([
            transforms.Resize(m_cfg['input_shape']),
            transforms.ToTensor(),
            transforms.Normalize(d_cfg['normalization']['mean'], d_cfg['normalization']['std']),
        ])
        dataset = ImageFolder(root=d_cfg['calib_path'], transform=transform)
        if len(dataset) > subset_len:
            random.seed(42)
            indices = random.sample(range(len(dataset)), subset_len)
            dataset = Subset(dataset, indices)
        collate_fn = None
    elif _is_detection_or_segmentation(m_cfg):
        subset_indices = build_or_load_subset_indices(
            split='train', n=subset_len, cache_dir=d_cfg['subset_cache_dir']
        )
        dataset = YoloDataset(
            images_dir=d_cfg['images_train'],
            labels_dir=d_cfg['labels_train'],
            input_shape=m_cfg['input_shape'],
            normalization=d_cfg['normalization'],
            augment=True,
            indices=subset_indices,
        )
        collate_fn = yolo_collate_fn
    else:
        transform = transforms.Compose([
            transforms.Resize(m_cfg['input_shape']),
            transforms.ToTensor(),
            transforms.Normalize(d_cfg['normalization']['mean'], d_cfg['normalization']['std']),
        ])
        dataset = FlatImageDataset(root_dir=d_cfg['calib_path'], transform=transform)
        if len(dataset) > subset_len:
            dataset = Subset(dataset, list(range(subset_len)))
        collate_fn = None

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


def build_analysis_loader(m_cfg, d_cfg, subset_len=200, batch_size=4):
    """Build a validation-style loader for analysis / evaluation."""
    task_type = m_cfg.get('type', 'classification')

    if task_type == 'classification':
        transform = transforms.Compose([
            transforms.Resize(m_cfg['input_shape']),
            transforms.ToTensor(),
            transforms.Normalize(d_cfg['normalization']['mean'], d_cfg['normalization']['std']),
        ])
        dataset = ImageFolder(root=d_cfg['calib_path'], transform=transform)
        if len(dataset) > subset_len:
            random.seed(42)
            indices = random.sample(range(len(dataset)), subset_len)
            dataset = Subset(dataset, indices)
        collate_fn = None
    elif _is_detection_or_segmentation(m_cfg):
        subset_indices = build_or_load_subset_indices(
            split='val', n=subset_len, cache_dir=d_cfg['subset_cache_dir']
        )
        dataset = YoloDataset(
            images_dir=d_cfg['images_val'],
            labels_dir=d_cfg['labels_val'],
            input_shape=m_cfg['input_shape'],
            normalization=d_cfg['normalization'],
            augment=False,
            indices=subset_indices,
        )
        collate_fn = yolo_collate_fn
    else:
        transform = transforms.Compose([
            transforms.Resize(m_cfg['input_shape']),
            transforms.ToTensor(),
            transforms.Normalize(d_cfg['normalization']['mean'], d_cfg['normalization']['std']),
        ])
        dataset = FlatImageDataset(root_dir=d_cfg['calib_path'], transform=transform)
        if len(dataset) > subset_len:
            dataset = Subset(dataset, list(range(subset_len)))
        collate_fn = None

    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


# ---------------------------------------------------------------------------
# CLI orchestration
# ---------------------------------------------------------------------------

def run_optimizer():
    parser = argparse.ArgumentParser(description='Vitis AI Optimizer / Pruning Pipeline')
    parser.add_argument('--model', type=str, default=None,
                        help='Model ID from model_config.py')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset ID from dataset_config.py')
    parser.add_argument('--method', choices=['iterative', 'one_step'], default='one_step',
                        help='Pruning algorithm')
    parser.add_argument('--mode', choices=['search', 'prune', 'finetune', 'all'], default='all',
                        help='Pipeline stage(s) to run')
    parser.add_argument('--ratio', type=float, default=0.2,
                        help='Channel removal ratio')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Fine-tuning epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Fine-tuning learning rate')
    parser.add_argument('--subset_len', type=int, default=200,
                        help='Number of samples for search/finetune')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Data loader batch size')
    parser.add_argument('--num_subnet', type=int, default=1000,
                        help='Candidate subnets for one-step search')
    parser.add_argument('--num_calib_forward', type=int, default=100,
                        help='BN calibration batches for one-step search')
    parser.add_argument('--device', type=str, default='auto',
                        help='auto | cuda | cpu')
    args = parser.parse_args()

    m_cfg = get_active_model(args.model)
    if args.dataset is None:
        if m_cfg.get('type') == 'segmentation':
            args.dataset = 'coco'
    d_cfg = get_active_dataset(args.dataset)

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f'[INFO] Using device: {device}')
    print(f'[INFO] Model: {m_cfg["name"]} | method: {args.method} | mode: {args.mode}')

    model = prepare_model(m_cfg, device, prune_threshold=None)
    input_h, input_w = m_cfg['input_shape']
    dummy_input = torch.randn([1, 3, input_h, input_w], dtype=torch.float32).to(device)
    runner = create_pruning_runner(model, dummy_input, args.method)

    excludes = []
    if _is_detection_or_segmentation(m_cfg):
        profile = get_profile(m_cfg)
        excludes = profile.prune_excludes(model)
        print(f'[INFO] Excluding {len(excludes)} modules from pruning')

    train_loader = build_finetune_loader(m_cfg, d_cfg, subset_len=args.subset_len, batch_size=args.batch_size)
    val_loader = build_analysis_loader(m_cfg, d_cfg, subset_len=args.subset_len, batch_size=args.batch_size)

    evaluator = detection_eval_fn if _is_detection_or_segmentation(m_cfg) else eval_fn
    eval_args = (train_loader, m_cfg) if _is_detection_or_segmentation(m_cfg) else (train_loader,)
    val_eval_args = (val_loader, m_cfg) if _is_detection_or_segmentation(m_cfg) else (val_loader,)

    output_dir = os.path.join('build', m_cfg['name'].lower(), 'optimizer_report')
    os.makedirs(output_dir, exist_ok=True)

    before_metrics = None
    after_metrics = None
    current_model = model

    if args.mode in ('search', 'all'):
        if args.method == 'one_step':
            import functools
            calib_fn = functools.partial(calibration_fn, number_forward=args.num_calib_forward)
            gpus = [str(torch.cuda.current_device())] if device.type == 'cuda' else []
            run_one_step_search(
                runner, train_loader,
                num_subnet=args.num_subnet,
                removal_ratio=args.ratio,
                gpus=gpus,
                calibration_fn=calib_fn,
                eval_fn=evaluator,
                excludes=excludes,
                eval_args=eval_args,
                calib_args=(train_loader,),
            )
        else:
            run_iterative_analysis(
                runner, val_loader, eval_fn=evaluator, eval_args=val_eval_args, excludes=excludes
            )

    if args.mode in ('prune', 'all'):
        if args.method == 'one_step':
            current_model = run_one_step_prune(runner, args.ratio)
        else:
            current_model = run_iterative_prune(runner, args.ratio, excludes=excludes)

        before_metrics = collect_model_metrics(model, (input_h, input_w), m_cfg.get('gops'))
        if hasattr(current_model, 'slim_state_dict'):
            after_metrics = compute_metrics_for_pruned_model(
                current_model, current_model.slim_state_dict(), (input_h, input_w), before_metrics
            )
        else:
            after_metrics = collect_model_metrics(current_model, (input_h, input_w), m_cfg.get('gops'))

        pruned_path = derive_weight_path(m_cfg['model_path'], '_pruned')
        torch.save(current_model.state_dict(), pruned_path)
        print(f'[INFO] Saved pruned model to: {pruned_path}')
        if hasattr(current_model, 'slim_state_dict'):
            slim_path = derive_weight_path(m_cfg['model_path'], '_slim')
            torch.save(current_model.slim_state_dict(), slim_path)
            print(f'[INFO] Saved slim model to: {slim_path}')

    if args.mode in ('finetune', 'all'):
        if _is_detection_or_segmentation(m_cfg):
            current_model = run_retrain_detection(
                current_model, m_cfg, train_loader,
                epochs=args.epochs, lr=args.lr,
                save_prefix=os.path.splitext(derive_weight_path(m_cfg['model_path'], '_finetuned'))[0]
            )
        else:
            current_model = run_retrain(
                current_model, train_loader, val_loader,
                nn.CrossEntropyLoss(), epochs=args.epochs, lr=args.lr,
                save_prefix=os.path.splitext(derive_weight_path(m_cfg['model_path'], '_finetuned'))[0]
            )

        finetuned_path = derive_weight_path(m_cfg['model_path'], '_finetuned')
        torch.save(current_model.state_dict(), finetuned_path)
        print(f'[INFO] Saved finetuned model to: {finetuned_path}')
        if hasattr(current_model, 'slim_state_dict'):
            finetuned_slim_path = derive_weight_path(m_cfg['model_path'], '_finetuned_slim')
            torch.save(current_model.slim_state_dict(), finetuned_slim_path)
            print(f'[INFO] Saved finetuned slim model to: {finetuned_slim_path}')

    if before_metrics and after_metrics:
        print('\n' + format_report(before_metrics, after_metrics, m_cfg['name']))
        report_path = save_report_json(before_metrics, after_metrics, m_cfg['name'], output_dir)
        print(f'[INFO] Report saved to: {report_path}')

    print('\n' + '=' * 70)
    print('  OPTIMIZER PIPELINE COMPLETE')
    print('=' * 70)


if __name__ == '__main__':
    run_optimizer()
