import random
from pathlib import Path

from kria_ai.classification.preprocess import build_transform


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def build_dataset(model_config, dataset_config, split="calibration", subset_len=None, seed=42):
    from torch.utils.data import Subset
    from torchvision.datasets import ImageFolder

    roots = {
        "calibration": dataset_config.calibration_root,
        "evaluation": dataset_config.evaluation_root,
    }
    try:
        root = _resolve(roots[split])
    except KeyError as error:
        raise ValueError(f"Unknown classification split {split!r}") from error
    if not root.is_dir():
        raise FileNotFoundError(f"Classification dataset directory not found: {root}")
    dataset = ImageFolder(root=root, transform=build_transform(model_config, dataset_config))
    if tuple(dataset.classes) != tuple(dataset_config.classes):
        raise ValueError(
            f"Dataset class order {tuple(dataset.classes)!r} does not match configured order {dataset_config.classes!r}"
        )
    if subset_len is not None and len(dataset) > subset_len:
        indices = random.Random(seed).sample(range(len(dataset)), subset_len)
        dataset = Subset(dataset, indices)
    return dataset


def build_loader(
    model_config,
    dataset_config,
    split="calibration",
    subset_len=None,
    batch_size=32,
    seed=42,
    shuffle=False,
    num_workers=0,
):
    from torch.utils.data import DataLoader

    dataset = build_dataset(model_config, dataset_config, split=split, subset_len=subset_len, seed=seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
