import random
from pathlib import Path

from kria_ai.yolov26.preprocess import preprocess_host_image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def list_image_files(images_dir):
    """Return a stable list of image filenames from one flat directory."""
    images_dir = _resolve(images_dir)
    if not images_dir.is_dir():
        raise FileNotFoundError(f"YOLO image directory not found: {images_dir}")
    return sorted(
        path.name
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _normalization_values(normalization):
    normalization = normalization or {
        "mean": (0.0, 0.0, 0.0),
        "std": (1.0, 1.0, 1.0),
    }
    return normalization["mean"], normalization["std"]


class CalibrationImageDataset:
    """Flat, image-only YOLO dataset used for quantizer calibration."""

    def __init__(
        self,
        images_dir,
        input_shape=(640, 640),
        normalization=None,
        indices=None,
    ):
        self.images_dir = _resolve(images_dir)
        self.input_shape = tuple(input_shape)
        self.mean, self.std = _normalization_values(normalization)
        image_files = list_image_files(self.images_dir)
        if indices is not None:
            image_files = [
                image_files[index]
                for index in indices
                if 0 <= index < len(image_files)
            ]
        self.image_files = image_files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        import cv2

        image_path = self.images_dir / self.image_files[index]
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise FileNotFoundError(f"Unable to read calibration image: {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return preprocess_host_image(
            image_rgb,
            self.input_shape,
            mean=self.mean,
            std=self.std,
        )


def build_or_load_subset_indices(
    split,
    n,
    seed=42,
    cache_dir="data/coco/.subsets",
    total_count=None,
    *,
    dataset_length=None,
):
    """Load or create deterministic, unique indices bounded by dataset size.

    ``total_count`` retains the legacy utility's argument name.  New callers may
    use the clearer ``dataset_length`` keyword.  Unlike the legacy default of
    1000, one of them must provide the actual scanned dataset length.
    """
    if dataset_length is not None:
        if total_count is not None and total_count != dataset_length:
            raise ValueError("total_count and dataset_length must match")
        total_count = dataset_length
    if total_count is None:
        raise ValueError("The actual dataset length is required")
    try:
        n = int(n)
        total_count = int(total_count)
    except (TypeError, ValueError) as error:
        raise ValueError("Subset length and dataset length must be integers") from error
    if n < 0 or total_count < 0:
        raise ValueError("Subset length and dataset length must be non-negative")

    subset_count = min(n, total_count)
    cache_dir = _resolve(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{split}_{n}_seed{seed}.txt"

    expected_indices = list(range(total_count))
    random.Random(seed).shuffle(expected_indices)
    expected_indices = expected_indices[:subset_count]

    cached_indices = None
    if cache_path.is_file():
        try:
            cached_indices = [
                int(line.strip())
                for line in cache_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except ValueError:
            cached_indices = None

    # Comparing with the deterministic result also detects a cache created for
    # a different dataset length even when all of its indices remain in range.
    if cached_indices != expected_indices:
        cache_path.write_text(
            "".join(f"{index}\n" for index in expected_indices),
            encoding="utf-8",
        )
    return expected_indices


def _split_images(dataset_config, split):
    roots = {
        "calibration": dataset_config.train_images,
        "train": dataset_config.train_images,
        "validation": dataset_config.validation_images,
        "val": dataset_config.validation_images,
        "evaluation": dataset_config.validation_images,
    }
    try:
        return roots[split]
    except KeyError as error:
        raise ValueError(f"Unknown YOLOv26 dataset split {split!r}") from error


def build_calibration_dataset(
    model_config,
    dataset_config,
    split="calibration",
    subset_len=None,
    seed=42,
):
    """Build a target-free dataset for calibration forward passes."""
    images_dir = _split_images(dataset_config, split)
    normalization = {"mean": dataset_config.mean, "std": dataset_config.std}
    dataset = CalibrationImageDataset(
        images_dir=images_dir,
        input_shape=model_config.input_size,
        normalization=normalization,
    )
    if subset_len is not None:
        indices = build_or_load_subset_indices(
            split=split,
            n=subset_len,
            seed=seed,
            cache_dir=dataset_config.subset_cache_dir,
            dataset_length=len(dataset),
        )
        dataset.image_files = [dataset.image_files[index] for index in indices]
    return dataset


def build_calibration_loader(
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

    dataset = build_calibration_dataset(
        model_config,
        dataset_config,
        split=split,
        subset_len=subset_len,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


# Keep the generic names parallel with ``kria_ai.classification.data``.
build_dataset = build_calibration_dataset
build_loader = build_calibration_loader
