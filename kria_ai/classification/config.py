from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ClassificationModelConfig:
    id: str
    name: str
    constructor: str
    checkpoint_path: Path
    input_size: tuple[int, int]
    num_classes: int
    head_attribute: str

    def __post_init__(self):
        if self.num_classes < 2:
            raise ValueError(f"{self.id}: num_classes must be at least 2")
        if len(self.input_size) != 2 or min(self.input_size) <= 0:
            raise ValueError(f"{self.id}: invalid input size {self.input_size}")


@dataclass(frozen=True)
class ClassificationDatasetConfig:
    id: str
    name: str
    calibration_root: Path
    evaluation_root: Path
    board_root: Path
    classes: tuple[str, ...]
    mean: tuple[float, float, float]
    std: tuple[float, float, float]

    def __post_init__(self):
        if not self.classes:
            raise ValueError(f"{self.id}: classes cannot be empty")
        if len(self.mean) != 3 or len(self.std) != 3 or min(self.std) <= 0:
            raise ValueError(f"{self.id}: invalid normalization")


MODELS = {
    "resnet18": ClassificationModelConfig(
        id="resnet18",
        name="ResNet18",
        constructor="resnet18",
        checkpoint_path=Path("models/resnet18.pt"),
        input_size=(224, 224),
        num_classes=6,
        head_attribute="fc",
    ),
    "resnet50": ClassificationModelConfig(
        id="resnet50",
        name="ResNet50",
        constructor="resnet50",
        checkpoint_path=Path("models/resnet50.pt"),
        input_size=(224, 224),
        num_classes=6,
        head_attribute="fc",
    ),
    "mobilenet_v2": ClassificationModelConfig(
        id="mobilenet_v2",
        name="MobileNetV2",
        constructor="mobilenet_v2",
        checkpoint_path=Path("models/mobilenet_v2.pt"),
        input_size=(224, 224),
        num_classes=6,
        head_attribute="classifier",
    ),
    "mobilenet_v3": ClassificationModelConfig(
        id="mobilenet_v3",
        name="MobileNetV3-Large",
        constructor="mobilenet_v3_large",
        checkpoint_path=Path("models/mobilenet_v3.pt"),
        input_size=(224, 224),
        num_classes=6,
        head_attribute="classifier",
    ),
}

DATASETS = {
    "intel_images": ClassificationDatasetConfig(
        id="intel_images",
        name="Intel Image Classification",
        calibration_root=Path("data/intel_images/calibration_data"),
        evaluation_root=Path("data/intel_images/calibration_data"),
        board_root=Path("datasets/intel_images/train_data"),
        classes=("buildings", "forest", "glacier", "mountain", "sea", "street"),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
}

DEFAULT_MODEL_ID = "resnet18"
DEFAULT_DATASET_ID = "intel_images"


def get_model(model_id=None):
    resolved_id = model_id or DEFAULT_MODEL_ID
    try:
        return MODELS[resolved_id]
    except KeyError as error:
        raise ValueError(f"Unknown classification model {resolved_id!r}; available: {', '.join(MODELS)}") from error


def get_dataset(dataset_id=None):
    resolved_id = dataset_id or DEFAULT_DATASET_ID
    try:
        return DATASETS[resolved_id]
    except KeyError as error:
        raise ValueError(f"Unknown classification dataset {resolved_id!r}; available: {', '.join(DATASETS)}") from error


def validate_model_dataset(model, dataset):
    if model.num_classes != len(dataset.classes):
        raise ValueError(
            f"Model {model.id!r} has {model.num_classes} classes but dataset {dataset.id!r} has {len(dataset.classes)}"
        )
