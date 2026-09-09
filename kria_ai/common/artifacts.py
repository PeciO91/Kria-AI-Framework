import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ArtifactPaths:
    model_id: str
    build_root: Path = Path("build")

    def __post_init__(self):
        if not self.model_id or Path(self.model_id).name != self.model_id:
            raise ValueError(f"Invalid model ID for artifact paths: {self.model_id!r}")
        object.__setattr__(self, "build_root", Path(self.build_root))

    @property
    def model_dir(self):
        return self.build_root / self.model_id

    @property
    def inspector_dir(self):
        return self.model_dir / "inspector_report"

    @property
    def optimizer_dir(self):
        return self.model_dir / "optimizer_report"

    @property
    def quantize_dir(self):
        return self.model_dir / "quantize_result"

    @property
    def compiled_dir(self):
        return self.model_dir / "compiled"

    @property
    def quantized_xmodel(self):
        return self.quantize_dir / f"{self.model_id}_int.xmodel"

    @property
    def compiled_xmodel(self):
        return self.compiled_dir / f"{self.model_id}_kria.xmodel"

    def ensure(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        return path


def write_manifest(path: Path, payload: dict[str, Any]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
