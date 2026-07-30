"""
src/config.py
=============

Purpose
-------
Central, configuration-driven settings for the entire landmark-detection
project.  Every other module imports the :class:`Config` dataclass instead
of reading hard-coded values, so hyper-parameters can be changed from a
single YAML file (``config.yaml``) without touching source code.

Design decisions
----------------
* **Dataclass + YAML** – gives type hints, IDE auto-complete and a clean
  ``cfg.training.batch_size`` API while staying serialisable.
* **Single source of truth** – the module exposes a process-wide
  ``get_config()`` singleton so there is never more than one parsed copy.
* **Path resolution** – relative paths in the YAML are resolved against
  the project root, which is discovered from this file's location.  This
  makes the project runnable from any working directory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


# ---------------------------------------------------------------------------
# Project root helper
# ---------------------------------------------------------------------------
def get_project_root() -> Path:
    """Return the absolute path to the project root.

    ``config.py`` lives in ``<root>/src/config.py`` so the root is two
    parents up.  We also fall back to the current working directory when
    the file is executed in unusual environments (e.g. interactive
    notebooks) where ``__file__`` may be absent.
    """
    try:
        return Path(__file__).resolve().parent.parent
    except NameError:  # pragma: no cover - notebook fallback
        return Path.cwd()


# ---------------------------------------------------------------------------
# Nested dataclasses for grouped settings
# ---------------------------------------------------------------------------
@dataclass
class PathsConfig:
    data_raw: str = "data/raw"
    data_processed: str = "data/processed"
    train_dir: str = "data/train"
    val_dir: str = "data/val"
    test_dir: str = "data/test"
    models: str = "models"
    checkpoints: str = "checkpoints"
    outputs: str = "outputs"
    figures: str = "outputs/figures"
    plots: str = "outputs/plots"
    logs: str = "logs"
    train_csv: str = "data/raw/train.csv"
    encoder_file: str = "models/label_encoder.joblib"

    def resolve(self, root: Path) -> "PathsConfig":
        """Convert every relative path to an absolute ``Path`` object."""
        for f in vars(self):
            setattr(self, f, (root / getattr(self, f)).resolve())
        return self


@dataclass
class ImageConfig:
    height: int = 224
    width: int = 224
    channels: int = 3
    mean: tuple = (103.939, 116.779, 123.68)
    normalize_mode: str = "tf"
    interpolation: str = "bilinear"

    @property
    def target_size(self) -> tuple:
        return (self.height, self.width)


@dataclass
class DatasetConfig:
    top_k_classes: int = 50
    min_samples_per_class: int = 40
    max_samples_per_class: int = 500
    val_split: float = 0.15
    test_split: float = 0.10
    shuffle_buffer: int = 2000


@dataclass
class TrainingConfig:
    batch_size: int = 32
    epochs_phase1: int = 15
    epochs_phase2: int = 10
    phase1_lr: float = 1e-3
    phase2_lr: float = 1e-4
    optimizer: str = "adam"
    sgd_momentum: float = 0.9
    sgd_nesterov: bool = True
    early_stopping_patience: int = 5
    reduce_lr_patience: int = 3
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-5
    mixed_precision: bool = False
    fine_tune_layers: int = 4
    cache_dataset: bool = True
    prefetch_buffer: str = "AUTOTUNE"


@dataclass
class ModelConfig:
    backbone: str = "vgg19"
    weights: str = "imagenet"
    include_top: bool = False
    dropout: float = 0.5
    dense_units: int = 1024
    use_batchnorm: bool = True
    use_global_avg_pool: bool = True
    label_smoothing: float = 0.1


@dataclass
class AugmentationConfig:
    enabled: bool = True
    random_flip: str = "horizontal"
    random_rotation: float = 20.0
    random_zoom: float = 0.2
    random_width_shift: float = 0.2
    random_height_shift: float = 0.2
    random_brightness: float = 0.2
    random_contrast: float = 0.2


@dataclass
class EvaluationConfig:
    top_k: int = 5


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """Aggregate every sub-config plus global scalars."""

    seed: int = 42
    project_name: str = "Google Landmark Detection"
    description: str = ""
    paths: PathsConfig = field(default_factory=PathsConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    # runtime-only
    root: Path = field(default_factory=get_project_root)

    # ------------------------------------------------------------------
    # Construction from YAML
    # ------------------------------------------------------------------
    @classmethod
    def from_yaml(cls, yaml_path: Optional[Path] = None) -> "Config":
        """Parse ``config.yaml`` and build a fully-resolved :class:`Config`.

        Parameters
        ----------
        yaml_path:
            Optional explicit path.  When *None* the default
            ``<root>/config.yaml`` is used.
        """
        root = get_project_root()
        if yaml_path is None:
            yaml_path = root / "config.yaml"
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {yaml_path}. "
                "Copy config.yaml.example to config.yaml and edit it."
            )

        with open(yaml_path, "r", encoding="utf-8") as fh:
            raw: Dict[str, Any] = yaml.safe_load(fh) or {}

        cfg = cls(
            seed=raw.get("seed", 42),
            project_name=raw.get("project_name", ""),
            description=raw.get("description", ""),
            paths=PathsConfig(**raw.get("paths", {})),
            image=ImageConfig(**raw.get("image", {})),
            dataset=DatasetConfig(**raw.get("dataset", {})),
            training=TrainingConfig(**raw.get("training", {})),
            model=ModelConfig(**raw.get("model", {})),
            augmentation=AugmentationConfig(**raw.get("augmentation", {})),
            evaluation=EvaluationConfig(**raw.get("evaluation", {})),
            root=root,
        )
        cfg.paths.resolve(root)
        return cfg

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def ensure_directories(self) -> None:
        """Create every output directory referenced in the config."""
        dir_attrs = [
            "data_raw", "data_processed", "train_dir", "val_dir", "test_dir",
            "models", "checkpoints", "outputs", "figures", "plots", "logs",
        ]
        for attr in dir_attrs:
            p: Path = getattr(self.paths, attr)
            p.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise back to a plain dict (handy for logging)."""
        import dataclasses
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Process-wide singleton
# ---------------------------------------------------------------------------
_CONFIG: Optional[Config] = None


def get_config(yaml_path: Optional[Path] = None) -> Config:
    """Return the cached :class:`Config`, loading it on first call."""
    global _CONFIG
    if _CONFIG is None or yaml_path is not None:
        _CONFIG = Config.from_yaml(yaml_path)
    return _CONFIG


def reset_config() -> None:
    """Clear the cache – mainly used by tests."""
    global _CONFIG
    _CONFIG = None
