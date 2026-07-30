"""
src/utils.py
============

Purpose
-------
Cross-cutting helpers used by every other module:

* **Logging** – a single configured logger that writes to both console
  and a rotating log file under ``logs/``.
* **Reproducibility** – set Python, NumPy and TensorFlow random seeds so
  experiments are deterministic and comparable.
* **Hardware detection** – discover available GPUs, enable mixed
  precision when requested, and fall back to CPU gracefully.
* **Persistence** – save / load Keras models and scikit-learn encoders
  (joblib) with defensive checks.
* **Path helpers** – ensure directories exist, validate image files.

Why a separate utils module?
----------------------------
Keeping these concerns in one place follows the *Single Responsibility
Principle*: dataset / model / training code does not need to know how
logging is configured or how seeds are set.
"""

from __future__ import annotations

import logging
import os
import random
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

# TensorFlow import is deferred inside functions where possible so that
# importing this module never forces TF initialisation (useful in
# lightweight scripts and notebooks).

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_LOGGER: Optional[logging.Logger] = None


def get_logger(name: str = "landmark", log_dir: Optional[Path] = None) -> logging.Logger:
    """Return a configured logger.

    The first caller creates a root logger that writes:

    * **Stream** – ``stdout`` at INFO level (coloured-ish prefix).
    * **File**   – ``logs/<timestamp>.log`` rotating at 5 MB, 3 backups.

    Subsequent calls with *name* return child loggers that inherit the
    handlers, so per-module log lines stay distinguishable.
    """
    global _LOGGER
    if _LOGGER is not None and name == "landmark":
        return _LOGGER

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if logger.handlers:  # already configured
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    if log_dir is None:
        log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = RotatingFileHandler(
        log_dir / f"run_{ts}.log",
        maxBytes=5_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    if name == "landmark":
        _LOGGER = logger
    return logger


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and TensorFlow for reproducible runs.

    Reproducibility matters in research and in production: it lets you
    compare two experiments fairly and debug an issue that only appears
    with a specific weight initialisation.  We set the seed on all
    three RNG sources and also disable TensorFlow's non-deterministic
    GPU op selection (at a performance cost – acceptable for dev).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        # Deterministic ops – slower but reproducible.
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:  # pragma: no cover - older TF versions
            pass
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Hardware detection & mixed precision
# ---------------------------------------------------------------------------
def detect_hardware(mixed_precision: bool = False) -> Tuple[str, list]:
    """Detect available compute devices and optionally enable mixed precision.

    Returns
    -------
    (strategy_name, devices)
        ``strategy_name`` is one of ``"gpu"``, ``"multi_gpu"`` or
        ``"cpu"``.  ``devices`` is the list of GPU device names (empty
        on CPU).

    Mixed precision (FP16 compute + FP32 master weights) can give a
    1.5-3× speed-up on modern NVIDIA GPUs (Tensor Cores).  It is disabled
    by default because on older GPUs or CPUs it can slow training and
    cause numerical instability.
    """
    try:
        import tensorflow as tf
    except ImportError:
        return "cpu", []

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return "cpu", []

    # Memory growth – avoid grabbing all GPU memory upfront.
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    if mixed_precision:
        try:
            from tensorflow.keras.mixed_precision import set_global_policy
            set_global_policy("mixed_float16")
        except Exception as e:  # pragma: no cover
            print(f"[utils] Mixed precision not enabled: {e}")

    strategy = "multi_gpu" if len(gpus) > 1 else "gpu"
    return strategy, [d.name for d in gpus]


def get_distribution_strategy() -> Any:
    """Return a TF distribution strategy for single- or multi-GPU training.

    On a single GPU (or CPU) we return *None* – the default strategy is
    fine.  With 2+ GPUs we use ``MirroredStrategy`` which mirrors
    variables across devices and synchronises gradients.
    """
    try:
        import tensorflow as tf
    except ImportError:
        return None

    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 1:
        return tf.distribute.MirroredStrategy()
    return None


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
def ensure_dir(path: Path | str) -> Path:
    """Create *path* (and parents) if it does not exist; return as ``Path``."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def validate_image_file(path: Path | str) -> bool:
    """Return *True* if *path* is a readable, non-corrupted image.

    We use OpenCV's ``imread`` which returns ``None`` for corrupt
    files, then verify the array has non-zero size.  This is faster and
    more robust than relying on file extensions alone.
    """
    import cv2
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return False
    try:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        return img is not None and img.size > 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Model / encoder persistence
# ---------------------------------------------------------------------------
def save_model(model, path: Path | str) -> Path:
    """Save a Keras model in the modern ``.keras`` format.

    The ``.keras`` format (TF 2.13+) stores architecture, weights and
    optimiser state in a single archive, which simplifies deployment.
    """
    ensure_dir(Path(path).parent)
    p = Path(path)
    if not p.suffix:
        p = p.with_suffix(".keras")
    model.save(p)
    return p


def load_model(path: Path | str, custom_objects: Optional[dict] = None):
    """Load a Keras model from disk with a defensive existence check."""
    from tensorflow.keras.models import load_model as _lm
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {p}")
    return _lm(str(p), custom_objects=custom_objects)


def save_encoder(encoder, path: Path | str) -> Path:
    """Persist a scikit-learn ``LabelEncoder`` with joblib."""
    import joblib
    ensure_dir(Path(path).parent)
    p = Path(path)
    joblib.dump(encoder, p)
    return p


def load_encoder(path: Path | str):
    """Load a joblib-persisted encoder, raising a clear error if absent."""
    import joblib
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Label encoder not found: {p}")
    return joblib.load(p)


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
def now_str() -> str:
    """Compact timestamp string for file naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def format_bytes(n: int) -> str:
    """Human-readable byte size (e.g. '1.4 GB')."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"
