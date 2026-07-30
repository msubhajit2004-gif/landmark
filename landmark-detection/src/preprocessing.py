"""
src/preprocessing.py
====================

Purpose
-------
Robust, reusable image preprocessing for the landmark-detection
pipeline.  Every function here is pure (no global state) and defensive:
corrupt files are skipped, missing paths are handled, and unexpected
exceptions never crash a training run.

Key operations
--------------
1. **Path generation** – Google Landmarks images are stored in a
   3-level folder hierarchy derived from the first characters of the
   image id (e.g. ``0a/1b/2c/0a1b2c....jpg``).  ``build_image_path``
   reconstructs that path from the id.
2. **Safe reading** – ``load_image`` uses OpenCV, validates the file,
   converts BGR→RGB, and returns ``None`` for corrupt inputs.
3. **Resizing** – every image is resized to 224×224 (VGG19 input size).
4. **Normalisation** – three modes:
   * ``"tf"``   – scale to [0, 1] (default for modern transfer learning).
   * ``"caffe"``– subtract ImageNet mean in BGR order (original VGG style).
   * ``"scale"``– scale to [-1, 1].
5. **Batch helpers** – ``preprocess_single`` wraps the steps for a
   ``tf.data`` map function.

Why not use Keras' built-in ``ImageDataGenerator``?
---------------------------------------------------
The generator hides too much logic and is hard to combine with
``tf.data``.  Explicit functions give us caching control, better error
handling, and the ability to mix NumPy/OpenCV operations with TF
augmentation layers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from src.config import Config
from src.utils import get_logger, validate_image_file

logger = get_logger("preprocessing")


# ---------------------------------------------------------------------------
# Path generation from image id
# ---------------------------------------------------------------------------
def build_image_path(image_id: str, base_dir: Path | str, extension: str = "jpg") -> Path:
    """Reconstruct the on-disk path of a Google-Landmarks image.

    The dataset stores images in a 3-level hierarchy where each level
    is the first three characters of the image id::

        base_dir/0a/1b/2c/0a1b2c3d....jpg

    Parameters
    ----------
    image_id : str
        Unique identifier without extension (e.g. ``"0a1b2c3d..."``).
    base_dir : Path
        Root directory that contains the hierarchy.
    extension : str
        File extension to append (default ``"jpg"``).

    Returns
    -------
    Path
        Full path to the image file.  Existence is *not* checked here –
        callers decide whether to skip missing files.
    """
    if not image_id or len(image_id) < 3:
        raise ValueError(f"Invalid image_id: {image_id!r}")
    parts = [image_id[i] for i in range(3)]
    return Path(base_dir).joinpath(*parts, f"{image_id}.{extension}")


# ---------------------------------------------------------------------------
# Single-image loading
# ---------------------------------------------------------------------------
def load_image(
    path: Path | str,
    target_size: Tuple[int, int] = (224, 224),
    interpolation: int = cv2.INTER_LINEAR,
) -> Optional[np.ndarray]:
    """Safely read an image, convert BGR→RGB, and resize.

    Returns ``None`` for missing or corrupt files so that callers can
    filter bad samples without try/except boilerplate.
    """
    p = Path(path)
    if not p.exists():
        logger.debug("Missing image: %s", p)
        return None
    try:
        # cv2.imread returns None for corrupt files.
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            logger.debug("Corrupt image (unreadable): %s", p)
            return None
        # unreadable
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, target_size, interpolation=interpolation)
        return rgb.astype(np.float32)
    except cv2.error as e:
        logger.debug("OpenCV error on %s: %s", p, e)
        return None
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("Unexpected error loading %s: %s", p, e)
        return None


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def normalize_image(
    image: np.ndarray,
    mode: str = "tf",
    mean: Tuple[float, float, float] = (103.939, 116.779, 123.68),
) -> np.ndarray:
    """Normalise pixel values according to *mode*.

    Parameters
    ----------
    image : np.ndarray
        Float32 image, values in [0, 255].
    mode : str
        * ``"tf"``    – divide by 255.0 → [0, 1].
        * ``"caffe"`` – subtract ImageNet mean (BGR order, no scaling).
        * ``"scale"`` – (img / 127.5) - 1 → [-1, 1].
    mean : tuple
        Only used when ``mode == "caffe"``.

    Returns
    -------
    np.ndarray
        Normalised float32 image.
    """
    if image.dtype != np.float32:
        image = image.astype(np.float32)

    if mode == "tf":
        return image / 255.0
    if mode == "caffe":
        # VGG19 was trained with Caffe-style mean subtraction in BGR.
        # Our images are RGB, so we subtract in RGB order by reversing mean.
        mean_rgb = np.array([mean[2], mean[1], mean[0]], dtype=np.float32)
        return image - mean_rgb
    if mode == "scale":
        return (image / 127.5) - 1.0
    raise ValueError(f"Unknown normalisation mode: {mode!r}")


# ---------------------------------------------------------------------------
# Full single-image pipeline (for tf.data / inference)
# ---------------------------------------------------------------------------
def preprocess_single(
    image_path: str,
    config: Config,
    augment_fn=None,
) -> Tuple[np.ndarray, None]:
    """Load + resize + normalise a single image by path.

    Designed to be wrapped in ``tf.data.Dataset.map`` via
    ``tf.py_function``.  If the image is corrupt it returns a black
    placeholder so the pipeline never breaks; callers should filter
    such rows beforehand.
    """
    img = load_image(image_path, config.image.target_size)
    if img is None:
        img = np.zeros((*config.image.target_size, 3), dtype=np.float32)
    img = normalize_image(img, mode=config.image.normalize_mode, mean=tuple(config.image.mean))
    return img, None


# ---------------------------------------------------------------------------
# Batch preprocessing (NumPy)
# ---------------------------------------------------------------------------
def preprocess_batch(
    paths: list,
    config: Config,
) -> np.ndarray:
    """Preprocess a list of image paths into a single NumPy batch.

    Useful for evaluation scripts that prefer plain NumPy arrays over
    ``tf.data``.  Corrupt images are replaced with zeros.
    """
    batch = np.zeros(
        (len(paths), config.image.height, config.image.width, 3),
        dtype=np.float32,
    )
    for i, p in enumerate(paths):
        img = load_image(p, config.image.target_size)
        if img is not None:
            img = normalize_image(
                img, mode=config.image.normalize_mode, mean=tuple(config.image.mean)
            )
            batch[i] = img
    return batch


# ---------------------------------------------------------------------------
# Public convenience: filter a DataFrame to existing, valid images
# ---------------------------------------------------------------------------
def filter_valid_images(
    df,
    base_dir: Path | str,
    path_col: str = "path",
    sample_limit: Optional[int] = None,
) -> "pd.DataFrame":
    """Return a copy of *df* with only rows whose image file is valid.

    Uses :func:`validate_image_file` for robust checking.  When
    *sample_limit* is given, at most that many rows are checked (useful
    for quick EDA on huge CSVs).
    """
    import pandas as pd
    df = df.copy()
    if sample_limit is not None:
        df = df.head(sample_limit)
    mask = df[path_col].apply(lambda p: validate_image_file(p))
    n_bad = (~mask).sum()
    if n_bad:
        logger.info("Filtered %d invalid/corrupt images", n_bad)
    return df[mask].reset_index(drop=True)
