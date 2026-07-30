"""
src/dataset.py
==============

Purpose
-------
End-to-end dataset management for Google Landmark Detection:

1. **Loading** – read ``train.csv`` metadata (image_id, landmark_id).
2. **EDA**    – exploratory data analysis: class counts, imbalance,
   dimension stats, missing/duplicate detection, sample visualisation.
3. **Filtering** – keep the top-K most frequent classes and cap
   per-class samples to fight imbalance.
4. **Label encoding** – ``LabelEncoder`` mapping landmark_id →
   contiguous integers [0, num_classes).
5. **Splitting** – stratified train / val / test split.
6. **tf.data pipeline** – efficient, memory-friendly input pipeline
   with caching, shuffling, augmentation, prefetching and parallel
   loading.

Dataset background
------------------
The Google Landmarks dataset v2 contains ~5M images spanning ~200k
landmark classes.  It is *extremely* long-tailed: a few landmarks have
thousands of photos while most have fewer than ten.  Training on the
full set is impractical on commodity hardware, so we keep the top-K
classes (configurable) and cap samples per class.  This produces a
manageable, reasonably balanced subset that still preserves the
real-world challenges of landmark recognition.

CSV columns
-----------
* ``id``          – unique image identifier (string).
* ``landmark_id`` – integer landmark class label.

The image files live in a 3-level folder hierarchy under
``data/raw`` (see :func:`preprocessing.build_image_path`).
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf

from src.config import Config
from src.preprocessing import build_image_path, load_image, normalize_image
from src.utils import get_logger, save_encoder, load_encoder

logger = get_logger("dataset")


# ---------------------------------------------------------------------------
# 1. CSV loading
# ---------------------------------------------------------------------------
def load_metadata(csv_path: Path | str) -> pd.DataFrame:
    """Load the landmark metadata CSV into a DataFrame.

    Expected columns: ``id`` (image id), ``landmark_id`` (class).
    Missing files raise a clear error.
    """
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(
            f"Metadata CSV not found: {p}. Download the Google Landmarks "
            "dataset and place train.csv under data/raw/."
        )
    df = pd.read_csv(p)
    required = {"id", "landmark_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    logger.info("Loaded metadata: %d rows from %s", len(df), p.name)
    return df


# ---------------------------------------------------------------------------
# 2. Exploratory data analysis
# ---------------------------------------------------------------------------
def exploratory_data_analysis(
    df: pd.DataFrame,
    base_dir: Optional[Path] = None,
    sample_size: int = 500,
    save_dir: Optional[Path] = None,
) -> Dict:
    """Run a complete EDA pass and return a stats dictionary.

    Computes:
    * total images, unique classes, missing values, duplicates.
    * per-class frequency and imbalance ratio.
    * image dimension statistics on a random sample (if base_dir given).
    * optionally saves distribution plots.

    The returned dict is handy for logging and for the notebook.
    """
    stats: Dict = {}
    stats["total_images"] = len(df)
    stats["unique_landmarks"] = df["landmark_id"].nunique()
    stats["missing_values"] = int(df.isnull().sum().sum())
    stats["duplicate_rows"] = int(df.duplicated().sum())

    class_counts = df["landmark_id"].value_counts()
    stats["class_counts"] = class_counts
    stats["max_class_size"] = int(class_counts.max())
    stats["min_class_size"] = int(class_counts.min())
    stats["imbalance_ratio"] = (
        class_counts.max() / max(class_counts.min(), 1)
    )
    stats["mean_class_size"] = float(class_counts.mean())
    stats["median_class_size"] = float(class_counts.median())

    logger.info(
        "EDA: %d images, %d classes, imbalance ratio %.1f",
        stats["total_images"], stats["unique_landmarks"],
        stats["imbalance_ratio"],
    )

    # Image dimension stats (sample)
    if base_dir is not None:
        sample_ids = df["id"].sample(min(sample_size, len(df)), random_state=42)
        dims = []
        for img_id in sample_ids:
            path = build_image_path(img_id, base_dir)
            if path.exists():
                import cv2
                img = cv2.imread(str(path))
                if img is not None:
                    dims.append(img.shape[:2])
        if dims:
            dims = np.array(dims)
            stats["image_dimensions"] = {
                "mean_height": float(dims[:, 0].mean()),
                "mean_width": float(dims[:, 1].mean()),
                "min_height": int(dims[:, 0].min()),
                "min_width": int(dims[:, 1].min()),
                "max_height": int(dims[:, 0].max()),
                "max_width": int(dims[:, 1].max()),
            }
            logger.info(
                "Image dims (sample %d): mean %.0f×%.0f",
                len(dims), dims[:, 0].mean(), dims[:, 1].mean(),
            )

    # Save plots
    if save_dir is not None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            # Class distribution (top 50)
            top = class_counts.head(50)
            fig, ax = plt.subplots(figsize=(14, 5))
            ax.bar(range(len(top)), top.values)
            ax.set_title("Top-50 Landmark Class Distribution")
            ax.set_xlabel("Class index (sorted by frequency)")
            ax.set_ylabel("Number of images")
            fig.tight_layout()
            fig.savefig(save_dir / "class_distribution.png", dpi=120)
            plt.close(fig)

            # Log-scale histogram of class sizes
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(class_counts.values, bins=50, log=True)
            ax.set_title("Class Size Histogram (log scale)")
            ax.set_xlabel("Images per class")
            ax.set_ylabel("Number of classes")
            fig.tight_layout()
            fig.savefig(save_dir / "class_size_histogram.png", dpi=120)
            plt.close(fig)
            logger.info("Saved EDA plots to %s", save_dir)
        except Exception as e:
            logger.warning("Could not save EDA plots: %s", e)

    return stats


# ---------------------------------------------------------------------------
# 3. Filtering & balancing
# ---------------------------------------------------------------------------
def filter_top_k_classes(
    df: pd.DataFrame,
    top_k: int,
    min_samples: int = 1,
    max_samples: Optional[int] = None,
) -> pd.DataFrame:
    """Keep only the top-K most frequent landmark classes.

    Optionally drop classes with fewer than *min_samples* images and cap
    each class at *max_samples* (random subsample) to reduce imbalance.
    """
    counts = df["landmark_id"].value_counts()
    top_ids = counts.head(top_k).index
    filtered = df[df["landmark_id"].isin(top_ids)].copy()

    if min_samples > 1:
        keep = counts[counts >= min_samples].index
        filtered = filtered[filtered["landmark_id"].isin(keep)]

    if max_samples is not None:
        parts = []
        for lid, grp in filtered.groupby("landmark_id"):
            if len(grp) > max_samples:
                grp = grp.sample(max_samples, random_state=42)
            parts.append(grp)
        filtered = pd.concat(parts, ignore_index=True)

    logger.info(
        "Filtered to %d classes, %d images (top_k=%d, min=%d, max=%s)",
        filtered["landmark_id"].nunique(), len(filtered),
        top_k, min_samples, max_samples,
    )
    return filtered.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. Label encoding
# ---------------------------------------------------------------------------
def encode_labels(
    df: pd.DataFrame,
    encoder_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, LabelEncoder]:
    """Fit a ``LabelEncoder`` on ``landmark_id`` and add an ``label`` column.

    Why neural networks need sequential integer labels
    ---------------------------------------------------
    A softmax output layer has one unit per class.  To compute the
    cross-entropy loss the network needs the *index* of the correct
    class, not the original (possibly large, non-contiguous) landmark
    id.  ``LabelEncoder`` maps the original ids to a dense range
    [0, num_classes) which matches the softmax units exactly.  Using
    ``SparseCategoricalCrossentropy`` we can pass these integers
    directly (no one-hot encoding needed), saving memory.
    """
    encoder = LabelEncoder()
    df = df.copy()
    df["label"] = encoder.fit_transform(df["landmark_id"])
    n_classes = len(encoder.classes_)
    logger.info("Encoded %d landmark classes → [0, %d)", n_classes, n_classes)

    if encoder_path is not None:
        save_encoder(encoder, encoder_path)
        logger.info("Saved label encoder to %s", encoder_path)

    return df, encoder


def decode_labels(labels, encoder: LabelEncoder):
    """Map encoded integer labels back to original landmark ids."""
    return encoder.inverse_transform(labels)


# ---------------------------------------------------------------------------
# 5. Path generation
# ---------------------------------------------------------------------------
def add_image_paths(df: pd.DataFrame, base_dir: Path | str) -> pd.DataFrame:
    """Add a ``path`` column with the on-disk image path for each row."""
    df = df.copy()
    df["path"] = df["id"].apply(lambda x: str(build_image_path(x, base_dir)))
    return df


# ---------------------------------------------------------------------------
# 6. Stratified split
# ---------------------------------------------------------------------------
def stratified_split(
    df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.10,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split *df* into train / val / test preserving class distribution.

    We use ``stratify`` so each subset has roughly the same class
    proportions as the full set.  This is critical for imbalanced
    datasets – a random split could leave some classes entirely out of
    the validation or test set.
    """
    train_val, test = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["label"],
    )
    train, val = train_test_split(
        train_val,
        test_size=val_size / (1 - test_size),
        random_state=seed,
        stratify=train_val["label"],
    )
    logger.info(
        "Split: train=%d, val=%d, test=%d", len(train), len(val), len(test)
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 7. tf.data pipeline
# ---------------------------------------------------------------------------
def _load_and_preprocess(
    path: tf.Tensor,
    label: tf.Tensor,
    config: Config,
    augment_model: Optional[tf.keras.Sequential] = None,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """tf.data-compatible image loader + preprocessor.

    Uses ``tf.py_function`` to call OpenCV inside the graph.  This
    keeps NumPy/OpenCV code reusable while staying in the efficient
    tf.data pipeline.
    """
    def _numpy_load(p: str) -> np.ndarray:
        img = load_image(p.decode("utf-8"), config.image.target_size)
        if img is None:
            img = np.zeros((*config.image.target_size, 3), dtype=np.float32)
        return img

    image = tf.py_function(_numpy_load, [path], tf.float32)
    image.set_shape((config.image.height, config.image.width, 3))

    # Normalise (tf mode → [0,1])
    if config.image.normalize_mode == "tf":
        image = image / 255.0
    elif config.image.normalize_mode == "scale":
        image = (image / 127.5) - 1.0
    elif config.image.normalize_mode == "caffe":
        mean = tf.constant(config.image.mean[::-1], dtype=tf.float32)  # BGR→RGB
        image = image - mean

    if augment_model is not None:
        image = augment_image(image, augment_model)

    return image, label


def augment_image(image: tf.Tensor, augment_model: tf.keras.Sequential) -> tf.Tensor:
    """Apply augmentation layers to a single image tensor (adds batch dim)."""
    image = tf.expand_dims(image, 0)
    image = augment_model(image, training=True)
    return tf.squeeze(image, axis=0)


def build_tf_dataset(
    df: pd.DataFrame,
    config: Config,
    augment_model: Optional[tf.keras.Sequential] = None,
    shuffle: bool = True,
    cache: bool = True,
) -> tf.data.Dataset:
    """Build an efficient ``tf.data.Dataset`` from a metadata DataFrame.

    Pipeline order (the classic "cache-shuffle-batch-prefetch" pattern):
    1. ``from_tensor_slices`` – (path, label) pairs.
    2. ``map``                – load + preprocess (+ augment).
    3. ``cache``              – cache preprocessed tensors in memory/disk.
    4. ``shuffle``            – randomise order (train only).
    5. ``batch``              – mini-batches.
    6. ``prefetch``           – overlap producer & consumer.

    Caching after the map step means the expensive OpenCV I/O and
    resizing happen only once per epoch.  Augmentation is *not* cached
    because it is stochastic – we apply it inside the map so it runs
    every epoch.
    """
    paths = df["path"].values.astype(str)
    labels = df["label"].values.astype(np.int32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    # Map with parallel calls for throughput.
    map_fn = lambda p, l: _load_and_preprocess(p, l, config, augment_model)
    ds = ds.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)

    if cache:
        ds = ds.cache()

    if shuffle:
        ds = ds.shuffle(buffer_size=config.dataset.shuffle_buffer)

    ds = ds.batch(config.training.batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ---------------------------------------------------------------------------
# 8. Convenience: full preparation
# ---------------------------------------------------------------------------
def prepare_dataset(config: Config) -> Dict[str, tf.data.Dataset]:
    """Run the full preparation pipeline and return ready-to-train datasets.

    Steps:
    1. Load CSV.
    2. EDA (save plots).
    3. Filter top-K classes.
    4. Encode labels.
    5. Add image paths.
    6. Stratified split.
    7. Build tf.data pipelines (augmentation on train only).
    """
    config.ensure_directories()

    df = load_metadata(config.paths.train_csv)
    stats = exploratory_data_analysis(
        df, base_dir=Path(config.paths.data_raw), save_dir=Path(config.paths.figures),
    )

    df = filter_top_k_classes(
        df,
        top_k=config.dataset.top_k_classes,
        min_samples=config.dataset.min_samples_per_class,
        max_samples=config.dataset.max_samples_per_class,
    )

    df, encoder = encode_labels(df, encoder_path=Path(config.paths.encoder_file))
    df = add_image_paths(df, base_dir=Path(config.paths.data_raw))

    train_df, val_df, test_df = stratified_split(
        df,
        val_size=config.dataset.val_split,
        test_size=config.dataset.test_split,
        seed=config.seed,
    )

    # Save splits for reproducibility / debugging.
    for name, sub in [("train", train_df), ("val", val_df), ("test", test_df)]:
        sub.to_csv(config.paths.data_processed / f"{name}_split.csv", index=False)

    # Augmentation model (training only).
    from src.augmentation import build_augmentation_layers
    augment_model = build_augmentation_layers(config) if config.augmentation.enabled else None

    train_ds = build_tf_dataset(
        train_df, config, augment_model=augment_model,
        shuffle=True, cache=config.training.cache_dataset,
    )
    val_ds = build_tf_dataset(
        val_df, config, augment_model=None,
        shuffle=False, cache=config.training.cache_dataset,
    )
    test_ds = build_tf_dataset(
        test_df, config, augment_model=None,
        shuffle=False, cache=config.training.cache_dataset,
    )

    return {
        "train": train_ds,
        "val": val_ds,
        "test": test_ds,
        "encoder": encoder,
        "num_classes": len(encoder.classes_),
        "stats": stats,
        "splits": {"train": train_df, "val": val_df, "test": test_df},
    }
