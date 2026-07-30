"""
src/visualization.py
=====================

Purpose
-------
Plotting utilities for training curves, confusion matrices, sample
images, prediction visualisations and confidence analysis.

All functions save figures to disk (Agg backend) and return the figure
path so callers can log it.  We avoid ``plt.show()`` because the
training pipeline runs headless on a server.

Visualisation concepts
---------------------
* **Training curves** – plot loss/accuracy vs epoch to spot
  over-fitting (train improves, val plateaus) or under-fitting (both
  high).  A gap between train and val curves that grows with epochs is
  the classic over-fitting signature.
* **Confusion matrix** – heatmap of true vs predicted class.  Bright
  off-diagonal cells reveal which landmark pairs the model confuses.
* **Prediction gallery** – grid of images with their predicted label,
  true label and confidence.  Green border = correct, red = wrong.
  This is the most intuitive way to understand model behaviour.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Use Agg backend by default for headless rendering.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.config import Config
from src.utils import get_logger

logger = get_logger("visualization")


# ---------------------------------------------------------------------------
# Training history
# ---------------------------------------------------------------------------
def plot_training_history(
    histories: Dict[str, "tf.keras.callbacks.History"],
    config: Config,
    save_dir: Optional[Path] = None,
) -> Path:
    """Plot accuracy, loss, precision and recall curves for each phase.

    Parameters
    ----------
    histories : dict
        ``{"phase1": History, "phase2": History}``.
    """
    if save_dir is None:
        save_dir = Path(config.paths.plots)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    metrics_to_plot = ["accuracy", "loss", "precision", "recall"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics_to_plot):
        for phase, hist in histories.items():
            h = hist.history
            epochs = range(1, len(h.get(metric, [])) + 1)
            ax.plot(epochs, h.get(metric, []), label=f"{phase} train")
            val_key = f"val_{metric}"
            ax.plot(epochs, h.get(val_key, []), "--", label=f"{phase} val")
        ax.set_title(metric.capitalize())
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.capitalize())
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Training History", fontsize=14)
    fig.tight_layout()
    out = save_dir / "training_history.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info("Saved training history plot to %s", out)
    return out


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------
def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Path,
    top_k: int = 30,
    normalize: bool = True,
) -> Path:
    """Plot a confusion-matrix heatmap (top-k classes by support)."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Subsample to the most populous classes for readability.
    if cm.shape[0] > top_k:
        support = cm.sum(axis=1)
        idx = np.argsort(-support)[:top_k]
        cm = cm[np.ix_(idx, idx)]
        class_names = [class_names[i] for i in idx]

    cm_display = cm.astype(float)
    if normalize:
        row_sums = cm_display.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_display = cm_display / row_sums

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_display, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (normalised)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    logger.info("Saved confusion matrix to %s", save_path)
    return save_path


# ---------------------------------------------------------------------------
# Sample images
# ---------------------------------------------------------------------------
def plot_sample_images(
    df,
    base_dir: Path,
    n: int = 16,
    save_path: Optional[Path] = None,
    config: Optional[Config] = None,
) -> Optional[Path]:
    """Plot a grid of sample images with their landmark id."""
    from src.preprocessing import load_image

    n = min(n, len(df))
    sample = df.sample(n, random_state=42)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).flatten()
    for ax, (_, row) in zip(axes, sample.iterrows()):
        path = row.get("path")
        img = load_image(path) if path else None
        if img is not None:
            ax.imshow(img.astype(np.uint8))
        ax.set_title(f"LM {row['landmark_id']}", fontsize=8)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    if save_path is None:
        if config is not None:
            save_path = Path(config.paths.figures) / "sample_images.png"
        else:
            return None
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    logger.info("Saved sample images to %s", save_path)
    return save_path


# ---------------------------------------------------------------------------
# Prediction gallery
# ---------------------------------------------------------------------------
def visualize_predictions(
    dataset,
    model,
    encoder,
    config: Config,
    save_path: Path,
    n: int = 16,
) -> Path:
    """Visualise a grid of images with predicted/true labels and confidence.

    Correct predictions get a green title; wrong ones get red.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    images_batch, labels_batch = next(iter(dataset.take(1)))
    probs = model.predict(images_batch, verbose=0)
    preds = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)

    n = min(n, len(images_batch))
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).flatten()

    for i, ax in enumerate(axes[:n]):
        img = images_batch[i].numpy()
        # Denormalise for display (tf mode: [0,1])
        if config.image.normalize_mode == "tf":
            img = np.clip(img, 0, 1)
        ax.imshow(img)
        true_lm = encoder.inverse_transform([labels_batch[i]])[0]
        pred_lm = encoder.inverse_transform([preds[i]])[0]
        correct = preds[i] == labels_batch[i]
        color = "green" if correct else "red"
        ax.set_title(
            f"True: {true_lm}\nPred: {pred_lm}\nConf: {confidences[i]:.2f}",
            fontsize=7, color=color,
        )
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    logger.info("Saved prediction gallery to %s", save_path)
    return save_path


# ---------------------------------------------------------------------------
# Top-5 predictions for a single image
# ---------------------------------------------------------------------------
def plot_top5_predictions(
    image: np.ndarray,
    probs: np.ndarray,
    encoder,
    save_path: Optional[Path] = None,
) -> Optional[Path]:
    """Bar chart of the top-5 predicted classes for one image."""
    top5_idx = np.argsort(probs)[-5:][::-1]
    top5_probs = probs[top5_idx]
    labels = [str(encoder.inverse_transform([i])[0]) for i in top5_idx]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.imshow(np.clip(image, 0, 1))
    ax1.axis("off")
    ax1.set_title("Input image")
    ax2.barh(range(5), top5_probs, color="steelblue")
    ax2.set_yticks(range(5))
    ax2.set_yticklabels(labels, fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Probability")
    ax2.set_title("Top-5 Predictions")
    fig.tight_layout()
    if save_path is None:
        plt.close(fig)
        return None
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return save_path
