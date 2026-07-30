"""
src/train.py
============

Purpose
-------
Orchestrates the full two-phase training workflow:

1. **Prepare data** – load CSV, EDA, filter, encode, split, build
   ``tf.data`` pipelines.
2. **Phase 1 – frozen backbone** – train only the custom classifier
   head with a larger learning rate.
3. **Phase 2 – fine-tuning** – unfreeze the last few VGG19 layers and
   retrain with a 10× smaller learning rate.
4. **Save artefacts** – best model, final model, label encoder,
   training history plots.

Run from the command line::

    python -m src.train            # uses config.yaml
    python -m src.train --epochs 20

Training concepts
-----------------
* **Mini-batch gradient descent** – we update weights on small batches
  (32 images) rather than the full dataset.  This is a trade-off
  between the stability of full-batch and the speed of stochastic
  gradient descent.
* **Validation monitoring** – after each epoch we evaluate on a held-out
  validation set to detect over-fitting early.
* **Checkpointing** – the best model (by val accuracy) is saved
  automatically so a crash never loses progress.
* **Automatic resume** – if a checkpoint exists we can resume from it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import tensorflow as tf

from src.config import Config, get_config
from src.dataset import prepare_dataset
from src.model import build_model, compile_model, unfreeze_backbone
from src.callbacks import build_callbacks
from src.utils import get_logger, set_seed, detect_hardware, save_model

logger = get_logger("train")


# ---------------------------------------------------------------------------
# History helpers
# ---------------------------------------------------------------------------
def save_history(history: tf.keras.callbacks.History, path: Path) -> None:
    """Serialise a training history object to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(history.history, fh, indent=2)
    logger.info("Saved training history to %s", path)


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------
def train_phase1(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    config: Config,
) -> tf.keras.callbacks.History:
    """Phase 1: train only the custom head (backbone frozen)."""
    logger.info("=" * 60)
    logger.info("PHASE 1: Frozen backbone, training classifier head")
    logger.info("=" * 60)

    callbacks = build_callbacks(config, phase="phase1")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.training.epochs_phase1,
        callbacks=callbacks,
        verbose=1,
    )
    save_history(history, Path(config.paths.logs) / "history_phase1.json")
    return history


def train_phase2(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    config: Config,
) -> tf.keras.callbacks.History:
    """Phase 2: fine-tune the last VGG19 layers with a smaller LR."""
    logger.info("=" * 60)
    logger.info("PHASE 2: Fine-tuning last %d backbone layers",
                config.training.fine_tune_layers)
    logger.info("=" * 60)

    model = unfreeze_backbone(model, config)
    callbacks = build_callbacks(config, phase="phase2")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.training.epochs_phase2,
        callbacks=callbacks,
        verbose=1,
    )
    save_history(history, Path(config.paths.logs) / "history_phase2.json")
    return history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(
    config_path: Optional[str] = None,
    epochs_phase1: Optional[int] = None,
    epochs_phase2: Optional[int] = None,
    skip_finetune: bool = False,
) -> None:
    """Run the complete training pipeline.

    Parameters
    ----------
    config_path : str, optional
        Path to a YAML config file.  Defaults to ``config.yaml``.
    epochs_phase1, epochs_phase2 : int, optional
        Override the config values from the command line.
    skip_finetune : bool
        If True, stop after phase 1.
    """
    config = get_config(Path(config_path) if config_path else None)
    if epochs_phase1 is not None:
        config.training.epochs_phase1 = epochs_phase1
    if epochs_phase2 is not None:
        config.training.epochs_phase2 = epochs_phase2

    # Reproducibility & hardware
    set_seed(config.seed)
    strategy, gpus = detect_hardware(config.training.mixed_precision)
    logger.info("Hardware: %s (GPUs: %s)", strategy, gpus)

    # Data
    data = prepare_dataset(config)
    train_ds, val_ds = data["train"], data["val"]
    num_classes = data["num_classes"]
    logger.info("Classes: %d", num_classes)

    # Model
    model = build_model(config, num_classes)
    compile_model(model, config, lr=config.training.phase1_lr)
    model.summary(print_fn=logger.info)

    # Phase 1
    hist1 = train_phase1(model, train_ds, val_ds, config)

    # Phase 2
    if not skip_finetune:
        hist2 = train_phase2(model, train_ds, val_ds, config)
    else:
        hist2 = None

    # Save final model
    final_path = Path(config.paths.models) / "landmark_vgg19_final.keras"
    save_model(model, final_path)
    logger.info("Saved final model to %s", final_path)

    # Plot training curves
    try:
        from src.visualization import plot_training_history
        histories = {"phase1": hist1}
        if hist2:
            histories["phase2"] = hist2
        plot_training_history(histories, config)
    except Exception as e:
        logger.warning("Could not plot training history: %s", e)

    logger.info("Training complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train landmark detection model")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--epochs1", type=int, default=None, help="Phase 1 epochs override")
    parser.add_argument("--epochs2", type=int, default=None, help="Phase 2 epochs override")
    parser.add_argument("--no-finetune", action="store_true", help="Skip fine-tuning phase")
    args = parser.parse_args()

    main(
        config_path=args.config,
        epochs_phase1=args.epochs1,
        epochs_phase2=args.epochs2,
        skip_finetune=args.no_finetune,
    )
