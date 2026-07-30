"""
src/callbacks.py
================

Purpose
-------
Factory functions that build the standard set of Keras callbacks used
during training.  Keeping them in one place lets both training phases
(phase-1 frozen, phase-2 fine-tune) reuse the same configuration.

Callbacks explained
-------------------
* **EarlyStopping** – stop training when a monitored metric stops
  improving.  Saves time and prevents over-fitting.
* **ReduceLROnPlateau** – shrink the learning rate by a factor when
  the loss plateaus, helping the model settle into a sharper minimum.
* **ModelCheckpoint** – save the best model (by val accuracy) so we
  always keep the strongest weights.
* **CSVLogger** – append per-epoch metrics to a CSV file for later
  plotting / comparison.
* **TensorBoard** – live training curves and graph visualisation in
  the browser.
* **LearningRateScheduler** – optional step-decay schedule.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional

import tensorflow as tf

from src.config import Config
from src.utils import get_logger, now_str

logger = get_logger("callbacks")


def step_decay(epoch: int, lr: float, initial_lr: float = 1e-3, drop: float = 0.5, epochs_drop: int = 5) -> float:
    """Classic step-decay schedule: multiply LR by *drop* every *epochs_drop* epochs."""
    if epoch == 0:
        return initial_lr
    return initial_lr * math.pow(drop, math.floor(epoch / epochs_drop))


def build_callbacks(
    config: Config,
    phase: str = "phase1",
    checkpoint_path: Optional[Path] = None,
) -> List[tf.keras.callbacks.Callback]:
    """Return the standard callback list for a training phase.

    Parameters
    ----------
    config : Config
    phase : str
        ``"phase1"`` or ``"phase2"`` – used for file naming.
    checkpoint_path : Path, optional
        Where to save the best model.  Defaults to
        ``checkpoints/<phase>_<timestamp>.keras``.
    """
    config.ensure_directories()
    ts = now_str()

    if checkpoint_path is None:
        checkpoint_path = Path(config.paths.checkpoints) / f"{phase}_best_{ts}.keras"
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    csv_path = Path(config.paths.logs) / f"{phase}_metrics_{ts}.csv"
    tb_dir = Path(config.paths.logs) / f"tensorboard_{phase}_{ts}"

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=config.training.early_stopping_patience,
            restore_best_weights=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.training.reduce_lr_factor,
            patience=config.training.reduce_lr_patience,
            min_lr=config.training.min_lr,
            mode="min",
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(csv_path), append=False),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(tb_dir),
            histogram_freq=1,
            write_graph=False,
            update_freq="epoch",
        ),
    ]

    # Optional LR scheduler (step decay).
    if phase == "phase2":
        initial_lr = config.training.phase2_lr
        callbacks.append(
            tf.keras.callbacks.LearningRateScheduler(
                lambda epoch, lr: step_decay(epoch, lr, initial_lr=initial_lr),
                verbose=1,
            )
        )

    logger.info("Built %d callbacks for %s", len(callbacks), phase)
    return callbacks
