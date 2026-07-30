"""
src/evaluate.py
===============

Purpose
-------
Evaluate a trained landmark-detection model on the validation and test
sets.  Produces a comprehensive report:

* Overall accuracy, top-5 accuracy, precision, recall, F1.
* Per-class precision / recall / F1 / support.
* Confusion matrix (saved as image + raw CSV).
* Error statistics – most confused pairs, overconfidence analysis.
* Visualisation of correctly and incorrectly classified samples.

Run::

    python -m src.evaluate --model models/landmark_vgg19_final.keras
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import tensorflow as tf

from src.config import Config, get_config
from src.metrics import (
    classification_report_dict,
    confusion_matrix,
    error_statistics,
    per_class_metrics,
    top_k_accuracy,
)
from src.utils import get_logger, load_model, load_encoder

logger = get_logger("evaluate")


# ---------------------------------------------------------------------------
# Prediction collection
# ---------------------------------------------------------------------------
def collect_predictions(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run *model* over *dataset* and return (y_true, y_pred, y_pred_probs).

    We iterate the (already batched) ``tf.data`` dataset and concatenate
    results.  This is memory-friendly because only the predictions
    (N × C floats) are kept, not the images.
    """
    y_true_list, prob_list = [], []
    for images, labels in dataset:
        probs = model.predict(images, verbose=0)
        y_true_list.append(labels.numpy())
        prob_list.append(probs)
    y_true = np.concatenate(y_true_list)
    y_pred_probs = np.concatenate(prob_list)
    y_pred = np.argmax(y_pred_probs, axis=1)
    return y_true, y_pred, y_pred_probs


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_dataset(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    config: Config,
    encoder,
    split_name: str = "test",
    save_dir: Optional[Path] = None,
) -> Dict:
    """Evaluate *model* on *dataset* and return a metrics dictionary.

    Also saves a confusion-matrix image and a JSON report under
    *save_dir* (defaults to ``outputs/figures``).
    """
    if save_dir is None:
        save_dir = Path(config.paths.figures)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Evaluating on %s split ...", split_name)
    y_true, y_pred, probs = collect_predictions(model, dataset)
    num_classes = len(encoder.classes_)

    # Core metrics
    top1 = float(np.mean(y_pred == y_true))
    top5 = top_k_accuracy(y_true, probs, k=config.evaluation.top_k)
    report = classification_report_dict(y_true, y_pred, num_classes)
    cm = confusion_matrix(y_true, y_pred, num_classes)
    errors = error_statistics(y_true, y_pred, probs)

    results = {
        "split": split_name,
        "num_samples": int(len(y_true)),
        "accuracy": top1,
        f"top{config.evaluation.top_k}_accuracy": top5,
        "macro_f1": report["macro_f1"],
        "weighted_f1": report["weighted_f1"],
        "per_class": report["per_class"],
        "error_stats": errors,
    }

    # Log summary
    logger.info(
        "%s — accuracy: %.4f | top-5: %.4f | macro-F1: %.4f | weighted-F1: %.4f",
        split_name, top1, top5, report["macro_f1"], report["weighted_f1"],
    )
    logger.info("Errors: %d (%.2f%%), overconfident: %d",
                 errors["num_errors"], errors["error_rate"] * 100,
                 errors["overconfident_errors"])

    # Save confusion matrix
    try:
        from src.visualization import plot_confusion_matrix
        plot_confusion_matrix(
            cm,
            class_names=[str(c) for c in encoder.classes_],
            save_path=save_dir / f"confusion_matrix_{split_name}.png",
            top_k=30,
        )
    except Exception as e:
        logger.warning("Could not plot confusion matrix: %s", e)

    # Save report JSON
    report_path = save_dir / f"report_{split_name}.json"
    # Convert numpy types for JSON.
    def _convert(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=_convert)
    logger.info("Saved evaluation report to %s", report_path)

    # Visualise correct / incorrect samples
    try:
        from src.visualization import visualize_predictions
        visualize_predictions(
            dataset, model, encoder, config,
            save_path=save_dir / f"predictions_{split_name}.png",
        )
    except Exception as e:
        logger.warning("Could not visualise predictions: %s", e)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(
    model_path: Optional[str] = None,
    config_path: Optional[str] = None,
) -> None:
    """Load model + encoder + data, then evaluate on val and test."""
    config = get_config(Path(config_path) if config_path else None)
    set_seed_and_hw(config)

    # Load model
    if model_path is None:
        model_path = Path(config.paths.models) / "landmark_vgg19_final.keras"
    model = load_model(model_path)
    logger.info("Loaded model from %s", model_path)

    # Load encoder
    encoder = load_encoder(config.paths.encoder_file)

    # Rebuild datasets (val + test only)
    from src.dataset import prepare_dataset
    data = prepare_dataset(config)

    val_results = evaluate_dataset(
        model, data["val"], config, encoder, split_name="val",
        save_dir=Path(config.paths.figures),
    )
    test_results = evaluate_dataset(
        model, data["test"], config, encoder, split_name="test",
        save_dir=Path(config.paths.figures),
    )

    logger.info("Evaluation complete.")
    return {"val": val_results, "test": test_results}


def set_seed_and_hw(config: Config):
    from src.utils import set_seed, detect_hardware
    set_seed(config.seed)
    detect_hardware(config.training.mixed_precision)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate landmark model")
    parser.add_argument("--model", type=str, default=None, help="Path to .keras model")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    args = parser.parse_args()
    main(model_path=args.model, config_path=args.config)
