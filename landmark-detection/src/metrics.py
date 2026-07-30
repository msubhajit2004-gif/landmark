"""
src/metrics.py
==============

Purpose
-------
Custom metrics and evaluation helpers that go beyond what Keras tracks
during training.  These are used by :mod:`evaluate` to produce a full
post-training report.

Concepts
--------
* **Accuracy** – fraction of correct predictions.  Misleading on
  imbalanced data.
* **Precision** – of the images we *predicted* as class X, how many
  were really X?  (quality of positive predictions)
* **Recall**    – of the images that *are* class X, how many did we
  find?  (coverage of the true positives)
* **F1-score**  – harmonic mean of precision and recall; high only when
  both are high.
* **Top-5 accuracy** – fraction of samples whose true label is among
  the model's top-5 predicted classes.  Important for landmark
  recognition because many landmarks are visually similar.
* **Confusion matrix** – N×N grid showing where misclassifications go.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def top_k_accuracy(y_true: np.ndarray, y_pred_probs: np.ndarray, k: int = 5) -> float:
    """Compute top-k accuracy from predicted probabilities.

    Parameters
    ----------
    y_true : (N,) int array of true class indices.
    y_pred_probs : (N, C) float array of class probabilities.
    k : int
    """
    topk = np.argsort(y_pred_probs, axis=1)[:, -k:]
    correct = np.any(topk == y_true.reshape(-1, 1), axis=1)
    return float(np.mean(correct))


def per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> Dict[int, Dict[str, float]]:
    """Compute precision, recall, F1 and support for every class."""
    results: Dict[int, Dict[str, float]] = {}
    for c in range(num_classes):
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        support = int(np.sum(y_true == c))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )
        results[c] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    return results


def macro_f1(per_class: Dict[int, Dict[str, float]]) -> float:
    """Macro-averaged F1 (unweighted mean across classes)."""
    f1s = [v["f1"] for v in per_class.values() if v["support"] > 0]
    return float(np.mean(f1s)) if f1s else 0.0


def weighted_f1(per_class: Dict[int, Dict[str, float]]) -> float:
    """Support-weighted F1 – better for imbalanced datasets."""
    total = sum(v["support"] for v in per_class.values())
    if total == 0:
        return 0.0
    return sum(v["f1"] * v["support"] for v in per_class.values()) / total


def error_statistics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_probs: np.ndarray,
) -> Dict:
    """Summarise where and how badly the model fails.

    Returns a dict with:
    * ``confusion_pairs`` – most frequent (true, predicted) confusions.
    * ``mean_confidence_on_error`` – avg predicted prob on wrong samples.
    * ``overconfident_errors`` – wrong predictions with prob > 0.9.
    """
    errors = y_pred != y_true
    n_err = int(np.sum(errors))

    # Most confused pairs
    pairs: Dict[Tuple[int, int], int] = {}
    for t, p in zip(y_true[errors], y_pred[errors]):
        key = (int(t), int(p))
        pairs[key] = pairs.get(key, 0) + 1
    top_pairs = sorted(pairs.items(), key=lambda kv: -kv[1])[:10]

    confidences = np.max(y_pred_probs, axis=1)
    mean_conf_err = float(np.mean(confidences[errors])) if n_err > 0 else 0.0
    overconfident = int(np.sum((errors) & (confidences > 0.9)))

    return {
        "num_errors": n_err,
        "error_rate": n_err / len(y_true) if len(y_true) else 0.0,
        "top_confusion_pairs": top_pairs,
        "mean_confidence_on_error": mean_conf_err,
        "overconfident_errors": overconfident,
    }


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """Return an (C, C) confusion matrix."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def classification_report_dict(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> Dict:
    """Full report combining per-class and aggregate metrics."""
    pc = per_class_metrics(y_true, y_pred, num_classes)
    return {
        "per_class": pc,
        "macro_f1": macro_f1(pc),
        "weighted_f1": weighted_f1(pc),
        "accuracy": float(np.mean(y_pred == y_true)),
    }
