"""Evaluation metrics for binary fraud classifiers.

All functions are pure: they accept arrays of true labels and predicted scores,
return floats or JSON-serializable dicts, and have no side effects. Designed to
be reused across training scripts, threshold tuning, and the analyst dashboard.

Why these metrics and not ROC-AUC:
  At ~3.5% fraud rate, the large pool of true negatives inflates ROC-AUC even
  for mediocre models. PR-AUC, Precision@K, and Recall@Precision all operate
  on the positive class only and reflect real operational constraints.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


def pr_auc(y_true, y_score) -> float:
    """Precision-Recall AUC (average precision).

    Primary metric for imbalanced datasets. Equivalent to the area under the
    precision-recall curve, computed as the weighted mean of precisions at each
    threshold, weighted by the recall increment.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    return float(average_precision_score(y_true, y_score))


def precision_at_k(y_true, y_score, k: int) -> float:
    """Fraction of true fraud among the top-K highest-scored transactions.

    Measures the quality of the model's highest-confidence predictions — the
    only ones that fit in a finite manual-review queue.

    Parameters
    ----------
    k:
        Number of top-scored transactions to inspect. Clamped to len(y_true)
        if k exceeds the sample count.

    Raises
    ------
    ValueError
        If k <= 0.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}.")

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    k = min(k, len(y_true))
    top_idx = np.argsort(-y_score)[:k]
    return float(y_true[top_idx].mean())


def recall_at_precision(y_true, y_score, min_precision: float) -> dict:
    """Recall achievable while maintaining at least `min_precision`.

    Finds the lowest score threshold where precision >= min_precision, then
    reports the recall and number of flagged transactions at that threshold.
    Useful for setting operational cutoffs: "flag as few transactions as needed
    to stay above X% precision, and report how much fraud we catch."

    Parameters
    ----------
    min_precision:
        Minimum acceptable precision (0.0–1.0).

    Returns
    -------
    dict with keys: threshold, precision, recall, n_flagged.
    If no threshold achieves min_precision, threshold and precision are None,
    recall is 0.0, and n_flagged is 0.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # precision_recall_curve returns arrays of length n_thresholds+1 for
    # precision/recall and n_thresholds for thresholds. The extra point at
    # index 0 (precision=1, recall=0) has no corresponding threshold entry.
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)

    # Restrict to indices that have a real threshold (exclude the appended point)
    valid = np.where(precisions[:-1] >= min_precision)[0]

    if len(valid) == 0:
        return {"threshold": None, "precision": None, "recall": 0.0, "n_flagged": 0}

    # Among valid indices, pick the one with the lowest threshold (= highest recall).
    # Lower threshold → more transactions flagged → higher recall.
    best = valid[np.argmin(thresholds[valid])]

    chosen_threshold = float(thresholds[best])
    return {
        "threshold": chosen_threshold,
        "precision": float(precisions[best]),
        "recall": float(recalls[best]),
        "n_flagged": int((y_score >= chosen_threshold).sum()),
    }


def evaluate_all(
    y_true,
    y_score,
    ks: tuple[int, ...] = (100, 500, 1000, 5000),
    precision_floors: tuple[float, ...] = (0.5, 0.7, 0.9),
) -> dict:
    """Compute all evaluation metrics and return a JSON-serializable dict.

    Returns
    -------
    {
        "pr_auc": float,
        "precision_at_k": {"100": float, "500": float, ...},
        "recall_at_precision": {"0.5": {...}, "0.7": {...}, "0.9": {...}},
        "n_samples": int,
        "n_positives": int,
        "base_rate": float,
    }
    Dict keys are strings so the structure round-trips through JSON unchanged.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    return {
        "pr_auc": pr_auc(y_true, y_score),
        "precision_at_k": {
            str(k): precision_at_k(y_true, y_score, k) for k in ks
        },
        "recall_at_precision": {
            str(floor): recall_at_precision(y_true, y_score, floor)
            for floor in precision_floors
        },
        "n_samples": int(len(y_true)),
        "n_positives": int(y_true.sum()),
        "base_rate": float(y_true.mean()),
    }