"""Example-dependent cost model for fraud threshold tuning.

Implements the cost matrix from Bahnsen, Aouada & Ottersten (2014): missing a
fraudulent transaction costs its full amount (Cost_FN), while flagging any
transaction for manual review — fraud or not — costs a fixed fee (Cost_FP =
Cost_TP = admin_cost). Correctly-ignored legitimate transactions cost nothing
(Cost_TN = 0).

All functions are pure: they accept arrays of true labels, transaction amounts,
and predicted scores, and return floats/dicts with no side effects. Designed to
be reused across threshold-tuning scripts, the dashboard, and the API.
"""

from __future__ import annotations

import numpy as np


def _validate_lengths(y_true: np.ndarray, amounts: np.ndarray, y_score: np.ndarray) -> None:
    if not (len(y_true) == len(amounts) == len(y_score)):
        raise ValueError(
            "y_true, amounts, and y_score must have the same length, got "
            f"{len(y_true)}, {len(amounts)}, {len(y_score)}."
        )


def expected_cost(
    y_true,
    amounts,
    y_score,
    threshold: float,
    admin_cost: float = 5.0,
) -> float:
    """Total expected cost of flagging at a given score threshold.

    Cost model (Bahnsen, Aouada & Ottersten, 2014), example-dependent:
      - Cost_FN(i) = amounts[i]     — a missed fraud costs the transaction amount.
      - Cost_FP(i) = Cost_TP(i) = admin_cost — every flagged transaction costs a
        fixed manual-review fee, whether it turns out to be fraud or not.
      - Cost_TN(i) = 0              — correctly ignored legitimate transactions
        cost nothing.

    Total_cost(t) = admin_cost * n_flagged(t)
                    + sum(amounts[i] for i not flagged with y_true[i] == 1)

    A transaction is flagged if its score >= threshold.

    Parameters
    ----------
    y_true:
        Binary fraud labels (1 = fraud).
    amounts:
        Transaction amounts, same currency units as the cost being minimized.
    y_score:
        Model scores.
    threshold:
        Score cutoff for flagging.
    admin_cost:
        Fixed cost of manually reviewing one flagged transaction.

    Returns
    -------
    float
        Total cost at this threshold.

    Raises
    ------
    ValueError
        If y_true, amounts, and y_score do not have the same length.
    """
    y_true = np.asarray(y_true)
    amounts = np.asarray(amounts)
    y_score = np.asarray(y_score)
    _validate_lengths(y_true, amounts, y_score)

    flagged = y_score >= threshold
    n_flagged = int(flagged.sum())

    missed_fraud = (~flagged) & (y_true == 1)
    fn_cost = float(amounts[missed_fraud].sum())

    return float(admin_cost * n_flagged + fn_cost)


def evaluate_at_threshold(
    y_true,
    amounts,
    y_score,
    threshold: float,
    admin_cost: float = 5.0,
) -> dict:
    """Cost, savings, and ranking metrics at a single fixed threshold.

    Reuses expected_cost for the cost figure so the cost formula lives in one
    place; this function adds the operational context around it — how many
    transactions get flagged, precision/recall of the flag, and how much the
    threshold saves relative to reviewing nothing.

    Parameters
    ----------
    y_true:
        Binary fraud labels (1 = fraud).
    amounts:
        Transaction amounts.
    y_score:
        Model scores.
    threshold:
        Score cutoff for flagging.
    admin_cost:
        Fixed cost of manually reviewing one flagged transaction.

    Returns
    -------
    dict with keys:
      - threshold: float, echoes the input.
      - cost: float, total cost at this threshold.
      - n_flagged: int, transactions flagged.
      - precision: float, fraction of flagged transactions that are fraud.
        0.0 if n_flagged is 0.
      - recall: float, fraction of fraud transactions that get flagged.
        0.0 if there are no positives in y_true.
      - baseline_cost: float, cost of flagging nothing (sum of amounts of all
        fraud — every fraud is a miss).
      - savings_pct: float, (baseline_cost - cost) / baseline_cost. 0.0 if
        baseline_cost is 0 (no fraud in y_true).
      - admin_cost: float, echoes the input parameter.
    All values are native Python floats/ints (JSON-serializable).

    Raises
    ------
    ValueError
        If y_true, amounts, and y_score do not have the same length.
    """
    y_true = np.asarray(y_true)
    amounts = np.asarray(amounts)
    y_score = np.asarray(y_score)
    _validate_lengths(y_true, amounts, y_score)

    cost = expected_cost(y_true, amounts, y_score, threshold, admin_cost)

    flagged = y_score >= threshold
    n_flagged = int(flagged.sum())
    n_positives = int((y_true == 1).sum())
    n_tp = int(((y_true == 1) & flagged).sum())

    precision = n_tp / n_flagged if n_flagged > 0 else 0.0
    recall = n_tp / n_positives if n_positives > 0 else 0.0

    baseline_cost = float(amounts[y_true == 1].sum())
    savings_pct = (baseline_cost - cost) / baseline_cost if baseline_cost > 0 else 0.0

    return {
        "threshold": float(threshold),
        "cost": cost,
        "n_flagged": n_flagged,
        "precision": float(precision),
        "recall": float(recall),
        "baseline_cost": baseline_cost,
        "savings_pct": float(savings_pct),
        "admin_cost": float(admin_cost),
    }


def evaluate_amount_aware(
    y_true,
    amounts,
    y_score,
    admin_cost: float = 5.0,
) -> dict:
    """Per-transaction flagging rule: flag i iff y_score[i] > admin_cost / amounts[i].

    This is the actual cost-minimizing rule under Cost_TP=Cost_FP=admin_cost —
    a single global threshold is a deployment simplification of this, not the
    Bayes-optimal decision. amounts[i] <= 0 is never flagged (no division,
    nothing to lose by missing it).

    Returns a dict with the same shape as evaluate_at_threshold, plus a "rule"
    key, minus "threshold" (there isn't a single one here).

    Raises
    ------
    ValueError
        If y_true, amounts, and y_score do not have the same length.
    """
    y_true = np.asarray(y_true)
    amounts = np.asarray(amounts)
    y_score = np.asarray(y_score)
    _validate_lengths(y_true, amounts, y_score)

    flagged = np.zeros(len(y_true), dtype=bool)
    has_amount = amounts > 0
    flagged[has_amount] = y_score[has_amount] > (admin_cost / amounts[has_amount])

    n_flagged = int(flagged.sum())
    n_positives = int((y_true == 1).sum())
    n_tp = int(((y_true == 1) & flagged).sum())

    precision = n_tp / n_flagged if n_flagged > 0 else 0.0
    recall = n_tp / n_positives if n_positives > 0 else 0.0

    missed_fraud = (~flagged) & (y_true == 1)
    fn_cost = float(amounts[missed_fraud].sum())
    cost = float(admin_cost * n_flagged + fn_cost)

    baseline_cost = float(amounts[y_true == 1].sum())
    savings_pct = (baseline_cost - cost) / baseline_cost if baseline_cost > 0 else 0.0

    return {
        "rule": "p > admin_cost / amount",
        "cost": cost,
        "n_flagged": n_flagged,
        "precision": float(precision),
        "recall": float(recall),
        "baseline_cost": baseline_cost,
        "savings_pct": float(savings_pct),
        "admin_cost": float(admin_cost),
    }


def find_optimal_threshold(
    y_true,
    amounts,
    y_score,
    admin_cost: float = 5.0,
) -> dict:
    """Find the score threshold that minimizes total expected cost.

    Sweeps every unique value in y_score as a candidate threshold — those are
    the only points where any transaction's flagging decision can change, so
    an arbitrary grid would either skip the optimum or waste candidates between
    them. Each candidate is scored via evaluate_at_threshold, so the returned
    dict already has the full report (cost, precision, recall, savings_pct...)
    for the winning threshold, not just the cost.

    Candidates are swept from highest to lowest score, so that ties in cost
    keep the higher threshold (fewer flagged transactions for the same cost).

    This sweep is O(n_unique * n) — deliberately simple and easy to verify
    against evaluate_at_threshold one candidate at a time, rather than a
    cumulative-sum formulation that would be O(n log n) but harder to eyeball.
    Fine for offline threshold tuning on a val set; would need revisiting if
    ever run inside a tight, latency-sensitive loop.

    Parameters
    ----------
    y_true:
        Binary fraud labels (1 = fraud).
    amounts:
        Transaction amounts.
    y_score:
        Model scores.
    admin_cost:
        Fixed cost of manually reviewing one flagged transaction.

    Returns
    -------
    dict
        Same shape as evaluate_at_threshold's return value, for the
        cost-minimizing threshold.

    Raises
    ------
    ValueError
        If y_true, amounts, and y_score do not have the same length.
    """
    y_true = np.asarray(y_true)
    amounts = np.asarray(amounts)
    y_score = np.asarray(y_score)
    _validate_lengths(y_true, amounts, y_score)

    candidates = np.unique(y_score)[::-1]  # descending: ties keep the higher threshold

    best_result = None
    for threshold in candidates:
        result = evaluate_at_threshold(y_true, amounts, y_score, float(threshold), admin_cost)
        if best_result is None or result["cost"] < best_result["cost"]:
            best_result = result

    return best_result


def cost_curve(
    y_true,
    amounts,
    y_score,
    admin_cost: float = 5.0,
) -> dict:
    """Total expected cost swept across every unique score threshold.

    Same candidate set as find_optimal_threshold, returned as a full curve
    (thresholds ascending) for plotting cost vs. threshold in the dashboard or
    README, rather than reduced to a single optimum.

    Parameters
    ----------
    y_true:
        Binary fraud labels (1 = fraud).
    amounts:
        Transaction amounts.
    y_score:
        Model scores.
    admin_cost:
        Fixed cost of manually reviewing one flagged transaction.

    Returns
    -------
    dict with keys:
      - thresholds: list[float], unique score values swept, ascending order.
      - costs: list[float], total cost at each threshold.
      - n_flagged: list[int], transactions flagged at each threshold.
    All values are native Python floats/ints (JSON-serializable).

    Raises
    ------
    ValueError
        If y_true, amounts, and y_score do not have the same length.
    """
    y_true = np.asarray(y_true)
    amounts = np.asarray(amounts)
    y_score = np.asarray(y_score)
    _validate_lengths(y_true, amounts, y_score)

    candidates = np.unique(y_score)  # already ascending

    thresholds: list[float] = []
    costs: list[float] = []
    n_flagged_list: list[int] = []

    for threshold in candidates:
        threshold = float(threshold)
        cost = expected_cost(y_true, amounts, y_score, threshold, admin_cost)
        n_flagged = int((y_score >= threshold).sum())

        thresholds.append(threshold)
        costs.append(cost)
        n_flagged_list.append(n_flagged)

    return {"thresholds": thresholds, "costs": costs, "n_flagged": n_flagged_list}
