"""Tests for src/models/cost.py.

Cases are built by hand so the expected cost can be verified without running
any code — see each docstring for the manual computation.
"""

import numpy as np
import pytest

from src.models.cost import (
    cost_curve,
    evaluate_amount_aware,
    evaluate_at_threshold,
    expected_cost,
    find_optimal_threshold,
)


def test_no_one_flagged_equals_baseline_cost():
    """Threshold above every score flags nobody: every fraud is a miss, so
    cost must equal baseline_cost (the sum of fraud amounts).

    y_true=[0,1,0,1,0], amounts=[10,20,30,40,50] → fraud amounts are 20+40=60.
    """
    y_true = np.array([0, 1, 0, 1, 0])
    amounts = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    y_score = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    threshold = 1.0  # higher than max score → nobody flagged
    admin_cost = 5.0

    cost = expected_cost(y_true, amounts, y_score, threshold, admin_cost)
    assert cost == pytest.approx(60.0)

    result = evaluate_at_threshold(y_true, amounts, y_score, threshold, admin_cost)
    assert result["cost"] == pytest.approx(60.0)
    assert result["baseline_cost"] == pytest.approx(60.0)
    assert result["n_flagged"] == 0
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["savings_pct"] == pytest.approx(0.0)


def test_everyone_flagged_equals_admin_cost_times_n():
    """Threshold below every score flags everybody: nothing is left unflagged,
    so the missed-fraud term is zero and cost is pure admin_cost * n_total.

    Same data as above, admin_cost=5.0, n=5 → cost = 5*5 = 25.
    """
    y_true = np.array([0, 1, 0, 1, 0])
    amounts = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    y_score = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    threshold = 0.0  # lower than min score → everyone flagged
    admin_cost = 5.0

    cost = expected_cost(y_true, amounts, y_score, threshold, admin_cost)
    assert cost == pytest.approx(25.0)

    result = evaluate_at_threshold(y_true, amounts, y_score, threshold, admin_cost)
    assert result["n_flagged"] == 5
    assert result["cost"] == pytest.approx(25.0)
    assert result["recall"] == 1.0
    assert result["precision"] == pytest.approx(2 / 5)  # 2 fraud out of 5 flagged


def test_find_optimal_threshold_matches_hand_computed_minimum():
    """5-row example where the cost at every unique threshold can be computed
    by hand; find_optimal_threshold must land exactly on the minimum.

    y_true=[1,0,1,0,0], amounts=[100,10,5,10,10], y_score=[.9,.8,.3,.2,.1], admin_cost=5.

    Hand-computed cost per candidate threshold (score >= threshold flags):
      t=0.9: flags {0}       → 1 flagged, missed fraud amt=5 (row 2) → cost = 5*1 + 5  = 10
      t=0.8: flags {0,1}     → 2 flagged, missed fraud amt=5         → cost = 5*2 + 5  = 15
      t=0.3: flags {0,1,2}   → 3 flagged, no missed fraud            → cost = 5*3      = 15
      t=0.2: flags {0,1,2,3} → 4 flagged, no missed fraud            → cost = 5*4      = 20
      t=0.1: flags all 5     → 5 flagged, no missed fraud            → cost = 5*5      = 25
    Minimum is 10.0 at threshold=0.9, no ties.
    """
    y_true = np.array([1, 0, 1, 0, 0])
    amounts = np.array([100.0, 10.0, 5.0, 10.0, 10.0])
    y_score = np.array([0.9, 0.8, 0.3, 0.2, 0.1])
    admin_cost = 5.0

    result = find_optimal_threshold(y_true, amounts, y_score, admin_cost)

    assert result["threshold"] == pytest.approx(0.9)
    assert result["cost"] == pytest.approx(10.0)
    assert result["n_flagged"] == 1
    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(0.5)
    assert result["baseline_cost"] == pytest.approx(105.0)  # 100 + 5
    assert result["savings_pct"] == pytest.approx((105.0 - 10.0) / 105.0)


def test_cost_curve_matches_hand_computed_costs():
    """Same 5-row example as the optimal-threshold test above; cost_curve must
    return all 5 unique-score candidates, ascending, with the hand-computed costs.
    """
    y_true = np.array([1, 0, 1, 0, 0])
    amounts = np.array([100.0, 10.0, 5.0, 10.0, 10.0])
    y_score = np.array([0.9, 0.8, 0.3, 0.2, 0.1])
    admin_cost = 5.0

    curve = cost_curve(y_true, amounts, y_score, admin_cost)

    assert curve["thresholds"] == pytest.approx([0.1, 0.2, 0.3, 0.8, 0.9])
    assert curve["costs"] == pytest.approx([25.0, 20.0, 15.0, 15.0, 10.0])
    assert curve["n_flagged"] == [5, 4, 3, 2, 1]


def test_amount_aware_matches_hand_computed_flags():
    """6-row case with varied amounts; only the row that clears its own
    admin_cost/amount threshold should be flagged.

    y_true=[1,0,1,0,0,1], amounts=[1000,50,20,200,5,10], y_score=[.003,.05,.5,.02,.9,.4], admin_cost=5.

    Per-row threshold = 5/amount, flagged iff score > threshold:
      row0: amt=1000, thresh=0.005, score=.003  → not flagged
      row1: amt=50,   thresh=0.1,   score=.05   → not flagged
      row2: amt=20,   thresh=0.25,  score=.5    → FLAGGED (fraud)
      row3: amt=200,  thresh=0.025, score=.02   → not flagged
      row4: amt=5,    thresh=1.0,   score=.9    → not flagged
      row5: amt=10,   thresh=0.5,   score=.4    → not flagged
    Only row2 flagged → n_flagged=1, it's a true positive, recall=1/3 (3 fraud rows).
    Missed fraud = row0 (1000) + row5 (10) = 1010 → cost = 5*1 + 1010 = 1015.
    baseline_cost = 1000+20+10 = 1030.
    """
    y_true = np.array([1, 0, 1, 0, 0, 1])
    amounts = np.array([1000.0, 50.0, 20.0, 200.0, 5.0, 10.0])
    y_score = np.array([0.003, 0.05, 0.5, 0.02, 0.9, 0.4])
    admin_cost = 5.0

    result = evaluate_amount_aware(y_true, amounts, y_score, admin_cost)

    assert result["rule"] == "p > admin_cost / amount"
    assert result["n_flagged"] == 1
    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(1 / 3)
    assert result["cost"] == pytest.approx(1015.0)
    assert result["baseline_cost"] == pytest.approx(1030.0)


def test_amount_aware_never_flags_zero_amount():
    """amount<=0 must never be flagged, and must not raise ZeroDivisionError.

    Row 0: amount=0, score=0.99 — nothing to lose by missing it, never flagged
    regardless of score. Row 1: amount=100, score=0.01 — below its own
    threshold (5/100=0.05), also not flagged. So n_flagged=0 and the only
    cost is the missed fraud amount in row 1.
    """
    y_true = np.array([0, 1])
    amounts = np.array([0.0, 100.0])
    y_score = np.array([0.99, 0.01])
    admin_cost = 5.0

    result = evaluate_amount_aware(y_true, amounts, y_score, admin_cost)

    assert result["n_flagged"] == 0
    assert result["cost"] == pytest.approx(100.0)
    assert result["baseline_cost"] == pytest.approx(100.0)
    assert result["savings_pct"] == pytest.approx(0.0)


def test_amount_aware_diverges_from_global_threshold():
    """Amount-aware and a global threshold can flag opposite rows entirely.

    Row 0: amount=2000 (fraud), score=0.004 — tiny score, huge amount. A
    global threshold of 0.01 misses it (score < threshold). Amount-aware
    flags it: 5/2000=0.0025, and 0.004 > 0.0025.

    Row 1: amount=10 (not fraud), score=0.05 — the same global threshold of
    0.01 flags it (0.05 >= 0.01). Amount-aware does not: 5/10=0.5, and 0.05
    is not > 0.5.

    Result: the global rule misses the fraud (cost=2005, worse than doing
    nothing), the amount-aware rule catches it (cost=5).
    """
    y_true = np.array([1, 0])
    amounts = np.array([2000.0, 10.0])
    y_score = np.array([0.004, 0.05])
    admin_cost = 5.0

    global_result = evaluate_at_threshold(y_true, amounts, y_score, threshold=0.01, admin_cost=admin_cost)
    amount_aware_result = evaluate_amount_aware(y_true, amounts, y_score, admin_cost)

    assert global_result["n_flagged"] == 1
    assert global_result["recall"] == 0.0
    assert global_result["cost"] == pytest.approx(2005.0)

    assert amount_aware_result["n_flagged"] == 1
    assert amount_aware_result["recall"] == 1.0
    assert amount_aware_result["cost"] == pytest.approx(5.0)


def test_mismatched_lengths_raises():
    """All five functions must reject y_true/amounts/y_score of different lengths."""
    y_true = np.array([1, 0, 1])
    amounts = np.array([10.0, 20.0])  # length mismatch
    y_score = np.array([0.5, 0.4, 0.3])

    with pytest.raises(ValueError):
        expected_cost(y_true, amounts, y_score, threshold=0.4)

    with pytest.raises(ValueError):
        evaluate_amount_aware(y_true, amounts, y_score)

    with pytest.raises(ValueError):
        evaluate_at_threshold(y_true, amounts, y_score, threshold=0.4)

    with pytest.raises(ValueError):
        find_optimal_threshold(y_true, amounts, y_score)

    with pytest.raises(ValueError):
        cost_curve(y_true, amounts, y_score)
