from __future__ import annotations

from headline.eval.sauto import WEIGHTS, compute_sauto

REF = ("**Findings:** No intracranial hemorrhage. Normal ventricular size. "
       "**Assessment:** Unremarkable non-contrast head CT.")


def test_weights_sum_to_one():
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9


def test_sauto_is_weighted_sum_of_components():
    out = compute_sauto([REF], [REF])
    expected = sum(WEIGHTS[k] * out[k] for k in WEIGHTS)
    assert abs(out["sauto"] - expected) < 1e-9


def test_identical_text_scores_near_one():
    out = compute_sauto([REF], [REF])
    assert out["bertscore_f1"] > 0.99
    assert out["rouge_l"] > 0.99


def test_unrelated_text_scores_far_below_identical():
    same = compute_sauto([REF], [REF])["sauto"]
    diff = compute_sauto(["The weather is pleasant today."], [REF])["sauto"]
    assert diff < same - 0.3


def test_empty_prediction_does_not_crash():
    out = compute_sauto(["", "  "], [REF, REF])
    assert out["sauto"] == out["sauto"]          # NaN 아님
