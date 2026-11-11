"""Metric tests: the scores have to mean what the README claims they mean."""
from __future__ import annotations

import numpy as np
import pytest

from src.evaluate import (
    calibration_table,
    evaluate,
    multiclass_brier,
    ranked_probability_score,
)

CLASSES = ["A", "D", "H"]  # scikit-learn's lexicographic ordering


def test_perfect_forecast_scores_zero():
    y = ["H", "D", "A"]
    proba = np.array([[0, 0, 1.0], [0, 1.0, 0], [1.0, 0, 0]])
    assert ranked_probability_score(y, proba, CLASSES) == pytest.approx(0.0)
    assert multiclass_brier(y, proba, CLASSES) == pytest.approx(0.0)
    assert evaluate(y, proba, CLASSES)["accuracy"] == 1.0


def test_rps_penalises_distance_on_the_ordinal_scale():
    """The point of RPS: how wrong you were matters, not just that you were wrong."""
    y_home = ["H"]
    draw_call = np.array([[0.0, 1.0, 0.0]])   # one step from the truth
    away_call = np.array([[1.0, 0.0, 0.0]])   # the opposite end of the scale
    assert (ranked_probability_score(y_home, draw_call, CLASSES)
            < ranked_probability_score(y_home, away_call, CLASSES))

    # A draw sits in the middle, so missing it either way costs the same.
    y_draw = ["D"]
    home_call = np.array([[0.0, 0.0, 1.0]])
    assert ranked_probability_score(y_draw, home_call, CLASSES) == pytest.approx(
        ranked_probability_score(y_draw, away_call, CLASSES))


def test_rps_is_bounded_by_one():
    y = ["H"]
    worst = np.array([[1.0, 0.0, 0.0]])
    assert 0.0 <= ranked_probability_score(y, worst, CLASSES) <= 1.0
    assert ranked_probability_score(y, worst, CLASSES) == pytest.approx(1.0)


def test_uniform_forecast_hits_the_analytic_values():
    """Hand-computable reference points that pin the normalisation down."""
    uniform = np.full((1, 3), 1 / 3)
    # cumulative (1/3, 2/3); against a win that is (4/9 + 1/9) / 2 = 5/18 ...
    assert ranked_probability_score(["H"], uniform, CLASSES) == pytest.approx(5 / 18)
    assert ranked_probability_score(["A"], uniform, CLASSES) == pytest.approx(5 / 18)
    # ... and against a draw only (1/9 + 1/9) / 2 = 1/9.
    assert ranked_probability_score(["D"], uniform, CLASSES) == pytest.approx(1 / 9)


def test_evaluate_reports_every_class():
    rng = np.random.default_rng(0)
    y = rng.choice(["H", "D", "A"], size=300)
    proba = rng.dirichlet([2, 2, 2], size=300)
    m = evaluate(y, proba, CLASSES)
    for cls in ("H", "D", "A"):
        assert 0.0 <= m[f"precision_{cls}"] <= 1.0
        assert 0.0 <= m[f"recall_{cls}"] <= 1.0
    assert m["n"] == 300
    assert m["log_loss"] > 0


def test_class_order_does_not_change_the_score():
    """Metrics must follow the labels, not the column order they arrive in."""
    y = ["H", "A", "D", "H"]
    p_adh = np.array([[0.2, 0.3, 0.5], [0.6, 0.2, 0.2], [0.3, 0.4, 0.3], [0.1, 0.2, 0.7]])
    reorder = [2, 1, 0]
    p_hda = p_adh[:, reorder]
    a = evaluate(y, p_adh, ["A", "D", "H"])
    b = evaluate(y, p_hda, ["H", "D", "A"])
    for key in ("accuracy", "log_loss", "rps", "brier", "f1_macro"):
        assert a[key] == pytest.approx(b[key]), key


def test_calibration_table_shape():
    rng = np.random.default_rng(1)
    y = rng.choice(["H", "D", "A"], size=800)
    proba = rng.dirichlet([3, 3, 3], size=800)
    table = calibration_table(y, proba, CLASSES)
    assert {"outcome", "bin", "n", "predicted", "observed"} == set(table.columns)
    assert (table.n >= 20).all()
