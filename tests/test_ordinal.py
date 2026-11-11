"""The proportional-odds model must respect the ordering it is built around."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone

from src.evaluate import evaluate
from src.ordinal import ProportionalOdds


@pytest.fixture(scope="module")
def latent():
    """Outcomes generated from one ordered latent score, as the model assumes."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3000, 3))
    eta = 1.2 * X[:, 0] - 0.8 * X[:, 1]
    u = eta + rng.logistic(size=3000)
    y = np.where(u < -0.6, "A", np.where(u < 0.6, "D", "H"))
    return X, y


def test_recovers_the_coefficients_it_was_generated_from(latent):
    X, y = latent
    m = ProportionalOdds(alpha=1.0).fit(X, y)
    assert m.converged_
    assert m.coef_[0] == pytest.approx(1.2, abs=0.15)
    assert m.coef_[1] == pytest.approx(-0.8, abs=0.15)
    assert m.coef_[2] == pytest.approx(0.0, abs=0.15)


def test_thresholds_stay_ordered(latent):
    """t2 = t1 + softplus(delta), so the optimiser cannot cross them."""
    X, y = latent
    for alpha in (0.1, 1.0, 100.0, 5000.0):
        m = ProportionalOdds(alpha=alpha).fit(X, y)
        assert m.theta_[0] < m.theta_[1], f"thresholds crossed at alpha={alpha}"


def test_probabilities_are_a_distribution(latent):
    X, y = latent
    p = ProportionalOdds().fit(X, y).predict_proba(X)
    assert p.shape == (len(y), 3)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert (p > 0).all()


def test_columns_follow_sklearn_class_order(latent):
    """Every metric here indexes probabilities by classes_, so this must hold."""
    X, y = latent
    m = ProportionalOdds().fit(X, y)
    assert list(m.classes_) == ["A", "D", "H"]
    p = m.predict_proba(X)
    # A higher score means a home win; its probability must rise with x0.
    high, low = X[:, 0] > 1, X[:, 0] < -1
    assert p[high][:, 2].mean() > p[low][:, 2].mean()
    assert p[low][:, 0].mean() > p[high][:, 0].mean()


def test_a_middling_score_favours_the_draw(latent):
    """The point of the ordering: the draw lives between the thresholds."""
    X, y = latent
    m = ProportionalOdds().fit(X, y)
    middle = np.abs(X[:, 0]) < 0.2
    extreme = np.abs(X[:, 0]) > 1.5
    assert m.predict_proba(X[middle])[:, 1].mean() > m.predict_proba(X[extreme])[:, 1].mean()


def test_stronger_regularisation_shrinks_the_coefficients(latent):
    X, y = latent
    weak = ProportionalOdds(alpha=1.0).fit(X, y)
    strong = ProportionalOdds(alpha=5000.0).fit(X, y)
    assert np.abs(strong.coef_).sum() < np.abs(weak.coef_).sum()


def test_beats_the_base_rate_on_data_with_signal(latent):
    X, y = latent
    m = ProportionalOdds(alpha=1.0).fit(X[:2000], y[:2000])
    proba = m.predict_proba(X[2000:])
    model = evaluate(y[2000:], proba, list(m.classes_))
    prior = np.tile([(y[:2000] == c).mean() for c in m.classes_], (len(y) - 2000, 1))
    base = evaluate(y[2000:], prior, list(m.classes_))
    assert model["rps"] < base["rps"]


def test_sample_weights_shift_the_fit(latent):
    X, y = latent
    plain = ProportionalOdds(alpha=1.0).fit(X, y)
    weighted = ProportionalOdds(alpha=1.0).fit(X, y, sample_weight=np.where(X[:, 0] > 0, 5.0, 0.2))
    assert not np.allclose(plain.coef_, weighted.coef_)


def test_works_as_a_scikit_learn_estimator(latent):
    X, y = latent
    fresh = clone(ProportionalOdds(alpha=42.0))
    assert fresh.get_params()["alpha"] == 42.0
    fresh.fit(X, y)
    assert set(fresh.predict(X)) <= {"A", "D", "H"}


def test_refuses_a_single_outcome(latent):
    X, _ = latent
    with pytest.raises(ValueError, match="two outcomes"):
        ProportionalOdds().fit(X, np.array(["H"] * len(X)))
