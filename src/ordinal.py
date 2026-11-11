"""A proportional-odds model for an ordered outcome.

Away win, draw, home win is an *ordered* scale - that is the whole reason the
ranked probability score is the headline metric. Multinomial logistic regression
does not know that: it fits each outcome its own independent set of weights, and
nothing stops it from deciding a rating gap raises the chance of an away win and
a home win at once.

A cumulative-link model builds the ordering in. One score per match, two
thresholds that must stay in order:

    P(away win)          = sigma(t1 - eta)
    P(away win or draw)  = sigma(t2 - eta)
    P(home win)          = 1 - sigma(t2 - eta)

so eta slides a match along a single axis from "away win" to "home win", and the
draw is what sits between the thresholds. That is a far closer description of
football than three independent outcomes, and it spends 12 parameters where the
multinomial spends 24.

Fitted by maximum likelihood with an L2 penalty (alpha = 1/C in scikit-learn's
terms) and analytic gradients. t2 = t1 + softplus(delta) keeps the thresholds
ordered by construction, so the optimiser cannot cross them.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

# Least to most home-favouring. classes_ is sorted to match scikit-learn's
# convention, so predict_proba lines up with every other estimator here.
ORDER = ["A", "D", "H"]


def _softplus(x: float) -> float:
    return np.logaddexp(0.0, x)


class ProportionalOdds(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha: float = 500.0, max_iter: int = 300, tol: float = 1e-8):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

    # -- likelihood ----------------------------------------------------------
    def _neg_log_likelihood(self, params, X, levels, weights):
        beta, t1, delta = params[:-2], params[-2], params[-1]
        t2 = t1 + _softplus(delta)
        eta = X @ beta
        s1, s2 = expit(t1 - eta), expit(t2 - eta)
        g1, g2 = s1 * (1 - s1), s2 * (1 - s2)

        p = np.where(levels == 0, s1, np.where(levels == 1, s2 - s1, 1 - s2))
        p = np.clip(p, 1e-12, 1.0)
        nll = -np.sum(weights * np.log(p)) + 0.5 * self.alpha * beta @ beta

        # d(-log p)/d eta, and the same for each threshold.
        d_eta = np.where(levels == 0, g1, np.where(levels == 1, g2 - g1, -g2)) / p
        d_t1 = np.where(levels == 0, -g1, np.where(levels == 1, g1, 0.0)) / p
        d_t2 = np.where(levels == 0, 0.0, np.where(levels == 1, -g2, g2)) / p

        grad_beta = X.T @ (weights * d_eta) + self.alpha * beta
        grad_t1 = np.sum(weights * (d_t1 + d_t2))          # t2 moves with t1
        grad_delta = np.sum(weights * d_t2) * expit(delta)  # d t2 / d delta
        return nll, np.concatenate([grad_beta, [grad_t1, grad_delta]])

    # -- scikit-learn API ----------------------------------------------------
    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        seen = [c for c in ORDER if c in set(y)]
        if len(seen) < 2:
            raise ValueError("need at least two outcomes to fit an ordered model")
        rank = {c: i for i, c in enumerate(seen)}
        levels = np.array([rank[v] for v in y])
        weights = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)

        start = np.concatenate([np.zeros(X.shape[1]), [-0.4, 0.0]])
        result = minimize(self._neg_log_likelihood, start, args=(X, levels, weights),
                          jac=True, method="L-BFGS-B",
                          options={"maxiter": self.max_iter, "ftol": self.tol})
        self.coef_ = result.x[:-2]
        self.theta_ = np.array([result.x[-2], result.x[-2] + _softplus(result.x[-1])])
        self.n_iter_ = result.nit
        self.converged_ = bool(result.success)
        self._seen = seen
        self.classes_ = np.array(sorted(seen))
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "coef_")
        eta = np.asarray(X, dtype=float) @ self.coef_
        cumulative = expit(self.theta_[:, None] - eta)          # (2, n)
        ordered = np.vstack([cumulative[0],
                             cumulative[1] - cumulative[0],
                             1 - cumulative[1]]).T              # columns in ORDER
        ordered = np.clip(ordered, 1e-12, 1.0)
        ordered /= ordered.sum(axis=1, keepdims=True)
        # Re-order columns to match classes_ (sorted), as scikit-learn expects.
        index = {c: i for i, c in enumerate(self._seen)}
        return ordered[:, [index[c] for c in self.classes_]]

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
