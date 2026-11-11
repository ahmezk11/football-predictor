"""Evaluation metrics and baselines for three-way match prediction."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)

from .config import RESULT_CLASSES

# H/D/A is ordinal (a draw sits between the two wins), which is what makes the
# ranked probability score the standard accuracy measure for football forecasts.
ORDINAL = ["H", "D", "A"]


def _align(proba: np.ndarray, classes: list[str], order: list[str]) -> np.ndarray:
    idx = [list(classes).index(c) for c in order]
    return proba[:, idx]


def ranked_probability_score(y_true, proba: np.ndarray, classes: list[str]) -> float:
    """Mean RPS over the ordinal H-D-A scale. Lower is better; 0 is perfect."""
    p = _align(proba, classes, ORDINAL)
    y = np.zeros_like(p)
    lookup = {c: i for i, c in enumerate(ORDINAL)}
    for i, label in enumerate(np.asarray(y_true)):
        y[i, lookup[label]] = 1.0
    cum_p, cum_y = np.cumsum(p, axis=1), np.cumsum(y, axis=1)
    # Sum the first r-1 cumulative differences, normalised by r-1 so the score
    # stays on the conventional 0..1 scale (Epstein 1969).
    r = len(ORDINAL)
    return float(np.mean(np.sum((cum_p[:, :-1] - cum_y[:, :-1]) ** 2, axis=1)) / (r - 1))


def multiclass_brier(y_true, proba: np.ndarray, classes: list[str]) -> float:
    p = _align(proba, classes, ORDINAL)
    y = np.zeros_like(p)
    lookup = {c: i for i, c in enumerate(ORDINAL)}
    for i, label in enumerate(np.asarray(y_true)):
        y[i, lookup[label]] = 1.0
    return float(np.mean(np.sum((p - y) ** 2, axis=1)))


def _log_loss(y_true, proba: np.ndarray, classes: list[str]) -> float:
    """log_loss assumes lexicographic label order, so sort before scoring."""
    order = sorted(classes)
    return float(log_loss(y_true, _align(proba, classes, order), labels=order))


def evaluate(y_true, proba: np.ndarray, classes: list[str]) -> dict:
    """Headline metrics for one set of predictions."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(classes)[np.argmax(proba, axis=1)]
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=RESULT_CLASSES, zero_division=0)

    metrics = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "log_loss": _log_loss(y_true, proba, classes),
        "brier": multiclass_brier(y_true, proba, classes),
        "rps": ranked_probability_score(y_true, proba, classes),
    }
    for i, cls in enumerate(RESULT_CLASSES):
        metrics[f"precision_{cls}"] = float(prec[i])
        metrics[f"recall_{cls}"] = float(rec[i])
        metrics[f"f1_{cls}"] = float(f1[i])
    return metrics


def report_text(y_true, proba: np.ndarray, classes: list[str]) -> str:
    y_pred = np.asarray(classes)[np.argmax(proba, axis=1)]
    rep = classification_report(y_true, y_pred, labels=RESULT_CLASSES,
                                target_names=["Home win", "Draw", "Away win"],
                                zero_division=0, digits=3)
    cm = confusion_matrix(y_true, y_pred, labels=RESULT_CLASSES)
    cm_df = pd.DataFrame(cm, index=[f"true {c}" for c in RESULT_CLASSES],
                         columns=[f"pred {c}" for c in RESULT_CLASSES])
    return f"{rep}\nConfusion matrix\n{cm_df.to_string()}"


# --- baselines --------------------------------------------------------------
def baseline_prior(y_train, y_test) -> dict:
    """Always predict the training-set outcome distribution."""
    prior = pd.Series(y_train).value_counts(normalize=True)
    classes = RESULT_CLASSES
    p = np.tile([prior.get(c, 0.0) for c in classes], (len(y_test), 1))
    p = p / p.sum(axis=1, keepdims=True)
    return evaluate(y_test, p, classes)


def baseline_always_home(y_test) -> dict:
    classes = RESULT_CLASSES
    p = np.tile([0.999, 0.0005, 0.0005], (len(y_test), 1))
    return evaluate(y_test, p, classes)


def baseline_market(test_df: pd.DataFrame) -> dict | None:
    """Bookmaker closing odds - the benchmark any model has to beat."""
    cols = ["p_market_home", "p_market_draw", "p_market_away"]
    if not set(cols).issubset(test_df.columns):
        return None
    sub = test_df.dropna(subset=cols)
    if len(sub) < 50:
        return None
    p = sub[cols].to_numpy(dtype=float)
    p = p / p.sum(axis=1, keepdims=True)
    return evaluate(sub["result"], p, RESULT_CLASSES)


def calibration_table(y_true, proba: np.ndarray, classes: list[str], bins: int = 10
                      ) -> pd.DataFrame:
    """Predicted vs observed frequency, pooled over all three outcomes."""
    y_true = np.asarray(y_true)
    rows = []
    edges = np.linspace(0, 1, bins + 1)
    for i, cls in enumerate(classes):
        p = proba[:, i]
        hit = (y_true == cls).astype(float)
        idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
        for b in range(bins):
            m = idx == b
            if m.sum() >= 20:
                rows.append({"outcome": cls, "bin": f"{edges[b]:.1f}-{edges[b+1]:.1f}",
                             "n": int(m.sum()), "predicted": float(p[m].mean()),
                             "observed": float(hit[m].mean())})
    return pd.DataFrame(rows)
