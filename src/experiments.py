"""Which changes actually make the model better? Measured, not assumed.

Every candidate is scored by rolling-origin cross-validation on the training
period only (matches before the holdout starts), using the same folds, so the
comparison is like for like. The holdout is not used to choose anything here -
it is reported once, by `src.train`, for whatever wins.

    python -m src.experiments         # writes reports/experiments.json
"""
from __future__ import annotations

import argparse
import json
import logging
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

from . import data as data_mod
from . import evaluate as ev
from .config import REPORTS
from .features import (
    BASE_FEATURES,
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    ORIGINAL_ELO,
    RATING_FEATURES,
    FeatureBuilder,
    build_features,
)
from .model import _preprocessor
from .ordinal import ProportionalOdds
from .ratings import PiParams, PoissonParams
from .train import chronological_split

log = logging.getLogger(__name__)

# The set the model now trains on; see features.MODEL_FEATURES.
LEAN_FEATURES = MODEL_FEATURES

COMPACT_FEATURES = [
    "elo_home", "elo_away", "elo_diff", "elo_abs_diff", "elo_expected_home",
    "pi_home", "pi_away", "pi_expected_gd",
    "xg_home", "xg_away", "xg_diff", "pois_p_home", "pois_p_draw", "pois_p_away",
    "sot_rate_home", "sot_rate_away", "sot_rate_diff",
    "league_home_rate", "league_draw_rate", "league_goals",
    "ppg10_diff", "gd5_diff", "venue_ppg_home", "venue_ppg_away",
    "rest_diff", "season_match_home",
]


def logistic(features, C=0.5):
    num = [f for f in features if f not in CATEGORICAL_FEATURES]
    cat = [f for f in features if f in CATEGORICAL_FEATURES]
    return Pipeline([("prep", _preprocessor(num, cat, scale=True)),
                     ("clf", LogisticRegression(max_iter=3000, C=C))])


def boosting(features, **kw):
    num = [f for f in features if f not in CATEGORICAL_FEATURES]
    cat = [f for f in features if f in CATEGORICAL_FEATURES]
    params = dict(max_iter=500, learning_rate=0.03, max_leaf_nodes=15,
                  min_samples_leaf=80, l2_regularization=1.0, random_state=42)
    params.update(kw)
    return Pipeline([("prep", _preprocessor(num, cat, scale=False)),
                     ("clf", HistGradientBoostingClassifier(**params))])


def ordinal(features, alpha=500.0):
    """Proportional-odds model: one score per match on the ordered A-D-H scale."""
    num = [f for f in features if f not in CATEGORICAL_FEATURES]
    cat = [f for f in features if f in CATEGORICAL_FEATURES]
    return Pipeline([("prep", _preprocessor(num, cat, scale=True)),
                     ("clf", ProportionalOdds(alpha=alpha))])


def with_league_interactions(df, features):
    """Let the rating gap and home advantage act differently in each league.

    The league-context features that did nothing were rolling rates - how often
    home teams won lately. This is a different question: whether league identity
    changes how much a rating gap is worth, which the pooled model cannot say.
    """
    out, names = df.copy(), list(features)
    leagues = sorted(df["league"].unique())[1:]      # one league is the baseline
    for lg in leagues:
        flag = (df["league"] == lg).astype(float)
        for base in ("elo_diff", "elo_expected_home"):
            name = f"{base}_x_{lg}"
            out[name] = df[base] * flag
            names.append(name)
    return out, names


def recency_weights(dates, halflife_years):
    if not halflife_years:
        return None
    age = (dates.max() - dates).dt.days.to_numpy() / 365.25
    return 0.5 ** (age / halflife_years)


POISSON_COLS = ["pois_p_away", "pois_p_draw", "pois_p_home"]   # sklearn class order A, D, H


def cv_score(df, make_model, features, *, halflife=None, blend_with=None,
             poisson_weight=0.0, league_blend=0.0, n_splits=5):
    """Mean fold metrics.

    make_model=None scores the Poisson rating model's own probabilities, with no
    learned layer at all. `poisson_weight` mixes those into a learned model's
    output; `blend_with` averages with a second learned model.
    """
    X, y = df[features], df["result"]
    folds = []
    classes = ["A", "D", "H"]
    for tr, te in TimeSeriesSplit(n_splits=n_splits).split(X):
        direct = df[POISSON_COLS].iloc[te].to_numpy()
        if make_model is None:
            proba = direct
        else:
            w = recency_weights(df.date.iloc[tr], halflife)
            fit_kw = {"clf__sample_weight": w} if w is not None else {}
            m = make_model(features)
            m.fit(X.iloc[tr], y.iloc[tr], **fit_kw)
            proba = m.predict_proba(X.iloc[te])
            assert list(m.named_steps["clf"].classes_) == classes
            if blend_with is not None:
                m2 = blend_with(features)
                m2.fit(X.iloc[tr], y.iloc[tr], **fit_kw)
                proba = (proba + m2.predict_proba(X.iloc[te])) / 2
            if league_blend:
                # Hierarchical-ish: mix the pooled model with one fitted on that
                # league alone, so a league can differ without going it alone.
                own = np.zeros_like(proba)
                train_lg, test_lg = df.league.iloc[tr], df.league.iloc[te]
                for lg in test_lg.unique():
                    rows = train_lg == lg
                    mask = (test_lg == lg).to_numpy()
                    if rows.sum() < 500:
                        own[mask] = proba[mask]
                        continue
                    local = make_model(features)
                    local.fit(X.iloc[tr][rows.to_numpy()], y.iloc[tr][rows.to_numpy()])
                    own[mask] = local.predict_proba(X.iloc[te][mask])
                proba = (1 - league_blend) * proba + league_blend * own
            if poisson_weight:
                proba = (1 - poisson_weight) * proba + poisson_weight * direct
        folds.append(ev.evaluate(y.iloc[te], proba, classes))
    return {k: float(np.mean([f[k] for f in folds])) for k in ("rps", "log_loss", "accuracy", "brier")}


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    t0 = time.time()
    matches = data_mod.load()

    tuned, _ = build_features(matches, min_history=5)
    original, _ = build_features(matches, min_history=5, builder=FeatureBuilder(
        elo=ORIGINAL_ELO, pi=PiParams(), poisson=PoissonParams()))
    train_t, test_t = chronological_split(tuned, 0.2)
    train_o, _ = chronological_split(original, 0.2)
    log.info("training period: %d matches to %s (holdout untouched from %s)\n",
             len(train_t), train_t.date.max().date(), test_t.date.min().date())

    base = BASE_FEATURES + CATEGORICAL_FEATURES
    full = BASE_FEATURES + RATING_FEATURES + CATEGORICAL_FEATURES
    compact = COMPACT_FEATURES + CATEGORICAL_FEATURES
    results = []

    def run(name, df, make, feats, **kw):
        t = time.time()
        r = cv_score(df, make, feats, **kw)
        r.update(name=name, n_features=len(feats), seconds=round(time.time() - t, 1))
        results.append(r)
        log.info("  %-52s rps %.5f  logloss %.4f  acc %.4f", name, r["rps"], r["log_loss"], r["accuracy"])
        return r

    log.info("Features (logistic regression, C=0.5 as in the current model)")
    run("1. current model (original Elo, original features)", train_o, logistic, base)
    run("2. + tuned Elo and promoted-club prior", train_t, logistic, base)
    run("3. + pi, Poisson, shots and league ratings", train_t, logistic, full)
    run("4. ratings-led compact set", train_t, logistic, compact)

    lean = LEAN_FEATURES + CATEGORICAL_FEATURES
    run("5. lean ratings set", train_t, logistic, lean)

    log.info("\nRegularisation")
    for label, feats in (("base", base), ("full", full), ("compact", compact), ("lean", lean)):
        grid = (0.002, 0.005, 0.01, 0.05, 0.1, 0.25) if label == "lean" else (0.01, 0.05, 0.1, 0.25)
        for C in grid:
            run(f"{label}  C={C}", train_t, lambda f, C=C: logistic(f, C), feats)

    log.info("\nThe Poisson rating model on its own, and mixed in")
    run("Poisson probabilities, no learned model", train_t, None, lean)
    for label, feats, C in (("base", base, 0.05), ("lean", lean, 0.05), ("lean", lean, 0.01)):
        for wt in (0.25, 0.5, 0.75):
            run(f"{label} C={C} mixed {int(wt*100)}% with Poisson", train_t,
                lambda f, C=C: logistic(f, C), feats, poisson_weight=wt)

    log.info("\nOther learners on the lean set")
    run("gradient boosting (lean)", train_t, boosting, lean)
    run("logistic C=0.05 + boosting blend (lean)", train_t,
        lambda f: logistic(f, 0.05), lean, blend_with=boosting)
    run("logistic C=0.05, recency-weighted 6y (lean)", train_t,
        lambda f: logistic(f, 0.05), lean, halflife=6)

    log.info("\nOrdered outcome: proportional odds vs multinomial")
    run("multinomial logistic C=0.002 (current)", train_t, lambda f: logistic(f, 0.002), lean)
    for alpha in (50, 200, 500, 1000, 2000, 5000):
        run(f"proportional odds alpha={alpha}", train_t, lambda f, a=alpha: ordinal(f, a), lean)

    log.info("\nLeague-specific structure")
    inter_df, inter_feats = with_league_interactions(train_t, lean)
    for C in (0.002, 0.01):
        run(f"league x rating interactions, C={C}", inter_df,
            lambda f, C=C: logistic(f, C), inter_feats)
    for w in (0.25, 0.5):
        run(f"pooled blended {int(w*100)}% with per-league models", train_t,
            lambda f: logistic(f, 0.002), lean, league_blend=w)

    best = min(results, key=lambda r: r["rps"])
    log.info("\nBEST: %s  rps %.5f  logloss %.4f  acc %.4f",
             best["name"], best["rps"], best["log_loss"], best["accuracy"])

    out = {"training_end": str(train_t.date.max().date()), "results": results,
           "seconds": round(time.time() - t0, 1)}
    (REPORTS / "experiments.json").write_text(json.dumps(out, indent=2))
    log.info("\n-> reports/experiments.json (%.0fs)", time.time() - t0)


if __name__ == "__main__":
    main()
