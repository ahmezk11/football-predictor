"""Train, cross-validate, compare and persist match-outcome models.

Splits are always chronological. Football data is a time series: shuffling it
would let a model learn from the future, which inflates every metric and tells
you nothing about how the thing will behave on next weekend's fixtures.
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.model_selection import TimeSeriesSplit

from . import data as data_mod
from . import evaluate as ev
from .config import MODELS, REPORTS
from .features import FEATURE_VERSION, FeatureBuilder, build_features, feature_columns
from .model import MODEL_NAMES, build_model

log = logging.getLogger(__name__)


def chronological_split(df: pd.DataFrame, test_fraction: float = 0.2
                        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out the most recent `test_fraction` of matches by date."""
    df = FeatureBuilder._chronological(df)
    cut = int(len(df) * (1 - test_fraction))
    # Do not split a single day across the boundary.
    cut_date = df.loc[cut, "date"]
    train = df[df.date < cut_date]
    test = df[df.date >= cut_date]
    return train.reset_index(drop=True), test.reset_index(drop=True)


def cross_validate(df: pd.DataFrame, model_name: str, features: list[str],
                   n_splits: int = 5, seed: int = 42, *, calibrate: bool = False) -> dict:
    """Rolling-origin CV: every fold trains on the past, tests on the future."""
    df = FeatureBuilder._chronological(df)
    X, y = df[features], df["result"]
    splitter = TimeSeriesSplit(n_splits=n_splits)

    fold_metrics = []
    for k, (tr, te) in enumerate(splitter.split(X), start=1):
        pipe = train_final(df.iloc[tr], model_name, features,
                           calibrate=calibrate, seed=seed)
        proba = pipe.predict_proba(X.iloc[te])
        m = ev.evaluate(y.iloc[te], proba, _classes_of(pipe))
        m["fold"] = k
        m["train_end"] = str(df.date.iloc[tr[-1]].date())
        fold_metrics.append(m)
        log.info("  fold %d (train<=%s, n=%d): acc %.3f  rps %.4f  logloss %.4f",
                 k, m["train_end"], m["n"], m["accuracy"], m["rps"], m["log_loss"])

    folds = pd.DataFrame(fold_metrics)
    numeric = folds.select_dtypes(include=[np.number])
    return {
        "folds": fold_metrics,
        "mean": numeric.mean().to_dict(),
        "std": numeric.std(ddof=0).to_dict(),
    }


def _classes_of(pipeline) -> list[str]:
    est = pipeline if hasattr(pipeline, "classes_") else pipeline.named_steps["clf"]
    return list(est.classes_)


def train_final(train_df: pd.DataFrame, model_name: str, features: list[str],
                *, calibrate: bool, seed: int = 42):
    """Fit on all training data, optionally wrapping in probability calibration."""
    pipe = build_model(model_name, features, seed=seed)
    X, y = train_df[features], train_df["result"]
    if calibrate:
        # Isotonic needs a decent sample; sigmoid is safer on smaller sets.
        method = "isotonic" if len(train_df) >= 5000 else "sigmoid"
        pipe = CalibratedClassifierCV(pipe, method=method, cv=TimeSeriesSplit(n_splits=4))
    pipe.fit(X, y)
    return pipe


def feature_importance(pipeline, test_df: pd.DataFrame, features: list[str],
                       *, n_repeats: int = 5, seed: int = 42, top: int = 20
                       ) -> pd.DataFrame:
    """Permutation importance on held-out data.

    Model-agnostic on purpose: it measures how much the held-out ranked
    probability score suffers when a feature is shuffled, so it is comparable
    across a linear model and a forest, and it reflects predictive value rather
    than how often a tree happened to split on something.
    """
    def scorer(est, X, y):
        proba = est.predict_proba(X)
        classes = list(est.classes_ if hasattr(est, "classes_")
                       else est.named_steps["clf"].classes_)
        return -ev.ranked_probability_score(y, proba, classes)  # higher is better

    result = permutation_importance(
        pipeline, test_df[features], test_df["result"],
        scoring=scorer, n_repeats=n_repeats, random_state=seed, n_jobs=1)
    imp = pd.DataFrame({
        "feature": features,
        "rps_increase": result.importances_mean,
        "std": result.importances_std,
    }).sort_values("rps_increase", ascending=False).reset_index(drop=True)
    return imp.head(top)


def _fmt(metrics: dict) -> str:
    return (f"acc {metrics['accuracy']:.3f}  f1m {metrics['f1_macro']:.3f}  "
            f"logloss {metrics['log_loss']:.4f}  rps {metrics['rps']:.4f}  "
            f"brier {metrics['brier']:.4f}")


def run(args) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    matches = data_mod.load(args.dataset)
    if args.leagues:
        matches = matches[matches.league.isin(args.leagues)]
    log.info("%d matches, %s .. %s, leagues: %s", len(matches),
             matches.date.min().date(), matches.date.max().date(),
             ", ".join(sorted(matches.league.unique())))

    if args.use_odds:
        coverage = matches.get("odds_home", pd.Series(dtype=float)).notna().mean()
        if coverage < 0.5:
            raise SystemExit(
                f"--use-odds needs bookmaker odds, but only {coverage:.1%} of rows have "
                "them. Run `python -m src.data --source both` first (football-data.co.uk "
                "rate-limits heavily, so this can take a while).")
        log.info("market odds present on %.1f%% of rows", 100 * coverage)

    log.info("building features ...")
    feats, builder = build_features(matches, min_history=args.min_history)
    features = feature_columns(args.use_odds)
    features = [f for f in features if f in feats.columns]
    log.info("%d usable rows, %d features (odds %s)",
             len(feats), len(features), "in" if args.use_odds else "out")

    train_df, test_df = chronological_split(feats, args.test_fraction)
    log.info("train %d (%s..%s) | test %d (%s..%s)\n",
             len(train_df), train_df.date.min().date(), train_df.date.max().date(),
             len(test_df), test_df.date.min().date(), test_df.date.max().date())

    # --- baselines ---
    baselines = {
        "prior (train distribution)": ev.baseline_prior(train_df.result, test_df.result),
        "always home win": ev.baseline_always_home(test_df.result),
    }
    market = ev.baseline_market(test_df)
    if market:
        baselines["bookmaker closing odds"] = market
    log.info("Baselines on the held-out period")
    for name, m in baselines.items():
        log.info("  %-26s %s", name, _fmt(m))

    # --- model comparison via rolling-origin CV ---
    candidates = args.models or MODEL_NAMES
    cv_results, holdout, fitted = {}, {}, {}
    for name in candidates:
        log.info("\n[%s] cross-validation", name)
        cv_results[name] = cross_validate(train_df, name, features,
                                          n_splits=args.cv_splits, seed=args.seed)
        pipe = train_final(train_df, name, features, calibrate=False, seed=args.seed)
        proba = pipe.predict_proba(test_df[features])
        classes = list(pipe.named_steps["clf"].classes_)
        holdout[name] = ev.evaluate(test_df.result, proba, classes)
        fitted[name] = pipe
        log.info("  holdout: %s", _fmt(holdout[name]))

    # Selection is on RPS: it scores the whole probability vector and respects
    # the H-D-A ordering, so it punishes confident wrong calls properly.
    best = min(candidates, key=lambda n: cv_results[n]["mean"]["rps"])
    log.info("\nselected: %s (mean CV RPS %.4f)", best, cv_results[best]["mean"]["rps"])

    # --- does post-hoc calibration actually help this model? ---
    # Decided on the training folds only. Choosing it by holdout score would be
    # tuning on the very data the headline numbers are supposed to be honest about.
    if args.calibrate == "auto":
        log.info("\n[%s] calibration check", best)
        cal_cv = cross_validate(train_df, best, features, n_splits=args.cv_splits,
                                seed=args.seed, calibrate=True)
        raw_rps = cv_results[best]["mean"]["rps"]
        cal_rps = cal_cv["mean"]["rps"]
        calibrate = cal_rps < raw_rps
        log.info("  raw CV RPS %.4f vs calibrated %.4f -> %s",
                 raw_rps, cal_rps, "calibrate" if calibrate else "leave uncalibrated")
    else:
        calibrate = args.calibrate == "on"

    # --- refit the winner on everything ---
    final_pipe = train_final(feats, best, features, calibrate=calibrate, seed=args.seed)
    eval_pipe = train_final(train_df, best, features, calibrate=calibrate, seed=args.seed)
    proba = eval_pipe.predict_proba(test_df[features])
    classes = _classes_of(eval_pipe)
    final_metrics = ev.evaluate(test_df.result, proba, classes)

    log.info("\n=== held-out performance: %s%s ===", best,
             " (calibrated)" if calibrate else "")
    log.info("%s", _fmt(final_metrics))
    log.info("\n%s", ev.report_text(test_df.result, proba, classes))
    # Season-by-season on the holdout: shows whether accuracy is stable over time
    # or the model is drifting against a changing sport.
    season_rows = []
    for season, idx in test_df.groupby("season").groups.items():
        pos = test_df.index.get_indexer(idx)
        m = ev.evaluate(test_df.loc[idx, "result"], proba[pos], classes)
        season_rows.append({"season": season, "n": m["n"], "accuracy": m["accuracy"],
                            "rps": m["rps"], "log_loss": m["log_loss"]})
    by_season = pd.DataFrame(season_rows).sort_values("season")
    log.info("\nHeld-out performance by season\n%s",
             by_season.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    calib = ev.calibration_table(test_df.result, proba, classes)
    if len(calib):
        log.info("\nCalibration (predicted vs observed)\n%s",
                 calib.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    log.info("\ncomputing permutation importance ...")
    importance = feature_importance(eval_pipe, test_df, features,
                                    n_repeats=args.importance_repeats, seed=args.seed)
    log.info("\nTop features (held-out RPS increase when shuffled)\n%s",
             importance.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    # --- persist ---
    bundle = {
        "pipeline": final_pipe,
        "builder": builder,
        "features": features,
        "classes": list(classes),
        "model_name": best,
        "use_odds": args.use_odds,
        "calibrated": calibrate,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "feature_version": FEATURE_VERSION,
        "train_rows": int(len(feats)),
        "data_range": [str(matches.date.min().date()), str(matches.date.max().date())],
        "leagues": sorted(matches.league.unique().tolist()),
        "holdout_metrics": final_metrics,
    }
    model_path = MODELS / args.out
    joblib.dump(bundle, model_path)
    log.info("\nmodel -> %s", model_path)

    report = {
        "trained_at": bundle["trained_at"],
        "rows": int(len(feats)),
        "features": features,
        "use_odds": args.use_odds,
        "calibrated": calibrate,
        "baselines": baselines,
        "cv": {k: {"mean": v["mean"], "std": v["std"], "folds": v["folds"]}
               for k, v in cv_results.items()},
        "holdout_by_model": holdout,
        "selected": best,
        "selected_holdout": final_metrics,
        "calibration": calib.to_dict("records") if len(calib) else [],
        "holdout_by_season": by_season.to_dict("records"),
        "feature_importance": importance.to_dict("records"),
    }
    report_path = REPORTS / f"training_report{'_odds' if args.use_odds else ''}.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    log.info("report -> %s", report_path)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Train match outcome models.")
    ap.add_argument("--dataset", default="matches.csv")
    ap.add_argument("--leagues", nargs="*", default=None)
    ap.add_argument("--models", nargs="*", default=None, choices=MODEL_NAMES)
    ap.add_argument("--use-odds", action="store_true",
                    help="include bookmaker implied probabilities as features")
    ap.add_argument("--calibrate", choices=["auto", "on", "off"], default="auto",
                    help="post-hoc probability calibration; 'auto' picks by CV")
    ap.add_argument("--test-fraction", type=float, default=0.2)
    ap.add_argument("--cv-splits", type=int, default=5)
    ap.add_argument("--min-history", type=int, default=5)
    ap.add_argument("--importance-repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="model.joblib")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
