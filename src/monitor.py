"""Monitoring: does the live model still work on the data arriving now?

Football is not stationary. Home advantage drifted sharply during the 2020-21
empty-stadium seasons, league scoring rates move, squads turn over. A model
trained once and left alone quietly decays, so this module re-scores the saved
model against recent matches and flags degradation against its own baseline.
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import data as data_mod
from . import evaluate as ev
from .config import REPORTS
from .features import FeatureBuilder, build_features
from .predict import Predictor

log = logging.getLogger(__name__)

# Relative degradation in RPS beyond which the model is considered stale.
DRIFT_TOLERANCE = 0.05


def _proba(predictor: Predictor, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    X = df.reindex(columns=predictor.features)
    return predictor.pipeline.predict_proba(X), predictor.classes


def breakdown(df: pd.DataFrame, proba: np.ndarray, classes: list[str],
              by: str = "season") -> pd.DataFrame:
    """Metrics sliced by season, league, or any other column."""
    rows = []
    for key, idx in df.groupby(by).groups.items():
        pos = df.index.get_indexer(idx)
        m = ev.evaluate(df.loc[idx, "result"], proba[pos], classes)
        rows.append({by: key, **{k: m[k] for k in
                                 ("n", "accuracy", "f1_macro", "log_loss", "rps")}})
    return pd.DataFrame(rows).sort_values(by).reset_index(drop=True)


def home_advantage_trend(matches: pd.DataFrame) -> pd.DataFrame:
    """Share of home wins per season - the clearest drift signal in this sport."""
    m = matches.copy()
    m["season_start"] = m["date"].dt.year - (m["date"].dt.month < 7).astype(int)
    g = m.groupby("season_start")
    return pd.DataFrame({
        "matches": g.size(),
        "home_win_rate": g["result"].apply(lambda s: (s == "H").mean()),
        "draw_rate": g["result"].apply(lambda s: (s == "D").mean()),
        "away_win_rate": g["result"].apply(lambda s: (s == "A").mean()),
        "goals_per_game": g.apply(lambda d: (d.home_goals + d.away_goals).mean(),
                                  include_groups=False),
    }).round(3).reset_index()


def season_drift(matches: pd.DataFrame, predictor) -> dict:
    """Season-by-season: how the sport changed, and how the model scored.

    The monitor writes this to JSON, where nobody reads it. It is the most
    interesting output in the project - home advantage visibly collapsing in
    2020 and never fully returning - so the app shows it.

    Honest caveat, carried in the response: the shipped model is trained on all
    of this, so its per-season scores are not out-of-sample. They still show how
    predictable each season was; they are not evidence of accuracy.
    """
    feats, _ = build_features(matches, min_history=5)
    feats = FeatureBuilder._chronological(feats)
    proba = predictor.pipeline.predict_proba(feats.reindex(columns=predictor.features))
    scored = breakdown(feats, proba, predictor.classes, "season")

    def season_code(value) -> str:
        """Reading the CSV turns '0506' into 506; put the zeros back."""
        try:
            return f"{int(value):04d}"
        except (TypeError, ValueError):
            return str(value)

    def season_start(code: str) -> int:
        """'1516' -> 2015, so seasons sort and label correctly."""
        head = int(season_code(code)[:2])
        return 2000 + head if head < 50 else 1900 + head

    # Group the outcome mix by the season label rather than by date: the
    # 2019/20 season finished in July 2020, and splitting on month puts those
    # matches in the wrong season.
    played = matches.assign(season=matches["season"].map(season_code))
    mix = played.groupby("season").apply(lambda g: pd.Series({
        "home_win_rate": float((g.result == "H").mean()),
        "draw_rate": float((g.result == "D").mean()),
        "away_win_rate": float((g.result == "A").mean()),
        "goals_per_game": float((g.home_goals + g.away_goals).mean()),
    }), include_groups=False).to_dict("index")

    rows = []
    for r in scored.to_dict("records"):
        start = season_start(r["season"])
        m = mix.get(season_code(r["season"]), {})
        rows.append({
            "season": f"{start}/{str(start + 1)[2:]}",
            "start": start,
            "matches": int(r["n"]),
            "rps": round(float(r["rps"]), 4),
            "accuracy": round(float(r["accuracy"]), 4),
            "home_win_rate": round(m.get("home_win_rate", float("nan")), 4),
            "draw_rate": round(m.get("draw_rate", float("nan")), 4),
            "away_win_rate": round(m.get("away_win_rate", float("nan")), 4),
            "goals_per_game": round(m.get("goals_per_game", float("nan")), 3),
        })
    rows.sort(key=lambda r: r["start"])
    return {
        "seasons": rows,
        "holdout_rps": (predictor.meta.get("holdout_metrics") or {}).get("rps"),
        "note": ("The model was trained on these matches, so the per-season scores "
                 "are not out-of-sample. They show how predictable each season was."),
    }


def drift_check(recent: dict, baseline: dict, tolerance: float = DRIFT_TOLERANCE) -> dict:
    """Compare recent RPS/accuracy against the metrics recorded at training time."""
    base_rps = baseline.get("rps")
    if not base_rps:
        return {"status": "unknown", "reason": "no baseline metrics in model bundle"}

    delta = (recent["rps"] - base_rps) / base_rps
    status = "degraded" if delta > tolerance else "ok"
    return {
        "status": status,
        "baseline_rps": round(base_rps, 4),
        "recent_rps": round(recent["rps"], 4),
        "relative_change": round(delta, 4),
        "tolerance": tolerance,
        "baseline_accuracy": round(baseline.get("accuracy", float("nan")), 4),
        "recent_accuracy": round(recent["accuracy"], 4),
        "recommendation": ("retrain: recent ranked probability score is worse than the "
                           "held-out baseline by more than the tolerance"
                           if status == "degraded" else
                           "no action: recent performance is within tolerance"),
    }


def run(args) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    predictor = Predictor.load(args.model)
    matches = data_mod.load(args.dataset)
    log.info("model trained %s on %s matches (%s)",
             predictor.meta.get("trained_at"), predictor.meta.get("train_rows"),
             predictor.meta.get("model_name"))

    feats, _ = build_features(matches, min_history=5)
    feats = FeatureBuilder._chronological(feats)

    cutoff = pd.Timestamp(args.since) if args.since else \
        feats.date.max() - pd.Timedelta(days=args.window_days)
    recent = feats[feats.date >= cutoff].reset_index(drop=True)
    if len(recent) < 100:
        raise SystemExit(f"only {len(recent)} matches since {cutoff.date()} - widen the window")

    proba, classes = _proba(predictor, recent)
    recent_metrics = ev.evaluate(recent["result"], proba, classes)
    log.info("\nrecent window: %s .. %s (%d matches)",
             recent.date.min().date(), recent.date.max().date(), len(recent))
    log.info("acc %.3f  f1m %.3f  logloss %.4f  rps %.4f",
             recent_metrics["accuracy"], recent_metrics["f1_macro"],
             recent_metrics["log_loss"], recent_metrics["rps"])

    # NOTE: these slices include matches the model was trained on, so they read as
    # a consistency check across eras rather than as clean out-of-sample scores.
    all_proba, _ = _proba(predictor, feats)
    by_season = breakdown(feats, all_proba, classes, "season")
    by_league = breakdown(feats, all_proba, classes, "league")
    log.info("\nBy season\n%s", by_season.to_string(index=False,
             float_format=lambda v: f"{v:.3f}"))
    log.info("\nBy league\n%s", by_league.to_string(index=False,
             float_format=lambda v: f"{v:.3f}"))

    trend = home_advantage_trend(matches)
    log.info("\nOutcome mix by season (drift in the sport itself)\n%s",
             trend.to_string(index=False))

    drift = drift_check(recent_metrics, predictor.meta.get("holdout_metrics", {}),
                        args.tolerance)
    log.info("\nDrift: %s (%+.1f%% RPS vs baseline)\n%s",
             drift["status"].upper(), 100 * drift.get("relative_change", 0),
             drift["recommendation"])

    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": predictor.meta.get("model_name"),
        "model_trained_at": predictor.meta.get("trained_at"),
        "window": {"from": str(recent.date.min().date()),
                   "to": str(recent.date.max().date()), "matches": len(recent)},
        "recent_metrics": recent_metrics,
        "drift": drift,
        "by_season": by_season.to_dict("records"),
        "by_league": by_league.to_dict("records"),
        "outcome_mix_by_season": trend.to_dict("records"),
    }
    path = REPORTS / "monitoring_report.json"
    path.write_text(json.dumps(report, indent=2, default=str))
    log.info("report -> %s", path)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Monitor a deployed model for drift.")
    ap.add_argument("--model", default="model.joblib")
    ap.add_argument("--dataset", default="matches.csv")
    ap.add_argument("--window-days", type=int, default=365)
    ap.add_argument("--since", default=None, help="explicit start date (YYYY-MM-DD)")
    ap.add_argument("--tolerance", type=float, default=DRIFT_TOLERANCE)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
