"""Keeping the app current: pull new results, settle predictions, retrain if needed.

Retraining is deliberately conditional. Pulling a handful of new matches rarely
moves a model fitted on 38,000, so this checks the drift monitor first and only
spends the time when recent performance has actually slipped - or when enough
new matches have accumulated to be worth it.
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta, timezone

import pandas as pd

from . import data as data_mod
from . import evaluate as ev
from .config import season_codes
from .features import build_features
from .monitor import drift_check
from .predict import Predictor
from .store import Store

log = logging.getLogger(__name__)

RETRAIN_AFTER_NEW_MATCHES = 400   # roughly a full round across five leagues


def refresh_data(*, to_year: int | None = None) -> dict:
    """Re-download recent seasons and rebuild the consolidated dataset."""
    now = datetime.now(timezone.utc).date()
    end = to_year if to_year is not None else (now.year if now.month >= 7 else now.year - 1)

    try:
        before = len(data_mod.load())
    except FileNotFoundError:
        before = 0

    # Only the last two seasons can still change; earlier files are final, so
    # they stay cached and cost nothing.
    fresh = season_codes(end - 1, end)
    for season in fresh:
        for league in data_mod.LEAGUES:
            data_mod.download_season(league, season, source="mirror", refresh=True)

    data = data_mod.build_dataset(seasons=season_codes(2005, end), source="mirror")
    data_mod.save(data)
    return {"matches_before": before, "matches_after": len(data),
            "new_matches": len(data) - before,
            "latest_match": str(data.date.max().date())}


def unseen_matches(matches, predictor: Predictor) -> int:
    """Matches played after the model's training data ends.

    Counting rows added to the local CSV instead was wrong on an app's first run:
    its data folder starts empty, so all 38,000 matches looked new and forced a
    full retrain whose output then shadowed the model shipped with the app.
    """
    end = (predictor.meta.get("data_range") or [None, None])[1]
    if not end:
        return 0
    return int((matches["date"] > pd.Timestamp(end)).sum())


def settle_predictions(store: Store, predictor: Predictor) -> dict:
    """Attach real results to predictions whose matches have now been played."""
    from .fixtures import FixtureFeed, TeamResolver

    open_preds = store.open_predictions()
    if not open_preds:
        return {"settled": 0, "still_open": 0}

    known = {lg["code"]: predictor.teams(lg["code"]) for lg in predictor.leagues()}
    feed = FixtureFeed(TeamResolver(known))
    since = min(p["match_date"] for p in open_preds)
    results = feed.results_since(sorted({p["league"] for p in open_preds}), since)

    settled = store.settle(results)
    return {"settled": settled, "still_open": len(store.open_predictions())}


def should_retrain(store: Store, predictor: Predictor, new_matches: int,
                   *, force: bool = False, tolerance: float = 0.05) -> dict:
    """Decide whether a retrain is worth the time, and say why."""
    if force:
        return {"retrain": True, "reason": "forced"}
    if new_matches >= RETRAIN_AFTER_NEW_MATCHES:
        return {"retrain": True,
                "reason": f"{new_matches} matches played since the model's training data ends"}

    try:
        matches = data_mod.load()
    except FileNotFoundError:
        return {"retrain": False, "reason": "no dataset"}

    feats, _ = build_features(matches, min_history=5)
    cutoff = feats.date.max() - timedelta(days=365)
    recent = feats[feats.date >= cutoff]
    if len(recent) < 200:
        return {"retrain": False, "reason": "not enough recent matches to judge drift"}

    proba = predictor.pipeline.predict_proba(recent.reindex(columns=predictor.features))
    metrics = ev.evaluate(recent["result"], proba, predictor.classes)
    drift = drift_check(metrics, predictor.meta.get("holdout_metrics", {}), tolerance)
    return {"retrain": drift["status"] == "degraded",
            "reason": drift["recommendation"], "drift": drift}


def run(args) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    store = Store()

    log.info("refreshing data ...")
    data_info = refresh_data()
    log.info("  %d matches (%+d new), latest %s", data_info["matches_after"],
             data_info["new_matches"], data_info["latest_match"])
    store.set_meta("last_data_refresh", datetime.now(timezone.utc).isoformat(timespec="seconds"))

    report = {"data": data_info}
    try:
        predictor = Predictor.load()
    except FileNotFoundError:
        log.info("no model yet - training from scratch")
        report["decision"] = {"retrain": True, "reason": "no model"}
        _retrain(args)
        return report

    settle = settle_predictions(store, predictor)
    log.info("settled %d prediction(s), %d still open",
             settle["settled"], settle["still_open"])
    report["settle"] = settle

    unseen = unseen_matches(data_mod.load(), predictor)
    report["unseen_matches"] = unseen
    decision = should_retrain(store, predictor, unseen, force=args.force_retrain)
    log.info("retrain? %s - %s", "yes" if decision["retrain"] else "no", decision["reason"])
    report["decision"] = decision

    if decision["retrain"] and not args.no_retrain:
        _retrain(args)
        store.set_meta("last_retrain",
                       datetime.now(timezone.utc).isoformat(timespec="seconds"))
        report["retrained"] = True
    return report


def _retrain(args) -> None:
    from types import SimpleNamespace

    from .train import run as train_run

    log.info("retraining ...")
    train_run(SimpleNamespace(
        dataset="matches.csv", leagues=None, models=None, use_odds=False,
        calibrate="auto", test_fraction=0.2, cv_splits=args.cv_splits,
        min_history=5, importance_repeats=3, seed=42, out="model.joblib"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Refresh data, settle predictions, retrain.")
    ap.add_argument("--force-retrain", action="store_true")
    ap.add_argument("--no-retrain", action="store_true",
                    help="refresh and settle only")
    ap.add_argument("--cv-splits", type=int, default=5)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
