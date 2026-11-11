"""How does the model compare with the betting market - on our own matches?

The README quotes ~0.19 RPS for bookmaker closing odds from published work. That
is somebody else's number, on somebody else's matches. This scores the market and
the model on exactly the same fixtures, and then asks the more interesting
question: when the two disagree, who is right?

Needs a dataset carrying closing odds:

    make data-odds                       # football-data.co.uk, throttles hard
    python -m src.benchmark --dataset matches_odds.csv

The odds source blocks bursts of requests aggressively - hard enough to cut off
the live fixtures feed the app needs - so the download paces itself and may have
to be run more than once.
"""
from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd

from . import data as data_mod
from . import evaluate as ev
from .config import REPORTS, RESULT_CLASSES
from .features import build_features, feature_columns
from .model import build_model
from .train import chronological_split

log = logging.getLogger(__name__)

MARKET_COLS = ["p_market_away", "p_market_draw", "p_market_home"]   # class order A, D, H
CLASSES = ["A", "D", "H"]


def market_probabilities(df: pd.DataFrame) -> np.ndarray | None:
    """Overround-free implied probabilities, or None when the odds are missing."""
    if not set(MARKET_COLS).issubset(df.columns):
        return None
    values = df[MARKET_COLS].to_numpy(dtype=float)
    if np.isnan(values).any():
        return None
    return values / values.sum(axis=1, keepdims=True)


def paired_bootstrap(model_loss: np.ndarray, market_loss: np.ndarray,
                     rounds: int = 5000, seed: int = 0) -> dict:
    """Is the gap bigger than sampling noise? Paired, because it is the same matches."""
    diff = market_loss - model_loss           # positive means the model is better
    rng = np.random.default_rng(seed)
    draws = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(rounds)])
    return {"model_minus_market": float(diff.mean()),
            "ci_low": float(np.percentile(draws, 2.5)),
            "ci_high": float(np.percentile(draws, 97.5)),
            "model_better_share": float((draws > 0).mean())}


def per_match_rps(proba: np.ndarray, y) -> np.ndarray:
    """RPS for each match, so model and market can be compared pair by pair."""
    y = np.asarray(y)
    cum_p = np.cumsum(proba, axis=1)
    truth = np.zeros_like(proba)
    index = {c: i for i, c in enumerate(CLASSES)}
    for i, label in enumerate(y):
        truth[i, index[label]] = 1.0
    cum_y = np.cumsum(truth, axis=1)
    return np.sum((cum_p[:, :-1] - cum_y[:, :-1]) ** 2, axis=1) / (len(CLASSES) - 1)


def disagreements(df: pd.DataFrame, model: np.ndarray, market: np.ndarray) -> dict:
    """Where the model and the market pick different winners, who is right?"""
    model_pick = np.array(CLASSES)[model.argmax(axis=1)]
    market_pick = np.array(CLASSES)[market.argmax(axis=1)]
    differ = model_pick != market_pick
    truth = df["result"].to_numpy()

    out = {"matches": int(differ.sum()), "share": float(differ.mean())}
    if differ.any():
        out.update({
            "model_right": float((model_pick[differ] == truth[differ]).mean()),
            "market_right": float((market_pick[differ] == truth[differ]).mean()),
            "model_rps": float(per_match_rps(model[differ], truth[differ]).mean()),
            "market_rps": float(per_match_rps(market[differ], truth[differ]).mean()),
        })
        gap = np.abs(model.max(axis=1) - market.max(axis=1))
        order = np.argsort(-np.where(differ, gap, -1))[:10]
        out["biggest"] = [{
            "date": str(df["date"].iloc[i].date()), "league": df["league"].iloc[i],
            "match": f"{df['home_team'].iloc[i]} v {df['away_team'].iloc[i]}",
            "model": model_pick[i], "market": market_pick[i], "result": truth[i],
        } for i in order if differ[i]]
    return out


def run(args) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    matches = data_mod.load(args.dataset)
    coverage = matches.get("odds_home", pd.Series(dtype=float)).notna().mean()
    if coverage < 0.2:
        raise SystemExit(
            f"only {coverage:.1%} of {args.dataset} carries closing odds. Run "
            "`make data-odds` first - see the note in src/benchmark.py about the "
            "source throttling.")

    feats, _ = build_features(matches, min_history=5)
    train, test = chronological_split(feats, args.test_fraction)
    features = [c for c in feature_columns(False) if c in feats.columns]
    log.info("training to %s, comparing on %d held-out matches",
             train.date.max().date(), len(test))

    pipe = build_model("logistic", features).fit(train[features], train["result"])
    priced = test[test[MARKET_COLS].notna().all(axis=1)].reset_index(drop=True)
    if len(priced) < 200:
        raise SystemExit(f"only {len(priced)} held-out matches have odds")

    market = market_probabilities(priced)
    model = pipe.predict_proba(priced[features])
    assert list(pipe.named_steps["clf"].classes_) == CLASSES

    report = {
        "matches": len(priced),
        "period": [str(priced.date.min().date()), str(priced.date.max().date())],
        "model": ev.evaluate(priced["result"], model, CLASSES),
        "market": ev.evaluate(priced["result"], market, CLASSES),
        "significance": paired_bootstrap(per_match_rps(model, priced["result"]),
                                         per_match_rps(market, priced["result"])),
        "disagreements": disagreements(priced, model, market),
    }
    log.info("\n%-8s  acc %.3f  rps %.4f  logloss %.4f", "model",
             report["model"]["accuracy"], report["model"]["rps"], report["model"]["log_loss"])
    log.info("%-8s  acc %.3f  rps %.4f  logloss %.4f", "market",
             report["market"]["accuracy"], report["market"]["rps"], report["market"]["log_loss"])
    sig = report["significance"]
    log.info("\nmodel minus market RPS: %+.5f  (95%% %+.5f to %+.5f)",
             sig["model_minus_market"], sig["ci_low"], sig["ci_high"])
    d = report["disagreements"]
    if d.get("matches"):
        log.info("\nthey pick different winners in %d matches (%.1f%%): "
                 "model right %.1f%%, market right %.1f%%",
                 d["matches"], 100 * d["share"], 100 * d["model_right"], 100 * d["market_right"])

    path = REPORTS / "market_benchmark.json"
    path.write_text(json.dumps(report, indent=2, default=str))
    log.info("-> %s", path)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="matches_odds.csv")
    ap.add_argument("--test-fraction", type=float, default=0.2)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
