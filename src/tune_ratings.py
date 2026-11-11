"""Tune the rating systems' parameters on training data only.

Each system is scored by its own prediction of the matches it has not seen yet,
walking forward through time: Elo by the Brier score of its expected result,
pi-ratings by squared error in predicted goal difference, the Poisson model by
the ranked probability score of the outcome probabilities it implies.

Only matches before the training cutoff are used, and the first two seasons are
excluded from the score while ratings warm up. The holdout period the headline
numbers are reported on is never touched here.

    python -m src.tune_ratings            # writes reports/rating_params.json
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, replace

import pandas as pd

from . import data as data_mod
from .config import REPORTS
from .features import FeatureBuilder, build_features
from .ratings import Elo, EloParams, OnlinePoisson, PiParams, PiRatings, PoissonParams
from .train import chronological_split

log = logging.getLogger(__name__)
BURN_IN_UNTIL = pd.Timestamp("2007-07-01")


def training_matches(test_fraction: float = 0.2) -> tuple[list[dict], pd.Timestamp]:
    """Chronological matches strictly before the holdout starts."""
    matches = data_mod.load()
    feats, _ = build_features(matches, min_history=5)
    _, test = chronological_split(feats, test_fraction)
    cutoff = test.date.min()
    ordered = FeatureBuilder._chronological(matches)
    rows = ordered[ordered.date < cutoff].to_dict("records")
    return rows, cutoff


def _rps(p_home: float, p_draw: float, outcome: str) -> float:
    y_h, y_d = (1.0, 0.0) if outcome == "H" else (0.0, 1.0) if outcome == "D" else (0.0, 0.0)
    return ((p_home - y_h) ** 2 + (p_home + p_draw - y_h - y_d) ** 2) / 2


def score_elo(rows, params: EloParams) -> float:
    elo, total, n = Elo(params), 0.0, 0
    for r in rows:
        lg, h, a = r["league"], r["home_team"], r["away_team"]
        hg, ag = float(r["home_goals"]), float(r["away_goals"])
        if r["date"] >= BURN_IN_UNTIL:
            s = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
            total += (elo.expected_home(lg, h, a) - s) ** 2
            n += 1
        elo.update(lg, h, a, hg, ag, str(r.get("season", "")))
    return total / n


def score_pi(rows, params: PiParams) -> float:
    pi, total, n = PiRatings(params), 0.0, 0
    for r in rows:
        lg, h, a = r["league"], r["home_team"], r["away_team"]
        hg, ag = float(r["home_goals"]), float(r["away_goals"])
        if r["date"] >= BURN_IN_UNTIL:
            total += ((hg - ag) - pi.expected_gd(lg, h, a)) ** 2
            n += 1
        pi.update(lg, h, a, hg, ag, str(r.get("season", "")))
    return total / n


def score_poisson(rows, params: PoissonParams, *, goals=("home_goals", "away_goals")) -> float:
    model, total, n = OnlinePoisson(params), 0.0, 0
    for r in rows:
        lg, h, a = r["league"], r["home_team"], r["away_team"]
        hg, ag = float(r[goals[0]]), float(r[goals[1]])
        if r["date"] >= BURN_IN_UNTIL and goals[0] == "home_goals":
            ph, pd_, _ = model.outcome_probs(lg, h, a)
            total += _rps(ph, pd_, r["result"])
            n += 1
        model.update(lg, h, a, hg, ag, str(r.get("season", "")))
    return total / n


def coordinate_descent(name, score, start, grid: dict[str, list], rounds: int = 3):
    """Improve one parameter at a time, keeping any change that lowers the score."""
    best, best_score = start, score(start)
    log.info("%s start %.5f  %s", name, best_score, asdict(best))
    for rnd in range(rounds):
        improved = False
        for field_name, values in grid.items():
            for v in values:
                if getattr(best, field_name) == v:
                    continue
                cand = replace(best, **{field_name: v})
                s = score(cand)
                if s < best_score - 1e-7:
                    best, best_score, improved = cand, s, True
            log.info("  round %d  %-18s -> %-8s score %.5f", rnd + 1, field_name,
                     getattr(best, field_name), best_score)
        if not improved:
            break
    return best, best_score


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    t0 = time.time()
    rows, cutoff = training_matches()
    log.info("tuning on %d matches before %s (scored from %s)\n",
             len(rows), cutoff.date(), BURN_IN_UNTIL.date())

    elo_base = score_elo(rows, EloParams())
    elo, elo_s = coordinate_descent("elo", lambda p: score_elo(rows, p), EloParams(), {
        "k": [5, 7.5, 10, 12.5, 15, 20, 25, 30],
        "home_adv": [40, 55, 65, 75, 90],
        "season_regression": [0.0, 0.1, 0.2, 0.3, 0.4],
        "promoted_offset": [0, 25, 50, 75, 100, 150, 200],
    })

    pi_base = score_pi(rows, PiParams())
    pi, pi_s = coordinate_descent("pi", lambda p: score_pi(rows, p), PiParams(), {
        "lr": [0.02, 0.035, 0.05, 0.07, 0.1],
        "cross": [0.3, 0.5, 0.7, 0.9],
        "promoted_offset": [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8],
        "season_regression": [0.0, 0.1, 0.2, 0.3],
    })

    po_base = score_poisson(rows, PoissonParams())
    po, po_s = coordinate_descent("poisson", lambda p: score_poisson(rows, p), PoissonParams(), {
        "lr": [0.01, 0.015, 0.02, 0.025, 0.03, 0.05],
        "season_regression": [0.0, 0.1, 0.2, 0.35, 0.5],
        "promoted_attack": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "promoted_defence": [0.0, 0.05, 0.1, 0.2, 0.3],
        "level_lr": [0.0005, 0.001, 0.002, 0.004],
        "rho": [0.0, -0.025, -0.05, -0.075, -0.1, -0.15],
    })

    out = {
        "cutoff": str(cutoff.date()), "matches": len(rows),
        "elo": {"params": asdict(elo), "brier": elo_s, "default_brier": elo_base},
        "pi": {"params": asdict(pi), "sq_error": pi_s, "default_sq_error": pi_base},
        "poisson": {"params": asdict(po), "rps": po_s, "default_rps": po_base},
        "seconds": round(time.time() - t0, 1),
    }
    path = REPORTS / "rating_params.json"
    path.write_text(json.dumps(out, indent=2))
    log.info("\nelo     brier %.5f -> %.5f", elo_base, elo_s)
    log.info("pi      sq err %.4f -> %.4f", pi_base, pi_s)
    log.info("poisson rps   %.5f -> %.5f", po_base, po_s)
    log.info("-> %s (%.0fs)", path, time.time() - t0)


if __name__ == "__main__":
    main()
