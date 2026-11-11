"""Shared fixtures: a synthetic league big enough to train a real model on."""
from __future__ import annotations

import os
import tempfile

# Set before anything imports src: config resolves its paths at import time, and
# starting the API runs a lifespan that would otherwise touch the developer's
# real database and go to the network for a staleness refresh.
_TMP = tempfile.mkdtemp(prefix="predictor-tests-")
os.environ.setdefault("PREDICTOR_DATA_DIR", _TMP)
os.environ["PREDICTOR_AUTO_REFRESH_DAYS"] = "0"

import numpy as np  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from src import api as api_mod  # noqa: E402


def synthetic_league(n_teams: int = 12, n_seasons: int = 6, seed: int = 7) -> pd.DataFrame:
    """A believable league where team strength drives results, plus home advantage.

    Real data is slow to fetch and changes over time; this gives the tests a
    stable dataset with genuine signal, so a model trained on it should beat a
    prior-only baseline.
    """
    rng = np.random.default_rng(seed)
    teams = [f"Team {chr(65 + i)}" for i in range(n_teams)]
    strength = dict(zip(teams, rng.normal(0, 0.35, n_teams)))

    rows, day = [], pd.Timestamp("2015-08-08")
    for s in range(n_seasons):
        season = f"{15 + s:02d}{16 + s:02d}"
        for team in teams:                       # squads drift between seasons
            strength[team] += rng.normal(0, 0.12)
        fixtures = [(h, a) for h in teams for a in teams if h != a]
        rng.shuffle(fixtures)
        for i, (home, away) in enumerate(fixtures):
            lam_h = np.exp(0.30 + strength[home] - strength[away] * 0.9)
            lam_a = np.exp(0.05 + strength[away] - strength[home] * 0.9)
            hg, ag = rng.poisson(lam_h), rng.poisson(lam_a)
            rows.append({
                "date": day + pd.Timedelta(days=3 * (i // (n_teams // 2))),
                "league": "SYN", "season": season,
                "home_team": home, "away_team": away,
                "home_goals": int(hg), "away_goals": int(ag),
                "home_shots": int(rng.poisson(6 * lam_h)), "away_shots": int(rng.poisson(6 * lam_a)),
                "home_sot": int(rng.poisson(3 * lam_h)), "away_sot": int(rng.poisson(3 * lam_a)),
                "home_corners": int(rng.poisson(5)), "away_corners": int(rng.poisson(4)),
                "home_fouls": int(rng.poisson(11)), "away_fouls": int(rng.poisson(12)),
                "home_yellows": int(rng.poisson(1.6)), "away_yellows": int(rng.poisson(1.9)),
                "home_reds": int(rng.random() < 0.04), "away_reds": int(rng.random() < 0.05),
                "odds_home": np.nan, "odds_draw": np.nan, "odds_away": np.nan,
            })
        day += pd.Timedelta(days=330)

    df = pd.DataFrame(rows)
    df["result"] = np.where(df.home_goals > df.away_goals, "H",
                            np.where(df.home_goals < df.away_goals, "A", "D"))
    df = df.sort_values(["date", "home_team"], kind="stable").reset_index(drop=True)
    df["match_id"] = range(len(df))
    return df


@pytest.fixture(scope="session")
def matches() -> pd.DataFrame:
    return synthetic_league()


@pytest.fixture(scope="session")
def predictor(matches):
    """A real model trained on the synthetic league, shared across test modules."""
    from src.features import build_features, feature_columns
    from src.predict import Predictor
    from src.train import train_final

    feats, builder = build_features(matches, min_history=5)
    features = [c for c in feature_columns(False) if c in feats.columns]
    pipe = train_final(feats, "logistic", features, calibrate=False)
    return Predictor({
        "pipeline": pipe, "builder": builder, "features": features,
        "classes": list(pipe.named_steps["clf"].classes_),
        "model_name": "logistic", "use_odds": False,
        "trained_at": "2025-01-01T00:00:00+00:00", "train_rows": len(feats),
        "data_range": ["2015-08-08", "2021-05-01"], "leagues": ["SYN"],
        "holdout_metrics": {"accuracy": 0.5, "rps": 0.2},
    })


@pytest.fixture(scope="module")
def client(predictor):
    # A loopback base URL: the API refuses requests that do not come from the
    # local machine, and TestClient's default host ("testserver") is not one.
    with TestClient(api_mod.app, base_url="http://127.0.0.1:8000") as c:
        api_mod.state["predictor"] = predictor   # replace the disk-loaded model
        yield c
