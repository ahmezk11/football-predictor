"""Entry point for the packaged desktop backend.

PyInstaller needs a real module to start from (not `uvicorn` on a command line),
and the packaged app must never bind anything but the loopback interface.
"""
from __future__ import annotations

import argparse
import multiprocessing

import uvicorn


def selftest() -> int:
    """Train, score and calibrate inside the packaged build.

    Serving a prediction only proves the model loads. The app can also retrain
    itself, which reaches much further into scikit-learn and scipy - exactly the
    parts a slimmed-down bundle might have dropped. This exercises that path, so
    trimming the bundle cannot quietly break the refresh.
    """
    import numpy as np
    import pandas as pd

    from src.evaluate import evaluate
    from src.features import build_features, feature_columns
    from src.model import build_model
    from src.ratings import poisson_outcomes

    rng = np.random.default_rng(0)
    teams = [f"T{i}" for i in range(8)]
    rows, day = [], pd.Timestamp("2024-08-01")
    for week in range(60):
        for i in range(0, len(teams) - 1, 2):
            home, away = teams[i], teams[(i + week) % len(teams)]
            if home == away:
                continue
            hg, ag = int(rng.poisson(1.6)), int(rng.poisson(1.1))
            rows.append({"league": "ST", "season": "2425", "date": day,
                         "home_team": home, "away_team": away,
                         "home_goals": hg, "away_goals": ag,
                         "home_sot": 5, "away_sot": 4,
                         "result": "H" if hg > ag else "A" if ag > hg else "D"})
        day += pd.Timedelta(days=7)
    matches = pd.DataFrame(rows)
    matches["match_id"] = range(len(matches))

    feats, builder = build_features(matches, min_history=3)
    cols = [c for c in feature_columns(False) if c in feats.columns]
    pipe = build_model("logistic", cols).fit(feats[cols], feats["result"])
    proba = pipe.predict_proba(feats[cols])
    metrics = evaluate(feats["result"], proba, list(pipe.named_steps["clf"].classes_))
    print(f"  trained on {len(feats)} rows, rps {metrics['rps']:.4f}")

    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import TimeSeriesSplit
    CalibratedClassifierCV(build_model("logistic", cols), method="sigmoid",
                           cv=TimeSeriesSplit(n_splits=3)).fit(feats[cols], feats["result"])
    print("  calibration ok")

    h, d, a = poisson_outcomes(1.7, 1.1, rho=-0.075)
    assert abs(h + d + a - 1) < 1e-9
    forecast = builder.score_forecast("ST", teams[0], teams[1])
    assert forecast["scorelines"], "no scorelines"
    print(f"  score distribution ok (most likely "
          f"{forecast['scorelines'][0]['home']}-{forecast['scorelines'][0]['away']})")
    print("selftest passed")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--selftest", action="store_true",
                    help="train a small model in-process and exit")
    args = ap.parse_args()

    if args.selftest:
        raise SystemExit(selftest())

    from src.api import app

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    multiprocessing.freeze_support()   # otherwise a frozen app can fork-bomb
    main()
