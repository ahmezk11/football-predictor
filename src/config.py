"""Central configuration: data sources, paths, and feature groups."""
from __future__ import annotations

import os
import sys
from pathlib import Path

FROZEN = getattr(sys, "frozen", False)

# The source tree, always knowable and never redirected by the environment.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if FROZEN:
    # In a packaged desktop build the code lives in a read-only bundle, so
    # read-only resources (the trained model, the web front-end) sit beside the
    # executable while everything writable goes to the user's data directory.
    BUNDLE = Path(sys.executable).resolve().parent
    ROOT = Path(os.environ.get("PREDICTOR_DATA_DIR", BUNDLE))
else:
    BUNDLE = ROOT = PROJECT_ROOT

_writable = Path(os.environ.get("PREDICTOR_DATA_DIR", ROOT))

DATA_RAW = _writable / "data" / "raw"
DATA_PROCESSED = _writable / "data" / "processed"
REPORTS = _writable / "reports"

# A retrained model must be able to replace the shipped one, so the writable
# copy wins - but if it is not there we fall back to wherever the model actually
# lives. The packaged layout puts the binary in Resources/backend/ and the model
# in Resources/models/, hence "beside the executable" and "one level up"; the
# project tree covers a dev run whose data directory has been redirected.
MODELS = Path(os.environ.get("PREDICTOR_MODEL_DIR", _writable / "models"))
_FALLBACK_MODEL_DIRS = (BUNDLE / "models", BUNDLE.parent / "models",
                        PROJECT_ROOT / "models")

for _d in (DATA_RAW, DATA_PROCESSED, MODELS, REPORTS):
    try:
        _d.mkdir(parents=True, exist_ok=True)
    except OSError:  # read-only location inside a bundle
        pass


def resolve_model(name: str, primary: Path, fallbacks) -> Path:
    """First existing copy of `name`, else `primary` so callers report that path.

    Deliberately independent of whether the build is frozen: setting
    PREDICTOR_DATA_DIR must never be able to hide a model that is sitting right
    there in the project, which is exactly what it used to do when the desktop
    shell redirected the data directory of a dev run.
    """
    local = primary / name
    if local.exists():
        return local
    return next((d / name for d in fallbacks if (d / name).exists()), local)


def model_path(name: str = "model.joblib") -> Path:
    """Where a newly trained model is written, or the first one found."""
    return resolve_model(name, MODELS, _FALLBACK_MODEL_DIRS)


def model_candidates(name: str = "model.joblib") -> list[Path]:
    """Every existing copy of the model: a local retrain and the one shipped.

    Which to use is decided by Predictor.load on what is inside each file - the
    location alone cannot tell a fresh retrain from a stale leftover.
    """
    seen, out = set(), []
    for d in (MODELS, *_FALLBACK_MODEL_DIRS):
        path = (d / name)
        key = path.resolve() if path.exists() else None
        if key and key not in seen:
            seen.add(key)
            out.append(path)
    return out

# Division codes follow the football-data.co.uk convention throughout. Only
# leagues the default source can actually fetch belong here: the Championship
# was listed without a MIRROR_SLUGS entry, so it silently returned no matches.
LEAGUES: dict[str, str] = {
    "E0": "England - Premier League",
    "SP1": "Spain - La Liga",
    "I1": "Italy - Serie A",
    "D1": "Germany - Bundesliga",
    "F1": "France - Ligue 1",
}

# Two interchangeable sources for the same underlying data:
#
#   mirror        - a GitHub mirror of the football-data.co.uk CSVs. Reliable and
#                   rate-limit friendly, carries every match statistic, but the
#                   bookmaker odds columns are stripped.
#   football-data - the original site. Adds closing odds (which is what lets us
#                   benchmark against the market), but throttles hard.
#
# The mirror is the default; pass --source football-data to pick up odds.
MIRROR_URL = ("https://raw.githubusercontent.com/datasets/football-datasets/"
              "main/datasets/{slug}/season-{season}.csv")
FOOTBALL_DATA_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"

MIRROR_SLUGS = {
    "E0": "premier-league",
    "SP1": "la-liga",
    "I1": "serie-a",
    "D1": "bundesliga",
    "F1": "ligue-1",
}
SOURCES = ("mirror", "football-data")


def season_codes(start_year: int, end_year: int) -> list[str]:
    """Season codes in football-data's "%y%y" form: 1516 == the 2015/16 season."""
    return [f"{y % 100:02d}{(y + 1) % 100:02d}" for y in range(start_year, end_year + 1)]


DEFAULT_SEASONS = season_codes(2005, 2024)

# Raw columns we keep, mapped to tidy names.
COLUMN_MAP = {
    "Date": "date",
    "Time": "time",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "FTR": "result",
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_sot",
    "AST": "away_sot",
    "HC": "home_corners",
    "AC": "away_corners",
    "HF": "home_fouls",
    "AF": "away_fouls",
    "HY": "home_yellows",
    "AY": "away_yellows",
    "HR": "home_reds",
    "AR": "away_reds",
    # Market consensus closing odds (average across bookmakers).
    "AvgH": "odds_home",
    "AvgD": "odds_draw",
    "AvgA": "odds_away",
    "BbAvH": "odds_home",  # older seasons used the Betbrain prefix
    "BbAvD": "odds_draw",
    "BbAvA": "odds_away",
}

RESULT_CLASSES = ["H", "D", "A"]  # home win, draw, away win
CLASS_LABELS = {"H": "Home win", "D": "Draw", "A": "Away win"}
