"""Feature engineering.

Everything here is built by one stateful, strictly chronological pass. For each
match we emit the features *as they stood before kick-off*, then fold the result
into team state. That gives two things at once:

  * no target leakage in training - a row never sees its own result, nor any
    later match;
  * an identical code path for an unplayed fixture, so what the API scores is
    exactly what the model was trained on.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from functools import partial

import numpy as np
import pandas as pd

from .ratings import (
    Elo,
    EloParams,
    LeagueContext,
    OnlinePoisson,
    PiParams,
    PiRatings,
    PoissonParams,
    align_matrix,
    markets,
    score_matrix,
)

# Bump whenever the builder's pickled state or the feature set changes in a way
# older code cannot read. A saved model carries the version it was built with,
# and one from a different version is ignored rather than loaded: unpickling
# succeeds, but the first prediction then fails on a missing attribute.
#   1  rolling form and a single Elo rating
#   2  Elo, pi-ratings and online Poisson ratings (src/ratings.py)
FEATURE_VERSION = 2

# --- knobs ------------------------------------------------------------------
SHORT_WINDOW = 5      # "current form"
LONG_WINDOW = 10      # "underlying level"
VENUE_WINDOW = 5      # last N at this venue specifically
H2H_WINDOW = 6        # last N meetings between the two clubs

# Rating-system settings. The values used for training come from
# `python -m src.tune_ratings`, which fits them on pre-holdout matches only;
# TUNED_* below are filled in from reports/rating_params.json.
ELO_START = 1500.0
ORIGINAL_ELO = EloParams(k=20.0, home_adv=65.0, season_regression=0.25, promoted_offset=0.0)
# Fitted on 30,520 matches before 2022-04-02 (reports/rating_params.json).
# Notable: a lower K (steadier ratings), and every system now starts a newly
# promoted club below the league average instead of at it.
TUNED_ELO = EloParams(k=7.5, home_adv=65.0, season_regression=0.1, promoted_offset=75)
TUNED_PI = PiParams(lr=0.035, cross=0.5, promoted_offset=0.5, season_regression=0.0)
TUNED_POISSON = PoissonParams(lr=0.015, level_lr=0.001, season_regression=0.0,
                              promoted_attack=0.3, promoted_defence=0.05, rho=-0.075)
# Shots on target: same model, different scale (~4.5 per side, not ~1.35).
SOT_POISSON = PoissonParams(lr=0.02, level_lr=0.002, season_regression=0.2,
                            start_level=float(np.log(4.5)), start_home=0.15)

POINTS = {"W": 3.0, "D": 1.0, "L": 0.0}

BASE_FEATURES = [
    "elo_home", "elo_away", "elo_diff", "elo_expected_home",
    "ppg5_home", "ppg5_away", "ppg5_diff",
    "ppg10_home", "ppg10_away", "ppg10_diff",
    "gf5_home", "ga5_home", "gf5_away", "ga5_away",
    "gd5_home", "gd5_away", "gd5_diff",
    "venue_ppg_home", "venue_ppg_away", "venue_gd_home", "venue_gd_away",
    "sot5_home", "sot5_away", "sot5_diff",
    "shots5_home", "shots5_away",
    "sot_conceded5_home", "sot_conceded5_away",
    "corners5_home", "corners5_away",
    "cards5_home", "cards5_away",
    "attack_strength_home", "defence_strength_home",
    "attack_strength_away", "defence_strength_away",
    "h2h_played", "h2h_ppg_home", "h2h_gd_home",
    "rest_days_home", "rest_days_away", "rest_diff",
    "matches_played_home", "matches_played_away",
    "season_match_home", "season_match_away",
]
# Opponent-adjusted strength ratings and league conditions (src/ratings.py).
RATING_FEATURES = [
    "elo_abs_diff",
    "pi_home", "pi_away", "pi_expected_gd",
    "xg_home", "xg_away", "xg_diff",
    "pois_p_home", "pois_p_draw", "pois_p_away",
    "sot_rate_home", "sot_rate_away", "sot_rate_diff",
    "league_home_rate", "league_draw_rate", "league_goals",
]
# What the model is actually trained on. Chosen by `python -m src.experiments`:
# the rating systems already summarise most of what the form, shots and venue
# columns carry, and feeding all 63 columns to the model scored *worse* in
# cross-validation (RPS 0.2012) than these 11 (0.1988). The other columns are
# still computed - the app shows them as the reasoning behind a prediction.
MODEL_FEATURES = [
    "elo_diff", "elo_expected_home",
    "pi_expected_gd",
    "xg_home", "xg_away", "pois_p_home", "pois_p_draw", "pois_p_away",
    "sot_rate_diff",
    "league_home_rate", "league_draw_rate",
]
ODDS_FEATURES = ["p_market_home", "p_market_draw", "p_market_away", "market_overround"]
CATEGORICAL_FEATURES = ["league"]


# Module-level factories (not lambdas) so a fitted builder stays picklable.
_recent_deque = partial(deque, maxlen=LONG_WINDOW)
_venue_deque = partial(deque, maxlen=VENUE_WINDOW)
_h2h_deque = partial(deque, maxlen=H2H_WINDOW)
_league_goals_deque = partial(deque, maxlen=760)


def _mean(values) -> float:
    return float(np.mean(values)) if len(values) else np.nan


@dataclass
class TeamState:
    """Rolling history for a single club."""
    last_date: pd.Timestamp | None = None
    last_season: str | None = None
    played: int = 0
    season_played: int = 0
    recent: deque = field(default_factory=_recent_deque)
    home_recent: deque = field(default_factory=_venue_deque)
    away_recent: deque = field(default_factory=_venue_deque)

    def roll_season(self, season: str) -> None:
        """New campaign: reset the season counter."""
        if self.last_season is not None and season != self.last_season:
            self.season_played = 0
        self.last_season = season

    def window(self, n: int) -> list[dict]:
        return list(self.recent)[-n:]


class FeatureBuilder:
    """Chronological feature extractor and rolling team-state store."""

    def __init__(self, *, elo: EloParams | None = None, pi: PiParams | None = None,
                 poisson: PoissonParams | None = None,
                 sot: PoissonParams | None = None) -> None:
        self.elo = Elo(elo or TUNED_ELO)
        self.pi = PiRatings(pi or TUNED_PI)
        self.goals = OnlinePoisson(poisson or TUNED_POISSON)
        self.sot = OnlinePoisson(sot or SOT_POISSON)
        self.league_ctx = LeagueContext()
        self.teams: dict[tuple[str, str], TeamState] = {}
        self.h2h: dict[tuple[str, str, str], deque] = defaultdict(_h2h_deque)
        # League scoring baselines, used to turn raw goal rates into strengths.
        self.league_goals: dict[str, deque] = defaultdict(_league_goals_deque)

    # -- state access --------------------------------------------------------
    def state(self, league: str, team: str) -> TeamState:
        key = (league, team)
        if key not in self.teams:
            self.teams[key] = TeamState()
        return self.teams[key]

    def known_teams(self, league: str | None = None) -> list[str]:
        return sorted({t for (lg, t) in self.teams if league in (None, lg)})

    def leagues(self) -> list[str]:
        return sorted({lg for (lg, _) in self.teams})

    def _league_avg_goals(self, league: str) -> float:
        vals = self.league_goals[league]
        return float(np.mean(vals)) if len(vals) >= 20 else 1.35

    # -- feature extraction --------------------------------------------------
    def features(self, league: str, home: str, away: str, date: pd.Timestamp,
                 odds: dict | None = None) -> dict:
        """Pre-kickoff feature row. Pure read - never mutates state."""
        h, a = self.state(league, home), self.state(league, away)
        lg_avg = self._league_avg_goals(league)

        f: dict[str, float | str] = {"league": league}

        # Elo and its implied home win expectancy.
        f["elo_home"] = self.elo.rating(league, home)
        f["elo_away"] = self.elo.rating(league, away)
        f["elo_diff"] = f["elo_home"] - f["elo_away"]
        f["elo_abs_diff"] = abs(f["elo_diff"])
        f["elo_expected_home"] = self.elo.expected_home(league, home, away)

        # Separate home and away strength, in goal-difference units.
        f["pi_home"] = self.pi.ratings(league, home)[0]
        f["pi_away"] = self.pi.ratings(league, away)[1]
        f["pi_expected_gd"] = self.pi.expected_gd(league, home, away)

        # Expected goals from opponent-adjusted attack and defence, and the
        # outcome probabilities they imply.
        f["xg_home"], f["xg_away"] = self.goals.rates(league, home, away)
        f["xg_diff"] = f["xg_home"] - f["xg_away"]
        f["pois_p_home"], f["pois_p_draw"], f["pois_p_away"] = \
            self.goals.outcome_probs(league, home, away)

        # The same model on shots on target: who is actually creating chances.
        f["sot_rate_home"], f["sot_rate_away"] = self.sot.rates(league, home, away)
        f["sot_rate_diff"] = f["sot_rate_home"] - f["sot_rate_away"]

        ctx = self.league_ctx.get(league)
        f["league_home_rate"], f["league_draw_rate"] = ctx["home"], ctx["draw"]
        f["league_goals"] = ctx["goals"]

        for side, st in (("home", h), ("away", a)):
            short, long = st.window(SHORT_WINDOW), st.window(LONG_WINDOW)
            f[f"ppg5_{side}"] = _mean([m["points"] for m in short])
            f[f"ppg10_{side}"] = _mean([m["points"] for m in long])
            f[f"gf5_{side}"] = _mean([m["gf"] for m in short])
            f[f"ga5_{side}"] = _mean([m["ga"] for m in short])
            f[f"gd5_{side}"] = _mean([m["gf"] - m["ga"] for m in short])
            f[f"sot5_{side}"] = _mean([m["sot"] for m in short])
            f[f"shots5_{side}"] = _mean([m["shots"] for m in short])
            f[f"sot_conceded5_{side}"] = _mean([m["sot_against"] for m in short])
            f[f"corners5_{side}"] = _mean([m["corners"] for m in short])
            f[f"cards5_{side}"] = _mean([m["cards"] for m in short])
            # Scoring / conceding rate relative to the league, over a long window.
            f[f"attack_strength_{side}"] = _mean([m["gf"] for m in long]) / lg_avg
            f[f"defence_strength_{side}"] = _mean([m["ga"] for m in long]) / lg_avg
            f[f"matches_played_{side}"] = st.played
            f[f"season_match_{side}"] = st.season_played

        f["ppg5_diff"] = f["ppg5_home"] - f["ppg5_away"]
        f["ppg10_diff"] = f["ppg10_home"] - f["ppg10_away"]
        f["gd5_diff"] = f["gd5_home"] - f["gd5_away"]
        f["sot5_diff"] = f["sot5_home"] - f["sot5_away"]

        # Venue-specific form: how the home side goes at home, the away side away.
        f["venue_ppg_home"] = _mean([m["points"] for m in h.home_recent])
        f["venue_gd_home"] = _mean([m["gf"] - m["ga"] for m in h.home_recent])
        f["venue_ppg_away"] = _mean([m["points"] for m in a.away_recent])
        f["venue_gd_away"] = _mean([m["gf"] - m["ga"] for m in a.away_recent])

        # Head-to-head, always from the current home side's point of view.
        meetings = self.h2h[self._h2h_key(league, home, away)]
        f["h2h_played"] = len(meetings)
        if meetings:
            pts, gds = [], []
            for m in meetings:
                gf, ga = (m["hg"], m["ag"]) if m["home"] == home else (m["ag"], m["hg"])
                pts.append(POINTS["W"] if gf > ga else POINTS["D"] if gf == ga else POINTS["L"])
                gds.append(gf - ga)
            f["h2h_ppg_home"], f["h2h_gd_home"] = float(np.mean(pts)), float(np.mean(gds))
        else:
            f["h2h_ppg_home"] = f["h2h_gd_home"] = np.nan

        # Congestion.
        f["rest_days_home"] = (date - h.last_date).days if h.last_date is not None else np.nan
        f["rest_days_away"] = (date - a.last_date).days if a.last_date is not None else np.nan
        f["rest_diff"] = f["rest_days_home"] - f["rest_days_away"]
        for k in ("rest_days_home", "rest_days_away", "rest_diff"):
            if pd.notna(f[k]):
                f[k] = float(np.clip(f[k], 0, 30))

        f.update(self.market_features(odds))
        return f

    def score_forecast(self, league: str, home: str, away: str,
                       outcome: tuple[float, float, float] | None = None) -> dict:
        """The full scoreline distribution behind a fixture, and what it implies.

        Pure read, like features(). `outcome` rescales the distribution so its
        win/draw/loss shares match the headline probabilities the app shows;
        without it the two would disagree on screen.
        """
        lam_h, lam_a = self.goals.rates(league, home, away)
        grid = score_matrix(lam_h, lam_a, self.goals.p.rho)
        if outcome:
            grid = align_matrix(grid, *outcome)
        return {"expected_goals": {"home": round(lam_h, 2), "away": round(lam_a, 2)},
                **markets(grid)}

    @staticmethod
    def market_features(odds: dict | None) -> dict:
        """Decimal odds -> overround-free implied probabilities."""
        blank = {k: np.nan for k in ODDS_FEATURES}
        if not odds:
            return blank
        try:
            raw = [1.0 / float(odds[k]) for k in ("home", "draw", "away")]
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            return blank
        if not all(np.isfinite(raw)) or min(raw) <= 0:
            return blank
        total = sum(raw)
        return {
            "p_market_home": raw[0] / total,
            "p_market_draw": raw[1] / total,
            "p_market_away": raw[2] / total,
            "market_overround": total - 1.0,
        }

    @staticmethod
    def _h2h_key(league: str, a: str, b: str) -> tuple[str, str, str]:
        x, y = sorted((a, b))
        return (league, x, y)

    # -- state update --------------------------------------------------------
    def update(self, row) -> None:
        """Fold a completed match into team state. Call *after* features()."""
        league, home, away = row["league"], row["home_team"], row["away_team"]
        hg, ag = float(row["home_goals"]), float(row["away_goals"])
        date, season = row["date"], str(row.get("season", ""))

        h, a = self.state(league, home), self.state(league, away)
        h.roll_season(season)
        a.roll_season(season)

        self.elo.update(league, home, away, hg, ag, season)
        self.pi.update(league, home, away, hg, ag, season)
        self.goals.update(league, home, away, hg, ag, season)
        self.league_ctx.update(league, hg, ag)

        def num(key, default=np.nan):
            v = row.get(key, default)
            try:
                v = float(v)
            except (TypeError, ValueError):
                return np.nan
            return v if np.isfinite(v) else np.nan

        h_pts = POINTS["W"] if hg > ag else POINTS["D"] if hg == ag else POINTS["L"]
        a_pts = POINTS["W"] if ag > hg else POINTS["D"] if hg == ag else POINTS["L"]
        h_rec = {
            "points": h_pts, "gf": hg, "ga": ag,
            "shots": num("home_shots"), "sot": num("home_sot"),
            "sot_against": num("away_sot"), "corners": num("home_corners"),
            "cards": num("home_yellows") + 2 * num("home_reds"),
        }
        a_rec = {
            "points": a_pts, "gf": ag, "ga": hg,
            "shots": num("away_shots"), "sot": num("away_sot"),
            "sot_against": num("home_sot"), "corners": num("away_corners"),
            "cards": num("away_yellows") + 2 * num("away_reds"),
        }
        hs, as_ = num("home_sot"), num("away_sot")
        if not (np.isnan(hs) or np.isnan(as_)):
            self.sot.update(league, home, away, hs, as_, season)

        h.recent.append(h_rec)
        a.recent.append(a_rec)
        h.home_recent.append(h_rec)
        a.away_recent.append(a_rec)

        for st, d in ((h, date), (a, date)):
            st.last_date = d
            st.played += 1
            st.season_played += 1

        self.h2h[self._h2h_key(league, home, away)].append(
            {"home": home, "hg": hg, "ag": ag, "date": date})
        self.league_goals[league].extend([hg, ag])

    # -- bulk pass -----------------------------------------------------------
    @staticmethod
    def _chronological(matches: pd.DataFrame) -> pd.DataFrame:
        """Deterministic ordering.

        Several matches share a kick-off date, and pandas' default sort is not
        stable, so without an explicit tiebreak the order of a given day could
        change between runs - and with it every rolling feature computed after it.
        """
        keys = ["date"] + [c for c in ("match_id", "league", "home_team")
                           if c in matches.columns]
        return matches.sort_values(keys, kind="stable").reset_index(drop=True)

    def transform(self, matches: pd.DataFrame) -> pd.DataFrame:
        """One chronological pass: extract features, then absorb each result."""
        matches = self._chronological(matches)
        rows = []
        for row in matches.to_dict("records"):
            odds = {"home": row.get("odds_home"), "draw": row.get("odds_draw"),
                    "away": row.get("odds_away")}
            feats = self.features(row["league"], row["home_team"], row["away_team"],
                                  row["date"], odds)
            feats.update({
                "match_id": row.get("match_id"),
                "date": row["date"],
                "season": row.get("season"),
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "result": row["result"],
            })
            rows.append(feats)
            self.update(row)
        return pd.DataFrame(rows)


def build_features(matches: pd.DataFrame, *, min_history: int = 5,
                   builder: FeatureBuilder | None = None
                   ) -> tuple[pd.DataFrame, FeatureBuilder]:
    """Feature table plus the fitted state store (for scoring future fixtures).

    Rows where either club has played fewer than `min_history` matches are
    dropped: their form features are mostly empty and only add noise.
    """
    builder = builder or FeatureBuilder()
    feats = builder.transform(matches)
    warm = (feats.matches_played_home >= min_history) & (feats.matches_played_away >= min_history)
    return feats.loc[warm].reset_index(drop=True), builder


def feature_columns(use_odds: bool) -> list[str]:
    return MODEL_FEATURES + (ODDS_FEATURES if use_odds else []) + CATEGORICAL_FEATURES
