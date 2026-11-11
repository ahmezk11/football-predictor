"""Team rating systems, each updated one match at a time.

These are what turn "team strength" into something better than a rolling
average of results. Every one of them rates a team relative to the opposition
it actually faced, so beating the champions counts for more than beating the
side in last place - which a flat "goals in the last five" cannot see.

All share the same shape as the feature builder: `predict` is a pure read of
the pre-match state, `update` folds a finished match in. That keeps them usable
both for training and for scoring a fixture that has not been played.

  Elo           one number per club, updated by result and margin.
  PiRatings     separate home and away ratings in goal-difference units
                (Constantinou & Fenton, 2013) - clubs are rarely equally good
                at home and away, and a single number cannot say so.
  OnlinePoisson attack and defence ratings fitted online to a Poisson model of
                scoring, with a per-league scoring level and home advantage
                that drift over time. Gives expected goals for each side and,
                from those, home/draw/away probabilities that handle draws
                properly. Run on shots on target too, as a less noisy read of
                who is creating chances.
  LeagueContext exponentially weighted outcome and scoring rates per league,
                so the model can see home advantage shrink when it does.

Every system gives a club that is new to an established league a below-average
starting rating. Such a club is almost always newly promoted, and starting it at
the league mean - as the original Elo did - overrates it for weeks.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

ESTABLISHED_LEAGUE = 380   # matches seen before a newcomer is treated as promoted
MAX_GOALS = 10


# --- Elo ----------------------------------------------------------------------
@dataclass(frozen=True)
class EloParams:
    k: float = 20.0
    home_adv: float = 65.0
    season_regression: float = 0.25
    promoted_offset: float = 0.0
    start: float = 1500.0


class Elo:
    def __init__(self, params: EloParams = EloParams()):
        self.p = params
        self.r: dict[tuple[str, str], float] = {}
        self.season: dict[tuple[str, str], str] = {}
        self.league_seen: dict[str, int] = defaultdict(int)

    def _prior(self, league: str) -> float:
        if self.league_seen[league] >= ESTABLISHED_LEAGUE:
            return self.p.start - self.p.promoted_offset
        return self.p.start

    def rating(self, league: str, team: str) -> float:
        return self.r.get((league, team), self._prior(league))

    def expected_home(self, league: str, home: str, away: str) -> float:
        diff = self.rating(league, home) + self.p.home_adv - self.rating(league, away)
        return 1.0 / (1.0 + 10 ** (-diff / 400))

    def _roll(self, key, season: str) -> None:
        if key not in self.r:
            self.r[key] = self._prior(key[0])
        elif season and self.season.get(key) not in (None, season):
            self.r[key] += (self.p.start - self.r[key]) * self.p.season_regression
        if season:
            self.season[key] = season

    def update(self, league: str, home: str, away: str, hg: float, ag: float,
               season: str = "") -> None:
        hk, ak = (league, home), (league, away)
        self._roll(hk, season)
        self._roll(ak, season)
        exp = self.expected_home(league, home, away)
        score = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        delta = self.p.k * (1.0 + math.log1p(abs(hg - ag))) * (score - exp)
        self.r[hk] += delta
        self.r[ak] -= delta
        self.league_seen[league] += 1


# --- pi-ratings ---------------------------------------------------------------
@dataclass(frozen=True)
class PiParams:
    lr: float = 0.035          # lambda in the paper
    cross: float = 0.7         # gamma: how much a home result moves the away rating
    base: float = 3.0          # c in the paper
    promoted_offset: float = 0.0
    season_regression: float = 0.0


class PiRatings:
    def __init__(self, params: PiParams = PiParams()):
        self.p = params
        self.r: dict[tuple[str, str], list[float]] = {}   # [home rating, away rating]
        self.season: dict[tuple[str, str], str] = {}
        self.league_seen: dict[str, int] = defaultdict(int)

    def _prior(self, league: str) -> list[float]:
        v = -self.p.promoted_offset if self.league_seen[league] >= ESTABLISHED_LEAGUE else 0.0
        return [v, v]

    def ratings(self, league: str, team: str) -> list[float]:
        return self.r.get((league, team)) or self._prior(league)

    def _goals(self, rating: float) -> float:
        sign = 1.0 if rating >= 0 else -1.0
        return sign * (10 ** (abs(rating) / self.p.base) - 1.0)

    def expected_gd(self, league: str, home: str, away: str) -> float:
        return (self._goals(self.ratings(league, home)[0])
                - self._goals(self.ratings(league, away)[1]))

    def _roll(self, key, season: str) -> None:
        if key not in self.r:
            self.r[key] = self._prior(key[0])
        elif season and self.season.get(key) not in (None, season) and self.p.season_regression:
            self.r[key] = [v * (1 - self.p.season_regression) for v in self.r[key]]
        if season:
            self.season[key] = season

    def update(self, league: str, home: str, away: str, hg: float, ag: float,
               season: str = "") -> None:
        hk, ak = (league, home), (league, away)
        self._roll(hk, season)
        self._roll(ak, season)
        error = (hg - ag) - self.expected_gd(league, home, away)
        weighted = self.p.base * math.log10(1.0 + abs(error))
        psi_h = weighted if error > 0 else -weighted
        h, a = self.r[hk], self.r[ak]
        old_hh, old_aa = h[0], a[1]
        h[0] += psi_h * self.p.lr
        h[1] += (h[0] - old_hh) * self.p.cross
        a[1] += -psi_h * self.p.lr
        a[0] += (a[1] - old_aa) * self.p.cross
        self.league_seen[league] += 1


# --- online Poisson attack / defence -----------------------------------------
@dataclass(frozen=True)
class PoissonParams:
    lr: float = 0.03
    level_lr: float = 0.002        # per-league scoring level and home advantage
    season_regression: float = 0.2
    promoted_attack: float = 0.0   # newcomers score less ...
    promoted_defence: float = 0.0  # ... and concede more
    rho: float = 0.0               # Dixon-Coles low-score correlation (draws)
    start_level: float = math.log(1.35)
    start_home: float = 0.2


class OnlinePoisson:
    """log E[home goals] = level + home_adv + attack(home) - defence(away)."""

    def __init__(self, params: PoissonParams = PoissonParams()):
        self.p = params
        self.att: dict[tuple[str, str], float] = {}
        self.dfn: dict[tuple[str, str], float] = {}
        self.season: dict[tuple[str, str], str] = {}
        self.level: dict[str, float] = {}
        self.home: dict[str, float] = {}
        self.league_seen: dict[str, int] = defaultdict(int)

    def _prior(self, league: str) -> tuple[float, float]:
        if self.league_seen[league] >= ESTABLISHED_LEAGUE:
            return -self.p.promoted_attack, -self.p.promoted_defence
        return 0.0, 0.0

    def team(self, league: str, team: str) -> tuple[float, float]:
        key = (league, team)
        if key in self.att:
            return self.att[key], self.dfn[key]
        return self._prior(league)

    def rates(self, league: str, home: str, away: str) -> tuple[float, float]:
        lvl = self.level.get(league, self.p.start_level)
        hfa = self.home.get(league, self.p.start_home)
        ah, dh = self.team(league, home)
        aa, da = self.team(league, away)
        return math.exp(lvl + hfa + ah - da), math.exp(lvl + aa - dh)

    def outcome_probs(self, league: str, home: str, away: str) -> tuple[float, float, float]:
        lam_h, lam_a = self.rates(league, home, away)
        return poisson_outcomes(lam_h, lam_a, self.p.rho)

    def _roll(self, key, season: str) -> None:
        if key not in self.att:
            self.att[key], self.dfn[key] = self._prior(key[0])
        elif season and self.season.get(key) not in (None, season):
            keep = 1.0 - self.p.season_regression
            self.att[key] *= keep
            self.dfn[key] *= keep
        if season:
            self.season[key] = season

    def update(self, league: str, home: str, away: str, hg: float, ag: float,
               season: str = "") -> None:
        if hg is None or ag is None or math.isnan(hg) or math.isnan(ag):
            return
        hk, ak = (league, home), (league, away)
        self._roll(hk, season)
        self._roll(ak, season)
        lam_h, lam_a = self.rates(league, home, away)
        e_h, e_a = hg - lam_h, ag - lam_a
        lr = self.p.lr
        self.att[hk] += lr * e_h
        self.dfn[ak] -= lr * e_h
        self.att[ak] += lr * e_a
        self.dfn[hk] -= lr * e_a
        self.level[league] = self.level.get(league, self.p.start_level) + self.p.level_lr * (e_h + e_a)
        self.home[league] = self.home.get(league, self.p.start_home) + self.p.level_lr * (e_h - e_a)
        self.league_seen[league] += 1


def _pmf(lam: float) -> list[float]:
    out, p = [], math.exp(-lam)
    for k in range(MAX_GOALS + 1):
        out.append(p)
        p *= lam / (k + 1)
    return out


def score_matrix(lam_h: float, lam_a: float, rho: float = 0.0) -> list[list[float]]:
    """P(home scores i, away scores j) for every scoreline, normalised.

    This is the full distribution the model actually has. Collapsing it to three
    numbers throws away the exact score, both-teams-to-score and the goal line,
    all of which come free from the same matrix.

    With rho != 0 the four lowest scorelines get the Dixon-Coles adjustment,
    which corrects plain Poisson's tendency to under-predict 0-0 and 1-1.
    """
    ph, pa = _pmf(lam_h), _pmf(lam_a)
    grid = [[pi * pj for pj in pa] for pi in ph]
    if rho:
        grid[0][0] *= 1 - lam_h * lam_a * rho
        grid[0][1] *= 1 + lam_h * rho
        grid[1][0] *= 1 + lam_a * rho
        grid[1][1] *= 1 - rho
    total = sum(sum(row) for row in grid)
    return [[p / total for p in row] for row in grid]


def outcomes_from_matrix(grid) -> tuple[float, float, float]:
    home = sum(p for i, row in enumerate(grid) for j, p in enumerate(row) if i > j)
    draw = sum(row[i] for i, row in enumerate(grid) if i < len(row))
    away = sum(p for i, row in enumerate(grid) for j, p in enumerate(row) if i < j)
    return home, draw, away


def poisson_outcomes(lam_h: float, lam_a: float, rho: float = 0.0) -> tuple[float, float, float]:
    """P(home win), P(draw), P(away win) from two scoring rates."""
    return outcomes_from_matrix(score_matrix(lam_h, lam_a, rho))


def align_matrix(grid, home: float, draw: float, away: float) -> list[list[float]]:
    """Rescale a score matrix so its win/draw/loss shares match given probabilities.

    The app's headline probabilities come from the trained model, the scorelines
    from the goal-rate model. Left alone the two would contradict each other on
    screen - "62% home win" above a score distribution implying 55%. Scaling each
    outcome's block keeps every scoreline's relative shape while making the totals
    agree, so the detail is a decomposition of the headline rather than a rival to it.
    """
    current = outcomes_from_matrix(grid)
    factors = [(target / actual if actual > 1e-12 else 0.0)
               for target, actual in zip((home, draw, away), current)]
    out = []
    for i, row in enumerate(grid):
        scaled = []
        for j, p in enumerate(row):
            k = 0 if i > j else 1 if i == j else 2
            scaled.append(p * factors[k])
        out.append(scaled)
    total = sum(sum(row) for row in out)
    return [[p / total for p in row] for row in out]


def markets(grid, top: int = 5) -> dict:
    """The bets a score distribution answers directly."""
    flat = [(p, i, j) for i, row in enumerate(grid) for j, p in enumerate(row)]
    flat.sort(reverse=True)
    over = sum(p for p, i, j in flat if i + j > 2.5)
    btts = sum(p for p, i, j in flat if i > 0 and j > 0)
    return {
        "scorelines": [{"home": i, "away": j, "probability": round(p, 4)}
                       for p, i, j in flat[:top]],
        "over_2_5": round(over, 4),
        "under_2_5": round(1 - over, 4),
        "both_score": round(btts, 4),
        "clean_sheet_home": round(sum(p for p, i, j in flat if j == 0), 4),
        "clean_sheet_away": round(sum(p for p, i, j in flat if i == 0), 4),
    }


# --- league context -----------------------------------------------------------
class LeagueContext:
    """Exponentially weighted outcome mix and scoring rate for each league."""

    def __init__(self, halflife: float = 380.0):
        self.alpha = 1.0 - 0.5 ** (1.0 / halflife)
        self.stats: dict[str, dict[str, float]] = {}

    def get(self, league: str) -> dict[str, float]:
        return self.stats.get(league, {"home": 0.45, "draw": 0.26, "away": 0.29,
                                       "goals": 2.65})

    def update(self, league: str, hg: float, ag: float) -> None:
        s = dict(self.get(league))
        a = self.alpha
        for key, hit in (("home", hg > ag), ("draw", hg == ag), ("away", hg < ag)):
            s[key] += a * (float(hit) - s[key])
        s["goals"] += a * ((hg + ag) - s["goals"])
        self.stats[league] = s
