"""Upcoming fixtures, from two sources that cover different horizons.

  football-data  the next few days, with bookmaker odds attached. Uses the same
                 team names as the training data, so no resolution is needed -
                 but it thins out to nothing during international breaks.
  openfootball   the whole season's schedule, plus results for matches already
                 played. Names are the long official forms ("Arsenal FC"), so
                 they have to be resolved against the names the model knows.

The app wants both: openfootball to answer "when do my teams next play", and
football-data to enrich the near-term ones with market odds.
"""
from __future__ import annotations

import difflib
import io
import json
import logging
import re
import time
import unicodedata
from dataclasses import dataclass, replace
from datetime import date as date_cls
from datetime import datetime, timedelta, timezone

import pandas as pd

from .data import get_session

log = logging.getLogger(__name__)

FOOTBALL_DATA_FIXTURES = "https://www.football-data.co.uk/fixtures.csv"
OPENFOOTBALL_URL = ("https://raw.githubusercontent.com/openfootball/football.json/"
                    "master/{season}/{code}.json")

# League code (football-data convention) -> openfootball file code.
OPENFOOTBALL_CODES = {
    "E0": "en.1", "SP1": "es.1",
    "I1": "it.1", "D1": "de.1", "F1": "fr.1",
}

# Names that normalisation alone will not reconcile.
ALIASES = {
    "manchester city": "Man City",
    "manchester united": "Man United",
    "nottingham forest": "Nott'm Forest",
    "tottenham hotspur": "Tottenham",
    "wolverhampton wanderers": "Wolves",
    "brighton hove albion": "Brighton",
    "west bromwich albion": "West Brom",
    "sheffield united": "Sheffield United",
    "sheffield wednesday": "Sheffield Weds",
    "queens park rangers": "QPR",
    "newcastle united": "Newcastle",
    "leeds united": "Leeds",
    "leicester city": "Leicester",
    "norwich city": "Norwich",
    "ipswich town": "Ipswich",
    "hull city": "Hull",
    "stoke city": "Stoke",
    "swansea city": "Swansea",
    "cardiff city": "Cardiff",
    "coventry city": "Coventry",
    "birmingham city": "Birmingham",
    "derby county": "Derby",
    "luton town": "Luton",
    "burnley": "Burnley",
    "athletic club": "Ath Bilbao",
    "athletic bilbao": "Ath Bilbao",
    "atletico madrid": "Ath Madrid",
    "club atletico de madrid": "Ath Madrid",
    "real sociedad de futbol": "Sociedad",
    "real sociedad": "Sociedad",
    "real betis balompie": "Betis",
    "real betis": "Betis",
    "celta de vigo": "Celta",
    "rc celta de vigo": "Celta",
    "deportivo alaves": "Alaves",
    "rayo vallecano de madrid": "Vallecano",
    "rayo vallecano": "Vallecano",
    "real valladolid": "Valladolid",
    "villarreal": "Villarreal",
    "espanyol": "Espanol",
    "rcd espanyol de barcelona": "Espanol",
    "internazionale": "Inter",
    "fc internazionale milano": "Inter",
    "ac milan": "Milan",
    "as roma": "Roma",
    "ss lazio": "Lazio",
    "juventus": "Juventus",
    "napoli": "Napoli",
    "hellas verona": "Verona",
    "bayern munich": "Bayern Munich",
    "fc bayern munchen": "Bayern Munich",
    "borussia dortmund": "Dortmund",
    "borussia monchengladbach": "M'gladbach",
    "bayer 04 leverkusen": "Leverkusen",
    "bayer leverkusen": "Leverkusen",
    "eintracht frankfurt": "Ein Frankfurt",
    "rb leipzig": "RB Leipzig",
    "1 fc koln": "FC Koln",
    "fc koln": "FC Koln",
    "vfb stuttgart": "Stuttgart",
    "sv werder bremen": "Werder Bremen",
    "werder bremen": "Werder Bremen",
    "tsg hoffenheim": "Hoffenheim",
    "fc st pauli": "St Pauli",
    "paris saint germain": "Paris SG",
    "olympique de marseille": "Marseille",
    "olympique marseille": "Marseille",
    "olympique lyonnais": "Lyon",
    "as monaco": "Monaco",
    "losc lille": "Lille",
    "stade rennais": "Rennes",
    "ogc nice": "Nice",
    "rc lens": "Lens",
    "fc nantes": "Nantes",
    "stade brestois 29": "Brest",
    "racing club de strasbourg": "Strasbourg",
}

_ALIAS_EXTRA = {
    "bayern munchen": "Bayern Munich",
    "borussia monchengladbach": "M'gladbach",
    "koln": "FC Koln",
    "mainz": "Mainz",
    "eintracht braunschweig": "Braunschweig",
    "hertha bsc": "Hertha",
    "hamburger sv": "Hamburg",
    "vfl bochum": "Bochum",
    "arminia bielefeld": "Bielefeld",
    "fortuna dusseldorf": "Fortuna Dusseldorf",
    "greuther furth": "Greuther Furth",
    "holstein kiel": "Holstein Kiel",
    "union berlin": "Union Berlin",
    "st pauli": "St Pauli",
    "heidenheim": "Heidenheim",
    "elversberg": None,
}

# Decorations that carry no identity: club-type prefixes and noise words.
_NOISE = re.compile(
    r"\b(fc|afc|cf|sc|ac|as|ss|ssc|rc|rcd|cd|ud|sd|club|calcio|futbol|football|"
    r"sv|tsg|vfb|vfl|bsc|fsv|sgd|spvgg|de|la|el)\b",
    re.IGNORECASE)


def _strip_accents(s: str) -> str:
    """Munchen and München have to land on the same key."""
    return "".join(c for c in unicodedata.normalize("NFKD", s)
                   if not unicodedata.combining(c))


def normalise(name: str) -> str:
    """Strip club decorations so 'Brighton & Hove Albion FC' meets 'Brighton'."""
    s = _strip_accents(str(name)).lower().strip()
    s = s.replace("&", " ").replace(".", " ").replace("-", " ").replace("'", "")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    # Founding years and squad numbers ("Mainz 05", "Paderborn 07", "1. FC Koln")
    # are decoration too, but only as standalone tokens - never inside a name.
    s = re.sub(r"\b\d+\b", " ", s)
    s = _NOISE.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


_NORMALISED_ALIASES = {normalise(k): v for k, v in
                       {**ALIASES, **_ALIAS_EXTRA}.items() if v}


class TeamResolver:
    """Maps an external team name onto a name the trained model actually knows."""

    def __init__(self, known: dict[str, list[str]]):
        # league -> {normalised known name: original}
        self.by_league = {lg: {normalise(t): t for t in teams}
                          for lg, teams in known.items()}

    def resolve(self, league: str, name: str) -> str | None:
        table = self.by_league.get(league)
        if not table:
            return None
        key = normalise(name)
        if key in table:
            return table[key]

        alias = _NORMALISED_ALIASES.get(key)
        if alias and normalise(alias) in table:
            return table[normalise(alias)]

        # The known name is nearly always a subset of the official one:
        # "AJ Auxerre" -> Auxerre, "US Sassuolo Calcio" -> Sassuolo, "CA Osasuna"
        # -> Osasuna. This catches every club-type abbreviation at once instead
        # of enumerating them. Prefer the most specific match, and refuse to
        # guess when two candidates are equally specific.
        tokens = set(key.split())
        if tokens:
            hits = [(len(kn.split()), kn) for kn in table
                    if set(kn.split()) <= tokens]
            if hits:
                hits.sort(reverse=True)
                if len(hits) == 1 or hits[0][0] > hits[1][0]:
                    return table[hits[0][1]]

        # Last resort: closest known name, but only on a strong match, so a
        # promoted club the model has never seen stays unresolved instead of
        # being silently mapped onto some other team.
        close = difflib.get_close_matches(key, list(table), n=1, cutoff=0.86)
        return table[close[0]] if close else None


@dataclass
class Fixture:
    league: str
    date: str
    time: str | None
    home_team: str
    away_team: str
    odds: dict | None = None
    source: str = ""
    played: bool = False
    home_goals: int | None = None
    away_goals: int | None = None
    round: str | None = None      # "Matchday 7", when the source says so
    # False when a club is not in the model's vocabulary - usually newly
    # promoted, with no history to predict from. The fixture is still listed,
    # because a matchday with a match missing is a confusing thing to show.
    predictable: bool = True
    unknown_teams: tuple[str, ...] = ()

    @property
    def result(self) -> str | None:
        if not self.played or self.home_goals is None:
            return None
        return ("H" if self.home_goals > self.away_goals
                else "A" if self.home_goals < self.away_goals else "D")

    @property
    def round_number(self) -> int:
        """Sort key: 'Matchday 10' must come after 'Matchday 9', not before it."""
        if not self.round:
            return 10_000
        digits = "".join(c for c in self.round if c.isdigit())
        return int(digits) if digits else 10_000

    def as_dict(self) -> dict:
        d = {"league": self.league, "date": self.date, "time": self.time,
             "home_team": self.home_team, "away_team": self.away_team,
             "odds": self.odds, "source": self.source, "round": self.round,
             "predictable": self.predictable}
        if self.unknown_teams:
            d["unknown_teams"] = list(self.unknown_teams)
        if self.played:
            d.update({"played": True, "home_goals": self.home_goals,
                      "away_goals": self.away_goals, "result": self.result})
        return d


def _season_code(on: date_cls) -> str:
    """openfootball's directory name for the season containing `on`."""
    start = on.year if on.month >= 7 else on.year - 1
    return f"{start}-{(start + 1) % 100:02d}"


class FeedUnavailable(RuntimeError):
    """The schedule source could not be reached - different from having no fixtures."""


def from_openfootball(league: str, resolver: TeamResolver, *,
                      season: str | None = None, timeout: int = 30) -> list[Fixture]:
    """The whole season's schedule, with results where matches have been played."""
    code = OPENFOOTBALL_CODES.get(league)
    if not code:
        return []
    season = season or _season_code(datetime.now(timezone.utc).date())
    url = OPENFOOTBALL_URL.format(season=season, code=code)
    try:
        resp = get_session().get(url, timeout=timeout)
        if resp.status_code == 404:
            log.info("no openfootball schedule for %s %s", league, season)
            return []
        resp.raise_for_status()
        payload = json.loads(resp.content.decode("utf-8"))
    except Exception as exc:
        log.warning("openfootball unavailable for %s: %s", league, exc)
        raise FeedUnavailable(str(exc)) from exc

    out, unresolved = [], set()
    for m in payload.get("matches", []):
        raw_home, raw_away = str(m.get("team1", "")), str(m.get("team2", ""))
        home = resolver.resolve(league, raw_home)
        away = resolver.resolve(league, raw_away)
        unknown = tuple(raw for raw, got in ((raw_home, home), (raw_away, away)) if not got)
        unresolved.update(unknown)
        # Keep the fixture either way; just flag that we cannot score it.
        home, away = home or raw_home, away or raw_away
        score = m.get("score")
        # openfootball is inconsistent here: usually {"ft": [h, a]}, sometimes a
        # bare [h, a], and occasionally absent for unplayed matches.
        if isinstance(score, dict):
            ft = score.get("ft")
        elif isinstance(score, (list, tuple)) and len(score) >= 2:
            ft = list(score[:2])
        else:
            ft = None
        if not (isinstance(ft, (list, tuple)) and len(ft) >= 2
                and all(isinstance(v, int) for v in ft[:2])):
            ft = None
        out.append(Fixture(
            league=league, date=str(m.get("date")), time=m.get("time"),
            home_team=home, away_team=away, source="openfootball",
            played=bool(ft), home_goals=ft[0] if ft else None,
            away_goals=ft[1] if ft else None, round=m.get("round"),
            predictable=not unknown, unknown_teams=unknown,
        ))
    if unresolved:
        log.info("%s: %d team(s) not in the model's vocabulary: %s",
                 league, len(unresolved), ", ".join(sorted(map(str, unresolved))))
    return out


def from_football_data(leagues: set[str] | None = None,
                       timeout: int = 30) -> list[Fixture]:
    """The next few days, carrying bookmaker odds."""
    try:
        resp = get_session().get(FOOTBALL_DATA_FIXTURES, timeout=timeout)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.content.decode("latin-1")),
                         on_bad_lines="skip")
    except Exception as exc:
        log.warning("football-data fixtures unavailable: %s", exc)
        return []

    df.columns = [str(c).strip().lstrip("﻿") for c in df.columns]
    if "HomeTeam" not in df.columns:
        return []

    out = []
    for _, r in df.iterrows():
        div = str(r.get("Div", "")).strip()
        if leagues and div not in leagues:
            continue
        odds = None
        for h, d, a in (("AvgH", "AvgD", "AvgA"), ("B365H", "B365D", "B365A")):
            try:
                odds = {"home": float(r[h]), "draw": float(r[d]), "away": float(r[a])}
                break
            except (KeyError, TypeError, ValueError):
                continue
        try:
            when = pd.to_datetime(str(r.get("Date")), dayfirst=True).date().isoformat()
        except Exception:
            continue
        out.append(Fixture(
            league=div, date=when, time=str(r.get("Time") or "") or None,
            home_team=str(r.get("HomeTeam", "")).strip(),
            away_team=str(r.get("AwayTeam", "")).strip(),
            odds=odds, source="football-data",
        ))
    return out


class FixtureFeed:
    """Merges the two sources and caches the result for a few minutes."""

    def __init__(self, resolver: TeamResolver, cache_seconds: int = 900):
        self.resolver = resolver
        self.cache_seconds = cache_seconds
        self._cache: dict[str, tuple[float, list[Fixture]]] = {}
        # Why a league is empty matters: no schedule published is not the same
        # as no internet, and the app should not report the second as the first.
        self.errors: dict[str, str] = {}

    def _cached(self, key: str):
        hit = self._cache.get(key)
        if hit and (time.time() - hit[0]) < self.cache_seconds:
            return hit[1]
        return None

    def season(self, league: str, *, refresh: bool = False) -> list[Fixture]:
        """Every fixture in the current season for one league."""
        if not refresh:
            hit = self._cached(league)
            if hit is not None:
                return hit
        try:
            fixtures = from_openfootball(league, self.resolver)
        except FeedUnavailable as exc:
            self.errors[league] = str(exc)
            stale = self._cache.get(league)
            return stale[1] if stale else []      # last known schedule, if any
        self.errors.pop(league, None)
        self._cache[league] = (time.time(), fixtures)
        return fixtures

    def upcoming(self, leagues: list[str], *, days: int = 14,
                 teams: list[str] | None = None, limit: int = 60) -> list[Fixture]:
        """Unplayed fixtures in the next `days`, enriched with odds where known."""
        today = datetime.now(timezone.utc).date()
        horizon = today + timedelta(days=days)

        near = {}
        for fx in from_football_data(set(leagues)):
            near[(fx.league, fx.home_team, fx.away_team)] = fx

        wanted = set(teams) if teams else None
        out = []
        for league in leagues:
            for fx in self.season(league):
                if fx.played or not fx.predictable:
                    continue
                try:
                    when = date_cls.fromisoformat(fx.date)
                except ValueError:
                    continue
                if not (today <= when <= horizon):
                    continue
                if wanted and not (wanted & {fx.home_team, fx.away_team}):
                    continue
                match = near.get((fx.league, fx.home_team, fx.away_team))
                # Copy before attaching odds: these Fixture objects are the
                # cached season, and writing to them left short-lived odds stuck
                # on every later caller, including ones outside the window the
                # odds applied to.
                out.append(replace(fx, odds=match.odds) if match and match.odds else fx)

        # Anything the season schedule missed but the near-term feed has.
        seen = {(f.league, f.home_team, f.away_team) for f in out}
        for key, fx in near.items():
            if key not in seen and fx.league in leagues:
                if not wanted or (wanted & {fx.home_team, fx.away_team}):
                    out.append(fx)

        out.sort(key=lambda f: (f.date, f.time or "", f.home_team))
        return out[:limit]

    def matchdays(self, league: str, *, refresh: bool = False) -> list[dict]:
        """The season split into rounds, in playing order.

        Sources that carry no round label fall back to grouping by date, so the
        list still reads as "the matches on this day" rather than one flat run.
        """
        fixtures = self.season(league, refresh=refresh)
        groups: dict[tuple, list[Fixture]] = {}
        for fx in fixtures:
            key = ((fx.round_number, fx.round) if fx.round
                   else (10_000, f"Matches on {fx.date}"))
            groups.setdefault(key, []).append(fx)

        out = []
        for (number, label), matches in sorted(groups.items(),
                                               key=lambda kv: (kv[0][0], min(f.date for f in kv[1]))):
            matches.sort(key=lambda f: (f.date, f.time or "", f.home_team))
            out.append({
                "round": label,
                "number": number if number < 10_000 else None,
                "first_date": min(f.date for f in matches),
                "last_date": max(f.date for f in matches),
                "played": sum(f.played for f in matches),
                "total": len(matches),
                "matches": matches,
            })
        return out

    def results_since(self, leagues: list[str], since: str) -> list[Fixture]:
        """Played matches on or after `since` - used to settle open predictions."""
        out = []
        for league in leagues:
            out += [fx for fx in self.season(league)
                    if fx.played and fx.predictable and fx.date >= since]
        return out
