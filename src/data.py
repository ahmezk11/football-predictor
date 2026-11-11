"""Data collection: download historical match results and consolidate them.

Each row is one played match: full-time score, match statistics and - when the
source provides them - closing bookmaker odds. See config.SOURCES for the two
interchangeable providers.
"""
from __future__ import annotations

import argparse
import io
import logging
import time

import pandas as pd
import requests

from .config import (
    COLUMN_MAP,
    DATA_PROCESSED,
    DATA_RAW,
    DEFAULT_SEASONS,
    FOOTBALL_DATA_URL,
    LEAGUES,
    MIRROR_SLUGS,
    MIRROR_URL,
    season_codes,
)

log = logging.getLogger(__name__)

USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
              "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36")
REQUEST_DELAY = 0.4     # polite pause between downloads
MAX_RETRIES = 3
BACKOFF = 5.0           # seconds, doubled per retry

NUMERIC_COLS = [
    "home_goals", "away_goals",
    "home_shots", "away_shots", "home_sot", "away_sot",
    "home_corners", "away_corners", "home_fouls", "away_fouls",
    "home_yellows", "away_yellows", "home_reds", "away_reds",
    "odds_home", "odds_draw", "odds_away",
]

_session: requests.Session | None = None


def get_session() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update({"User-Agent": USER_AGENT, "Accept": "text/csv,*/*"})
    return _session


def _fetch(url: str, timeout: int) -> str | None:
    """GET with exponential backoff. Returns None if it never succeeds."""
    global _session
    delay = BACKOFF
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = get_session().get(url, timeout=timeout)
            if resp.status_code == 404:
                return None            # season simply does not exist
            resp.raise_for_status()
            return resp.content.decode("latin-1")
        except requests.RequestException as exc:
            if attempt == MAX_RETRIES:
                log.warning("giving up on %s: %s", url, exc)
                return None
            log.debug("retry %d in %.0fs (%s)", attempt, delay, exc)
            time.sleep(delay)
            delay *= 2
            _session = None            # force a fresh connection
    return None


def _url(league: str, season: str, source: str) -> str | None:
    if source == "mirror":
        slug = MIRROR_SLUGS.get(league)
        return MIRROR_URL.format(slug=slug, season=season) if slug else None
    return FOOTBALL_DATA_URL.format(season=season, league=league)


def _cache_path(league: str, season: str, source: str):
    return DATA_RAW / f"{source}_{league}_{season}.csv"


def download_season(league: str, season: str, *, source: str = "mirror",
                    refresh: bool = False, timeout: int = 30,
                    delay: float | None = None) -> pd.DataFrame | None:
    """Fetch one league-season CSV, caching it on disk."""
    url = _url(league, season, source)
    if url is None:
        return None

    path = _cache_path(league, season, source)
    if path.exists() and not refresh:
        text = path.read_text(encoding="latin-1")
    else:
        text = _fetch(url, timeout)
        if text is None:
            return None
        path.write_text(text, encoding="latin-1")
        time.sleep(REQUEST_DELAY if delay is None else delay)

    try:
        df = pd.read_csv(io.StringIO(text), encoding="latin-1", on_bad_lines="skip")
    except Exception as exc:
        log.warning("unparseable %s %s: %s", league, season, exc)
        return None

    if "HomeTeam" not in df.columns:
        return None
    df = df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns})
    df = df.loc[:, ~df.columns.duplicated()]
    df = df[[c for c in dict.fromkeys(COLUMN_MAP.values()) if c in df.columns]].copy()
    df["league"] = league
    df["season"] = season
    df["source"] = source
    return df


def _parse_dates(s: pd.Series) -> pd.Series:
    """Sources mix ISO (mirror) with dd/mm/yy and dd/mm/yyyy (football-data)."""
    s = s.astype(str).str.strip()
    out = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        out = out.fillna(pd.to_datetime(s, format=fmt, errors="coerce"))
    return out


def build_dataset(leagues: list[str] | None = None,
                  seasons: list[str] | None = None,
                  *, source: str = "mirror", refresh: bool = False,
                  delay: float | None = None) -> pd.DataFrame:
    """Download every league-season and return one tidy, chronological frame."""
    leagues = leagues or list(LEAGUES)
    seasons = seasons or DEFAULT_SEASONS

    frames = []
    for league in leagues:
        got = 0
        for season in seasons:
            df = download_season(league, season, source=source, refresh=refresh, delay=delay)
            if df is not None and len(df):
                frames.append(df)
                got += len(df)
        log.info("%-4s %-28s %6d matches", league, LEAGUES.get(league, ""), got)
    if not frames:
        raise RuntimeError("no data downloaded - check network access")

    data = pd.concat(frames, ignore_index=True)
    data["date"] = _parse_dates(data["date"])

    for col in NUMERIC_COLS:
        data[col] = pd.to_numeric(data[col], errors="coerce") if col in data.columns else pd.NA

    # A row is usable only if we know who played and how it finished.
    data = data.dropna(subset=["date", "home_team", "away_team", "home_goals", "away_goals"])
    data["home_team"] = data["home_team"].astype(str).str.strip()
    data["away_team"] = data["away_team"].astype(str).str.strip()
    data = data[data.home_team.ne("") & data.away_team.ne("")]

    # Derive the label from the score rather than trusting the FTR column.
    data["result"] = "D"
    data.loc[data.home_goals > data.away_goals, "result"] = "H"
    data.loc[data.home_goals < data.away_goals, "result"] = "A"

    # If both sources were cached, keep one row per fixture (prefer the one with odds).
    data["_has_odds"] = data.get("odds_home", pd.Series(index=data.index)).notna()
    data = (data.sort_values("_has_odds", ascending=False)
                .drop_duplicates(subset=["league", "date", "home_team", "away_team"])
                .drop(columns="_has_odds"))

    data = data.sort_values(["date", "league", "home_team"]).reset_index(drop=True)
    data["match_id"] = data.index
    return data


def save(data: pd.DataFrame, name: str = "matches.csv") -> str:
    out = DATA_PROCESSED / name
    data.to_csv(out, index=False)
    return str(out)


def load(name: str = "matches.csv") -> pd.DataFrame:
    path = DATA_PROCESSED / name
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `python -m src.data` first")
    return pd.read_csv(path, parse_dates=["date"])


def main() -> None:
    ap = argparse.ArgumentParser(description="Download historical match data.")
    ap.add_argument("--source", default="mirror", choices=["mirror", "football-data", "both"])
    ap.add_argument("--leagues", nargs="*", default=None, choices=list(LEAGUES))
    ap.add_argument("--from-year", type=int, default=2005)
    ap.add_argument("--to-year", type=int, default=2024)
    ap.add_argument("--refresh", action="store_true", help="re-download cached files")
    ap.add_argument("--delay", type=float, default=None,
                    help="seconds between downloads; raise it for football-data.co.uk, "
                         "which throttles hard enough to cut off the live fixtures feed")
    ap.add_argument("--out", default="matches.csv")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    seasons = season_codes(args.from_year, args.to_year)
    sources = ["mirror", "football-data"] if args.source == "both" else [args.source]

    frames = []
    for src in sources:
        log.info("\nsource: %s", src)
        try:
            frames.append(build_dataset(args.leagues, seasons, source=src,
                                        refresh=args.refresh, delay=args.delay))
        except RuntimeError as exc:
            log.warning("  %s", exc)
    if not frames:
        raise SystemExit("no data available from any source")

    data = pd.concat(frames, ignore_index=True)
    data["_has_odds"] = data["odds_home"].notna()
    data = (data.sort_values("_has_odds", ascending=False)
                .drop_duplicates(subset=["league", "date", "home_team", "away_team"])
                .drop(columns="_has_odds")
                .sort_values(["date", "league", "home_team"]).reset_index(drop=True))
    data["match_id"] = data.index

    path = save(data, args.out)
    print(f"\n{len(data):,} matches -> {path}")
    print(f"date range : {data.date.min():%Y-%m-%d} .. {data.date.max():%Y-%m-%d}")
    print(f"leagues    : {', '.join(sorted(data.league.unique()))}")
    print(f"teams      : {len(set(data.home_team) | set(data.away_team))}")
    print("outcome mix:", data.result.value_counts(normalize=True).round(3).to_dict())
    print(f"shot stats : {data.home_sot.notna().mean():.1%} of rows")
    print(f"odds       : {data.odds_home.notna().mean():.1%} of rows")


if __name__ == "__main__":
    main()
