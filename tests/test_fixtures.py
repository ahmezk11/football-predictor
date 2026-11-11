"""Team-name resolution: the join between external feeds and the model."""
from __future__ import annotations

import pytest

from src.fixtures import Fixture, TeamResolver, normalise

KNOWN = {
    "E0": ["Arsenal", "Man City", "Man United", "Nott'm Forest", "Brighton",
           "Tottenham", "Wolves", "Sheffield United"],
    "SP1": ["Real Madrid", "Ath Madrid", "Ath Bilbao", "Barcelona", "Osasuna",
            "Sociedad", "Betis", "Vallecano"],
    "I1": ["Milan", "Inter", "Roma", "Lazio", "Fiorentina", "Sassuolo", "Atalanta"],
    "D1": ["Bayern Munich", "Dortmund", "M'gladbach", "FC Koln", "Mainz", "Leverkusen"],
    "F1": ["Paris SG", "Marseille", "Lyon", "Lens", "Lille", "Auxerre", "Strasbourg"],
}


@pytest.fixture(scope="module")
def resolver():
    return TeamResolver(KNOWN)


def test_normalise_strips_club_decoration():
    assert normalise("Brighton & Hove Albion FC") == "brighton hove albion"
    assert normalise("1. FC Köln") == "koln"
    assert normalise("SC Paderborn 07") == "paderborn"


def test_normalise_folds_accents():
    assert normalise("FC Bayern München") == normalise("FC Bayern Munchen")


@pytest.mark.parametrize("league,external,expected", [
    ("E0", "Arsenal FC", "Arsenal"),
    ("E0", "Manchester City FC", "Man City"),
    ("E0", "Manchester United FC", "Man United"),
    ("E0", "Nottingham Forest FC", "Nott'm Forest"),
    ("E0", "Tottenham Hotspur FC", "Tottenham"),
    ("E0", "Wolverhampton Wanderers FC", "Wolves"),
    ("SP1", "Real Madrid CF", "Real Madrid"),
    ("SP1", "Club Atletico de Madrid", "Ath Madrid"),
    ("SP1", "Athletic Club", "Ath Bilbao"),
    ("SP1", "CA Osasuna", "Osasuna"),
    ("SP1", "Rayo Vallecano de Madrid", "Vallecano"),
    ("I1", "AC Milan", "Milan"),
    ("I1", "FC Internazionale Milano", "Inter"),
    ("I1", "US Sassuolo Calcio", "Sassuolo"),
    ("I1", "Atalanta BC", "Atalanta"),
    ("D1", "FC Bayern München", "Bayern Munich"),
    ("D1", "Borussia Mönchengladbach", "M'gladbach"),
    ("D1", "1. FC Köln", "FC Koln"),
    ("D1", "1. FSV Mainz 05", "Mainz"),
    ("F1", "Paris Saint-Germain FC", "Paris SG"),
    ("F1", "AJ Auxerre", "Auxerre"),
    ("F1", "RC Strasbourg Alsace", "Strasbourg"),
    ("F1", "Racing Club de Lens", "Lens"),
])
def test_known_clubs_resolve(resolver, league, external, expected):
    assert resolver.resolve(league, external) == expected


def test_unknown_club_stays_unresolved(resolver):
    """A promoted side the model has never seen must not be silently remapped."""
    assert resolver.resolve("E0", "Coventry City FC") is None
    assert resolver.resolve("D1", "SV 07 Elversberg") is None


def test_the_two_madrid_clubs_do_not_collide(resolver):
    assert resolver.resolve("SP1", "Real Madrid CF") == "Real Madrid"
    assert resolver.resolve("SP1", "Atletico Madrid") == "Ath Madrid"


def test_the_two_manchester_clubs_do_not_collide(resolver):
    assert resolver.resolve("E0", "Manchester City") == "Man City"
    assert resolver.resolve("E0", "Manchester United") == "Man United"


def test_resolution_is_scoped_to_the_league(resolver):
    assert resolver.resolve("SP1", "Arsenal FC") is None


def test_fixture_result_derives_from_the_score():
    assert Fixture("E0", "2026-09-06", None, "A", "B",
                   played=True, home_goals=2, away_goals=0).result == "H"
    assert Fixture("E0", "2026-09-06", None, "A", "B",
                   played=True, home_goals=1, away_goals=1).result == "D"
    assert Fixture("E0", "2026-09-06", None, "A", "B",
                   played=True, home_goals=0, away_goals=3).result == "A"
    assert Fixture("E0", "2026-09-06", None, "A", "B").result is None


# --- matchday grouping -------------------------------------------------------
def _fx(home, away, date, rnd=None, played=False, hg=None, ag=None, predictable=True):
    return Fixture("E0", date, None, home, away, played=played, home_goals=hg,
                   away_goals=ag, round=rnd, predictable=predictable)


def test_round_number_sorts_numerically():
    """'Matchday 10' must follow 'Matchday 9', which string sorting gets wrong."""
    assert _fx("a", "b", "2026-01-01", "Matchday 9").round_number == 9
    assert _fx("a", "b", "2026-01-01", "Matchday 10").round_number == 10
    assert (_fx("a", "b", "2026-01-01", "Matchday 9").round_number
            < _fx("a", "b", "2026-01-01", "Matchday 10").round_number)


def test_round_number_of_an_unlabelled_fixture_sorts_last():
    assert _fx("a", "b", "2026-01-01").round_number == 10_000


def _feed_with(fixtures, resolver):
    from src.fixtures import FixtureFeed

    feed = FixtureFeed(resolver)
    feed._cache["E0"] = (float("inf"), fixtures)   # never expires during the test
    return feed


def test_matchdays_group_and_order_by_round(resolver):
    fixtures = [
        _fx("Arsenal", "Man City", "2026-09-19", "Matchday 10"),
        _fx("Wolves", "Brighton", "2026-08-15", "Matchday 9"),
        _fx("Tottenham", "Man United", "2026-08-16", "Matchday 9"),
    ]
    days = _feed_with(fixtures, resolver).matchdays("E0")
    assert [d["round"] for d in days] == ["Matchday 9", "Matchday 10"]
    assert days[0]["total"] == 2
    assert days[0]["first_date"] == "2026-08-15"
    assert days[0]["last_date"] == "2026-08-16"


def test_matchdays_report_how_many_have_been_played(resolver):
    fixtures = [
        _fx("Arsenal", "Man City", "2026-09-19", "Matchday 1", played=True, hg=2, ag=1),
        _fx("Wolves", "Brighton", "2026-09-19", "Matchday 1"),
    ]
    day = _feed_with(fixtures, resolver).matchdays("E0")[0]
    assert (day["played"], day["total"]) == (1, 2)


def test_matchdays_fall_back_to_dates_without_round_labels(resolver):
    fixtures = [_fx("Arsenal", "Man City", "2026-09-19"),
                _fx("Wolves", "Brighton", "2026-09-26")]
    days = _feed_with(fixtures, resolver).matchdays("E0")
    assert len(days) == 2
    assert all(d["number"] is None for d in days)
    assert "2026-09-19" in days[0]["round"]


def test_unpredictable_fixtures_stay_in_the_list(resolver):
    """A matchday with a match quietly missing is worse than one flagged."""
    fixtures = [_fx("Arsenal", "Coventry City FC", "2026-09-19", "Matchday 1",
                    predictable=False)]
    day = _feed_with(fixtures, resolver).matchdays("E0")[0]
    assert day["total"] == 1
    assert day["matches"][0].predictable is False


def test_unpredictable_fixtures_are_excluded_from_scoring_paths(resolver):
    """They must never reach the model or the settling logic."""
    fixtures = [
        _fx("Arsenal", "Coventry City FC", "2026-09-19", "Matchday 1", predictable=False),
        _fx("Wolves", "Brighton", "2026-09-19", "Matchday 1", played=True, hg=1, ag=0),
        _fx("Man City", "Coventry City FC", "2026-09-19", "Matchday 1",
            played=True, hg=3, ag=0, predictable=False),
    ]
    feed = _feed_with(fixtures, resolver)
    assert [f.home_team for f in feed.results_since(["E0"], "2026-01-01")] == ["Wolves"]


def test_as_dict_exposes_the_round_and_predictability():
    d = _fx("Arsenal", "Chelsea", "2026-09-19", "Matchday 4").as_dict()
    assert d["round"] == "Matchday 4" and d["predictable"] is True


def test_enriching_with_odds_leaves_the_cached_season_untouched(resolver):
    """Odds from the near-term feed are valid for days, the cache lives for minutes."""
    from datetime import date, timedelta

    from src.fixtures import FixtureFeed

    soon = (date.today() + timedelta(days=2)).isoformat()
    season = [Fixture("E0", soon, "15:00", "Arsenal", "Man City", round="Matchday 1")]
    feed = FixtureFeed(resolver)
    feed._cache["E0"] = (float("inf"), season)
    priced = Fixture("E0", soon, "15:00", "Arsenal", "Man City",
                     odds={"home": 2.0, "draw": 3.4, "away": 3.6}, source="football-data")
    import src.fixtures as fixtures_mod
    original = fixtures_mod.from_football_data
    fixtures_mod.from_football_data = lambda leagues=None, timeout=30: [priced]
    try:
        out = feed.upcoming(["E0"], days=14)
    finally:
        fixtures_mod.from_football_data = original

    assert out[0].odds == priced.odds, "the returned fixture should carry the odds"
    assert season[0].odds is None, "the cached season must not keep them"
