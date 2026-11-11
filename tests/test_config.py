"""Path resolution: a redirected data directory must not hide a real model."""
from __future__ import annotations

from src.config import resolve_model


def test_writable_copy_wins(tmp_path):
    """A retrained model in the writable directory beats the shipped one."""
    primary, bundled = tmp_path / "user", tmp_path / "bundle"
    primary.mkdir(), bundled.mkdir()
    (primary / "model.joblib").write_bytes(b"new")
    (bundled / "model.joblib").write_bytes(b"shipped")
    assert resolve_model("model.joblib", primary, [bundled]) == primary / "model.joblib"


def test_falls_back_when_the_writable_copy_is_missing(tmp_path):
    """The regression: redirecting the data directory used to hide the model."""
    primary, bundled = tmp_path / "user", tmp_path / "bundle"
    primary.mkdir(), bundled.mkdir()
    (bundled / "model.joblib").write_bytes(b"shipped")
    assert resolve_model("model.joblib", primary, [bundled]) == bundled / "model.joblib"


def test_fallbacks_are_tried_in_order(tmp_path):
    primary = tmp_path / "user"
    first, second = tmp_path / "a", tmp_path / "b"
    for d in (primary, first, second):
        d.mkdir()
    (second / "model.joblib").write_bytes(b"found here")
    assert resolve_model("model.joblib", primary, [first, second]) == second / "model.joblib"


def test_missing_everywhere_reports_the_primary_path(tmp_path):
    """So the error message names the place the user would look."""
    primary = tmp_path / "user"
    primary.mkdir()
    got = resolve_model("model.joblib", primary, [tmp_path / "nope"])
    assert got == primary / "model.joblib"
    assert not got.exists()


def test_every_league_can_be_fetched_by_the_default_source():
    """A league listed without a mirror slug silently returns no matches."""
    from src.config import LEAGUES, MIRROR_SLUGS

    assert set(LEAGUES) == set(MIRROR_SLUGS)


def test_fixture_feed_covers_exactly_those_leagues():
    from src.config import LEAGUES
    from src.fixtures import OPENFOOTBALL_CODES

    assert set(OPENFOOTBALL_CODES) == set(LEAGUES)
