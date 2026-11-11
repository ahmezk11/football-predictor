"""The desktop app's API must answer only its own page, not any website."""
from __future__ import annotations

import pytest

LOCAL = "http://127.0.0.1:8000"


def test_same_origin_requests_are_served(client):
    r = client.get("/health", headers={"Origin": LOCAL})
    assert r.status_code == 200


def test_requests_without_an_origin_are_served(client):
    """The app's own page loads, and curl or a test has no Origin at all."""
    assert client.get("/health").status_code == 200


def test_a_website_cannot_read_your_data(client):
    """Any page open in a browser could reach a known loopback port."""
    r = client.get("/favourites", headers={"Origin": "https://evil.example"})
    assert r.status_code == 403
    assert "cross-origin" in r.json()["detail"]


@pytest.mark.parametrize("method,path", [
    ("post", "/refresh"),        # downloads data and can trigger a retrain
    ("post", "/settle"),
    ("delete", "/predictions/1"),
])
def test_a_website_cannot_change_anything(client, method, path):
    """Simple POSTs skip the preflight, so removing CORS alone would not stop these."""
    r = getattr(client, method)(path, headers={"Origin": "https://evil.example"})
    assert r.status_code == 403


def test_a_page_on_another_local_port_is_refused(client):
    """Another app on this machine is not this app."""
    r = client.get("/favourites", headers={"Origin": "http://127.0.0.1:9999"})
    assert r.status_code == 403


def test_dns_rebinding_is_refused(client):
    """A hostname the attacker controls, pointed at 127.0.0.1, shares our origin."""
    r = client.get("/favourites", headers={"Host": "evil.example", "Origin": "http://evil.example"})
    assert r.status_code == 403
    assert "local machine" in r.json()["detail"]


def test_localhost_is_still_allowed(client):
    r = client.get("/health", headers={"Host": "localhost:8000", "Origin": "http://localhost:8000"})
    assert r.status_code == 200


# --- the refresh lock ---------------------------------------------------------
def test_a_finished_refresh_frees_the_lock_for_the_next_one(monkeypatch):
    """The job releases the lock it took, so a later refresh can still start."""
    from types import SimpleNamespace

    from src import api as api_mod

    monkeypatch.setattr("src.refresh.run", lambda args: {"data": {"new_matches": 0}})
    assert api_mod._job_lock.acquire(blocking=False)
    api_mod._run_refresh(False)
    assert api_mod._job_lock.acquire(blocking=False), "lock was not released"
    api_mod._job_lock.release()
    assert api_mod.state["job"]["status"] == "done"


def test_a_failing_refresh_still_frees_the_lock(monkeypatch):
    from src import api as api_mod

    def boom(args):
        raise RuntimeError("network down")

    monkeypatch.setattr("src.refresh.run", boom)
    assert api_mod._job_lock.acquire(blocking=False)
    api_mod._run_refresh(False)
    assert api_mod._job_lock.acquire(blocking=False), "lock was not released after a failure"
    api_mod._job_lock.release()
    assert api_mod.state["job"]["status"] == "failed"


def test_releasing_without_holding_the_lock_is_harmless(monkeypatch):
    """Guards the old bug: the job used to release whatever lock it found held."""
    from src import api as api_mod

    monkeypatch.setattr("src.refresh.run", lambda args: {})
    assert not api_mod._job_lock.locked()
    api_mod._run_refresh(False)          # nothing acquired; must not raise
    assert not api_mod._job_lock.locked()
