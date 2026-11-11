"""FastAPI service: prediction API plus the static web front-end."""
from __future__ import annotations

import logging
import os
import threading
from contextlib import asynccontextmanager
from datetime import date as date_cls
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from .config import BUNDLE, CLASS_LABELS, LEAGUES, ROOT
from .fixtures import FixtureFeed, TeamResolver
from .predict import Predictor, UnknownTeamError
from .store import Store

log = logging.getLogger(__name__)


def _web_dir():
    """The front-end lives beside the source in dev, inside the binary once frozen.

    PyInstaller unpacks bundled data to _MEIPASS, which is the only copy in a
    packaged build - deliberately, so there is nothing to fall out of sync.
    """
    import sys
    candidates = [BUNDLE / "web", ROOT / "web"]
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.insert(0, Path(meipass) / "web")
    return next((c for c in candidates if (c / "index.html").exists()), candidates[-1])


WEB_DIR = _web_dir()

state: dict = {"predictor": None, "feed": None, "store": None, "job": None}
_job_lock = threading.Lock()


def get_predictor() -> Predictor:
    p = state.get("predictor")
    if p is None:
        raise HTTPException(status_code=503,
                            detail="model not loaded - run `python -m src.train` and restart")
    return p


def get_store() -> Store:
    if state.get("store") is None:
        state["store"] = Store()
    return state["store"]


def get_feed() -> FixtureFeed:
    """Fixture feed bound to the vocabulary of the currently loaded model."""
    if state.get("feed") is None:
        p = get_predictor()
        known = {lg["code"]: p.teams(lg["code"]) for lg in p.leagues()}
        state["feed"] = FixtureFeed(TeamResolver(known))
    return state["feed"]


def _load_model() -> None:
    state["predictor"] = Predictor.load()
    state["feed"] = None       # vocabulary may have changed, rebuild on demand


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _load_model()
        log.info("loaded model: %s", state["predictor"].meta.get("model_name"))
    except FileNotFoundError as exc:
        log.warning("%s", exc)
        state["predictor"] = None
    state["store"] = Store()
    _maybe_auto_refresh()
    _forecast_in_background("startup")
    yield
    state.clear()


app = FastAPI(
    title="Football Match Outcome Predictor",
    description="Predicts home win / draw / away win from historical form, "
                "Elo ratings, head-to-head records and match statistics.",
    version="1.0.0",
    lifespan=lifespan,
)
# The API is the desktop app's own backend on a loopback port, and the page is
# served from the same origin, so it needs no cross-origin access at all. It used
# to allow any origin, which let any website the user happened to have open read
# their followed teams and prediction history, delete predictions, or set off a
# refresh and retrain. Two checks close that:
#
#   Origin  a browser sets it on every cross-origin request and a page cannot
#           forge it, so requiring it to match this server stops a website
#           calling us - including the simple POSTs that skip a preflight and
#           would otherwise still fire even with CORS removed.
#   Host    blocks DNS rebinding, where a hostname the attacker controls is
#           pointed at 127.0.0.1 so their page and this server share an origin.
LOOPBACK = {"127.0.0.1", "localhost", "::1"}


def _forbidden(reason: str) -> JSONResponse:
    return JSONResponse({"detail": f"refused: {reason}"}, status_code=403)


@app.middleware("http")
async def local_only(request: Request, call_next):
    host_header = request.headers.get("host", "")
    hostname = host_header.rsplit(":", 1)[0].strip("[]") if host_header else ""
    if hostname and hostname not in LOOPBACK:
        return _forbidden("this API only answers on the local machine")

    origin = request.headers.get("origin")
    if origin:
        parsed = urlparse(origin)
        if (parsed.hostname or "") not in LOOPBACK or parsed.netloc != host_header:
            return _forbidden("cross-origin requests are not accepted")
    return await call_next(request)


AUTO_REFRESH_DAYS = int(os.environ.get("PREDICTOR_AUTO_REFRESH_DAYS", "7"))


def _forecast_in_background(trigger: str) -> None:
    """Predict every league without blocking the window from opening."""
    if state.get("predictor") is None:
        return

    def work():
        from .autopredict import forecast_all
        try:
            counts = forecast_all(get_predictor(), get_feed(), get_store())
            total = sum(c.get("forecast", 0) for c in counts.values())
            locked = sum(c.get("locked_in", 0) for c in counts.values())
            log.info("auto-forecast (%s): %d fixtures forecast, %d locked in",
                     trigger, total, locked)
        except Exception:
            log.exception("auto-forecast failed")

    threading.Thread(target=work, daemon=True, name=f"forecast-{trigger}").start()


def _maybe_auto_refresh() -> None:
    """Refresh in the background if the data has gone stale.

    A desktop app has no cron, but it does get launched. Checking staleness at
    startup gives the same effect without anything to install, and the work
    happens on a background thread so the window still opens immediately.
    Set PREDICTOR_AUTO_REFRESH_DAYS=0 to turn it off.
    """
    if AUTO_REFRESH_DAYS <= 0 or state.get("predictor") is None:
        return
    last = get_store().get_meta("last_data_refresh")
    if last:
        try:
            age = datetime.now(timezone.utc) - datetime.fromisoformat(last)
            if age.days < AUTO_REFRESH_DAYS:
                return
        except ValueError:
            pass   # unparseable timestamp: treat as stale

    if not _job_lock.acquire(blocking=False):
        return
    log.info("data is stale - refreshing in the background")
    state["job"] = {"status": "running", "trigger": "startup",
                    "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    threading.Thread(target=_run_refresh, args=(False,), daemon=True).start()


# --- schemas ----------------------------------------------------------------
class Odds(BaseModel):
    home: float = Field(gt=1.0, description="decimal odds for a home win")
    draw: float = Field(gt=1.0)
    away: float = Field(gt=1.0)


class PredictionRequest(BaseModel):
    league: str = Field(examples=["E0"])
    home_team: str = Field(examples=["Arsenal"])
    away_team: str = Field(examples=["Chelsea"])
    match_date: date_cls | None = Field(default=None, description="defaults to today")
    odds: Odds | None = Field(default=None,
                              description="optional bookmaker decimal odds")

    @model_validator(mode="after")
    def _distinct(self):
        if self.home_team.strip().lower() == self.away_team.strip().lower():
            raise ValueError("home_team and away_team must differ")
        return self


class BatchRequest(BaseModel):
    matches: list[PredictionRequest] = Field(min_length=1, max_length=100)


class FollowRequest(BaseModel):
    league: str
    team: str


# --- routes -----------------------------------------------------------------
@app.get("/health", tags=["meta"])
def health():
    p = state.get("predictor")
    return {
        "status": "ok" if p else "model_missing",
        "model": p.meta.get("model_name") if p else None,
        "trained_at": p.meta.get("trained_at") if p else None,
        "training_rows": p.meta.get("train_rows") if p else None,
        "data_range": p.meta.get("data_range") if p else None,
    }


@app.get("/model", tags=["meta"])
def model_info():
    p = get_predictor()
    return {
        "name": p.meta.get("model_name"),
        "uses_market_odds": p.meta.get("use_odds"),
        "trained_at": p.meta.get("trained_at"),
        "training_rows": p.meta.get("train_rows"),
        "data_range": p.meta.get("data_range"),
        "leagues": p.meta.get("leagues"),
        "features": p.features,
        "holdout_metrics": p.meta.get("holdout_metrics"),
        "classes": {c: CLASS_LABELS[c] for c in p.classes},
    }


@app.get("/leagues", tags=["reference"])
def leagues():
    return get_predictor().leagues()


@app.get("/teams", tags=["reference"])
def teams(league: str | None = Query(default=None, description="e.g. E0")):
    p = get_predictor()
    if league and league not in dict.fromkeys(t["code"] for t in p.leagues()):
        raise HTTPException(404, f"unknown league '{league}'. Known: {list(LEAGUES)}")
    return {"league": league, "teams": p.teams(league)}


@app.get("/teams/{league}/{team}/form", tags=["reference"])
def team_form(league: str, team: str):
    p = get_predictor()
    if team not in set(p.teams(league)):
        raise HTTPException(404, f"unknown team '{team}' in {league}")
    return p.team_form(league, team)


@app.post("/predict", tags=["prediction"])
def predict(req: PredictionRequest):
    p = get_predictor()
    try:
        return p.predict(req.league, req.home_team, req.away_team,
                         req.match_date, req.odds.model_dump() if req.odds else None)
    except UnknownTeamError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/predict/batch", tags=["prediction"])
def predict_batch(req: BatchRequest):
    p = get_predictor()
    return {"predictions": p.predict_many([
        {"league": m.league, "home_team": m.home_team, "away_team": m.away_team,
         "date": m.match_date, "odds": m.odds.model_dump() if m.odds else None}
        for m in req.matches])}


@app.get("/fixtures", tags=["prediction"])
def fixtures(league: str | None = None, days: int = Query(14, ge=1, le=120)):
    """Every upcoming fixture in the window, scored - not just followed teams."""
    p, feed = get_predictor(), get_feed()
    leagues = [league] if league else [lg["code"] for lg in p.leagues()]
    found = feed.upcoming(leagues, days=days)
    if not found:
        return {"fixtures": [], "note": "no fixtures found in this window"}
    out = []
    for fx in found:
        try:
            out.append({**fx.as_dict(),
                        **p.predict(fx.league, fx.home_team, fx.away_team,
                                    fx.date, fx.odds)})
        except (UnknownTeamError, ValueError) as exc:
            out.append({**fx.as_dict(), "error": str(exc)})
    return {"fixtures": out}


@app.post("/reload", tags=["meta"])
def reload_model():
    """Pick up a freshly trained model without restarting the process."""
    try:
        _load_model()
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc
    return {"status": "reloaded", "model": state["predictor"].meta.get("model_name")}


# --- following ---------------------------------------------------------------
@app.get("/favourites", tags=["personal"])
def list_favourites():
    return {"favourites": get_store().favourites()}


@app.post("/favourites", tags=["personal"])
def add_favourite(req: FollowRequest):
    p = get_predictor()
    if req.team not in set(p.teams(req.league)):
        raise HTTPException(404, f"unknown team '{req.team}' in {req.league}")
    get_store().follow(req.league, req.team)
    return {"status": "following", **req.model_dump()}


@app.delete("/favourites", tags=["personal"])
def remove_favourite(league: str, team: str):
    get_store().unfollow(league, team)
    return {"status": "unfollowed", "league": league, "team": team}


# --- the home screen ---------------------------------------------------------
@app.get("/upcoming", tags=["personal"])
def upcoming(days: int = Query(14, ge=1, le=120),
             following_only: bool = True,
             league: str | None = None,
             save: bool = False):
    """Next fixtures, already scored. This is what the app opens on."""
    p, store, feed = get_predictor(), get_store(), get_feed()

    leagues = [league] if league else [lg["code"] for lg in p.leagues()]
    teams = store.favourite_teams() if following_only else None
    if following_only and not teams:
        return {"fixtures": [], "following": [],
                "note": "not following any teams yet"}

    fixtures = feed.upcoming(leagues, days=days, teams=teams)
    out = []
    for fx in fixtures:
        try:
            pred = p.predict(fx.league, fx.home_team, fx.away_team, fx.date, fx.odds)
        except (UnknownTeamError, ValueError) as exc:
            out.append({**fx.as_dict(), "error": str(exc)})
            continue
        if save:
            store.save_prediction(pred)
        out.append({**fx.as_dict(), **pred})
    return {"fixtures": out, "following": teams or [], "days": days}


@app.get("/schedule", tags=["prediction"])
def schedule(league: str, round: str | None = None):
    """A league's season by matchday, every unplayed fixture forecast automatically.

    Fixtures in the next few days are also locked in to the prediction history
    so they get graded; see src/autopredict.py for why only those.
    """
    from .autopredict import forecast_league

    p = get_predictor()
    if league not in {lg["code"] for lg in p.leagues()}:
        raise HTTPException(404, f"unknown league '{league}'")

    data = forecast_league(p, get_feed(), get_store(), league)
    # The picker needs every round, even when the response carries only one.
    data["rounds"] = [{"round": g["round"], "first_date": g["first_date"],
                       "played": g["played"], "total": g["total"]}
                      for g in data["matchdays"]]
    if round:
        data["matchdays"] = [g for g in data["matchdays"] if g["round"] == round]
        if not data["matchdays"]:
            raise HTTPException(404, f"no matchday '{round}' in {league}")
    return data


@app.get("/forecast/summary", tags=["prediction"])
def forecast_summary():
    """Forecast every league at once and report what was scored and locked in."""
    from .autopredict import forecast_all
    return {"leagues": forecast_all(get_predictor(), get_feed(), get_store())}


# --- prediction history ------------------------------------------------------
@app.post("/predictions", tags=["personal"])
def save_prediction(req: PredictionRequest):
    """Predict a fixture and keep it, so it can be scored once it is played."""
    p, store = get_predictor(), get_store()
    try:
        pred = p.predict(req.league, req.home_team, req.away_team,
                         req.match_date, req.odds.model_dump() if req.odds else None)
    except UnknownTeamError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"id": store.save_prediction(pred), "saved": True, **pred}


@app.get("/predictions", tags=["personal"])
def list_predictions(limit: int = Query(200, ge=1, le=2000), settled_only: bool = False):
    return {"predictions": get_store().history(limit=limit, settled_only=settled_only)}


@app.delete("/predictions/{prediction_id}", tags=["personal"])
def delete_prediction(prediction_id: int):
    if not get_store().forget(prediction_id):
        raise HTTPException(404, f"no prediction with id {prediction_id}")
    return {"status": "deleted", "id": prediction_id}


@app.get("/scorecard", tags=["personal"])
def scorecard():
    """How the model has done on the fixtures you actually asked about."""
    return get_store().scorecard()


@app.post("/settle", tags=["personal"])
def settle():
    """Look up results for saved predictions whose matches have been played."""
    from .refresh import settle_predictions
    return settle_predictions(get_store(), get_predictor())


# --- maintenance -------------------------------------------------------------
def _run_refresh(force: bool) -> None:
    from types import SimpleNamespace

    from .refresh import run as refresh_run
    try:
        report = refresh_run(SimpleNamespace(
            force_retrain=force, no_retrain=False, cv_splits=5))
        if report.get("retrained"):
            _load_model()
        _forecast_in_background("refresh")
        state["job"] = {"status": "done",
                        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "report": report}
    except Exception as exc:                      # keep the endpoint truthful
        log.exception("refresh failed")
        state["job"] = {"status": "failed", "error": str(exc)}
    finally:
        # This runs only on the thread that acquired the lock before starting the
        # job. Lock.locked() cannot tell whose lock it is, so testing it risked
        # releasing a refresh started by someone else.
        try:
            _job_lock.release()
        except RuntimeError:            # already released; nothing to undo
            pass


@app.post("/refresh", tags=["maintenance"])
def refresh(background: BackgroundTasks, force_retrain: bool = False):
    """Pull new results, settle predictions, retrain if the model has drifted."""
    if not _job_lock.acquire(blocking=False):
        raise HTTPException(409, "a refresh is already running")
    state["job"] = {"status": "running",
                    "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    background.add_task(_run_refresh, force_retrain)
    return {"status": "started"}


@app.get("/refresh/status", tags=["maintenance"])
def refresh_status():
    return state.get("job") or {"status": "idle"}


@app.get("/drift", tags=["monitoring"])
def drift():
    """How the sport and the model have moved, season by season.

    Costs a feature pass over every match, so it is computed once per loaded
    model and kept.
    """
    from . import data as data_mod
    from .monitor import season_drift

    p = get_predictor()
    key = p.meta.get("trained_at")
    cached = state.get("drift")
    if cached and cached.get("key") == key:
        return cached["data"]
    try:
        data = season_drift(data_mod.load(), p)
    except FileNotFoundError as exc:
        raise HTTPException(503, f"no match data to analyse: {exc}") from exc
    state["drift"] = {"key": key, "data": data}
    return data


@app.get("/stats", tags=["meta"])
def app_stats():
    return get_store().stats()


# --- front-end --------------------------------------------------------------
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(WEB_DIR / "index.html")
