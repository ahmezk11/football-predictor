# Match Predictor

A macOS desktop app that predicts **home win / draw / away win** for football
fixtures, from historical results, team form, Elo ratings, head-to-head records
and match statistics.

Follow your teams, see their next fixtures already scored, and keep a running
record of how the model actually does on them. Underneath it is a Python
training pipeline, a local FastAPI service and a drift monitor.

Built on **38,023 matches** across five European leagues (Premier League, La Liga,
Serie A, Bundesliga, Ligue 1), 2005/06 – 2025/26 — 37,236 of them usable once every
club has a five-match history behind it.

![the app](desktop/icon-preview.png)

## Running it

```bash
make install && make data && make train   # one-time: deps, data, model
make app-dev                              # run the desktop app
```

To build a real double-clickable `Match Predictor.app` and a `.dmg`:

```bash
make app
```

That freezes the Python server into a standalone binary with PyInstaller, wraps
it with Electron, ad-hoc signs the bundle and rebuilds the disk image from the
signed copy. The finished app needs no Python, no venv and no terminal, and comes
to about 359 MB installed (233 MB of that is Electron's own Chromium).

**Signing.** `scripts-sign.sh` gives the bundle one consistent ad-hoc signature
under its own identifier. Without it, electron-builder rewrites `Contents/Resources`
after the linker's signature and leaves it invalid — `spctl` reports "code has no
resources but signature indicates they must be present", and a copy carrying the
quarantine flag is refused outright rather than showing the usual prompt.

Ad-hoc is not notarization. A build **you** create locally has no quarantine flag
and opens normally. A copy someone downloads or AirDrops does, and will still need
right-click → Open. Removing that step for other people needs an Apple Developer ID
certificate and `xcrun notarytool submit --keychain-profile ... --wait`, which
needs a paid account.

Signing runs in a temporary directory on purpose: this project lives on an
iCloud-synced Desktop, and the sync agent reattaches Finder metadata while codesign
walks the bundle ("resource fork, Finder information, or similar detritus not
allowed"). Signing on local disk and copying the result back is what works.

**Size.** `make selftest` trains a model *inside* the frozen build. Serving a
prediction only proves the model loads; the app can also retrain itself, which
reaches much further into scikit-learn and scipy. Trimming the bundle without that
check is how you ship a build that dies on its first refresh — it caught exactly
that three times (`sklearn.externals`, then `scipy.integrate`, then `scipy.ndimage`,
each pulled in lazily by something that looked unrelated).

Prefer a browser? `make serve` gives the same interface at
<http://localhost:8000>, and `/docs` has the generated API reference.

## What the app does

The interface borrows the conventions of football apps like FotMob and SofaScore:
sidebar navigation with league shortcuts, fixtures grouped by day with the home side
on the left and kick-off or score in the middle, and a match sheet that slides over
with the detail.

| | |
|---|---|
| **Matches** | Upcoming fixtures for the clubs you follow, or every league, over 7 to 90 days. Each row shows the forecast as three outcome boxes — **1** home win, **X** draw, **2** away win — with the model's pick highlighted. |
| **Predictions** | Runs itself. Every unplayed match in all five leagues is forecast automatically. Pick a league and step through its matchdays. A lock marks matches in the next 7 days, which are saved to History to be graded; played rounds show the score with a tick or cross against the call. |
| **Match sheet** | Click any fixture: a "Who will win?" bar, the most likely scorelines with over/under 2.5 goals and both-teams-to-score, both sides' last-five form as W/D/L, and mirrored comparison bars for Elo, points per game, goals scored and conceded, and venue form, plus head to head. |
| **History** | Accuracy as a ring, how many calls are graded or still waiting, RPS, and a predicted-versus-happened breakdown by outcome — which makes the model's reluctance to call draws plain to see. Every figure sits next to what guessing would have scored on the same matches, and the goals and exact-score calls are graded separately. Two charts show the drift the monitor found: the outcome mix by season, and how predictable each season was. |
| **My teams** | Follow clubs; each gets a card with recent form and Elo. |

Club badges are initials on the club's colours, not real crests — those are
trademarks. Clubs without a colour entry get one derived from their name. The app
uses the system font, so it starts and renders identically offline.

The app keeps its database in `~/Library/Application Support/Match Predictor/`,
so a rebuild never destroys your history.

### Staying current

Data goes stale, so the app refreshes itself. On launch, if the data is more than
7 days old it pulls new results in the background, settles any predictions whose
matches have been played, and checks the drift monitor. It only retrains when
recent performance has actually slipped or ~400 new matches have accumulated —
a handful of new results does not move a model fitted on 37,000. There is also a
**Refresh data** button and ⌘⇧R.

Set `PREDICTOR_AUTO_REFRESH_DAYS=0` to turn the automatic check off.

### When something is unreachable

A failed request means one of two things, and the app does not treat them alike.
If the server answers with an error, it is shown. If the request never lands — the
machine is offline, the backend is restarting — the last good answer is served from
the browser's storage under a quiet banner saying how old it is, rather than a dead
screen. The backend draws the same distinction: if the fixture source cannot be
reached it keeps the last schedule it downloaded and says so, instead of reporting
that a league has no fixtures.

---

## Results

Evaluated on the most recent 20% of matches (7,448 games, April 2022 – May 2026),
never seen during training, tuning or model selection.

| | Accuracy | Macro F1 | Log loss | RPS ↓ | Brier ↓ |
|---|---|---|---|---|---|
| Always predict home win | 0.437 | 0.203 | 4.281 | 0.436 | 1.125 |
| Base rates from training data | 0.437 | 0.203 | 1.074 | 0.231 | 0.650 |
| Previous model (Elo + rolling form) | 0.525 | 0.395 | 0.988 | 0.2008 | 0.589 |
| **This model** | **0.529** | **0.394** | **0.982** | **0.1995** | **0.585** |

**RPS** (ranked probability score) is the headline number. Accuracy throws away the
probabilities and treats "away win at 34%" the same as "away win at 90%"; RPS scores
the whole distribution *and* respects the fact that H–D–A is an ordered scale, so
calling a home win when it finished a draw costs less than calling an away win.
Model selection is done on RPS, not accuracy.

For scale: published work puts bookmaker closing odds at roughly **0.19 RPS** on
comparable data — they see injuries, lineups and team news that no feature here can.
The retrain closed about an eighth of the gap to that.

That 0.19 is someone else's number on someone else's matches. `src/benchmark.py`
scores the market and the model on *identical* fixtures and reports where the two
disagree and who was right. It is written and tested, but not yet run: it needs
closing odds, and the only source for them (football-data.co.uk) is currently
refusing our requests entirely. `make data-odds` then `python -m src.benchmark`
produces the number as soon as it lets us back in.

### Is the improvement real?

The gain over the previous model is modest, so it was checked rather than assumed.
Scored match by match on the same 7,448 holdout games, a paired bootstrap puts the
RPS improvement at **+0.0014 (95% interval +0.0004 to +0.0023)**; the new model is
better in 99.6% of 5,000 resamples, and better in four of the five holdout seasons
(level in 2023/24). Cross-validation on the training period agreed beforehand:
0.2008 → 0.1988.

Calibration also tightened at the confident end, where the previous model overstated
its certainty:

| Predicted home win | 0.25 | 0.35 | 0.45 | 0.55 | 0.65 | 0.74 | 0.84 |
|---|---|---|---|---|---|---|---|
| Observed rate | 0.237 | 0.355 | 0.439 | 0.525 | 0.654 | 0.744 | 0.841 |

### What the model uses

Eleven features, almost all from the rating systems in `src/ratings.py`. Permutation
importance on held-out data — how much RPS worsens when a feature is shuffled:

| Feature | What it is | RPS increase |
|---|---|---|
| `sot_rate_diff` | opponent-adjusted shots-on-target rating gap | 0.0047 |
| `elo_diff` | Elo gap | 0.0011 |
| `elo_expected_home` | Elo home-win expectancy | 0.0007 |
| `pois_p_away` / `pois_p_home` / `pois_p_draw` | outcome probabilities from the goal model | 0.0005 / 0.0005 / 0.0003 |
| `xg_away` / `xg_home` | expected goals from attack and defence ratings | 0.0005 / 0.0001 |
| `pi_expected_gd` | goal difference expected from home/away pi-ratings | 0.0005 |
| `league_home_rate`, `league_draw_rate` | recent outcome mix in the league | ~0 |

The shots-on-target rating carries four times Elo's weight — consistent with the
literature, where chance creation predicts future results better than past goals
do. The Elo, pi and Poisson features overlap heavily, so shuffling any one of them
costs little while the others stand in; their individual numbers understate how
much the group matters. The league-context features do essentially nothing: the
idea of tracking home advantage as it drifts did not pan out as a feature.

### Two honest caveats

**The model almost never predicts a draw.** Draws are ~25% of matches but are rarely the
*single most likely* outcome, so `argmax` picks none of the 7,448 holdout matches as a
draw. This is a property of the sport, not a bug:
the model assigns draws a well-calibrated 20–30% throughout, which is why RPS and log
loss look fine while draw F1 does not. If you need draw calls, threshold on probability
instead of taking the argmax — and expect accuracy to fall when you do.

**Home advantage moved, and the model was trained through the shift.** The monitor's
season table shows home-win rate falling from ~46% (2005–2018) to 40.4% in 2020/21 —
the empty-stadium season — and staying at 42–43% since, with away wins up from 27% to
33%. This is exactly the kind of drift that silently degrades a deployed model, which is
what `src/monitor.py` is for; see [Monitoring](#6-monitoring).

---

## How it works

### 1. Data collection

`src/data.py` pulls season CSVs — one row per played match with the full-time score,
shots, shots on target, corners, fouls and cards. Two interchangeable sources:

| Source | Coverage | Notes |
|---|---|---|
| `mirror` (default) | 5 leagues, 2005–2026, all match stats | A GitHub mirror of the football-data.co.uk CSVs. Reliable, no key, no rate limits. |
| `football-data` | Adds closing bookmaker odds | The original site. Throttles aggressively; expect failures. |

```bash
python -m src.data --source mirror --from-year 2005 --to-year 2025
python -m src.data --source both      # merge in odds where available
```

Odds are only needed to benchmark against the market and to train the `--use-odds`
variant. Everything else works without them.

### 2. Feature engineering

`src/features.py` makes a single chronological pass. For each match it emits the
features **as they stood before kick-off**, then folds the result into team state.
That one design decision does two jobs:

- **No leakage.** A row cannot see its own result or any later match. `tests/test_features.py`
  asserts this directly: features computed over the full dataset must be bit-identical
  to features computed over a truncated prefix.
- **Training and serving share a code path.** Scoring an unplayed fixture calls exactly
  the same function, so what the API sees is what the model was trained on.

63 features are computed in seven groups. The model trains on the 11 rating-based
ones (see [section 3](#3-ratings-and-how-the-model-was-improved)); the rest are kept
because the app shows them as the reasoning behind a prediction.

| Group | Examples |
|---|---|
| **Ratings** | Elo, pi-ratings, expected goals and outcome probabilities from the Poisson model, shots-on-target rating, league outcome mix — see section 3 |
| **Recent form** | points per game, goals for/against over the last 5 and 10 matches |
| **Venue form** | how the home side goes *at home*, the away side *away* |
| **Match statistics** | shots, shots on target, shots on target conceded, corners, cards (last 5) |
| **Team strength** | scoring and conceding rates relative to a rolling league average |
| **Context** | head-to-head over the last 6 meetings, days of rest, matches played, season stage |
| **Market** *(optional)* | overround-free implied probabilities from closing odds |

Teams with fewer than 5 recorded matches are dropped — their form features are mostly
empty and add noise.

### 3. Ratings, and how the model was improved

`src/ratings.py` holds four systems, each updated one match at a time so they share
the feature builder's no-leakage, train-equals-serve design:

| System | What it adds |
|---|---|
| **Elo** | one strength number per club |
| **Pi-ratings** (Constantinou & Fenton, 2013) | separate home and away strength, in goal-difference units |
| **Online Poisson** | attack and defence ratings fitted to a Poisson model of scoring, with a per-league scoring level and home advantage that drift; gives expected goals and outcome probabilities, with a Dixon-Coles correction for low-scoring draws |
| **Online Poisson on shots on target** | the same model on chances created rather than goals |

Unlike rolling averages, every one rates a club relative to the opposition it
actually faced. Each gives a club new to an established league a below-average
starting rating, since such a club is almost always newly promoted.

**Tuning** (`python -m src.tune_ratings`). Each system's settings are fitted by
coordinate descent on matches before April 2022 only, scored walk-forward by its own
predictions: Elo's Brier score improved 1.9% (notably K 20 → 7.5, and promoted clubs
starting 75 points below average), the Poisson model's RPS 0.2012 → 0.1999.

**What made the model better, and what did not** (`python -m src.experiments`,
cross-validated RPS on the training period, same folds for every row):

| Change | CV RPS |
|---|---|
| Previous model | 0.2008 |
| + tuned Elo and promoted-club prior | 0.2001 |
| + all four rating systems on top of the 47 old features | 0.2012 — *worse* |
| Poisson ratings alone, no learned model | 0.1996 |
| **11 rating features, strong regularisation (C = 0.002)** | **0.1988** |
| … blended 50/50 with the Poisson probabilities | 0.1989 — no better, dropped |
| Gradient boosting / logistic–boosting blend / recency weighting | 0.1993 – 0.2046 |
| Proportional-odds (ordinal) model, best alpha | 0.19879 — a tie |
| League × rating interactions | 0.1989 – 0.1992 — worse |
| Pooled blended with per-league models | 0.1989 – 0.1990 — worse |

**The ordering is already in the features.** H–D–A is an ordered scale, which is
why RPS is the headline metric, and multinomial logistic regression does not know
that — it fits each outcome independently. A proportional-odds model
(`src/ordinal.py`) builds the ordering in: one score per match and two thresholds,
so a match slides along a single axis from away win to home win with the draw in
between. It matches the multinomial almost exactly — 0.19879 against 0.19882, and
fold by fold it wins 3 of 5 by a mean of 0.00003, about fifty times smaller than
the gain from the ratings rework and smaller than the spread between folds. It is
a tie, so the simpler shipped path stays; the model remains in the repo as tested
experiment code. Worth noting it matches on half the parameters.

**League identity adds nothing either.** The earlier null result was for rolling
league *rates*; this was the other test — letting the rating gap and home
advantage differ by league, through interactions and through blending the pooled
model with per-league ones. Both were worse. Accuracy does vary by league (0.545
in the Premier League against 0.501 in Ligue 1), but that looks like differences
in how predictable the leagues are, not in how the features should be weighted.

Two lessons. Adding good features to a crowded model made it *worse* — the ratings
already summarise what the form, shots and venue columns were carrying, and the
overlap is noise to a linear model; cutting to eleven features and regularising hard
is what worked. And the promoted-club prior, the most intuitive fix, is not where the
gain comes from: matches involving a promoted club improved by about as much as
everything else. The improvement is broad.

The regularisation strength is not an artefact of the small early folds either: on
the three largest folds alone, RPS is flat from C = 0.001 to 0.01 and best at 0.002.

Three estimators are still compared in `src/train.py` by rolling-origin
cross-validation (`TimeSeriesSplit`, 5 folds — every fold trains on the past and
tests on the future). With these features logistic regression wins (0.1988) over
gradient boosting (0.1999) and random forest (0.2000). Post-hoc calibration is
evaluated on training folds only and applied only if it helps; it does not
(0.1995 calibrated vs 0.1988 raw), so the model ships uncalibrated.

### 4. What the score distribution gives you

The Poisson rating model computes a probability for every scoreline, not three
numbers. Collapsing it to home/draw/away threw away most of what it knew, so the
app now shows the rest: the most likely scorelines, over/under 2.5 goals, both
teams to score, and clean-sheet chances, all from the same matrix.

Those scorelines are rescaled so their win/draw/loss shares match the trained
model's probabilities (`align_matrix`). Left alone the two would contradict each
other on screen — "62% home win" above a score distribution implying 55% — so the
detail reads as a decomposition of the headline rather than a rival to it.

Each market is graded on its own once results arrive, which is the point: the
outcome probabilities can look well calibrated while the goal model is off, and
over/under and both-teams-to-score are independent checks that would show it.

### 5. Evaluation

`src/evaluate.py` reports accuracy, macro/weighted F1, per-class precision/recall/F1,
log loss, multi-class Brier, and RPS, plus a confusion matrix, a calibration table
(predicted vs observed frequency in probability bins) and three baselines.

```bash
python -m src.train --cv-splits 5           # full run, writes reports/training_report.json
python -m src.train --use-odds              # market-aware variant (needs odds data)
python -m src.train --models logistic random_forest
```

### 6. Fixtures

`src/fixtures.py` merges two feeds, because neither alone is enough:

| Source | Horizon | Why |
|---|---|---|
| openfootball (GitHub JSON) | the whole season, plus results already played | Answers "when do my teams next play", and supplies the results used to settle predictions. |
| football-data.co.uk `fixtures.csv` | the next few days | Carries bookmaker odds. Thins to nothing during international breaks, so it cannot be the only source. |

openfootball uses official long names (`Borussia Mönchengladbach`) while the model
knows the short forms (`M'gladbach`), so names have to be resolved. That is done by
normalising away accents, punctuation, founding years and club-type tokens, then a
curated alias table, then a **token-subset rule**: the known name is nearly always a
subset of the official one, so `AJ Auxerre` → Auxerre and `US Sassuolo Calcio` →
Sassuolo fall out without enumerating every abbreviation.

The rule refuses to guess when two candidates are equally specific, and a club the
model has never seen stays unresolved rather than being silently mapped onto another
team — currently 94 of 96 clubs resolve, and the two that do not are newly promoted
sides with no history to predict from. Their fixtures are still listed, marked as
unpredictable: a matchday with a match quietly missing is more confusing than one
that says why it cannot be scored.

### 7. The API

`src/api.py` is a FastAPI service; `web/index.html` is a dependency-free front-end
served from the same process. The desktop app is this service plus an Electron shell.

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness plus loaded-model metadata |
| `GET /model` | features, training range, holdout metrics |
| `GET /leagues`, `GET /teams?league=E0` | reference data for the UI |
| `GET /teams/{league}/{team}/form` | current Elo and last-5 form |
| `POST /predict` | one fixture |
| `POST /predict/batch` | up to 100 fixtures, failures isolated per row |
| `GET /fixtures` | every upcoming fixture in a window, scored |
| `GET /schedule?league=E0` | a league's season by matchday, every unplayed fixture forecast |
| `GET /forecast/summary` | forecast every league at once; per-league counts |
| `GET /upcoming` | next fixtures for followed teams — what the app opens on |
| `GET`/`POST`/`DELETE` `/favourites` | teams you follow |
| `POST` `/predictions` | predict and save, for later grading |
| `GET /predictions`, `GET /scorecard` | history and how it has done |
| `POST /settle` | attach real results to saved predictions |
| `POST /refresh`, `GET /refresh/status` | background data refresh + conditional retrain |
| `POST /reload` | pick up a newly trained model without a restart |

```bash
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \
  -d '{"league":"E0","home_team":"Arsenal","away_team":"Chelsea"}'
```

```json
{
  "prediction": "H",
  "prediction_label": "Home win",
  "confidence": 0.5316,
  "probabilities": {"home_win": 0.5316, "draw": 0.2447, "away_win": 0.2237},
  "fair_odds": {"home_win": 1.88, "draw": 4.09, "away_win": 4.47},
  "context": {"elo": {"Arsenal": 1667.6, "Chelsea": 1588.4, "difference": 79.2}, "...": "..."}
}
```

Every prediction carries the reasoning alongside it — Elo, form, venue record,
head-to-head and rest days — so the number is inspectable rather than a black box.

### 8. The desktop app

`desktop/main.js` is a thin Electron shell that owns the lifecycle: it asks the OS
for a free port (so two copies never collide), spawns the backend, waits for
`/health`, then swaps its loading window for the app. The renderer gets no Node
access — it only ever talks to `127.0.0.1` over `fetch` — and nothing is ever bound
beyond loopback.

`make app` freezes the Python server with PyInstaller and packages it with
electron-builder. Because an installed bundle is read-only, `src/config.py`
distinguishes bundled read-only resources (the shipped model, the front-end) from
the writable user directory where the database and any retrained model live.

Still deployable as a web service if you want it:

```bash
docker build -t football-predictor .
docker run -p 8000:8000 -v "$PWD/models:/app/models" football-predictor
```

### 9. Monitoring

Football is not stationary — home advantage drifted during the empty-stadium seasons,
scoring rates move, squads turn over. `src/monitor.py` re-scores the deployed model
against recent matches and compares it to the metrics recorded in the model bundle:

```bash
python -m src.monitor --window-days 365
```

The prediction history carries its own benchmark too: the scorecard reports what
base rates and always-backing-the-home-team would have scored **on exactly the
same graded matches**, because "56% correct" is not interpretable on its own —
home wins alone are about 45% of matches, and these are the fixtures the user
happened to ask about rather than a balanced sample. On the current history that
reads 56% (RPS 0.192) against 48% (0.233) for the base rates.

It reports performance for the recent window, breakdowns by season and league, the
outcome mix per season (drift in the sport itself), and a pass/fail drift verdict
against a configurable tolerance. Writes `reports/monitoring_report.json`, so it can
be run on a schedule and alerted on.

The app shows this on the History tab now, as two charts: the outcome mix by
season, and ranked probability score by season against the held-out baseline. The
per-season scores are labelled as not out-of-sample — the shipped model is trained
on all of it — because the shape is informative but the level is not evidence of
accuracy.

The chart colours are chart-only tokens, not the app's semantic green/grey/blue:
those failed a colour-blindness check as adjacent stacked segments (blue against
grey, ΔE 13.8, below the 15 floor where even full colour vision struggles). The
replacements pass in both light and dark, and every chart carries a legend, direct
labels and a table view.

The season table is the useful part. Current output:

```
 season_start  matches  home_win_rate  draw_rate  away_win_rate  goals_per_game
         2005     1826          0.455      0.276          0.269           2.488
         ...
         2019     1504          0.440      0.247          0.313           2.777
         2020     2047          0.404      0.251          0.345           2.810   <- empty stadiums
         2021     1826          0.428      0.258          0.314           2.811
         2022     1826          0.457      0.243          0.300           2.766
         2023     1752          0.431      0.264          0.305           2.885
         2024     1752          0.420      0.249          0.330           2.827
```

Home advantage has not fully recovered, and scoring is up almost half a goal a game
since 2005. Per-league accuracy varies too — 0.545 in the Premier League against 0.501
in Ligue 1 — which is worth knowing before quoting a single headline number.

`src/refresh.py` ties it together for the app: pull new results, settle predictions,
then consult the drift check and only retrain when it is actually warranted.

```bash
make refresh              # or the Refresh data button / ⌘⇧R in the app
```

---

## CI

`.github/workflows/football-predictor.yml` runs on every push touching this
project: the test suite, then a real train on two seasons of one league followed
by `python -m src.smoke`, which loads the model the way the app does and scores a
fixture. That second job exists because training and serving run in different
processes, and the one failure that reached a user sat in that gap - a saved model
whose feature builder the running code could no longer use. Every test passed; the
app broke on launch. `make smoke` runs the same check locally.

## Tests

```bash
make test        # 204 tests
```

Beyond the usual coverage, the suite pins down the things that would silently
invalidate the results:

- **Leakage**: features over the full dataset must equal features over a prefix.
- **Determinism**: same-day matches must not reorder between runs (pandas' default
  sort is not stable — this was a real bug the test caught).
- **Metric correctness**: RPS checked against hand-computed analytic values, and
  proven invariant to the order class columns arrive in.
- **Negative control**: a model trained on shuffled labels must lose to the real one.
- **Serving**: the fitted feature-state store must pickle, prediction must not mutate
  state, and repeated calls must return identical probabilities.
- **Golden values**: a scripted run of matches and a set of fixtures have their
  feature values and probabilities pinned in `tests/golden/`. The other tests prove
  a run is *valid*; these prove it still gives the same answers, so a refactor of
  `features.py` or `ratings.py` cannot quietly move every prediction. A deliberate
  change is `make golden`, and the diff says exactly which numbers moved.
- **Local-only API**: a website's origin, a DNS-rebinding host and a cross-site
  `POST /refresh` must all be refused, while the app's own page is served.
- **Name resolution**: the two Madrid clubs and the two Manchester clubs must never
  collide, and an unknown club must stay unresolved rather than be remapped.
- **Settling**: a prediction must survive its fixture being rescheduled, and must
  never settle twice.

Tests are pinned to a temporary data directory and cannot touch the real database
or the network.

## Layout

```
src/
  config.py     data sources, paths (bundle-aware), league codes
  data.py       download + consolidate historical matches
  features.py   chronological feature builder and team-state store
  ratings.py    Elo, pi-ratings, online Poisson (goals and shots), league context
  tune_ratings.py  fits the rating systems' settings on pre-holdout data
  experiments.py   cross-validated comparison of every candidate change
  model.py      candidate estimators behind preprocessing pipelines
  train.py      CV, model comparison, calibration decision, persistence
  evaluate.py   metrics, baselines, calibration table
  predict.py    scoring fixtures from a saved bundle
  fixtures.py   upcoming fixtures + team-name resolution
  store.py      SQLite: followed teams, saved predictions, scorecard
  refresh.py    new results, settling, conditional retrain
  autopredict.py  season-wide forecasts and the 7-day lock-in rule
  smoke.py      loads the model on disk and scores a fixture, as the app does
  ordinal.py    proportional-odds model (an experiment; a tie, not shipped)
  benchmark.py  model against the betting market on identical fixtures
  monitor.py    drift detection against recent results
  api.py        FastAPI service
web/index.html  the app front-end
desktop/        Electron shell, icon, electron-builder config
tests/          204 tests
```

## Notes

Predictions are probabilistic forecasts, not tips. Roughly half of matches are called
correctly and about a quarter end in draws; the value is in the calibrated distribution,
not the headline pick. The model has no knowledge of injuries, suspensions, lineups or
motivation, which is most of the gap between it and the betting market.
