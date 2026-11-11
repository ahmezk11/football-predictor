VENV := ./.venv/bin

.PHONY: install data data-odds train train-odds serve test smoke selftest golden monitor refresh app app-dev dist clean

install:                 ## venv + python deps
	python3 -m venv .venv && $(VENV)/pip install -U pip && $(VENV)/pip install -r requirements.txt

data:                    ## download ~37k matches, 5 leagues, 21 seasons
	$(VENV)/python -m src.data --source mirror --from-year 2005 --to-year 2025

data-odds:               ## also pull bookmaker odds (football-data.co.uk rate-limits)
	$(VENV)/python -m src.data --source both --from-year 2005 --to-year 2025

train:
	$(VENV)/python -m src.train --cv-splits 5

train-odds:              ## needs `make data-odds` first
	$(VENV)/python -m src.train --use-odds --cv-splits 5 --out model_odds.joblib

serve:                   ## API + web UI in a browser
	$(VENV)/uvicorn src.api:app --reload --port 8000

test:
	$(VENV)/python -m pytest tests/ -q

smoke:                   ## load the model on disk and score a fixture, as the app does
	$(VENV)/python -m src.smoke

golden:                  ## rewrite the pinned feature and prediction values
	$(VENV)/python -m tests.test_golden --update

monitor:                 ## drift report against recent results
	$(VENV)/python -m src.monitor --window-days 365

refresh:                 ## new results, settle predictions, retrain if drifted
	$(VENV)/python -m src.refresh

# --- desktop app ---
app-dev:                 ## run the desktop app against this checkout
	cd desktop && npm install && npm start

backend-bin:             ## freeze the Python server into a standalone binary
	# Start clean: PyInstaller leaves the old tree in place, and on a synced
	# folder repeated builds accumulate "predictor-server 2" conflict copies that
	# are then packaged into the app.
	rm -rf dist/pyi build/pyi
	$(VENV)/pyinstaller predictor-server.spec --distpath dist/pyi --workpath build/pyi --noconfirm

app: backend-bin selftest  ## build and ad-hoc sign Match Predictor.app + .dmg
	cd desktop && npm install && npx electron-builder --mac
	# Sign, then rebuild the disk image from the signed bundle: electron-builder
	# assembles the first one during packaging, before signing. The image is built
	# from the staged copy on local disk, which carries no synced-folder metadata.
	staged=$$(./scripts-sign.sh "dist/desktop/mac-arm64/Match Predictor.app" --stage) && \
	  cd desktop && npx electron-builder --mac dmg --prepackaged "$$staged" && rm -rf "$$staged"

selftest:                ## train inside the packaged backend (catches over-trimming)
	# A writable directory of its own: a frozen build with nowhere else to write
	# creates data/ models/ reports/ beside the executable, which then ship.
	PREDICTOR_DATA_DIR=$$(mktemp -d) ./dist/pyi/predictor-server/predictor-server --selftest

clean:
	rm -rf build dist models/*.joblib reports/*.json data/processed/*.csv
