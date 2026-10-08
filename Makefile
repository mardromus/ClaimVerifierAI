CONFIG ?= configs/default.yaml
PY ?= python

.PHONY: install install-lite setup setup-lite train-rationale train-nli calibrate evaluate serve web-install web-dev web-build test

install:            ## full stack (Sentence-BERT, SciBERT, DeBERTa-v3, Qwen/Llama)
	$(PY) -m pip install -r requirements.txt
install-lite:       ## offline stack (no PyTorch)
	$(PY) -m pip install -r requirements-lite.txt

setup:              ## download SciFact, build FAISS index, train lightweight fallbacks, calibrate
	$(PY) -m claimverifier setup --config $(CONFIG)
setup-lite:
	$(PY) -m claimverifier setup --config configs/lite.yaml

train-rationale:    ## fine-tune SciBERT rationale selector (GPU recommended)
	$(PY) -m claimverifier train-rationale --config $(CONFIG) --neg-ratio 4 --lr 3e-5
train-nli:          ## fine-tune the NLI verifier on SciFact (GPU recommended)
	$(PY) -m claimverifier train-nli --config $(CONFIG) --retrieval-negatives 2 --augment
calibrate:
	$(PY) -m claimverifier calibrate --config $(CONFIG) --split train
evaluate:
	$(PY) -m claimverifier evaluate --config $(CONFIG) --split dev

serve:              ## web UI + REST API on http://localhost:8000
	$(PY) -m claimverifier serve --config $(CONFIG) --host 0.0.0.0 --port 8000

web-install:
	cd web && npm ci
web-dev:            ## hot-reloading UI on :5173 (run `make serve` in another terminal)
	cd web && npm run dev
web-build:          ## rebuild claimverifier/web/dist
	cd web && npm run build

test:
	$(PY) -m pytest -q
	cd web && npm test
