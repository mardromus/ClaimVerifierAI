CONFIG ?= configs/default.yaml
PY ?= python

.PHONY: install install-lite setup setup-lite train-rationale train-nli calibrate evaluate ui api test

install:            ## full stack (Sentence-BERT, SciBERT, DeBERTa-v3, Qwen/Llama)
	$(PY) -m pip install -r requirements.txt
install-lite:       ## offline stack (no PyTorch)
	$(PY) -m pip install -r requirements-lite.txt

setup:              ## download SciFact, build FAISS index, train lightweight fallbacks, calibrate
	$(PY) -m claimverifier setup --config $(CONFIG)
setup-lite:
	$(PY) -m claimverifier setup --config configs/lite.yaml

train-rationale:    ## fine-tune SciBERT rationale selector (GPU recommended)
	$(PY) -m claimverifier train-rationale --config $(CONFIG) --neg-ratio 4 --retrieval-negatives 3
train-nli:          ## fine-tune DeBERTa-v3 NLI on SciFact (GPU recommended)
	$(PY) -m claimverifier train-nli --config $(CONFIG) --retrieval-negatives 3
calibrate:
	$(PY) -m claimverifier calibrate --config $(CONFIG) --split train
evaluate:
	$(PY) -m claimverifier evaluate --config $(CONFIG) --split dev

ui:
	streamlit run app/streamlit_app.py
api:
	$(PY) -m claimverifier serve --config $(CONFIG) --host 0.0.0.0 --port 8000
test:
	$(PY) -m pytest -q
