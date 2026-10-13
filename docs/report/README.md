# Project report figures

Original figures used in the CA-3 project report. Diagrams are Graphviz sources (`*.dot`, editable as text,
rendered to SVG/PNG with `dot -Tsvg file.dot -o file.svg`); charts are generated from the evaluation reports.

| Figure | File | Source |
|---|---|---|
| 1 System architecture | `figures/fig01_architecture.*` | Graphviz |
| 2 SciFact label distribution | `figures/fig02_dataset.png` | `make_charts.py` |
| 3 Pre-processing workflow | `figures/fig03_preprocessing.*` | Graphviz |
| 4 Text representations | `figures/fig04_representations.*` | Graphviz |
| 5 Model architecture | `figures/fig05_models.*` | Graphviz |
| 6 Proposed novelty | `figures/fig06_novelty.*` | Graphviz |
| 7 Pipeline comparison | `figures/fig07_pipelines.png` | `make_charts.py` |
| 8 Per-class F1 | `figures/fig08_perclass.png` | `make_charts.py` |
| 9 Confusion matrix (verisci) | `figures/fig09_confusion.png` | `make_charts.py` |
| 10 Retrieval Recall@K | `figures/fig10_retrieval.png` | `make_charts.py` + `retrieval_dev.json` |
| 11 Stance probes | `figures/fig11_probes.png` | `make_charts.py` |

Regenerate the charts from the repository root with `python docs/report/make_charts.py` (needs matplotlib).
