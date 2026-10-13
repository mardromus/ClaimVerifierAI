"""Charts for the CA-3 project report, generated from the evaluation reports in reports/*/metrics.json.

Retrieval curves come from docs/report/retrieval_dev.json (`python -m claimverifier evaluate --retrieval-only`
with the dense/BM25 weights listed there); stance-probe counts from `python -m claimverifier probe`.
Run from the repository root:  python docs/report/make_charts.py
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "report" / "figures"
RETR = json.load(open(REPO / "docs" / "report" / "retrieval_dev.json"))

for f in font_manager.findSystemFonts():
    if "Inter" in f:
        font_manager.fontManager.addfont(f)
plt.rcParams.update({
    "font.family": "Inter", "font.size": 10.5, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#52514e",
    "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e8e7e3", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 100, "savefig.dpi": 220, "savefig.bbox": "tight",
    "axes.titlesize": 11.5, "axes.titleweight": "semibold", "axes.titlecolor": "#0b0b0b",
})
INK, INK2 = "#0b0b0b", "#52514e"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]

# Entity -> fixed colour (colour follows the pipeline, never its rank)
PIPES = [
    ("lite_dev", "Lite (feature LR + lexical stance)", SLOTS[0]),
    ("scibert-rationale+lite-nli_dev", "SciBERT evidence + lexical stance", SLOTS[1]),
    ("scibert_dev", "SciBERT evidence + SciBERT verifier", SLOTS[2]),
    ("verisci_dev", "SciBERT evidence + RoBERTa-large verifier", SLOTS[3]),
    ("default_dev", "SBERT + SciBERT + DeBERTa-v3 (default)", SLOTS[4]),
]
LAB = ["SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"]
LABN = ["Supported", "Contradicted", "Insufficient Evidence"]


def load():
    out = []
    for key, name, col in PIPES:
        p = REPO / "reports" / key / "metrics.json"
        if p.exists():
            out.append((key, name, col, json.load(open(p))))
    return out


def bar_labels(ax, bars, fmt="{:.1f}", size=8.5):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 0.8, fmt.format(h), ha="center", va="bottom", fontsize=size,
                color=INK2)


def legend_below(ax, n):
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=min(n, 2), fontsize=9.5, handlelength=1.2,
              columnspacing=1.5)


def grouped(ax, groups, series, width_total=0.84, fmt="{:.1f}"):
    n = len(series)
    w = width_total / n
    x = np.arange(len(groups))
    for i, (name, col, vals) in enumerate(series):
        bars = ax.bar(x - width_total / 2 + w * (i + 0.5), vals, w - 0.02, color=col, label=name,
                      edgecolor="white", linewidth=1.2)
        bar_labels(ax, bars, fmt)
    ax.set_xticks(x, groups)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", length=0)


def fig_dataset():
    stats = {"train": [332, 173, 304], "dev": [124, 64, 112]}
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    grouped(ax, LABN, [("Train split (809 claims)", SLOTS[0], stats["train"]),
                       ("Dev split (300 claims)", SLOTS[1], stats["dev"])], 0.7, fmt="{:.0f}")
    for c in ax.containers:
        pass
    ax.set_ylabel("Number of claims")
    ax.set_ylim(0, 380)
    legend_below(ax, 2)
    fig.savefig(OUT / "fig02_dataset.png")
    plt.close(fig)


def fig_pipelines(runs):
    groups = ["Accuracy", "Macro F1", "SciFact abstract F1\n(label only)"]
    series = []
    for key, name, col, m in runs:
        c = m["verdict_classification"]
        series.append((name, col, [100 * c["accuracy"], 100 * c["macro_f1"],
                                   100 * m["scifact_abstract_level"]["label_only"]["f1"]]))
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    grouped(ax, groups, series)
    ax.plot([-0.46, 0.46], [41.3, 41.3], color=INK2, lw=1.2, ls=(0, (4, 3)))
    ax.plot([0.54, 1.46], [19.5, 19.5], color=INK2, lw=1.2, ls=(0, (4, 3)))
    ax.plot([], [], color=INK2, lw=1.2, ls=(0, (4, 3)), label="Majority-class baseline (always Supported)")
    ax.set_ylabel("Score on SciFact dev (%)")
    ax.set_ylim(0, 85)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=9.5, handlelength=1.6, columnspacing=1.5)
    fig.savefig(OUT / "fig07_pipelines.png")
    plt.close(fig)


def fig_perclass(runs):
    series = [(name, col, [100 * m["verdict_classification"]["per_class"][l]["f1"] for l in LAB])
              for key, name, col, m in runs]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    grouped(ax, LABN, series)
    ax.set_ylabel("F1 score (%)")
    ax.set_ylim(0, 90)
    legend_below(ax, len(series))
    fig.savefig(OUT / "fig08_perclass.png")
    plt.close(fig)


def fig_confusion(runs, key="verisci_dev", name="fig09_confusion.png"):
    m = dict((k, mm) for k, _, _, mm in runs)[key]
    cm = np.array(m["verdict_classification"]["confusion_matrix"]["matrix"], dtype=float)
    rowp = cm / cm.sum(1, keepdims=True)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue", ["#f4f8fe", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#104281"])
    fig, ax = plt.subplots(figsize=(5.2, 4.3))
    ax.imshow(rowp, cmap=cmap, vmin=0, vmax=1)
    ax.grid(False)
    for i in range(3):
        for j in range(3):
            dark = rowp[i, j] > 0.5
            ax.text(j, i - 0.08, f"{int(cm[i, j])}", ha="center", va="center", fontsize=14, fontweight="semibold",
                    color="white" if dark else INK)
            ax.text(j, i + 0.22, f"{100 * rowp[i, j]:.0f}% of row", ha="center", va="center", fontsize=8.5,
                    color="white" if dark else INK2)
    short = ["Supported", "Contradicted", "Insufficient"]
    ax.set_xticks(range(3), short)
    ax.set_yticks(range(3), short)
    ax.set_xlabel("Predicted verdict")
    ax.set_ylabel("Gold label")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.savefig(OUT / name)
    plt.close(fig)


def fig_retrieval():
    ks = [1, 3, 5, 10, 20]
    series = [("LSA dense vectors (FAISS)", SLOTS[0], RETR["lsa"]),
              ("LSA + BM25 (reciprocal-rank fusion)", SLOTS[1], RETR["lsa_bm25_rrf"]),
              ("BM25 (sparse)", SLOTS[2], RETR["bm25"])]
    if "sbert_bm25_rrf" in RETR:
        series.append(("Sentence-BERT + BM25 (fusion)", SLOTS[3], RETR["sbert_bm25_rrf"]))
    if "sbert" in RETR:
        series.append(("Sentence-BERT dense (FAISS)", SLOTS[4], RETR["sbert"]))
    fig, ax = plt.subplots(figsize=(7.0, 3.9))
    for name, col, r in series:
        ys = [100 * r[f"recall@{k}"] for k in ks]
        ax.plot(range(len(ks)), ys, color=col, lw=2, marker="o", ms=6, mec="white", mew=1.5, label=name)
    ax.set_xticks(range(len(ks)), [f"@{k}" for k in ks])
    ax.set_xlim(-0.25, len(ks) - 0.75)
    ax.set_ylim(30, 100)
    ax.set_xlabel("Number of retrieved abstracts (K)")
    ax.set_ylabel("Recall@K (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=9.5)
    fig.savefig(OUT / "fig10_retrieval.png")
    plt.close(fig)


def fig_probes(probes):
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    names = [p[0] for p in probes][::-1]
    vals = [p[1] for p in probes][::-1]
    cols = [p[2] for p in probes][::-1]
    bars = ax.barh(range(len(vals)), vals, color=cols, height=0.62, edgecolor="white")
    for i, v in enumerate(vals):
        ax.text(v + 0.3, i, f"{v} / 24  ({100 * v / 24:.0f}%)", va="center", fontsize=9, color=INK2)
    ax.set_yticks(range(len(vals)), names)
    ax.set_xlim(0, 28.5)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_xlabel("Hand-written stance probes classified correctly (out of 24)")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    fig.savefig(OUT / "fig11_probes.png")
    plt.close(fig)


if __name__ == "__main__":
    runs = load()
    print("reports:", [r[0] for r in runs])
    fig_dataset()
    fig_pipelines(runs)
    fig_perclass(runs)
    fig_confusion(runs)
    fig_retrieval()
    probes = json.loads(sys.argv[1]) if len(sys.argv) > 1 else [
        ["Lexical stance model (lite)", 20, SLOTS[0]],
        ["SciBERT verifier (SciFact only)", 14, SLOTS[2]],
        ["SciBERT verifier + augmentation", 10, "#6250d6"],
        ["RoBERTa-large verifier (FEVER + SciFact)", 24, SLOTS[3]],
    ]
    fig_probes(probes)
    print("done")
