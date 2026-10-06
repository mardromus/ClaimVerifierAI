"""ClaimVerifier AI - Streamlit web interface.

Run:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import html
import json
import os
import re
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # config paths (data/, artifacts/, models/) are relative to the repository root

from claimverifier.config import load_config  # noqa: E402
from claimverifier.data.scifact import Document, load_claims  # noqa: E402
from claimverifier.labels import CONTRADICTED, DISPLAY_NAMES, INSUFFICIENT, LABELS, SUPPORTED  # noqa: E402
from claimverifier.pipeline import MIN_SHOWN_WEIGHT  # noqa: E402
from claimverifier.text import split_sentences  # noqa: E402

st.set_page_config(page_title="ClaimVerifier AI", page_icon="🔬", layout="wide")

COLORS = {SUPPORTED: "#1a7f37", CONTRADICTED: "#c62828", INSUFFICIENT: "#b26a00"}
HIGHLIGHT = {SUPPORTED: "rgba(26,127,55,0.18)", CONTRADICTED: "rgba(198,40,40,0.18)",
             INSUFFICIENT: "rgba(178,106,0,0.16)"}
STANCE_TEXT = {SUPPORTED: "supports", CONTRADICTED: "contradicts", INSUFFICIENT: "neutral"}

st.markdown("""
<style>
.cv-verdict {border-radius: 12px; padding: 18px 22px; color: white; margin-bottom: 8px;}
.cv-verdict h2 {margin: 0; color: white; font-size: 1.9rem;}
.cv-verdict p {margin: 4px 0 0 0; opacity: 0.95;}
.cv-bar {height: 10px; border-radius: 5px; background: rgba(128,128,128,0.2); overflow: hidden;}
.cv-bar > div {height: 100%;}
.cv-row {display: flex; align-items: center; gap: 10px; margin: 6px 0; font-size: 0.92rem;}
.cv-row .lbl {width: 170px;} .cv-row .val {width: 56px; text-align: right; font-variant-numeric: tabular-nums;}
.cv-row .cv-bar {flex: 1;}
.cv-abstract {line-height: 1.65; font-size: 0.95rem;}
.cv-abstract mark {padding: 2px 3px; border-radius: 4px; color: inherit;}
.cv-chip {display: inline-block; padding: 1px 9px; border-radius: 10px; font-size: 0.8rem; color: white;}
.cv-meta {font-size: 0.82rem; opacity: 0.75;}
.cv-expl {border-left: 4px solid rgba(128,128,128,0.5); padding: 6px 14px; margin: 6px 0 14px 0;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------- loading


def available_configs():
    return sorted(str(p.relative_to(ROOT)) for p in (ROOT / "configs").glob("*.yaml"))


@st.cache_resource(show_spinner="Loading models and index (first run can take a while)...")
def get_verifier(config_path: str, overrides: tuple):
    from claimverifier.pipeline import ClaimVerifier

    return ClaimVerifier.from_config(load_config(config_path, list(overrides)))


@st.cache_data
def example_claims(data_dir: str):
    try:
        claims = load_claims(data_dir, "dev")
    except FileNotFoundError:
        return []
    return [(c.claim, c.label) for c in claims]


# ---------------------------------------------------------------------------- rendering


def score_bars(scores: dict) -> str:
    rows = []
    for lab in LABELS:
        v = scores[lab]
        rows.append(f'<div class="cv-row"><span class="lbl">{DISPLAY_NAMES[lab]}</span>'
                    f'<div class="cv-bar"><div style="width:{100 * v:.1f}%;background:{COLORS[lab]}"></div></div>'
                    f'<span class="val">{100 * v:.1f}%</span></div>')
    return "".join(rows)


def link_citations(text: str, docs: dict) -> str:
    text = html.escape(text).replace("\n", "<br>")

    def repl(m):
        n = int(m.group(1))
        d = docs.get(n)
        if d and d["url"]:
            return f'<a href="{d["url"]}" target="_blank" title="{html.escape(d["title"])}">[{n}]</a>'
        return f'<b title="{html.escape(d["title"]) if d else ""}">[{n}]</b>'

    return re.sub(r"\[(\d+)\]", repl, text)


def is_shown(doc: dict) -> bool:
    return doc["evidence_weight"] >= MIN_SHOWN_WEIGHT


def strength(doc: dict) -> float:
    return doc["evidence_weight"] * max(doc["stance_probs"][SUPPORTED], doc["stance_probs"][CONTRADICTED])


def render_abstract(doc: dict) -> str:
    evidence = {e["index"]: e for e in doc["evidence"]}
    parts = []
    for i, sent in enumerate(doc["sentences"]):
        s = html.escape(sent)
        if i in evidence and is_shown(doc):
            e = evidence[i]
            tip = (f"rationale score {e['rationale_score']:.2f} | "
                   f"{STANCE_TEXT[e['stance']]} ({e['stance_probs'][e['stance']]:.2f})")
            line = "solid" if doc["has_evidence"] else "dashed"
            parts.append(f'<mark style="background:{HIGHLIGHT[e["stance"]]};'
                         f'border-bottom:2px {line} {COLORS[e["stance"]]}" title="{tip}">{s}</mark>')
        elif i in evidence:
            parts.append(f'<span style="text-decoration: underline dotted" '
                         f'title="closest sentence (below evidence threshold)">{s}</span>')
        else:
            parts.append(s)
    return f'<div class="cv-abstract">{" ".join(parts)}</div>'


def render_result(result: dict, show_all: bool):
    verdict = result["verdict"]
    st.markdown(
        f'<div class="cv-verdict" style="background:{COLORS[verdict]}">'
        f'<h2>{DISPLAY_NAMES[verdict]}</h2>'
        f'<p>Confidence {100 * result["confidence"]:.1f}%'
        f'{" · mixed evidence" if result["mixed_evidence"] else ""}</p></div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("**Confidence scores**")
        st.markdown(score_bars(result["scores"]), unsafe_allow_html=True)
    with col2:
        st.markdown("**Evidence strength**")
        st.markdown(
            f'<div class="cv-row"><span class="lbl">Strongest support</span><div class="cv-bar"><div style="width:'
            f'{100 * result["support_strength"]:.1f}%;background:{COLORS[SUPPORTED]}"></div></div>'
            f'<span class="val">{result["support_strength"]:.2f}</span></div>'
            f'<div class="cv-row"><span class="lbl">Strongest contradiction</span><div class="cv-bar"><div '
            f'style="width:{100 * result["contradict_strength"]:.1f}%;background:{COLORS[CONTRADICTED]}"></div>'
            f'</div><span class="val">{result["contradict_strength"]:.2f}</span></div>', unsafe_allow_html=True)
        t = result.get("timings_ms", {})
        st.markdown(f'<p class="cv-meta">Latency: ' + " · ".join(f"{k} {v:.0f} ms" for k, v in t.items()) + "</p>",
                    unsafe_allow_html=True)

    docs_by_citation = {d["citation"]: d for d in result["documents"]}
    if result.get("explanation"):
        st.markdown(f"**Explanation** <span class='cv-meta'>({html.escape(result['explanation_backend'])})</span>",
                    unsafe_allow_html=True)
        st.markdown(f'<div class="cv-expl">{link_citations(result["explanation"], docs_by_citation)}</div>',
                    unsafe_allow_html=True)

    docs = result["documents"]
    shown = sorted([d for d in docs if is_shown(d)], key=lambda d: -strength(d))
    st.markdown(f"**Retrieved papers** — {len(shown)} of {len(docs)} provide evidence for the verdict "
                f"(highlighted: <span style='color:{COLORS[SUPPORTED]}'>supports</span>, "
                f"<span style='color:{COLORS[CONTRADICTED]}'>contradicts</span>, "
                f"<span style='color:{COLORS[INSUFFICIENT]}'>neutral</span>; dashed = partial evidence)",
                unsafe_allow_html=True)
    ordered = shown + ([d for d in docs if not is_shown(d)] if show_all else [])
    for d in ordered:
        stance = d["stance"] if is_shown(d) else INSUFFICIENT
        if not is_shown(d):
            label = "no evidence"
        else:
            label = STANCE_TEXT[stance] + ("" if d["has_evidence"] else " (partial)")
        header = f"[{d['citation']}] {d['title']}  —  {label}"
        with st.expander(header, expanded=bool(shown) and d is shown[0]):
            chip = (f'<span class="cv-chip" style="background:{COLORS[stance]}">{label} '
                    f'{d["stance_probs"][stance]:.2f}</span>') if is_shown(d) else ""
            link = f' · <a href="{d["url"]}" target="_blank">Semantic Scholar</a>' if d["url"] else ""
            st.markdown(
                f'{chip} <span class="cv-meta">doc {d["doc_id"]} · retrieval rank {d["rank"]} · '
                f'cosine {d["dense_score"]:.3f} · BM25 {d["bm25_score"]:.1f} · rationale score '
                f'{d["relevance"]:.2f} · evidence weight {d["evidence_weight"]:.2f}{link}</span>',
                unsafe_allow_html=True)
            st.markdown(render_abstract(d), unsafe_allow_html=True)
    if not shown:
        st.info("None of the retrieved papers is relevant enough to count as evidence. Enable "
                "“Show papers without evidence” in the sidebar to inspect the closest papers.")
    with st.expander("Raw JSON"):
        st.code(json.dumps(result, indent=2)[:200000], language="json")


# ---------------------------------------------------------------------------- evaluation tab


def render_reports():
    reports = sorted((ROOT / "reports").glob("*/metrics.json"))
    if not reports:
        st.info("No evaluation reports yet. Run `python -m claimverifier evaluate --config <config>`.")
        return
    rows = []
    for path in reports:
        m = json.loads(path.read_text())
        c, r = m["verdict_classification"], m["evidence_retrieval"]
        a = m["scifact_abstract_level"]
        rows.append({"report": path.parent.name, "claims": m["num_claims"],
                     "accuracy": round(100 * c["accuracy"], 1), "macro P": round(100 * c["macro_precision"], 1),
                     "macro R": round(100 * c["macro_recall"], 1), "macro F1": round(100 * c["macro_f1"], 1),
                     "Recall@1": round(100 * r["recall@1"], 1), "Recall@5": round(100 * r["recall@5"], 1),
                     "Recall@20": round(100 * r["recall@20"], 1), "MRR": round(r["mrr"], 3),
                     "abstract F1 (label)": round(100 * a["label_only"]["f1"], 1)})
    st.dataframe(rows, use_container_width=True, hide_index=True)
    choice = st.selectbox("Report details", [p.parent.name for p in reports])
    report_md = ROOT / "reports" / choice / "report.md"
    if report_md.exists():
        st.markdown(report_md.read_text())


# ---------------------------------------------------------------------------- page


with st.sidebar:
    st.header("Settings")
    configs = available_configs()
    default_idx = configs.index("configs/default.yaml") if "configs/default.yaml" in configs else 0
    config_path = st.selectbox("Pipeline configuration", configs, index=default_idx,
                               help="configs/lite.yaml runs fully offline on CPU (no transformer downloads).")
    top_k = st.slider("Papers to retrieve (top-k)", 1, 10, 5)
    explain = st.toggle("Generate explanation", value=True)
    show_all = st.toggle("Show papers without evidence", value=False)

st.title("🔬 ClaimVerifier AI")
st.caption("Scientific claim verification with semantic retrieval (Sentence-BERT + FAISS), evidence selection "
           "(SciBERT), natural language inference (DeBERTa-v3) and LLM explanations (Llama 3 / Qwen 2.5) "
           "over 5,183 research abstracts from SciFact.")

try:
    verifier = get_verifier(config_path, ())
except Exception as e:  # noqa: BLE001
    st.error(f"Could not load the pipeline for `{config_path}`: {e}")
    st.markdown("Set it up first, e.g.\n```bash\npython -m claimverifier setup --config " + config_path + "\n```")
    st.stop()

with st.sidebar:
    st.subheader("Loaded components")
    st.json(verifier.components(), expanded=False)

tab_verify, tab_custom, tab_eval = st.tabs(["Verify a claim", "Verify against your own abstract", "Evaluation"])

with tab_verify:
    examples = example_claims(verifier.cfg.data.data_dir)
    options = ["(type your own claim)"] + [f"{c}  [gold: {DISPLAY_NAMES[lab]}]" for c, lab in examples[:150]]
    picked = st.selectbox("Example claims from the SciFact dev set", options)
    default_claim = examples[options.index(picked) - 1][0] if picked != options[0] else ""
    claim = st.text_area("Scientific claim", value=default_claim, height=90,
                         placeholder="e.g. Vitamin D supplementation reduces the risk of fractures in older adults.")
    if st.button("Verify claim", type="primary"):
        if len(claim.strip()) < 3:
            st.warning("Please enter a claim.")
        else:
            with st.spinner("Retrieving papers, selecting evidence and running NLI..."):
                result = verifier.verify(claim.strip(), top_k=top_k, explain=explain).to_dict()
            st.session_state["last_result"] = result
    if "last_result" in st.session_state:
        render_result(st.session_state["last_result"], show_all)

with tab_custom:
    st.markdown("Paste one or more abstracts (separate them with a line containing only `---`). "
                "The first line of each block is used as its title.")
    c_claim = st.text_input("Claim", key="custom_claim")
    c_text = st.text_area("Abstract(s)", height=220, key="custom_text")
    if st.button("Verify against these abstracts"):
        docs = []
        for i, block in enumerate(b.strip() for b in re.split(r"^\s*---\s*$", c_text, flags=re.M)):
            if not block:
                continue
            lines = block.splitlines()
            title, body = (lines[0], " ".join(lines[1:])) if len(lines) > 1 else (f"Abstract {i + 1}", lines[0])
            sentences = split_sentences(body)
            if sentences:
                docs.append(Document(doc_id=-(i + 1), title=title.strip(), sentences=sentences))
        if docs and len(c_claim.strip()) >= 3:
            with st.spinner("Analysing..."):
                st.session_state["custom_result"] = verifier.verify_against(c_claim.strip(), docs, explain).to_dict()
        else:
            st.warning("Please enter a claim and at least one abstract.")
    if "custom_result" in st.session_state:
        render_result(st.session_state["custom_result"], show_all=True)

with tab_eval:
    render_reports()
