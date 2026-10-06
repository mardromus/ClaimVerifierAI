"""Command-line interface: ``python -m claimverifier <command> [options]``.

Typical workflow::

    python -m claimverifier setup    --config configs/default.yaml   # download SciFact, build index, train lite models
    python -m claimverifier train-rationale --config configs/default.yaml   # fine-tune SciBERT (GPU recommended)
    python -m claimverifier train-nli       --config configs/default.yaml   # fine-tune DeBERTa-v3 on SciFact
    python -m claimverifier verify   --config configs/default.yaml "Vitamin D supplementation reduces fractures."
    python -m claimverifier evaluate --config configs/default.yaml --split dev
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import textwrap
from pathlib import Path

from .config import Config, load_config, resolve_device

logger = logging.getLogger("claimverifier")


def _cfg(args) -> Config:
    return load_config(args.config, args.set)


# ---------------------------------------------------------------------------- commands


def cmd_download(args):
    from .data.scifact import dataset_statistics, download_scifact

    cfg = _cfg(args)
    download_scifact(cfg.data.data_dir, force=args.force)
    print(json.dumps(dataset_statistics(cfg.data.data_dir), indent=2))


def cmd_stats(args):
    from .data.scifact import dataset_statistics

    print(json.dumps(dataset_statistics(_cfg(args).data.data_dir), indent=2))


def cmd_index(args):
    from .data.scifact import load_corpus
    from .retrieval.retriever import Retriever

    cfg = _cfg(args)
    corpus = load_corpus(cfg.data.data_dir, cfg.data.corpus_path)
    Retriever.build(cfg.retrieval, corpus, cfg.artifacts_dir, resolve_device(cfg.device), cfg.seed)
    cfg.save(Path(cfg.artifacts_dir) / "config.yaml")
    print(f"Index built for {len(corpus)} documents in {Retriever.index_dir(cfg.artifacts_dir)}")


def _load_retriever(cfg):
    from .data.scifact import load_corpus
    from .retrieval.retriever import Retriever

    corpus = load_corpus(cfg.data.data_dir, cfg.data.corpus_path)
    device = resolve_device(cfg.device)
    if Retriever.exists(cfg.artifacts_dir):
        return corpus, Retriever.load(cfg.retrieval, corpus, cfg.artifacts_dir, device)
    return corpus, Retriever.build(cfg.retrieval, corpus, cfg.artifacts_dir, device, cfg.seed)


def cmd_train_lite(args):
    from .data.scifact import load_claims
    from .training.lite import train_lite_models

    cfg = _cfg(args)
    corpus, retriever = _load_retriever(cfg)
    report = train_lite_models(cfg, retriever, corpus, load_claims(cfg.data.data_dir, "train"),
                               load_claims(cfg.data.data_dir, "dev"))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "coefficients"} for k, v in report.items()},
                     indent=2))


def cmd_setup(args):
    from .data.scifact import download_scifact

    cfg = _cfg(args)
    download_scifact(cfg.data.data_dir)
    cmd_index(args)
    cmd_train_lite(args)
    args.split, args.limit = "train", None
    cmd_calibrate(args)
    print("Setup complete. Try: python -m claimverifier verify --config", args.config or "<config>", '"<claim>"')


def _training_args(args, defaults):
    from .training.transformer_trainer import TrainingArgs

    ta = TrainingArgs(**defaults)
    for name in ("epochs", "learning_rate", "batch_size", "grad_accum", "max_length", "max_steps", "seed"):
        value = getattr(args, name, None)
        if value is not None:
            setattr(ta, name, value)
    ta.device = args.device or "auto"
    return ta


def cmd_train_rationale(args):
    from .training.tasks import train_rationale_model

    cfg = _cfg(args)
    ta = _training_args(args, {"max_length": 128, "learning_rate": 2e-5, "epochs": 3, "batch_size": 16})
    info = train_rationale_model(cfg, args.base_model, args.output, ta, neg_ratio=args.neg_ratio,
                                 retrieval_negatives=args.retrieval_negatives)
    print(json.dumps({k: info[k] for k in ("best_epoch", "best_score", "history")}, indent=2))


def cmd_train_nli(args):
    from .training.tasks import train_nli_model

    cfg = _cfg(args)
    ta = _training_args(args, {"max_length": 256, "learning_rate": 1e-5, "epochs": 3, "batch_size": 8,
                               "grad_accum": 2})
    info = train_nli_model(cfg, args.base_model, args.output, ta, balanced=not args.no_balance)
    print(json.dumps({k: info[k] for k in ("best_epoch", "best_score", "history")}, indent=2))


def print_result(result, show_all: bool = False) -> None:
    from .pipeline import shown_documents

    width = 100
    print("=" * width)
    print(f"CLAIM: {result.claim}")
    print(f"VERDICT: {result.verdict_display.upper()}  (confidence {result.confidence:.1%})")
    print("Scores: " + ", ".join(f"{k}: {v:.3f}" for k, v in result.scores.items()))
    if result.mixed_evidence:
        print("Note: mixed evidence (strong support and contradiction).")
    print("-" * width)
    shown = shown_documents(result)
    docs = shown + [d for d in result.documents if d not in shown] if show_all else shown
    for d in docs:
        kind = "evidence" if d.has_evidence else ("partial evidence" if d in shown else "no evidence")
        print(f"[{d.citation}] {d.title}")
        print(f"     doc {d.doc_id} | {kind} (weight {d.evidence_weight:.2f}) | retrieval cos {d.dense_score:.3f} | "
              f"stance {d.stance} ({max(d.stance_probs.values()):.2f}) | {d.url}")
        for e in d.evidence:
            mark = {"SUPPORTED": "+", "CONTRADICTED": "-"}.get(e.stance, "~")
            for j, line in enumerate(textwrap.wrap(e.text, width - 12)):
                print(f"   {mark if j == 0 else ' '} {line}")
    if not shown:
        print("No abstract is relevant enough to count as evidence. Closest papers:")
        for d in result.documents[:3]:
            print(f"[{d.citation}] {d.title} (doc {d.doc_id}, relevance {d.relevance:.2f})")
    if result.explanation:
        print("-" * width)
        print(f"EXPLANATION ({result.explanation_backend}):")
        for para in result.explanation.split("\n"):
            print(textwrap.fill(para, width) if para.strip() else "")
    print("=" * width)


def cmd_verify(args):
    from .pipeline import ClaimVerifier

    cfg = _cfg(args)
    if args.no_explain:
        cfg.explanation.backend = "none"
    verifier = ClaimVerifier.from_config(cfg)
    claims = args.claims or [line.strip() for line in sys.stdin if line.strip()]
    for claim in claims:
        result = verifier.verify(claim, top_k=args.top_k, explain=not args.no_explain)
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print_result(result, show_all=args.show_all)


def cmd_evaluate(args):
    from .data.scifact import load_claims
    from .evaluation.evaluate import evaluate, markdown_report, save_report
    from .pipeline import ClaimVerifier

    cfg = _cfg(args)
    if not args.explain:
        cfg.explanation.backend = "none"
    verifier = ClaimVerifier.from_config(cfg)
    claims = load_claims(cfg.data.data_dir, args.split)
    if args.limit:
        claims = claims[: args.limit]
    result = evaluate(verifier, claims, top_k=args.top_k, explain=args.explain)
    out_dir = Path(args.out or Path(cfg.reports_dir) / f"{cfg.name}_{args.split}")
    title = f"ClaimVerifier AI - evaluation of '{cfg.name}' on SciFact {args.split}"
    save_report(result, out_dir, title)
    print(markdown_report(result["metrics"], title))
    print(f"\nSaved metrics, predictions and report to {out_dir}")


def cmd_calibrate(args):
    from .data.scifact import load_claims
    from .evaluation.calibrate import calibrate, save_calibration
    from .pipeline import ClaimVerifier

    cfg = _cfg(args)
    cfg.explanation.backend = "none"
    cfg.aggregation.use_calibration = False
    verifier = ClaimVerifier.from_config(cfg)
    claims = load_claims(cfg.data.data_dir, args.split)
    if args.limit:
        claims = claims[: args.limit]
    result = calibrate(verifier, claims)
    components = {"rationale": verifier.rationale.describe(), "nli": verifier.nli.describe()}
    path = save_calibration(cfg.artifacts_dir, result, args.split, components)
    default = next((g for g in result["grid"] if abs(g["threshold"] - 0.5) < 1e-9 and g["nei_weight"] == 1.0), None)
    print(json.dumps({"best": result["best"], "default_threshold_0.5_nei_1.0": default}, indent=2))
    print(f"Saved calibration to {path}")


def cmd_serve(args):
    import os

    import uvicorn

    if args.config:
        os.environ["CLAIMVERIFIER_CONFIG"] = args.config
    if args.set:
        os.environ["CLAIMVERIFIER_OVERRIDES"] = json.dumps(args.set)
    uvicorn.run("claimverifier.api:app", host=args.host, port=args.port, log_level="info")


# ---------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", "-c", default=None, help="YAML config (default: built-in defaults)")
    common.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config value, e.g. --set retrieval.top_k=10 (repeatable)")
    common.add_argument("--verbose", "-v", action="store_true")

    parser = argparse.ArgumentParser(prog="claimverifier", description="Scientific claim verification (SciFact).")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", parents=[common], help="download the SciFact dataset")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_download)

    sub.add_parser("stats", parents=[common], help="print dataset statistics").set_defaults(func=cmd_stats)
    sub.add_parser("index", parents=[common], help="embed the corpus and build the FAISS index") \
        .set_defaults(func=cmd_index)
    sub.add_parser("train-lite", parents=[common], help="train the lightweight rationale + NLI models") \
        .set_defaults(func=cmd_train_lite)
    sub.add_parser("setup", parents=[common], help="download + index + train-lite + calibrate") \
        .set_defaults(func=cmd_setup)

    for name, func, helptext in (("train-rationale", cmd_train_rationale, "fine-tune SciBERT rationale selector"),
                                 ("train-nli", cmd_train_nli, "fine-tune DeBERTa-v3 NLI on SciFact")):
        p = sub.add_parser(name, parents=[common], help=helptext)
        p.add_argument("--base-model", default=None, help="base checkpoint (default from config)")
        p.add_argument("--output", default=None, help="output directory (default from config)")
        p.add_argument("--epochs", type=int)
        p.add_argument("--learning-rate", "--lr", dest="learning_rate", type=float)
        p.add_argument("--batch-size", type=int)
        p.add_argument("--grad-accum", type=int)
        p.add_argument("--max-length", type=int)
        p.add_argument("--max-steps", type=int)
        p.add_argument("--seed", type=int)
        p.add_argument("--device", default=None)
        if name == "train-rationale":
            p.add_argument("--neg-ratio", type=float, default=None,
                           help="down-sample negatives to this many per positive (faster training)")
            p.add_argument("--retrieval-negatives", type=int, default=0, metavar="K",
                           help="add each claim's top-K retrieved abstracts as hard negatives")
        else:
            p.add_argument("--no-balance", action="store_true", help="disable class-balanced loss")
        p.set_defaults(func=func)

    p = sub.add_parser("verify", parents=[common], help="verify one or more claims (args or stdin)")
    p.add_argument("claims", nargs="*")
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--no-explain", action="store_true")
    p.add_argument("--json", action="store_true")
    p.add_argument("--show-all", action="store_true", help="also list retrieved papers without evidence")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("evaluate", parents=[common], help="evaluate on a labelled SciFact split")
    p.add_argument("--split", default="dev", choices=["train", "dev"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--explain", action="store_true", help="also generate explanations (slow)")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("calibrate", parents=[common],
                       help="tune the evidence threshold and NEI weight on a labelled split (default: train)")
    p.add_argument("--split", default="train", choices=["train", "dev"])
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("serve", parents=[common], help="start the REST API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "urllib3", "sentence_transformers", "filelock", "faiss"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    args.func(args)


if __name__ == "__main__":
    main()
