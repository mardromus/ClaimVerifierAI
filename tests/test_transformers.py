"""Transformer code paths (SciBERT selector, DeBERTa-style NLI, Sentence-BERT, chat LLM, fine-tuning),
exercised with tiny randomly initialised models built locally - no downloads needed."""

import json
import re
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from claimverifier.labels import CONTRADICTED, INSUFFICIENT, LABEL2ID, SUPPORTED  # noqa: E402
from claimverifier.training.transformer_trainer import TrainingArgs  # noqa: E402

FIXTURE_DATA = Path(__file__).parent / "fixtures" / "scifact_mini"

PAIRS = [("Aspirin reduced colorectal cancer risk.", "Aspirin prevents colorectal cancer."),
         ("Sleep deprivation impaired recall.", "Sleep loss improves memory."),
         ("Coffee was associated with lower risk.", "Coffee lowers Parkinson disease risk.")]


def _vocab():
    words = set()
    for line in (FIXTURE_DATA / "corpus.jsonl").read_text().splitlines() + \
            (FIXTURE_DATA / "claims_train.jsonl").read_text().splitlines():
        words.update(re.findall(r"[a-z]+|\d+|[^\sa-z\d]", line.lower()))
    return ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + sorted(words)


@pytest.fixture(scope="module")
def tiny_tokenizer_dir(tmp_path_factory):
    from transformers import BertTokenizer

    d = tmp_path_factory.mktemp("tiny_tok")
    (d / "vocab.txt").write_text("\n".join(_vocab()))
    BertTokenizer(vocab_file=str(d / "vocab.txt"), do_lower_case=True).save_pretrained(d)
    return d


def make_bert(tmp_path_factory, tok_dir, labels):
    from transformers import AutoTokenizer, BertConfig, BertForSequenceClassification

    d = tmp_path_factory.mktemp("tiny_bert")
    tok = AutoTokenizer.from_pretrained(tok_dir)
    config = BertConfig(vocab_size=len(tok), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                        intermediate_size=64, max_position_embeddings=256, num_labels=len(labels),
                        id2label=dict(enumerate(labels)), label2id={lab: i for i, lab in enumerate(labels)})
    torch.manual_seed(0)
    BertForSequenceClassification(config).save_pretrained(d)
    tok.save_pretrained(d)
    return d


@pytest.fixture(scope="module")
def nli_dir(tmp_path_factory, tiny_tokenizer_dir):
    return make_bert(tmp_path_factory, tiny_tokenizer_dir, ["entailment", "neutral", "contradiction"])


@pytest.fixture(scope="module")
def rationale_dir(tmp_path_factory, tiny_tokenizer_dir):
    return make_bert(tmp_path_factory, tiny_tokenizer_dir, ["NOT_RATIONALE", "RATIONALE"])


def test_transformer_nli_maps_labels(nli_dir):
    from claimverifier.hf_utils import load_sequence_classifier, predict_pair_probs
    from claimverifier.nli import TransformerNLI

    nli = TransformerNLI(str(nli_dir), batch_size=2, max_length=64)
    probs = nli.predict(PAIRS)
    assert probs.shape == (3, 3) and np.allclose(probs.sum(1), 1.0, atol=1e-5)
    tok, model = load_sequence_classifier(str(nli_dir))
    raw = predict_pair_probs(model, tok, PAIRS, batch_size=3, max_length=64)  # entailment, neutral, contradiction
    assert np.allclose(probs[:, LABEL2ID[SUPPORTED]], raw[:, 0], atol=1e-5)
    assert np.allclose(probs[:, LABEL2ID[INSUFFICIENT]], raw[:, 1], atol=1e-5)
    assert np.allclose(probs[:, LABEL2ID[CONTRADICTED]], raw[:, 2], atol=1e-5)


def test_two_way_nli_head(tmp_path_factory, tiny_tokenizer_dir):
    from claimverifier.nli import TransformerNLI

    d = make_bert(tmp_path_factory, tiny_tokenizer_dir, ["entailment", "not_entailment"])
    probs = TransformerNLI(str(d)).predict(PAIRS)
    assert np.all(probs[:, LABEL2ID[CONTRADICTED]] == 0) and np.allclose(probs.sum(1), 1.0, atol=1e-5)


def test_scibert_selector(rationale_dir):
    from claimverifier.rationale import SciBertRationaleSelector

    sel = SciBertRationaleSelector(str(rationale_dir), batch_size=4, max_length=64)
    scores = sel.score_batch([("Aspirin prevents cancer.", ["A first sentence.", "Aspirin reduced risk.", "x"]),
                              ("Coffee lowers risk.", ["Coffee intake was examined."])])
    assert [len(s) for s in scores] == [3, 1]
    assert all(((0 <= s) & (s <= 1)).all() for s in scores)


def test_finetune_new_head_and_kept_nli_head(tmp_path, rationale_dir, nli_dir, lite_cfg):
    from claimverifier.training.tasks import nli_head_mapping, train_nli_model, train_rationale_model

    args = TrainingArgs(epochs=1, batch_size=4, max_length=64, max_steps=2, log_every=1, device="cpu")
    # rationale model trained from a checkpoint with a different head, using retrieval negatives
    info = train_rationale_model(lite_cfg, base_model=str(rationale_dir), output_dir=str(tmp_path / "rat"),
                                 args=args, retrieval_negatives=2)
    assert (tmp_path / "rat" / "config.json").exists() and info["history"][0]["epoch"] == 1
    assert "f1" in info["history"][0]
    # NLI: the entailment/neutral/contradiction head is kept and its label names preserved
    assert nli_head_mapping(str(nli_dir)) == {SUPPORTED: 0, INSUFFICIENT: 1, CONTRADICTED: 2}
    info = train_nli_model(lite_cfg, base_model=str(nli_dir), output_dir=str(tmp_path / "nli"), args=args)
    saved = json.loads((tmp_path / "nli" / "config.json").read_text())
    assert saved["id2label"]["0"] == "entailment" and "macro_f1" in info["history"][0]
    # a base model without an NLI head gets a new canonical 3-way head
    assert nli_head_mapping(str(rationale_dir)) is None
    train_nli_model(lite_cfg, base_model=str(rationale_dir), output_dir=str(tmp_path / "nli2"), args=args)
    saved = json.loads((tmp_path / "nli2" / "config.json").read_text())
    assert set(saved["id2label"].values()) == {SUPPORTED, CONTRADICTED, INSUFFICIENT}


def test_full_pipeline_with_transformer_components(lite_verifier, nli_dir, rationale_dir):
    from claimverifier.nli import TransformerNLI
    from claimverifier.pipeline import ClaimVerifier
    from claimverifier.rationale import SciBertRationaleSelector

    v = ClaimVerifier(lite_verifier.cfg, lite_verifier.corpus, lite_verifier.retriever,
                      SciBertRationaleSelector(str(rationale_dir), max_length=64),
                      TransformerNLI(str(nli_dir), max_length=64), lite_verifier.explainer)
    r = v.verify("Aspirin reduces colorectal cancer risk.", top_k=2)
    assert r.documents[0].doc_id == 101 and abs(sum(r.scores.values()) - 1) < 1e-6
    assert v.components()["rationale"]["method"] == "scibert"


def test_create_selector_and_nli_fallbacks(lite_verifier, tmp_path):
    import copy

    from claimverifier.nli import create_nli_model
    from claimverifier.rationale import create_rationale_selector

    cfg = copy.deepcopy(lite_verifier.cfg)
    cfg.rationale.method, cfg.rationale.model_path = "scibert", str(tmp_path / "missing")
    cfg.rationale.fallbacks = ["features", "similarity"]
    sel = create_rationale_selector(cfg, lite_verifier.retriever.embedder, cfg.artifacts_dir)
    assert sel.method == "features"
    cfg.nli.method, cfg.nli.model_name, cfg.nli.fallbacks = "transformer", str(tmp_path / "missing"), ["lite"]
    cfg.nli.finetuned_path = str(tmp_path / "also-missing")
    assert create_nli_model(cfg, cfg.artifacts_dir).method == "lite"
    cfg.nli.fallbacks = []
    with pytest.raises(RuntimeError, match="No NLI model available"):
        create_nli_model(cfg, cfg.artifacts_dir)


def test_sentence_bert_embedder(rationale_dir):
    pytest.importorskip("sentence_transformers")
    from claimverifier.retrieval.embedders import SentenceBertEmbedder

    emb = SentenceBertEmbedder(str(rationale_dir), max_seq_length=64)
    x = emb.encode(["aspirin colorectal cancer", "sleep memory"])
    assert x.shape == (2, 32) and np.allclose(np.linalg.norm(x, axis=1), 1.0, atol=1e-5)


def test_transformers_chat_llm(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    from claimverifier.explain import EvidenceForPrompt, TransformersChatLLM, build_messages

    vocab = {w: i for i, w in enumerate(["<unk>", "<pad>", "</s>"] + _vocab()[5:] + ["system:", "user:", "assistant:"])}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", pad_token="<pad>", eos_token="</s>")
    tok.chat_template = ("{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}"
                         "{% if add_generation_prompt %}assistant:{% endif %}")
    tok.save_pretrained(tmp_path)
    torch.manual_seed(0)
    GPT2LMHeadModel(GPT2Config(vocab_size=len(vocab), n_embd=32, n_layer=2, n_head=2, n_positions=1024)) \
        .save_pretrained(tmp_path)
    llm = TransformersChatLLM(str(tmp_path))
    msgs = build_messages("Aspirin prevents cancer.", SUPPORTED, 0.9,
                          [EvidenceForPrompt(1, "Aspirin study", SUPPORTED, 0.9, ["Aspirin reduced risk."])])
    out = llm.chat(msgs, max_new_tokens=5, temperature=0.0)
    assert isinstance(out, str)


def test_nli_fallback_models_and_probes(tmp_path, nli_dir, lite_verifier):
    import copy

    from claimverifier.evaluation.probes import load_probes, run_probes
    from claimverifier.nli import create_nli_model

    cfg = copy.deepcopy(lite_verifier.cfg)
    cfg.nli.method, cfg.nli.model_name, cfg.nli.finetuned_path = "transformer", str(tmp_path / "missing"), ""
    cfg.nli.fallback_models = [str(tmp_path / "also-missing"), str(nli_dir)]
    model = create_nli_model(cfg, cfg.artifacts_dir)
    assert model.describe()["model"] == str(nli_dir)
    probes = load_probes()
    assert len(probes) == 24 and sum(p["label"] == "SUPPORTED" for p in probes) == 12
    out = run_probes(model, probes)
    assert out["total"] == 24 and 0 <= out["accuracy"] <= 1


@pytest.mark.skipif(not (Path(__file__).resolve().parents[1] / "models" / "verisci-roberta-large" / "config.json").exists(),
                    reason="run `python -m claimverifier fetch-verifier` to enable this model-quality test")
def test_verisci_passes_stance_probes():
    from claimverifier.evaluation.probes import run_probes
    from claimverifier.nli import TransformerNLI

    model = TransformerNLI(str(Path(__file__).resolve().parents[1] / "models" / "verisci-roberta-large"), batch_size=8)
    assert run_probes(model)["accuracy"] >= 0.9
