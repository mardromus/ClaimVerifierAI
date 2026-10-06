import io
import tarfile

import pytest

from claimverifier.config import Config, apply_overrides, load_config
from claimverifier.data.scifact import _safe_members, dataset_statistics, load_claims, load_corpus
from claimverifier.labels import CONTRADICTED, INSUFFICIENT, SUPPORTED, normalize_model_label
from claimverifier.text import cue_counts, split_sentences, tokenize


def test_load_fixture(data_dir):
    corpus = load_corpus(data_dir)
    assert len(corpus) == 10 and corpus[0].doc_id == 101
    assert corpus[0].full_text.startswith("Aspirin use and colorectal cancer risk")
    assert corpus[0].url.endswith("CorpusID:101")
    train = load_claims(data_dir, "train")
    assert train[0].label == SUPPORTED and train[1].label == CONTRADICTED
    assert train[0].rationale_sentences(101) == {2}
    assert [c for c in train if c.id == 13][0].label == INSUFFICIENT
    test = load_claims(data_dir, "test")
    assert test[0].label is None and not test[0].labelled
    stats = dataset_statistics(data_dir)
    assert stats["corpus_documents"] == 10 and stats["dev_claims"] == 6


def test_missing_data_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_corpus(tmp_path)
    with pytest.raises(ValueError):
        load_claims(tmp_path, "validation")


def test_safe_members_rejects_path_traversal(tmp_path):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo("../evil.txt")
        info.size = 1
        tar.addfile(info, io.BytesIO(b"x"))
    buf.seek(0)
    with tarfile.open(fileobj=buf, mode="r:gz") as tar, pytest.raises(RuntimeError):
        list(_safe_members(tar))


def test_config_defaults_and_overrides(tmp_path):
    cfg = load_config(None, ["retrieval.top_k=7", "nli.method=lite", "rationale.fallbacks=[similarity]"])
    assert isinstance(cfg, Config)
    assert cfg.retrieval.top_k == 7 and cfg.nli.method == "lite"
    assert cfg.rationale.fallbacks == ["similarity"]
    assert cfg.explanation.model_name.startswith("Qwen/")
    with pytest.raises(ValueError):
        load_config(None, ["retrieval.not_a_field=1"])
    path = tmp_path / "c.yaml"
    cfg.save(path)
    assert load_config(path).retrieval.top_k == 7
    assert apply_overrides({}, ["a.b=1.5"]) == {"a": {"b": 1.5}}


def test_shipped_configs_parse():
    for name in ("default", "lite", "large"):
        cfg = load_config(f"configs/{name}.yaml")
        assert cfg.name == name


@pytest.mark.parametrize("name,expected", [
    ("ENTAILMENT", SUPPORTED), ("entailment", SUPPORTED), ("SUPPORTS", SUPPORTED),
    ("contradiction", CONTRADICTED), ("REFUTES", CONTRADICTED), ("CONTRADICT", CONTRADICTED),
    ("neutral", INSUFFICIENT), ("NOT ENOUGH INFO", INSUFFICIENT), ("not_entailment", INSUFFICIENT),
    ("INSUFFICIENT_EVIDENCE", INSUFFICIENT),
])
def test_normalize_model_label(name, expected):
    assert normalize_model_label(name) == expected


def test_normalize_model_label_unknown():
    with pytest.raises(ValueError):
        normalize_model_label("LABEL_0")


def test_text_utils():
    assert "not" in tokenize("Aspirin does not reduce risk")
    assert "the" not in tokenize("the risk")
    assert split_sentences("First sentence. Second one (n = 3). Third!") == [
        "First sentence.", "Second one (n = 3).", "Third!"]
    cues = cue_counts("Statins did not reduce stroke but increased survival")
    assert cues["neg"] == 1 and cues["dec"] == 1 and cues["inc"] == 1
