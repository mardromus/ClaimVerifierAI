import json

import yaml

from claimverifier.cli import main


def _write_cfg(lite_cfg, tmp_path):
    path = tmp_path / "cfg.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(lite_cfg.to_dict(), f)
    return str(path)


def test_cli_stats(lite_cfg, tmp_path, capsys):
    main(["stats", "--config", _write_cfg(lite_cfg, tmp_path)])
    assert json.loads(capsys.readouterr().out)["corpus_documents"] == 10


def test_cli_verify_json(lite_verifier, lite_cfg, tmp_path, capsys):
    cfg = _write_cfg(lite_cfg, tmp_path)
    main(["verify", "--config", cfg, "--json", "--no-explain", "Exercise improves insulin sensitivity."])
    out = json.loads(capsys.readouterr().out)
    assert out["documents"][0]["doc_id"] == 103 and out["explanation"] == ""


def test_cli_verify_pretty(lite_verifier, lite_cfg, tmp_path, capsys):
    main(["verify", "--config", _write_cfg(lite_cfg, tmp_path), "--show-all", "Sleep loss impairs memory."])
    out = capsys.readouterr().out
    assert "VERDICT:" in out and "EXPLANATION (template)" in out and "[1]" in out


def test_cli_evaluate(lite_verifier, lite_cfg, tmp_path, capsys):
    out_dir = tmp_path / "report"
    main(["evaluate", "--config", _write_cfg(lite_cfg, tmp_path), "--split", "dev", "--out", str(out_dir)])
    assert "Verdict classification" in capsys.readouterr().out
    assert json.loads((out_dir / "metrics.json").read_text())["num_claims"] == 6


def test_cli_retrieval_only(lite_verifier, lite_cfg, tmp_path, capsys):
    main(["evaluate", "--config", _write_cfg(lite_cfg, tmp_path), "--split", "dev", "--retrieval-only"])
    out = capsys.readouterr().out
    metrics = json.loads(out[out.index("{"):])
    assert metrics["num_queries"] == 4 and 0 < metrics["mrr"] <= 1


def test_shipped_scibert_config_uses_base_model():
    from claimverifier.config import load_config

    cfg = load_config("configs/scibert.yaml")
    assert cfg.nli.base_model == "allenai/scibert_scivocab_uncased"
    assert cfg.nli.finetuned_path == cfg.nli.model_name == "models/scibert-verifier"
