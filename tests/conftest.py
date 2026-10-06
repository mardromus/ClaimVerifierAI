import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from claimverifier.config import load_config  # noqa: E402

FIXTURE_DATA = Path(__file__).parent / "fixtures" / "scifact_mini"


def make_lite_config(artifacts_dir: Path, data_dir: Path = FIXTURE_DATA):
    return load_config(ROOT / "configs" / "lite.yaml", [
        f"data.data_dir={data_dir}",
        f"artifacts_dir={artifacts_dir}",
        f"reports_dir={artifacts_dir / 'reports'}",
        "retrieval.lsa_dim=8",
        "retrieval.top_k=3",
        "retrieval.candidate_pool=10",
    ])


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    target = tmp_path_factory.mktemp("data") / "scifact_mini"
    shutil.copytree(FIXTURE_DATA, target)
    return target


@pytest.fixture(scope="session")
def lite_cfg(tmp_path_factory, data_dir):
    return make_lite_config(tmp_path_factory.mktemp("artifacts"), data_dir)


@pytest.fixture(scope="session")
def lite_verifier(lite_cfg):
    """A fully set-up lite pipeline (index + trained lightweight models) on the mini corpus."""
    from claimverifier.data import load_claims, load_corpus
    from claimverifier.pipeline import ClaimVerifier
    from claimverifier.retrieval import Retriever
    from claimverifier.training.lite import train_lite_models

    corpus = load_corpus(lite_cfg.data.data_dir)
    retriever = Retriever.build(lite_cfg.retrieval, corpus, lite_cfg.artifacts_dir, show_progress=False)
    train_lite_models(lite_cfg, retriever, corpus, load_claims(lite_cfg.data.data_dir, "train"),
                      load_claims(lite_cfg.data.data_dir, "dev"))
    return ClaimVerifier.from_config(lite_cfg)
