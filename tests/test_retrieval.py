import numpy as np
import pytest

from claimverifier.data import load_corpus
from claimverifier.retrieval import BM25Index, FaissIndex, LSAEmbedder, Retriever


def test_lsa_embedder_normalised(data_dir, tmp_path):
    corpus = load_corpus(data_dir)
    emb = LSAEmbedder(dim=8).fit([d.full_text for d in corpus])
    x = emb.encode(["aspirin colorectal cancer", "vitamin D fractures"])
    assert x.shape == (2, emb.dim) and x.dtype == np.float32
    assert np.allclose(np.linalg.norm(x, axis=1), 1.0, atol=1e-5)
    emb.save(tmp_path)
    assert np.allclose(LSAEmbedder.load(tmp_path).encode(["aspirin"]), emb.encode(["aspirin"]))


@pytest.mark.parametrize("index_type", ["flat", "hnsw", "ivf"])
def test_faiss_index_types(index_type, tmp_path):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 16)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    index = FaissIndex(16, index_type, ivf_nlist=8, ivf_nprobe=8).build(x)
    scores, ids = index.search(x[:5], 3)
    assert ids.shape == (5, 3)
    if index_type == "flat":
        assert list(ids[:, 0]) == [0, 1, 2, 3, 4]
        assert np.allclose(scores[:, 0], 1.0, atol=1e-5)
    index.save(tmp_path / "x.index")
    loaded = FaissIndex.load(tmp_path / "x.index", index_type, ivf_nprobe=8)
    assert loaded.size == 500
    assert loaded.search(x[:1], 1)[1].shape == (1, 1)


def test_bm25_ranks_lexical_match(data_dir):
    corpus = load_corpus(data_dir)
    bm25 = BM25Index([d.full_text for d in corpus])
    scores, ids = bm25.search("coffee Parkinson disease", 3)
    assert corpus[ids[0]].doc_id == 108
    assert scores[0] >= scores[1] >= scores[2]


def test_retriever_build_load_search(lite_cfg, data_dir, tmp_path):
    corpus = load_corpus(data_dir)
    retriever = Retriever.build(lite_cfg.retrieval, corpus, tmp_path, show_progress=False)
    res = retriever.search("Does aspirin lower colorectal cancer risk?", top_k=3)
    assert len(res) == 3 and [r.rank for r in res] == [1, 2, 3]
    assert res[0].doc.doc_id == 101
    assert res[0].score >= res[1].score >= res[2].score
    loaded = Retriever.load(lite_cfg.retrieval, corpus, tmp_path)
    assert [r.doc.doc_id for r in loaded.search("Does aspirin lower colorectal cancer risk?", 3)] == \
           [r.doc.doc_id for r in res]
    ranked = loaded.rank_doc_ids(["statin cardiovascular events", "sleep memory"], depth=10)
    assert ranked[0][0] == 106 and ranked[1][0] == 107 and len(ranked[0]) == 10


def test_retriever_dense_only(lite_cfg, data_dir, tmp_path):
    import copy

    cfg = copy.deepcopy(lite_cfg.retrieval)
    cfg.hybrid = False
    retriever = Retriever.build(cfg, load_corpus(data_dir), tmp_path, show_progress=False)
    res = retriever.search("Mediterranean diet and cognitive decline", top_k=2)
    assert res[0].doc.doc_id == 110 and res[0].score == pytest.approx(res[0].dense_score)


def test_retriever_rejects_mismatched_index(lite_cfg, data_dir, tmp_path):
    import copy

    corpus = load_corpus(data_dir)
    Retriever.build(lite_cfg.retrieval, corpus, tmp_path, show_progress=False)
    cfg = copy.deepcopy(lite_cfg.retrieval)
    cfg.embedder = "sbert"
    with pytest.raises(RuntimeError, match="Rebuild"):
        Retriever.load(cfg, corpus, tmp_path)
    with pytest.raises(RuntimeError, match="different corpus"):
        Retriever.load(lite_cfg.retrieval, corpus[:-1], tmp_path)
