import json
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from claimverifier.sources import EuropePMCSource, PubMedSource, create_source, keyword_query, parse_pubmed_xml
from claimverifier.text import clean_abstract_html, split_sentences

EPMC_HIT = {
    "id": "31234567", "source": "MED", "pmid": "31234567", "doi": "10.1000/xyz",
    "title": "Green tea extract and <i>blood pressure</i>: a randomized trial",
    "authorString": "Doe J, Roe R.", "pubYear": "2019",
    "journalInfo": {"journal": {"title": "Hypertension"}},
    "abstractText": "<h4>Background</h4>Tea may affect blood pressure. <h4>Results</h4>Green tea extract reduced "
                    "systolic blood pressure by 5.1 mmHg (e.g. in Fig. 2) versus placebo. No adverse events occurred.",
}
PUBMED_XML = b"""<?xml version="1.0"?>
<PubmedArticleSet>
 <PubmedArticle><MedlineCitation><PMID>111</PMID><Article>
  <Journal><Title>The Lancet</Title><JournalIssue><PubDate><Year>2020</Year></PubDate></JournalIssue></Journal>
  <ArticleTitle>Aspirin and <i>colorectal</i> cancer</ArticleTitle>
  <Abstract><AbstractText Label="BACKGROUND">Aspirin is widely used.</AbstractText>
   <AbstractText Label="RESULTS">Aspirin reduced colorectal cancer incidence (HR 0.77).</AbstractText></Abstract>
  <AuthorList><Author><LastName>Smith</LastName><Initials>AB</Initials></Author></AuthorList>
 </Article></MedlineCitation>
 <PubmedData><ArticleIdList><ArticleId IdType="doi">10.1/abc</ArticleId></ArticleIdList></PubmedData></PubmedArticle>
 <PubmedArticle><MedlineCitation><PMID>222</PMID><Article><ArticleTitle>No abstract</ArticleTitle></Article></MedlineCitation></PubmedArticle>
</PubmedArticleSet>"""


@pytest.fixture()
def fake_apis():
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            url = urllib.parse.urlparse(self.path)
            q = urllib.parse.parse_qs(url.query)
            calls.append((url.path, q))
            if url.path.endswith("/search"):
                strict = " AND " in q["query"][0] and " OR " not in q["query"][0]
                body = json.dumps({"resultList": {"result": [EPMC_HIT] if strict else [EPMC_HIT, {**EPMC_HIT, "id": "PPR1", "pmid": None, "source": "PPR"}]}}).encode()
            elif url.path.endswith("esearch.fcgi"):
                body = json.dumps({"esearchresult": {"idlist": ["111", "222"]}}).encode()
            elif url.path.endswith("efetch.fcgi"):
                body = PUBMED_XML
            else:
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}", calls
    server.shutdown()


def test_keyword_query_and_text_helpers():
    terms = keyword_query("Does green tea consumption lower the blood pressure of adults in 2019?")
    assert "consumption" in terms and "the" not in terms and "2019" not in terms
    assert clean_abstract_html("<h4>Methods</h4>We <b>did</b> it &amp; more.") == "Methods: We did it & more."
    assert split_sentences("A was shown by Smith et al. in mice. B followed.") == ["A was shown by Smith et al. in mice.", "B followed."]


def test_europepmc_source(fake_apis):
    url, calls = fake_apis
    src = EuropePMCSource(base_url=f"{url}/search")
    docs = src.search("Green tea lowers blood pressure", limit=2)
    assert len(docs) == 2
    d = docs[0]
    assert d.doc_id == 31234567 and d.title.startswith("Green tea extract and blood pressure")
    assert d.meta["journal"] == "Hypertension" and d.meta["year"] == "2019" and d.url.endswith("/31234567/")
    assert d.sentences[0].startswith("Background: Tea may affect") and any("e.g. in Fig. 2" in s for s in d.sentences)
    assert docs[1].doc_id < 0 and "europepmc.org/article/PPR/PPR1" in docs[1].url  # preprint without PMID
    assert len(calls) == 2  # strict AND query returned too few -> widened to OR
    src.search("Green tea lowers blood pressure", limit=2)
    assert len(calls) == 2  # cached


def test_pubmed_source(fake_apis):
    url, calls = fake_apis
    src = PubMedSource(base_url=url, api_key="k", email="a@b.c")
    docs = src.search("aspirin colorectal cancer", limit=2)
    assert [d.doc_id for d in docs] == [111]  # the article without an abstract is skipped
    d = docs[0]
    assert d.title == "Aspirin and colorectal cancer." and d.meta["authors"] == "Smith AB" and d.meta["doi"] == "10.1/abc"
    assert d.sentences == ["Background: Aspirin is widely used.", "Results: Aspirin reduced colorectal cancer incidence (HR 0.77)."]
    esearch = [q for p, q in calls if p.endswith("esearch.fcgi")][0]
    assert esearch["api_key"] == ["k"] and esearch["sort"] == ["relevance"]


def test_parse_pubmed_xml_and_factory():
    assert parse_pubmed_xml(PUBMED_XML)[0].meta["year"] == "2020"
    assert create_source("pubmed").name == "pubmed"
    with pytest.raises(ValueError):
        create_source("google")


def test_fetch_checkpoint_from_fake_server(tmp_path):
    import io
    import tarfile
    from http.server import SimpleHTTPRequestHandler

    from claimverifier import hub

    src = tmp_path / "srv"
    model_dir = tmp_path / "pack" / "label_model"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(json.dumps({"id2label": {"0": "LABEL_0", "1": "LABEL_1", "2": "LABEL_2"}}))
    (model_dir / "vocab.json").write_text("{}")
    src.mkdir()
    with tarfile.open(src / "m.tar.gz", "w:gz") as tar:
        tar.add(model_dir, arcname="label_model")

    class Quiet(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(src), **kw)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Quiet)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        spec = {**hub.CHECKPOINTS["verisci"], "url": f"http://127.0.0.1:{server.server_address[1]}/m.tar.gz"}
        hub.CHECKPOINTS["test"] = spec
        dest = hub.fetch_checkpoint("test", tmp_path / "models" / "v")
    finally:
        server.shutdown()
        hub.CHECKPOINTS.pop("test", None)
    cfg = json.loads((dest / "config.json").read_text())
    assert cfg["id2label"]["0"] == "CONTRADICT" and cfg["label2id"]["SUPPORT"] == 2 and (dest / "vocab.json").exists()
    with pytest.raises(ValueError):
        hub.fetch_checkpoint("nope", tmp_path / "x")
