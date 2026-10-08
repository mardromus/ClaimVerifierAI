"""Live literature sources: fetch candidate abstracts for a claim from public APIs.

* :class:`EuropePMCSource` - Europe PMC REST API (``/search`` with ``resultType=core`` returns abstracts).
* :class:`PubMedSource`    - NCBI E-utilities (``esearch`` for PMIDs, ``efetch`` for abstracts as XML).

Both turn the claim into a keyword query, return :class:`~claimverifier.data.scifact.Document` objects
(sentence-split, with bibliographic metadata) and are re-ranked by the pipeline's rationale selector,
so the verdict is computed exactly as for the indexed SciFact corpus. Responses are cached in memory.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zlib
from collections import OrderedDict
from typing import Dict, List, Optional

from .data.scifact import Document
from .text import clean_abstract_html, split_sentences, tokenize

logger = logging.getLogger(__name__)

USER_AGENT = "ClaimVerifierAI/1.0 (scientific claim verification; https://github.com/mardromus/ClaimVerifierAI)"


def keyword_query(claim: str, max_terms: int = 8) -> List[str]:
    """Content words of a claim, longest (most specific) first, keeping their original order on ties."""
    seen: List[str] = []
    for tok in tokenize(claim):
        if len(tok) > 2 and tok not in seen and not tok.isdigit():
            seen.append(tok)
    ranked = sorted(seen, key=lambda t: (-len(t), seen.index(t)))[:max_terms]
    return [t for t in seen if t in ranked]


def _get(url: str, timeout: float) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


class LiteratureSource:
    name = "base"
    label = "Base"
    description = ""

    def __init__(self, timeout: float = 15.0, cache_size: int = 256):
        self.timeout = timeout
        self._cache: "OrderedDict[tuple, List[Document]]" = OrderedDict()
        self._cache_size = cache_size
        self._lock = threading.Lock()

    def search(self, claim: str, limit: int = 20) -> List[Document]:
        key = (claim.strip().lower(), limit)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        docs = self._search(claim, limit)
        with self._lock:
            self._cache[key] = docs
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return docs

    def _search(self, claim: str, limit: int) -> List[Document]:
        raise NotImplementedError

    def describe(self) -> Dict[str, str]:
        return {"name": self.name, "label": self.label, "description": self.description}


def _make_document(doc_id: int, title: str, abstract: str, meta: Dict) -> Optional[Document]:
    sentences = split_sentences(abstract)
    if not sentences:
        return None
    return Document(doc_id=doc_id, title=title.strip().rstrip(".") + "." if title else "Untitled",
                    sentences=sentences, meta=meta)


class EuropePMCSource(LiteratureSource):
    name = "europepmc"
    label = "Europe PMC"
    description = "Live search over 40M+ life-science abstracts (PubMed, PMC, preprints) via the Europe PMC API"
    BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

    def __init__(self, timeout: float = 15.0, base_url: Optional[str] = None, **kw):
        super().__init__(timeout, **kw)
        self.base_url = base_url or self.BASE_URL

    def _query(self, terms: List[str], operator: str, page_size: int) -> List[dict]:
        query = f" {operator} ".join(terms)
        params = {"query": f"({query}) AND HAS_ABSTRACT:y", "format": "json", "resultType": "core",
                  "pageSize": str(page_size)}
        data = json.loads(_get(f"{self.base_url}?{urllib.parse.urlencode(params)}", self.timeout))
        return (data.get("resultList") or {}).get("result") or []

    def _search(self, claim: str, limit: int) -> List[Document]:
        terms = keyword_query(claim)
        if not terms:
            return []
        page_size = min(max(limit, 1), 100)
        results = self._query(terms, "AND", page_size)
        if len(results) < limit:  # strict query too narrow: widen to OR (the re-ranker sorts it out)
            seen = {r.get("id") for r in results}
            results += [r for r in self._query(terms, "OR", page_size) if r.get("id") not in seen]
        docs = []
        for r in results[:limit]:
            abstract = clean_abstract_html(r.get("abstractText", ""))
            journal = ((r.get("journalInfo") or {}).get("journal") or {}).get("title") or r.get("journalTitle", "")
            pmid = r.get("pmid")
            url = (f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else
                   f"https://europepmc.org/article/{r.get('source', 'MED')}/{r.get('id')}")
            meta = {"source": self.label, "journal": journal, "year": str(r.get("pubYear") or ""),
                    "authors": (r.get("authorString") or "").rstrip("."), "pmid": pmid or "",
                    "doi": r.get("doi") or "", "url": url}
            doc_id = int(pmid) if pmid and str(pmid).isdigit() else -1 - zlib.crc32(str(r.get("id")).encode())
            doc = _make_document(doc_id, clean_abstract_html(r.get("title", "")), abstract, meta)
            if doc:
                docs.append(doc)
        return docs


class PubMedSource(LiteratureSource):
    name = "pubmed"
    label = "PubMed"
    description = "Live search over 37M+ biomedical citations via NCBI E-utilities"
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def __init__(self, timeout: float = 15.0, base_url: Optional[str] = None, api_key: Optional[str] = None,
                 email: Optional[str] = None, **kw):
        super().__init__(timeout, **kw)
        self.base_url = (base_url or self.BASE_URL).rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("NCBI_API_KEY")
        self.email = email

    def _params(self, **params) -> str:
        params["tool"] = "claimverifier"
        if self.api_key:
            params["api_key"] = self.api_key
        if self.email:
            params["email"] = self.email
        return urllib.parse.urlencode(params)

    def _esearch(self, terms: List[str], operator: str, limit: int) -> List[str]:
        term = f" {operator} ".join(terms) + " AND hasabstract[text]"
        url = f"{self.base_url}/esearch.fcgi?{self._params(db='pubmed', term=term, retmode='json', retmax=limit, sort='relevance')}"
        data = json.loads(_get(url, self.timeout))
        return list((data.get("esearchresult") or {}).get("idlist") or [])

    def _search(self, claim: str, limit: int) -> List[Document]:
        terms = keyword_query(claim)
        if not terms:
            return []
        ids = self._esearch(terms, "AND", limit)
        if len(ids) < limit:
            ids += [i for i in self._esearch(terms, "OR", limit) if i not in ids]
        ids = ids[:limit]
        if not ids:
            return []
        url = f"{self.base_url}/efetch.fcgi?{self._params(db='pubmed', id=','.join(ids), retmode='xml')}"
        return parse_pubmed_xml(_get(url, self.timeout), self.label)


def parse_pubmed_xml(xml_bytes: bytes, source_label: str = "PubMed") -> List[Document]:
    root = ET.fromstring(xml_bytes)
    docs = []
    for article in root.iter("PubmedArticle"):
        pmid = (article.findtext(".//MedlineCitation/PMID") or "").strip()
        art = article.find(".//MedlineCitation/Article")
        if art is None or not pmid.isdigit():
            continue
        title = "".join(art.find("ArticleTitle").itertext()) if art.find("ArticleTitle") is not None else ""
        parts = []
        for node in art.findall("./Abstract/AbstractText"):
            text = " ".join("".join(node.itertext()).split())
            label = node.get("Label")
            parts.append(f"{label.capitalize()}: {text}" if label and text else text)
        journal = art.findtext("./Journal/Title") or ""
        year = (art.findtext("./Journal/JournalIssue/PubDate/Year")
                or (art.findtext("./Journal/JournalIssue/PubDate/MedlineDate") or "")[:4])
        authors = []
        for a in art.findall("./AuthorList/Author"):
            last, initials = a.findtext("LastName"), a.findtext("Initials")
            if last:
                authors.append(f"{last} {initials or ''}".strip())
        doi = next((e.text for e in article.findall(".//ArticleIdList/ArticleId") if e.get("IdType") == "doi"), "")
        meta = {"source": source_label, "journal": journal, "year": year,
                "authors": ", ".join(authors[:6]) + (" et al" if len(authors) > 6 else ""),
                "pmid": pmid, "doi": doi or "", "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"}
        doc = _make_document(int(pmid), title, " ".join(p for p in parts if p), meta)
        if doc:
            docs.append(doc)
    return docs


SOURCES = {"europepmc": EuropePMCSource, "pubmed": PubMedSource}


def create_source(name: str, timeout: float = 15.0, **kwargs) -> LiteratureSource:
    if name not in SOURCES:
        raise ValueError(f"Unknown literature source {name!r} (available: {sorted(SOURCES)})")
    return SOURCES[name](timeout=timeout, **kwargs)
