"""PubMed E-utilities: randomized controlled trial abstracts (2018 onward) for the in-scope drugs -> `documents`.

Run from the repo root:  python -m clinical_rag.ingest.pubmed
"""

import os
import re
from datetime import date

import requests
from lxml import etree

from clinical_rag.db import connect, drug_aliases
from clinical_rag.ingest.drugs import matched_drugs, search_terms
from clinical_rag.ingest.load import content_hash, upsert_documents
from clinical_rag.parsing.tables import clean

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
BATCH = 200
NCT_RE = re.compile(r"NCT\d{8}")


def search_query(terms: list[str]) -> str:
    any_drug = " OR ".join(f'"{t}"' for t in terms)
    return f"obesity AND ({any_drug}) AND randomized controlled trial[pt] AND 2018:3000[dp]"


def _ncbi_params() -> dict:
    return {"tool": "clinical-rag", "email": os.environ["NCBI_EMAIL"]}


def fetch_pmids(query: str) -> list[str]:
    resp = requests.get(f"{EUTILS}/esearch.fcgi", timeout=60, params={
        **_ncbi_params(), "db": "pubmed", "term": query, "retmode": "json", "retmax": 10000})
    resp.raise_for_status()
    return resp.json()["esearchresult"]["idlist"]


def fetch_articles(pmids: list[str]) -> list[etree._Element]:
    articles = []
    for i in range(0, len(pmids), BATCH):
        resp = requests.post(f"{EUTILS}/efetch.fcgi", timeout=120, data={
            **_ncbi_params(), "db": "pubmed", "id": ",".join(pmids[i:i + BATCH]), "retmode": "xml"})
        resp.raise_for_status()
        articles += etree.fromstring(resp.content).findall("PubmedArticle")
    return articles


def _text(el: etree._Element | None) -> str:
    return clean("".join(el.itertext())) if el is not None else ""


def to_document(article: etree._Element, aliases: dict[str, str]) -> dict | None:
    """None when the article has no abstract (nothing to search)."""
    cit = article.find("MedlineCitation")
    pmid = cit.findtext("PMID")
    title = _text(cit.find("Article/ArticleTitle"))
    sections: dict[str, str] = {}
    for part in cit.findall("Article/Abstract/AbstractText"):
        label = (part.get("Label") or "Abstract").capitalize()
        sections[label] = (sections.get(label, "") + "\n" + _text(part)).strip()
    if not sections:
        return None
    # 'pubmed' history date is always a full Y/M/D, unlike the journal issue date.
    hist = article.find("PubmedData/History/PubMedPubDate[@PubStatus='pubmed']")
    published = date(int(hist.findtext("Year")), int(hist.findtext("Month")), int(hist.findtext("Day")))
    registered = cit.findall("Article/DataBankList/DataBank[DataBankName='ClinicalTrials.gov']"
                             "/AccessionNumberList/AccessionNumber")
    nct_ids = sorted({a.text for a in registered} | set(NCT_RE.findall(" ".join(sections.values()))))
    return {
        "doc_id": f"pubmed:{pmid}",
        "source": "pubmed",
        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        "title": title,
        "published_date": published,
        "license": None,  # abstracts remain the publisher's copyright; we store them and show short snippets only
        "sections": sections,
        "metadata": {
            "drugs": matched_drugs(" ".join([title, *sections.values()]), aliases),
            "nct_ids": nct_ids,
            "journal": cit.findtext("Article/Journal/Title"),
            "year": published.year,
        },
        "content_hash": content_hash(title, sections),
    }


def main() -> None:
    with connect() as conn:
        aliases = drug_aliases(conn)
        pmids = fetch_pmids(search_query(search_terms(aliases)))
        print(f"found {len(pmids)} PubMed records")
        docs = [d for a in fetch_articles(pmids) if (d := to_document(a, aliases))]
        print(f"{len(docs)} with abstracts; documents: {upsert_documents(conn, docs)}")


if __name__ == "__main__":
    main()
