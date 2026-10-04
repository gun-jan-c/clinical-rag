"""Europe PMC open-access RCT full text (JATS XML): text sections, markdown tables, figure captions -> `documents`.

Run from the repo root:  python -m clinical_rag.ingest.europepmc
Downloaded XML is cached in data/raw/europepmc/ (gitignored), so re-runs only fetch new articles.
"""

import re
import time
from datetime import date, datetime, timezone
from pathlib import Path

import requests
from lxml import etree

from clinical_rag.db import connect, drug_aliases
from clinical_rag.ingest.drugs import matched_drugs, search_terms
from clinical_rag.ingest.load import content_hash, upsert_documents
from clinical_rag.parsing.tables import clean, table_to_markdown

API = "https://www.ebi.ac.uk/europepmc/webservices/rest"
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "europepmc"
NCT_RE = re.compile(r"NCT\d{8}")
SKIP_INSIDE = {"table-wrap", "fig", "caption"}  # their text goes to tables/figures, not sections


def search_query(terms: list[str]) -> str:
    any_drug = " OR ".join(f'"{t}"' for t in terms)
    return (f'(obesity) AND ({any_drug}) AND OPEN_ACCESS:y AND PUB_TYPE:"randomized controlled trial" '
            f"AND PUB_YEAR:[2018 TO {datetime.now(timezone.utc).year}]")


def search(query: str) -> list[dict]:
    params = {"query": query, "format": "json", "resultType": "core", "pageSize": 1000, "cursorMark": "*"}
    hits = []
    while True:
        resp = requests.get(f"{API}/search", params=params, timeout=120)
        resp.raise_for_status()
        page = resp.json()
        hits += page["resultList"]["result"]
        if not page["resultList"]["result"] or page.get("nextCursorMark") in (None, params["cursorMark"]):
            return [h for h in hits if h.get("pmcid")]
        params["cursorMark"] = page["nextCursorMark"]


def fetch_xml(session: requests.Session, pmcid: str) -> bytes | None:
    """Full-text JATS XML, from the local cache when present.

    None when Europe PMC has no full text, or still returns a server error after 3 tries.
    """
    path = RAW_DIR / f"{pmcid}.xml"
    if path.exists():
        return path.read_bytes()
    for attempt in range(3):
        resp = session.get(f"{API}/{pmcid}/fullTextXML", timeout=120)
        if resp.status_code < 500:
            break
        time.sleep(2 ** attempt)
    if resp.status_code == 404 or resp.status_code >= 500:
        return None
    resp.raise_for_status()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path.write_bytes(resp.content)
    return resp.content


def _local(el: etree._Element) -> str:
    return etree.QName(el).localname if isinstance(el.tag, str) else ""


def _text(el: etree._Element | None) -> str:
    return clean("".join(el.itertext())) if el is not None else ""


def _skipped(el: etree._Element) -> bool:
    return any(_local(a) in SKIP_INSIDE for a in el.iterancestors())


def _itertext_outside_skipped(el: etree._Element):
    """Like itertext(), but leaves out tables and figures placed inside a paragraph."""
    if el.text:
        yield el.text
    for child in el:
        if isinstance(child.tag, str) and _local(child) not in SKIP_INSIDE:
            yield from _itertext_outside_skipped(child)
        if child.tail:
            yield child.tail


def _section_text(sec: etree._Element) -> str:
    """Paragraphs of a section in reading order, with nested subsection titles on their own lines."""
    parts = []
    for el in sec.iter("p", "title"):
        if el is sec.find("title") or _skipped(el):
            continue
        if el.tag == "title" and el.getparent().tag != "sec":
            continue  # titles of lists, boxes, etc.
        parts.append(clean("".join(_itertext_outside_skipped(el))))
    return "\n\n".join(p for p in parts if p)


def _nearest_section(el: etree._Element) -> str | None:
    for a in el.iterancestors("sec"):
        if a.getparent() is not None and a.getparent().tag == "body":
            return _text(a.find("title")) or None
    return None


def parse_jats(xml: bytes) -> dict:
    root = etree.fromstring(xml)
    meta = root.find(".//article-meta")
    sections: dict[str, str] = {}
    abstract = meta.find("abstract") if meta is not None else None
    if abstract is not None:
        sections["Abstract"] = _section_text(abstract)
    body = root.find(".//body")
    for i, sec in enumerate(body.findall("sec") if body is not None else []):
        name = _text(sec.find("title")) or f"Section {i + 1}"
        if name in sections:
            name = f"{name} ({i + 1})"
        if text := _section_text(sec):
            sections[name] = text
    tables = []
    for wrap in root.iter("table-wrap"):
        table = next((t for t in wrap.iter("table")), None)
        markdown, foot_rows = table_to_markdown(table) if table is not None else ("", "")
        footnotes = "\n".join(f for f in [_text(wrap.find("table-wrap-foot")), foot_rows] if f)
        tables.append({"label": _text(wrap.find("label")) or None, "caption": _text(wrap.find("caption")),
                       "section": _nearest_section(wrap), "markdown": markdown, "footnotes": footnotes})
    figures = [{"label": _text(fig.find("label")) or None, "caption": _text(fig.find("caption")),
                "section": _nearest_section(fig)}
               for fig in root.iter("fig") if fig.find("caption") is not None]
    title = _text(meta.find("title-group/article-title")) if meta is not None else ""
    return {"title": title, "sections": {k: v for k, v in sections.items() if v},
            "tables": tables, "figures": figures}


def to_document(hit: dict, xml: bytes, aliases: dict[str, str]) -> dict:
    parsed = parse_jats(xml)
    pmcid = hit["pmcid"]
    title = parsed["title"] or hit.get("title")
    sections, tables, figures = parsed["sections"], parsed["tables"], parsed["figures"]
    all_text = " ".join([title or "", *sections.values(), *(t["caption"] for t in tables)])
    # Drugs from title + abstract only: introductions often name other drugs as background.
    about_text = " ".join([title or "", sections.get("Abstract", "")])
    published = date.fromisoformat(hit["firstPublicationDate"]) if hit.get("firstPublicationDate") else None
    return {
        "doc_id": f"epmc:{pmcid}",
        "source": "europepmc",
        "url": f"https://europepmc.org/article/PMC/{pmcid}",
        "title": title,
        "published_date": published,
        "license": hit.get("license"),
        "sections": sections,
        "metadata": {
            "drugs": matched_drugs(about_text, aliases),
            "nct_ids": sorted(set(NCT_RE.findall(all_text))),
            "pmid": hit.get("pmid"),  # the same paper may also be in the corpus as pubmed:<pmid>
            "journal": hit.get("journalInfo", {}).get("journal", {}).get("title"),
            "year": published.year if published else None,
            "tables": tables,
            "figures": figures,
        },
        "content_hash": content_hash(title, sections, tables, figures),
    }


def main() -> None:
    with connect() as conn:
        aliases = drug_aliases(conn)
        hits = search(search_query(search_terms(aliases)))
        print(f"found {len(hits)} open-access articles")
        docs, missing = [], []
        with requests.Session() as session:
            for n, hit in enumerate(hits, start=1):
                xml = fetch_xml(session, hit["pmcid"])
                if xml is None:
                    missing.append(hit["pmcid"])
                else:
                    docs.append(to_document(hit, xml, aliases))
                if n % 50 == 0:
                    print(f"  {n}/{len(hits)}")
        n_tables = sum(len(d["metadata"]["tables"]) for d in docs)
        n_figures = sum(len(d["metadata"]["figures"]) for d in docs)
        print(f"{len(docs)} parsed ({n_tables} tables, {n_figures} figures); "
              f"{len(missing)} without full text: {missing}")
        print(f"documents: {upsert_documents(conn, docs)}")


if __name__ == "__main__":
    main()
