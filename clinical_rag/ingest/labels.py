"""openFDA drug labels: current prescribing information for the in-scope brands -> `documents`.

Run from the repo root:  python -m clinical_rag.ingest.labels
"""

import re
from datetime import date

import lxml.html
import requests

from clinical_rag.db import connect, drug_aliases
from clinical_rag.ingest.drugs import matched_drugs
from clinical_rag.ingest.load import content_hash, upsert_documents
from clinical_rag.parsing.tables import clean, table_to_markdown

API_URL = "https://api.fda.gov/drug/label.json"

# Pinned to the original manufacturer's label (set IDs checked on 2026-10-03). Searching by brand name
# also returns repackager copies, and misses the current Wegovy label entirely.
LABEL_SET_IDS = {
    "Wegovy": "ee06186f-2aa3-4990-a760-757579d8f77b",
    "Ozempic": "27f15fac-7d98-4114-a2ec-92494a91da98",
    "Saxenda": "3946d389-0926-4f77-a708-0acb8153b143",
    "Zepbound": "487cd7e7-434c-4925-99fa-aa80b1cc776b",
    "Mounjaro": "d2d7da5d-ad07-4228-955f-cf7e355c8cc0",
}

# Label fields that are identifiers or packaging, not clinical content.
SKIP_FIELDS = {"id", "set_id", "version", "effective_time", "openfda", "spl_product_data_elements",
               "package_label_principal_display_panel"}
TABLE_LABEL_RE = re.compile(r"\bTable\s+\d+[A-Za-z]?", re.IGNORECASE)


def fetch_label(set_id: str) -> dict:
    resp = requests.get(API_URL, params={"search": f'set_id:"{set_id}"'}, timeout=60)
    resp.raise_for_status()
    return resp.json()["results"][0]


def _field_title(field: str) -> str:
    return field.replace("_", " ").capitalize()


def parse_tables(field: str, html_tables: list[str]) -> list[dict]:
    tables = []
    for html in html_tables:
        table = lxml.html.fragment_fromstring(html, create_parent="div")
        caption_el = table.find(".//caption")
        caption = clean(caption_el.text_content()) if caption_el is not None else ""
        markdown, footnotes = table_to_markdown(table)
        if not markdown:
            continue
        label = TABLE_LABEL_RE.search(caption)
        tables.append({"label": label.group(0) if label else None, "caption": caption,
                       "section": _field_title(field.removesuffix("_table")),
                       "markdown": markdown, "footnotes": footnotes})
    return tables


def _without_tables(text: str, html_tables: list[str]) -> str:
    """openFDA section text also contains each of its tables as one run of words; cut that copy out."""
    text = clean(text)
    for html in html_tables:
        flat = clean(" ".join(lxml.html.fragment_fromstring(html, create_parent="div").itertext()))
        if flat:
            text = text.replace(flat, " ")
    return clean(text)


def to_document(brand: str, label: dict, aliases: dict[str, str]) -> dict:
    sections, tables = {}, []
    for field, value in label.items():
        if field in SKIP_FIELDS or not isinstance(value, list):
            continue
        if field.endswith("_table"):
            tables += parse_tables(field, value)
        else:
            html_tables = label.get(f"{field}_table", [])
            sections[_field_title(field)] = "\n".join(_without_tables(v, html_tables) for v in value)
    effective = date.fromisoformat(label["effective_time"])  # 'YYYYMMDD'
    title = f"{brand} prescribing information"
    return {
        "doc_id": f"label:{label['set_id']}",
        "source": "dailymed",
        "url": f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={label['set_id']}",
        "title": title,
        "published_date": effective,
        "license": None,  # FDA-approved labeling, published by the US government
        "sections": sections,
        "metadata": {
            "drugs": matched_drugs(" ".join([brand, *label["openfda"].get("generic_name", [])]), aliases),
            "brand": brand,
            "manufacturer": (label["openfda"].get("manufacturer_name") or [None])[0],
            "label_effective_date": effective.isoformat(),
            "year": effective.year,
            "tables": tables,
            "figures": [],
        },
        "content_hash": content_hash(title, sections, tables),
    }


def main() -> None:
    with connect() as conn:
        aliases = drug_aliases(conn)
        docs = [to_document(brand, fetch_label(set_id), aliases) for brand, set_id in LABEL_SET_IDS.items()]
        for d in docs:
            print(f"{d['title']}: {len(d['sections'])} sections, {len(d['metadata']['tables'])} tables, "
                  f"effective {d['published_date']}")
        print(f"documents: {upsert_documents(conn, docs)}")


if __name__ == "__main__":
    main()
