"""The two chunking strategies compared in the eval (ProjectSpec.md section 5.5) -> `chunks`.

- `fixed` (naive baseline): all kept text of a document, tables flattened into it as plain words, cut into
  ~512-token pieces with 64 tokens of overlap. A piece's context is the piece itself.
- `section`: one piece per section. A section over 512 tokens is cut at numbered subheadings (label
  "14.1 ...", "14.2 ..."), then paragraph and line breaks, then sentence ends. Its context is the whole
  section. Every table and figure is its own piece.

Run from the repo root:  python -m clinical_rag.chunking.strategies
Only documents without chunks are chunked (load.py deletes the chunks of changed documents).
"""

import re
from collections import Counter
from functools import cache

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
from psycopg.types.json import Jsonb

from clinical_rag.db import connect
from clinical_rag.llm import EMBEDDING_MODELS

MAX_TOKENS = 512
FIXED_OVERLAP = 64
HEADING_TOKENS = 40 # a line this short with no sentence end is treated as a subheading

# Back matter and patient leaflets are not chunked. Funding is kept, even in e.g. "Study funding/competing interests".
KEEP_SECTION_RE = re.compile(r"fund|financial support|sponsor", re.IGNORECASE)
SKIP_SECTION_RE = re.compile(
    r"acknowledg|disclos|conflict|competing interest|declaration|duality of interest|relationships and activities"
    r"|data (availab|availib|sharing|accessib|statement)|availability of data|associated data|source data"
    r"|code availability|resource availability|ethic|consent|institutional review board|contribut|authorship"
    r"|authors.? roles|guarantor|footnote|supplement|peer review|provenance|publisher|online content"
    r"|abbreviation|glossary|generative ai|orcid|^references$|^spl medguide$",
    re.IGNORECASE)

NCT_RE = re.compile(r"NCT\d{8}")
OWN_TRIAL_SECTION_RE = re.compile(r"abstract|regist", re.IGNORECASE)  # where a paper names its own trial

# Where a long section may be cut, from the biggest break to the smallest: (pattern, glue to rejoin pieces).
BREAKS = [(r"\n\n", "\n\n"), (r"\n", "\n"), (r"(?<=[.!?])\s+(?=[A-Z0-9(\[•])|\s+(?=• )", " ")]


@cache
def encoding() -> tiktoken.Encoding:
    """The tokenizer of the embedding model, so chunk sizes match what is billed."""
    try:
        return tiktoken.encoding_for_model(EMBEDDING_MODELS["oai-3-small"])
    except KeyError:
        return tiktoken.get_encoding("o200k_base")


def count_tokens(text: str) -> int:
    return len(encoding().encode(text))


def is_skipped(section: str | None) -> bool:
    return bool(section) and not KEEP_SECTION_RE.search(section) and bool(SKIP_SECTION_RE.search(section))


def _subsections(text: str) -> list[str]:
    """Cut a label section like '14 CLINICAL STUDIES 14.1 ... 14.2 ...' before each '14.<n> <Capital>'."""
    m = re.match(r"\s*(\d{1,2})\s+[A-Z]", text)
    if not m:
        return [text]
    parts = [p.strip() for p in re.split(rf"(?=\b{m.group(1)}\.\d{{1,2}}\s+[A-Z])", text) if p.strip()]
    if len(parts) > 1 and not re.match(rf"{m.group(1)}\.\d", parts[0]):
        parts[:2] = [f"{parts[0]} {parts[1]}"]  # keep the '14 CLINICAL STUDIES' heading with 14.1
    return parts


def _hard_split(text: str) -> list[str]:
    """Last resort for a single 'sentence' over the limit: cut every MAX_TOKENS tokens."""
    tokens = encoding().encode(text)
    return [encoding().decode(tokens[i:i + MAX_TOKENS]).strip() for i in range(0, len(tokens), MAX_TOKENS)]


def _attach_headings(units: list[str], glue: str) -> list[str]:
    """Join a short line without a sentence end (a subheading like 'Outcomes') to the paragraph after it.

    Only the last of several such lines in a row is joined, so list items stay separate units.
    """
    attached, pending = [], ""
    for unit in units:
        if count_tokens(unit) <= HEADING_TOKENS and not unit.endswith((".", "!", "?")):
            attached += [pending] if pending else []
            pending = unit
        else:
            attached.append(f"{pending}{glue}{unit}" if pending else unit)
            pending = ""
    return attached + ([pending] if pending else [])


def _pack(text: str, breaks: list[tuple[str, str]]) -> list[str]:
    """Cut text at the first kind of break, then fill each piece up to MAX_TOKENS with whole units."""
    if count_tokens(text) <= MAX_TOKENS:
        return [text]
    if not breaks:
        return _hard_split(text)
    (pattern, glue), smaller = breaks[0], breaks[1:]
    units = _attach_headings([u.strip() for u in re.split(pattern, text) if u.strip()], glue)
    if len(units) == 1:
        return _pack(text, smaller)
    pieces, current = [], ""
    for unit in units:
        for small in _pack(unit, smaller):
            joined = f"{current}{glue}{small}" if current else small
            if count_tokens(joined) <= MAX_TOKENS:
                current = joined
            else:
                pieces.append(current)
                current = small
    return pieces + [current]


def split_section(text: str) -> list[str]:
    """`section` strategy: the whole section, or pieces of at most MAX_TOKENS that never cut a sentence."""
    if count_tokens(text) <= MAX_TOKENS:
        return [text]
    return [piece for part in _subsections(text) for piece in _pack(part, BREAKS)]


def trial_label(doc: dict, acronyms: dict[str, str | None]) -> str | None:
    """'SURMOUNT-1 (NCT04184622)' for each registered trial a paper reports. A paper's Results part often never
    names its trial (only the Conclusions do), so the label goes in every chunk header. None for registry entries.

    Europe PMC `nct_ids` come from the full text, which also cites other trials (561 IDs vs 238 in title +
    abstract), so for those papers only the title, abstract and trial-registration sections count."""
    if doc["source"] == "clinicaltrials.gov":
        return None
    ids = doc["metadata"].get("nct_ids", [])
    if doc["source"] == "europepmc":
        own = " ".join([doc["title"], *(t for n, t in doc["sections"].items() if OWN_TRIAL_SECTION_RE.search(n))])
        ids = sorted(set(NCT_RE.findall(own)))
    return ", ".join(f"{acronyms[i]} ({i})" if acronyms.get(i) else i for i in ids) or None


def _prefix(doc: dict, section: str | None) -> str:
    parts = [f"Title: {doc['title']}", f"Trial: {doc['trials']}" if doc.get("trials") else "",
             f"Section: {section}" if section else ""]
    return " | ".join(p for p in parts if p) + "\n\n"


def _label_caption(item: dict) -> str:
    return " ".join(x for x in [item.get("label"), item.get("caption")] if x)


def _flat_table(table: dict) -> str:
    """A markdown table as one run of words, the way a naive parser sees it."""
    rows = [line.strip("| ").replace(" | ", " ") for line in table["markdown"].splitlines()
            if not set(line) <= set("|- ")]
    return " ".join(x for x in [_label_caption(table), *rows, table.get("footnotes")] if x)


def _chunk(doc: dict, strategy: str, n: int, section: str | None, content: str, context: str,
           content_type: str, label: str | None = None) -> dict:
    meta = doc["metadata"]
    metadata = {"content_type": content_type, "drugs": meta.get("drugs", []), "year": meta.get("year"),
                "source": doc["source"], "license": doc["license"]}
    if label:
        metadata["label"] = label
    return {"chunk_id": f"{doc['doc_id']}:{strategy}:{n}", "doc_id": doc["doc_id"], "strategy": strategy,
            "section": section, "content": content, "context": context, "token_count": count_tokens(content),
            "metadata": metadata}


def _kept(doc: dict) -> tuple[dict[str, str], list[dict], list[dict]]:
    """Sections, tables and figures that get chunked (not back matter)."""
    meta = doc["metadata"]
    sections = {name: text for name, text in doc["sections"].items() if not is_skipped(name)}
    tables = [t for t in meta.get("tables", []) if t.get("markdown") and not is_skipped(t.get("section"))]
    figures = [f for f in meta.get("figures", []) if f.get("caption") and not is_skipped(f.get("section"))]
    return sections, tables, figures


def fixed_chunks(doc: dict) -> list[dict]:
    sections, tables, figures = _kept(doc)
    extras: dict[str | None, list[str]] = {}  # flattened tables and captions, placed after their section
    for item, text in [(t, _flat_table(t)) for t in tables] + [(f, _label_caption(f)) for f in figures]:
        extras.setdefault(item.get("section") if item.get("section") in sections else None, []).append(text)
    blocks = []
    for name, text in sections.items():
        blocks += [f"{name}: {text}", *extras.get(name, [])]  # on one line, so a name is never a piece alone
    body = "\n\n".join(blocks + extras.get(None, []))
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=encoding().name, chunk_size=MAX_TOKENS, chunk_overlap=FIXED_OVERLAP)
    return [_chunk(doc, "fixed", n, None, _prefix(doc, None) + piece, piece, "text")
            for n, piece in enumerate(splitter.split_text(body))]


def section_chunks(doc: dict) -> list[dict]:
    sections, tables, figures = _kept(doc)
    chunks = []
    for name, text in sections.items():
        for piece in split_section(text):
            chunks.append(_chunk(doc, "section", len(chunks), name, _prefix(doc, name) + piece, text, "text"))
    for t in tables:
        # The spec's one-line summary by the fast model is added to `content` in a later step.
        header = t["markdown"].splitlines()[0]
        content = _prefix(doc, t.get("section")) + "\n".join(x for x in [_label_caption(t), header] if x)
        context = "\n\n".join(x for x in [_label_caption(t), t["markdown"], t.get("footnotes")] if x)
        chunks.append(_chunk(doc, "section", len(chunks), t.get("section"), content, context, "table", t.get("label")))
    for f in figures:
        caption = _label_caption(f)
        chunks.append(_chunk(doc, "section", len(chunks), f.get("section"), _prefix(doc, f.get("section")) + caption,
                             caption, "figure", f.get("label")))
    return chunks


def chunk_document(doc: dict) -> list[dict]:
    return fixed_chunks(doc) + section_chunks(doc)


def main() -> None:
    with connect() as conn:
        cur = conn.execute("select doc_id, source, title, license, sections, metadata from documents d "
                           "where not exists (select 1 from chunks c where c.doc_id = d.doc_id)")
        cols = [c.name for c in cur.description]
        docs = [dict(zip(cols, row)) for row in cur.fetchall()]
        acronyms = dict(conn.execute("select split_part(doc_id, ':', 2), metadata->>'acronym' from documents "
                                     "where source = 'clinicaltrials.gov'").fetchall())
        for d in docs:
            d["trials"] = trial_label(d, acronyms)
        print(f"{len(docs)} documents to chunk")
        chunks = [c for d in docs for c in chunk_document(d)]
        with conn.cursor() as cur:
            cur.executemany(
                "insert into chunks (chunk_id, doc_id, strategy, section, content, context, token_count, metadata) "
                "values (%(chunk_id)s, %(doc_id)s, %(strategy)s, %(section)s, %(content)s, %(context)s, "
                "%(token_count)s, %(metadata)s)",
                [{**c, "metadata": Jsonb(c["metadata"])} for c in chunks])
        counts = Counter((c["strategy"], c["metadata"]["source"], c["metadata"]["content_type"]) for c in chunks)
        for (strategy, source, content_type), n in sorted(counts.items()):
            print(f"  {strategy:8} {source:18} {content_type:6} {n}")
        print(f"{len(chunks)} chunks written")


if __name__ == "__main__":
    main()
