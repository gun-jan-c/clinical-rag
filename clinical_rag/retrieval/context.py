"""Context expansion (ProjectSpec.md section 3, step 5): search small pieces, give the model the bigger thing.

A hit is found by its short `content`; the model reads its `context`: the full section for text, the full
markdown table (+ footnotes) for tables, caption + description for figures. Hits that share a section send it
once. Text sections over MAX_SECTION_TOKENS (the biggest ~10%, up to ~17k tokens) send only the matched
piece. Items that would go over the budget are skipped, so the lowest-ranked ones are dropped first.

Every item starts with the chunk's header line (`Title: … | Trial: SURMOUNT-1 (NCT04184622) | Section: …`), which
is only stored in `content`; without it the model (and verify) could not see which trial a section reports.
"""

from langchain_core.documents import Document

from clinical_rag.chunking.strategies import count_tokens

MAX_SECTION_TOKENS = 2_000
CONTEXT_BUDGET = 6_000


def _header(content: str) -> str:
    first = content.split("\n\n", 1)[0]
    return first if first.startswith("Title: ") else ""


def expand_context(docs: list[Document], budget: int = CONTEXT_BUDGET) -> list[dict]:
    """Ranked hits -> what the model reads, in rank order: {text, tokens, chunk_ids, doc_id, section}."""
    items: dict[str, dict] = {}
    used = 0
    for d in docs:
        m = d.metadata
        text = m.get("context") or d.page_content
        if m["metadata"]["content_type"] == "text" and count_tokens(text) > MAX_SECTION_TOKENS:
            text = d.page_content  # the piece, which already starts with the header
        elif (header := _header(d.page_content)) and not text.startswith(header):
            text = f"{header}\n\n{text}"
        tokens = count_tokens(text)
        if text in items:
            items[text]["chunk_ids"].append(m["chunk_id"])
        elif used + tokens <= budget:
            items[text] = {"text": text, "tokens": tokens, "chunk_ids": [m["chunk_id"]],
                           "doc_id": m["doc_id"], "section": m["section"]}
            used += tokens
    return list(items.values())
