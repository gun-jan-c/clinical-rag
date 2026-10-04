"""Context expansion (ProjectSpec.md section 3, step 5): search small pieces, give the model the bigger thing.

A hit is found by its short `content`; the model reads its `context`: the full section for text, the full
markdown table (+ footnotes) for tables, caption + description for figures. Hits that share a section send it
once. Text sections over MAX_SECTION_TOKENS (the biggest ~10%, up to ~17k tokens) send only the matched
piece. Items that would go over the budget are skipped, so the lowest-ranked ones are dropped first.
"""

from langchain_core.documents import Document

from clinical_rag.chunking.strategies import count_tokens

MAX_SECTION_TOKENS = 2_000
CONTEXT_BUDGET = 6_000


def expand_context(docs: list[Document], budget: int = CONTEXT_BUDGET) -> list[dict]:
    """Ranked hits -> what the model reads, in rank order: {text, tokens, chunk_ids, doc_id, section}."""
    items: dict[str, dict] = {}
    used = 0
    for d in docs:
        m = d.metadata
        text = m.get("context") or d.page_content
        tokens = count_tokens(text)
        if m["metadata"]["content_type"] == "text" and tokens > MAX_SECTION_TOKENS:
            text, tokens = d.page_content, count_tokens(d.page_content)
        if text in items:
            items[text]["chunk_ids"].append(m["chunk_id"])
        elif used + tokens <= budget:
            items[text] = {"text": text, "tokens": tokens, "chunk_ids": [m["chunk_id"]],
                           "doc_id": m["doc_id"], "section": m["section"]}
            used += tokens
    return list(items.values())
