"""Cohere Rerank on the hybrid-search candidates (ProjectSpec.md section 5.8).

RERANK_ENABLED=false skips the call and keeps the hybrid order, so everything works without Cohere.
The Cohere trial key allows 10 rerank calls per minute and 1,000 API calls per month.
"""

import os

from langchain_core.documents import Document

from clinical_rag.llm import get_reranker


def rerank_enabled() -> bool:
    return os.environ.get("RERANK_ENABLED", "true").strip().lower() == "true"


def rerank(docs: list[Document], query: str, top_n: int) -> list[Document]:
    """Best `top_n` documents by Cohere relevance score (stored in metadata["rerank_score"])."""
    if not docs or not rerank_enabled():
        return docs[:top_n]
    ranked = get_reranker(top_n=top_n).compress_documents(docs, query)
    for doc in ranked:
        doc.metadata["rerank_score"] = doc.metadata.pop("relevance_score")
    return list(ranked)
