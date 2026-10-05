"""Cohere Rerank on the hybrid-search candidates (ProjectSpec.md section 5.8).

RERANK_ENABLED=false skips the call and keeps the hybrid order, so everything works without Cohere.
The Cohere trial key allows 10 rerank calls per minute and 1,000 API calls per month.
"""

import os
import time

from cohere.errors import TooManyRequestsError
from langchain_core.documents import Document

from clinical_rag.llm import get_reranker

RATE_LIMIT_WAIT_SECONDS = 10
RATE_LIMIT_TRIES = 6  # ~1 minute in all: the trial key's limit is per minute


def rerank_enabled() -> bool:
    return os.environ.get("RERANK_ENABLED", "true").strip().lower() == "true"


def rerank(docs: list[Document], query: str, top_n: int) -> list[Document]:
    """Best `top_n` documents by Cohere relevance score (stored in metadata["rerank_score"]).
    When Cohere answers "too many requests", wait and try again, up to about a minute."""
    if not docs or not rerank_enabled():
        return docs[:top_n]
    for attempt in range(RATE_LIMIT_TRIES):
        try:
            ranked = get_reranker(top_n=top_n).compress_documents(docs, query)
            break
        except TooManyRequestsError:
            if attempt == RATE_LIMIT_TRIES - 1:
                raise
            time.sleep(RATE_LIMIT_WAIT_SECONDS)
    for doc in ranked:
        doc.metadata["rerank_score"] = doc.metadata.pop("relevance_score")
    return list(ranked)
