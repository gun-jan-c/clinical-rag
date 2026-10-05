"""Hybrid retrieval over Postgres (ProjectSpec.md sections 3 and 5.4) and the `search()` behind the Search Lab.

One `hybrid_search` SQL call per filter combination (drug x content type) returns candidates with their
dense rank, keyword rank, and RRF score. Dense-only and keyword-only modes are read from those ranks.
Results from several drugs are interleaved by rank, so one drug with many papers cannot crowd out the rest.
"""

from functools import cache
from itertools import product

import numpy as np
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from psycopg.types.json import Jsonb

from clinical_rag.db import drug_aliases, pool
from clinical_rag.llm import EMBEDDING_DIMENSIONS, get_embeddings
from clinical_rag.retrieval.query_expansion import expand
from clinical_rag.retrieval.rerank import rerank as rerank_docs
from clinical_rag.schemas import (
    ContentType,
    EmbeddingModel,
    SearchHit,
    SearchMode,
    Source,
    Strategy,
)

CANDIDATES = 20  # per filter combination, before rerank (spec: "20 candidates per sub-query")
CANDIDATE_POOL = 50  # hybrid_search's own dense and keyword pools
SNIPPET_CHARS = 300

RANK_KEY = {"hybrid": "rrf_score", "dense": "dense_rank", "keyword": "keyword_rank"}


@cache
def _embedder(model: EmbeddingModel):
    return get_embeddings(model, max_retries=6)


def rank_rows(rows: list[dict], mode: SearchMode) -> list[dict]:
    """Order one hybrid_search result for a mode. Dense/keyword drop rows that the other side found alone."""
    if mode == "hybrid":
        return sorted(rows, key=lambda r: -r["rrf_score"])
    key = RANK_KEY[mode]
    return sorted((r for r in rows if r[key] is not None), key=lambda r: r[key])


def interleave(lists: list[list[dict]], limit: int) -> list[dict]:
    """Round-robin over ranked lists (one per drug / content type), skipping chunks already taken."""
    out, seen = [], set()
    for i in range(max((len(x) for x in lists), default=0)):
        for ranked in lists:
            if i < len(ranked) and ranked[i]["chunk_id"] not in seen:
                seen.add(ranked[i]["chunk_id"])
                out.append(ranked[i])
    return out[:limit]


def snippet(content: str) -> str:
    """Chunk text without the "Title: ... | Section: ..." prefix, cut to a short preview."""
    body = content.split("\n\n", 1)[1] if content.startswith("Title: ") and "\n\n" in content else content
    body = " ".join(body.split())
    return body if len(body) <= SNIPPET_CHARS else body[:SNIPPET_CHARS].rsplit(" ", 1)[0] + " ..."


class HybridPostgresRetriever(BaseRetriever):
    """LangChain retriever over `hybrid_search`. Each Document's page_content is the chunk `content`;
    metadata carries chunk_id, doc_id, section, the ranks, and `context` (what the LLM will read)."""

    mode: SearchMode = "hybrid"
    strategy: Strategy = "section"
    embedding_model: EmbeddingModel = "oai-3-small"
    drugs: list[str] | None = None
    content_types: list[ContentType] | None = None
    limit: int = CANDIDATES

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        if self.mode == "keyword":  # no embedding needed; the dense side is ignored
            vector = np.zeros(EMBEDDING_DIMENSIONS, dtype=np.float32)
        else:
            vector = np.array(_embedder(self.embedding_model).embed_query(query), dtype=np.float32)
        # Hybrid asks for the fused top N; the single modes ask for everything so their own top N is complete.
        match_count = CANDIDATES if self.mode == "hybrid" else 2 * CANDIDATE_POOL
        with pool().connection() as conn:
            keyword_text = expand(query, drug_aliases(conn))
            ranked_lists = []
            for drug, content_type in product(self.drugs or [None], self.content_types or [None]):
                where = {k: v for k, v in [("drugs", [drug] if drug else None), ("content_type", content_type)] if v}
                cur = conn.execute(
                    "select * from hybrid_search(%s, %s, %s, %s, %s, %s, %s)",
                    (keyword_text, vector, self.embedding_model, self.strategy, Jsonb(where), match_count,
                     CANDIDATE_POOL))
                cols = [c.name for c in cur.description]
                ranked_lists.append(rank_rows([dict(zip(cols, r)) for r in cur.fetchall()], self.mode))
        return [Document(page_content=r.pop("content"), metadata=r) for r in interleave(ranked_lists, self.limit)]


def to_sources(docs: list[Document]) -> list[Source]:
    ids = list({d.metadata["doc_id"] for d in docs})
    with pool().connection() as conn:
        rows = conn.execute("select doc_id, source, title, url, published_date, license from documents "
                            "where doc_id = any(%s)", (ids,)).fetchall()
    meta = {r[0]: r for r in rows}
    out = []
    for d in docs:
        m = d.metadata
        _, source, title, url, published, license_ = meta[m["doc_id"]]
        out.append(Source(chunk_id=m["chunk_id"], doc_id=m["doc_id"], source=source,
                          content_type=m["metadata"]["content_type"], title=title or "", url=url,
                          published_date=published, license=license_, section=m["section"],
                          label=m["metadata"].get("label"), snippet=snippet(d.page_content)))
    return out


def search(query: str, mode: SearchMode = "hybrid", strategy: Strategy = "section",
           embedding_model: EmbeddingModel = "oai-3-small", drugs: list[str] | None = None,
           content_types: list[ContentType] | None = None,
           k: int = 10, rerank: bool = True) -> list[SearchHit]:
    """Contract function (ProjectSpec.md section 6.2): top k chunks, with ranks for the Search Lab."""
    retriever = HybridPostgresRetriever(mode=mode, strategy=strategy, embedding_model=embedding_model,
                                        drugs=drugs, content_types=content_types)
    docs = retriever.invoke(query)
    docs = rerank_docs(docs, query, k) if rerank else docs[:k]
    return [SearchHit(source=s, dense_rank=d.metadata["dense_rank"], keyword_rank=d.metadata["keyword_rank"],
                      rrf_score=d.metadata["rrf_score"], rerank_score=d.metadata.get("rerank_score"))
            for s, d in zip(to_sources(docs), docs)]

