"""Single-question RAG (ProjectSpec.md section 3, steps 2-7). services/api.ask() adds the daily cap and audit.

hybrid retrieval (top 20) -> rerank (top 8) -> context expansion (6,000-token budget) -> the generation model
writes claims with citations (structured output, one retry) -> every claim is verified. Step times are logged.
"""

import logging
import os
import time

from pydantic import BaseModel, Field

from clinical_rag.config import cost_usd
from clinical_rag.generation.verify import verify
from clinical_rag.llm import get_chat
from clinical_rag.retrieval.context import expand_context
from clinical_rag.retrieval.rerank import rerank
from clinical_rag.retrieval.retriever import HybridPostgresRetriever, to_sources
from clinical_rag.schemas import Answer, Claim, Source, Usage

log = logging.getLogger(__name__)

TOP_K = 8
MAX_OUTPUT_TOKENS = 4_000  # includes the model's reasoning tokens

SYSTEM_PROMPT = """You answer questions about obesity drugs using only the sources provided.

1. Use only the provided sources. Each source is labeled with its id and content type (text, table, figure).
2. Every claim must cite one or more source ids that directly support it.
3. Copy numbers exactly as written in the source (values, units, confidence intervals, timepoints, dose arms).
4. If information the question asks for is not in the sources, add it to not_found. Never fill gaps from general \
knowledge, even when you know the answer.
5. When citing a figure, describe only what the caption or description states; never estimate values from a figure.
6. Neutral, scientific tone. No promotional language, no treatment recommendations, no comparative superiority \
claims unless a head-to-head trial in the sources states it."""


class DraftClaim(BaseModel):
    text: str = Field(description="One short factual statement")
    citation_ids: list[str] = Field(description="Ids of the sources that directly support it")


class AnswerDraft(BaseModel):
    """What the model writes. verified/issues are set by verify(), never by the model."""

    claims: list[DraftClaim]
    not_found: list[str] = Field(description="Parts of the question the sources do not answer")


def format_sources(items: list[dict], sources: list[Source]) -> str:
    return "\n\n".join(
        f'<source id="{s.chunk_id}" type="{s.content_type}" title="{s.title}" section="{s.section or ""}">\n'
        f'{i["text"]}\n</source>'
        for i, s in zip(items, sources))


def generate(messages: list[tuple[str, str]]) -> tuple[AnswerDraft, int, int]:
    """Structured output with one retry on a format error. Returns the draft and the tokens used."""
    model = get_chat(max_tokens=MAX_OUTPUT_TOKENS).with_structured_output(AnswerDraft, include_raw=True)
    input_tokens = output_tokens = 0
    for _ in range(2):
        out = model.invoke(messages)
        usage = out["raw"].usage_metadata or {}
        input_tokens += usage.get("input_tokens", 0)
        output_tokens += usage.get("output_tokens", 0)
        if out["parsed"] is not None:
            return out["parsed"], input_tokens, output_tokens
        retry = f"Your reply did not match the required format: {out['parsing_error']}. Reply again in that format."
        messages = [*messages, ("human", retry)]
    raise ValueError(f"Answer format invalid after one retry: {out['parsing_error']}")


def answer_question(question: str) -> Answer:
    times, last = {}, time.perf_counter()

    def lap(step: str) -> None:
        nonlocal last
        now = time.perf_counter()
        times[step], last = round((now - last) * 1000), now

    docs = HybridPostgresRetriever().invoke(question)
    lap("retrieve")
    docs = rerank(docs, question, TOP_K)
    lap("rerank")
    items = expand_context(docs)
    first_hit = {d.metadata["chunk_id"]: d for d in docs}
    sources = to_sources([first_hit[i["chunk_ids"][0]] for i in items])
    lap("context")
    draft, input_tokens, output_tokens = generate(
        [("system", SYSTEM_PROMPT), ("human", f"Sources:\n\n{format_sources(items, sources)}\n\nQuestion: {question}")])
    lap("generate")
    sent = {cid: i["text"] for i in items for cid in i["chunk_ids"]}
    claims = verify([Claim(text=c.text, citation_ids=c.citation_ids) for c in draft.claims], sent)
    lap("verify")
    log.info("answer_question ms: %s", times)

    usage = Usage(input_tokens=input_tokens, output_tokens=output_tokens,
                  cost_usd=cost_usd(os.environ["GEN_MODEL_ID"], input_tokens, output_tokens),
                  latency_ms=sum(times.values()))
    return Answer(question=question, claims=claims, not_found=draft.not_found, sources=sources, usage=usage)
