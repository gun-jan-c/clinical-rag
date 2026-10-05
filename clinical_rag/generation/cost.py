"""Cost of a brief (ProjectSpec.md section 5.7). Per-call cost comes from config.cost_usd with token counts from
LangChain's usage_metadata; this adds the sections up. Latency is the brief's wall-clock time, not the sum, because
sections run at the same time."""

from clinical_rag.schemas import Usage


def total(usages: list[Usage], latency_ms: int) -> Usage:
    return Usage(input_tokens=sum(u.input_tokens for u in usages),
                 output_tokens=sum(u.output_tokens for u in usages),
                 cost_usd=round(sum(u.cost_usd for u in usages), 6), latency_ms=latency_ms)
