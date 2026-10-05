"""One brief section (ProjectSpec.md sections 2 and 3).

- pipeline: the trials table, SQL only.
- efficacy, safety, competitive_positioning: per drug, the template's search questions -> Cohere keeps the best 8
  -> up to 6,000 tokens of context; then the model writes claims for all drugs, and every claim is verified.
- evidence_gaps: the drugs' ongoing trials plus the other sections' not_found items.
A reply that fails the format twice marks the section "failed" instead of stopping the brief.
"""

import os
import time
from collections.abc import Callable
from itertools import chain, zip_longest

from langchain_core.documents import Document

from clinical_rag.config import cost_usd
from clinical_rag.db import pool
from clinical_rag.generation.answer import format_sources, generate, seen_by_model
from clinical_rag.generation.templates import (
    EVIDENCE_GAPS_WRITE,
    SECTION_TITLES,
    TEMPLATES,
    section_prompt,
)
from clinical_rag.generation.verify import verify
from clinical_rag.retrieval.context import expand_context
from clinical_rag.retrieval.rerank import rerank
from clinical_rag.retrieval.retriever import HybridPostgresRetriever, to_sources
from clinical_rag.schemas import Claim, SectionDraft, SectionKey, TrialRow, Usage

Progress = Callable[[SectionKey, str], None]

PER_DRUG_TOP_K = 8  # spec: "keep the top 8 per drug"
MAX_SECTION_OUTPUT_TOKENS = 8_000  # includes reasoning tokens; at most ~$0.004 per section
ONGOING = ["RECRUITING", "NOT_YET_RECRUITING", "ACTIVE_NOT_RECRUITING", "ENROLLING_BY_INVITATION"]


def trial_rows(drugs: list[str] | None = None, phases: list[str] | None = None,
               statuses: list[str] | None = None) -> list[TrialRow]:
    """Trials from the registry, highest phase and newest first. None = no filter. Also behind api.get_pipeline."""
    with pool().connection() as conn:
        rows = conn.execute("""
            select t.nct_id, t.title, d.metadata->'drugs', t.phase, t.status, t.sponsor, t.enrollment, t.first_posted
            from trials t join documents d on d.doc_id = 'ctgov:' || t.nct_id
            where (%(drugs)s::text[] is null or d.metadata->'drugs' ?| %(drugs)s)
              and (%(phases)s::text[] is null or t.phase && %(phases)s)
              and (%(statuses)s::text[] is null or t.status = any(%(statuses)s))
            order by t.phase desc, t.first_posted desc""",
            {"drugs": drugs, "phases": phases, "statuses": statuses}).fetchall()
    return [TrialRow(nct_id=nct, title=title or "",
                     drug=", ".join(d for d in (drugs or row_drugs) if d in row_drugs),
                     phase=phase or [], status=status, sponsor=sponsor, enrollment=enrollment,
                     first_posted=first_posted, url=f"https://clinicaltrials.gov/study/{nct}")
            for nct, title, row_drugs, phase, status, sponsor, enrollment, first_posted in rows]


def pipeline_section(drugs: list[str], progress: Progress) -> SectionDraft:
    start = time.perf_counter()
    progress("pipeline", "retrieving")
    table = trial_rows(drugs)
    progress("pipeline", "done")
    return SectionDraft(section_key="pipeline", title=SECTION_TITLES["pipeline"], table=table,
                        usage=Usage(latency_ms=round((time.perf_counter() - start) * 1000)))


def _merge(per_drug_items: list[list[dict]]) -> list[dict]:
    """Context items of all drugs; a source found for two drugs (e.g. a head-to-head trial) is sent once."""
    merged: dict[str, dict] = {}
    for item in chain.from_iterable(per_drug_items):
        if item["text"] in merged:
            merged[item["text"]]["chunk_ids"] += [c for c in item["chunk_ids"] if c not in merged[item["text"]]["chunk_ids"]]
        else:
            merged[item["text"]] = {**item, "chunk_ids": list(item["chunk_ids"])}
    return list(merged.values())


def _write(key: SectionKey, drugs: list[str], items: list[dict], hits: dict[str, Document], instructions: str,
           extra: str, progress: Progress, start: float) -> SectionDraft:
    """Generate claims from the context items, verify them, and package the section."""
    sources = to_sources([hits[i["chunk_ids"][0]] for i in items])
    progress(key, "generating")
    human = f"Drugs: {', '.join(drugs)}\n\n{extra}Sources:\n\n{format_sources(items, sources)}"
    try:
        draft, input_tokens, output_tokens = generate([("system", section_prompt(key, instructions)),
                                                       ("human", human)], MAX_SECTION_OUTPUT_TOKENS)
    except ValueError as e:
        progress(key, "failed")
        return SectionDraft(section_key=key, title=SECTION_TITLES[key], sources=sources, review_status="failed",
                            not_found=[f"Section could not be generated: {e}"])
    progress(key, "verifying")
    sent = seen_by_model(items, sources)
    claims = verify([Claim(text=c.text, citation_ids=c.citation_ids) for c in draft.claims], sent)
    usage = Usage(input_tokens=input_tokens, output_tokens=output_tokens,
                  cost_usd=cost_usd(os.environ["GEN_MODEL_ID"], input_tokens, output_tokens),
                  latency_ms=round((time.perf_counter() - start) * 1000))
    progress(key, "done")
    return SectionDraft(section_key=key, title=SECTION_TITLES[key], claims=claims, not_found=draft.not_found,
                        sources=sources, usage=usage)


def llm_section(key: SectionKey, drugs: list[str], progress: Progress) -> SectionDraft:
    start = time.perf_counter()
    template = TEMPLATES[key]
    progress(key, "retrieving")
    per_drug_items, hits = [], {}
    for drug in drugs:
        lists = [HybridPostgresRetriever(drugs=[drug]).invoke(q.format(drug=drug)) for q in template["queries"]]
        candidates, seen = [], set()
        for d in chain.from_iterable(zip_longest(*lists)):  # take turns between the questions, no duplicates
            if d is not None and d.metadata["chunk_id"] not in seen:
                seen.add(d.metadata["chunk_id"])
                candidates.append(d)
        top = rerank(candidates, template["rerank"].format(drug=drug), PER_DRUG_TOP_K)
        hits |= {d.metadata["chunk_id"]: d for d in top}
        per_drug_items.append(expand_context(top))  # 6,000 tokens per drug
    return _write(key, drugs, _merge(per_drug_items), hits, template["write"], "", progress, start)


def _ongoing_trials(drug: str) -> list[Document]:
    """Each ongoing trial's summary, headed by its status, phase and enrollment so claims about them can be verified."""
    with pool().connection() as conn:
        rows = conn.execute("""
            select distinct on (c.doc_id) c.chunk_id, c.doc_id, c.section, c.content, c.context, c.metadata,
                   t.status, t.phase, t.enrollment
            from chunks c join trials t on c.doc_id = 'ctgov:' || t.nct_id
            where c.strategy = 'section' and c.section = 'Brief summary' and t.status = any(%s)
              and c.metadata->'drugs' ? %s
            order by c.doc_id, c.chunk_id""", (ONGOING, drug)).fetchall()
    rows.sort(key=lambda r: (r[7] or [], r[8] or 0), reverse=True)  # latest phase, then largest, first
    docs = []
    for chunk_id, doc_id, section, content, context, metadata, status, phase, enrollment in rows:
        header = f"Status: {status}; phase: {', '.join(phase or []) or 'not stated'}; enrollment: {enrollment}"
        docs.append(Document(page_content=content, metadata={"chunk_id": chunk_id, "doc_id": doc_id,
                                                             "section": section, "metadata": metadata,
                                                             "context": f"{header}\n\n{context}"}))
    return docs


def evidence_gaps_section(drugs: list[str], open_questions: list[str], progress: Progress) -> SectionDraft:
    start = time.perf_counter()
    progress("evidence_gaps", "retrieving")
    per_drug_items, hits = [], {}
    for drug in drugs:
        trials = _ongoing_trials(drug)
        hits |= {d.metadata["chunk_id"]: d for d in trials}
        per_drug_items.append(expand_context(trials))  # 6,000 tokens per drug
    extra = "".join(f"- {q}\n" for q in open_questions)
    extra = f"Open questions reported by the other sections:\n{extra}\n" if extra else ""
    return _write("evidence_gaps", drugs, _merge(per_drug_items), hits, EVIDENCE_GAPS_WRITE, extra, progress, start)
