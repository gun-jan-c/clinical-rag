"""Fake backend: same functions as services/api.py (ProjectSpec.md section 6.2), no database or model calls.

Used when USE_MOCK=1 so the frontend can be built before the real backend exists.
Every title, number, and trial ID here is made up and marked "[Mock]"; none are clinical facts.
"""

import os
import time
import uuid
from collections.abc import Callable
from datetime import date, datetime, timezone

from dotenv import load_dotenv

from clinical_rag.schemas import (
    Answer,
    AuditEvent,
    Brief,
    BriefRequest,
    BriefSummary,
    Claim,
    ContentType,
    EmbeddingModel,
    EvalRunSummary,
    PipelineFilters,
    ReviewAction,
    SearchHit,
    SearchMode,
    SectionDraft,
    SectionKey,
    Source,
    Strategy,
    TrialRow,
    Usage,
    UsageStatus,
)

load_dotenv()

DELAY_SECONDS = 0.3  # simulated latency per step; tests set this to 0
DATA_AS_OF = date(2026, 10, 1)
BRIEFS_CAP = int(os.environ.get("MAX_BRIEFS_PER_DAY", "30"))
ASKS_CAP = int(os.environ.get("MAX_ASKS_PER_DAY", "200"))
OUT_OF_SCOPE_WORDS = ("price", "alzheimer")  # questions containing these return not_found


class CapExceededError(Exception):
    """Daily brief or ask cap reached."""


SECTION_TITLES: dict[SectionKey, str] = {
    "pipeline": "Development pipeline by phase",
    "efficacy": "Key efficacy results",
    "safety": "Safety and tolerability",
    "competitive_positioning": "Competitive positioning",
    "evidence_gaps": "Open questions and evidence gaps",
}

DRUGS = ["semaglutide", "tirzepatide", "liraglutide", "orforglipron",
         "retatrutide", "survodutide", "cagrilintide", "mazdutide"]

# One source of each content type, so the UI can show text, table, and figure badges.
SOURCES = [
    Source(chunk_id="pubmed:00000001:section:0", doc_id="pubmed:00000001", source="pubmed",
           content_type="text", title="[Mock] Randomized trial of a weekly incretin therapy in adults with obesity",
           url="https://pubmed.ncbi.nlm.nih.gov/", published_date=date(2024, 6, 1), section="Results",
           snippet="[Mock] Mean change in body weight at week 72 was X% with treatment versus Y% with placebo."),
    Source(chunk_id="label:mock-set-1:section:3", doc_id="label:mock-set-1", source="dailymed",
           content_type="table", title="[Mock] Prescribing information", url="https://dailymed.nlm.nih.gov/",
           published_date=date(2025, 3, 1), section="Adverse Reactions", label="Table 1",
           snippet="[Mock] Adverse reactions occurring in at least 5% of patients, by dose."),
    Source(chunk_id="epmc:PMC0000001:section:7", doc_id="epmc:PMC0000001", source="europepmc",
           content_type="figure", title="[Mock] Long-term weight outcomes with an incretin therapy",
           url="https://europepmc.org/", published_date=date(2025, 1, 15), license="cc by",
           section="Results", label="Figure 2",
           snippet="[Mock] Mean percent change in body weight over 72 weeks by treatment arm."),
]

TRIALS = [
    TrialRow(nct_id=f"NCT9{2 * i + j:07d}", title=f"[Mock] Phase {phase} study of {drug} in adults with obesity",
             drug=drug, phase=[f"PHASE{phase}"], status=status, sponsor="[Mock] Sponsor",
             enrollment=300 * phase, first_posted=date(2023, 1 + i, 1), url="https://clinicaltrials.gov/")
    for i, drug in enumerate(DRUGS)
    for j, (phase, status) in enumerate([(3, "RECRUITING"), (2, "COMPLETED")])
]

_briefs: dict[str, Brief] = {}
_audit: dict[str, list[AuditEvent]] = {}
_counts = {"briefs": 0, "asks": 0}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _log(brief_id: str, actor: str, event: str, payload: dict) -> None:
    _audit.setdefault(brief_id, []).append(AuditEvent(actor=actor, event=event, payload=payload, created_at=_now()))


def _section(key: SectionKey, drugs: list[str]) -> SectionDraft:
    if key == "pipeline":
        return SectionDraft(section_key=key, title=SECTION_TITLES[key],
                            table=[t for t in TRIALS if t.drug in drugs], usage=Usage(latency_ms=40))
    cited = SOURCES[1] if key == "safety" else SOURCES[0]
    claims = [Claim(text=f"[Mock] Example {key} finding for {drug} (X%).", citation_ids=[cited.chunk_id])
              for drug in drugs]
    # One flagged claim per section so the Review page can show what an unverified claim looks like.
    claims.append(Claim(text="[Mock] A claim whose number (12.3%) is not in its cited source.",
                        citation_ids=[SOURCES[2].chunk_id], verified=False,
                        issues=["Number 12.3% not found in cited sources"]))
    return SectionDraft(section_key=key, title=SECTION_TITLES[key], claims=claims,
                        not_found=[f"[Mock] No head-to-head data found for {drugs[0]}"], sources=SOURCES,
                        usage=Usage(input_tokens=6000, output_tokens=800, cost_usd=0.004, latency_ms=9000))


def generate_brief(req: BriefRequest,
                   on_progress: Callable[[SectionKey, str], None] | None = None) -> Brief:
    if _counts["briefs"] >= BRIEFS_CAP:
        raise CapExceededError(f"Daily limit of {BRIEFS_CAP} briefs reached")
    _counts["briefs"] += 1
    sections = []
    for key in req.sections:
        for step in ("retrieving", "generating", "verifying"):
            if on_progress:
                on_progress(key, step)
            time.sleep(DELAY_SECONDS)
        sections.append(_section(key, req.drugs))
        if on_progress:
            on_progress(key, "done")
    usage = Usage(input_tokens=sum(s.usage.input_tokens for s in sections),
                  output_tokens=sum(s.usage.output_tokens for s in sections),
                  cost_usd=round(sum(s.usage.cost_usd for s in sections), 4),
                  latency_ms=sum(s.usage.latency_ms for s in sections))
    brief = Brief(brief_id=str(uuid.uuid4()), request=req, data_as_of=DATA_AS_OF, status="draft",
                  sections=sections, usage=usage, created_at=_now())
    _briefs[brief.brief_id] = brief
    _log(brief.brief_id, req.requested_by, "generated", {"sections": req.sections, "drugs": req.drugs})
    return brief


def get_brief(brief_id: str) -> Brief:
    return _briefs[brief_id]


def list_briefs(limit: int = 20) -> list[BriefSummary]:
    newest = sorted(_briefs.values(), key=lambda b: b.created_at, reverse=True)[:limit]
    return [BriefSummary(brief_id=b.brief_id, requested_by=b.request.requested_by, drugs=b.request.drugs,
                         status=b.status, cost_usd=b.usage.cost_usd, created_at=b.created_at) for b in newest]


def submit_review(action: ReviewAction) -> SectionDraft:
    brief = get_brief(action.brief_id)
    section = next(s for s in brief.sections if s.section_key == action.section_key)
    section.review_status = {"approve": "approved", "edit": "edited", "reject": "rejected"}[action.action]
    section.reviewer_text = action.edited_text
    all_approved = all(s.review_status in ("approved", "edited") for s in brief.sections)
    brief.status = "approved" if all_approved else "in_review"
    _log(brief.brief_id, action.reviewer, section.review_status,
         {"section_key": action.section_key, "comment": action.comment})
    return section


def get_audit_log(brief_id: str) -> list[AuditEvent]:
    return _audit.get(brief_id, [])


def ask(question: str, asked_by: str) -> Answer:
    if _counts["asks"] >= ASKS_CAP:
        raise CapExceededError(f"Daily limit of {ASKS_CAP} questions reached")
    _counts["asks"] += 1
    time.sleep(DELAY_SECONDS)
    usage = Usage(input_tokens=3000, output_tokens=200, cost_usd=0.002, latency_ms=4000)
    if any(word in question.lower() for word in OUT_OF_SCOPE_WORDS):
        return Answer(question=question, claims=[], not_found=[question], sources=[], usage=usage)
    return Answer(question=question,
                  claims=[Claim(text=f"[Mock] Answer to: {question}", citation_ids=[SOURCES[0].chunk_id])],
                  not_found=[], sources=[SOURCES[0]], usage=usage)


def search(query: str, mode: SearchMode = "hybrid", strategy: Strategy = "section",
           embedding_model: EmbeddingModel = "oai-3-small", drugs: list[str] | None = None,
           content_types: list[ContentType] | None = None,
           k: int = 10, rerank: bool = True) -> list[SearchHit]:
    time.sleep(DELAY_SECONDS)
    matches = [s for s in SOURCES if not content_types or s.content_type in content_types][:k]
    hits = []
    for i, source in enumerate(matches, start=1):
        dense = i if mode != "keyword" else None
        keyword = len(matches) - i + 1 if mode != "dense" else None  # reversed, so the columns differ
        rrf = sum(1 / (60 + r) for r in (dense, keyword) if r is not None)
        hits.append(SearchHit(source=source, dense_rank=dense, keyword_rank=keyword, rrf_score=rrf,
                              rerank_score=round(0.9 - 0.1 * i, 2) if rerank else None))
    return hits


def get_pipeline(filters: PipelineFilters) -> list[TrialRow]:
    return [t for t in TRIALS
            if (not filters.drugs or t.drug in filters.drugs)
            and (not filters.phases or set(t.phase) & set(filters.phases))
            and (not filters.statuses or t.status in filters.statuses)]


def get_usage_today() -> UsageStatus:
    return UsageStatus(briefs_today=_counts["briefs"], briefs_cap=BRIEFS_CAP,
                       asks_today=_counts["asks"], asks_cap=ASKS_CAP)


def list_eval_runs() -> list[EvalRunSummary]:
    configs = [("fixed", False, 0.62), ("section", True, 0.81)]
    return [EvalRunSummary(run_id=f"mock-run-{n}", created_at=_now(),
                           config={"strategy": strategy, "embedding_model": "oai-3-small", "mode": "hybrid",
                                   "rerank": rerank, "k": 10},
                           metrics={"recall_at_5": recall - 0.1, "recall_at_10": recall, "mrr": recall - 0.2},
                           notes="[Mock] Made-up numbers for UI development")
            for n, (strategy, rerank, recall) in enumerate(configs, start=1)]
