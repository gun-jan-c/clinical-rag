"""CONTRACT: shared Pydantic models between backend and frontend (ProjectSpec.md section 6.1).

Frozen. Changing anything here needs an architect decision and a decision-log row.
"""

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field

SectionKey = Literal["pipeline", "efficacy", "safety", "competitive_positioning", "evidence_gaps"]
SearchMode = Literal["dense", "keyword", "hybrid"]
Strategy = Literal["fixed", "section"]
EmbeddingModel = Literal["oai-3-small", "oai-3-large"]
ContentType = Literal["text", "table", "figure"]
SourceName = Literal["clinicaltrials.gov", "pubmed", "europepmc", "dailymed"]


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0


class Source(BaseModel):
    chunk_id: str
    doc_id: str
    source: SourceName
    content_type: ContentType
    title: str
    url: str
    published_date: date | None = None
    license: str | None = None
    section: str | None = None
    label: str | None = None  # e.g. "Table 2", "Figure 1"
    snippet: str  # short text shown in the UI (never a full figure or table dump)


class Claim(BaseModel):
    text: str
    citation_ids: list[str]
    verified: bool = True
    issues: list[str] = Field(default_factory=list)


class TrialRow(BaseModel):
    nct_id: str
    title: str
    drug: str | None
    phase: list[str]
    status: str | None  # as of ingestion date
    sponsor: str | None
    enrollment: int | None
    first_posted: date | None
    url: str


class SectionDraft(BaseModel):
    section_key: SectionKey
    title: str
    claims: list[Claim] = Field(default_factory=list)
    table: list[TrialRow] | None = None  # only for 'pipeline'
    not_found: list[str] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    usage: Usage = Usage()
    review_status: Literal["pending", "approved", "edited", "rejected", "failed"] = "pending"
    reviewer_text: str | None = None


class BriefRequest(BaseModel):
    indication: Literal["obesity"] = "obesity"
    drugs: list[str]
    sections: list[SectionKey]
    requested_by: str


class Brief(BaseModel):
    brief_id: str
    request: BriefRequest
    data_as_of: date  # date of the last ingestion run
    status: Literal["generating", "draft", "in_review", "approved", "failed"]
    sections: list[SectionDraft]
    usage: Usage
    created_at: datetime


class BriefSummary(BaseModel):
    brief_id: str
    requested_by: str
    drugs: list[str]
    status: str
    cost_usd: float
    created_at: datetime


class ReviewAction(BaseModel):
    brief_id: str
    section_key: SectionKey
    action: Literal["approve", "edit", "reject"]
    reviewer: str
    edited_text: str | None = None
    comment: str | None = None


class AuditEvent(BaseModel):
    actor: str
    event: str
    payload: dict
    created_at: datetime


class SearchHit(BaseModel):
    source: Source
    dense_rank: int | None
    keyword_rank: int | None
    rrf_score: float | None
    rerank_score: float | None = None


class Answer(BaseModel):
    question: str
    claims: list[Claim]
    not_found: list[str]
    sources: list[Source]
    usage: Usage


class PipelineFilters(BaseModel):
    drugs: list[str] | None = None
    phases: list[str] | None = None
    statuses: list[str] | None = None


class UsageStatus(BaseModel):
    briefs_today: int
    briefs_cap: int
    asks_today: int
    asks_cap: int


class EvalRunSummary(BaseModel):
    run_id: str
    created_at: datetime
    config: dict  # strategy, embedding_model, mode, rerank, k
    metrics: dict  # recall_at_5, recall_at_10, mrr, citation_valid, numbers_grounded, not_found_accuracy, by question type
    notes: str | None
