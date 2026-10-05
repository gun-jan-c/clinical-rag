"""Real backend: the functions the frontend calls (ProjectSpec.md section 6.2). Same signatures as services/mock.py.

Adds to generation/ and retrieval/: saving briefs, the daily caps (checked before any model call), and the audit log.
A brief's row is written before generation starts (status 'generating'), so a brief that crashes is still listed,
as 'failed'. Events: generated, claim_flagged (one per unverified claim), failed, approved/edited/rejected, asked.
"""

import os
from collections.abc import Callable
from datetime import date, datetime, timezone

from psycopg import Connection
from psycopg.types.json import Jsonb

from clinical_rag.db import pool
from clinical_rag.generation.answer import answer_question
from clinical_rag.generation.brief import build_brief, data_as_of
from clinical_rag.generation.sections import PER_DRUG_TOP_K, trial_rows
from clinical_rag.retrieval.rerank import rerank_enabled
from clinical_rag.retrieval.retriever import (
    CANDIDATES,
    HybridPostgresRetriever,
    search,  # noqa: F401 (contract function, used as is)
)
from clinical_rag.schemas import (
    Answer,
    AuditEvent,
    Brief,
    BriefRequest,
    BriefSummary,
    EvalRunSummary,
    PipelineFilters,
    ReviewAction,
    SectionDraft,
    SectionKey,
    TrialRow,
    Usage,
    UsageStatus,
)

BRIEFS_CAP = int(os.environ.get("MAX_BRIEFS_PER_DAY", "30"))
ASKS_CAP = int(os.environ.get("MAX_ASKS_PER_DAY", "200"))
REVIEWED = {"approve": "approved", "edit": "edited", "reject": "rejected"}


class CapExceededError(Exception):
    """Daily brief or ask cap reached."""


def _day() -> date:
    """Caps count per UTC day (the same day data_as_of uses). Tests replace this with a made-up date."""
    return datetime.now(timezone.utc).date()


def _take(kind: str, cap: int) -> None:
    """Count one brief or ask for today, or raise CapExceededError if the cap is reached.
    Check and count are one statement, so two requests at the same moment can't both take the last slot."""
    assert kind in ("briefs", "asks")
    with pool().connection() as conn:
        row = conn.execute(f"""
            insert into usage_daily (day, {kind}) values (%s, 1)
            on conflict (day) do update set {kind} = usage_daily.{kind} + 1
            where usage_daily.{kind} < %s
            returning {kind}""", (_day(), cap)).fetchone()
    if row is None:
        raise CapExceededError(f"Daily limit of {cap} {'briefs' if kind == 'briefs' else 'questions'} reached")


def _log(conn: Connection, brief_id: str | None, actor: str, event: str, payload: dict) -> None:
    conn.execute("insert into audit_log (brief_id, actor, event, payload) values (%s, %s, %s, %s)",
                 (brief_id, actor, event, Jsonb(payload)))


def _config() -> dict:
    """What produced the brief, so two briefs can be compared later."""
    retriever = HybridPostgresRetriever()
    return {"strategy": retriever.strategy, "embedding_model": retriever.embedding_model,
            "candidates": CANDIDATES, "top_k_per_drug": PER_DRUG_TOP_K,
            "rerank_model": os.environ["RERANK_MODEL_ID"] if rerank_enabled() else None,
            "data_as_of": data_as_of().isoformat()}


def generate_brief(req: BriefRequest,
                   on_progress: Callable[[SectionKey, str], None] | None = None) -> Brief:
    _take("briefs", BRIEFS_CAP)
    with pool().connection() as conn:
        brief_id = conn.execute(
            "insert into briefs (requested_by, request, model_id, config) values (%s, %s, %s, %s) "
            "returning brief_id::text",
            (req.requested_by, Jsonb(req.model_dump(mode="json")), os.environ["GEN_MODEL_ID"], Jsonb(_config()))
        ).fetchone()[0]
    try:
        brief = build_brief(req, on_progress)
    except Exception as e:
        with pool().connection() as conn:
            conn.execute("update briefs set status = 'failed' where brief_id = %s", (brief_id,))
            _log(conn, brief_id, req.requested_by, "failed", {"error": f"{type(e).__name__}: {e}"})
        raise
    with pool().connection() as conn:  # one transaction: the brief, its sections and its events
        u = brief.usage
        conn.execute("""
            update briefs set status = %s, total_input_tokens = %s, total_output_tokens = %s, total_cost_usd = %s,
                              latency_ms = %s
            where brief_id = %s""",
            (brief.status, u.input_tokens, u.output_tokens, u.cost_usd, u.latency_ms, brief_id))
        for s in brief.sections:
            conn.execute("""
                insert into brief_sections (brief_id, section_key, draft, retrieved_ids, review_status)
                values (%s, %s, %s, %s, %s)""",
                (brief_id, s.section_key, Jsonb(s.model_dump(mode="json")), [x.chunk_id for x in s.sources],
                 s.review_status))
        _log(conn, brief_id, req.requested_by, "generated",
             {"drugs": req.drugs, "sections": req.sections, "status": brief.status,
              "cost_usd": u.cost_usd, "latency_ms": u.latency_ms})
        for s in brief.sections:
            for c in s.claims:
                if not c.verified:
                    _log(conn, brief_id, req.requested_by, "claim_flagged",
                         {"section_key": s.section_key, "claim": c.text, "issues": c.issues})
    return get_brief(brief_id)


def to_brief(brief_id: str, row: tuple, section_rows: list[tuple]) -> Brief:
    """Rebuild a Brief from its saved rows. The review columns, not the saved draft, hold the review state."""
    request, status, config, input_tokens, output_tokens, cost, latency_ms, created_at = row
    req = BriefRequest(**request)
    saved = {key: SectionDraft(**{**draft, "review_status": review_status, "reviewer_text": reviewer_text})
             for key, draft, review_status, reviewer_text in section_rows}
    return Brief(brief_id=brief_id, request=req, data_as_of=date.fromisoformat(config["data_as_of"]),
                 status=status, sections=[saved[k] for k in req.sections if k in saved],
                 usage=Usage(input_tokens=input_tokens or 0, output_tokens=output_tokens or 0,
                             cost_usd=float(cost or 0), latency_ms=latency_ms or 0),
                 created_at=created_at)


def get_brief(brief_id: str) -> Brief:
    with pool().connection() as conn:
        row = conn.execute("""
            select request, status, config, total_input_tokens, total_output_tokens, total_cost_usd, latency_ms,
                   created_at
            from briefs where brief_id = %s""", (brief_id,)).fetchone()
        if row is None:
            raise KeyError(f"No brief {brief_id}")
        section_rows = conn.execute("select section_key, draft, review_status, reviewer_text from brief_sections "
                                    "where brief_id = %s", (brief_id,)).fetchall()
    return to_brief(brief_id, row, section_rows)


def list_briefs(limit: int = 20) -> list[BriefSummary]:
    with pool().connection() as conn:
        rows = conn.execute("""
            select brief_id::text, requested_by, request->'drugs', status, total_cost_usd, created_at
            from briefs order by created_at desc limit %s""", (limit,)).fetchall()
    return [BriefSummary(brief_id=i, requested_by=by, drugs=drugs, status=status, cost_usd=float(cost or 0),
                         created_at=created) for i, by, drugs, status, cost, created in rows]


def brief_status(review_statuses: list[str]) -> str:
    """'approved' once every section is approved or edited; otherwise 'in_review'."""
    return "approved" if all(s in ("approved", "edited") for s in review_statuses) else "in_review"


def submit_review(action: ReviewAction) -> SectionDraft:
    status = REVIEWED[action.action]
    with pool().connection() as conn:
        row = conn.execute("""
            update brief_sections set review_status = %s, reviewer_text = %s, reviewed_by = %s, reviewed_at = now()
            where brief_id = %s and section_key = %s
            returning draft""",
            (status, action.edited_text, action.reviewer, action.brief_id, action.section_key)).fetchone()
        if row is None:
            raise KeyError(f"No section {action.section_key} in brief {action.brief_id}")
        statuses = [r[0] for r in conn.execute("select review_status from brief_sections where brief_id = %s",
                                               (action.brief_id,))]
        conn.execute("update briefs set status = %s where brief_id = %s", (brief_status(statuses), action.brief_id))
        _log(conn, action.brief_id, action.reviewer, status,
             {"section_key": action.section_key, "edited_text": action.edited_text, "comment": action.comment})
    return SectionDraft(**{**row[0], "review_status": status, "reviewer_text": action.edited_text})


def get_audit_log(brief_id: str) -> list[AuditEvent]:
    with pool().connection() as conn:
        rows = conn.execute("select actor, event, payload, created_at from audit_log where brief_id = %s "
                            "order by id", (brief_id,)).fetchall()
    return [AuditEvent(actor=a, event=e, payload=p, created_at=t) for a, e, p, t in rows]


def ask(question: str, asked_by: str) -> Answer:
    _take("asks", ASKS_CAP)
    answer = answer_question(question)
    with pool().connection() as conn:
        _log(conn, None, asked_by, "asked",
             {"question": question, "claims": len(answer.claims),
              "flagged": sum(not c.verified for c in answer.claims), "not_found": answer.not_found,
              "cost_usd": answer.usage.cost_usd, "latency_ms": answer.usage.latency_ms})
    return answer


def get_pipeline(filters: PipelineFilters) -> list[TrialRow]:
    return trial_rows(filters.drugs or None, filters.phases or None, filters.statuses or None)


def get_usage_today() -> UsageStatus:
    with pool().connection() as conn:
        row = conn.execute("select briefs, asks from usage_daily where day = %s", (_day(),)).fetchone()
    briefs, asks = row or (0, 0)
    return UsageStatus(briefs_today=briefs, briefs_cap=BRIEFS_CAP, asks_today=asks, asks_cap=ASKS_CAP)


def list_eval_runs() -> list[EvalRunSummary]:
    with pool().connection() as conn:
        rows = conn.execute("select run_id::text, created_at, config, metrics, notes from eval_runs "
                            "order by created_at desc").fetchall()
    return [EvalRunSummary(run_id=i, created_at=t, config=c, metrics=m, notes=n) for i, t, c, m, n in rows]
