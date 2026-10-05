"""services/api.py. The Railway tests write real rows (no model calls: build_brief and answer_question are faked)
and delete them afterwards; usage is counted under a made-up day so today's real caps are untouched."""

import os
from datetime import date, datetime, timezone

import pytest

from clinical_rag.schemas import (
    Answer,
    Brief,
    BriefRequest,
    Claim,
    PipelineFilters,
    ReviewAction,
    SectionDraft,
    Usage,
)
from clinical_rag.services import api

ACTOR = "pytest"
TEST_DAY = date(2000, 1, 1)

railway = pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="needs DATABASE_URL (Railway)")


def test_brief_is_approved_only_when_every_section_is_approved_or_edited():
    assert api.brief_status(["approved", "edited"]) == "approved"
    assert api.brief_status(["approved", "pending"]) == "in_review"
    assert api.brief_status(["approved", "rejected"]) == "in_review"


def test_to_brief_takes_review_state_from_the_columns_and_keeps_requested_order():
    req = BriefRequest(drugs=["tirzepatide"], sections=["safety", "pipeline"], requested_by="me")
    row = (req.model_dump(mode="json"), "in_review", {"data_as_of": "2026-10-04"}, 10, 2, 0.0123, 900,
           datetime(2026, 10, 4, tzinfo=timezone.utc))
    draft = lambda key: SectionDraft(section_key=key, title=key).model_dump(mode="json")
    b = api.to_brief("id-1", row, [("pipeline", draft("pipeline"), "pending", None),
                                   ("safety", draft("safety"), "edited", "New text")])
    assert [s.section_key for s in b.sections] == ["safety", "pipeline"]
    assert (b.sections[0].review_status, b.sections[0].reviewer_text) == ("edited", "New text")
    assert b.data_as_of == date(2026, 10, 4) and b.usage.cost_usd == 0.0123


@pytest.fixture
def db(monkeypatch):
    """Made-up usage day; afterwards delete everything the test wrote."""
    monkeypatch.setattr(api, "_day", lambda: TEST_DAY)
    yield
    with api.pool().connection() as conn:
        conn.execute("delete from briefs where requested_by = %s", (ACTOR,))  # sections and events cascade
        conn.execute("delete from audit_log where actor = %s", (ACTOR,))
        conn.execute("delete from usage_daily where day = %s", (TEST_DAY,))


def fake_brief(req, on_progress=None):
    flagged = Claim(text="Weight fell 99.9%.", citation_ids=["x"], verified=False, issues=["Number 99.9 not found"])
    sections = [SectionDraft(section_key="efficacy", title="Key efficacy results",
                             claims=[Claim(text="ok", citation_ids=["x"]), flagged]),
                SectionDraft(section_key="safety", title="Safety and tolerability", review_status="failed")]
    return Brief(brief_id="replaced", request=req, data_as_of=date(2026, 10, 4), status="draft", sections=sections,
                 usage=Usage(input_tokens=100, output_tokens=10, cost_usd=0.0011, latency_ms=500),
                 created_at=datetime.now(timezone.utc))


@railway
def test_cap_refuses_before_any_model_call(db, monkeypatch):
    calls = []
    monkeypatch.setattr(api, "BRIEFS_CAP", 1)
    monkeypatch.setattr(api, "build_brief", lambda req, on_progress=None: calls.append(req) or fake_brief(req))
    req = BriefRequest(drugs=["tirzepatide"], sections=["efficacy", "safety"], requested_by=ACTOR)
    api.generate_brief(req)
    with pytest.raises(api.CapExceededError):
        api.generate_brief(req)
    assert len(calls) == 1
    assert api.get_usage_today().briefs_today == 1


@railway
def test_brief_is_saved_flagged_claims_logged_and_reviews_update_status(db, monkeypatch):
    monkeypatch.setattr(api, "build_brief", fake_brief)
    req = BriefRequest(drugs=["tirzepatide"], sections=["efficacy", "safety"], requested_by=ACTOR)
    b = api.generate_brief(req)
    assert b.brief_id != "replaced" and b.status == "draft"
    assert api.get_brief(b.brief_id) == b
    assert b.sections[1].review_status == "failed"
    assert any(s.brief_id == b.brief_id for s in api.list_briefs())

    api.submit_review(ReviewAction(brief_id=b.brief_id, section_key="efficacy", action="approve", reviewer=ACTOR))
    assert api.get_brief(b.brief_id).status == "in_review"
    section = api.submit_review(ReviewAction(brief_id=b.brief_id, section_key="safety", action="edit",
                                             reviewer=ACTOR, edited_text="Rewritten by hand"))
    assert (section.review_status, section.reviewer_text) == ("edited", "Rewritten by hand")
    assert api.get_brief(b.brief_id).status == "approved"

    events = [(e.event, e.payload.get("section_key")) for e in api.get_audit_log(b.brief_id)]
    assert events == [("generated", None), ("claim_flagged", "efficacy"), ("approved", "efficacy"),
                      ("edited", "safety")]


@railway
def test_a_crashed_brief_is_kept_as_failed(db, monkeypatch):
    def crash(req, on_progress=None):
        raise RuntimeError("Railway went away")

    monkeypatch.setattr(api, "build_brief", crash)
    with pytest.raises(RuntimeError):
        api.generate_brief(BriefRequest(drugs=["tirzepatide"], sections=["efficacy"], requested_by=ACTOR))
    saved = next(s for s in api.list_briefs() if s.requested_by == ACTOR)
    assert saved.status == "failed"
    assert [e.event for e in api.get_audit_log(saved.brief_id)] == ["failed"]


@railway
def test_ask_is_counted_and_logged(db, monkeypatch):
    answer = Answer(question="q", claims=[], not_found=["q"], sources=[], usage=Usage(cost_usd=0.0005))
    monkeypatch.setattr(api, "answer_question", lambda question: answer)
    assert api.ask("q", ACTOR) == answer
    assert api.get_usage_today().asks_today == 1
    with api.pool().connection() as conn:
        payload = conn.execute("select payload from audit_log where actor = %s and event = 'asked' "
                               "and brief_id is null", (ACTOR,)).fetchone()[0]
    assert payload["not_found"] == ["q"] and payload["cost_usd"] == 0.0005


@railway
def test_pipeline_filters():
    rows = api.get_pipeline(PipelineFilters(drugs=["tirzepatide"], phases=["PHASE3"]))
    assert rows and all("PHASE3" in r.phase and r.drug == "tirzepatide" for r in rows)
