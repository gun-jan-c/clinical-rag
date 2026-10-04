import pytest

from clinical_rag.schemas import BriefRequest, PipelineFilters, ReviewAction
from clinical_rag.services import mock


@pytest.fixture(autouse=True)
def fast_mock(monkeypatch):
    monkeypatch.setattr(mock, "DELAY_SECONDS", 0)


def make_brief(sections=("pipeline", "safety")):
    req = BriefRequest(drugs=["tirzepatide", "semaglutide"], sections=list(sections), requested_by="tester")
    return mock.generate_brief(req)


def test_generate_brief_reports_progress_for_every_section():
    calls = []
    req = BriefRequest(drugs=["tirzepatide"], sections=["pipeline", "safety"], requested_by="tester")
    brief = mock.generate_brief(req, on_progress=lambda key, step: calls.append((key, step)))
    assert [s.section_key for s in brief.sections] == ["pipeline", "safety"]
    assert ("pipeline", "done") in calls and ("safety", "done") in calls


def test_pipeline_section_is_a_table_without_claims():
    pipeline = make_brief().sections[0]
    assert pipeline.claims == []
    assert {row.drug for row in pipeline.table} == {"tirzepatide", "semaglutide"}


def test_llm_section_includes_a_flagged_claim():
    safety = make_brief().sections[1]
    assert any(not c.verified and c.issues for c in safety.claims)


def test_review_updates_section_and_audit_log():
    brief = make_brief()
    mock.submit_review(ReviewAction(brief_id=brief.brief_id, section_key="safety",
                                    action="reject", reviewer="tester"))
    assert mock.get_brief(brief.brief_id).sections[1].review_status == "rejected"
    assert [e.event for e in mock.get_audit_log(brief.brief_id)] == ["generated", "rejected"]


def test_cap_blocks_new_briefs(monkeypatch):
    monkeypatch.setattr(mock, "BRIEFS_CAP", 0)
    with pytest.raises(mock.CapExceededError):
        make_brief()


def test_out_of_scope_question_is_not_found():
    answer = mock.ask("What is the list price of Wegovy in Germany?", asked_by="tester")
    assert answer.claims == [] and answer.not_found


def test_search_filters_by_content_type():
    hits = mock.search("nausea", content_types=["table"])
    assert [h.source.content_type for h in hits] == ["table"]


def test_pipeline_filters_by_phase():
    rows = mock.get_pipeline(PipelineFilters(phases=["PHASE3"]))
    assert rows and all(r.phase == ["PHASE3"] for r in rows)
