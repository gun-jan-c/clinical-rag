import subprocess
import sys
from datetime import date

from cohere.errors import TooManyRequestsError
from langchain_core.documents import Document

from clinical_rag.generation import brief
from clinical_rag.generation.cost import total
from clinical_rag.retrieval import rerank as rerank_module
from clinical_rag.schemas import BriefRequest, SectionDraft, Usage


def test_rerank_waits_and_retries_when_cohere_is_busy(monkeypatch):
    calls, waits = [], []

    class FakeReranker:
        def compress_documents(self, docs, query):
            calls.append(query)
            if len(calls) < 3:
                raise TooManyRequestsError(body=None)
            return [Document(page_content=d.page_content, metadata={"relevance_score": 0.9}) for d in docs]

    monkeypatch.setenv("RERANK_ENABLED", "true")
    monkeypatch.setattr(rerank_module, "get_reranker", lambda top_n: FakeReranker())
    monkeypatch.setattr(rerank_module.time, "sleep", waits.append)
    out = rerank_module.rerank([Document(page_content="a")], "q", 1)
    assert [d.metadata["rerank_score"] for d in out] == [0.9]
    assert len(calls) == 3 and waits == [10, 10]


def test_total_adds_sections_and_keeps_wall_clock_latency():
    usage = total([Usage(input_tokens=100, output_tokens=10, cost_usd=0.001, latency_ms=5000),
                   Usage(input_tokens=50, output_tokens=5, cost_usd=0.0005, latency_ms=7000)], 7500)
    assert usage == Usage(input_tokens=150, output_tokens=15, cost_usd=0.0015, latency_ms=7500)


def fake_sections(monkeypatch, failed=()):
    seen = {}

    def section(key, not_found=()):
        return SectionDraft(section_key=key, title=key, not_found=list(not_found),
                            review_status="failed" if key in failed else "pending")

    monkeypatch.setattr(brief, "pipeline_section", lambda drugs, progress: section("pipeline"))
    monkeypatch.setattr(brief, "llm_section", lambda key, drugs, progress: section(key, [f"{key} gap"]))

    def gaps(drugs, open_questions, progress):
        seen["open_questions"] = open_questions
        return section("evidence_gaps")

    monkeypatch.setattr(brief, "evidence_gaps_section", gaps)
    monkeypatch.setattr(brief, "data_as_of", lambda: date(2026, 10, 4))
    return seen


def test_brief_keeps_the_requested_order_and_passes_open_questions_on(monkeypatch):
    seen = fake_sections(monkeypatch, failed={"safety"})
    keys = ["evidence_gaps", "safety", "pipeline", "efficacy"]
    b = brief.build_brief(BriefRequest(drugs=["tirzepatide"], sections=keys, requested_by="me"))
    assert [s.section_key for s in b.sections] == keys
    assert seen["open_questions"] == ["efficacy gap"]  # the failed section's items are left out
    assert b.status == "draft"


def test_brief_fails_only_when_every_llm_section_failed(monkeypatch):
    fake_sections(monkeypatch, failed={"efficacy", "safety"})
    req = BriefRequest(drugs=["tirzepatide"], sections=["pipeline", "efficacy", "safety"], requested_by="me")
    assert brief.build_brief(req).status == "failed"


def test_a_section_that_crashes_is_marked_failed_and_the_rest_finish(monkeypatch):
    fake_sections(monkeypatch)

    def section(key, drugs, progress):
        if key == "safety":
            raise AttributeError("boom")
        return SectionDraft(section_key=key, title=key)

    monkeypatch.setattr(brief, "llm_section", section)
    steps = []
    req = BriefRequest(drugs=["tirzepatide"], sections=["efficacy", "safety", "evidence_gaps"], requested_by="me")
    b = brief.build_brief(req, lambda key, step: steps.append((key, step)))
    assert [s.review_status for s in b.sections] == ["pending", "failed", "pending"]
    assert b.sections[1].not_found == ["Section could not be generated (AttributeError)"]
    assert ("safety", "failed") in steps
    assert b.status == "draft"


def test_httpx_is_loaded_before_sections_run_in_threads():
    # openai looks httpx up in sys.modules without importing it. If another section's thread is still loading
    # httpx (Cohere's first call), openai finds it half-loaded and crashes (first brief after start, 2026-10-04).
    code = "import sys, clinical_rag.generation.brief; assert 'httpx' in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
