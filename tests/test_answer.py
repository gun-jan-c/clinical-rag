import pytest
from langchain_core.messages import AIMessage

from clinical_rag.generation import answer
from clinical_rag.generation.answer import (
    AnswerDraft,
    format_sources,
    generate,
    seen_by_model,
)
from clinical_rag.generation.verify import verify
from clinical_rag.schemas import Claim, Source


class FakeModel:
    """Stands in for ChatOpenAI(...).with_structured_output(..., include_raw=True)."""

    def __init__(self, replies):
        self.replies, self.calls = replies, []

    def with_structured_output(self, schema, include_raw):
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        parsed = self.replies.pop(0)
        raw = AIMessage(content="", usage_metadata={"input_tokens": 100, "output_tokens": 10, "total_tokens": 110})
        return {"raw": raw, "parsed": parsed, "parsing_error": None if parsed else "claims: field required"}


def use_fake(monkeypatch, replies):
    fake = FakeModel(replies)
    monkeypatch.setattr(answer, "get_chat", lambda **kw: fake)
    return fake


def test_generate_retries_once_with_the_error_and_adds_up_tokens(monkeypatch):
    draft = AnswerDraft(claims=[], not_found=["list price"])
    fake = use_fake(monkeypatch, [None, draft])
    assert generate([("human", "q")]) == (draft, 200, 20)
    assert "claims: field required" in fake.calls[1][-1][1]


def test_generate_gives_up_after_one_retry(monkeypatch):
    use_fake(monkeypatch, [None, None])
    with pytest.raises(ValueError, match="after one retry"):
        generate([("human", "q")])


def test_format_sources_labels_each_source_with_its_chunk_id():
    s = Source(chunk_id="label:oz:section:29", doc_id="label:oz", source="dailymed", content_type="table",
               title="Ozempic prescribing information", url="u", section="Adverse reactions", snippet="")
    assert format_sources([{"text": "| Nausea | 15.8 |"}], [s]) == (
        '<source id="label:oz:section:29" type="table" title="Ozempic prescribing information" '
        'section="Adverse reactions">\n| Nausea | 15.8 |\n</source>')


def test_verify_sees_the_source_label_too():
    s = Source(chunk_id="ctgov:NCT05556512:section:3", doc_id="ctgov:NCT05556512", source="clinicaltrials.gov",
               content_type="text", title="A Study of Tirzepatide (SURMOUNT-MMO)", url="u", snippet="")
    sent = seen_by_model([{"text": "Status: ACTIVE_NOT_RECRUITING", "chunk_ids": [s.chunk_id]}], [s])
    claim = verify([Claim(text="NCT05556512 is still running.", citation_ids=[s.chunk_id])], sent)[0]
    assert claim.verified
