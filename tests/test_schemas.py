import pytest
from pydantic import ValidationError

from clinical_rag.schemas import Claim, SectionDraft, Source


def test_source_rejects_unknown_content_type():
    with pytest.raises(ValidationError):
        Source(chunk_id="c1", doc_id="d1", source="pubmed", content_type="video",
               title="t", url="https://example.org", snippet="s")


def test_section_draft_defaults():
    draft = SectionDraft(section_key="efficacy", title="Key efficacy results",
                         claims=[Claim(text="x", citation_ids=["c1"])])
    assert draft.review_status == "pending"
    assert draft.not_found == [] and draft.sources == []
    assert draft.usage.cost_usd == 0.0


def test_section_draft_usage_not_shared_between_instances():
    a = SectionDraft(section_key="safety", title="a")
    b = SectionDraft(section_key="safety", title="b")
    a.usage.input_tokens = 5
    assert b.usage.input_tokens == 0
