from clinical_rag.generation.verify import numbers, verify
from clinical_rag.schemas import Claim

SOURCES = {
    "label:oz:section:29": "| Adverse reaction | Placebo | 0.5 mg | 1 mg |\n| Nausea | 6.1 | 15.8 | 20.3 |",
    "ctgov:NCT1:section:3": "Percent change in body weight at week 72: −14.9 (95% CI −15.4 to −14.4); 1,961 participants",
}


def check(text, ids):
    return verify([Claim(text=text, citation_ids=ids)], SOURCES)[0]


def test_numbers_extracts_distinct_numbers_in_order():
    assert numbers("15.8% vs 6.1% at week 72, 15.8% again; 1,961 people") == ["15.8", "6.1", "72", "1961"]


def test_grounded_claim_passes_with_signs_and_thousands_commas_ignored():
    claim = check("Mean weight change was 14.9% at week 72 (95% CI -15.4 to -14.4) in 1961 participants.",
                  ["ctgov:NCT1:section:3"])
    assert claim.verified and claim.issues == []


def test_number_may_come_from_any_cited_source():
    claim = check("Nausea occurred in 15.8% on 0.5 mg; weight change was 14.9% at week 72.",
                  ["label:oz:section:29", "ctgov:NCT1:section:3"])
    assert claim.verified


def test_missing_citation_and_invalid_citation_are_flagged():
    assert check("Nausea was common.", []).issues == ["no citation"]
    claim = check("Nausea was common.", ["label:oz:section:99"])
    assert not claim.verified
    assert claim.issues == ["cites chunks that were not among the sources: label:oz:section:99"]


def test_number_inside_a_longer_number_does_not_count():
    claim = check("Weight change was 4.9% at week 7.", ["ctgov:NCT1:section:3"])
    assert claim.issues == ["numbers not found in the cited sources: 4.9, 7"]
