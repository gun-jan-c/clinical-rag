"""Chunking strategies on small hand-written documents (placeholder values)."""

from clinical_rag.chunking import strategies
from clinical_rag.chunking.strategies import (
    MAX_TOKENS,
    count_tokens,
    is_skipped,
    split_section,
)

SENTENCE = "Patients in the treatment group lost more weight than those given placebo over the study period. "


def _doc(**overrides) -> dict:
    doc = {"doc_id": "epmc:PMC1", "source": "europepmc", "title": "Drug X trial", "license": "cc by",
           "sections": {"Results": "Weight fell.", "Acknowledgments": "We thank the nurses.",
                        "Funding": "Funded by Company Y."},
           "metadata": {"drugs": ["tirzepatide"], "year": 2024,
                        "tables": [{"label": "Table 2", "caption": "Adverse events", "section": "Results",
                                    "markdown": "| Event | n |\n| --- | --- |\n| Nausea | X |",
                                    "footnotes": "Data are n."}],
                        "figures": [{"label": "Figure 1", "caption": "Weight over time.", "section": "Results"}]}}
    return {**doc, **overrides}


def test_back_matter_is_skipped_but_funding_is_kept():
    for name in ["Acknowledgements", "CONFLICT OF INTEREST STATEMENT", "Data Availability Statement",
                 "Author contributions", "Footnotes", "Supplementary Materials", "Spl medguide", "Associated Data"]:
        assert is_skipped(name), name
    for name in ["Results", "Funding Statement", "Study funding/competing interest(s)", "Instructions for use",
                 "Article Information", "Registration"]:
        assert not is_skipped(name), name


def test_short_section_stays_whole():
    assert split_section("Weight fell.") == ["Weight fell."]


def test_long_section_is_cut_at_sentence_ends_only():
    text = SENTENCE * 60
    pieces = split_section(text)
    assert len(pieces) > 1
    assert all(count_tokens(p) <= MAX_TOKENS for p in pieces)
    assert all(p.endswith("study period.") for p in pieces)
    assert " ".join(pieces) == text.strip()


def test_label_section_is_cut_at_numbered_subheadings_first():
    text = "14 CLINICAL STUDIES 14.1 Weight Study " + SENTENCE * 30 + "14.2 Sleep Apnea Study " + SENTENCE * 3
    pieces = split_section(text)
    assert pieces[0].startswith("14 CLINICAL STUDIES 14.1 Weight Study")
    assert pieces[-1].startswith("14.2 Sleep Apnea Study")  # 14.2 is never mixed into a 14.1 piece
    assert all(count_tokens(p) <= MAX_TOKENS for p in pieces)


def test_paragraphs_that_fit_are_not_cut():
    paragraphs = [SENTENCE * 20, SENTENCE * 20, SENTENCE * 20]  # each fits, but no two fit together
    pieces = split_section("\n\n".join(p.strip() for p in paragraphs))
    assert pieces == [p.strip() for p in paragraphs]


def test_subheading_stays_with_the_paragraph_after_it():
    first, second = (SENTENCE * 20).strip(), (SENTENCE * 20).strip()
    pieces = split_section(f"{first}\n\nOutcomes\n\n{second}")
    assert pieces == [first, f"Outcomes\n\n{second}"]


def test_long_list_is_cut_between_items():
    items = [f"- Change in Body Weight From Baseline at Week {n} (Week 0, week {n})" for n in range(60)]
    pieces = split_section("\n".join(items))
    assert len(pieces) > 1
    assert all(count_tokens(p) <= MAX_TOKENS for p in pieces)
    assert all(line in items for p in pieces for line in p.splitlines())


def test_section_strategy_makes_text_table_and_figure_chunks():
    chunks = strategies.section_chunks(_doc())
    assert [c["chunk_id"] for c in chunks] == ["epmc:PMC1:section:0", "epmc:PMC1:section:1",
                                               "epmc:PMC1:section:2", "epmc:PMC1:section:3"]
    text, funding, table, figure = chunks
    assert text["content"] == "Title: Drug X trial | Section: Results\n\nWeight fell."
    assert funding["section"] == "Funding"
    assert table["content"] == "Title: Drug X trial | Section: Results\n\nTable 2 Adverse events\n| Event | n |"
    assert "| Nausea | X |" in table["context"] and "Data are n." in table["context"]
    assert table["metadata"] == {"content_type": "table", "drugs": ["tirzepatide"], "year": 2024,
                                 "source": "europepmc", "license": "cc by", "label": "Table 2"}
    assert figure["metadata"]["content_type"] == "figure" and figure["context"] == "Figure 1 Weight over time."


def test_fixed_strategy_flattens_tables_into_the_text():
    [chunk] = strategies.fixed_chunks(_doc())
    assert chunk["chunk_id"] == "epmc:PMC1:fixed:0" and chunk["section"] is None
    assert "Table 2 Adverse events Event n Nausea X Data are n." in chunk["content"]
    assert "We thank the nurses" not in chunk["content"]
    assert chunk["content"].index("Nausea X") < chunk["content"].index("Funding:")  # table follows its section


def test_fixed_strategy_overlaps_pieces():
    doc = _doc(sections={"Results": SENTENCE * 60}, metadata={"drugs": [], "year": 2024})
    chunks = strategies.fixed_chunks(doc)
    assert len(chunks) > 1
    assert all(c["token_count"] <= MAX_TOKENS + 20 for c in chunks)  # + the 'Title: ...' prefix
    assert chunks[0]["context"].startswith("Results: Patients")
    assert chunks[0]["context"][-100:] in chunks[1]["context"]


def test_paper_chunks_name_their_trial_in_the_header():
    acronyms = {"NCT04184622": "SURMOUNT-1", "NCT00000001": None}
    paper = _doc(source="pubmed", metadata={**_doc()["metadata"], "nct_ids": ["NCT04184622", "NCT00000001"]})
    paper["trials"] = strategies.trial_label(paper, acronyms)
    assert paper["trials"] == "SURMOUNT-1 (NCT04184622), NCT00000001"
    chunk = strategies.section_chunks(paper)[0]
    assert chunk["content"].startswith(
        "Title: Drug X trial | Trial: SURMOUNT-1 (NCT04184622), NCT00000001 | Section: Results\n\n")
    assert strategies.section_chunks(_doc())[0]["content"].startswith("Title: Drug X trial | Section: Results\n\n")
    registry = _doc(source="clinicaltrials.gov", metadata={**_doc()["metadata"], "nct_ids": ["NCT04184622"]})
    assert strategies.trial_label(registry, acronyms) is None


def test_europe_pmc_trial_label_ignores_trials_only_cited_in_the_body():
    paper = _doc(sections={"Abstract": "A trial (NCT00000001) of drug X.", "Discussion": "Unlike NCT04184622, ..."},
                 metadata={**_doc()["metadata"], "nct_ids": ["NCT00000001", "NCT04184622"]})
    assert strategies.trial_label(paper, {"NCT04184622": "SURMOUNT-1"}) == "NCT00000001"
