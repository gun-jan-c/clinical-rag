"""Parsers for PubMed, Europe PMC, and openFDA, on small hand-written samples (placeholder values)."""

from datetime import date

import lxml.html
from lxml import etree

from clinical_rag.ingest import europepmc, labels, pubmed
from clinical_rag.parsing.tables import table_to_markdown

ALIASES = {"zepbound": "tirzepatide"}


def test_table_to_markdown_handles_colspan_and_footnotes():
    table = lxml.html.fragment_fromstring(
        "<table><thead><tr><th>Reaction</th><th colspan='2'>Dose</th></tr></thead>"
        "<tbody><tr><td>Nausea<sup>a</sup></td><td>X%</td><td>Y%</td></tr></tbody>"
        "<tfoot><tr><td>a Footnote</td></tr></tfoot></table>")
    markdown, footnotes = table_to_markdown(table)
    assert markdown.splitlines() == ["| Reaction | Dose |  |", "| --- | --- | --- |", "| Nausea^a | X% | Y% |"]
    assert footnotes == "a Footnote"


PUBMED_XML = b"""<PubmedArticleSet><PubmedArticle>
<MedlineCitation><PMID>123</PMID><Article>
  <Journal><Title>Journal</Title></Journal>
  <ArticleTitle>Tirzepatide in obesity</ArticleTitle>
  <Abstract><AbstractText Label="METHODS">Adults were randomized.</AbstractText>
            <AbstractText Label="RESULTS">Weight fell (NCT00000001).</AbstractText></Abstract>
  <DataBankList><DataBank><DataBankName>ClinicalTrials.gov</DataBankName>
    <AccessionNumberList><AccessionNumber>NCT00000002</AccessionNumber></AccessionNumberList></DataBank></DataBankList>
</Article></MedlineCitation>
<PubmedData><History><PubMedPubDate PubStatus="pubmed"><Year>2022</Year><Month>6</Month><Day>4</Day>
</PubMedPubDate></History></PubmedData>
</PubmedArticle></PubmedArticleSet>"""


def test_pubmed_keeps_abstract_sections_and_trial_links():
    doc = pubmed.to_document(etree.fromstring(PUBMED_XML).find("PubmedArticle"), ALIASES)
    assert doc["doc_id"] == "pubmed:123" and doc["published_date"] == date(2022, 6, 4)
    assert list(doc["sections"]) == ["Methods", "Results"]
    assert doc["metadata"]["nct_ids"] == ["NCT00000001", "NCT00000002"]
    assert doc["metadata"]["drugs"] == ["tirzepatide"]


JATS_XML = b"""<article><front><article-meta>
  <title-group><article-title>Zepbound trial</article-title></title-group>
  <abstract><p>Short abstract.</p></abstract></article-meta></front>
<body>
  <sec><title>Results</title><p>Weight fell.</p>
    <sec><title>Safety</title><p>Nausea was common.</p></sec>
    <table-wrap><label>Table 2</label><caption><p>Adverse events</p></caption>
      <table><tr><th>Event</th><th>n</th></tr><tr><td>Nausea</td><td>X</td></tr></table>
      <table-wrap-foot><p>Data are n.</p></table-wrap-foot></table-wrap>
    <fig><label>Figure 1</label><caption><p>Body weight over time.</p></caption></fig>
  </sec>
</body></article>"""


def test_jats_splits_text_tables_and_figures():
    parsed = europepmc.parse_jats(JATS_XML)
    assert parsed["sections"] == {"Abstract": "Short abstract.",
                                  "Results": "Weight fell.\n\nSafety\n\nNausea was common."}
    [table] = parsed["tables"]
    assert (table["label"], table["caption"], table["section"]) == ("Table 2", "Adverse events", "Results")
    assert table["footnotes"] == "Data are n." and "| Nausea | X |" in table["markdown"]
    assert parsed["figures"] == [{"label": "Figure 1", "caption": "Body weight over time.", "section": "Results"}]


def test_jats_leaves_tables_inside_a_paragraph_out_of_the_text():
    xml = b"""<article><body><sec><title>Results</title>
      <p>See Table 1.<table-wrap><label>Table 1</label><caption><p>Baseline</p></caption>
        <table><tr><td>Age</td><td>X</td></tr></table></table-wrap> Groups were similar.</p>
    </sec></body></article>"""
    parsed = europepmc.parse_jats(xml)
    assert parsed["sections"] == {"Results": "See Table 1. Groups were similar."}
    assert [t["label"] for t in parsed["tables"]] == ["Table 1"]


def test_europepmc_tags_drugs_from_title_and_abstract_only():
    xml = b"""<article><front><article-meta>
      <title-group><article-title>Compound X in obesity</article-title></title-group>
      <abstract><p>Compound X lowered weight.</p></abstract></article-meta></front>
    <body><sec><title>Introduction</title><p>Zepbound is approved.</p></sec></body></article>"""
    doc = europepmc.to_document({"pmcid": "PMC1"}, xml, ALIASES)
    assert doc["metadata"]["drugs"] == []


def test_label_removes_the_pasted_copy_of_its_tables():
    table = "<table><caption>Table 1: Adverse Reactions</caption><tr><td>Nausea</td><td>X%</td></tr></table>"
    label = {"set_id": "abc", "effective_time": "20260828", "openfda": {},
             "adverse_reactions": ["Nausea occurred. Table 1: Adverse Reactions Nausea X% It was mild."],
             "adverse_reactions_table": [table]}
    doc = labels.to_document("Zepbound", label, ALIASES)
    assert doc["sections"] == {"Adverse reactions": "Nausea occurred. It was mild."}


def test_label_separates_text_sections_from_tables():
    label = {"set_id": "abc", "effective_time": "20260828", "openfda": {"generic_name": ["TIRZEPATIDE"]},
             "adverse_reactions": ["Nausea occurred."],
             "adverse_reactions_table": [("<table><caption>Table 1: Adverse Reactions</caption>"
                                          "<tr><td>Nausea</td><td>X%</td></tr></table>")]}
    doc = labels.to_document("Zepbound", label, ALIASES)
    assert doc["sections"] == {"Adverse reactions": "Nausea occurred."}
    [table] = doc["metadata"]["tables"]
    assert table["label"] == "Table 1" and table["section"] == "Adverse reactions"
    assert doc["metadata"]["drugs"] == ["tirzepatide"] and doc["published_date"] == date(2026, 8, 28)
