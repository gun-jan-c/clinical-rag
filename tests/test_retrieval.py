from langchain_core.documents import Document

from clinical_rag.retrieval.query_expansion import expand
from clinical_rag.retrieval.rerank import rerank
from clinical_rag.retrieval.retriever import interleave, rank_rows, snippet

ALIASES = {"ly3502970": "orforglipron", "wegovy": "semaglutide", "ozempic": "semaglutide",
           "bi 456906": "survodutide"}


def test_expand_code_name_adds_generic_name():
    assert expand("LY3502970 weight loss", ALIASES) == "LY3502970 weight loss orforglipron"


def test_expand_generic_name_adds_brand_names_and_quotes_phrases():
    assert expand("semaglutide nausea", ALIASES) == "semaglutide nausea ozempic wegovy"
    assert expand("survodutide", ALIASES) == 'survodutide "bi 456906"'


def test_expand_leaves_query_without_drug_unchanged():
    assert expand("nausea incidence by dose", ALIASES) == "nausea incidence by dose"


def row(cid, dense, kw, rrf):
    return {"chunk_id": cid, "dense_rank": dense, "keyword_rank": kw, "rrf_score": rrf}


ROWS = [row("a", 3, None, 0.016), row("b", None, 1, 0.016), row("c", 1, 2, 0.032), row("d", 2, None, 0.0161)]


def test_rank_rows_hybrid_orders_by_rrf_score():
    assert [r["chunk_id"] for r in rank_rows(ROWS, "hybrid")] == ["c", "d", "a", "b"]


def test_rank_rows_single_modes_use_their_own_rank_and_drop_the_rest():
    assert [r["chunk_id"] for r in rank_rows(ROWS, "dense")] == ["c", "d", "a"]
    assert [r["chunk_id"] for r in rank_rows(ROWS, "keyword")] == ["b", "c"]


def test_interleave_takes_turns_and_skips_duplicates():
    tirzepatide = [row("t1", 1, 1, 0), row("shared", 2, 2, 0), row("t3", 3, 3, 0)]
    semaglutide = [row("shared", 1, 1, 0), row("s2", 2, 2, 0)]
    assert [r["chunk_id"] for r in interleave([tirzepatide, semaglutide], 10)] == ["t1", "shared", "s2", "t3"]
    assert len(interleave([tirzepatide, semaglutide], 2)) == 2


def test_snippet_drops_title_prefix_and_shortens():
    content = "Title: SURMOUNT-1 | Section: Results\n\nMean change in body weight at week 72 was large."
    assert snippet(content) == "Mean change in body weight at week 72 was large."
    long = snippet("Title: X\n\n" + "word " * 200)
    assert len(long) <= 304 and long.endswith(" ...")


def test_rerank_disabled_keeps_hybrid_order(monkeypatch):
    monkeypatch.setenv("RERANK_ENABLED", "false")
    docs = [Document(page_content=str(i)) for i in range(5)]
    assert [d.page_content for d in rerank(docs, "q", 3)] == ["0", "1", "2"]
