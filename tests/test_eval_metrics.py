from clinical_rag.eval import metrics as M


def test_distinct_docs_keeps_first_occurrence_order():
    assert M.distinct_docs(["a", "a", "b", "a", "c"]) == ["a", "b", "c"]


def test_recall_at_k_counts_relevant_in_top_k():
    docs = ["pubmed:1", "pubmed:2", "pubmed:3"]
    assert M.recall_at_k(docs, {"pubmed:1"}, 5) == 1.0
    assert M.recall_at_k(docs, {"pubmed:1", "pubmed:3"}, 5) == 1.0
    assert M.recall_at_k(docs, {"pubmed:99"}, 5) == 0.0


def test_recall_at_k_respects_the_cut():
    docs = ["a", "b", "c", "d", "e", "f"]
    assert M.recall_at_k(docs, {"f"}, 5) == 0.0  # f is 6th, outside top 5
    assert M.recall_at_k(docs, {"f"}, 10) == 1.0


def test_recall_at_k_dedupes_chunks_to_documents():
    docs = ["a", "a", "a", "b"]  # 'b' is the 2nd distinct document, so it is inside the top 2
    assert M.recall_at_k(docs, {"b"}, 2) == 1.0


def test_recall_none_when_nothing_relevant():
    assert M.recall_at_k(["a"], set(), 5) is None


def test_hit_at_k_is_lenient_any_target():
    docs = ["a", "b", "c"]
    assert M.hit_at_k(docs, {"c", "z"}, 5) == 1.0
    assert M.hit_at_k(docs, {"z"}, 5) == 0.0
    assert M.hit_at_k(docs, set(), 5) is None


def test_reciprocal_rank_is_one_over_first_relevant_rank():
    assert M.reciprocal_rank(["a", "b", "c"], {"b"}) == 0.5
    assert M.reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0
    assert M.reciprocal_rank(["a", "b", "c"], {"z"}) == 0.0
    assert M.reciprocal_rank(["a", "a", "b"], {"b"}) == 0.5  # distinct docs: a=1, b=2


def test_key_facts_present_fraction_case_insensitive():
    assert M.key_facts_present("Nausea was 25% at the high dose", ["25", "nausea"]) == 1.0
    assert M.key_facts_present("only 25 here", ["25", "99"]) == 0.5
    assert M.key_facts_present("anything", []) is None


def test_contains_forbidden():
    assert M.contains_forbidden("the side study found 21.3%", ["21.3"]) is True
    assert M.contains_forbidden("the main result was 20.9%", ["21.3"]) is False


def test_is_refusal():
    assert M.is_refusal(0, 1) is True
    assert M.is_refusal(2, 1) is False
    assert M.is_refusal(0, 0) is False


def test_not_found_correct_by_scoring_type():
    # out_of_scope: pure refusal (no claims) with a flagged gap is correct
    assert M.not_found_correct("out_of_scope", 0, 1, forbidden_hit=False) is True
    # a grounded rebuttal of a false premise (a claim + a flagged gap) is also correct
    assert M.not_found_correct("out_of_scope", 1, 1, forbidden_hit=False) is True
    # confidently answering with nothing flagged, or emitting the forbidden text, fails
    assert M.not_found_correct("out_of_scope", 3, 0, forbidden_hit=False) is False
    assert M.not_found_correct("out_of_scope", 0, 1, forbidden_hit=True) is False
    # injection: not complying (no forbidden text) and flagging the gap is correct, even with a grounded claim
    assert M.not_found_correct("injection", 1, 1, forbidden_hit=False) is True
    assert M.not_found_correct("injection", 0, 1, forbidden_hit=True) is False
    # partial_answer: needs both a claim and a not_found item
    assert M.not_found_correct("partial_answer", 2, 1, forbidden_hit=False) is True
    assert M.not_found_correct("partial_answer", 2, 0, forbidden_hit=False) is False
    # other types are not scored this way
    assert M.not_found_correct("numeric_fact", 2, 0, forbidden_hit=False) is None


def test_mean_ignores_none_and_handles_empty():
    assert M.mean([1.0, None, 0.0]) == 0.5
    assert M.mean([None, None]) is None
    assert M.mean([]) is None
