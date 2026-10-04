from clinical_rag.config import cost_usd
from clinical_rag.index.embed import MAX_BATCH_ITEMS, MAX_BATCH_TOKENS, batches
from clinical_rag.parsing.table_summaries import add_summary


def test_batches_respect_token_budget():
    rows = [(f"c{i}", "text", 30_000) for i in range(7)]
    groups = batches(rows)
    assert [len(g) for g in groups] == [3, 3, 1]
    assert all(sum(r[2] for r in g) <= MAX_BATCH_TOKENS for g in groups)
    assert [r for g in groups for r in g] == rows  # nothing lost, order kept


def test_batches_respect_item_limit():
    rows = [(f"c{i}", "t", 1) for i in range(MAX_BATCH_ITEMS + 5)]
    assert [len(g) for g in batches(rows)] == [MAX_BATCH_ITEMS, 5]


def test_batches_empty():
    assert batches([]) == []


def test_add_summary_goes_before_header_row():
    content = "Title: SURMOUNT-1 | Section: Results\n\nTable 2 Adverse events\n| Event | Placebo | 15 mg |"
    out = add_summary(content, "Adverse events by tirzepatide dose versus placebo.")
    assert out.splitlines()[-2:] == ["Adverse events by tirzepatide dose versus placebo.", "| Event | Placebo | 15 mg |"]
    assert out.startswith("Title: SURMOUNT-1 | Section: Results\n\nTable 2 Adverse events\n")


def test_cost_usd():
    assert round(cost_usd("text-embedding-3-small", 12_430_000), 4) == 0.2486
    assert round(cost_usd("gpt-6-luna", 1_000_000, 100_000), 2) == 0.15
