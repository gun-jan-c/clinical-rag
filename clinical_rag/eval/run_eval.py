"""Run the golden set through retrieval (and optionally generation), compute metrics, store one run.

    uv run python -m clinical_rag.eval.run_eval --strategy section --embedding-model oai-3-small \
        --mode hybrid --rerank on [--with-generation] [--split dev|test|all] [--report]

Writes one row to `eval_runs` and one per question to `eval_results` (ProjectSpec.md section 8.3), then prints a
summary. Recall is document level (spec section 12). Generation adds OpenAI + Cohere calls (~$0.0005/question),
so it is off by default. The golden set itself is the architect's; this only reads it.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from psycopg.types.json import Jsonb

from clinical_rag.db import pool
from clinical_rag.eval import metrics as M
from clinical_rag.retrieval.rerank import rerank
from clinical_rag.retrieval.retriever import CANDIDATES, HybridPostgresRetriever
from clinical_rag.schemas import EmbeddingModel, SearchMode, Strategy

GOLDEN_PATH = Path("golden_set.jsonl")
REPORT_PATH = Path("docs/eval_report.md")


def load_golden(path: Path, split: str) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows if split == "all" else [q for q in rows if q.get("split") == split]


def ranked_doc_ids(question: str, mode: SearchMode, strategy: Strategy, embedding_model: EmbeddingModel,
                   rerank_on: bool, candidates: int) -> list[str]:
    """The candidate chunks for a question, reduced to a ranked list of document ids (the pipeline's view)."""
    retriever = HybridPostgresRetriever(mode=mode, strategy=strategy, embedding_model=embedding_model,
                                        filter_named_drugs=True, limit=candidates)
    docs = retriever.invoke(question)
    if rerank_on:
        docs = rerank(docs, question, top_n=len(docs))  # reorder the whole candidate set, not just the top few
    return [d.metadata["doc_id"] for d in docs]


def score_generation(question: str, q: dict) -> dict:
    """Run the real single-question pipeline and score the written answer. Returns per-question generation scores."""
    # Imported lazily: only --with-generation needs it (it pulls in the LLM and Cohere clients).
    from clinical_rag.generation.answer import answer_question

    ans = answer_question(question)
    source_ids = {s.chunk_id for s in ans.sources}
    answer_text = " ".join(c.text for c in ans.claims)
    forbidden = M.contains_forbidden(answer_text, q.get("must_not_contain", []))
    citation_valid = M.mean([1.0 if c.citation_ids and all(cid in source_ids for cid in c.citation_ids) else 0.0
                             for c in ans.claims])
    numbers_grounded = M.mean([1.0 if c.verified else 0.0 for c in ans.claims])
    return {
        "claims": [c.text for c in ans.claims],
        "not_found": ans.not_found,
        "citation_valid": citation_valid,
        "numbers_grounded": numbers_grounded,
        "key_facts_present": M.key_facts_present(answer_text, q.get("key_facts", [])),
        "forbidden_hit": forbidden,
        "not_found_correct": M.not_found_correct(q["type"], len(ans.claims), len(ans.not_found), forbidden),
        "cost_usd": ans.usage.cost_usd,
    }


def evaluate(golden: list[dict], mode: SearchMode, strategy: Strategy, embedding_model: EmbeddingModel,
             rerank_on: bool, with_generation: bool, candidates: int) -> list[dict]:
    results = []
    for q in golden:
        docs = ranked_doc_ids(q["question"], mode, strategy, embedding_model, rerank_on, candidates)
        relevant = set(q.get("relevant_doc_ids", []))
        # Lenient is scored only where there is a relevant document, so it stays over the same questions as strict
        # recall (a behavioural question like medical_advice may list an acceptable doc but nothing strictly relevant).
        lenient = (relevant | set(q.get("acceptable_doc_ids", []))) if relevant else set()
        row = {
            "question_id": q["id"],
            "type": q["type"],
            "retrieved_doc_ids": M.distinct_docs(docs),
            "recall_at_5": M.recall_at_k(docs, relevant, 5),
            "recall_at_10": M.recall_at_k(docs, relevant, 10),
            "reciprocal_rank": M.reciprocal_rank(docs, relevant),
            "lenient_hit_at_10": M.hit_at_k(docs, lenient, 10),
            "generation": score_generation(q["question"], q) if with_generation else None,
        }
        results.append(row)
    return results


def aggregate(results: list[dict]) -> dict:
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_type[r["type"]].append(r)
    gen = [r["generation"] for r in results if r["generation"] is not None]
    nf = [r["generation"]["not_found_correct"] for r in results
          if r["generation"] and r["generation"]["not_found_correct"] is not None]
    return {
        "recall_at_5": M.mean([r["recall_at_5"] for r in results]),
        "recall_at_10": M.mean([r["recall_at_10"] for r in results]),
        "mrr": M.mean([r["reciprocal_rank"] for r in results]),
        "lenient_recall_at_10": M.mean([r["lenient_hit_at_10"] for r in results]),
        "citation_valid": M.mean([g["citation_valid"] for g in gen]) if gen else None,
        "numbers_grounded": M.mean([g["numbers_grounded"] for g in gen]) if gen else None,
        "key_facts_present": M.mean([g["key_facts_present"] for g in gen]) if gen else None,
        "not_found_accuracy": (sum(1 for x in nf if x) / len(nf)) if nf else None,
        "by_type": {t: {"n": len(rs), "recall_at_10": M.mean([r["recall_at_10"] for r in rs])}
                    for t, rs in sorted(by_type.items())},
    }


def store_run(config: dict, metrics: dict, results: list[dict], notes: str | None) -> str:
    with pool().connection() as conn:
        run_id = conn.execute("insert into eval_runs (config, metrics, notes) values (%s, %s, %s) "
                              "returning run_id::text", (Jsonb(config), Jsonb(metrics), notes)).fetchone()[0]
        for r in results:
            g = r["generation"]
            conn.execute(
                "insert into eval_results (run_id, question_id, retrieved_doc_ids, recall_at_5, recall_at_10, "
                "reciprocal_rank, answer, citation_valid, numbers_grounded, not_found_correct) "
                "values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (run_id, r["question_id"], r["retrieved_doc_ids"], r["recall_at_5"], r["recall_at_10"],
                 r["reciprocal_rank"],
                 Jsonb({"type": r["type"], "lenient_hit_at_10": r["lenient_hit_at_10"], "generation": g}),
                 g["citation_valid"] if g else None, g["numbers_grounded"] if g else None,
                 g["not_found_correct"] if g else None))
    return run_id


def pct(x: float | None) -> str:
    return "  n/a" if x is None else f"{x * 100:5.1f}%"


def summary_lines(config: dict, metrics: dict) -> list[str]:
    lines = [f"config: {config}",
             (f"recall@5 {pct(metrics['recall_at_5'])}   recall@10 {pct(metrics['recall_at_10'])}   "
              f"MRR {pct(metrics['mrr'])}   lenient recall@10 {pct(metrics['lenient_recall_at_10'])}")]
    if metrics["citation_valid"] is not None:
        lines.append(f"citation_valid {pct(metrics['citation_valid'])}   "
                     f"numbers_grounded {pct(metrics['numbers_grounded'])}   "
                     f"key_facts {pct(metrics['key_facts_present'])}   "
                     f"not_found_accuracy {pct(metrics['not_found_accuracy'])}")
    lines.append("by type (recall@10):")
    for t, d in metrics["by_type"].items():
        lines.append(f"  {t:18} n={d['n']:<2} {pct(d['recall_at_10'])}")
    return lines


def write_report(run_id: str, config: dict, metrics: dict) -> None:
    body = ["# Evaluation report", "",
            f"Run `{run_id}` — golden-set `{config['split']}` split, {config['n_questions']} questions.", "",
            "```", *summary_lines(config, metrics), "```", ""]
    REPORT_PATH.write_text("\n".join(body), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Run the golden set and store an eval run.")
    p.add_argument("--strategy", choices=["section", "fixed"], default="section")
    p.add_argument("--embedding-model", choices=["oai-3-small", "oai-3-large"], default="oai-3-small")
    p.add_argument("--mode", choices=["hybrid", "dense", "keyword"], default="hybrid")
    p.add_argument("--rerank", choices=["on", "off"], default="on")
    p.add_argument("--candidates", type=int, default=CANDIDATES,
                   help=f"pool retrieved before rerank (default {CANDIDATES}; answer_question uses 50)")
    p.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    p.add_argument("--with-generation", action="store_true", help="also run the LLM answer (costs ~$0.0005/question)")
    p.add_argument("--report", action="store_true", help="write docs/eval_report.md")
    p.add_argument("--golden", type=Path, default=GOLDEN_PATH)
    p.add_argument("--notes", default=None)
    args = p.parse_args()

    golden = load_golden(args.golden, args.split)
    if not golden:
        raise SystemExit(f"No questions in split '{args.split}' of {args.golden}")
    rerank_on = args.rerank == "on"
    config = {"strategy": args.strategy, "embedding_model": args.embedding_model, "mode": args.mode,
              "rerank": rerank_on, "candidates": args.candidates, "with_generation": args.with_generation,
              "split": args.split, "n_questions": len(golden)}

    results = evaluate(golden, args.mode, args.strategy, args.embedding_model, rerank_on, args.with_generation,
                       args.candidates)
    metrics = aggregate(results)
    run_id = store_run(config, metrics, results, args.notes)

    print(f"\neval run {run_id}")
    print("\n".join(summary_lines(config, metrics)))
    if args.report:
        write_report(run_id, config, metrics)
        print(f"\nwrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
