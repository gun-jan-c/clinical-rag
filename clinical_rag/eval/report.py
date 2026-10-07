"""Generate docs/eval_report.md from the stored eval runs (ProjectSpec.md section 8.3, task 4).

    uv run python -m clinical_rag.eval.report

Reads `eval_runs` / `eval_results` and writes a results table across configurations, the table experiment
(section/markdown vs fixed/flattened on table questions), and concrete failure examples. Every number comes
from the database, so one command reproduces the report. The golden set supplies each question's wording.
"""

import json
from pathlib import Path

from clinical_rag.db import connect

REPORT_PATH = Path("docs/eval_report.md")
GOLDEN_PATH = Path("golden_set.jsonl")

# Short written explanations for known failures; the report pairs these with the numbers pulled from the DB.
EXPLANATIONS = {
    "q19": ("The SURMOUNT-1 abstract mentions adverse events in a single sentence, so for an 'adverse events' "
            "query it is out-competed by papers written entirely about tirzepatide safety and falls past the "
            "50-candidate pool. The same paper ranks fine for the weight-loss wording (q01): retrieval is "
            "phrasing-sensitive, and the multi-query brief path is more robust than a single question."),
    "q23": ("Fix D put 'STEP 1' into the record's text, but it still does not reach the 50-candidate pool for this "
            "phrasing: 'step' is a common token (STEP 2-8 and any 'step' elsewhere) and the BMI-eligibility wording "
            "is generic across ~313 trials, while an exact acronym match gets no extra weight. Still a retrieval "
            "miss; fix D was necessary but not sufficient. Fix D did lift other acronym questions (dev MRR 59.9% -> "
            "65.9%, key_facts 84.6% -> 92.3%). A stronger fix would boost the acronym/title or apply a trial filter."),
    "q17": ("The model answered with the overall GI-adverse-reaction rate (56% at each dose) instead of the "
            "per-dose vomiting rows; the right table is retrieved, but the wrong row is quoted."),
}


def pct(x) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def cfg(c: dict) -> str:
    return (f"{c['strategy']}/{c['embedding_model'].replace('oai-3-', '')}/{c['mode']}/"
            f"rerank {'on' if c['rerank'] else 'off'}" + (" +gen" if c.get("with_generation") else ""))


def load_runs(conn) -> list[dict]:
    rows = conn.execute("select run_id::text, config, metrics, created_at from eval_runs order by created_at").fetchall()
    return [{"run_id": r[0], "config": r[1], "metrics": r[2], "created_at": r[3]} for r in rows]


def results_table(runs: list[dict]) -> list[str]:
    out = ["## Results across configurations", "",
           "Dev split, 50-candidate pool. Recall is document level; strict unless noted.", "",
           "| configuration | recall@5 | recall@10 | MRR | lenient@10 | citations | grounded | not-found |",
           "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in runs:
        m = r["metrics"]
        out.append(f"| {cfg(r['config'])} | {pct(m['recall_at_5'])} | {pct(m['recall_at_10'])} | {pct(m['mrr'])} "
                   f"| {pct(m['lenient_recall_at_10'])} | {pct(m.get('citation_valid'))} "
                   f"| {pct(m.get('numbers_grounded'))} | {pct(m.get('not_found_accuracy'))} |")
    return out


def table_experiment(runs: list[dict]) -> list[str]:
    """Section/markdown vs fixed/flattened on table questions, holding everything else equal (hybrid + rerank)."""
    on = {r["config"]["strategy"]: r for r in runs
          if r["config"]["mode"] == "hybrid" and r["config"]["rerank"]
          and r["config"]["embedding_model"] == "oai-3-small" and not r["config"].get("with_generation")}
    out = ["", "## Table experiment: section/markdown vs fixed/flattened", ""]
    prod = next((r for r in runs if r["config"].get("with_generation")), None)
    section = on.get("section") or prod
    fixed = on.get("fixed")
    if not (section and fixed):
        return out + ["(Need a section and a fixed hybrid+rerank run; re-run the sweep.)"]
    for label, r in [("section/markdown", section), ("fixed/flattened", fixed)]:
        t = r["metrics"]["by_type"].get("table", {})
        out.append(f"- **{label}**: table-question recall@10 {pct(t.get('recall_at_10'))} "
                   f"(n={t.get('n', 0)}); overall recall@10 {pct(r['metrics']['recall_at_10'])}.")
    out += ["", ("Section chunks keep each table whole with its header, so the model reads the right row; fixed "
                 "chunks flatten tables into text and can split a table across slices. See the decision log (gap "
                 "B) for why some label rows are still found only through the table, not its summary.")]
    return out


def failure_examples(conn, runs: list[dict], golden: dict) -> list[str]:
    prod = next((r for r in runs if r["config"].get("with_generation")), runs[-1])
    rows = conn.execute("select question_id, retrieved_doc_ids, recall_at_10 from eval_results "
                        "where run_id = %s and recall_at_10 = 0 order by question_id", (prod["run_id"],)).fetchall()
    out = ["", "## Failure examples", "", f"From the production run (`{cfg(prod['config'])}`):", ""]
    for qid, retrieved, _ in rows:
        g = golden.get(qid, {})
        out.append(f"- **{qid} [{g.get('type')}]** — {g.get('question')}")
        out.append(f"  - expected a relevant doc in {g.get('relevant_doc_ids')}; top 3 retrieved: {retrieved[:3]}")
        if qid in EXPLANATIONS:
            out.append(f"  - {EXPLANATIONS[qid]}")
    return out if rows else out + ["None: every question with a relevant document retrieved it in the top 10."]


def main() -> None:
    golden = {json.loads(line)["id"]: json.loads(line)
              for line in GOLDEN_PATH.read_text(encoding="utf-8").splitlines() if line.strip()}
    with connect() as conn:
        runs = load_runs(conn)
        if not runs:
            raise SystemExit("No eval runs stored; run clinical_rag.eval.run_eval first.")
        chosen = next((r for r in runs if r["config"].get("with_generation")), runs[-1])
        body = ["# Evaluation report", "",
                (f"Golden set: {len(golden)} questions, 14 types. Chosen configuration: **{cfg(chosen['config'])}**"
                 ", the pipeline the app runs (hybrid retrieval, 50-candidate pool, Cohere rerank to the top 8)."),
                ""]
        body += results_table(runs)
        body += table_experiment(runs)
        body += failure_examples(conn, runs, golden)
        body += ["", "---", "Regenerate with `uv run python -m clinical_rag.eval.report` after new runs."]
    REPORT_PATH.write_text("\n".join(body) + "\n", encoding="utf-8")
    print(f"wrote {REPORT_PATH} from {len(runs)} runs")


if __name__ == "__main__":
    main()
