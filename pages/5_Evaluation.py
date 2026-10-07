"""Evaluation page (ProjectSpec.md sections 7 and 8.3): the golden-set eval runs, read-only.

Reads `list_eval_runs()` (the `eval_runs` table). Runs are produced by `python -m clinical_rag.eval.run_eval`,
not here: running an eval costs money and hits the Cohere rate limit, so this page only displays finished runs.
Honest evaluation is a core goal of the project, so this is where a visitor sees the quality numbers and the gaps.
"""

import altair as alt
import pandas as pd
import streamlit as st

from clinical_rag.schemas import EvalRunSummary
from ui.backend import backend
from ui.layout import friendly_errors

G1_RECALL_AT_10 = 0.85  # spec goal G1: doc-level recall@10 on the golden set

METRIC_COLUMNS = [  # (metrics key, column label); shown as "n/a" when a run did not record it
    ("recall_at_5", "recall@5"), ("recall_at_10", "recall@10"), ("mrr", "MRR"),
    ("lenient_recall_at_10", "lenient@10"), ("citation_valid", "citations"),
    ("numbers_grounded", "grounded"), ("not_found_accuracy", "not-found"),
]


def config_label(c: dict) -> str:
    """One-line configuration, e.g. 'section / small / hybrid / rerank on'. Keys vary by run, so read defensively."""
    model = str(c.get("embedding_model", "")).replace("oai-3-", "")
    parts = [c.get("strategy", "?"), model, c.get("mode", "?"),
             "rerank on" if c.get("rerank") else "rerank off"]
    if c.get("with_generation"):
        parts.append("+gen")
    return " / ".join(p for p in parts if p)


def pct(x) -> float | None:
    return round(x * 100, 1) if isinstance(x, (int, float)) else None


def runs_frame(runs: list[EvalRunSummary]) -> pd.DataFrame:
    rows = []
    for r in runs:
        row = {"configuration": config_label(r.config), "split": r.config.get("split", "?"),
               "candidates": r.config.get("candidates", r.config.get("k"))}
        row |= {label: pct(r.metrics.get(key)) for key, label in METRIC_COLUMNS}
        rows.append(row)
    return pd.DataFrame(rows)


def recall_chart(df: pd.DataFrame) -> None:
    data = df.dropna(subset=["recall@10"])
    if data.empty:
        return
    bars = alt.Chart(data).mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4, color="#2a78d6").encode(
        x=alt.X("recall@10:Q", title="recall@10 (%)", scale=alt.Scale(domain=[0, 100])),
        y=alt.Y("configuration:N", sort="-x", title=None),
        tooltip=["configuration", "recall@10"],
    )
    goal = alt.Chart(pd.DataFrame({"g": [G1_RECALL_AT_10 * 100]})).mark_rule(
        color="#e34948", strokeDash=[4, 4]).encode(x="g:Q")
    st.altair_chart(bars + goal, width="stretch")
    st.caption(f"Dashed line: spec goal G1, recall@10 ≥ {G1_RECALL_AT_10:.0%}.")


def by_type_table(run: EvalRunSummary) -> None:
    by_type = run.metrics.get("by_type") or {}
    if not by_type:
        st.caption("No per-type breakdown recorded for this run.")
        return
    rows = [{"question type": t, "n": d.get("n"), "recall@10": pct(d.get("recall_at_10"))}
            for t, d in sorted(by_type.items())]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


st.title("Evaluation")
st.caption("Golden-set quality, measured by `clinical_rag/eval`. Each row is one eval run. Runs are produced by the "
           "eval CLI (they cost money and hit the rerank rate limit), so this page only displays finished runs.")

with friendly_errors("loading the eval runs"):
    runs = backend.list_eval_runs()

if not runs:
    st.info("No eval runs yet. Produce one with "
            "`uv run python -m clinical_rag.eval.run_eval --split dev --rerank on`.", icon=":material/science:")
else:
    df = runs_frame(runs)
    best = df["recall@10"].dropna()
    if not best.empty:
        top = df.loc[df["recall@10"].idxmax()]
        meets = "meets" if top["recall@10"] >= G1_RECALL_AT_10 * 100 else "below"
        st.metric(f"Best recall@10 ({top['configuration']})", f"{top['recall@10']:.1f}%",
                  delta=f"{meets} the 85% goal", delta_color="normal" if meets == "meets" else "inverse")

    st.subheader("All runs")
    st.dataframe(df, width="stretch", hide_index=True)
    recall_chart(df)

    st.subheader("Per-question-type recall")
    labels = [f"{config_label(r.config)} — {r.created_at:%Y-%m-%d %H:%M}" for r in runs]
    choice = st.selectbox("Run", range(len(runs)), format_func=lambda i: labels[i])
    chosen = runs[choice]
    by_type_table(chosen)
    if chosen.notes:
        st.caption(chosen.notes)
