"""Pipeline page (ProjectSpec.md section 7): registry trials by drug, phase and status. Filters -> get_pipeline ->
a phase x drug bar chart + the full trial table. Deterministic: no language model.

Chart (decision log 2026-10-05): one group of bars per phase, one bar per drug; a trial testing two drugs counts
under each. Each drug keeps one color whatever the filters (palette: the dataviz skill's categorical order).
"""

import altair as alt
import pandas as pd
import streamlit as st

from clinical_rag.schemas import PipelineFilters, TrialRow
from ui import components
from ui.backend import backend
from ui.components import DRUGS
from ui.layout import data_as_of, friendly_errors

DRUG_COLORS = dict(zip(DRUGS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                               "#e87ba4", "#008300", "#4a3aa7", "#e34948"]))
PHASES = ["EARLY_PHASE1", "PHASE1", "PHASE2", "PHASE3", "PHASE4", "NA"]
STATUSES = ["RECRUITING", "NOT_YET_RECRUITING", "ACTIVE_NOT_RECRUITING", "ENROLLING_BY_INVITATION", "COMPLETED",
            "TERMINATED", "SUSPENDED", "WITHDRAWN", "UNKNOWN", "AVAILABLE"]  # registry codes seen in our data


@st.cache_data(ttl=300, show_spinner=False)
def pipeline(drugs: tuple[str, ...], phases: tuple[str, ...], statuses: tuple[str, ...]) -> list[TrialRow]:
    return backend.get_pipeline(PipelineFilters(drugs=list(drugs) or None, phases=list(phases) or None,
                                                statuses=list(statuses) or None))


def chart(rows: list[TrialRow]) -> None:
    pairs = [{"Drug": d.strip(), "Phase": components.phase_label(t.phase)}
             for t in rows if t.drug for d in t.drug.split(",")]
    if not pairs:
        st.info("None of these trials name one of the 8 drugs, so there is nothing to chart.")
        return
    counts = pd.DataFrame(pairs).groupby(["Phase", "Drug"]).size().reset_index(name="Trials")
    drugs = [d for d in DRUGS if d in set(counts["Drug"])]
    bars = alt.Chart(counts).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("Phase:N", sort=components.phase_order(counts["Phase"]), title=None, axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("Drug:N", sort=drugs, scale=alt.Scale(paddingInner=0.1)),
        y=alt.Y("Trials:Q", title="Trials", axis=alt.Axis(tickMinStep=1, format="d")),  # whole trials only
        color=alt.Color("Drug:N", scale=alt.Scale(domain=drugs, range=[DRUG_COLORS[d] for d in drugs]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=["Drug", "Phase", "Trials"],
    )
    st.altair_chart(bars, width="stretch")


st.title("Development pipeline")
st.caption(f"Trials straight from ClinicalTrials.gov, status as of {data_as_of():%B %d, %Y}. "
           "No language model is used on this page.")

with st.container(horizontal=True):
    drugs = st.multiselect("Drugs", DRUGS, placeholder="All drugs")
    phases = st.multiselect("Phases", PHASES, format_func=lambda p: components.phase_label([p]),
                            placeholder="All phases")
    statuses = st.multiselect("Status", STATUSES, format_func=components.status_label, placeholder="All statuses")

with friendly_errors("loading the pipeline"):
    rows = pipeline(tuple(drugs), tuple(phases), tuple(statuses))
    if not rows:
        st.info("No trials match these filters.")
    else:
        st.subheader("Trials by phase and drug")
        chart(rows)
        unnamed = sum(not t.drug for t in rows)
        st.caption("A trial testing two drugs is counted under each."
                   + (f" {unnamed} trials that name none of the 8 drugs are in the table but not the chart."
                      if unnamed else ""))
        st.subheader(f"All {len(rows)} trials")
        st.dataframe(components.trial_table(rows), hide_index=True, column_config=components.TRIAL_COLUMNS)
