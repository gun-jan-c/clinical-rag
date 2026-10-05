"""Review page (ProjectSpec.md section 7): read a brief as one document and approve, edit or reject each section.

Layout (decision log 2026-10-05): one scrolling document with a contents list; the pipeline as a drug x phase count
table with every trial in a collapsed table; review controls under each section; audit trail and cost at the bottom.
"""

import pandas as pd
import streamlit as st

from clinical_rag.schemas import Brief, ReviewAction, SectionDraft, TrialRow
from ui import components
from ui.backend import backend
from ui.layout import friendly_errors, user

STATUS_BADGE = {"pending": ("Pending review", "gray"), "approved": ("Approved", "green"),
                "edited": ("Edited", "blue"), "rejected": ("Rejected", "red"), "failed": ("Failed", "orange")}
BRIEF_STATUS = {"generating": "Generating", "draft": "Draft", "in_review": "In review", "approved": "Approved",
                "failed": "Failed"}


def phase_label(phase: list[str]) -> str:
    """["PHASE2", "PHASE3"] -> "Phase 2/3"; [] -> "Not stated" (registry codes are not for readers)."""
    if not phase:
        return "Not stated"
    if phase == ["EARLY_PHASE1"]:
        return "Early phase 1"
    if phase == ["NA"]:
        return "Not applicable"  # registry: studies without drug phases, e.g. behavioural
    numbers = [p.removeprefix("PHASE") for p in phase]
    return "Phase " + "/".join(numbers) if all(n.isdigit() for n in numbers) else ", ".join(phase)


def status_label(status: str | None) -> str:
    return status.replace("_", " ").capitalize() if status else "Not stated"


def pipeline(section: SectionDraft, brief: Brief) -> None:
    rows = section.table or []
    st.caption(f"Trial status as of {brief.data_as_of:%B %d, %Y}, straight from ClinicalTrials.gov. "
               "No language model is used in this section.")
    if not rows:
        st.info("No trials found for these drugs.")
        return
    trials = pd.DataFrame([{"Drug": d.strip(), "Phase": phase_label(t.phase)}
                           for t in rows for d in (t.drug or "Not stated").split(",")])
    phases = sorted(trials["Phase"].unique(),  # Early phase 1, Phase 1 ... Phase 4, Not applicable, Not stated
                    key=lambda p: (not p.startswith(("Early", "Phase")), not p.startswith("Early"), p))
    counts = pd.crosstab(trials["Drug"], trials["Phase"]).reindex(columns=phases, fill_value=0)
    counts["Total"] = counts.sum(axis=1)
    st.markdown("**Number of trials by phase**")
    st.dataframe(counts)
    with st.expander(f"All {len(rows)} trials"):
        st.dataframe(trial_table(rows), hide_index=True,
                     column_config={"Link": st.column_config.LinkColumn("Link", display_text="Open"),
                                    "Enrollment": st.column_config.NumberColumn(format="%d"),
                                    "First posted": st.column_config.DateColumn()})


def trial_table(rows: list[TrialRow]) -> pd.DataFrame:
    return pd.DataFrame([{"NCT ID": t.nct_id, "Title": t.title, "Drug": t.drug, "Phase": phase_label(t.phase),
                          "Status": status_label(t.status), "Sponsor": t.sponsor, "Enrollment": t.enrollment,
                          "First posted": t.first_posted, "Link": t.url} for t in rows])


def written_section(section: SectionDraft) -> None:
    if section.review_status == "failed":
        st.error("This section could not be generated. " + " ".join(section.not_found))
        return
    if section.review_status == "edited" and section.reviewer_text:
        st.markdown(section.reviewer_text)
        st.caption("Edited by the reviewer. The AI draft with its citations is below.")
        with st.expander("Original AI draft"):
            components.claims(section.claims, section.sources, key=f"{section.section_key}-original")
    else:
        if not section.claims:
            st.info("The sources don't cover this section.")
        components.claims(section.claims, section.sources, key=section.section_key)
    if section.not_found:
        with st.expander(f"Not found in the sources ({len(section.not_found)})"):
            st.markdown("\n".join(f"- {q}" for q in section.not_found))


def review_controls(brief: Brief, section: SectionDraft) -> None:
    key = section.section_key
    with st.container(border=True):
        label, color = STATUS_BADGE[section.review_status]
        st.badge(label, color=color)
        comment = st.text_input("Comment (optional)", key=f"comment-{key}")
        action = None
        with st.container(horizontal=True):
            if st.button("Approve", key=f"approve-{key}", icon=":material/check:"):
                action = ReviewAction(brief_id=brief.brief_id, section_key=key, action="approve",
                                      reviewer=user(), comment=comment or None)
            if st.button("Reject", key=f"reject-{key}", icon=":material/close:"):
                action = ReviewAction(brief_id=brief.brief_id, section_key=key, action="reject",
                                      reviewer=user(), comment=comment or None)
            if key != "pipeline":
                with st.popover("Edit", icon=":material/edit:"), st.form(f"edit-{key}"):
                    draft = section.reviewer_text or "\n".join(f"- {c.text}" for c in section.claims)
                    text = st.text_area("Section text", value=draft, height=300)
                    if st.form_submit_button("Save edit", type="primary"):
                        action = ReviewAction(brief_id=brief.brief_id, section_key=key, action="edit",
                                              reviewer=user(), edited_text=text, comment=comment or None)
        if action:
            with friendly_errors("saving the review"):
                backend.submit_review(action)
                st.rerun()


def audit_trail(brief_id: str) -> None:
    with friendly_errors("loading the audit trail"):
        events = backend.get_audit_log(brief_id)
        st.dataframe(pd.DataFrame([{"Time (UTC)": e.created_at.strftime("%Y-%m-%d %H:%M:%S"), "Who": e.actor,
                                    "Event": e.event, "Section": e.payload.get("section_key", ""),
                                    "Comment": e.payload.get("comment") or ""} for e in events]),
                     hide_index=True)


st.title("Review a brief")

briefs = []
with friendly_errors("loading the briefs"):
    briefs = backend.list_briefs()
if not briefs:
    st.info("No briefs yet. Create one on the Generate page.")
    st.page_link("pages/1_Generate.py", label="Generate a brief", icon=":material/edit_note:")
    st.stop()

ids = [b.brief_id for b in briefs]
labels = {b.brief_id: f"{' vs '.join(b.drugs)} · {b.created_at:%b %d, %H:%M} UTC · "
                      f"{BRIEF_STATUS.get(b.status, b.status)} · by {b.requested_by}" for b in briefs}
current = st.session_state.get("brief_id")
brief_id = st.selectbox("Brief", ids, index=ids.index(current) if current in ids else 0, format_func=labels.get)
st.session_state.brief_id = brief_id

brief = None
with friendly_errors("loading the brief"):
    brief = backend.get_brief(brief_id)
if brief is None:
    st.stop()

st.header(f"Competitive landscape: {' vs '.join(brief.request.drugs)} in obesity", divider="gray")
st.caption(f"Data as of {brief.data_as_of:%B %d, %Y} · {BRIEF_STATUS.get(brief.status, brief.status)} · "
           f"Generated by {brief.request.requested_by} on {brief.created_at:%B %d, %Y} · "
           "Learning project on public data. Not medical advice.")
st.markdown("**Contents:** " + " · ".join(f"[{s.title}](#{s.section_key.replace('_', '-')})"
                                          for s in brief.sections))
st.caption("Each claim has numbered sources under it: hover a number to see the passage, click it to open the "
           "source. Claims in yellow failed the automatic check and need a closer look.")

for section in brief.sections:
    st.subheader(section.title, anchor=section.section_key.replace("_", "-"))
    if section.section_key == "pipeline":
        pipeline(section, brief)
    else:
        written_section(section)
    review_controls(brief, section)

st.subheader("Audit trail")
audit_trail(brief.brief_id)
components.usage_footer(brief.usage)
