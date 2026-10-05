"""Generate page (ProjectSpec.md section 7): pick drugs and sections, then watch each section's progress live.

generate_brief runs in a background thread. Its sections report progress from their own worker threads, and
Streamlit can only draw from the page's thread, so progress events go through a queue that this page reads.
If the reader leaves the page mid-run, the brief still finishes and is saved.
"""

import queue
import threading

import streamlit as st

from clinical_rag.schemas import Brief, BriefRequest, SectionDraft, SectionKey
from ui.backend import backend
from ui.layout import friendly_errors, user

DRUGS = ["semaglutide", "tirzepatide", "liraglutide", "orforglipron",
         "retatrutide", "survodutide", "cagrilintide", "mazdutide"]  # spec section 2
MAX_DRUGS = 3  # keeps a brief near 2 minutes and under Cohere's 10 rerank calls a minute (decision log)
SECTIONS: dict[SectionKey, str] = {
    "pipeline": "Development pipeline by phase",
    "efficacy": "Key efficacy results",
    "safety": "Safety and tolerability",
    "competitive_positioning": "Competitive positioning",
    "evidence_gaps": "Open questions and evidence gaps",
}
STEPS = {"retrieving": "searching sources", "generating": "writing claims", "verifying": "checking claims",
         "done": "done", "failed": "failed"}


def generate_with_progress(req: BriefRequest) -> Brief:
    boxes = {k: st.status(f"{SECTIONS[k]}: waiting", state="running") for k in req.sections}
    events: queue.Queue = queue.Queue()
    result: dict = {}
    finished: set[SectionKey] = set()

    def work() -> None:
        try:
            result["brief"] = backend.generate_brief(req, on_progress=lambda key, step: events.put((key, step)))
        except Exception as e:  # noqa: BLE001 (re-raised on the page's thread below)
            result["error"] = e

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    while thread.is_alive() or not events.empty():
        try:
            key, step = events.get(timeout=0.2)
        except queue.Empty:
            continue
        state = {"done": "complete", "failed": "error"}.get(step, "running")
        boxes[key].update(label=f"{SECTIONS[key]}: {STEPS[step]}", state=state)
        if step in ("done", "failed"):
            finished.add(key)
    if "error" in result:
        for key in boxes.keys() - finished:  # the brief stopped: no spinner may keep going
            boxes[key].update(label=f"{SECTIONS[key]}: stopped", state="error")
        raise result["error"]
    for section in result["brief"].sections:  # so an expanded box isn't empty
        with boxes[section.section_key]:
            st.caption(summary(section))
    return result["brief"]


def summary(s: SectionDraft) -> str:
    if s.review_status == "failed":
        return " ".join(s.not_found)
    if s.table is not None:
        return f"{len(s.table)} trials from the registry"
    flagged = sum(not c.verified for c in s.claims)
    return (f"{len(s.claims)} claims · {flagged} flagged for review · {len(s.sources)} sources · "
            f"{len(s.not_found)} not found")


st.title("Generate a brief")
st.write("Pick up to 3 drugs and the sections you want. A brief takes about 1–2 minutes; "
         "you can watch each section below as it is written.")

cap_reached = False
with friendly_errors("checking today's usage"):
    usage = backend.get_usage_today()
    cap_reached = usage.briefs_today >= usage.briefs_cap
    if cap_reached:
        st.warning(f"Today's limit of {usage.briefs_cap} briefs has been reached. Please come back tomorrow.",
                   icon=":material/hourglass_disabled:")

with st.form("generate"):
    drugs = st.multiselect("Drugs", DRUGS, default=["semaglutide", "tirzepatide"], max_selections=MAX_DRUGS)
    st.write("Sections")
    chosen = [key for key, title in SECTIONS.items() if st.checkbox(title, value=True, key=f"section-{key}")]
    submitted = st.form_submit_button("Generate brief", type="primary", disabled=cap_reached)

if submitted:
    if not drugs or not chosen:
        st.error("Pick at least one drug and one section.")
    else:
        with friendly_errors("generating the brief"):
            brief = generate_with_progress(BriefRequest(drugs=drugs, sections=chosen, requested_by=user()))
            st.session_state.brief_id = brief.brief_id
            flagged = sum(not c.verified for s in brief.sections for c in s.claims)
            failed = [SECTIONS[s.section_key] for s in brief.sections if s.review_status == "failed"]
            st.success(f"Brief ready: {len(brief.sections)} sections, "
                       f"{sum(len(s.claims) for s in brief.sections)} claims ({flagged} flagged for review), "
                       f"${brief.usage.cost_usd:.4f}, {brief.usage.latency_ms / 1000:.0f} s.",
                       icon=":material/check_circle:")
            if failed:
                st.error(f"These sections could not be generated: {', '.join(failed)}.")
            st.page_link("pages/2_Review.py", label="Read and review this brief", icon=":material/fact_check:")
