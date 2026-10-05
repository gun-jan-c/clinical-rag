"""Claims with citation chips (ProjectSpec.md section 7), shared by Home's Ask box and the Review page.

Each citation is a chip under its claim: clicking it opens the source document in a new tab; hovering shows the
cited chunk's snippet with its type, section, label, date and license. Only the snippet, never the full passage or
table (spec section 7: licensed articles). Unverified claims are shown in a warning box with their issues, never hidden.
"""

import pandas as pd
import streamlit as st

from clinical_rag.schemas import Claim, Source, TrialRow, Usage

TYPE_ICON = {"text": ":material/article:", "table": ":material/table_chart:", "figure": ":material/image:"}
DRUGS = ["semaglutide", "tirzepatide", "liraglutide", "orforglipron",
         "retatrutide", "survodutide", "cagrilintide", "mazdutide"]  # spec section 2
TRIAL_COLUMNS = {"Link": st.column_config.LinkColumn("Link", display_text="Open"),
                 "Enrollment": st.column_config.NumberColumn(format="%d"),
                 "First posted": st.column_config.DateColumn()}


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


def phase_order(labels) -> list[str]:
    """Early phase 1, Phase 1 ... Phase 4, then Not applicable and Not stated."""
    return sorted(set(labels), key=lambda p: (not p.startswith(("Early", "Phase")), not p.startswith("Early"), p))


def status_label(status: str | None) -> str:
    return status.replace("_", " ").capitalize() if status else "Not stated"


def trial_table(rows: list[TrialRow]) -> pd.DataFrame:
    return pd.DataFrame([{"NCT ID": t.nct_id, "Title": t.title, "Drug": t.drug, "Phase": phase_label(t.phase),
                          "Status": status_label(t.status), "Sponsor": t.sponsor, "Enrollment": t.enrollment,
                          "First posted": t.first_posted, "Link": t.url} for t in rows])


def tooltip(s: Source) -> str:
    """Hover text for a citation chip: where the passage is from, then the start of the cited chunk."""
    facts = [s.content_type, s.label, s.section and f"Section: {s.section}", s.source,
             s.published_date and f"Published {s.published_date}", s.license and f"License: {s.license}"]
    return f"**{s.title}**\n\n{' · '.join(f for f in facts if f)}\n\n> {s.snippet}"


def claims(items: list[Claim], sources: list[Source], key: str) -> None:
    """`key` keeps chips unique when several claim lists are on one page (e.g. one per brief section)."""
    number = {s.chunk_id: n for n, s in enumerate(sources, start=1)}
    by_id = {s.chunk_id: s for s in sources}
    for i, c in enumerate(items):
        if c.verified:
            st.markdown(c.text)
        else:
            st.warning(f"**Unverified claim.** {c.text}\n\n_{'; '.join(c.issues)}_", icon=":material/warning:")
        cited = [cid for cid in c.citation_ids if cid in by_id]
        if cited:
            with st.container(horizontal=True):
                for cid in cited:
                    s = by_id[cid]
                    st.link_button(f"[{number[cid]}]", s.url, help=tooltip(s), icon=TYPE_ICON[s.content_type],
                                   key=f"{key}-{i}-{cid}")


def not_found(items: list[str]) -> None:
    if items:
        st.markdown("**Not found in the sources**")
        st.markdown("\n".join(f"- {q}" for q in items))


def usage_footer(u: Usage) -> None:
    st.caption(f"{u.input_tokens:,} input + {u.output_tokens:,} output tokens · ${u.cost_usd:.4f} · "
               f"{u.latency_ms / 1000:.1f} s")
