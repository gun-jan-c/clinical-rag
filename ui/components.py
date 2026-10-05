"""Claims with citation chips (ProjectSpec.md section 7), shared by Home's Ask box and the Review page.

Each citation is a chip under its claim: clicking it opens the source document in a new tab; hovering shows the
cited chunk's snippet with its type, section, label, date and license. Only the snippet, never the full passage or
table (spec section 7: licensed articles). Unverified claims are shown in a warning box with their issues, never hidden.
"""

import streamlit as st

from clinical_rag.schemas import Claim, Source, Usage

TYPE_ICON = {"text": ":material/article:", "table": ":material/table_chart:", "figure": ":material/image:"}


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
