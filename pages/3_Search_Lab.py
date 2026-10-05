"""Search Lab (ProjectSpec.md section 7): the same query through dense, keyword, hybrid and hybrid + rerank, side by
side, so a visitor can see why the app uses hybrid search and reranking.

Rerank is a 4th column rather than a toggle (decision log 2026-10-05): one Cohere call per search instead of three,
and the first three columns stay unreranked, so their differences stay visible.
"""

from concurrent.futures import ThreadPoolExecutor

import streamlit as st

from clinical_rag.schemas import ContentType, EmbeddingModel, SearchHit, Strategy
from ui.backend import backend
from ui.components import TYPE_ICON
from ui.layout import friendly_errors

# Column title -> (mode, rerank, what it does)
COLUMNS = {
    "Dense": ("dense", False, "Closest in meaning (embeddings). Good with plain-English questions, weak on exact codes."),
    "Keyword": ("keyword", False, "Exact words (Postgres full-text search). Good with trial IDs and drug codes."),
    "Hybrid": ("hybrid", False, "Both lists merged by rank (RRF): high on either list means high here."),
    "Hybrid + rerank": ("hybrid", True, "Cohere re-reads hybrid's top 20 and reorders them. This is what the app uses."),
}
# Spec section 7 presets: query -> content types it is meant for
PRESETS: dict[str, list[ContentType]] = {
    "NCT05872620": [], "LY3502970": [], "weight loss at week 72": [],
    "nausea incidence by dose": ["table"], "body weight over time figure": ["figure"],
}
STRATEGIES: dict[Strategy, str] = {"section": "By document section", "fixed": "Fixed size"}
MODELS: dict[EmbeddingModel, str] = {"oai-3-small": "text-embedding-3-small", "oai-3-large": "text-embedding-3-large"}


@st.cache_data(ttl=300, show_spinner=False)
def search_all(query: str, strategy: Strategy, model: EmbeddingModel,
               content_types: tuple[ContentType, ...]) -> dict[str, list[SearchHit]]:
    """All four columns at once (each is a separate database search, 1.5-3.7 s on Railway)."""
    with ThreadPoolExecutor(max_workers=len(COLUMNS)) as pool:
        futures = {title: pool.submit(backend.search, query, mode=mode, strategy=strategy, embedding_model=model,
                                      content_types=list(content_types) or None, rerank=rerank)
                   for title, (mode, rerank, _) in COLUMNS.items()}
        return {title: f.result() for title, f in futures.items()}


def use_preset(query: str, content_types: list[ContentType]) -> None:
    st.session_state["lab-query"] = query
    st.session_state["lab-types"] = content_types


def rank(n: int | None) -> str:
    return f"#{n}" if n is not None else "not found"


def show_hit(n: int, hit: SearchHit) -> None:
    s = hit.source
    with st.container(border=True):
        st.markdown(f"**{n}. [{s.title}]({s.url})**")
        facts = [f"{TYPE_ICON[s.content_type]} {s.content_type}", s.label, s.section, s.source]
        st.caption(" · ".join(f for f in facts if f))
        st.caption(s.snippet)
        ranks = f"Dense {rank(hit.dense_rank)} · Keyword {rank(hit.keyword_rank)}"
        if hit.rerank_score is not None:
            ranks += f" · Rerank score {hit.rerank_score:.2f}"
        st.caption(f"**{ranks}**")


st.title("Search Lab")
st.write("One search, four ways. Each column shows the top 10 passages from the same trial registry entries, papers and "
         "FDA labels. Under each result you can see where the other searches ranked it. "
         "No language model writes anything on this page.")

st.write("Try one:")
with st.container(horizontal=True):
    for query, types in PRESETS.items():
        st.button(query, on_click=use_preset, args=(query, types), key=f"preset-{query}")

query = st.text_input("Search", key="lab-query", placeholder="e.g. tirzepatide gastrointestinal adverse events")
with st.container(horizontal=True):
    strategy = st.radio("Chunking", list(STRATEGIES), format_func=STRATEGIES.get, horizontal=True)
    model = st.radio("Embedding model", list(MODELS), format_func=MODELS.get, horizontal=True)
    types = st.pills("Content type (none = all)", ["text", "table", "figure"], selection_mode="multi",
                     key="lab-types")

if query.strip():
    with friendly_errors("the search"), st.spinner("Searching four ways..."):
        results = search_all(query.strip(), strategy, model, tuple(types))
        for col, (title, hits) in zip(st.columns(len(COLUMNS)), results.items()):
            with col:
                st.subheader(title)
                st.caption(COLUMNS[title][2])
                if not hits:
                    st.info("No results.")
                for n, hit in enumerate(hits, start=1):
                    show_hit(n, hit)
