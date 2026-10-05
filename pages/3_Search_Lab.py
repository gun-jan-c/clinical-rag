"""Search Lab (ProjectSpec.md section 7), a guided tutorial for demos (decision log 2026-10-05).

Step 1, ranking: the same query through dense, keyword, hybrid and hybrid + rerank, side by side. Step 2, chunking:
hybrid + rerank over section chunks vs fixed-size chunks. Every column and result gets a note computed from its
ranks, so the notes are true for any query; the five presets also get a hand-written lesson.
"""

import re
from concurrent.futures import ThreadPoolExecutor

import streamlit as st

from clinical_rag.schemas import ContentType, EmbeddingModel, SearchHit
from ui.backend import backend
from ui.components import TYPE_ICON
from ui.layout import friendly_errors

RRF_K = 60  # hybrid_search's rrf_k (db/migrations/002_hybrid_search.sql)
POOL = 50  # candidates each list keeps (candidate_pool)
SHOWN = 10
HYBRID_POOL = 20  # rerank reorders hybrid's top 20 (retriever.CANDIDATES)

RANKING = {
    "Dense": "Closest in meaning (embeddings). Good with plain-English questions, weak on exact codes.",
    "Keyword": "Your exact words, plus a drug's other names (Postgres full-text search). Good with IDs and codes.",
    "Hybrid": f"Both lists merged by rank (RRF): each list adds 1/({RRF_K} + rank). High on either list, or found by "
              "both, means high here.",
    "Hybrid + rerank": f"Cohere reads hybrid's top {HYBRID_POOL} next to your search and reorders them. "
                       "This is what the app uses.",
}
CHUNKING = {
    "Section chunks": "One chunk per document section; tables and figure captions are chunks of their own. "
                      "This is what the app uses.",
    "Fixed-size chunks": "The baseline: all text cut into ~512-token slices, tables flattened into plain text.",
}
MODELS: dict[EmbeddingModel, str] = {"oai-3-small": "text-embedding-3-small", "oai-3-large": "text-embedding-3-large"}

# Spec section 7 presets: query -> (content types, headline, ranking lesson, chunking lesson). Written from the real
# results on 2026-10-05 (text-embedding-3-small); the computed notes under each column always reflect the live data.
LESSONS_DATE = "October 5, 2026"
PRESETS: dict[str, tuple[list[ContentType], str, str, str]] = {
    "NCT05872620": (
        [], "Exact IDs need keyword search",
        ("A trial ID has no meaning, so dense search returns other obesity trials that merely look similar. Keyword "
         "search matches the ID itself: the ATTAIN-2 registry entry comes first, because a document's own ID is "
         "indexed with extra weight. Hybrid keeps it first. Rerank then lifts the published ATTAIN-2 paper from "
         "hybrid #11 to #1: Cohere judges which passage is most useful, and the results paper beats the registry form."),
        ("Both chunkings find the same two documents (the paper and the registry entry): an ID is found the same way "
         "however the text is cut.")),
    "LY3502970": (
        [], "Code names: query expansion + hybrid scoring",
        ("LY3502970 is orforglipron's code name. Dense search returns unrelated papers: a code carries no meaning. "
         "Keyword search also searches \"orforglipron\" (query expansion adds a drug's other names), so it finds the "
         "trials. Look at the hybrid scores: a trial ranked #49 by dense and #3 by keyword (1/109 + 1/63 = 0.025) beats "
         "the trial ranked #1 by keyword alone (1/61 = 0.016). Being found by both lists adds up."),
        ("With fixed chunks, dense search also finds these trials (dense #1 to #4 instead of #45 to #49), probably "
         "because a short slice that repeats the code is closer to the query than a whole section is. Both chunkings "
         "end with largely the same trials on top.")),
    "weight loss at week 72": (
        [], "Plain-English questions need dense search",
        ("Keyword search scores word matches, so its #1 is a liraglutide trial in teenagers whose outcome list repeats "
         "\"weight\", \"loss\" and \"week\". Dense search finds passages about weight change over time. Rerank puts a "
         "semaglutide 7.2 mg trial with week-72 weight outcomes first."),
        ("Fixed chunks cut through a results table: several slices are rows of numbers (\"−9.2 (3.1) −9.0 (3.1) …\") "
         "whose column headers are in another chunk, so a model can't tell which number belongs to which group. "
         "Section chunks keep a table whole, headers included.")),
    "nausea incidence by dose": (
        ["table"], "Tables: found by caption, or by flattened rows",
        ("With section chunks a table is searched by its caption, a one-line summary and its header row, not its "
         "rows. Dense finds a tirzepatide table of nausea by week; keyword finds label adverse-reaction tables "
         "(Zepbound, Mounjaro), but rerank tends to drop them: Cohere sees the same caption and summary, not the rows "
         "that list nausea (a known gap, future enhancement B)."),
        ("Fixed chunks contain the table rows as plain text, so they find rows that name nausea, such as the Wegovy "
         "label's \"Placebo … WEGOVY 2.4 mg … Nausea 16 44\". But other slices start mid-table (\"Nausea 2 (22.2) 0 7 "
         "(77.8) …\") with no column names, so a model can't tell which dose each number belongs to. Trade-off: fixed "
         "finds rows better; section is safer to answer from. Fixed chunks have no table type, so the filter is off.")),
    "body weight over time figure": (
        ["figure"], "Figures are found by their captions",
        ("Section chunking stores each figure caption as its own chunk, so the figure filter returns plots of body "
         "weight over time. Dense and keyword pick different figures; rerank puts the captions that describe body "
         "weight over time most directly first. The app shows captions and links, never the image."),
        ("Fixed chunks have no figures: captions are mixed into text slices, so the filter can't apply and most "
         "results are text about body-weight models. The Wegovy label's \"Figure 4. Change in Body Weight (%) …\" "
         "still appears, inside a slice of text.")),
}


@st.cache_data(ttl=300, show_spinner=False)
def search_all(query: str, model: EmbeddingModel, content_types: tuple[ContentType, ...]) -> dict[str, list]:
    """All six searches at once (each is a separate database search, 1.5-3.7 s on Railway; 2 Cohere calls)."""
    base = {"strategy": "section", "embedding_model": model, "content_types": list(content_types) or None,
            "rerank": False}
    calls = {"Dense": dict(base, mode="dense"), "Keyword": dict(base, mode="keyword"),
             "Hybrid": dict(base, mode="hybrid", k=HYBRID_POOL),  # 10 shown; all 20 tell rerank's moves
             "Hybrid + rerank": dict(base, mode="hybrid", rerank=True),
             "Fixed-size chunks": dict(base, mode="hybrid", rerank=True, strategy="fixed",
                                       content_types=None)}  # fixed chunks are all plain text
    with ThreadPoolExecutor(max_workers=len(calls)) as pool:
        futures = {name: pool.submit(backend.search, query, **kwargs) for name, kwargs in calls.items()}
        return {name: f.result() for name, f in futures.items()}


def use_preset(query: str) -> None:
    st.session_state["lab-query"] = query
    st.session_state["lab-types"] = PRESETS[query][0]


def plain(text: str) -> str:
    """Show source text as typed: "- " must not become a bullet, "*" italics, "$" a formula."""
    return re.sub(r"([\\`*_{}\[\]()#+\-.!|>~<$])", r"\\\1", text)


def card(n: int, hit: SearchHit, why: str) -> None:
    s = hit.source
    with st.container(border=True):
        st.markdown(f"**{n}. [{plain(s.title)}]({s.url})**")
        facts = [f"{TYPE_ICON[s.content_type]} {s.content_type}", s.label, s.section, s.source]
        st.caption(" · ".join(plain(f) for f in facts if f))
        st.markdown(why)
        with st.expander("Passage"):
            st.caption(plain(s.snippet))


def why_dense(h: SearchHit) -> str:
    other = (f"Keyword ranked it #{h.keyword_rank} too: it also shares your words." if h.keyword_rank
             else f"Not in keyword's top {POOL}: it doesn't share enough of your words.")
    return f"**Dense #{h.dense_rank}**: close in meaning to your search. {other}"


def why_keyword(h: SearchHit) -> str:
    other = (f"Dense ranked it #{h.dense_rank} too." if h.dense_rank
             else f"Not in dense's top {POOL}: the words match, the meaning is not close enough.")
    return f"**Keyword #{h.keyword_rank}**: contains your words (or a drug's other name, or its own ID). {other}"


def why_hybrid(h: SearchHit) -> str:
    parts = [f"1/({RRF_K}+{h.dense_rank}) from dense" if h.dense_rank else "0 from dense",
             f"1/({RRF_K}+{h.keyword_rank}) from keyword" if h.keyword_rank else "0 from keyword"]
    both = " Found by both lists." if h.dense_rank and h.keyword_rank else ""
    return f"**Score {h.rrf_score:.4f}** = {' + '.join(parts)}.{both}"


def why_rerank(n: int, h: SearchHit, hybrid_pos: dict[str, int]) -> str:
    was = hybrid_pos.get(h.source.chunk_id)
    if was is None:
        move = ""
    elif was > n:
        move = f" Was hybrid #{was}: moved up {was - n}."
    elif was < n:
        move = f" Was hybrid #{was}: moved down {n - was}."
    else:
        move = f" Was hybrid #{was}: same place."
    return f"**Relevance {h.rerank_score:.2f}** (Cohere, 0 to 1).{move}"


def why_chunk(h: SearchHit, fixed: bool) -> str:
    score = f"**Relevance {h.rerank_score:.2f}**. " if h.rerank_score is not None else ""
    if fixed:
        return (score + "A ~512-token slice: it can start or end mid-sentence or mid-table, and table rows lose their "
                "column layout. The model reads just this slice.")
    return score + {"table": "A whole table: the model reads every row with its column headers.",
                    "figure": "A figure caption: the model reads the caption, never the image.",
                    "text": "A document section: the model reads the whole section (or the matching piece if the "
                            "section is very long)."}[h.source.content_type]


def headline(name: str, hits: list[SearchHit], hybrid_pos: dict[str, int]) -> str:
    n = len(hits)
    if name == "Dense":
        return f"{sum(h.keyword_rank is not None for h in hits)} of {n} also in keyword's top {POOL}."
    if name == "Keyword":
        return f"{sum(h.dense_rank is not None for h in hits)} of {n} also in dense's top {POOL}."
    if name == "Hybrid":
        both = sum(bool(h.dense_rank and h.keyword_rank) for h in hits)
        dense_only = sum(bool(h.dense_rank and not h.keyword_rank) for h in hits)
        return f"{both} found by both lists, {dense_only} by dense only, {n - both - dense_only} by keyword only."
    if name == "Hybrid + rerank":
        below = sum(hybrid_pos.get(h.source.chunk_id, 0) > SHOWN for h in hits)
        return f"{below} of {n} came from below hybrid's top {SHOWN}."
    docs = len({h.source.doc_id for h in hits})
    if name == "Fixed-size chunks":
        return f"{n} plain-text slices from {docs} documents."
    kinds = [f"{sum(h.source.content_type == t for h in hits)} {t}" for t in ("text", "table", "figure")]
    return f"{', '.join(kinds)} chunks from {docs} documents."


def column(name: str, how: str, hits: list[SearchHit], hybrid_pos: dict[str, int]) -> None:
    st.subheader(name)
    st.caption(how)
    if not hits:
        st.info("No results.")
        return
    st.markdown(f"**{headline(name, hits, hybrid_pos)}**")
    for n, h in enumerate(hits, start=1):
        if name == "Dense":
            why = why_dense(h)
        elif name == "Keyword":
            why = why_keyword(h)
        elif name == "Hybrid":
            why = why_hybrid(h)
        elif name == "Hybrid + rerank":
            why = why_rerank(n, h, hybrid_pos)
        else:
            why = why_chunk(h, fixed=name == "Fixed-size chunks")
        card(n, h, why)


def lesson(query: str, part: int) -> None:
    if query in PRESETS:
        _, title, ranking, chunking = PRESETS[query]
        st.info(f"**{title}.** {ranking if part == 1 else chunking}\n\n_Lesson written from the results on "
                f"{LESSONS_DATE}; the notes under each column always show today's results._",
                icon=":material/school:")


st.title("Search Lab")
st.write("How the app finds the passages a model reads, step by step. No language model writes anything on this "
         "page: every note below is computed from the search results.")

st.write("Try one:")
with st.container(horizontal=True):
    for preset in PRESETS:
        st.button(preset, on_click=use_preset, args=(preset,), key=f"preset-{preset}")

query = st.text_input("Search", key="lab-query", placeholder="e.g. tirzepatide gastrointestinal adverse events")
with st.container(horizontal=True):
    model = st.radio("Embedding model", list(MODELS), format_func=MODELS.get, horizontal=True)
    types = st.pills("Content type (none = all)", ["text", "table", "figure"], selection_mode="multi",
                     key="lab-types")

if query.strip():
    with friendly_errors("the search"), st.spinner("Searching six ways..."):
        results = search_all(query.strip(), model, tuple(types))
        hybrid_pos = {h.source.chunk_id: n for n, h in enumerate(results["Hybrid"], start=1)}
        shown = {name: hits[:SHOWN] for name, hits in results.items()}

        st.header("Step 1: Ranking")
        dense_ids = {h.source.chunk_id for h in shown["Dense"]}
        shared = sum(h.source.chunk_id in dense_ids for h in shown["Keyword"])
        st.write(f"Four ways to rank the same section chunks. Dense and keyword share **{shared} of their top "
                 f"{SHOWN}** passages, so each finds things the other misses.")
        lesson(query.strip(), 1)
        for col, (name, how) in zip(st.columns(len(RANKING)), RANKING.items()):
            with col:
                column(name, how, shown[name], hybrid_pos)

        st.header("Step 2: Chunking")
        st.write("How documents are cut decides what can be found and what the model gets to read. "
                 "Both columns use hybrid + rerank.")
        lesson(query.strip(), 2)
        chunk_results = {"Section chunks": shown["Hybrid + rerank"], "Fixed-size chunks": shown["Fixed-size chunks"]}
        for col, (name, how) in zip(st.columns(len(CHUNKING)), CHUNKING.items()):
            with col:
                column(name, how, chunk_results[name], hybrid_pos)
