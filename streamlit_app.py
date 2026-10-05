"""Streamlit entry point and Home page (ProjectSpec.md section 7). Run: streamlit run streamlit_app.py

Logged-out users get only the login page: st.navigation lists the other pages only after login.
"""

import streamlit as st

from ui import components
from ui.backend import backend
from ui.layout import data_as_of, friendly_errors, login_page, sidebar, user

EXAMPLE_QUESTION = "What weight reduction did tirzepatide achieve versus placebo in SURMOUNT-1?"


def home() -> None:
    st.title("Clinical RAG Brief Generator")
    with friendly_errors("loading the data date"):
        st.info(f"Data as of **{data_as_of():%B %d, %Y}** (date of the last ingestion run).",
                icon=":material/calendar_today:")
    st.write("This app writes cited competitive-landscape briefs on GLP-1 and incretin therapies for obesity, "
             "from public sources: ClinicalTrials.gov, PubMed, Europe PMC open-access papers and FDA labels. "
             "Every claim cites the passage it came from, and a person approves, edits or rejects each section.")
    with st.expander("How it works"):
        st.markdown(
            "1. **Search:** keyword and vector search over the source passages, combined (hybrid), "
            "then reranked by Cohere.\n"
            "2. **Write:** the model writes claims from those passages only, each citing its sources. "
            "What the sources don't cover is listed as *not found*, never filled in from general knowledge.\n"
            "3. **Check:** every number in a claim must appear in the source it cites. "
            "Claims that fail are flagged, never hidden.\n"
            "4. **Review:** a person approves, edits or rejects each section; every action is logged.\n\n"
            "The trial pipeline tables come straight from the registry by SQL. No language model is used there.")

    st.subheader("Ask a question")
    with st.form("ask"):
        question = st.text_input("Question", placeholder=EXAMPLE_QUESTION, label_visibility="collapsed")
        asked = st.form_submit_button("Ask", type="primary")
    if asked and question.strip():
        with friendly_errors("answering the question"), st.spinner("Searching the sources and writing the answer..."):
            st.session_state.answer = backend.ask(question.strip(), user())
    answer = st.session_state.get("answer")
    if answer:
        st.markdown(f"**{answer.question}**")
        if not answer.claims and answer.not_found:
            st.info("The sources don't answer this question.", icon=":material/search_off:")
        components.claims(answer.claims, answer.sources, key="answer")
        components.not_found(answer.not_found)
        components.usage_footer(answer.usage)


st.set_page_config(page_title="Clinical RAG Brief Generator", page_icon=":material/biotech:", layout="wide")
if user() is None:
    st.navigation([st.Page(login_page, title="Log in", icon=":material/login:")]).run()
else:
    page = st.navigation([st.Page(home, title="Home", icon=":material/home:", default=True),
                          st.Page("pages/1_Generate.py", title="Generate", icon=":material/edit_note:"),
                          st.Page("pages/2_Review.py", title="Review", icon=":material/fact_check:"),
                          st.Page("pages/3_Search_Lab.py", title="Search Lab", icon=":material/manage_search:")])
    page.run()
    sidebar()  # after the page, so the usage meter includes what the page just did
