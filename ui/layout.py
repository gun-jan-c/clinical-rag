"""Shared layout (ProjectSpec.md section 7): access-code login, sidebar (user, usage meter, disclaimer), errors."""

import json
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from ui.backend import USING_MOCK, backend

log = logging.getLogger(__name__)

DISCLAIMER = "Learning project on public data. Not medical advice."


def access_codes() -> dict[str, str]:
    """Access code -> display name, from Streamlit secrets (deployed) or ACCESS_CODES in .env (local)."""
    try:
        codes = st.secrets["ACCESS_CODES"]
    except (StreamlitSecretNotFoundError, KeyError):
        codes = os.environ.get("ACCESS_CODES", "{}")
    return json.loads(codes) if isinstance(codes, str) else dict(codes)


def user() -> str | None:
    return st.session_state.get("user")


def login_page() -> None:
    st.title("Clinical RAG Brief Generator")
    st.write("Cited, reviewable competitive-landscape briefs on GLP-1 and incretin therapies for obesity, "
             "built from public trial registries, papers and FDA labels.")
    st.caption(DISCLAIMER)
    with st.form("login"):
        code = st.text_input("Access code", type="password")
        if st.form_submit_button("Log in", type="primary"):
            name = access_codes().get(code.strip())
            if name:
                st.session_state.user = name
                st.rerun()
            st.error("That access code isn't valid.")


@st.cache_data(ttl=300)
def data_as_of():
    return backend.get_data_as_of()


def sidebar() -> None:
    with st.sidebar:
        st.write(f"Signed in as **{user()}**")
        if USING_MOCK:
            st.warning("Mock data: every result is made up.", icon=":material/science:")
        try:
            usage = backend.get_usage_today()
            st.caption("Today's usage")
            st.progress(min(usage.briefs_today / usage.briefs_cap, 1.0),
                        text=f"Briefs: {usage.briefs_today} of {usage.briefs_cap}")
            st.progress(min(usage.asks_today / usage.asks_cap, 1.0),
                        text=f"Questions: {usage.asks_today} of {usage.asks_cap}")
        except Exception:
            log.exception("usage meter")
            st.caption("Usage is unavailable right now.")
        st.caption(DISCLAIMER)
        if st.button("Log out"):
            del st.session_state.user
            st.rerun()


@contextmanager
def friendly_errors(what: str) -> Iterator[None]:
    """Daily cap reached -> a friendly note; any other error -> a plain message on the page and the details in
    the server log, never a stack trace."""
    try:
        yield
    except backend.CapExceededError as e:
        st.warning(f"{e}. Please come back tomorrow.", icon=":material/hourglass_disabled:")
    except Exception:
        log.exception(what)
        st.error(f"Sorry, {what} didn't work. Please try again in a minute.")
