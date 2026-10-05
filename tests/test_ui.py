"""The Streamlit app, run headless on the mock backend with a made-up access code."""

import os

os.environ["USE_MOCK"] = "1"  # before ui.backend is imported; .env never overrides it

import pytest
from streamlit.testing.v1 import AppTest

from clinical_rag.services import mock


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("ACCESS_CODES", '{"test-code": "Tester"}')
    monkeypatch.setattr(mock, "DELAY_SECONDS", 0)
    return AppTest.from_file("../streamlit_app.py", default_timeout=30).run()


def log_in(app, code="test-code"):
    app.text_input[0].input(code)
    app.button[0].click().run()


def test_logged_out_users_see_only_the_login_page(app):
    assert not app.exception
    assert [t.value for t in app.title] == ["Clinical RAG Brief Generator"]
    assert not app.sidebar.button  # no log-out button, no usage meter
    log_in(app, "wrong")
    assert app.error[0].value == "That access code isn't valid."
    assert "user" not in app.session_state


def test_home_shows_data_date_and_answers_a_question(app):
    log_in(app)
    assert app.session_state.user == "Tester"
    assert "Data as of" in app.info[0].value
    app.text_input[0].input("What weight loss did tirzepatide show?")
    app.button[0].click().run()
    assert not app.exception
    assert any("[Mock] Answer to" in m.value for m in app.markdown)


def test_usage_meter_counts_the_question_just_asked(app):
    log_in(app)
    before = mock.get_usage_today().asks_today
    app.text_input[0].input("Any question")
    app.button[0].click().run()
    assert any(f"Questions: {before + 1} of" in p.proto.text for p in app.sidebar.get("progress"))


def test_out_of_scope_question_says_not_found(app):
    log_in(app)
    app.text_input[0].input("What is the list price of Wegovy in Germany?")
    app.button[0].click().run()
    assert any("don't answer this question" in i.value for i in app.info)
