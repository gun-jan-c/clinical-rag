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


def button(app, label):
    return next(b for b in app.button if b.label == label)


def test_generate_shows_each_section_done_and_a_summary(app):
    log_in(app)
    app.switch_page("pages/1_Generate.py").run()
    button(app, "Generate brief").click().run()
    assert not app.exception
    labels = [s.proto.label for s in app.get("status")]
    assert len(labels) == 5 and all(label.endswith(": done") for label in labels)
    assert app.success[0].value.startswith("Brief ready: 5 sections")
    assert app.session_state.brief_id


def test_generate_is_disabled_when_the_daily_cap_is_reached(app, monkeypatch):
    monkeypatch.setattr(mock, "BRIEFS_CAP", 0)
    log_in(app)
    app.switch_page("pages/1_Generate.py").run()
    assert "limit of 0 briefs" in app.warning[-1].value
    assert button(app, "Generate brief").disabled


def test_when_the_brief_fails_no_section_is_left_spinning(app, monkeypatch):
    def fail_midway(req, on_progress=None):
        on_progress("pipeline", "done")
        on_progress("efficacy", "retrieving")
        raise RuntimeError("database down")

    monkeypatch.setattr(mock, "generate_brief", fail_midway)
    log_in(app)
    app.switch_page("pages/1_Generate.py").run()
    button(app, "Generate brief").click().run()
    labels = [s.proto.label for s in app.get("status")]
    assert labels[0].endswith(": done")
    assert all(label.endswith(": stopped") for label in labels[1:])
    assert "didn't work" in app.error[0].value


@pytest.fixture
def no_briefs(monkeypatch):
    monkeypatch.setattr(mock, "_briefs", {})
    monkeypatch.setattr(mock, "_audit", {})


def generate_and_open_review(app):
    log_in(app)
    app.switch_page("pages/1_Generate.py").run()
    button(app, "Generate brief").click().run()
    app.switch_page("pages/2_Review.py").run()
    assert not app.exception


def test_generate_fills_each_box_with_a_summary(app, no_briefs):
    log_in(app)
    app.switch_page("pages/1_Generate.py").run()
    button(app, "Generate brief").click().run()
    captions = [c.value for c in app.caption]
    assert "4 trials from the registry" in captions  # 2 drugs x 2 mock trials
    assert "3 claims · 1 flagged for review · 3 sources · 1 not found" in captions


def test_review_says_so_when_there_are_no_briefs(app, no_briefs):
    log_in(app)
    app.switch_page("pages/2_Review.py").run()
    assert "No briefs yet" in app.info[0].value


def test_review_shows_the_brief_as_one_document(app, no_briefs):
    generate_and_open_review(app)
    assert [s.value for s in app.subheader] == ["Development pipeline by phase", "Key efficacy results",
                                                "Safety and tolerability", "Competitive positioning",
                                                "Open questions and evidence gaps", "Audit trail"]
    counts = app.dataframe[0].value
    assert list(counts.columns) == ["Phase 2", "Phase 3", "Total"] and counts["Total"].sum() == 4
    assert sum("Unverified claim" in w.value for w in app.warning) == 4  # one flagged claim per written section


def test_approve_updates_the_status_and_the_audit_trail(app, no_briefs):
    generate_and_open_review(app)
    app.text_input(key="comment-efficacy").input("Checked against STEP 1")
    button_by_key(app, "approve-efficacy").click().run()
    assert not app.exception
    audit = app.dataframe[-1].value
    last = audit.iloc[-1]
    assert (last["Event"], last["Section"], last["Comment"]) == ("approved", "efficacy", "Checked against STEP 1")
    assert "In review" in app.selectbox[0].format_func(app.selectbox[0].value)


def test_edit_replaces_the_section_text(app, no_briefs):
    generate_and_open_review(app)
    safety = next(t for t in app.text_area if t.form_id == "edit-safety")
    safety.input("- Reviewer's rewritten safety text.")  # a form sends typed text only with its submit click
    next(b for b in app.button if b.label == "Save edit" and "safety" in b.form_id).click().run()
    assert not app.exception
    assert any(m.value == "- Reviewer's rewritten safety text." for m in app.markdown)
    assert app.dataframe[-1].value.iloc[-1]["Event"] == "edited"


def button_by_key(app, key):
    return next(b for b in app.button if b.key == key)
