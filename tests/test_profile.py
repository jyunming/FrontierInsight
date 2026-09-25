"""The author line kept once for every later quest (core/profile.py), and the web page's use of it."""

from __future__ import annotations

from core import profile


def test_no_profile_until_saved_then_every_field_is_kept_on_one_line() -> None:
    assert profile.load() is None, "the test's own path, empty (tests/conftest.py)"
    kept = profile.save({"author": "  Jane \n Chen ", "affiliation": "R&D Lab"})
    assert kept == {"author": "Jane Chen", "affiliation": "R&D Lab", "contact_email": "", "url": ""}
    assert profile.load() == kept


def test_saving_blank_fields_still_counts_as_asked() -> None:
    profile.save({})
    assert profile.load() == {k: "" for k in profile.FIELDS}


def test_an_unreadable_profile_is_asked_again() -> None:
    profile.path().parent.mkdir(parents=True, exist_ok=True)
    profile.path().write_text("{half", encoding="utf-8")
    assert profile.load() is None


def test_the_web_page_reads_the_profile_and_a_submit_keeps_the_author_line(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from fastapi.testclient import TestClient

    from tests.test_web_interview_tier_flow import _author_payload
    from web.server import make_app

    client = TestClient(make_app(tmp_path))
    assert client.get("/api/profile").json()["profile"] is None
    r = client.post("/api/interview/submit", json=_author_payload(author="Jane Chen", url="https://example.org"))
    assert r.status_code == 200, r.text
    assert client.get("/api/profile").json()["profile"]["author"] == "Jane Chen"
    assert profile.load()["url"] == "https://example.org"


def test_the_web_parser_keeps_what_the_review_screen_offers_and_needs_a_provider() -> None:
    import pytest

    from tests.test_web_interview_tier_flow import _author_payload
    from web.interview_routes import _parse_answers

    answers = _parse_answers(_author_payload(pause_for_user_input="after_design", paper_style="briefing",
                                             node_models="write:model-x"))
    assert (answers.pause_for_user_input, answers.paper_style, answers.node_models) == (
        "after_design", "briefing", "write:model-x")
    with pytest.raises(ValueError, match="provider"):
        _parse_answers(_author_payload(provider=""))


def test_the_profile_is_not_served_to_a_page_on_another_machine(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from fastapi.testclient import TestClient

    from web.server import make_app

    profile.save({"author": "Jane Chen"})
    remote = TestClient(make_app(tmp_path), client=("192.168.1.20", 5000))
    assert remote.get("/api/profile").json()["profile"] is None


def test_a_page_behind_a_proxy_is_not_local_and_mapped_loopback_is(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from fastapi.testclient import TestClient

    from web.server import make_app

    profile.save({"author": "Jane Chen"})
    local = TestClient(make_app(tmp_path))
    assert local.get("/api/profile", headers={"X-Forwarded-For": "203.0.113.9"}).json()["profile"] is None
    mapped = TestClient(make_app(tmp_path), client=("::ffff:127.0.0.1", 5000))
    assert mapped.get("/api/profile").json()["profile"]["author"] == "Jane Chen"


def test_a_page_that_did_not_change_the_author_line_does_not_overwrite_a_newer_one(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from fastapi.testclient import TestClient

    from tests.test_web_interview_tier_flow import _author_payload
    from web.server import make_app

    seen = profile.save({"author": "Jane"})
    profile.save({"author": "Alice"})  # another tab changed it meanwhile
    client = TestClient(make_app(tmp_path))
    r = client.post("/api/interview/submit", json=_author_payload(author="Jane", profile_seen=seen))
    assert r.status_code == 200 and r.json()["profile_saved"] is None
    assert profile.load()["author"] == "Alice"


def test_a_malformed_per_step_model_entry_is_refused() -> None:
    import pytest

    from tests.test_web_interview_tier_flow import _author_payload
    from web.interview_routes import _parse_answers

    with pytest.raises(ValueError, match="node_models"):
        _parse_answers(_author_payload(node_models="bogus"))


def test_an_update_that_does_not_send_the_review_fields_keeps_the_quests_own() -> None:
    from types import SimpleNamespace

    from web.interview_routes import _review_extras

    current = SimpleNamespace(pause_for_user_input="after_design", paper_style="briefing", node_models="write:m")
    assert _review_extras({}, current) == {"pause_for_user_input": "after_design", "paper_style": "briefing",
                                           "node_models": "write:m"}
    assert _review_extras({"paper_style": "latex"}, current)["paper_style"] == "latex"


def test_the_deliverables_rule_reads_word_boundaries_as_the_other_interfaces_do() -> None:
    from core.interview import smart_default_output_kinds

    assert "poster" in smart_default_output_kinds({"topic": "an époster session"})
