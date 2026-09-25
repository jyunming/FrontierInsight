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
    assert client.get("/api/profile").json() == {"profile": None}
    r = client.post("/api/interview/submit", json=_author_payload(author="Jane Chen", url="https://example.org"))
    assert r.status_code == 200, r.text
    assert client.get("/api/profile").json()["profile"]["author"] == "Jane Chen"
    assert profile.load()["url"] == "https://example.org"
