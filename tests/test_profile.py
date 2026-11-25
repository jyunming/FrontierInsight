"""The author line kept once for every later quest (core/profile.py)."""

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


