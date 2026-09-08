"""Tests for ``core.numeric_oracle`` — the one gate that is arithmetic.

Every other correctness check in FI ends in a language model reading text.
This one compares the paper's numbers to the computed ones directly, so its
value depends entirely on two properties: it must catch a real
mis-transcription, and it must not fire on the many numbers a paper
legitimately contains that never came from ``result_json``.

A false positive here is expensive — the finding is a blocking must-flag and
costs a whole revision iteration — so most of these tests pin the *silence*.
"""
from __future__ import annotations

import pytest

from core import numeric_oracle as no


# ---------------------------------------------------------------------------
# The failure mode this exists for
# ---------------------------------------------------------------------------


def test_transposed_digits_are_caught() -> None:
    """The classic mis-copy: the run produced 2.14, the paper says 2.41.

    Same digits, different order. This is the case that passes every other
    gate in the pipeline, because 2.41 is a perfectly plausible number.
    """
    report = no.check(
        "The dipole source improves NILS to 2.41, a substantial gain.",
        {"nils_dipole": 2.14},
    )
    assert not report.ok
    assert len(report.findings) == 1
    f = report.findings[0]
    assert f.kind == "transposed"
    assert f.paper_value == 2.41
    assert f.result_value == 2.14
    assert f.result_path == "nils_dipole"
    assert "nils_dipole" in f.describe()


def test_near_miss_is_caught_with_its_path() -> None:
    report = no.check(
        "Contrast reached 0.812 under quadrupole illumination.",
        {"metrics": {"contrast": {"quadrupole": 0.79}}},
    )
    assert not report.ok
    f = report.findings[0]
    assert f.kind == "near_miss"
    assert f.result_path == "metrics.contrast.quadrupole"
    assert 0 < f.rel_error <= no.NEAR_REL


def test_transposed_outranks_near_miss_in_the_report() -> None:
    """A truncated report must still lead with the strongest signal."""
    report = no.check(
        "CD came out at 37.4 nm and the contrast was 0.812.",
        {"cd_nm": 34.7, "contrast": 0.79},
    )
    kinds = [f.kind for f in report.findings]
    assert kinds[0] == "transposed", kinds


# ---------------------------------------------------------------------------
# Silence: the expensive half
# ---------------------------------------------------------------------------


def test_exact_match_is_silent() -> None:
    report = no.check("NILS reached 2.14 at best focus.", {"nils": 2.14})
    assert report.ok


def test_correctly_rounded_quote_is_silent() -> None:
    """A paper quoting 2.1 for a computed 2.14 is right, not wrong.

    Rounding to the paper's own stated precision is the test — this is the
    case the roadmap flagged as the likeliest false positive.
    """
    for quoted, actual in [("2.1", 2.14), ("0.045", 0.04503), ("37", 37.2)]:
        report = no.check(f"The measured value was {quoted}.", {"v": actual})
        assert report.ok, f"{quoted} vs {actual} should round-match: {report.findings}"


def test_unrelated_numbers_are_ignored() -> None:
    """Papers are full of numbers that never came from the experiment.

    Anything beyond NEAR_REL of a result is assumed to come from the design,
    the literature, or the prose itself.
    """
    paper = (
        "We used a 193 nm source at NA 1.35 across 74 nm pitch, following "
        "the approach of prior work from 2019. Runtime was 12 minutes."
    )
    report = no.check(paper, {"nils": 2.14, "contrast": 0.79})
    assert report.ok, report.findings


@pytest.mark.parametrize(
    "paper",
    [
        "As shown in Figure 2.4, the trend is clear.",
        "See Table 2.4 for the full sweep.",
        "Described in Section 2.4 of the methods.",
        "Reported in reference 2.4 of the bibliography.",
        "Available at doi 2.4 (illustrative).",
    ],
)
def test_structural_references_are_not_measurements(paper: str) -> None:
    """Figure / table / section / reference numbers must never fire."""
    report = no.check(paper, {"nils": 2.14})
    assert report.ok, f"{paper!r} -> {report.findings}"


def test_code_blocks_are_ignored() -> None:
    paper = "Results follow.\n\n```python\nthreshold = 2.41\n```\n\nNo claim here.\n"
    report = no.check(paper, {"threshold": 2.14})
    assert report.ok, report.findings


def test_headings_are_ignored() -> None:
    report = no.check("## 2.41 Results\n\nNothing claimed.\n", {"v": 2.14})
    assert report.ok, report.findings


def test_low_precision_numbers_do_not_fire() -> None:
    """"about 2" against a computed 2.14 is not evidence of anything."""
    report = no.check("The value was about 2.", {"v": 2.14})
    assert report.ok, report.findings


def test_markdown_tables_are_checked() -> None:
    """A results table is where a wrong number does the most damage, so
    table rows are deliberately NOT stripped."""
    paper = "| Source | NILS |\n|---|---|\n| dipole | 2.41 |\n"
    report = no.check(paper, {"nils_dipole": 2.14})
    assert not report.ok
    assert report.findings[0].kind == "transposed"


# ---------------------------------------------------------------------------
# Degrading safely
# ---------------------------------------------------------------------------


def test_empty_result_json_skips_rather_than_flags() -> None:
    """No results means nothing to contradict — a survey paper must pass."""
    report = no.check("The literature reports values near 2.41.", {})
    assert report.ok and report.skipped
    assert "numeric" in report.skip_reason


def test_empty_paper_skips() -> None:
    report = no.check("   ", {"v": 2.14})
    assert report.ok and report.skipped


def test_booleans_are_not_metrics() -> None:
    report = no.check("The value was 1.02.", {"converged": True, "ok": False})
    assert report.ok and report.skipped


def test_non_dict_result_json_is_handled() -> None:
    report = no.check("Value 2.41 observed.", [2.14])
    assert not report.ok
    assert report.findings[0].result_path == "[0]"


# ---------------------------------------------------------------------------
# Flattening
# ---------------------------------------------------------------------------


def test_flatten_walks_nested_structures_with_paths() -> None:
    got = dict(no.flatten_numbers({"a": {"b": [1.5, {"c": 2.5}]}, "d": 3.5}))
    assert got == {"a.b[0]": 1.5, "a.b[1].c": 2.5, "d": 3.5}


def test_flatten_drops_booleans_and_zeros() -> None:
    got = dict(no.flatten_numbers({"flag": True, "zero": 0.0, "real": 4.2}))
    assert got == {"real": 4.2}


def test_report_serialises_for_the_audit_file() -> None:
    report = no.check("NILS of 2.41 was measured.", {"nils": 2.14})
    d = report.to_dict()
    assert d["ok"] is False
    assert d["findings"][0]["result_path"] == "nils"
    assert "message" in d["findings"][0]
