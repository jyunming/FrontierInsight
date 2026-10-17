"""A result key FI cannot fill: the plan's reference is found, the writer is told a form that works, and what is still
unfilled when no rewrite round is left is taken out of the paper, listed in QUALITY.md, and never stops the quest."""
from __future__ import annotations

import json
from pathlib import Path

from core import paper_facts as pf
from core import paper_values as pv
from core import quality_report as qr
from core import reference_values as rv

M0 = "(m - floor(pi1*m + 0.5))"


def _protocol(reference_value: str | None = "fdr_bh_ref") -> dict:
    ref = {"formula": f"{M0}/m*alpha", "kind": "value", "note": "pi0 x alpha"}
    if reference_value:
        ref["value"] = reference_value
    return {"grid": {"m": [20, 200], "pi1": [0.05, 0.2]}, "thresholds": {"alpha": 0.05},
            "metrics": [{"id": "fdr_bh", "reference": ref}]}


def _cells(extra: dict | None = None) -> list[dict]:
    out = []
    for m in (20, 200):
        for pi1 in (0.05, 0.2):
            ref = (m - int(pi1 * m + 0.5)) / m * 0.05
            values = [ref * f for f in (0.97, 1.0, 1.03, 0.99, 1.01, 1.0, 0.98, 1.02)]
            metrics = {"fdr_bh": {"values": values}}
            metrics.update(extra or {})
            out.append({"cell": {"m": m, "pi1": pi1}, "key": f"m={m},pi1={pi1}", "planned": 8, "ok": 8, "failed": 0,
                        "metrics": metrics})
    return out


# --- the root cause: a plan names its reference value by a name the trials never return -------------------------------


def test_a_reference_named_by_a_value_the_trials_do_not_return_is_for_the_metric_they_do() -> None:
    computed = rv.compute(_protocol(), [{"m": 20, "pi1": 0.05}], {"fdr_bh"})
    q = computed["quantities"][0]
    assert q["status"] == "computed" and q["value"] == "fdr_bh"
    assert computed["by_setting"][0]["reference"] == 0.95 * 0.05


def test_a_reference_for_a_value_that_is_not_returned_at_all_still_reads_no_reference() -> None:
    computed = rv.compute(_protocol(), [{"m": 20, "pi1": 0.05}], {"something_else"})
    assert computed["quantities"][0]["status"] == "no reference" and computed["by_setting"] == []


def test_a_reference_value_the_trials_do_return_is_unchanged() -> None:
    computed = rv.compute(_protocol(), [{"m": 20, "pi1": 0.05}], {"fdr_bh", "fdr_bh_ref"})
    assert computed["quantities"][0]["value"] == "fdr_bh_ref"


def test_the_key_the_writer_placed_is_filled_from_the_plan_reference() -> None:
    rec = pv.from_cells(_cells(), _protocol())
    got = pv.resolve("fdr_bh:mean @ m=each, pi1=0.2 | vs ref", rec)
    assert got.filled, got.hit and got.hit.reason
    assert got.text.startswith("consistent with the plan's reference")
    assert pv.resolve("fdr_bh:ref @ m=20, pi1=0.05", rec).text == "0.04750"


# --- the reason the writer is sent names a form that works ---------------------------------------------------------------


def test_the_reason_for_a_reference_the_plan_does_not_give_says_what_to_do_instead() -> None:
    cells = _cells()
    for c in cells:  # a value each setting sets is in the record, but it is not offered: whether it is the right one is not FI's guess
        c["metrics"]["m1"] = {"values": [c["cell"]["pi1"] * c["cell"]["m"]] * 4}
    rec = pv.from_cells(cells, {"grid": {"m": [20, 200], "pi1": [0.05, 0.2]}, "metrics": [{"id": "fdr_bh"}]})
    got = pv.resolve("fdr_bh:mean @ m=each, pi1=0.2 | vs ref", rec)
    assert not got.filled
    assert "the plan gives no reference for fdr_bh" in got.hit.reason
    assert "no key can make this comparison" in got.hit.reason and "with no number" in got.hit.reason
    assert "vs m1" not in got.hit.reason


# --- what is still unfilled when no rewrite round is left ------------------------------------------------------------------


def _mark(key: str) -> str:
    return f"<!-- fi:v {{{{v:{key}}}}} -->not recorded<!-- /fi:v -->"


def _filled(key: str, text: str) -> str:
    return f"<!-- fi:v {{{{v:{key}}}}} -->{text}<!-- /fi:v -->"


PAPER = (
    "# Results\n\n"
    "**False discovery rate.** BH sits near its reference line. For pi1=0.2, BH against pi0x0.05 across m: "
    + _mark("fdr_bh:mean @ m=each, pi1=0.2 | vs ref") + ". The headline setting is within tolerance.\n\n"
    "**Unresolved discrepancy at pi1 = 0.** Under the global null, the Bonferroni estimates against their analytic value were "
    + _mark("anyfd_bonf:mean @ m=each, pi1=0 | vs ref") + ". At m=2000 the BH estimate was "
    + _filled("anyfd_bh:mean @ m=2000, pi1=0", "0.0582") + ".\n\n"
    "Both together: " + _filled("a:mean @ m=20", "0.1") + " and " + _mark("b:mean @ m=20") + " in one sentence.\n\n"
    "| setting | value |\n|---|---|\n| m=20 | " + _mark("c:mean @ m=20") + " |\n"
)


def test_a_sentence_that_only_carries_unfilled_keys_is_dropped_and_the_rest_kept() -> None:
    text, taken = pf.take_out_unfilled(PAPER)
    assert "not recorded" not in text and "<!-- fi:v {{v:fdr_bh" not in text and "<!-- fi:v {{v:anyfd_bonf" not in text
    assert "For pi1=0.2, BH against" not in text and "Under the global null" not in text
    # the paragraph's other sentences and its bold run-in heading stay
    assert "BH sits near its reference line. The headline setting is within tolerance." in text
    assert "**Unresolved discrepancy at pi1 = 0.** At m=2000 the BH estimate was" in text
    assert "0.0582" in text


def test_a_key_among_filled_ones_or_in_a_table_is_replaced_by_the_marker_and_nothing_else_is_lost() -> None:
    text, taken = pf.take_out_unfilled(PAPER)
    assert "Both together: " in text and "[number removed] in one sentence." in text and "0.1" in text
    assert "| m=20 | [number removed] |" in text
    assert text.count("\n|") == 3  # the table keeps its three rows


def test_what_was_taken_out_is_listed_with_how() -> None:
    _text, taken = pf.take_out_unfilled(PAPER)
    by_key = {t["key"]: t for t in taken}
    assert set(by_key) == {"fdr_bh:mean @ m=each, pi1=0.2 | vs ref", "anyfd_bonf:mean @ m=each, pi1=0 | vs ref",
                           "b:mean @ m=20", "c:mean @ m=20"}
    assert by_key["fdr_bh:mean @ m=each, pi1=0.2 | vs ref"]["how"] == "sentence removed"
    assert by_key["b:mean @ m=20"]["how"] == "replaced by [number removed]"
    assert "BH against pi0x0.05" in by_key["fdr_bh:mean @ m=each, pi1=0.2 | vs ref"]["sentence"]


def test_a_paper_with_every_key_filled_is_untouched() -> None:
    paper = "A sentence. " + _filled("x:mean @ m=20", "0.3") + " and more.\n"
    text, taken = pf.take_out_unfilled(paper)
    assert text == paper and taken == []


def test_nothing_is_invented_what_remains_has_no_number_that_was_not_there() -> None:
    import re
    text, _ = pf.take_out_unfilled(PAPER)
    digits = lambda s: re.findall(r"\d+(?:\.\d+)?", pv.without_marks(s))  # noqa: E731
    assert set(digits(text)) <= set(digits(PAPER))


def test_the_quality_report_lists_what_was_taken_out(tmp_path: Path) -> None:
    _text, taken = pf.take_out_unfilled(PAPER)
    pf.keep_check(tmp_path, [], pv.without_marks(PAPER), dropped=taken)
    kept = json.loads((tmp_path / pf.RECORD).read_text(encoding="utf-8"))
    assert len(kept["taken_out_values"]) == 4
    lines = qr._paper_record_lines({"paper_check": {"taken_out": kept["taken_out_values"]}})
    joined = "\n".join(lines)
    assert "Values taken out of the paper" in joined and "sentence removed" in joined
    assert "BH against pi0x0.05" in joined
