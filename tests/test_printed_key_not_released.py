"""A finding whose helper no function of a result reaches is not released when the run prints a value for the quantity it names.

"No function of a result or a figure reaches it" is a guess from the call graph; a printed key is FI's own record. The key is
left out of the results with the note; a quantity the run prints no number for is released as before. Made-up study."""
from __future__ import annotations

from pathlib import Path

from core import shown_reach
from tests.test_shown_reach import _engine, _run

UNUSED = "from model import helper_unused\n\n\ndef test_x():\n    assert helper_unused(1) == 2, 'wrong'\n"
# a helper the call graph cannot connect to a result, whose value the run prints under a key no declared result owns
FILES = {"model.py": "def helper_unused(x):\n    return x\n"}


def test_a_printed_key_no_function_reaches_is_left_out_not_released(tmp_path: Path) -> None:
    """Fails on the earlier gate: the finding was released ("changes no reported result") and the wrong value stayed."""
    engine = _engine(tmp_path)
    out, record = _run(engine, UNUSED, file="model.py", function="helper_unused", quantity="est_b", headline=("est_a",))
    assert out is None  # not a stop: the key is left out with the note
    gate = record["issues"]["F-1"]["at_gate"]
    assert gate["outcome"] == "left_out"
    assert "est_b" in gate["keys"]


def test_the_key_left_out_is_gone_from_the_results_the_paper_may_use(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    _run(engine, UNUSED, file="model.py", function="helper_unused", quantity="est_b", headline=("est_a",))
    check = (engine.quest_root / "needs" / "RESULT_CHECK.json").read_text(encoding="utf-8")
    assert "est_b" in check


def test_a_quantity_the_run_prints_no_number_for_is_still_released(tmp_path: Path) -> None:
    out, record = _run(_engine(tmp_path), UNUSED, file="model.py", function="helper_unused", quantity="cover")
    assert out is None
    assert "at_gate" not in record["issues"]["F-1"]


def test_a_helper_a_result_reaches_is_placed_on_that_result_as_before(tmp_path: Path) -> None:
    src = "from model import helper_for_cover\n\n\ndef test_x():\n    assert helper_for_cover(1) == 2, 'wrong'\n"
    out, record = _run(_engine(tmp_path), src, file="model.py", function="helper_for_cover", quantity="est_b")
    assert out is None
    assert record["issues"]["F-1"]["at_gate"]["keys"][0] == "cover"  # the result its function makes, not the printed key named


def test_the_effect_flags_a_helper_no_result_reaches(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    root = engine.quest_root
    _run(engine, UNUSED, file="model.py", function="helper_unused", quantity="cover")
    reach = engine._shown_fault_reach(["est_a"], {"design": None})
    assert isinstance(reach, shown_reach.Reach)
    eff = reach._helper_effect("model.py", "helper_unused")
    assert eff is not None and eff.unreached and not eff.left_out and root.exists()


def test_a_printed_key_that_is_the_main_result_stops_the_paper_as_a_reached_result_does(tmp_path: Path) -> None:
    out, _record = _run(_engine(tmp_path), UNUSED, file="model.py", function="helper_unused", quantity="est_a", headline=("est_a",))
    assert out is not None and out["stuck_reason"] == "shown_fault_open"
    assert "est_a" in out["rationale"]


def test_only_a_printed_key_that_is_the_quantity_itself_is_left_out(tmp_path: Path, monkeypatch) -> None:
    """A longer quantity name does not take a shorter printed key with it (`accuracy_bias` is not `accuracy`)."""
    monkeypatch.setattr("tests.test_shown_reach.RESULT", {"noise=0.1": {"accuracy": 0.9, "est_a": 2.0}})
    out, record = _run(_engine(tmp_path), UNUSED, file="model.py", function="helper_unused", quantity="accuracy_bias")
    assert out is None
    assert "at_gate" not in record["issues"]["F-1"]


def test_a_printed_nan_is_no_value_and_releases_the_finding(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("tests.test_shown_reach.RESULT", {"noise=0.1": {"est_b": float("nan"), "est_a": 2.0}})
    out, record = _run(_engine(tmp_path), UNUSED, file="model.py", function="helper_unused", quantity="est_b")
    assert out is None
    assert "at_gate" not in record["issues"]["F-1"]
