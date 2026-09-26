"""The topic's own numbers, checked against the experiment (``core/goal_coverage.py``).

A graded run of a topic that named ``R0 in {0.9, 1.5, 3.0}`` drew its grid at {0.8, 1.2, 2.0, 3.0} and another drew five
figures for a topic that asked for three: nothing compared the experiment with the topic, and the paper described a study
nobody asked for. The comparison is advice for the human, never a must-fix hit.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from core import goal_coverage as gc
from tests.test_human_feedback_gate import _drive_node, gated_engine  # noqa: F401 — the fixture
from tests.test_page_limit import _engine, _review

SIR_TOPIC = """TOPIC: Compare a deterministic SIR epidemic model (ODE) with its stochastic counterpart.

GOALS:
1. Implement the SIR ODE and a Gillespie stochastic simulation in Python.
2. For R0 in {0.9, 1.5, 3.0} and population N in {100, 1000, 5000}, with one
   initial infective, estimate the probability of a major outbreak from 300 stochastic runs per setting.
3. Compare the final sizes with the deterministic relation and the outbreak probability with 1 - 1/R0.
5. Produce a short markdown paper (<= 4 pages, IMRAD) with three figures.
6. Cite at least three primary references, from 1927 on.
"""

FAITHFUL = "R0S = [0.9, 1.5, 3.0]\nNS = [100, 1000, 5000]\nNUM_RUNS = 300\n"
CHANGED = "R0S = [0.8, 1.2, 2.0, 3.0]\nNS = [100, 1000, 5000]\nNUM_RUNS = 300\n"


# --- what the topic asks -------------------------------------------------------------------

def test_the_settings_and_the_run_count_of_a_topic_are_the_numbers_asked_for() -> None:
    asked = {a.value: a for a in gc.asked_numbers(SIR_TOPIC)}
    assert list(asked) == [0.9, 1.5, 3.0, 100.0, 1000.0, 5000.0, 300.0]
    assert not any(a.scaled for a in asked.values())


def test_a_number_in_running_prose_is_not_a_setting() -> None:
    topic = ("High-NA EUV at 13.5 nm (~92 eV photons). Reach ROC-AUC > 0.9 on held-out data, in at most 4 pages, "
             "citing 12 references from 2018 on; the slope is -0.5. Use one seed.")
    assert gc.asked_numbers(topic) == []


def test_a_set_needs_two_numbers_and_a_small_integer_is_left_out() -> None:
    assert gc.asked_numbers("Use {25} as the size, and {1, 2, 3} as the depths.") == []
    assert [a.value for a in gc.asked_numbers("Sweep L in {16, 32, 64} and depth in {2, 3, 4}.")] == [16.0, 32.0, 64.0]


def test_a_count_of_runs_is_read_with_its_thousands_separator() -> None:
    asked = gc.asked_numbers("Average over 1,000 Monte Carlo samples and 25 independent replicates.")
    assert [a.value for a in asked] == [1000.0, 25.0]


def test_a_set_given_with_a_unit_or_a_percent_sign_may_be_held_at_another_power_of_ten() -> None:
    (a, b, c) = gc.asked_numbers("Sweep dose D in {10,20,30} mJ/cm^2.")
    assert a.scaled and b.scaled and c.scaled
    assert [(a.value, a.scaled) for a in gc.asked_numbers("Threshold levels in {5%, 10%}")] == [(10.0, True)]


# --- what the script holds --------------------------------------------------------------------

def test_the_numbers_of_a_script_are_its_literals_and_the_values_its_ranges_generate() -> None:
    source = (
        "import numpy as np\n"
        "A = [0.9, 1.5]\nB = -2.5\nC = np.linspace(0.5, 3.5, 7)\nD = np.arange(10, 60, 10)\nE = np.logspace(0, 3, 4)\n"
        "F = list(range(16, 80, 16))\nG = np.geomspace(1, 1000, 4)\n"
    )
    held = gc.code_numbers(source)
    assert {0.9, 1.5, -2.5, 0.5, 2.0, 3.5, 10.0, 50.0, 1.0, 10.0, 100.0, 1000.0, 16.0, 64.0}.issubset(held)
    assert 1.5 in held and 3.0 in held  # 3.0 from the linspace


def test_a_script_that_does_not_parse_still_gives_its_numbers() -> None:
    assert {0.9, 300.0} <= gc.code_numbers("R0 = [0.9, 1.5\nRUNS = 300 +++")


def test_the_numbers_in_the_keys_of_the_results_count() -> None:
    assert {1.5, 1000.0} <= gc.result_key_numbers({"by_R0": {"R0_1.5_N_1000": {"p": 0.3}}, "n": 5})


# --- the comparison --------------------------------------------------------------------------------

def test_a_script_that_follows_the_topic_has_no_note() -> None:
    assert gc.notes(SIR_TOPIC, [FAITHFUL], {}, 3) == []


def test_a_changed_grid_is_reported_with_the_number_and_its_words() -> None:
    notes = gc.notes(SIR_TOPIC, [CHANGED], {}, 3)
    assert len(notes) == 2
    assert notes[0].startswith("The topic gives 0.9") and "R0 in {0.9, 1.5, 3.0}" in notes[0]
    assert notes[1].startswith("The topic gives 1.5")
    assert all("neither the experiment's code nor the keys of its results" in n for n in notes)


def test_a_number_that_a_range_generates_or_that_a_result_key_names_is_not_missing() -> None:
    script = "import numpy as np\nR0S = np.linspace(0.9, 3.0, 5)\nNS = [100, 1000, 5000]\nNUM_RUNS = 300\n"
    assert [n.split(" ")[3] for n in gc.notes(SIR_TOPIC, [script], {}, 3)] == ["1.5"]
    assert gc.notes(SIR_TOPIC, [script], {"R0_1.5": 1}, 3) == []


def test_a_unit_scaled_number_matches_at_another_power_of_ten_and_a_plain_one_does_not() -> None:
    topic = "Sweep the wavelength in {193, 248} nm and use N in {500, 5000}."
    assert gc.notes(topic, ["W = [1.93e-7, 2.48e-7]\nN = [500, 5000]\n"], None, None) == []
    notes = gc.notes(topic, ["W = [193, 248]\nN = [500, 500000]\n"], None, None)
    assert [n.split(" ")[3] for n in notes] == ["5000"]  # 5000 is not 500000


@pytest.mark.parametrize("topic, expected", [
    ("with three figures", (3, "exact")),
    ("Produce 4 figures.", (4, "exact")),
    ("at least three figures", (3, "at least")),
    ("no more than 2 figures", (2, "at most")),
    ("one figure per method and a single figure of the errors", (1, "exact")),
    ("three figures and four figures", None),
    ("a set of plots", None),
])
def test_the_number_of_figures_a_topic_asks_for(topic: str, expected: Any) -> None:
    assert gc.asked_figure_count(topic) == expected


def test_a_figure_count_that_differs_from_the_topic_is_reported() -> None:
    assert gc.notes("with three figures", [], None, 5) == ["The topic asks for 3 figures; the run drew 5."]
    assert gc.notes("with three figures", [], None, 3) == []
    assert gc.notes("at least three figures", [], None, 4) == []
    assert gc.notes("at least three figures", [], None, 2) == ["The topic asks for at least 3 figures; the run drew 2."]
    assert gc.notes("at most two figures", [], None, 3) == ["The topic asks for at most 2 figures; the run drew 3."]
    # A run that drew none failed, which other checks report.
    assert gc.notes("with three figures", [], None, 0) == []


def test_at_most_eight_numbers_are_listed_and_the_rest_counted() -> None:
    topic = "Sweep x in {" + ", ".join(str(v) for v in range(11, 31)) + "}."
    notes = gc.notes(topic, ["A = 1\n"], None, None)
    assert len(notes) == 9 and notes[-1] == "12 more numbers of the topic are not in the experiment either."


# --- in the engine ------------------------------------------------------------------------------------

def _quest(tmp_path: Path, topic: str, script: str, **state: Any) -> tuple[Any, dict[str, Any]]:
    eng = _engine(tmp_path, topic=topic)
    code = tmp_path / "code"
    code.mkdir(parents=True, exist_ok=True)
    (code / "experiment.py").write_text(script, encoding="utf-8")
    return eng, {"topic": topic, "figures": ["a.png", "b.png", "c.png"], "result_json": {"p": 0.3}, **state}


def test_the_engine_reports_a_changed_grid_and_leaves_an_audit_of_what_it_checked(tmp_path: Path) -> None:
    eng, state = _quest(tmp_path, SIR_TOPIC, CHANGED)
    logged: list[str] = []
    eng._log = SimpleNamespace(  # type: ignore[assignment]
        warning=lambda m, *a: logged.append(m % a if a else m), info=lambda m, *a: logged.append(m % a if a else m),
    )
    notes = eng._goal_coverage_notes(state)  # type: ignore[arg-type]
    assert len(notes) == 2 and any("[goal_coverage] %s" == m or "[goal_coverage]" in m for m in logged)
    audit = json.loads((tmp_path / "paper" / "goal_coverage.json").read_text(encoding="utf-8"))
    assert audit["numbers_the_topic_sets"] == [0.9, 1.5, 3.0, 100.0, 1000.0, 5000.0, 300.0]
    assert audit["figures_the_topic_asks_for"] == {"count": 3, "kind": "exact"}
    assert audit["figures_drawn"] == 3 and audit["notes"] == notes


def test_a_faithful_experiment_leaves_a_clean_audit(tmp_path: Path) -> None:
    eng, state = _quest(tmp_path, SIR_TOPIC, FAITHFUL)
    assert eng._goal_coverage_notes(state) == []  # type: ignore[arg-type]
    audit = json.loads((tmp_path / "paper" / "goal_coverage.json").read_text(encoding="utf-8"))
    assert audit["notes"] == []


@pytest.mark.parametrize("extra", [
    {"no_simulation_resolved": True}, {"survey_mode_resolved": True}, {"result_json": {}},  # no experiment, or one that failed
])
def test_a_study_with_no_experiment_or_no_results_is_not_checked(tmp_path: Path, extra: dict) -> None:
    eng, state = _quest(tmp_path, SIR_TOPIC, CHANGED, **extra)
    assert eng._goal_coverage_notes(state) == []  # type: ignore[arg-type]
    assert not (tmp_path / "paper" / "goal_coverage.json").exists()


def test_a_missing_or_unreadable_script_is_skipped(tmp_path: Path) -> None:
    eng = _engine(tmp_path, topic=SIR_TOPIC)
    assert eng._goal_coverage_notes({"topic": SIR_TOPIC, "figures": []}) == []  # type: ignore[arg-type]
    assert not (tmp_path / "paper" / "goal_coverage.json").exists()


def test_a_failing_check_never_touches_the_quest(tmp_path: Path, monkeypatch) -> None:
    eng, state = _quest(tmp_path, SIR_TOPIC, CHANGED)
    monkeypatch.setattr(gc, "notes", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert eng._goal_coverage_notes(state) == []  # type: ignore[arg-type]


@pytest.mark.parametrize("panel", [False, True])
def test_the_review_records_the_notes_and_forces_nothing(tmp_path: Path, panel: bool) -> None:
    topic = "Compare two models. For R0 in {0.9, 1.5, 3.0}, from 300 runs per setting. Use three figures."
    eng = _engine(tmp_path, topic=topic, panel=panel)
    code = tmp_path / "code"
    code.mkdir(parents=True)
    (code / "experiment.py").write_text(CHANGED, encoding="utf-8")
    patch = _review(eng, tmp_path, figures=["a.png", "b.png"], result_json={"p": 0.3})
    review = patch["review"]
    assert len(review["goal_coverage_notes"]) == 3  # 0.9, 1.5 and the figure count
    assert review["must_flag_hits"] == [] and "iteration" not in patch
    assert review["verdict"] == "accept"


def test_the_human_review_snapshot_carries_the_notes(gated_engine, tmp_path: Path) -> None:  # noqa: F811, ANN001
    state = {"review": {"verdict": "accept", "goal_coverage_notes": ["The topic asks for 3 figures; the run drew 5."]},
             "iteration": 0}
    _drive_node(gated_engine, state, {"action": "accept"})
    snapshot = json.loads((gated_engine.fi_dir / "human_review.at_pause.json").read_text(encoding="utf-8"))
    assert snapshot["goal_coverage_notes"] == ["The topic asks for 3 figures; the run drew 5."]


def test_every_interface_shows_the_notes() -> None:
    repo = Path(__file__).resolve().parent.parent
    assert "goal_coverage_notes" in (repo / "web" / "static" / "quest.html").read_text(encoding="utf-8")
    assert "goal_coverage_notes" in (repo / "vscode-frontier-insight" / "src" / "bridge.ts").read_text(encoding="utf-8")
    assert "goal_coverage_notes" in (repo / "launch.py").read_text(encoding="utf-8")


def test_the_terminal_gate_prints_the_notes(monkeypatch, capsys) -> None:  # noqa: ANN001
    import asyncio

    import launch

    monkeypatch.setattr("builtins.input", lambda *_a: "accept")
    snapshot = {"verdict": "accept", "goal_coverage_notes": ["The topic gives 0.9 (\"...\"), and neither ..."]}
    asyncio.run(launch._cli_human_feedback_callback(snapshot))
    out = capsys.readouterr().out
    assert "Topic coverage (advisory, not blocking):" in out and "The topic gives 0.9" in out
