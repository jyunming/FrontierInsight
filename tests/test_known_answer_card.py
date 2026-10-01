"""The card a person reads when the known-answer checks stop a quest (core/oracle_card.py), on every surface.

Before it, a quest stopped at its checks printed no number in the terminal, and the web page and VS Code showed one
generic sentence: no source for the expected value, no case, no likely cause, and on the default two-script layout an
instruction to print an ``ORACLE_JSON`` line that does not exist there. These tests hold the card to what a person
needs to decide whether the simulation or the check is wrong, from the real cases that stopped quests: an RK4 check
whose expected value was miscalculated (1.637e-08 where the error is 3.33241e-07), a crash on a check's case, a repair
that called the check wrong, and two checks with one name. No verdict is changed by any of it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from core import oracle_card, oracle_check as oc, todo
from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig
from core.engine import Engine

ROOT = Path(__file__).resolve().parent.parent

RK4 = {
    "name": "rk4_closed_form_h01", "kind": "closed_form",
    "check": "RK4 global error at t=1 with step 0.1 against the exact solution exp(-1)",
    "expected": 1.637e-08, "tolerance": 1e-09, "case": {"dt": 0.1, "t_end": 1}, "measure": "err",
    "reference": "derivation: error = h^4/120 * max|y''''| = 1.637e-08",
}
SIMULATE = '''\
import math


def rk4(f, y, h, n):
    for _ in range(n):
        y = y + h * f(y)
    return y


def run_cell(cell):
    h = cell["dt"]
    y = rk4(lambda v: -v, 1.0, h, int(round(cell["t_end"] / h)))
    return {"err": abs(y - math.exp(-1)), "steps": cell["t_end"] / h}
'''


def _quest(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "q-1"
    (root / "code").mkdir(parents=True)
    script = root / "code" / "simulate.py"
    script.write_text(SIMULATE, encoding="utf-8")
    return root, script


def _judged(value: float | None, *, by: str = "engine", oracle: dict[str, Any] = RK4) -> list[dict[str, Any]]:
    expected, limit, mode = oc.limit_of(oracle)
    passed = None if value is None else abs(value - expected) <= limit
    return [{"name": oracle["name"], "value": value, "expected": expected, "limit": limit, "mode": mode,
             "passed_by_engine": passed, "measured_by": by}]


def _rk4_card(tmp_path: Path, **kw: Any) -> dict[str, Any]:
    root, script = _quest(tmp_path)
    reported = {"checks": [{"name": RK4["name"], "value": 3.33241e-07, "measured_by": "engine"}], "engine_measured": True}
    found = oc.problems([RK4], reported, 0)
    attempts = [{"attempt": 0, "problems": found, "checks": reported["checks"], "judged": _judged(3.33241e-07),
                 "repair": "applied", "patch_summary": "reduced the step of the integrator"},
                {"attempt": 1, "problems": found, "checks": reported["checks"], "judged": _judged(3.33241e-07)}]
    return oracle_card.build(quest_id="q-1", quest_root=root, script=script, found=found, oracles=[RK4],
                             judged=_judged(3.33241e-07), attempts=attempts, trial=True, **kw)


# --- S6: a value measured outside its tolerance (the real RK4 case) ---------------------------------------------------


def test_the_rk4_card_shows_every_number_a_person_needs_and_who_measured_it(tmp_path: Path) -> None:
    card = _rk4_card(tmp_path)
    (check,) = card["checks"]
    assert card["type"] == "known_answer_check" and check["status"] == "failed"
    assert check["name"] == RK4["check"] and check["id"] == RK4["name"]
    assert check["kind_words"] == "a special or limiting case with a known answer"
    assert check["expected_text"] == "1.637e-08" and check["reference"] == RK4["reference"]
    assert check["measured_text"] == "3.33241e-07" and check["measured_by"] == "fi"
    assert "FI, which called run_cell()" in check["measured_by_text"]
    assert check["limit_text"] == "1e-09" and check["tolerance_mode"] == "absolute"
    gap = check["gap"]
    assert gap["absolute"] == pytest.approx(3.16871e-07) and round(gap["times_tolerance"]) == 317
    assert "about 20 times the expected one" in gap["text"] and "317 times the tolerance" in gap["text"]
    assert check["case"] == {"dt": 0.1, "t_end": 1} and check["measure"] == "err"
    # Where the number is computed: the line of run_cell that returns `err`, with a short excerpt.
    where = check["where"]
    assert where["file"] == "code/simulate.py" and where["line"] == 13 and "run_cell()" in where["what"]
    assert any('"err": abs(y - math.exp(-1))' in line for line in where["excerpt"])
    assert any("reduced the step of the integrator" in t for t in card["tried"])


def test_the_terminal_prints_the_numbers_not_only_a_sentence(tmp_path: Path) -> None:
    card = _rk4_card(tmp_path)
    root, fi = tmp_path / "q-1", tmp_path / "q-1" / ".fi"
    todo.write(root, fi, "q-1", todo.pause_item("oracle", "the script has not passed its known-answer checks", [],
                                                card=card))
    text = todo.text(fi)
    for needle in ("[FI] Waiting for you: the script has not passed its known-answer checks",
                   "Expected: 1.637e-08 — from: derivation: error = h^4/120",
                   "Measured: 3.33241e-07 — by FI", "Tolerance: ±1e-09 (absolute)",
                   "about 20 times the expected one", "Case: dt=0.1, t_end=1; measure: `err`",
                   "code/simulate.py line 13", "python launch.py --resume q-1"):
        assert needle in text, needle
    markdown = (root / "NEXT_STEP.md").read_text(encoding="utf-8")
    for needle in ("## Why it stopped", "### Check “RK4 global error", "## Most likely cause", "## What FI already tried",
                   "## What you can do", "```python", '"err": abs(y - math.exp(-1))', "## Then go on"):
        assert needle in markdown, needle


def test_resume_says_it_repairs_again_and_the_change_to_the_check_never_pastes_the_measured_value(tmp_path: Path) -> None:
    card = _rk4_card(tmp_path)
    ids = [a["id"] for a in card["actions"]]
    assert ids == ["resume", "edit", "revise_check"]
    resume, edit, revise = card["actions"]
    assert "repairs the script up to 2 more time(s)" in resume["detail"] and resume["cli"] == "python launch.py --resume q-1"
    assert edit["file"] == "code/simulate.py" and edit["line"] == 13
    assert "Re-derive the expected value of the known-answer check 'rk4_closed_form_h01'" in revise["prefill"]
    assert "3.33241e-07" not in revise["prefill"], "the measured value is never offered as the new expected one"
    assert revise["cli"] == f'python launch.py --resume q-1 --revise-plan "{revise["prefill"]}"'
    assert revise["vscode"] == f"@fi /plan q-1 {revise['prefill']}"
    for hazard in ('"', "$", "`", "\\"):
        assert hazard not in revise["prefill"], hazard


def test_the_two_script_card_and_repair_request_never_mention_oracle_json(tmp_path: Path) -> None:
    card = _rk4_card(tmp_path)
    markdown = "\n".join(todo.card_lines(card)) + "\n".join(todo.card_lines(card, markdown=False))
    assert "ORACLE_JSON" not in markdown and "FI_ORACLE" not in markdown
    request = oc.directive([RK4], ["x"], trial=True)
    assert "There is no FI_ORACLE variable and no ORACLE_JSON line" in request
    assert "prints ONE line `ORACLE_JSON" not in request and "run with the environment variable FI_ORACLE=1" not in request
    assert "calls the simulation function (run_trial or run_cell" in request
    one_script = oc.directive([RK4], ["x"])
    assert "FI_ORACLE=1 to check its oracles" in one_script and "prints ONE line `ORACLE_JSON" in one_script


def test_who_measured_a_value_is_said_as_it_was(tmp_path: Path) -> None:
    by_fi = oc.problems([RK4], {"checks": [{"name": RK4["name"], "value": 3.33241e-07, "measured_by": "engine"}],
                                "engine_measured": True}, 0)[0]
    by_script = oc.problems([RK4], {"checks": [{"name": RK4["name"], "value": 3.33241e-07}]}, 0)[0]
    assert "failed: FI measured 3.33241e-07, the protocol expects 1.637e-08" in by_fi
    assert "failed: the script measured 3.33241e-07" in by_script
    # A script that only CLAIMS FI measured it is still the script's number.
    claimed = oc.problems([RK4], {"checks": [{"name": RK4["name"], "value": 3.33241e-07, "measured_by": "engine"}]}, 0)[0]
    assert "the script measured" in claimed


def test_enough_digits_are_shown_to_see_a_gap_the_tolerance_cares_about() -> None:
    oracle = {"name": "y1", "expected": 0.3678794641, "tolerance": 1e-12}
    found = oc.problems([oracle], {"checks": [{"name": "y1", "value": 0.36787977}]}, 0)[0]
    # At six digits both read 0.367879/0.36788: they look the same, and the gap is 3e-7 against a tolerance of 1e-12.
    assert "measured 0.36787977, the protocol expects 0.3678794641" in found
    shown, expected = oc.fmt_pair(0.36787977, 0.3678794641, 1e-12)
    assert shown != expected and len(expected.replace("0.", "")) >= 10
    # A plain gap keeps the short form a person is used to.
    assert oc.fmt_pair(0.5, 1.0, 0.05) == ("0.5", "1")


# --- S5: nothing measured ------------------------------------------------------------------------------------------------


def test_a_crash_on_the_checks_case_shows_the_error_and_where_and_offers_no_change_to_the_check(tmp_path: Path) -> None:
    root, script = _quest(tmp_path)
    oracle = {"name": "r0_threshold", "kind": "special_case", "check": "SIR at R0=1 does not grow", "expected": 0.0,
              "tolerance": 1e-3, "case": {"beta": 0.2, "gamma": 0.2}, "measure": "growth", "reference": "derivation: R0 = beta/gamma = 1"}
    problem = ("the oracle 'r0_threshold': the simulation could not be run on its case (KeyError: 'FI_RAW_DIR' "
               "(at code/simulate.py line 12))")
    attempts = [{"attempt": i, "problems": [problem], "checks": [], "judged": _judged(None, oracle=oracle),
                 **({"repair": "applied", "patch_summary": "rewrote the oracle branch"} if i < 2 else {})}
                for i in range(3)]
    card = oracle_card.build(quest_id="q-1", quest_root=root, script=script, found=[problem], oracles=[oracle],
                             judged=_judged(None, oracle=oracle), attempts=attempts, trial=True)
    (check,) = card["checks"]
    assert check["status"] == "not_measured" and check["error"] == "KeyError: 'FI_RAW_DIR'"
    assert check["error_at"] == "code/simulate.py line 12"
    assert "could not be measured" in card["summary"]
    cause = card["causes"][0]
    assert "not the expected value" in cause["text"]
    assert "KeyError: 'FI_RAW_DIR' (at code/simulate.py line 12)" in cause["evidence"]
    assert "the same error came back after each of FI's 2 repair(s)" in cause["evidence"]
    assert [a["id"] for a in card["actions"]] == ["resume", "edit"], "nothing was measured: the check is not the suspect"
    assert card["actions"][1]["file"] == "code/simulate.py" and card["actions"][1]["line"] == 12


def test_a_one_script_crash_reads_the_last_exception_of_the_traceback(tmp_path: Path) -> None:
    root, script = _quest(tmp_path)
    oracle = {"name": "final size", "expected": 1.0, "tolerance": 0.05}
    found = oc.problems([oracle], None, 1)
    stderr = ('Traceback (most recent call last):\n  File "' + str(root / "code" / "experiment.py") + '", line 7, in '
              "<module>\n    import simpy\nModuleNotFoundError: No module named 'simpy'\n")
    card = oracle_card.build(quest_id="q-1", quest_root=root, script=root / "code" / "experiment.py", found=found,
                             oracles=[oracle], judged=[], attempts=[{"problems": found}], stderr_tail=stderr)
    assert card["checks"][0]["status"] == "not_measured"
    evidence = " ".join(c["evidence"] for c in card["causes"])
    assert "ModuleNotFoundError: No module named 'simpy'" in evidence and "code/experiment.py line 7" in evidence


# --- S7: the repair says the check itself is wrong -------------------------------------------------------------------------


def test_a_disputed_check_shows_the_proposal_and_offers_it_as_one_step(tmp_path: Path) -> None:
    proposal = {"name": RK4["name"], "expected": 3.33241e-07, "tolerance": 1e-09, "tolerance_mode": "absolute",
                "reason": "the global RK4 error at h=0.1 is 3.33e-07; 1.637e-08 is the local error of one step"}
    card = _rk4_card(tmp_path, proposals=[proposal], disputed=[RK4["name"]], kept="as_it_was")
    (check,) = card["checks"]
    assert check["disputed"] and check["proposal"]["expected"] == 3.33241e-07
    assert "FI's repair says the check itself is wrong" in card["summary"]
    cause = card["causes"][0]
    assert "judged the check `rk4_closed_form_h01` itself wrong" in cause["text"]
    assert "accepting this makes that measurement pass, so check the reason, not the result" in cause["evidence"]
    assert "the local error of one step" in cause["evidence"]
    accept = next(a for a in card["actions"] if a["id"] == "accept_proposal")
    assert accept["prefill"] == oc.proposal_request(proposal)
    assert "not carried over" in card["actions"][0]["detail"], "a resume does not remember the dispute (yet)"
    assert any("kept as it was" in t for t in card["tried"])


# --- D13: two checks under one name ------------------------------------------------------------------------------------------


def test_two_checks_with_one_name_are_named_on_the_card_without_changing_the_verdict(tmp_path: Path) -> None:
    root, script = _quest(tmp_path)
    euler = {"name": "order", "kind": "convergence_rate", "check": "Euler order", "expected": 1.0, "tolerance": 0.1}
    rk = {"name": "Order", "kind": "convergence_rate", "check": "RK4 order", "expected": 4.0, "tolerance": 0.1}
    reported = {"checks": [{"name": "order", "value": 1.0}, {"name": "order", "value": 4.01}]}
    assert len(oc.duplicate_names([euler, rk], reported)) == 2
    found = oc.problems([euler, rk], reported, 0)
    card = oracle_card.build(quest_id="q-1", quest_root=root, script=script, found=found, oracles=[euler, rk],
                             judged=oc.judged([euler, rk], reported), attempts=[{"problems": found, "checks": reported["checks"]}])
    texts = " ".join(c["evidence"] for c in card["causes"])
    assert "The plan declares 2 known-answer checks named 'order'" in texts
    assert "The script reported 2 values named 'order'" in texts
    assert oc.duplicate_names([RK4], {"checks": [{"name": RK4["name"], "value": 1}]}) == []


# --- after the freeze ------------------------------------------------------------------------------------------------------


def test_after_the_freeze_an_interview_quest_is_told_to_approve_the_setting_with_update(tmp_path: Path) -> None:
    made = _rk4_card(tmp_path, frozen=True, interview_made=True)
    go_on = next(a for a in made["actions"] if a["id"] == "go_on_recorded")
    assert "`engine.oracle_check: warn`" in go_on["detail"] and "python launch.py --update q-1" in go_on["detail"]
    assert go_on["cli"] == "python launch.py --update q-1" and go_on["vscode"] == "@fi /update q-1"
    assert not any(a["id"] == "revise_check" for a in made["actions"]), "a frozen protocol is not changed from here"
    by_hand = _rk4_card(tmp_path / "b", frozen=True)
    assert "--update" not in next(a for a in by_hand["actions"] if a["id"] == "go_on_recorded")["detail"]
    research = _rk4_card(tmp_path / "c", frozen=True, research=True)
    text = json.dumps(research)
    assert "oracle_check: warn" not in text and "start a new quest" in text


# --- the stop itself: pause.json carries the card for the web page and VS Code ----------------------------------------------


def _cfg(tmp_path: Path) -> Config:
    return Config(
        topic="the known-answer card", title="card", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, pilot_run=False),
        execution=ExecutionConfig(sandbox="venv", timeout_s=120, split_analysis=False),
        knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "outputs"),
    )


def _paused(tmp_path: Path) -> Engine:
    engine = Engine(_cfg(tmp_path))
    (engine.quest_root / "code").mkdir(parents=True, exist_ok=True)
    script = engine.quest_root / "code" / "simulate.py"
    script.write_text(SIMULATE, encoding="utf-8")
    engine._trial_mode = True
    reported = {"checks": [{"name": RK4["name"], "value": 3.33241e-07, "measured_by": "engine"}], "engine_measured": True}
    found = oc.problems([RK4], reported, 0)
    with pytest.raises(BaseException):  # the pause interrupts; outside a graph that raises
        engine._pause_for_oracle(found, script, [], _judged(3.33241e-07), [RK4],
                                 attempts=[{"problems": found, "checks": reported["checks"], "judged": _judged(3.33241e-07)}])
    return engine


def test_the_stop_writes_the_card_into_pause_json_next_step_and_the_todo(tmp_path: Path) -> None:
    engine = _paused(tmp_path)
    pause = json.loads((engine.fi_dir / "pause.json").read_text(encoding="utf-8"))
    assert pause["kind"] == "oracle" and pause["headline"] == "the script has not passed its known-answer checks"
    assert pause["card"]["checks"][0]["measured_text"] == "3.33241e-07"
    assert pause["problems"] and "FI measured 3.33241e-07" in pause["problems"][0]
    markdown = (engine.quest_root / "NEXT_STEP.md").read_text(encoding="utf-8")
    assert "about 20 times the expected one" in markdown and "ORACLE_JSON" not in markdown
    assert todo.read(engine.fi_dir)[0]["card"]["type"] == "known_answer_check"


def test_the_web_page_gets_the_card_and_offers_its_actions(tmp_path: Path) -> None:
    try:
        from fastapi.testclient import TestClient
    except Exception:  # pragma: no cover
        pytest.skip("fastapi not installed")
    from web.server import make_app

    engine = _paused(tmp_path)
    client = TestClient(make_app(tmp_path / "outputs"))
    data = client.get(f"/api/quests/{engine.quest_id}/next-step").json()
    assert data["waiting"] and data["kind"] == "oracle"
    assert [a["id"] for a in data["card"]["actions"]] == ["resume", "edit", "revise_check"]
    assert "about 20 times the expected one" in data["markdown"], "the banner text is the same card"
    page = (ROOT / "web" / "static" / "quest.html").read_text(encoding="utf-8")
    assert "renderCardActions(data.card)" in page and 'id="next-step-card-actions"' in page
    assert "fillPlanRequest(a.prefill)" in page and "oracle: 'known-answer check'" in page


def test_vs_code_turns_the_same_card_into_chat_buttons(tmp_path: Path) -> None:
    ext = ROOT / "vscode-frontier-insight"
    compiled, source = ext / "out" / "stop-card.js", ext / "src" / "stop-card.ts"
    node = shutil.which("node")
    if node is None or not compiled.is_file() or compiled.stat().st_mtime < source.stat().st_mtime:
        pytest.skip("node, or a compiled vscode-frontier-insight/out/stop-card.js as new as its source, is not here "
                    "(npm run compile)")
    card = _rk4_card(tmp_path, proposals=[{"name": RK4["name"], "expected": 3.33241e-07, "tolerance": 1e-09,
                                           "tolerance_mode": "absolute", "reason": "the global error"}])
    pause = {"kind": "oracle", "card": {**card, "actions": [*card["actions"], {"id": "something_new", "label": "x"}]}}
    script = ("const m = require(process.argv[1]); const p = JSON.parse(require('fs').readFileSync(0, 'utf8'));"
              "console.log(JSON.stringify([m.cardButtons(m.cardOf(p), 'q-1'), m.cardOf({}), m.cardButtons(m.cardOf(p), 'q 1; x')]));")
    done = subprocess.run([node, "-e", script, str(compiled)], input=json.dumps(pause), capture_output=True,
                          text=True, encoding="utf-8", timeout=60)
    assert done.returncode == 0, done.stderr
    buttons, none, refused = json.loads(done.stdout)
    assert [b["title"] for b in buttons] == ["Let FI try again", "Accept the repair's proposed change"]
    assert buttons[0] == {"title": "Let FI try again", "query": "@fi /resume q-1", "send": True,
                          "tooltip": card["actions"][0]["detail"]}
    assert buttons[1]["query"] == f"@fi /plan q-1 {oc.proposal_request(card['checks'][0]['proposal'])}"
    assert buttons[1]["send"] is False, "a change to a check is put in the box to read, never sent at once"
    assert none is None and refused == []
    extension = (ext / "src" / "extension.ts").read_text(encoding="utf-8")
    assert "await offerCardButtons(outputsDir, card.questId, stream)" in extension
    assert "cardButtons(cardOf(JSON.parse(raw)), questId)" in extension and "stream.button(" in extension
