"""Warnings from a run's own numerics fail closed (core/numeric_warnings.py and the gate in execute_reflect)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from core import numeric_warnings as nw
from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig,
)
from core.engine import Engine
from tests.test_engine_smoke import _FAKE_RESPONSES, _classify, _fake_response_for

STDERR = """C:/x/experiment.py:26: UserWarning: The following arguments have no effect for a chosen solver: `abs_tol`, `rel_tol`.
  sol = integrate.solve_ivp(
C:/x/experiment.py:60: RuntimeWarning: overflow encountered in exp
C:/x/experiment.py:61: RuntimeWarning: invalid value encountered in divide
C:/x/experiment.py:70: DeprecationWarning: something old
C:/x/experiment.py:3: UserWarning: This figure includes Axes that are not compatible with tight_layout, so results might be incorrect.
C:/x/experiment.py:9: IntegrationWarning: The maximum number of subdivisions (50) has been achieved.
C:/x/experiment.py:9: RuntimeWarning: Mean of empty slice
C:/x/experiment.py:9: FutureWarning: a future thing
"""


# --- reading a run -----------------------------------------------------------------------------------------------------


def test_the_warnings_that_mean_the_numbers_may_be_wrong_are_named() -> None:
    found = nw.scan(STDERR)
    assert [(w.kind, w.text) for w in found] == [
        ("ignored_argument", "The following arguments have no effect for a chosen solver: `abs_tol`, `rel_tol`."),
        ("runtime", "overflow encountered in exp"),
        ("runtime", "invalid value encountered in divide"),
        ("solver", "IntegrationWarning: The maximum number of subdivisions (50) has been achieved."),
    ]
    assert "an argument had no effect" in found[0].describe() and "abs_tol" in found[0].describe()


@pytest.mark.parametrize("line", [
    "f.py:1: DeprecationWarning: x", "f.py:1: FutureWarning: x", "f.py:1: ResourceWarning: unclosed file",
    "f.py:1: UserWarning: Glyph 8722 missing from current font.", "f.py:1: UserWarning: No artists with labels found to put in legend.",
    "f.py:1: RuntimeWarning: Mean of empty slice", "not a warning line", "",
])
def test_warnings_about_fonts_deprecations_and_the_like_do_not_count(line: str) -> None:
    assert nw.scan(line) == []


@pytest.mark.parametrize("line, kind", [
    ("f.py:1: RuntimeWarning: divide by zero encountered in log", "runtime"),
    ("f.py:1: OptimizeWarning: Covariance of the parameters could not be estimated", "solver"),
    ("f.py:1: ConvergenceWarning: Solver did not converge", "solver"),
    ("f.py:1: LinAlgWarning: Ill-conditioned matrix", "solver"),
    ("f.py:1: RuntimeWarning: The iteration is not making good progress", "solver"),
    ("f.py:1: ComplexWarning: Casting complex values to real discards the imaginary part", "complex"),
    ("f.py:1: UserWarning: keyword `foo` will be ignored", "ignored_argument"),
])
def test_each_class_of_numeric_warning_is_recognised(line: str, kind: str) -> None:
    assert [w.kind for w in nw.scan(line)] == [kind]


def test_a_nan_or_an_infinity_in_a_result_is_a_warning_and_a_finite_one_is_not() -> None:
    found = nw.scan("", {"a": {"b": math.nan}, "c": [1.0, math.inf], "d": 0.5, "e": None, "f": "nan"})
    assert [(w.kind, w.text) for w in found] == [("non_finite", "a.b is nan"), ("non_finite", "c[1] is inf")]
    assert nw.scan("", {"a": 1.0, "b": [0.1, 0.2]}) == []


def test_a_warning_repeated_on_every_step_is_named_once_and_the_list_is_bounded() -> None:
    line = "f.py:1: RuntimeWarning: overflow encountered in exp\n"
    assert len(nw.scan(line * 500)) == 1
    many = "".join(f"f.py:1: RuntimeWarning: invalid value encountered in op{i}\n" for i in range(50))
    assert len(nw.scan(many)) == 8


def test_the_repair_request_forbids_hiding_the_warning_and_names_the_scipy_arguments() -> None:
    text = nw.directive(nw.scan(STDERR))
    for needle in ("atol", "rtol", "Do NOT silence the warning", "np.errstate", "patch_summary", "no effect"):
        assert needle in text, needle


# --- the gate, through the real graph and a real checkpoint --------------------------------------------------------------

_BODY = """\
import os, json, warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
"""
_TAIL = """\
os.makedirs('figures', exist_ok=True)
plt.figure(); plt.plot(list(range(3)), [i * i for i in range(3)]); plt.savefig('figures/result.png', dpi=72)
print('RESULT_JSON: {"score": 0.987}')
"""
_WARNS = _BODY + "warnings.warn('The following arguments have no effect for a chosen solver: abs_tol', UserWarning)\n" + _TAIL
_QUIET = _BODY + _TAIL
_DEPRECATED = _BODY + "warnings.warn('an old thing', DeprecationWarning)\n" + _TAIL


def _cfg(tmp_path: Path, **engine: Any) -> Config:
    return Config(
        topic="smoke topic for the numeric gate", title="numeric-smoke", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, auto_accept_on_pass=True, execute_replicates=1,
                            pilot_run=False, **{"exec_reflect_max_iterations": 2, **engine}),
        execution=ExecutionConfig(sandbox="venv", timeout_s=120, split_analysis=False),
        knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "outputs"),
    )


def _fake(calls: list[str], *, implement: str, repair: str):
    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        kind = _classify(prompt)
        calls.append(kind)
        if kind == "Implementation":
            return json.dumps({"code": implement, "deps": ["matplotlib"]})
        if kind == "ExecuteReflect" and "its numerics reported problems" in prompt:
            calls.append("NumericRepair")
            return json.dumps({"code": repair, "deps": [], "patch_summary": "used atol and rtol"})
        return _fake_response_for(prompt)

    return fake_chat


def _recorded(engine: Engine) -> list[dict[str, Any]]:
    return json.loads((engine.quest_root / "needs" / "NUMERIC_WARNINGS.json").read_text(encoding="utf-8"))


@pytest.mark.asyncio
async def test_a_run_that_warned_is_sent_back_like_a_failed_one_and_the_repair_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake(calls, implement=_WARNS, repair=_QUIET))
    engine = Engine(_cfg(tmp_path))
    artifacts = await engine.run()
    assert artifacts.paper_md is not None and calls.count("NumericRepair") == 1
    assert "warnings.warn" not in (engine.quest_root / "code" / "experiment.py").read_text(encoding="utf-8")
    record = _recorded(engine)
    assert len(record) == 1 and record[0]["warnings"][0]["kind"] == "ignored_argument"


@pytest.mark.asyncio
async def test_warnings_the_repairs_do_not_remove_stop_the_quest_and_a_fixed_script_is_run_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake(calls, implement=_WARNS, repair=_WARNS))
    cfg = _cfg(tmp_path)
    first = Engine(cfg)
    await first.run()

    log = (first.quest_root / ".fi" / "run.log").read_text(encoding="utf-8")
    assert calls.count("NumericRepair") == 2, "the repair budget (exec_reflect_max_iterations) was spent"
    assert "[numeric] paused" not in log and "paused for the run's numeric warnings" not in log
    assert not (first.fi_dir / "pause.json").exists(), "nobody is asked to fix the script"
    record = json.loads((first.quest_root / "needs" / "STUCK.json").read_text(encoding="utf-8"))
    assert "an argument had no effect" in record["problem"] and "did not remove it" in record["problem"]
    assert any("sent the script back to be repaired" in t for t in record["tried"])
    assert not (first.quest_root / "paper" / "paper.md").exists()

    # Run again from the run step with a script that no longer warns: the quest goes on.
    (first.quest_root / "code" / "experiment.py").write_text(_QUIET, encoding="utf-8")
    second = Engine(cfg, resume_quest_id=first.quest_id)
    artifacts = await second.run(from_step="run")
    assert artifacts.paper_md is not None and artifacts.paper_md.exists()


@pytest.mark.asyncio
async def test_running_again_without_changing_the_script_ends_in_the_same_stop_not_in_accepted_warnings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake(calls, implement=_WARNS, repair=_WARNS))
    cfg = _cfg(tmp_path)
    first = Engine(cfg)
    await first.run()

    second = Engine(cfg, resume_quest_id=first.quest_id)
    artifacts = await second.run(from_step="run")

    assert artifacts.paper_md is None, "warnings that stand are never accepted for a person"
    assert (second.quest_root / "needs" / "STUCK.json").is_file()
    assert _recorded(second), "the warnings stay on record"


@pytest.mark.asyncio
async def test_warn_records_the_warnings_and_goes_on_without_a_repair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake(calls, implement=_WARNS, repair=_QUIET))
    engine = Engine(_cfg(tmp_path, numeric_warnings="warn"))
    artifacts = await engine.run()
    assert artifacts.paper_md is not None and "NumericRepair" not in calls
    assert _recorded(engine)[0]["mode"] == "warn"


@pytest.mark.asyncio
async def test_off_does_not_look_and_a_harmless_warning_is_not_a_problem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name, extra, code in (("off", {"numeric_warnings": "off"}, _WARNS), ("harmless", {}, _DEPRECATED)):
        calls: list[str] = []
        monkeypatch.setattr("core.engine.LLMClient.chat", _fake(calls, implement=code, repair=_QUIET))
        engine = Engine(_cfg(tmp_path / name, **extra))
        artifacts = await engine.run()
        assert artifacts.paper_md is not None and "NumericRepair" not in calls
        assert not (engine.quest_root / "needs" / "NUMERIC_WARNINGS.json").exists()


def test_the_dashboard_and_the_quest_page_name_the_numeric_stop() -> None:
    static = Path(__file__).resolve().parent.parent / "web" / "static"
    for page in ("index.html", "quest.html"):
        assert "numeric: 'numeric'" in (static / page).read_text(encoding="utf-8"), page


@pytest.mark.asyncio
async def test_a_scanner_that_crashes_is_recorded_as_unknown_not_as_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real bypass (the re-audit's P1-1): before this fix, a scan that raised returned `[]` exactly like a scan
    that ran and found nothing, and nothing was ever written to disk to tell the two apart."""
    calls: list[str] = []
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake(calls, implement=_QUIET, repair=_QUIET))
    monkeypatch.setattr("core.numeric_warnings.scan", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    engine = Engine(_cfg(tmp_path))
    artifacts = await engine.run()
    assert artifacts.paper_md is not None

    record = _recorded(engine)
    assert record[-1] == {"error": "RuntimeError('boom')", "mode": "block"}

    from core import evidence

    got = evidence.assess(engine.quest_root, artifacts.raw_state, settings={"numeric_warnings": "block"})
    assert not got["levels"]["protocol_runtime_matched"]
    assert any("scanner failed to run" in g for g in got["all_gaps"]["protocol_runtime_matched"])


def test_a_crash_followed_by_a_clean_scan_clears_the_gap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A scanner crash on one run must not read as "still unknown" forever: a repair that made the scan itself
    run clean again (as opposed to fixing the warnings it found) has to say so, or the stale crash entry would
    outlive it and permanently gap a quest whose numerics were, in the end, actually checked."""
    from core import evidence

    engine = Engine(_cfg(tmp_path))
    monkeypatch.setattr("core.numeric_warnings.scan", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert engine._scan_numeric_warnings({}, "", None) == []
    assert _recorded(engine)[-1] == {"error": "RuntimeError('boom')", "mode": "block"}
    got = evidence.assess(engine.quest_root, {}, settings={"numeric_warnings": "block"})
    assert any("scanner failed to run" in g for g in got["all_gaps"]["protocol_runtime_matched"])

    monkeypatch.setattr("core.numeric_warnings.scan", lambda *a, **k: [])
    assert engine._scan_numeric_warnings({}, "", None) == []
    record = _recorded(engine)
    assert record[-1] == {"warnings": [], "mode": "block"}, "the crash must not stay the last word once a scan actually ran clean"
    assert record[0] == {"error": "RuntimeError('boom')", "mode": "block"}, "the crash itself stays on record"
    got = evidence.assess(engine.quest_root, {}, settings={"numeric_warnings": "block"})
    assert not any("scanner failed to run" in g for g in got["all_gaps"]["protocol_runtime_matched"])


def test_a_clean_scan_after_warnings_were_found_does_not_add_a_synthetic_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A clean scan following an earlier crash is news (above); a clean scan following earlier WARNINGS is not —
    the repair-loop tests elsewhere read the recorded history as exactly what was found, with no "and now it's
    fine" entries added on top for every successful repair."""
    engine = Engine(_cfg(tmp_path))
    monkeypatch.setattr("core.numeric_warnings.scan", lambda *a, **k: [nw.NumericWarning("ignored_argument", "abs_tol")])
    assert len(engine._scan_numeric_warnings({}, "", None)) == 1
    assert len(_recorded(engine)) == 1

    monkeypatch.setattr("core.numeric_warnings.scan", lambda *a, **k: [])
    assert engine._scan_numeric_warnings({}, "", None) == []
    assert len(_recorded(engine)) == 1, "a clean scan after warnings (not a crash) adds nothing"


@pytest.mark.asyncio
async def test_the_stop_offers_the_setting_and_running_again_with_it_goes_on_with_the_warnings_noted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[str] = []
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake(calls, implement=_WARNS, repair=_WARNS))
    first = Engine(_cfg(tmp_path))
    await first.run()
    record = json.loads((first.quest_root / "needs" / "STUCK.json").read_text(encoding="utf-8"))
    assert "engine.numeric_warnings: warn" in record["choice"] and "--from run" in record["choice"]
    assert "engine.numeric_warnings: warn" in capsys.readouterr().out
    assert not (first.fi_dir / "pause.json").exists()

    second = Engine(_cfg(tmp_path, numeric_warnings="warn"), resume_quest_id=first.quest_id)
    artifacts = await second.run(from_step="run")
    assert artifacts.paper_md is not None and artifacts.paper_md.exists()
    assert _recorded(second)[-1]["mode"] == "warn", "the warnings are on record"


# --- the warnings' own repair budget: repairs spent on crashes never leave them with none ------------------------------

_CRASHES = _BODY + "raise RuntimeError('the first version crashes')\n"


def _fake_crash_then(calls: list[str], *, after_crash: str, numeric_repair: str):
    """The first script crashes; its crash repair gives ``after_crash``; a repair for the warnings gives ``numeric_repair``."""
    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        kind = _classify(prompt)
        calls.append(kind)
        if kind == "Implementation":
            return json.dumps({"code": _CRASHES, "deps": ["matplotlib"]})
        if kind == "ExecuteReflect" and "its numerics reported problems" in prompt:
            calls.append("NumericRepair")
            return json.dumps({"code": numeric_repair, "deps": [], "patch_summary": "used atol and rtol"})
        if kind == "ExecuteReflect":
            calls.append("CrashRepair")
            return json.dumps({"code": after_crash, "deps": [], "patch_summary": "removed the raise"})
        return _fake_response_for(prompt)

    return fake_chat


@pytest.mark.asyncio
async def test_warnings_after_the_crash_repairs_are_spent_get_a_repair_of_their_own_and_the_quest_goes_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The recorded case: a crash repair used the last repair, the repaired run then warned, and the quest stopped saying
    the repairs did not remove the warnings, though none was made for them."""
    calls: list[str] = []
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake_crash_then(calls, after_crash=_WARNS, numeric_repair=_QUIET))
    engine = Engine(_cfg(tmp_path, exec_reflect_max_iterations=1))
    artifacts = await engine.run()
    assert calls.count("CrashRepair") == 1 and calls.count("NumericRepair") == 1
    assert not (engine.quest_root / "needs" / "STUCK.json").exists()
    assert artifacts.paper_md is not None and artifacts.paper_md.exists()


@pytest.mark.asyncio
async def test_a_stop_for_warnings_counts_only_the_repairs_made_for_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake_crash_then(calls, after_crash=_WARNS, numeric_repair=_WARNS))
    engine = Engine(_cfg(tmp_path, exec_reflect_max_iterations=1))
    await engine.run()
    assert calls.count("CrashRepair") == 1 and calls.count("NumericRepair") == 1
    record = json.loads((engine.quest_root / "needs" / "STUCK.json").read_text(encoding="utf-8"))
    assert "an argument had no effect" in record["problem"]
    assert any("repaired 1 time(s) for these warnings; they stood" in t for t in record["tried"]), record["tried"]
    assert not (engine.quest_root / "paper" / "paper.md").exists()


@pytest.mark.asyncio
async def test_a_refused_repair_for_the_warnings_after_the_crash_repairs_are_spent_never_goes_on_to_the_analysis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The repair for the warnings is not Python, so it is not used: the run is not taken on to a paper with its warnings
    standing because the crash repairs are spent; the warnings' own budget ends in the stop."""
    calls: list[str] = []
    monkeypatch.setattr("core.engine.LLMClient.chat",
                        _fake_crash_then(calls, after_crash=_WARNS, numeric_repair="def broken(:\n"))
    engine = Engine(_cfg(tmp_path, exec_reflect_max_iterations=1))
    artifacts = await engine.run()
    assert calls.count("CrashRepair") == 1 and calls.count("NumericRepair") == 1
    assert artifacts.paper_md is None and (engine.quest_root / "needs" / "STUCK.json").is_file()


def test_every_repair_made_is_counted_whichever_budget_it_is_on() -> None:
    history = [{"iter": 1, "returncode": 1}, {"for": nw.REPAIR_FOR, "iter": 2, "returncode": 0},
               {"for": nw.REPAIR_FOR, "iter": 3, "returncode": 0}]
    assert nw.repairs_for_warnings(history) == 2
    assert nw.repairs_made({"exec_reflect_iter": 1, "exec_reflect_history": history}) == 3
    assert nw.repairs_made({}) == 0 and nw.repairs_for_warnings(None) == 0



# ---- a NaN in a number the plan does not report is disclosed, not repaired for (-172 part b, option B) ----------------

_PROTOCOL = {"metrics": [{"id": "rate"}, {"id": "spread"}]}


def test_a_nan_in_a_declared_result_counts_and_one_in_a_diagnostic_does_not() -> None:
    result = {"by_setting": {"low": {"rate": math.nan, "spread": 0.2}}, "debug_ratio": math.nan, "timing": [math.inf]}
    found = nw.scan("", result, results=nw.declared_results(_PROTOCOL))
    assert [(w.kind, w.text) for w in found] == [("non_finite", "by_setting.low.rate is nan")]
    assert nw.disclosed_non_finite(result, nw.declared_results(_PROTOCOL)) == ["debug_ratio is nan", "timing[0] is inf"]


def test_a_plan_that_declares_no_result_keeps_every_non_finite_number() -> None:
    result = {"debug_ratio": math.nan}
    assert [w.text for w in nw.scan("", result, results=nw.declared_results({}))] == ["debug_ratio is nan"]
    assert nw.disclosed_non_finite(result, nw.declared_results({})) == []


def test_a_nan_only_in_a_diagnostic_asks_for_no_repair_and_is_disclosed_in_the_evidence(tmp_path: Path) -> None:
    from core import evidence

    engine = Engine(_cfg(tmp_path))
    state: dict[str, Any] = {"design": {"protocol": _PROTOCOL}}
    assert engine._scan_numeric_warnings(state, "", {"rate": 0.5, "debug_ratio": math.nan}) == []  # type: ignore[arg-type]
    assert _recorded(engine)[-1]["disclosed"] == ["debug_ratio is nan"]
    got = evidence.assess(engine.quest_root, {}, settings={"numeric_warnings": "block"})
    assert any("debug_ratio is nan" in g and "not stored under the name of one of the plan's results" in g
               for g in got["all_gaps"]["protocol_runtime_matched"])
    # The next run prints it finite: the disclosure is no longer the last word.
    assert engine._scan_numeric_warnings(state, "", {"rate": 0.5, "debug_ratio": 1.0}) == []  # type: ignore[arg-type]
    assert _recorded(engine)[-1] == {"warnings": [], "mode": "block"}
    got = evidence.assess(engine.quest_root, {}, settings={"numeric_warnings": "block"})
    assert not any("debug_ratio" in g for g in got["all_gaps"]["protocol_runtime_matched"])


def test_a_nan_in_a_declared_result_still_asks_for_the_repair(tmp_path: Path) -> None:
    engine = Engine(_cfg(tmp_path))
    state: dict[str, Any] = {"design": {"protocol": _PROTOCOL}}
    got = engine._scan_numeric_warnings(state, "", {"rate": math.nan, "debug_ratio": math.nan})  # type: ignore[arg-type]
    assert got == [{"kind": "non_finite", "text": "rate is nan"}]
    assert _recorded(engine)[-1]["disclosed"] == ["debug_ratio is nan"]


def test_a_key_is_matched_whole_even_when_it_holds_a_dot_or_a_bracket() -> None:
    results = nw.declared_results({"metrics": [{"id": "a.b"}, {"id": "rate"}]})
    found = nw.scan("", {"a.b": math.nan, "rate[nm]": math.nan, "x": {"rate": [1.0, math.inf]}}, results=results)
    assert [w.text for w in found] == ["a.b is nan", "x.rate[1] is inf"]
    assert nw.disclosed_non_finite({"rate[nm]": math.nan}, results) == ["rate[nm] is nan"]


def test_a_top_level_list_of_results_is_read_by_its_keys() -> None:
    found = nw.scan("", [{"rate": math.nan}, {"other": math.nan}], results={"rate"})
    assert [w.text for w in found] == ["[0].rate is nan"]


def test_a_clean_scan_after_a_warning_with_a_disclosure_ends_the_disclosure(tmp_path: Path) -> None:
    engine = Engine(_cfg(tmp_path))
    state: dict[str, Any] = {"design": {"protocol": _PROTOCOL}}
    assert engine._scan_numeric_warnings(state, "", {"rate": math.nan, "debug_ratio": math.nan})  # type: ignore[arg-type]
    assert engine._scan_numeric_warnings(state, "", {"rate": 0.5, "debug_ratio": 1.0}) == []  # type: ignore[arg-type]
    record = _recorded(engine)
    assert record[0]["warnings"] and record[0]["disclosed"] == ["debug_ratio is nan"]
    assert record[-1] == {"warnings": [], "mode": "block"}


def test_the_frozen_protocol_names_the_results_when_there_is_one(tmp_path: Path) -> None:
    from core import frozen_protocol

    engine = Engine(_cfg(tmp_path))
    frozen_path = frozen_protocol.frozen_path(engine.quest_root)
    frozen_path.parent.mkdir(parents=True, exist_ok=True)
    protocol = {"metrics": [{"id": "debug_ratio"}]}
    frozen_path.write_text(json.dumps({"schema": frozen_protocol.FROZEN_SCHEMA, "protocol": protocol,
                                       "sha256": frozen_protocol.sha256(protocol)}), encoding="utf-8")
    assert frozen_protocol.load(engine.quest_root) is not None
    state: dict[str, Any] = {"design": {"protocol": _PROTOCOL}}
    got = engine._scan_numeric_warnings(state, "", {"rate": math.nan, "debug_ratio": math.nan})  # type: ignore[arg-type]
    assert got == [{"kind": "non_finite", "text": "debug_ratio is nan"}]


def test_the_uncertainty_a_script_prints_beside_a_declared_result_is_disclosed_not_repaired_for() -> None:
    """The maintainer's decision (2026-10-27): only a key equal to a declared result's id counts; rate_se beside the
    declared rate is disclosed (FI's own intervals come from its records, and FI fills no non-finite number)."""
    results = nw.declared_results(_PROTOCOL)
    assert nw.scan("", {"rate": 0.4, "rate_se": math.nan}, results=results) == []
    assert nw.disclosed_non_finite({"rate": 0.4, "rate_se": math.nan}, results) == ["rate_se is nan"]


def test_a_frozen_protocol_with_no_results_keeps_every_non_finite_number(tmp_path: Path) -> None:
    from core import frozen_protocol

    engine = Engine(_cfg(tmp_path))
    frozen_path = frozen_protocol.frozen_path(engine.quest_root)
    frozen_path.parent.mkdir(parents=True, exist_ok=True)
    frozen_path.write_text(json.dumps({"schema": frozen_protocol.FROZEN_SCHEMA, "protocol": {},
                                       "sha256": frozen_protocol.sha256({})}), encoding="utf-8")
    state: dict[str, Any] = {"design": {"protocol": _PROTOCOL}}
    got = engine._scan_numeric_warnings(state, "", {"debug_ratio": math.nan})  # type: ignore[arg-type]
    assert got == [{"kind": "non_finite", "text": "debug_ratio is nan"}]


# ---- the role of a result decides which non-finite number stops (contract roles) ---------------------------------------

_ROLES = {"metrics": [{"id": "gain"}, {"id": "ideal", "reference": {"formula": "2 * x"}},
                      {"id": "agreement"}, {"id": "wall", "role": "timing"}, {"id": "edge", "role": "physical_bound"}],
          "oracles": [{"name": "agree_a", "kind": "second_implementation", "measure": "agreement"},
                      {"name": "agree_b", "kind": "invariant", "metric": "agreement"}]}


def test_only_a_result_a_reference_and_a_physical_bound_ask_for_a_repair() -> None:
    assert nw.declared_results(_ROLES) == {"gain", "ideal", "edge"}
    result = {"gain": math.nan, "ideal": math.inf, "edge": math.nan, "agreement": math.nan, "wall": math.nan}
    found = nw.scan("", result, results=nw.declared_results(_ROLES))
    assert [w.text for w in found] == ["gain is nan", "ideal is inf", "edge is nan"]
    assert nw.disclosed_non_finite(result, nw.declared_results(_ROLES)) == ["agreement is nan", "wall is nan"]


def test_a_plan_whose_results_are_all_diagnostics_stops_for_none_of_them() -> None:
    protocol = {"metrics": [{"id": "agreement"}], "oracles": [{"name": "agree", "kind": "symmetry", "measure": "agreement"}]}
    results = nw.declared_results(protocol)
    assert results == set()
    assert nw.scan("", {"agreement": math.nan, "other": math.nan}, results=results) == []


def test_a_check_s_measure_printed_as_nan_is_disclosed_with_fi_s_own_value(tmp_path: Path) -> None:
    engine = Engine(_cfg(tmp_path))
    record = {"attempts": [{"checks": [{"name": "agree_a", "value": 2.3e-15, "measured_by": "engine"},
                                       {"name": "agree_b", "value": 9.0, "measured_by": "script"}]}]}
    (engine.quest_root / "needs").mkdir(parents=True, exist_ok=True)
    (engine.quest_root / "needs" / "ORACLE_CHECK.json").write_text(json.dumps(record), encoding="utf-8")
    state: dict[str, Any] = {"design": {"protocol": _ROLES}}
    assert engine._scan_numeric_warnings(state, "", {"gain": 0.5, "agreement": math.nan}) == []  # type: ignore[arg-type]
    assert _recorded(engine)[-1]["disclosed"] == [
        "agreement is nan (FI's own run of the check `agree_a` measured 2.3e-15)"]


def test_the_contract_is_built_when_the_protocol_is_read_and_its_gaps_are_kept(tmp_path: Path) -> None:
    from core import disclosure

    engine = Engine(_cfg(tmp_path))
    design = {"protocol": {"metrics": [{"id": "gain"}, {"id": "gap"}],
                           "oracles": [{"name": "agree", "kind": "invariant", "measure": "gap"}]}}
    state: dict[str, Any] = {"design": design, "iteration": 1}
    protocol = engine._protocol_block(state)  # type: ignore[arg-type]
    assert "role" not in protocol["metrics"][0]  # never written back
    record = json.loads((engine.quest_root / ".fi" / "contract.json").read_text(encoding="utf-8"))
    assert any("no role for" in g for g in record["gaps"])
    assert all("no role" not in g and "older plan" not in g for g in record["shown"]) and len(record["shown"]) == 1
    text = disclosure.run_record(engine.quest_root)
    assert "In reading the plan, FI noted: result `gap` is the measure of the check `agree`" in text
    assert "no role" not in text and "older plan" not in text


def test_a_plan_with_only_the_default_gaps_adds_nothing_to_the_run_record(tmp_path: Path) -> None:
    from core import disclosure

    engine = Engine(_cfg(tmp_path))
    engine._protocol_block({"design": {"protocol": {"metrics": [{"id": "gain"}]}}, "iteration": 1})  # type: ignore[arg-type]
    record = json.loads((engine.quest_root / ".fi" / "contract.json").read_text(encoding="utf-8"))
    assert record["gaps"] and record["shown"] == []
    assert "In reading the plan" not in disclosure.run_record(engine.quest_root)


def test_a_candidate_design_does_not_write_the_contract_file(tmp_path: Path) -> None:
    engine = Engine(_cfg(tmp_path))
    state: dict[str, Any] = {"design": {"protocol": {"metrics": [{"id": "gain"}]}}, "iteration": 1}
    engine._protocol_block(state, adopted=False)  # type: ignore[arg-type]
    assert not (engine.quest_root / ".fi" / "contract.json").exists()
    engine._protocol_block(state)  # type: ignore[arg-type]
    assert (engine.quest_root / ".fi" / "contract.json").is_file()


def test_a_headline_result_that_a_special_case_check_measures_still_stops_on_a_nan() -> None:
    protocol = {"metrics": [{"id": "gain"}, {"id": "other"}],
                "oracles": [{"name": "limit", "kind": "special_case", "measure": "gain", "case": {"x": 0}}]}
    results = nw.declared_results(protocol)
    assert results == {"gain", "other"}
    assert [w.text for w in nw.scan("", {"gain": math.nan}, results=results)] == ["gain is nan"]


def test_a_declared_result_the_contract_cannot_place_still_counts_as_a_result() -> None:
    protocol = {"metrics": [{"id": "gain"}, {"id": "rate", "kind": "ratio"}, {"id": "ideal", "reference": "theory"}]}
    assert nw.declared_results(protocol) == {"gain", "rate", "ideal"}
    only_odd = {"metrics": [{"id": "rate", "kind": "ratio"}]}
    assert nw.declared_results(only_odd) == {"rate"}
    assert [w.text for w in nw.scan("", {"rate": math.nan}, results=nw.declared_results(only_odd))] == ["rate is nan"]


def test_a_result_on_its_bound_in_every_setting_reaches_the_run_record(tmp_path: Path) -> None:
    from core import disclosure

    engine = Engine(_cfg(tmp_path))
    (engine.quest_root / ".fi").mkdir(parents=True, exist_ok=True)
    text = "share is on its bound in every setting; the result may carry no information"
    (engine.quest_root / ".fi" / "edge_everywhere.json").write_text(json.dumps({"results": [text]}), encoding="utf-8")
    assert text + "." in disclosure.run_record(engine.quest_root)
