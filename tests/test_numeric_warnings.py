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
plt.figure(); plt.plot([0, 1, 2], [0, 1, 4]); plt.savefig('figures/result.png', dpi=72)
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
    assert "[numeric] paused" in log and "[FI] paused for the run's numeric warnings" in log and "paused for clarify" not in log
    assert not (first.fi_dir / "clarify_questions.json").exists()
    descriptor = json.loads((first.fi_dir / "pause.json").read_text(encoding="utf-8"))
    assert descriptor["kind"] == "numeric" and descriptor["interaction"] == "supply"
    text = (first.quest_root / "NEXT_STEP.md").read_text(encoding="utf-8")
    assert "an argument had no effect" in text and "warnings accepted" in text
    assert not (first.quest_root / "paper" / "paper.md").exists()

    (first.quest_root / "code" / "experiment.py").write_text(_QUIET, encoding="utf-8")
    second = Engine(cfg, resume_quest_id=first.quest_id)
    artifacts = await second.run()
    assert artifacts.paper_md is not None and artifacts.paper_md.exists()
    assert "the script was changed while the quest was stopped; running it again" in (second.quest_root / ".fi" / "run.log").read_text(encoding="utf-8")
    assert not (second.fi_dir / "numeric_stop.json").exists()


@pytest.mark.asyncio
async def test_resuming_without_changing_the_script_accepts_the_warnings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake(calls, implement=_WARNS, repair=_WARNS))
    cfg = _cfg(tmp_path)
    first = Engine(cfg)
    await first.run()
    repairs = calls.count("NumericRepair")

    second = Engine(cfg, resume_quest_id=first.quest_id)
    artifacts = await second.run()

    assert artifacts.paper_md is not None and artifacts.paper_md.exists()
    assert calls.count("NumericRepair") == repairs, "no further repair was asked for"
    assert "resumed with the script unchanged: the run's warnings are accepted" in (
        second.quest_root / ".fi" / "run.log").read_text(encoding="utf-8")
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
