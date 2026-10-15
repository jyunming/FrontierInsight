"""A repair may change what a correct fix needs (core/finding_repair.py): add a parameter after a function's own (with a
default value) and pass it on from the functions that call it directly, in another file too; and move what runs when a
file is imported under ``if __name__ == "__main__":``. Every acceptance check stays: the finding's own assertion passes,
every check that passed before still passes, the static checks find no more faults, and FI's small run still runs.

Replays on real code (copies; the quests are not touched):

* r12 / study 2: the delivered simulation types the noise level into ``fit_rate`` (tests/fixtures/replay/r12_code/). The
  correct repair (a ``noise_std_rel`` parameter with a default, passed on by ``run_trial`` in simulate.py) was rejected by
  the old parameter rule; it is accepted now. A wrong repair (the parameter added, the typed value kept) is still
  rejected, by the finding's own assertion. The finding's test is the one the quest kept
  (tests/fixtures/replay/finding_v2/r12_written/.fi/finding_tests/F-b166c00c_1.py).
* r8: experiment.py ran its whole analysis when a test imported it (``KeyError: 'FI_TRIALS'``); the repairs that put the
  last line under a main guard were rejected as "changed the module" (tests/fixtures/replay/finding_repairs/r8/)."""

from __future__ import annotations

import ast
import asyncio
import logging
import os
import shutil
import string
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from core import finding_repair as fr

FIX = Path(__file__).parent / "fixtures" / "replay"
AGENTS = Path(__file__).parent.parent / "agents"


def _env(root: Path, reply: str, *, small_run: Any = None) -> fr.Env:
    async def ask(prompt: str, who: str, kind: str) -> str:
        return reply

    async def run_tests(paths: list[str]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for path in paths:
            for cache in (root / "code").glob("**/__pycache__"):
                shutil.rmtree(cache, ignore_errors=True)
            done = subprocess.run([sys.executable, "-B", "-c", fr.harness_source("code", [path])], cwd=root,
                                  capture_output=True, text=True, timeout=300,
                                  env={**os.environ, **fr.run_environment(root)})
            out.update(fr.parse_runs(done.stdout, [path]))
        return out

    async def snapshot() -> dict[str, str]:
        return {}

    prompts = {"code_review_repair": string.Template((AGENTS / "code_review_repair.md").read_text(encoding="utf-8"))}
    return fr.Env(root=root, code=root / "code", record={}, save=lambda: None, ask=ask, run_tests=run_tests,
                  snapshot=snapshot, static=dict, keeps=lambda *a: "", prompts=prompts, topic="t", equations="[]",
                  scrub=lambda t: t, log=logging.getLogger("test"), small_run=small_run)


def _fn(source: str, name: str) -> str:
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    lines = source.splitlines()
    return "\n".join(lines[node.lineno - 1:node.end_lineno])


# ---- the parameter rule ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("new,ok", [
    ("def f(a, b, c=1): pass", True),
    ("def f(a, b, *, c=1): pass", True),
    ("def f(a, b, *args, **kw): pass", True),
    ("def f(a, b, c): pass", False),          # an added parameter without a default breaks every call
    ("def f(a, c=1, b=2): pass", False),      # a parameter moved
    ("def f(a): pass", False),                # a parameter dropped
    ("def f(a, bb): pass", False),            # a parameter renamed
    ("def f(a, b, *, c): pass", False),
])
def test_parameters_may_only_be_added_after_with_a_default(new: str, ok: bool) -> None:
    old = ast.parse("def f(a, b): pass").body[0]
    assert (fr.params_kept(old, ast.parse(new).body[0]) == "") is ok  # type: ignore[arg-type]


def test_a_default_the_function_had_stays() -> None:
    old = ast.parse("def f(a, b=2): pass").body[0]
    assert fr.params_kept(old, ast.parse("def f(a, b=2, c=3): pass").body[0]) == ""  # type: ignore[arg-type]
    assert fr.params_kept(old, ast.parse("def f(a, b, c=3): pass").body[0])  # type: ignore[arg-type]


# ---- r12: the correct repair is accepted, the wrong one is not ---------------------------------------------------------


def _r12(tmp_path: Path) -> Path:
    root = tmp_path / "r12"
    shutil.copytree(FIX / "r12_code", root / "code")
    tests = root / ".fi" / "finding_tests"
    tests.mkdir(parents=True)
    shutil.copy2(FIX / "finding_v2" / "r12_written" / ".fi" / "finding_tests" / "F-b166c00c_1.py", tests)
    return root


ISSUE = {"id": "F-b166c00c", "file": "kinetics_parameter_recovery/model.py", "function": "fit_rate", "changes_result": True,
         "text": "fit_rate types the observation noise level in instead of using the one the trial was run with",
         "test": ".fi/finding_tests/F-b166c00c_1.py", "failure": "TypeError at the test's own call"}


def _replies(root: Path) -> tuple[str, str]:
    model = (root / "code" / "kinetics_parameter_recovery" / "model.py").read_text(encoding="utf-8")
    sim = (root / "code" / "simulate.py").read_text(encoding="utf-8")
    head = "def fit_rate(time: np.ndarray, concentration: np.ndarray) -> tuple[float, float]:"
    fit = _fn(model, "fit_rate")
    assert head in fit and "    noise_std_rel = 0.01\n" in fit
    with_param = fit.replace(head, "def fit_rate(time: np.ndarray, concentration: np.ndarray,\n"
                                   "             noise_std_rel: float = 0.01) -> tuple[float, float]:")
    caller = _fn(sim, "run_trial").replace("model.fit_rate(time, observed)", "model.fit_rate(time, observed, noise_std_rel)")
    correct = with_param.replace("    noise_std_rel = 0.01\n", "")
    wrong = with_param  # the parameter added, the typed-in value still used
    as_reply = ("```python\n{fit}\n```\n\n```python\n# file: simulate.py\nimport math\nimport numpy as np\n"
                "from kinetics_parameter_recovery import model\n\n\n{caller}\n```")
    return as_reply.format(fit=correct, caller=caller), as_reply.format(fit=wrong, caller=caller)


def test_replay_r12_the_correct_repair_is_accepted_with_its_caller(tmp_path: Path) -> None:
    pytest.importorskip("scipy")
    root = _r12(tmp_path)
    correct, _wrong = _replies(root)
    ran: list[int] = []

    async def small_run() -> str:
        ran.append(1)
        return ""

    env = _env(root, correct, small_run=small_run)
    ok, why = asyncio.run(fr._attempt(env, dict(ISSUE), "kinetics_parameter_recovery/model.py", "code", ""))
    assert ok, why
    assert sorted(env.changed) == ["kinetics_parameter_recovery/model.py", "simulate.py"] and ran == [1]
    sim = (root / "code" / "simulate.py").read_text(encoding="utf-8")
    assert "model.fit_rate(time, observed, noise_std_rel)" in sim and "def oracle()" in sim
    assert "noise_std_rel = 0.01" not in (root / "code" / "kinetics_parameter_recovery" / "model.py").read_text(encoding="utf-8")


def test_replay_r12_a_wrong_repair_is_still_rejected_by_the_findings_assertion(tmp_path: Path) -> None:
    pytest.importorskip("scipy")
    root = _r12(tmp_path)
    model_before = (root / "code" / "kinetics_parameter_recovery" / "model.py").read_text(encoding="utf-8")
    sim_before = (root / "code" / "simulate.py").read_text(encoding="utf-8")
    _correct, wrong = _replies(root)
    ok, why = asyncio.run(fr._attempt(_env(root, wrong), dict(ISSUE), "kinetics_parameter_recovery/model.py", "code", ""))
    assert not ok and "the finding's own test still fails" in why and "does not scale" in why
    # both files are put back
    assert (root / "code" / "kinetics_parameter_recovery" / "model.py").read_text(encoding="utf-8") == model_before
    assert (root / "code" / "simulate.py").read_text(encoding="utf-8") == sim_before


def test_a_repair_the_small_run_no_longer_runs_is_not_kept(tmp_path: Path) -> None:
    pytest.importorskip("scipy")
    root = _r12(tmp_path)
    correct, _wrong = _replies(root)
    before = (root / "code" / "simulate.py").read_text(encoding="utf-8")

    async def small_run() -> str:
        return "every trial at noise_std_rel=0.08 failed"

    ok, why = asyncio.run(fr._attempt(_env(root, correct, small_run=small_run), dict(ISSUE),
                                      "kinetics_parameter_recovery/model.py", "code", ""))
    assert not ok and "small run" in why
    assert (root / "code" / "simulate.py").read_text(encoding="utf-8") == before


def test_a_change_to_a_function_that_does_not_call_it_is_not_used(tmp_path: Path) -> None:
    pytest.importorskip("scipy")
    root = _r12(tmp_path)
    correct, _wrong = _replies(root)
    reply = correct + "\n\n```python\n# file: simulate.py\ndef oracle() -> dict:\n    return {}\n```"
    env = _env(root, reply)
    ok, why = asyncio.run(fr._attempt(env, dict(ISSUE), "kinetics_parameter_recovery/model.py", "code", ""))
    assert ok, why  # the caller's change is used ...
    sim = (root / "code" / "simulate.py").read_text(encoding="utf-8")
    assert "model.fit_rate(time, observed, noise_std_rel)" in sim
    assert "return {}" not in sim  # ... a function in simulate.py that does not call fit_rate is not changed


# ---- r8: the main guard ------------------------------------------------------------------------------------------------


def test_replay_r8_the_main_guard_repair_is_accepted(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    root = tmp_path / "r8"
    shutil.copytree(FIX / "finding_repairs" / "r8", root)
    old = (root / "code" / "experiment.py").read_text(encoding="utf-8")
    last = 'print("RESULT_JSON: " + json.dumps(main(), sort_keys=True, allow_nan=False))'
    assert last in old
    new = (old.replace(last, 'if __name__ == "__main__":\n    ' + last)
              .replace('    summary["k_relative_bias_by_temp"] = k_bias\n',
                       '    summary["k_relative_bias_by_temp"] = k_bias\n    summary["k_relative_bias"] = k_bias\n'))
    issue = {"id": "F-b5e6edb8", "file": "", "function": "", "quantity": "krelativebias", "changes_result": True,
             "text": "the dependent variable k_relative_bias is computed and reported in the result",
             "test": ".fi/finding_tests/F-b5e6edb8_1.py"}
    # what the old rule said about it
    assert fr.unspoken_frame_changes(old, new, issue)
    assert not fr.unspoken_frame_changes(*fr.without_main_guard(old, new), issue)
    reply = f"```python\n# file: experiment.py\n{new}\n```"
    ok, why = asyncio.run(fr._attempt(_env(root, reply), dict(issue), "experiment.py", "code", ""))
    assert ok, why
    assert (root / "code" / "experiment.py").read_text(encoding="utf-8").rstrip() == new.rstrip()


def test_a_guard_that_drops_what_ran_is_not_a_move() -> None:
    old = "import os\nX = 1\nprint(X)\n"
    new = 'import os\nX = 1\nif __name__ == "__main__":\n    print(2)\n'
    was, now = fr.without_main_guard(old, new)
    assert (was, now) == (old, new)  # print(X) is gone, not moved: compared as it is
    assert fr.frame_changes(was, now)
