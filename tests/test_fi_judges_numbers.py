"""FI compares every expected number in its own process (review P0-1, probe_inprocess_theft.py).

The process that runs the quest's code (model-written or agent-written) is given the function, its inputs and how to
report the output, never the expected value, the tolerance or a path to them; it returns what the function computed
and FI judges it (core/equation_measure.py, core/equation_tests.run_and_judge). Before this, the equation test ran the
function in the very process that held the expected values, and a wrong Arrhenius law that looked the case up in the
test's own memory passed.
"""

from __future__ import annotations

import asyncio
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

from core import code_layout
from core import equation_tests as eqt
from core.execution import ExecutionResult
from core.outside_run import OutsideExecutor
from tests.equation_judge_helpers import fi_verdict

FIX = Path(__file__).parent / "fixtures" / "agent_cheats" / "kinetics"
HONEST = "    return A * np.exp(-Ea / (R * T))\n"
WRONG = "    return A * np.exp(-Ea / (2.0 * R * T))   # WRONG law (factor 2)\n"
# The probe's cheat: look the case up in any loaded module's CASES and return its expected value.
STEAL = (
    "    import sys\n"
    "    for mod in list(sys.modules.values()):\n"
    "        for case in (getattr(mod, 'CASES', None) or []):\n"
    "            if isinstance(case, dict) and case.get('function') == 'rate_constant' and "
    "case.get('inputs') == {'A': A, 'Ea': Ea, 'R': R, 'T': T} and 'expected' in case:\n"
    "                return case['expected']\n")
# The same, through every frame of the stack and every global: anything in the process that holds the expected value.
STEAL_ANYWHERE = (
    "    import gc, inspect\n"
    "    for obj in gc.get_objects():\n"
    "        if isinstance(obj, dict) and obj.get('function') == 'rate_constant' and 'expected' in obj:\n"
    "            return obj['expected']\n"
    "    for f in inspect.stack():\n"
    "        for v in list(f.frame.f_locals.values()):\n"
    "            if isinstance(v, dict) and 'expected' in v:\n"
    "                return v['expected']\n")
# Patch the comparison: make abs() always 0 in the process that imports the code.
PATCH = "    import builtins as _b\n    _b.abs = lambda x: 0.0\n"


def _quest(tmp_path: Path, body: str = HONEST) -> tuple[Path, list[dict]]:
    root = tmp_path / "quest"
    shutil.copytree(FIX / "clean", root / "code")
    model = root / "code" / "kinetics_parameter_recovery" / "model.py"
    src = model.read_text(encoding="utf-8")
    assert src.count(HONEST) == 1
    model.write_text(src.replace(HONEST, body), encoding="utf-8")
    protocol = json.loads((FIX / "protocol.json").read_text(encoding="utf-8"))
    sources = dict(code_layout.package_sources(root / "code"))
    sources["simulate.py"] = (root / "code" / "simulate.py").read_text(encoding="utf-8")
    cases = eqt.cases(eqt.example_rows(protocol), eqt.locate(protocol, sources))
    code_layout.write_check_values(root / "code", protocol)
    eqt.write(root / "code", cases)
    return root, cases


def _statuses(results: list[dict]) -> dict[str, str]:
    return {r["id"]: r["status"] for r in results}


def test_the_clean_code_passes_and_the_wrong_law_alone_is_caught(tmp_path: Path) -> None:
    root, cases = _quest(tmp_path)
    assert _statuses(fi_verdict(cases, root, code="code")) == {"E1": "ok", "E2": "ok", "E3": "ok"}
    root, cases = _quest(tmp_path / "w", WRONG)
    assert _statuses(fi_verdict(cases, root, code="code"))["E2"] == eqt.MISMATCH


def test_the_wrong_law_that_reads_the_tests_memory_is_caught(tmp_path: Path) -> None:
    """The probe: before, E1 ok, E2 ok, E3 ok (the theft passed); now E2 is a mismatch."""
    root, cases = _quest(tmp_path, STEAL + WRONG)
    # A person's copy of the test still runs it in one process and is still fooled: FI never takes its verdict.
    person = subprocess.run([sys.executable, "-B", "code/" + eqt.TEST_PATH, "--json"], cwd=root, capture_output=True,
                            text=True, timeout=300)
    assert {r["id"]: r["status"] for r in eqt.parse_results(person.stdout)}["E2"] == eqt.OK
    assert _statuses(fi_verdict(cases, root, code="code"))["E2"] == eqt.MISMATCH


def test_nothing_in_the_measuring_process_holds_an_expected_value(tmp_path: Path) -> None:
    root, cases = _quest(tmp_path, STEAL_ANYWHERE + WRONG)
    assert _statuses(fi_verdict(cases, root, code="code"))["E2"] == eqt.MISMATCH


def test_patching_the_comparison_in_the_code_changes_nothing(tmp_path: Path) -> None:
    root, cases = _quest(tmp_path, PATCH + WRONG)
    assert _statuses(fi_verdict(cases, root, code="code"))["E2"] == eqt.MISMATCH


def test_what_the_measuring_process_is_given_holds_no_value_to_compare_with(tmp_path: Path) -> None:
    _root, cases = _quest(tmp_path)
    spec = eqt.measure_spec(cases, nonce="n")
    for case in spec["cases"]:
        assert set(case) == {"id", "module", "function", "inputs", "random", "names", "outputs"}
    argv = eqt.measure_argv(cases, nonce="n")
    text = " ".join(argv)
    for c in cases:
        for value in (c["expected"], c["rel"], c["abs"]):
            if isinstance(value, float) and value not in (0.0, 1.0):
                assert repr(value) not in text and f"{value:g}" not in text, value
    assert ".fi" not in text and "checks" not in text


class _Recorder:
    """An executor that runs for real and keeps where it ran and what it was given."""

    def __init__(self) -> None:
        self.cwd: Path | None = None
        self.listing: set[str] = set()

    async def execute(self, cmd, *, cwd, timeout_s, env=None):
        self.cwd = Path(cwd)
        self.listing = {p.relative_to(cwd).as_posix() for p in Path(cwd).rglob("*")}
        done = await asyncio.to_thread(subprocess.run, cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout_s,
                                       env=env)
        return ExecutionResult(done.returncode, done.stdout, done.stderr, 0.0)


async def test_the_measuring_run_is_made_from_a_copy_outside_the_quest(tmp_path: Path) -> None:
    root, cases = _quest(tmp_path, STEAL + WRONG)
    (root / "needs").mkdir()
    (root / "needs" / "FROZEN_PROTOCOL.json").write_text("{}", encoding="utf-8")
    inner = _Recorder()
    executor = OutsideExecutor(inner, root, base=tmp_path / "scratch")

    async def run(argv, **kw):
        ran = await executor.execute([sys.executable, "-B", *argv], cwd=root, timeout_s=300, env=kw.get("env"))
        return ran.returncode, ran.stdout, ran.stderr

    results, _rc, err = await eqt.run_and_judge(run, cases)
    assert results is not None, err
    assert _statuses(results)["E2"] == eqt.MISMATCH
    assert inner.cwd is not None and not inner.cwd.is_relative_to(root.resolve())
    assert root.name not in str(inner.cwd)
    assert not any(p.split("/")[0] in ("needs", ".fi", ".venv", ".git") for p in inner.listing), inner.listing
    assert not any(p.endswith(tuple(eqt.STALE_NAMES)) for p in inner.listing)
    assert not inner.cwd.exists(), "the copy is removed after the run"


def test_the_wrong_law_is_really_wrong_away_from_the_example(tmp_path: Path) -> None:
    root, _cases = _quest(tmp_path, WRONG)
    out = subprocess.run([sys.executable, "-B", "-c",
                          "from kinetics_parameter_recovery.model import rate_constant as k;"
                          "print(k(1e7, 5e4, 8.314462618, 310.0))"], cwd=root / "code", capture_output=True, text=True)
    assert float(out.stdout) != __import__("pytest").approx(1e7 * math.exp(-5e4 / (8.314462618 * 310.0)), rel=1e-3)
