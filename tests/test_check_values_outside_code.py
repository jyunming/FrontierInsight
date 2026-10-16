"""The values FI's own tests compare with never land in ``code/`` (core/code_layout.py, core/equation_tests.py).

A coding agent that works in ``code/`` must not be able to read the expected value of a check, its tolerance, or a worked
example's inputs and output: with them it could special-case the check. FI writes its tests into ``code/tests/`` as
before, but the values they compare with go to the quest's ``.fi/checks/``, beside ``code/``, and the tests read them
when they run. The tests still run and give the same verdicts; a grep of everything under ``code/`` finds none of the
values.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

from core import code_layout as cl
from core import code_project
from core import equation_tests as eqt
from tests.equation_judge_helpers import fi_verdict

PKG = "decay_study"
# Distinctive numbers: none of them can turn up in code/ by chance.
EXPECTED = 0.7385113
TOLERANCE = 0.0031415
INPUTS = {"c0": 1.8371, "k": 0.6293, "t": 2.4717}
EXAMPLE_OUT = INPUTS["c0"] * math.exp(-INPUTS["k"] * INPUTS["t"])
PROTOCOL = {
    "grid": {"k": [0.3031, 0.9]},
    "runs_per_setting": 2,
    "metrics": [{"id": "c_end"}],
    "model": {"equations": [{"id": "E1", "formula": "c = c0 * exp(-k * t)", "role": "generates", "source": "textbook",
                             "example": {"inputs": INPUTS, "expected_formula": "c0 * exp(-k * t)"}}]},
    "oracles": [{"name": "decay at k 0.3031", "kind": "special_case", "check": "c at t = 1 for c0 = 1",
                 "expected": EXPECTED, "tolerance": TOLERANCE, "reference": "E1", "case": {"k": 0.3031},
                 "measure": "c_end"}],
}
MODEL = '''"""First-order decay."""
import math


# E1
def concentration(c0, k, t):
    return c0 * math.exp(-k * t)
'''
SIMULATE = f'''from {PKG} import model


def run_trial(cell, trial_id, seed):
    return {{"c_end": model.concentration(1.0, cell["k"], 1.0)}}
'''
ANALYSIS = 'print("RESULT_JSON: {}")\n'


def _quest(tmp_path: Path, model: str = MODEL) -> Path:
    """A quest folder whose code/ FI has laid out the way the engine does (core/engine.py _refresh_code_project and the
    equation gate)."""
    root = tmp_path / "quest"
    code = root / "code"
    (code / PKG).mkdir(parents=True)
    (code / PKG / "__init__.py").write_text('"""Decay."""\n', encoding="utf-8")
    (code / PKG / "model.py").write_text(model, encoding="utf-8")
    (code / "simulate.py").write_text(SIMULATE, encoding="utf-8")
    (code / "experiment.py").write_text(ANALYSIS, encoding="utf-8")
    cl.write_check_values(code, PROTOCOL)
    code_project.refresh(root, deps=[], protocol=PROTOCOL, title="Decay study", question="How does c decay?",
                         extra_files=cl.project_files(code, PROTOCOL, PKG), readme_files=cl.readme_lines(PKG))
    sources = dict(cl.package_sources(code))
    eqt.write(code, eqt.cases(eqt.example_rows(PROTOCOL), eqt.locate(PROTOCOL, sources)))
    return root


def _cases_of(code: Path) -> list[dict]:
    return eqt.cases(eqt.example_rows(PROTOCOL), eqt.locate(PROTOCOL, dict(cl.package_sources(code))))


def _literals(value: float) -> set[str]:
    return {repr(value), f"{value:.6g}", f"{value:.5f}", f"{value:.4f}"}


def _grep(folder: Path, needles: set[str]) -> list[str]:
    hits = []
    for path in folder.rglob("*"):
        if not path.is_file() or ".git" in path.relative_to(folder).parts:
            continue
        text = path.read_bytes().decode("utf-8", "replace")
        hits += [f"{path.relative_to(folder).as_posix()}: {n}" for n in needles if n in text]
    return hits


def test_a_generated_code_tree_holds_no_expected_value_tolerance_or_worked_example(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    code = root / "code"
    assert (code / cl.TEST_PATH).is_file() and (code / eqt.TEST_PATH).is_file()
    needles = _literals(EXPECTED) | _literals(TOLERANCE) | _literals(EXAMPLE_OUT)
    needles |= {repr(v) for v in INPUTS.values()}
    assert not _grep(code, needles)
    # The same grep does find them where FI keeps them (so a clean grep of code/ means something).
    found = _grep(root / eqt.CHECKS_DIR, needles)
    assert any(repr(EXPECTED) in h for h in found) and any(repr(INPUTS["k"]) in h for h in found)
    # ... and the values are where the tests look for them, outside code/.
    checks = root / eqt.CHECKS_DIR
    assert json.loads((checks / cl.ORACLE_VALUES_NAME).read_text(encoding="utf-8")) == {
        "decay at k 0.3031": {"expected": EXPECTED, "limit": TOLERANCE}}
    assert json.loads((checks / eqt.VALUES_NAME).read_text(encoding="utf-8"))["E1"]["inputs"] == INPUTS
    # The README says where they are.
    assert ".fi/checks/" in (code / "README.md").read_text(encoding="utf-8")


def test_the_tests_still_run_from_code_and_give_the_same_verdicts(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    code = root / "code"
    env = dict(os.environ)
    oracles = subprocess.run([sys.executable, "-B", cl.TEST_PATH], cwd=code, capture_output=True, text=True, env=env,
                             timeout=60)
    assert oracles.returncode == 0, oracles.stdout + oracles.stderr
    # A person's run of the equation tests from the quest folder, as `code/tests/test_equations.py --json`.
    eq = subprocess.run([sys.executable, "-B", "code/" + eqt.TEST_PATH, "--json"], cwd=root, capture_output=True,
                        text=True, env=env, timeout=60)
    assert [r["status"] for r in eqt.parse_results(eq.stdout)] == [eqt.OK]
    # FI's own verdict: measured in a process given only the inputs, judged in FI's process with the numbers for the log.
    fi = fi_verdict(_cases_of(code), root, code="code")
    assert [r["status"] for r in fi] == [eqt.OK] and fi[0]["expected"] == EXAMPLE_OUT
    assert not any((code / eqt.TESTS_DIR / n).exists() for n in eqt.STALE_NAMES)
    # pytest finds the same tests.
    run = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"], cwd=code,
                         capture_output=True, text=True, env=env, timeout=120)
    assert run.returncode == 0 and "2 passed" in run.stdout, run.stdout + run.stderr


def test_a_wrong_equation_still_fails_both_tests(tmp_path: Path) -> None:
    root = _quest(tmp_path, MODEL.replace("c0 * math.exp(-k * t)", "c0 * math.exp(-1.1 * k * t)"))
    code = root / "code"
    oracles = subprocess.run([sys.executable, "-B", cl.TEST_PATH], cwd=code, capture_output=True, text=True, timeout=60)
    assert oracles.returncode == 1 and "FAIL" in oracles.stdout
    eq = subprocess.run([sys.executable, "-B", "code/" + eqt.TEST_PATH, "--json"], cwd=root, capture_output=True,
                        text=True, timeout=60)
    result = eqt.parse_results(eq.stdout)[0]
    assert result["status"] == eqt.MISMATCH and repr(EXAMPLE_OUT) not in result["detail"]


def test_without_fis_values_a_test_fails_and_never_passes(tmp_path: Path) -> None:
    """A copy of code/ taken out of the quest folder (or a test whose values are gone) does not pass: it says where the
    values are kept."""
    root = _quest(tmp_path)
    for path in (root / eqt.CHECKS_DIR).iterdir():
        path.unlink()
    code = root / "code"
    oracles = subprocess.run([sys.executable, "-B", cl.TEST_PATH], cwd=code, capture_output=True, text=True, timeout=60)
    assert oracles.returncode == 1 and ".fi/checks/" in oracles.stdout
    eq = subprocess.run([sys.executable, "-B", "code/" + eqt.TEST_PATH, "--json"], cwd=root, capture_output=True,
                        text=True, timeout=60)
    result = eqt.parse_results(eq.stdout)[0]
    assert result["status"] == eqt.ERROR and ".fi/checks/" in result["detail"]


def test_a_person_names_another_checks_folder_on_the_command_line_and_never_by_the_environment(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    moved = tmp_path / "elsewhere"
    (root / eqt.CHECKS_DIR).rename(moved)
    run = subprocess.run([sys.executable, "-B", cl.TEST_PATH, "--checks", str(moved)], cwd=root / "code",
                         capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stdout + run.stderr
    # A variable left in the environment (an older FI's FI_CHECKS_DIR) is never read.
    stray = subprocess.run([sys.executable, "-B", cl.TEST_PATH], cwd=root / "code", capture_output=True, text=True,
                           timeout=60, env={**os.environ, "FI_CHECKS_DIR": str(moved)})
    assert stray.returncode == 1, stray.stdout + stray.stderr
