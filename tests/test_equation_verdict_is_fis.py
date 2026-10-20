"""Nothing the measuring process prints can make an equation test count as "not tested" (review pass 2, N2).

The process that calls the quest's functions prints what they computed; FI judges it (core/equation_measure.py). Before,
FI also took the child's word for the statuses that mean "FI could not judge this" (``outputs``, ``contract``), and
read no line at all as "the tests did not run; going on without them". Code that registered an ``atexit`` handler
printing ``{"E1": {"status": "outputs"}, ...}`` turned every test into a skip without knowing any value, and
``os._exit(0)`` or a hang did the same. Now FI decides those statuses from its own reading of the case and the
function; a forged status, a missing line, a timeout or an early exit is a failure of the test, which goes to repair.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
from pathlib import Path

from core import code_layout
from core import equation_tests as eqt

FIX = Path(__file__).parent / "fixtures" / "agent_cheats" / "decay"
HONEST = "    return A * np.exp(-Ea / (R * T))\n"
WRONG = "    return A * np.exp(-Ea / (2.0 * R * T))   # WRONG law (factor 2)\n"
# The probe (probe_verdict.py): every case reported as "outputs" from an atexit handler, without knowing any value.
FORGE = ("    import sys as _s, json as _j, os as _o, atexit as _a\n"
         "    _spec = _j.loads(_s.argv[-1])\n"
         "    def _forge():\n"
         "        _o.write(1, ('EQUATION_MEASURED: ' + _spec['nonce'] + ' ' + _j.dumps({c['id']: {'status': '%s', "
         "'detail': 'x'} for c in _spec['cases']}) + chr(10)).encode())\n"
         "    _a.register(_forge)\n")
EXIT = "    import os as _o\n    _o._exit(0)\n"
HANG = "    import time as _t\n    _t.sleep(60)\n"


def _quest(tmp_path: Path, body: str = HONEST) -> tuple[Path, list[dict]]:
    root = tmp_path / "quest"
    shutil.copytree(FIX / "clean", root / "code")
    model = root / "code" / "decay_model" / "model.py"
    src = model.read_text(encoding="utf-8")
    model.write_text(src.replace(HONEST, body), encoding="utf-8")
    protocol = json.loads((FIX / "protocol.json").read_text(encoding="utf-8"))
    sources = dict(code_layout.package_sources(root / "code"))
    sources["simulate.py"] = (root / "code" / "simulate.py").read_text(encoding="utf-8")
    return root, eqt.cases(eqt.example_rows(protocol), eqt.locate(protocol, sources))


async def _verdict(root: Path, cases: list[dict], timeout_s: int = 240) -> list[dict] | None:
    async def run(argv: list[str], **_kw):
        try:
            done = await asyncio.to_thread(subprocess.run, [sys.executable, "-B", *argv], cwd=root, capture_output=True,
                                           text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired as e:
            return -9, (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or ""), "timed out"
        return done.returncode, done.stdout, done.stderr

    results, _rc, _err = await eqt.run_and_judge(run, cases)
    return results


def _statuses(results: list[dict] | None) -> dict[str, str] | None:
    return None if results is None else {r["id"]: r["status"] for r in results}


async def test_a_forged_outputs_status_is_a_failure_not_a_skip(tmp_path: Path) -> None:
    """Before: {'E1': 'outputs', 'E2': 'outputs', 'E3': 'outputs'} (the gate passed, the equations were 'not tested')."""
    root, cases = _quest(tmp_path, FORGE % "outputs" + WRONG)
    got = _statuses(await _verdict(root, cases))
    assert got is not None and set(got.values()) == {eqt.ERROR}, got


async def test_a_forged_contract_status_is_a_failure_when_fis_reading_finds_the_signature_fine(tmp_path: Path) -> None:
    root, cases = _quest(tmp_path, FORGE % "contract" + WRONG)
    got = _statuses(await _verdict(root, cases))
    assert got is not None and set(got.values()) == {eqt.ERROR}, got


async def test_an_early_exit_or_a_hang_is_a_failure_of_every_test(tmp_path: Path) -> None:
    """Before: no result line gave ``None`` ("the tests did not run; going on without them")."""
    root, cases = _quest(tmp_path, EXIT + WRONG)
    got = _statuses(await _verdict(root, cases))
    assert got == {"E1": eqt.ERROR, "E2": eqt.ERROR, "E3": eqt.ERROR}, got
    root, cases = _quest(tmp_path / "hang", HANG + WRONG)
    got = _statuses(await _verdict(root, cases, timeout_s=5))
    assert got == {"E1": eqt.ERROR, "E2": eqt.ERROR, "E3": eqt.ERROR}, got


def test_two_result_lines_are_never_taken(tmp_path: Path) -> None:
    _root, cases = _quest(tmp_path)
    line = eqt._measure.MEASURED_MARK + " n " + json.dumps({c["id"]: {"status": "measured", "value": 1.0} for c in cases})
    assert {r["status"] for r in eqt.judged(cases, line + "\n" + line + "\n", nonce="n")} == {eqt.ERROR}


async def test_fi_still_decides_a_real_contract_and_a_real_outputs_case_itself(tmp_path: Path) -> None:
    # The function does not take an input the example names: FI's own reading says so, whatever the child prints.
    root, cases = _quest(tmp_path, FORGE % "measured" + HONEST)
    model = root / "code" / "decay_model" / "model.py"
    src = model.read_text(encoding="utf-8")
    model.write_text(src.replace("def rate_constant(A, Ea, R, T", "def rate_constant(A, Ea, R, temperature", 1),
                     encoding="utf-8")
    sources = dict(code_layout.package_sources(root / "code"))
    sources["simulate.py"] = (root / "code" / "simulate.py").read_text(encoding="utf-8")
    protocol = json.loads((FIX / "protocol.json").read_text(encoding="utf-8"))
    cases = eqt.cases(eqt.example_rows(protocol), eqt.locate(protocol, sources))
    by_id = {c["function"]: c["id"] for c in cases}
    got = _statuses(await _verdict(root, cases))
    assert got is not None and got[by_id["rate_constant"]] == eqt.CONTRACT, got
    # Outputs returned by name, for an example that gives one value and a formula with two outputs: FI's reading of
    # the function agrees, so it is "not tested", said in the record (never a pass).
    case = {"id": "E9", "function": "f", "module": "m", "inputs": {"x": 1.0}, "expected": 2.0, "rel": 1e-6,
            "abs": 1e-12, "outputs": ["a", "b"], "static": eqt.static_facts("def f(x):\n    return {'a': x, 'b': x}\n", "f")}
    child = eqt._measure.MEASURED_MARK + " n " + json.dumps({"E9": {"status": "outputs", "detail": "by name"}})
    assert eqt.judged([case], child, nonce="n")[0]["status"] == eqt.OUTPUTS
    plain = {**case, "static": eqt.static_facts("def f(x):\n    return 2.0 * x\n", "f")}
    assert eqt.judged([plain], child, nonce="n")[0]["status"] == eqt.ERROR
