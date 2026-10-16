"""Everything the code writer sees holds no expected value and no tolerance of FI's checks (review P1-1).

What the writer (a model, or the coding agent) is given: the design block of every code-writing prompt
(``Engine._code_design_block``, both forms), the protocol shown to a repair (``_stripped``), the copy of ``code/`` it
works on (FI's own files there: README, run.py, study.json, METHODS.md, the tests; and the code as it stands), and the
failure messages FI sends back. Each check's expected value and tolerance, and each worked example's output, is
searched for in all of it, spelled the usual ways (``0.6415``, ``50000.0``, ``50000``, ``95%``...).

Two exclusions, stated rather than hidden:

* a number that is also an INPUT the plan gives the code (a grid value, a threshold, the setting a check runs at):
  the code must be given it to run at all (``Ea_true_J_per_mol: 50000`` is both the true value the simulation draws
  from and what the noise-free check recovers). Each such case is listed in :data:`EXCUSED` and the test fails when the
  list is not exactly what it finds, so nothing is excused quietly;
* a constant of FI's own files that is the same for every quest (:data:`FI_CONSTANTS`, by the file it is in);
* 0 and 1, which every program spells.
"""

from __future__ import annotations

import copy
import json
import logging
import re
import shutil
from pathlib import Path

import pytest

from core import agent_coder as ac
from core import code_layout, code_project
from core import equation_measure
from core import equation_tests as eqt
from core.engine import Engine, _stripped

FIXTURES = Path(__file__).parent / "fixtures" / "agent_cheats"

#: (topic, check, which number, value): the expected values and tolerances that are also inputs the code is given.
EXCUSED = {
    ("kinetics", "noise_free_Ea", "expected", 50000.0),          # thresholds.Ea_true_J_per_mol
    ("kinetics", "noise_free_Ea", "tolerance", 0.05),            # grid.sigma
    ("kinetics", "noise_free_lnA", "expected", 16.0),            # thresholds.lnA_true
    ("kinetics", "low_noise_coverage_95", "expected", 0.95),     # thresholds.confidence_level
    ("kinetics", "low_noise_SD_over_theory_SE", "tolerance", 0.1),     # grid.sigma
    ("kinetics", "low_noise_reportedSE_over_SD", "tolerance", 0.1),    # grid.sigma
    ("fdr", "bh_fwer_all_null", "expected", 0.05),               # the procedures' alpha
}
#: Numbers of FI's own files that are the same for every quest and happen to equal a check's number, with where.
FI_CONSTANTS = {
    ("kinetics", "sampling_window_kt_max", "tolerance", 1e-09): "copy: run.py",   # run.py's float comparison, rel_tol=1e-9
}


def _numbers(node, skip=("oracles", "model", "criteria")) -> set[float]:
    """Every number of the protocol the code is given as an input (not the checks' values, not the criteria built on
    them, not the worked examples)."""
    out: set[float] = set()
    if isinstance(node, dict):
        for k, v in node.items():
            if k not in skip:
                out |= _numbers(v, ())
    elif isinstance(node, list):
        for v in node:
            out |= _numbers(v, ())
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        out.add(float(node))
    return out


def _inputs(protocol: dict) -> set[float]:
    found = _numbers(protocol)
    for o in protocol.get("oracles") or []:
        found |= _numbers(o.get("case") or {}, ())
    return found


def _spellings(v: float) -> set[str]:
    out = {repr(v), str(v), f"{v:g}"}
    if v == int(v):
        out.add(str(int(v)))
    if 0 < abs(v) < 1:
        out.add(f"{v * 100:g}%")
    out |= {s.replace("e-0", "e-").replace("e+0", "e+") for s in out}   # 1e-06 is also written 1e-6
    return {s for s in out if s not in ("0", "0.0", "1", "1.0", "-0.0")}


def _hits(text: str, v: float) -> list[str]:
    return sorted(s for s in _spellings(v) if re.search(r"(?<![\w.])" + re.escape(s) + r"(?![\w])", text))


def _visible(tmp_path: Path, topic: str) -> dict[str, str]:
    """Everything the writer sees, by where it comes from."""
    protocol = json.loads((FIXTURES / topic / "protocol.json").read_text(encoding="utf-8"))
    state = {"design": {"hypothesis": "the study's hypothesis", "protocol": copy.deepcopy(protocol)}}
    shown = {"design block": Engine._code_design_block(state),
             "design block (outline)": Engine._code_design_block(state, names_only=True),
             "repair protocol": json.dumps(_stripped(protocol), indent=2)}
    root = tmp_path / topic
    shutil.copytree(FIXTURES / topic / "clean", root / "code")
    code = root / "code"
    package = next(p.name for p in code.iterdir() if p.is_dir() and (p / "__init__.py").is_file())
    code_layout.write_check_values(code, protocol)
    code_project.refresh(root, deps=["numpy"], protocol=protocol, title=topic, question="the question",
                         extra_files=code_layout.project_files(code, protocol, package),
                         readme_files=code_layout.readme_lines(package))
    sources = dict(code_layout.package_sources(code))
    sources["simulate.py"] = (code / "simulate.py").read_text(encoding="utf-8")
    cases = eqt.cases(eqt.example_rows(protocol), eqt.locate(protocol, sources))
    eqt.write(code, cases)
    coder = ac.AgentCoder(root, ac.ClaudeCliRunner(mode="host", work_root=tmp_path / "work"),
                          log=logging.getLogger("t"), startup=[])
    work = coder.runner.workdir(root)
    coder._fresh_copy(work)
    for path in sorted(work.rglob("*")):
        if path.is_file():
            shown["copy: " + path.relative_to(work).as_posix()] = path.read_text(encoding="utf-8", errors="replace")
    # What FI says back about a failed equation (never the number).
    for c in cases:
        status, detail, *_ = equation_measure.judge(c, {"status": "measured", "value": 1e9})
        shown[f"failure message {c['id']}"] = eqt.failure_message({"id": c["id"], "function": c["function"],
                                                                   "status": status, "detail": detail}, c["formula"])
    shown["measure spec"] = json.dumps(eqt.measure_spec(cases, nonce="n"))
    return shown


@pytest.mark.parametrize("topic", ["kinetics", "fdr"])
def test_the_writer_sees_no_expected_value_and_no_tolerance(tmp_path: Path, topic: str) -> None:
    protocol = json.loads((FIXTURES / topic / "protocol.json").read_text(encoding="utf-8"))
    shown = _visible(tmp_path, topic)
    inputs = _inputs(protocol)
    leaks, excused = [], set()
    values = []
    for o in protocol.get("oracles") or []:
        for which in ("expected", "tolerance"):
            v = o.get(which)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                values.append((o["name"], which, float(v)))
    for eq in (protocol.get("model") or {}).get("equations") or []:
        for row in eqt.example_rows({"model": {"equations": [eq]}}):
            if isinstance(row.get("expected"), (int, float)):
                values.append((f"equation {row['id']}", "example output", float(row["expected"])))
    assert values, "the fixture has checks with numbers"
    for name, which, v in values:
        if not _spellings(v):
            continue
        for where, text in shown.items():
            found = _hits(text, v)
            if not found:
                continue
            if v in inputs:
                excused.add((topic, name, which, v))
            elif FI_CONSTANTS.get((topic, name, which, v)) == where:
                excused.add((topic, name, which, v))
            else:
                leaks.append(f"{name} {which} {v}: {found} in {where}")
    assert leaks == [], "\n".join(leaks)
    assert excused == {e for e in {*EXCUSED, *FI_CONSTANTS} if e[0] == topic}, sorted(excused)


def test_the_check_prose_keeps_its_meaning_without_the_numbers() -> None:
    protocol = json.loads((FIXTURES / "kinetics" / "protocol.json").read_text(encoding="utf-8"))
    shown = _stripped(protocol)
    o = next(x for x in shown["oracles"] if x["name"] == "noise_free_lnA")
    assert set(o) >= {"name", "kind", "check", "case", "measure"}
    assert not set(o) & {"expected", "expected_formula", "tolerance", "tolerance_mode", "limit"}
    assert "recover ln A exactly" in o["check"] and "1e-6" not in o["check"]


MASKED = "[the value FI checks it against]"


def test_a_number_at_the_end_of_a_sentence_is_masked_too() -> None:
    """review pass 2 N3: ``(?![\\w.])`` after the number left "FWER should be 0.6415." as written."""
    protocol = {"grid": {"m": [20]}, "oracles": [
        {"name": "fwer", "kind": "special_case", "case": {"m": 20}, "expected": 0.6415, "tolerance": 0.015,
         "check": "FWER should be 0.6415. Within 0.015.", "reference": "1-(1-0.05)^20 = 0.6415"}]}
    shown = _stripped(protocol)
    text = json.dumps(shown)
    assert "0.6415" not in text and "0.015" not in text, text
    assert shown["oracles"][0]["check"] == f"FWER should be {MASKED}. Within {MASKED}."


def test_a_setting_the_code_is_given_is_never_masked_even_when_it_equals_a_checks_number() -> None:
    """review pass 2 N4: a setting written as text (a grid value, a fixed condition, a check's case, a value compared
    inside each run) that equals a tolerance or an expected value was masked, so the code writer could not know it."""
    protocol = {
        "grid": {"noise": ["0.25"]},
        "fixed": {"q": "0.01", "note": "Apply BH at q = 0.01 to the m=20 p-values"},
        "in_each_run": {"alpha": ["0.2", "0.5"]},
        "oracles": [
            {"name": "a", "kind": "special_case", "case": {"pi1": "0.3"}, "expected": 0.25, "tolerance": 0.01,
             "check": "with noise sd 0.25 and q = 0.01, the rate is 0.25"},
            {"name": "b", "kind": "special_case", "case": {"m": 20}, "expected": 0.3, "tolerance": 0.2,
             "check": "at pi1 0.3 the power is 0.3 within 0.2"},
            {"name": "c", "kind": "special_case", "case": {"m": 20}, "expected": 0.7385, "tolerance": 0.0031,
             "check": "c is 0.7385 within 0.0031"}]}
    shown = _stripped(protocol)
    assert shown["fixed"]["note"] == "Apply BH at q = 0.01 to the m=20 p-values"
    assert shown["oracles"][0]["check"] == "with noise sd 0.25 and q = 0.01, the rate is 0.25"
    assert shown["oracles"][1]["check"] == "at pi1 0.3 the power is 0.3 within 0.2"
    assert shown["oracles"][2]["check"] == f"c is {MASKED} within {MASKED}", "a number that is no setting is masked"
