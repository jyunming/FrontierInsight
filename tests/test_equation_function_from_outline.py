"""Which function an equation test calls, and which parameters a repair must keep (core/equation_tests.py::locate,
Engine._outline_signatures). Replay of the shape of suite round d4 (DUV, quest 35167b).

The outline mapped E8 to ``coherence_first_zero(sigma, NA, lam) -> float``, which the first-written code computed
correctly. The model had also labelled a helper, ``coherence_zeros(sigma, NA, lam, kmax=1)`` (an array), ``# E8``, and
the test found the function by the label alone: it called the helper, got an array, and the repair rewrote the helper
into a scalar, breaking its two callers. The repair of that crash was then refused three times, because every equation
function was held to the outline's ``def`` and two functions no test calls (``abbe_image``, ``hopkins_image``) had been
first written with one more parameter than the outline gave. The outline names the tested function; a function no test
calls keeps the parameters it has in the code."""

from __future__ import annotations

from pathlib import Path

import pytest

from core import code_layout as cl, equation_tests as eqt
from tests.test_equation_gate import GOOD, OUTLINE, PROTOCOL, SPRING, _engine, _plan_md, _Writer, _write_code

# The outline's E2 function is `quality`; the model labelled a helper of it `# E2` (a list, like d4's array).
HELPER_LABELLED = ("def quality(m, w, c):\n    return quality_terms(m, w, c)[0]\n\n\n"
                   "# E2\ndef quality_terms(m, w, c, kmax=1):\n    return [m * w / c] * kmax\n")

ENERGY = {"id": "E3", "formula": "U = k * x ** 2 / 2", "role": "generates", "source": "derivation",
          "derivation": "work done on the spring", "example": {"untestable": "no worked example in this test"}}
PROTOCOL3 = {**PROTOCOL, "model": {**SPRING, "equations": [*SPRING["equations"], ENERGY]}}
OUTLINE3 = {**OUTLINE, "functions": [*OUTLINE["functions"], {
    "name": "energy", "signature": "def energy(k, x) -> float:", "one_line_purpose": "stored energy", "implements": "E3"}]}
# First written with one more parameter than the outline gave (d4's `abbe_image(..., o)`); no test calls it.
ENERGY_AS_WRITTEN = "\n\n# E3\ndef energy(k, x, o):\n    return o * k * x ** 2 / 2\n"
STATE3 = {"design": {"protocol": PROTOCOL3}, "implement_outline": OUTLINE3}


def test_the_outline_names_the_tested_function_and_a_label_is_for_an_equation_it_names_none_for() -> None:
    sources = {"spring/model.py": GOOD["frequency"] + "\n\n" + HELPER_LABELLED}
    by_label = eqt.locate(PROTOCOL, sources)
    assert by_label["E2"]["function"] == "quality_terms", "the label alone names the helper (the d4 fault)"
    by_outline = eqt.locate(PROTOCOL, sources, {"E1": "frequency", "E2": "quality"})
    assert by_outline["E2"] == {"module": "spring.model", "function": "quality", "file": "spring/model.py",
                                "static": eqt.static_facts(sources["spring/model.py"], "quality")}
    assert by_outline["E1"]["function"] == "frequency"
    assert eqt.locate(PROTOCOL, sources, {"e2": "quality"})["E2"]["function"] == "quality", "ids match in any case"
    # A function the outline names that the code does not have: the outline is still the one source, so the label on a
    # neighbour is not looked for (it used to be, and a test then called a function of another equation).
    assert eqt.locate(PROTOCOL, sources, {"E2": "q_factor"})["E2"] is None


@pytest.mark.asyncio
async def test_a_helper_labelled_with_the_equation_is_not_the_one_tested(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    _plan_md(engine)
    _write_code(engine, HELPER_LABELLED)
    writer = _Writer({})
    engine._client = writer
    assert await engine._equation_gate({"design": {"protocol": PROTOCOL}, "implement_outline": OUTLINE}, None, None) is None
    assert writer.prompts == [], "the outline's function computes E2 correctly: nothing is repaired"
    model = (engine.quest_root / "code" / "spring" / "model.py").read_text(encoding="utf-8")
    assert "def quality_terms(m, w, c, kmax=1):\n    return [m * w / c] * kmax" in model, "the helper is left as written"
    test_file = (engine.quest_root / "code" / eqt.TEST_PATH).read_text(encoding="utf-8")
    assert "quality_terms" not in test_file
    # Known positive: the same code without the outline is tested by the label, and the helper fails.
    engine2 = _engine(tmp_path / "no_outline")
    _plan_md(engine2)
    _write_code(engine2, HELPER_LABELLED)
    engine2._client = _Writer({"quality_terms": [HELPER_LABELLED.split("\n\n\n")[1]]})
    patch = await engine2._equation_gate({"design": {"protocol": PROTOCOL}}, None, None)
    assert patch is not None and "quality_terms" in patch["exec_result"]["stderr_tail"]


def test_a_function_no_test_calls_keeps_its_own_parameters_and_a_tested_one_the_outlines(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    _plan_md(engine, PROTOCOL3)
    _write_code(engine, GOOD["quality"] + ENERGY_AS_WRITTEN)
    cl.save(engine.quest_root, {"shape": cl.PACKAGE, "package": "spring"})
    fixed = engine._outline_signatures(STATE3)
    # Both tested functions are already written so that their tests can call them by the example's names: the def they
    # have in the code is the one interface (the outline's `-> float` annotation is no second claim on it).
    assert fixed["quality"] == "def quality(m, w, c):" and fixed["frequency"] == "def frequency(k, m):"
    assert fixed["energy"] == "def energy(k, x, o):", "untested: the parameters it was written with"
    model = GOOD["frequency"] + "\n\n" + GOOD["quality"]
    # The repair the quest needed: a fix elsewhere in the file, `energy` as it was written. Refused before this change.
    needed = {"package_files": {"spring/model.py": model.replace("m * w / c", "(m * w) / c") + ENERGY_AS_WRITTEN}}
    assert engine._apply_package_repair(STATE3, needed, node="execute_reflect") == ["spring/model.py"]
    # Planted: the untested function gets another parameter; the tested one loses one.
    for planted, name in ((ENERGY_AS_WRITTEN.replace("(k, x, o)", "(k, x, o, zz)"), "energy"),
                          (ENERGY_AS_WRITTEN, "quality")):
        text = model + planted
        if name == "quality":
            text = text.replace("def quality(m, w, c)", "def quality(m, w)")
        assert engine._apply_package_repair(STATE3, {"package_files": {"spring/model.py": text}},
                                            node="execute_reflect") == [], name
    log = (engine.fi_dir / "run.log").read_text(encoding="utf-8")
    assert "`energy` must keep the parameters (k, x, o)" in log and "`quality` must keep the parameters (m, w, c)" in log
