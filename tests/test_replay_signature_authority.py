"""Replay of suite round r12, first_order_kinetics (quest 1791991314): the parameters of an equation's function moved
back and forth between repairs, and the quest stopped without ever running the study.

The outline fixed ``arrhenius_parameters(alpha, beta, R, mean_invT)``. The whole-script repair rewrote the package with
other parameters (``T_i``, then ``covariance``); the repair of the one function was then held to the parameters the
function had *now*, not to the outline's, and the next whole-script repair changed them again (run.log 19:51-20:13:
"the function's parameters must be exactly (alpha, beta, R, T_i), as the outline fixed them; the reply has (alpha, beta,
R, mean_invT)", then the reverse, then ``covariance`` added and removed). The outline is the one authority: a repair of
the function is held to it, and a package file a whole-script repair gives back that changes it is not used."""

from __future__ import annotations

from pathlib import Path

import pytest

from core import code_layout as cl, function_steps as fs
from tests.test_equation_gate import GOOD, OUTLINE, STATE, _engine, _plan_md, _Writer, _write_code

R12_OUTLINE_SIG = "def arrhenius_parameters(alpha: float, beta: float, R: float, mean_invT: float) -> dict:"
R12_REPAIRED = ("import numpy as np\n\n\n# E4\ndef arrhenius_parameters(alpha: float, beta: float, covariance: np.ndarray,\n"
                "                         mean_invT: float, R: float) -> dict:\n"
                "    return {'Ea_hat': -R * beta, 'ln_A_hat': alpha - beta * mean_invT}\n")
R12_KEPT = ("# E4\ndef arrhenius_parameters(alpha, beta, R, mean_invT):\n"
            "    return {'Ea_hat': -R * beta, 'ln_A_hat': alpha - beta * mean_invT}\n")

WITH_OUTLINE = {**STATE, "implement_outline": OUTLINE}
DRIFTED_Q = "# E2\ndef quality(m, w, c, scale):\n    return scale * w / c\n"  # a whole-script repair added `scale`


def test_a_changed_parameter_list_is_named_against_the_outline_and_an_unchanged_one_is_not() -> None:
    fixed = {"arrhenius_parameters": R12_OUTLINE_SIG}
    changed = fs.changed_signatures(R12_REPAIRED, fixed)
    assert changed == ["`arrhenius_parameters` must keep the parameters (alpha, beta, R, mean_invT) the outline fixed; "
                       "the repair gave (alpha, beta, covariance, mean_invT, R)"]
    assert fs.changed_signatures(R12_KEPT, fixed) == []
    assert fs.changed_signatures("def other(x):\n    return x\n", fixed) == [], "a function the file does not define"


@pytest.mark.asyncio
async def test_the_repair_of_one_function_is_held_to_the_outline_not_to_what_an_earlier_repair_left(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    _plan_md(engine)
    _write_code(engine, DRIFTED_Q)
    writer = _Writer({"quality": [GOOD["quality"]]})  # the reply has the outline's parameters (m, w, c)
    engine._client = writer
    assert await engine._equation_gate(WITH_OUTLINE, None, None) is None
    model = (engine.quest_root / "code" / "spring" / "model.py").read_text(encoding="utf-8")
    assert "def quality(m, w, c):" in model and "scale" not in model
    log = (engine.fi_dir / "run.log").read_text(encoding="utf-8")
    assert "as the outline fixed them" not in log, "the reply with the outline's parameters is used"


@pytest.mark.asyncio
async def test_without_an_outline_the_repair_keeps_the_parameters_the_function_has(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    _plan_md(engine)
    _write_code(engine, DRIFTED_Q)
    engine._client = _Writer({"quality": [GOOD["quality"]]})
    await engine._equation_gate(STATE, None, None)
    log = (engine.fi_dir / "run.log").read_text(encoding="utf-8")
    assert "must be exactly (m, w, c, scale)" in log, "as before: no outline, the function's own parameters"


def test_a_whole_script_repair_that_changes_an_outline_signature_is_not_used_and_the_request_names_them(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    _plan_md(engine)
    _write_code(engine, GOOD["quality"])
    cl.save(engine.quest_root, {"shape": cl.PACKAGE, "package": "spring"})  # decided at plan time
    good_file = GOOD["frequency"] + "\n\n" + GOOD["quality"].replace("m * w / c", "(m * w) / c")
    parsed = {"package_files": {"spring/model.py": GOOD["frequency"] + "\n\n" + DRIFTED_Q}}
    assert engine._package_repair_files(WITH_OUTLINE, parsed) == {}
    assert engine._apply_package_repair(WITH_OUTLINE, parsed, node="execute_reflect") == []
    assert "scale" not in (engine.quest_root / "code" / "spring" / "model.py").read_text(encoding="utf-8")
    log = (engine.fi_dir / "run.log").read_text(encoding="utf-8")
    assert "`quality` must keep the parameters (m, w, c) the outline fixed; the repair gave (m, w, c, scale)" in log
    # A repair that keeps the parameters is used as before.
    kept = {"package_files": {"spring/model.py": good_file}}
    assert engine._apply_package_repair(WITH_OUTLINE, kept, node="execute_reflect") == ["spring/model.py"]
    note = engine._package_repair_note(WITH_OUTLINE)
    assert "`def frequency(k, m) -> float`" in note and "`def quality(m, w, c) -> float`" in note
    # Without an outline nothing is held: the note and the repair are as they were.
    assert "Keep the parameters of these functions" not in engine._package_repair_note(STATE)
    assert engine._package_repair_files(STATE, parsed) == {"spring/model.py": parsed["package_files"]["spring/model.py"].strip("\n") + "\n"}
