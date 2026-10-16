"""A shown fault whose test errors in its own code is asked for once more, as an assertion about values (core/finding_repair.py
``ask_again``, ``RunChecksMixin._ask_finding_test_again``, called by ``Engine._shown_fault_check``).

Replay of r7 (EUV, a dose-to-size typed into the analysis: ``nominal = 60.0 if platform == "CAR" else 40.0`` and an
"interval" of ``[nominal, nominal]``). Its fault F-bed59bff was shown by a test, and on the code that ran its test
(``F-bed59bff_2.py``, kept as the quest wrote it) raises ``AttributeError: 'function' object has no attribute 'co_names'``
in its own code. The old gate counted that error as the fault and stopped; the gate that only takes an assertion lost the
stop. Now FI asks the code model once more, shown the error. The reply below stands in for the model's answer (no model is
called): a test that runs the analysis on the study's own trials and asserts on the interval it reports. It fails by its
assertion, so the quest stops as before, on a test that shows the fault by its values.

Fixture tests/fixtures/replay/finding_v2/r7_final/: ``code/experiment.py`` and the kept test as the quest left them,
``raw/trials.json`` cut to the first 40 trials of each of its 40 settings (``non_finite`` counted again); the code-review
record and the results are tests/fixtures/shown_fault_names/euv_r7/."""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from core import code_findings, code_review
from core.engine import Engine

FIX = Path(__file__).parent / "fixtures"

REPLY = '''```python
import json
import os
import tempfile
from pathlib import Path


def test_dose_to_size_interval_comes_from_the_data():
    trials = Path("raw/trials.json").resolve()
    here = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            os.environ["FI_TRIALS"] = str(trials)
            import experiment
            experiment.main()
            out = json.loads(Path("data/results/results.json").read_text(encoding="utf-8"))
        finally:
            os.chdir(here)
    for platform, pdata in out["by_resist_platform"].items():
        for pattern, rows in pdata["by_pattern"].items():
            lo, hi = rows["dose_to_size_mJ_cm2"]["uncertainty_95_ci_mJ_cm2"]
            assert hi > lo, f"{platform} {pattern}: the dose-to-size interval is [{lo}, {hi}], one value typed in"
```'''


@pytest.mark.usefixtures("this_python")
def test_replay_r7_the_test_asked_for_again_fails_by_its_assertion_and_the_quest_still_stops(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("scipy")
    pytest.importorskip("matplotlib")
    from tests.test_disclosure import _engine

    root = tmp_path / "r7"
    shutil.copytree(FIX / "replay" / "finding_v2" / "r7_final", root)
    shutil.copytree(FIX / "shown_fault_names" / "euv_r7", root, dirs_exist_ok=True)
    record = code_review.load(root)
    for issue in record["issues"].values():  # only F-bed59bff is replayed; it was shown by its test
        issue.pop("failure_kind", None)
    record["issues"]["F-bed59bff"]["failure_kind"] = code_findings.ASSERTION
    code_review.save(root, record)
    assert [i["id"] for i in code_findings.shown_open(record)] == ["F-bed59bff"]
    asked: list[str] = []

    async def chat(self, prompt, *a, **kw):  # noqa: ANN001, ANN002, ANN003
        asked.append(str(prompt))
        return REPLY

    monkeypatch.setattr(Engine, "_chat", chat)
    eng = _engine(root)
    result: dict[str, Any] = json.loads((root / "result_json.json").read_text(encoding="utf-8"))
    verdict = asyncio.run(eng._shown_fault_verdict(  # type: ignore[arg-type]
        {"exec_result": {"returncode": 0}, "result_json": result}))
    assert len(asked) == 1 and "co_names" in asked[0]
    assert verdict is not None and verdict["stuck_reason"] == "shown_fault_open"
    assert "asserting on the values" in verdict["rationale"]
    issue = code_review.load(root)["issues"]["F-bed59bff"]
    assert issue["asked_again"]["outcome"] == "fail" and issue["test"] == ".fi/finding_tests/F-bed59bff_3.py"
    assert "one value typed in" in issue["failure"]
    # the test ran the analysis in a folder of its own: the quest's results are not rewritten
    assert not (root / "data" / "results" / "results.json").exists()
