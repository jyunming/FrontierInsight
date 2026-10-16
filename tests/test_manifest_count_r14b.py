"""FI's count of what the analysis reports, replayed on the r14b false-discovery quest, which stopped on it.

The quest's grid is m x pi1 (9 settings, 1500 runs each: 13,500 runs); each run applies three procedures to the same data
(``protocol.in_each_run``), so FI's record holds 40,500 values of ``fdr`` (one per run and procedure) and of ``power``
(27,000 with a value, 13,500 undefined: no real effect when pi1 = 0). The analysis prints each list under
``by_m/<m>/by_pi1/<pi1>/by_method/<procedure>``, every one identical, value for value, to FI's record of that setting and
procedure. FI counted 13,500 and 9,000 + 9,000: the three procedures of a setting were read as copies of one list, because
``by_method`` is not the name FI knows the factor by and the procedure's own name under it was not read as one of its
values; and an undefined value was counted twice when the list holds it as ``null`` and ``power_undefined`` counts it too
(FI's own trials.json keeps both the same way). FI sent experiment.py back three times and stopped.

``tests/fixtures/run_size/r14b_fdr/manifest_check.json`` keeps the check's inputs from the quest: the analysis's RESULT_JSON
(the final experiment.py run on FI's stored trials) with each per-trial list replaced by its counts, FI's per-setting record
as counts, the protocol and the manifest.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from core import run_manifest

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "run_size" / "r14b_fdr" / "manifest_check.json"


def _load() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _lists(node: Any) -> Any:
    """The RESULT_JSON back with lists of the recorded lengths (0.5 for a value, None for an undefined one)."""
    if isinstance(node, dict):
        if set(node) == {"__finite__", "__null__"}:
            return [0.5] * node["__finite__"] + [None] * node["__null__"]
        return {k: _lists(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_lists(v) for v in node]
    return node


def _inputs() -> tuple[dict[str, Any], dict[str, Any], Any, dict[str, Any], dict[str, Any]]:
    fx = _load()
    per_cell = {key: {name: Counter({0.5: d, math.nan: u}) for name, (d, u) in by.items()}
                for key, by in fx["recorded_by_cell"].items()}
    recorded: dict[str, Counter] = {}
    for by in per_cell.values():
        for name, c in by.items():
            recorded.setdefault(name, Counter()).update(c)
    manifest = {"successful_per_cell": fx["successful_per_cell"]}
    return fx["protocol"], manifest, _lists(fx["result_json"]), recorded, per_cell


def test_fis_record_holds_a_value_per_run_and_procedure() -> None:
    _protocol, _manifest, _rj, recorded, _per_cell = _inputs()
    assert run_manifest.recorded_counts(recorded, "fdr") == (40500, 0)
    assert run_manifest.recorded_counts(recorded, "power") == (27000, 13500)


def test_each_procedures_list_is_counted_as_its_own_and_a_null_is_not_counted_twice() -> None:
    protocol, _manifest, rj, recorded, per_cell = _inputs()
    for metric, want in (("fdr", (40500, 0)), ("power", (27000, 13500))):
        grid = run_manifest.metric_grid(protocol, metric, per_cell)
        assert "procedure" in grid
        defined, undefined, excesses, acc_defined, acc_undefined = run_manifest._listed_counts(
            rj, metric, grid, run_manifest.recorded_counts(recorded, metric), per_cell)
        assert (acc_defined, acc_undefined) == want, metric
        assert (defined, undefined) == want and excesses == [], metric


def test_the_r14b_analysis_passes_the_count_check() -> None:
    """Before: "the analysis accounts for 13500 value(s)" (fdr) and "9000 value(s) + 9000 undefined" (power)."""
    protocol, manifest, rj, recorded, per_cell = _inputs()
    found = run_manifest._value_count_findings(protocol, manifest, rj, trial_mode=True, recorded=recorded,
                                               recorded_by_cell=per_cell)
    assert found == []


def test_an_analysis_that_really_drops_two_procedures_is_still_caught() -> None:
    protocol, manifest, rj, recorded, per_cell = _inputs()
    for by_pi1 in rj["by_m"].values():
        for entry in by_pi1["by_pi1"].values():
            for procedure in ("bonferroni", "benjamini_hochberg"):
                entry["by_method"][procedure].pop("fdr_values")
    found = [s for s, _ in run_manifest._value_count_findings(protocol, manifest, rj, trial_mode=True, recorded=recorded,
                                                              recorded_by_cell=per_cell)]
    assert any("`fdr`" in s and "accounts for 13500" in s for s in found), found


def test_a_null_and_a_count_of_the_same_undefined_trials_are_one_account() -> None:
    both = {"power_values": [0.1, None, None], "power_undefined": 2}
    assert run_manifest._mapping_counts(both, "power") == (1, 2)
    only_count = {"power_values": [0.1], "power_undefined": 2}
    assert run_manifest._mapping_counts(only_count, "power") == (1, 2)
    only_null = {"power_values": [0.1, None, float("nan")]}
    assert run_manifest._mapping_counts(only_null, "power") == (1, 2)


def test_a_bare_name_of_a_value_compared_in_each_run_names_it() -> None:
    grid = {"m": [20, 200], "procedure": ["uncorrected", "bonferroni"]}
    path = ("by_m", "20", "by_method", "bonferroni", "fdr_values")
    assert run_manifest._path_cells(path, grid) == frozenset({("m", 20), ("procedure", "bonferroni")})
    # A number is not read this way (it could be an index): only a name that is one value of one factor.
    assert run_manifest._path_cells(("by_m", "20", "rows", "0"), {"m": [20, 200], "k": [0, 1]}) == frozenset({("m", 20)})


def test_by_default_a_difference_is_recorded_and_the_quest_goes_on(tmp_path: Path) -> None:
    """The default is 'warn': a difference the check finds is in QUALITY.md as recorded, not acted on (the engine goes on:
    ``tests/test_run_manifest.py`` follows a quest under 'warn' to its paper)."""
    from core import quality_report
    from core.config import EngineConfig

    assert EngineConfig().run_manifest_check == "warn"
    fi = tmp_path / ".fi"
    fi.mkdir()
    entry = {"kind": "check_result", "check": "run_manifest", "status": "warned", "seq": 7,
             "problems": ["the protocol fixes 1500 runs per setting; 1 setting(s) ran another number"]}
    (fi / "audit.jsonl").write_text(json.dumps(entry) + "\n", encoding="utf-8")
    found = [p for p in quality_report._run_problems(quality_report._Quest(tmp_path)) if "1500 runs" in p["text"]]
    assert len(found) == 1
    assert found[0]["found_by"] == "FI's run-record check (recorded only)" and found[0]["status"] == "not fixed"
