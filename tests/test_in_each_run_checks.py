"""The checks that read a run's numbers beyond the study itself understand what each run compares inside it
(``protocol.in_each_run``, core/in_run.py): FI's behaviour probes, the stuck and constant-value checks, the pilot's flat-output
check. A number per run is judged over the settings, a number per value over the settings and values."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

from core import degenerate_results as dr
from core import flat_output, in_run, probes as P, probes_run as PR, reported_settings as rs, run_estimate
from core.execution import SharedInterpreterExecutor

INSIDE = {"procedure": ["a", "b", "c"]}
PROTOCOL = {"grid": {"m": [10, 20, 30, 40]}, "in_each_run": INSIDE, "metrics": [{"id": "power"}, {"id": "total"}]}

SIM = '''
import random

def run_trial(cell, trial_id, seed):
    rng = random.Random(seed)
    assert cell["procedure"] == ["a", "b", "c"], cell      # FI gives the whole list
    noise = rng.random() * 0.01
    return {"power": {"a": cell["m"] * 0.1 + noise, "b": cell["m"] * 0.2 + noise, "c": cell["m"] * 0.3 + noise},
            "total": cell["m"] * 1.0 + noise}
'''
# `b` does not respond to the setting
B_FLAT = SIM.replace('"b": cell["m"] * 0.2 + noise', '"b": 5.0 + noise')
# nothing per value moves, and nothing per value is anywhere but 0
ZERO = SIM.replace('"a": cell["m"] * 0.1 + noise, "b": cell["m"] * 0.2 + noise, "c": cell["m"] * 0.3 + noise',
                   '"a": 0.0, "b": 0.0, "c": 0.0')


class Counting(SharedInterpreterExecutor):
    def __init__(self) -> None:
        super().__init__(python_version=f"{sys.version_info[0]}.{sys.version_info[1]}")
        self.processes = 0

    async def execute(self, *a: Any, **kw: Any) -> Any:
        self.processes += 1
        return await super().execute(*a, **kw)


def _probe(tmp_path: Path, source: str, relations: list[P.Relation]) -> tuple[dict[str, Any], Counting, Path]:
    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text(source, encoding="utf-8")
    ex = Counting()
    rec = asyncio.run(PR.run(ex, sys.executable, root, PROTOCOL, relations, confirm=False))
    return rec["results"][PR.code_sha(root)], ex, root


def _monotone(metric: str = "power") -> P.Relation:
    return P.Relation(id="R1", kind="monotone", metric=metric, setting="m", direction="up", quote="q", strength="core")


def test_a_probe_gives_the_simulation_the_whole_list_and_judges_a_number_per_value_at_each_value(tmp_path: Path) -> None:
    res, ex, _ = _probe(tmp_path, SIM, [
        _monotone(), P.Relation(id="R2", kind="monotone", metric="total", setting="m", direction="up", quote="q", strength="core")])
    assert res["R1"]["verdict"] == "held" and res["R2"]["verdict"] == "held"
    assert set(res["R1"]["numbers"]) == {"procedure=a", "procedure=b", "procedure=c"}   # judged at each value
    assert res["saturation"]["verdict"] == "held"
    assert ex.processes == 4        # one process per grid setting, not one per value


def test_a_value_the_setting_does_not_move_is_found_at_that_value(tmp_path: Path) -> None:
    res, _, _ = _probe(tmp_path, B_FLAT, [_monotone()])
    assert res["R1"]["verdict"] == "broken" and res["R1"]["says"].startswith("At procedure=b")


def test_a_number_per_value_stuck_at_zero_for_every_value_is_caught_by_the_probes(tmp_path: Path) -> None:
    res, _, _ = _probe(tmp_path, ZERO, [])
    assert res["saturation:power"]["verdict"] == "broken"
    assert "saturation:total" not in res        # the number per run moves


def _cells() -> list[dict[str, Any]]:
    """A trials.json as FI writes it for 4 settings x 3 values: ``total`` per run, ``power`` per value."""
    def metric(vals: list[float]) -> dict[str, Any]:
        return {"values": vals, "trials": list(range(len(vals))), "count": len(vals), "total": sum(vals), "non_finite": 0}

    out = []
    for m in (10, 20, 30, 40):
        out.append({"cell": {"m": m}, "key": f"m={m}", "metrics": {"total": metric([float(m), m + 0.1])},
                    "compared_in_each_run": ["procedure=a", "procedure=b", "procedure=c"]})
        for p in "abc":
            out.append({"cell": {"m": m, "procedure": p}, "key": f"m={m},procedure={p}", "level": {"procedure": p},
                        "metrics": {"power": metric([0.0, 0.0, 0.0, 0.0])}})
    return out


def _cells_with(power: Any) -> list[dict[str, Any]]:
    out = _cells()
    for c in out:
        if "level" in c:
            vals = power(c["cell"])
            c["metrics"]["power"] = {**c["metrics"]["power"], "values": vals, "count": len(vals)}
    return out


def test_the_stuck_check_judges_each_number_over_the_records_that_hold_it() -> None:
    cells = rs.cells_of({"cells": _cells()})
    stuck = P.saturation([{**c, "metrics": {k: v * 20 for k, v in c["metrics"].items()}} for c in cells])
    assert {v.relation_id: v.verdict for v in stuck} == {"saturation:power": "broken"}
    moving = rs.cells_of({"cells": _cells_with(lambda cell: [cell["m"] * 0.1 + 0.01 * (cell["procedure"] == "b")] * 3)})
    assert P.saturation(moving) == []


def test_the_constant_check_judges_each_number_over_the_records_that_hold_it() -> None:
    def cells(power: Any, other: Any) -> list[dict[str, Any]]:
        out = _cells_with(power)
        for c in out:
            if "level" in c:
                c["metrics"]["other"] = {**c["metrics"]["power"], "values": other(c["cell"])}
        return rs.cells_of({"cells": out})

    reported = {"power", "other", "total"}
    # a number per value that never changes while another per value does: caught, though only the value records hold it
    found = dr.quantity_problems(cells(lambda c: [0.3, 0.3], lambda c: [c["m"] * 1.0] * 2), reported)
    assert [(q.name, q.kind) for q in found] == [("power", "constant")]
    # a number per value that moves with the setting, and a number per run that moves: fine
    assert dr.quantity_problems(cells(lambda c: [c["m"] * 0.1] * 2, lambda c: [c["m"] * 1.0] * 2), reported) == []
    # a number that only differs between the values is moving too
    assert dr.quantity_problems(cells(lambda c: [ord(c["procedure"]) * 1.0] * 2, lambda c: [ord(c["procedure"]) * 2.0] * 2),
                                reported) == []


def test_a_plain_study_is_one_group_as_it_was() -> None:
    cells = [{"cell": {"m": m}, "metrics": {"x": [1.0, 2.0]}} for m in (1, 2, 3)]
    assert in_run.groups(cells) == [cells]
    assert in_run.groups([]) == [[]]
    assert [v.relation_id for v in P.saturation(cells)] == ["saturation:x"]      # flat over 3 settings, and alone
    assert dr.quantity_problems(cells, {"x"}) == []


def test_stored_cells_keep_the_value_a_record_belongs_to(tmp_path: Path) -> None:
    import json
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "trials.json").write_text(json.dumps({"cells": _cells()}), encoding="utf-8")
    stored = PR.stored_cells(tmp_path)
    assert sum(1 for c in stored if c.get("level")) == 12 and len(stored) == 16


# --- the pilot ----------------------------------------------------------------------------------------------------------


class _Cell:
    def __init__(self, cell: dict[str, Any], values: list[dict[str, Any]], levels: list[dict[str, Any]]) -> None:
        self.cell, self.levels = cell, levels
        self.rows = [{"status": "ok", "values": v} for v in values]


def test_flat_output_judges_a_number_per_run_and_a_number_per_value_each_over_its_own_cells() -> None:
    levels = in_run.level_cells(INSIDE)
    flat_all = [_Cell({"m": m}, [{"total": 3.0, "power": {"procedure=a": 1.0, "procedure=b": 1.0, "procedure=c": 1.0}}] * 2,
                      levels) for m in (1, 2, 3)]
    seen = flat_output.observed_flat_sweep(flat_all)
    assert seen is not None and seen.names == ["power", "total"] and seen.varied == ["m", "procedure"]
    moves = [_Cell({"m": m}, [{"total": 3.0, "power": {"procedure=a": m * 1.0, "procedure=b": 1.0, "procedure=c": 1.0}}] * 2,
                   levels) for m in (1, 2, 3)]
    assert flat_output.observed_flat_sweep(moves) is None       # a number per value moves: not flat


def test_the_pilot_counts_settings_not_values_and_names_a_level_that_reads_as_a_number() -> None:
    obs = [({"m": m}, {"total": 1.0}) for m in (1, 2)] + [({"m": m, "procedure": p}, {"power": 1.0}) for m in (1, 2) for p in "ab"]
    assert run_estimate._setting_count(obs) == 6 and run_estimate._setting_count(obs, ["procedure"]) == 2
    assert in_run.parse_level_key("method=nan,T=300.5") == {"method": "nan", "T": 300.5}
