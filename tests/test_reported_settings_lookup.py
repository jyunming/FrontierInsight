"""core/reported_settings.py matches a printed row to the cells by lookup (sorted values, bisected), not by scanning every
trial value. The verdicts must be the ones the scan gave: the frozen scan below is the reference."""

from __future__ import annotations

import asyncio
import math
import random
import time
from typing import Any

from core import reported_settings as rs


# ---- the reference: core/reported_settings.py before the lookup, copied unchanged -----------------------------------------
def _reference_distinct(values: list[float]) -> list[float]:
    out: list[float] = []
    for v in values:
        if not any(rs.same(v, o) for o in out):
            out.append(v)
    return out


def _reference_candidates(row: dict[str, Any], cells: list[dict[str, Any]], skip: set[str]) -> list[dict[str, Any]] | None:
    shared = [k for k, v in row.items() if k not in skip and rs._is_number(v) and any(k in c["metrics"] for c in cells)]
    if len(shared) < rs.MIN_SHARED:
        return None
    matched = [c for c in cells if all(
        k not in c["metrics"] or any(rs.same(row[k], v) for v in c["metrics"][k]) or
        rs.same(row[k], sum(c["metrics"][k]) / len(c["metrics"][k])) for k in shared)
        and any(k in c["metrics"] for k in shared)]
    discriminating = [k for k in shared if len(_reference_distinct([v for c in cells for v in c["metrics"].get(k, [])])) > 1]
    if not matched or len(discriminating) < 1 or len(matched) > rs.MAX_CANDIDATES:
        return None
    return matched


def _reference_problems(cells: list[dict[str, Any]], reported: list[tuple[str, Any]]) -> list[rs.Mismatch]:
    varying = rs._varying(cells)
    if len(cells) < 3 or not varying:
        return []
    out: list[rs.Mismatch] = []
    for where, value in reported:
        for path, rows in rs.tables(value, where if where else ""):
            root = "" if where else path.split(".")[0].split("[")[0]
            columns = {}
            for k in {k for r in rows for k in r}:
                axis = next((a for a in varying if rs.names_setting(k, a)), None)
                if axis and all(rs._is_number(r.get(k)) for r in rows):
                    columns[k] = axis
            if not columns:
                continue
            for column, axis in sorted(columns.items()):
                verified = wrong = 0
                reported_values: list[float] = []
                ran_values: list[float] = []
                for row in rows:
                    near = _reference_candidates(row, cells, set(columns))
                    if near is None:
                        continue
                    verified += 1
                    actual = [float(c["cell"][axis]) for c in near if rs._is_number(c["cell"].get(axis))]
                    if actual and not any(rs.same(row[column], a) for a in actual):
                        wrong += 1
                        reported_values.append(float(row[column]))
                        ran_values.extend(actual)
                if verified and wrong >= max(1, math.ceil(0.5 * verified)):
                    out.append(rs.Mismatch(path, column, axis, _reference_distinct(reported_values),
                                           _reference_distinct(ran_values), len(rows), wrong, top=root))
                elif not verified and len(rows) in rs._per_record_kind(cells, axis):
                    values = _reference_distinct([float(r[column]) for r in rows])
                    if len(values) == 1:
                        out.append(rs.Mismatch(path, column, axis, values, varying[axis], len(rows), len(rows),
                                               collapsed=True, top=root))
    return out


# ---- inputs ----------------------------------------------------------------------------------------------------------------
def _random_cells(rng: random.Random) -> list[dict[str, Any]]:
    pool = [round(rng.uniform(-5, 5), rng.choice([0, 1, 3])) for _ in range(rng.randint(2, 6))]
    cells = []
    for a in range(rng.randint(1, 4)):
        for b in range(rng.randint(1, 3)):
            metrics = {}
            for name in rng.sample(["x", "y", "z", "w"], rng.randint(1, 4)):
                metrics[name] = [rng.choice(pool) * rng.choice([1.0, 1.0, 1.0 + 4e-10, 1.0 + 3e-9]) for _ in range(rng.randint(1, 5))]
            cells.append({"cell": {"rate": float(a + 1), "size": float(10 * (b + 1))}, "metrics": metrics})
    return cells


def _random_rows(rng: random.Random, cells: list[dict[str, Any]]) -> list[dict[str, float]]:
    rows = []
    for _ in range(rng.randint(2, 6)):
        c = rng.choice(cells)
        row: dict[str, float] = {}
        for name, values in c["metrics"].items():
            pick = rng.random()
            row[name] = (rng.choice(values) if pick < 0.4 else sum(values) / len(values) if pick < 0.6
                         else rng.choice(values) * (1 + rng.choice([5e-10, 2e-9, -2e-9])) if pick < 0.8
                         else round(rng.uniform(-5, 5), 2))
        if rng.random() < 0.3:
            row[rng.choice(["x", "y", "z", "w"])] = round(rng.uniform(-5, 5), 2)
        row["rate_hz"] = c["cell"]["rate"] if rng.random() < 0.5 else float(rng.randint(1, 4))
        row["size"] = c["cell"]["size"] if rng.random() < 0.7 else 10.0 * rng.randint(1, 3)
        rows.append(row)
    return rows


def _shown(found: list[rs.Mismatch]) -> list[Any]:
    return [(m.path, m.column, m.axis, m.reported, m.ran, m.rows, m.wrong, m.collapsed, m.top) for m in found]


def test_lookup_gives_the_verdicts_of_the_scan_on_random_small_tables() -> None:
    rng = random.Random(20261017)
    flagged = 0
    for _ in range(600):
        cells = _random_cells(rng)
        reported = [("", {"table": _random_rows(rng, cells)})]
        want = _reference_problems(cells, reported)
        got = rs.problems(cells, reported)
        assert _shown(got) == _shown(want)
        flagged += bool(want)
        for row in reported[0][1]["table"]:
            assert rs._candidates(row, cells, {"rate_hz", "size"}) == _reference_candidates(row, cells, {"rate_hz", "size"})
    assert flagged > 20  # the inputs include tables the check flags, so equality is not equality of empty lists


def test_a_printed_row_under_the_wrong_setting_is_still_reported() -> None:
    rng = random.Random(3)
    cells = [{"cell": {"rate": float(r)}, "metrics": {"x": [r + rng.random() for _ in range(300)],
                                                      "y": [10 * r + rng.random() for _ in range(300)]}} for r in range(1, 6)]
    right = [{"rate_hz": float(r), "x": sum(c["metrics"]["x"]) / 300, "y": sum(c["metrics"]["y"]) / 300}
             for r, c in zip(range(1, 6), cells)]
    wrong = [{**row, "rate_hz": 1.0} for row in right]
    assert rs.problems(cells, [("", {"t": right})]) == []
    found = rs.problems(cells, [("", {"t": wrong})])
    assert len(found) == 1 and found[0].wrong == 4
    assert _shown(found) == _shown(_reference_problems(cells, [("", {"t": wrong})]))


def test_many_trial_values_are_not_scanned_for_every_row() -> None:
    cells = [{"cell": {"rate": float(r), "size": float(s)},
              "metrics": {m: [r + s + i * 1e-3 for i in range(3000)] for m in ("a", "b")}}
             for r in range(1, 8) for s in (1, 2)]
    rows = [{"rate_hz": float(c["cell"]["rate"]), "size": 1.0, "a": c["metrics"]["a"][5], "b": c["metrics"]["b"][7]}
            for c in cells[::2]]
    t = time.perf_counter()
    rs.problems(cells, [("", {"t": rows})])
    assert time.perf_counter() - t < 2.0  # the scan took minutes at this size's neighbours


def test_the_result_check_runs_off_the_event_loop() -> None:
    from core.result_checks import ResultChecksMixin

    ticks: list[int] = []
    seen: list[int] = []

    class Quest(ResultChecksMixin):
        _trial_mode = True

        def _result_findings(self, state: Any, result_json: Any) -> list[Any]:
            time.sleep(0.3)
            seen.append(len(ticks))
            return []

        def __getattr__(self, name: str) -> Any:  # the rest of the engine is not needed for a run that did not finish
            raise AttributeError(name)

    async def beat() -> None:
        for _ in range(5):
            await asyncio.sleep(0.02)
            ticks.append(1)

    async def go() -> None:
        from core.execution import ExecutionResult

        quest = Quest()
        quest.quest_root = __import__("pathlib").Path(".pytest_tmp") / "offloop"  # type: ignore[attr-defined]
        quest._result_kept = []  # type: ignore[attr-defined]
        quest._code_reader_findings = lambda r: []  # type: ignore[attr-defined]
        quest._figure_gap_findings = lambda d: []  # type: ignore[attr-defined]
        quest._no_figures_standing = lambda s: []  # type: ignore[attr-defined]
        await asyncio.gather(quest._results_match_run({}, ExecutionResult(0, "", "", 0.1, False), None, False), beat())

    asyncio.run(go())
    assert len(ticks) == 5
    assert seen and seen[0] >= 3  # the loop kept beating while the check ran; on the loop it would be 0
