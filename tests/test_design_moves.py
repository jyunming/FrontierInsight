"""FI moves each design variable once before a search (core/design_moves.py): a number to its low and its high, a choice
to up to two other values, an array whole, each in a part of the study that does not hold it; movement judged on the
objective and on each limit on its own; no change at two points is a disclosure, never a stop. Neutral made-up study."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from core import design_moves as dm
from core.execution import VenvExecutor


def _block(**extra: Any) -> dict[str, Any]:
    block = {
        "objective": {"quantity": "loss", "direction": "minimise"},
        "constraints": [{"quantity": "cost", "limit": "<= 10"}],
        "design_variables": [
            {"name": "width", "kind": "continuous", "low": 1.0, "high": 5.0},
            {"name": "unused", "kind": "continuous", "low": 0.0, "high": 1.0},
            {"name": "mode", "kind": "choice", "values": ["plain", "ribbed", "folded", "hollow"]},
            {"name": "weights", "kind": "continuous", "shape": [2], "low": 0.0, "high": 1.0},
        ],
        "baseline": {"values": {"width": 2.0, "unused": 0.5, "mode": "plain", "weights": [0.5, 0.5]}},
        "fixed": {"climate": "dry"},
    }
    block.update(extra)
    return block


def _model(cell: dict[str, Any]) -> dict[str, Any]:
    """``loss`` uses ``width`` and ``weights``; ``mode`` changes only ``cost``; ``unused`` is read by nothing."""
    loss = (cell["width"] - 2.5) ** 2 + sum(cell["weights"]) + (1.0 if cell.get("climate") == "wet" else 0.0)
    cost = 2.0 + {"plain": 0.0, "ribbed": 1.0, "folded": 20.0, "hollow": 3.0}[cell["mode"]]
    return {"loss": loss, "cost": cost, "spare": cell["unused"]}  # `spare` moves with `unused`: never judged


class _Calls:
    """The designs asked for at once; the first ``reached`` of them answered, the rest cut off by the time."""

    def __init__(self, model: Any = _model, reached: int | None = None) -> None:
        self.cells: list[dict[str, Any]] = []
        self.model = model
        self.reached = reached

    async def __call__(self, cells: list[dict[str, Any]], budget: float) -> list[dict[str, Any]]:
        out = []
        for cell in cells:
            self.cells.append(dict(cell))
            if self.reached is not None and len(out) >= self.reached:
                out.append({"stop": "time"})
                continue
            try:
                out.append({"values": self.model(cell), "elapsed_s": 0.01})
            except Exception as e:  # noqa: BLE001 - a design the simulation cannot answer for
                out.append({"failed": f"the simulation raised {e}", "elapsed_s": 0.01})
        return out


def _measure(block: dict[str, Any], calls: _Calls) -> dm.Moves:
    return asyncio.run(dm.measure(block, calls, budget_s=100.0))


def _by(moves: dm.Moves) -> dict[str, dm.Variable]:
    return {v.name: v for v in moves.variables}


def test_a_variable_that_moves_the_objective_is_said_to_and_an_unread_one_is_not() -> None:
    calls = _Calls()
    moves = _measure(_block(), calls)
    got = _by(moves)
    assert got["width"].verdict == "moves"
    assert [e.objective for e in got["width"].ends] == [3.25, 7.25]  # (1-2.5)^2 + 1 and (5-2.5)^2 + 1, against 1.25
    # `unused` moves `spare`, a number the objective and the limits do not read: no change of the objective
    assert got["unused"].verdict == "no_change"
    assert "`unused`" in moves.notice() and "`width`" not in moves.notice()
    assert "stays in the search" in moves.notice()
    assert got["weights"].verdict == "moves"
    # one evaluation per distinct design: the baseline once, then two ends of each variable
    assert moves.evaluations == len(calls.cells) == len({json.dumps(c, sort_keys=True) for c in calls.cells}) == 1 + 2 + 2 + 2 + 2


def test_a_flat_objective_is_said_even_when_a_limit_moves() -> None:
    moves = _measure(_block(), _Calls())
    mode = _by(moves)["mode"]
    assert mode.verdict == "no_change"
    assert [e.label for e in mode.ends] == ["at ribbed", "at folded"]  # two other values, in the plan's order
    assert mode.moved_constraints == ["cost"]
    sentence = mode.sentence("loss")
    assert "no change of the objective `loss`" in sentence and "`cost`" in sentence and "moved" in sentence
    # `folded` breaks the limit: the design was still judged, and the broken limit is said in the record
    assert mode.ends[1].feasible is False and "breaks a limit" in mode.ends[1].reason


def test_an_array_moves_as_a_whole_to_its_low_and_its_high() -> None:
    calls = _Calls()
    _measure(_block(), calls)
    arrays = [c["weights"] for c in calls.cells]
    assert [0.0, 0.0] in arrays and [1.0, 1.0] in arrays
    assert not any(a not in ([0.5, 0.5], [0.0, 0.0], [1.0, 1.0]) for a in arrays)  # never one element alone


def test_an_end_at_the_starting_value_is_not_moved_and_costs_nothing() -> None:
    block = _block()
    block["baseline"]["values"]["width"] = 1.0  # its low
    moves = _measure(block, _Calls())
    width = _by(moves)["width"]
    assert [e.status for e in width.ends] == ["not_moved", "tested"]
    assert width.verdict == "moves"
    assert moves.evaluations == 1 + 1 + 2 + 2 + 2
    flat = _block(design_variables=[{"name": "width", "kind": "continuous", "low": 2.0, "high": 2.0}])
    flat["baseline"]["values"] = {"width": 2.0}
    only = _measure(flat, _Calls())
    assert only.variables[0].verdict == "not_moved" and "not moved" in only.variables[0].sentence("loss")


def test_a_variable_is_moved_in_a_part_that_does_not_hold_it_with_that_part_s_conditions() -> None:
    block = _block(grid={"climate": ["dry", "wet"], "mode": ["plain", "ribbed", "folded", "hollow"],
                         "unused": [0.2, 0.5]},
                   sub_studies=[{"name": "dry part", "vary": ["mode"], "hold": {"climate": "dry", "unused": 0.5}},
                                {"name": "wet part", "vary": ["climate"], "hold": {"mode": "hollow"}}])
    calls = _Calls()
    moves = _measure(block, calls)
    got = _by(moves)
    assert got["mode"].part == "dry part"  # the part whose `vary` names it
    assert got["unused"].part == "wet part"  # the first part that does not hold it
    unused_cells = [c for c in calls.cells if c["unused"] in (0.0, 1.0)]
    assert unused_cells and all(c["mode"] == "hollow" for c in unused_cells)  # what the wet part holds
    # the wet part's starting design (mode held at hollow) is evaluated, so each move is compared within its part
    assert any(c["mode"] == "hollow" and c["unused"] == 0.5 and c["width"] == 2.0 for c in calls.cells)
    assert got["unused"].start.objective == 1.25 and "in the part `wet part`" in got["unused"].sentence("loss")


def test_a_part_without_a_name_is_named_by_its_place_and_a_hold_the_search_never_hands_stays_out() -> None:
    block = _block(grid={"climate": ["dry", "wet"], "unused": [0.2, 0.5], "season": ["spring", "autumn"]},
                   sub_studies=[{"vary": ["climate"], "hold": {"unused": 0.5, "season": "autumn"}},
                                {"name": "second", "vary": ["unused"], "hold": {"climate": "wet"}}])
    calls = _Calls()
    got = _by(_measure(block, calls))
    assert got["width"].part == "sub-study 1" and "(in the part `sub-study 1`)" in got["width"].sentence("loss")
    assert not any("season" in c for c in calls.cells)  # not in `fixed`: the search never hands it to the simulation
    assert got["unused"].part == "second" and all(c["climate"] == "wet" for c in calls.cells if c["unused"] in (0.0, 1.0))


def test_a_variable_every_part_holds_is_not_moved() -> None:
    block = _block(grid={"climate": ["dry", "wet"], "unused": [0.2, 0.5]},
                   sub_studies=[{"name": "one", "vary": ["climate"], "hold": {"unused": 0.5}},
                                {"name": "two", "vary": ["climate"], "hold": {"unused": 0.2}}])
    calls = _Calls()
    moves = _measure(block, calls)
    unused = _by(moves)["unused"]
    assert unused.held and unused.verdict == "not_moved" and "every part" in unused.sentence("loss")
    assert not any(c["unused"] in (0.0, 1.0) for c in calls.cells)


def test_a_refused_design_is_not_tested_and_an_answer_without_an_objective_is_said() -> None:
    def refuses_wide(cell: dict[str, Any]) -> dict[str, Any]:
        if cell["width"] > 4.0:
            raise ValueError("too wide")
        return _model(cell)

    width = _by(_measure(_block(), _Calls(refuses_wide)))["width"]
    assert [e.status for e in width.ends] == ["tested", "not_tested"]  # a design the simulation refuses is not tested
    assert "too wide" in width.ends[1].reason and "not tested at its high" in width.sentence("loss")
    assert width.verdict == "moves"  # its low still moved the objective

    def refuses_all_zero(cell: dict[str, Any]) -> dict[str, Any]:
        if not any(cell["weights"]):
            raise ValueError("the weights are all zero")
        return {**_model(cell), "loss": 1.0}  # the objective reads no weight

    weights = _by(_measure(_block(), _Calls(refuses_all_zero)))["weights"]
    assert weights.verdict == "no_change"  # a refused end is no sign that the variable acts

    def nothing_at_the_high(cell: dict[str, Any]) -> dict[str, Any]:
        return {**_model(cell), "loss": float("nan")} if cell["width"] == 5.0 else _model(cell)

    width = _by(_measure(_block(), _Calls(nothing_at_the_high)))["width"]
    assert width.ends[1].status == "no_objective" and "no objective at its high" in width.sentence("loss")

    def no_objective(cell: dict[str, Any]) -> dict[str, Any]:
        return {"cost": 1.0}

    moves = _measure(_block(), _Calls(no_objective))
    assert "not the objective `loss`" in moves.baseline_reason
    assert all(v.verdict == "not_tested" for v in moves.variables)
    assert moves.notice() == ""  # nothing is said to change nothing when nothing could be judged


def test_an_evaluation_stopped_from_outside_counts_for_nothing() -> None:
    class _Killed(_Calls):
        async def __call__(self, cells: list[dict[str, Any]], budget: float) -> list[dict[str, Any]]:
            out = await super().__call__(cells, budget)
            return [{"failed": "the evaluation's process stopped (exit code 9) before it reported", "process": True}
                    if c["width"] == 5.0 else a for c, a in zip(cells, out)]

    width = _by(_measure(_block(), _Killed()))["width"]
    assert width.ends[1].status == "not_tested" and width.verdict == "moves"  # its low still moved it
    flat = _block()
    flat["design_variables"][0]["low"] = 2.0  # its low is now its starting value: only its high is left, and it was stopped
    width = _by(_measure(flat, _Killed()))["width"]
    assert width.verdict == "not_tested" and "not tested" in width.sentence("loss")


def test_a_starting_design_without_an_objective_is_judged_with_its_ends() -> None:
    def no_value_at_the_start(cell: dict[str, Any]) -> dict[str, Any]:
        got = _model(cell)
        if cell["width"] == 2.0:  # every starting design: the objective is not a finite number there
            got["loss"] = float("nan")
        return got

    got = _by(_measure(_block(), _Calls(no_value_at_the_start)))
    assert got["width"].start.status == "no_objective" and got["width"].verdict == "moves"  # 3.25 and 7.25 at its ends
    assert "no objective at its starting design" in got["width"].sentence("loss")
    # every design of `unused` has width 2.0: none gave an objective, so it cannot be judged
    assert got["unused"].verdict == "not_tested" and "could not be judged" in got["unused"].sentence("loss")


def test_a_variable_the_time_did_not_reach_is_not_measured() -> None:
    moves = _measure(_block(), _Calls(reached=5))  # the baseline, width's two ends, unused's two ends
    got = _by(moves)
    assert got["width"].verdict == "moves" and got["unused"].verdict == "no_change"
    assert got["mode"].verdict == got["weights"].verdict == "not_measured"
    assert "ran out" in got["mode"].sentence("loss") and "were not measured" in moves.says()
    assert moves.evaluations == 5


def test_disclose_writes_the_record_the_plan_line_and_the_reader_fact_once(tmp_path: Path, monkeypatch: Any) -> None:
    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text("def run_cell(cell):\n    return {}\n", encoding="utf-8")
    (root / "plan.md").write_text("# Plan\n\n## Checks already made\n\n- nothing yet\n", encoding="utf-8")
    runs: list[int] = []

    async def fake_run(executor: Any, python: Any, quest_root: Path, module: str, block: dict[str, Any],
                       **kwargs: Any) -> dm.Moves:
        runs.append(1)
        assert kwargs["at_once"] >= 1
        return await dm.measure(block, _Calls(), budget_s=kwargs["timeout_s"])

    monkeypatch.setattr(dm, "run", fake_run)
    logged: list[str] = []

    class _Log:
        def __getattr__(self, name: str) -> Any:
            return lambda msg, *a, **k: logged.append(msg % a if a else msg)

    class _Exec:
        timeout_s = 600
        settings_at_once = 2

    class _Cfg:
        execution = _Exec()

    class _Engine:
        quest_root = root
        executor = None
        config = _Cfg()
        _log = _Log()
        _small_run_note = "FI ran the simulation with its own harness before this reading."

        def _protocol_block(self, state: Any) -> dict[str, Any]:
            return {"optimisation": _block()}

    engine = _Engine()
    asyncio.run(dm.disclose(engine, {}, sys.executable, {}, {"key": "code-a", "crashed": False}))
    record = dm.read_record(root)
    assert record["key"] and record["variables"][1]["verdict"] == "no_change"
    assert "`unused`" in (root / "plan.md").read_text(encoding="utf-8")
    assert "`unused`: no change of the objective" in engine._small_run_note
    assert any("moved each design variable" in line for line in logged)
    # the same code and block: the record is read back, nothing runs again, nothing is added twice
    asyncio.run(dm.disclose(engine, {}, sys.executable, {}, {"key": "code-a", "crashed": False}))
    assert len(runs) == 1 and engine._small_run_note.count("`unused`: no change") == 1
    assert (root / "plan.md").read_text(encoding="utf-8").count("`unused`") == 1
    # a crashed small run: the pass does not run
    asyncio.run(dm.disclose(engine, {}, sys.executable, {}, {"key": "code-b", "crashed": True}))
    assert len(runs) == 1


def test_the_small_run_gate_calls_the_pass(monkeypatch: Any) -> None:
    from core.run_checks import RunChecksMixin

    seen: list[Any] = []

    async def fake_disclose(engine: Any, state: Any, py: Any, env: Any, record: Any) -> None:
        seen.append(record)

    monkeypatch.setattr(dm, "disclose", fake_disclose)

    class _Engine:
        quest_root = Path(".")

        async def _small_run(self, state: Any, py: Any, env: Any) -> dict[str, Any]:
            return {"key": "k", "crashed": False}

        def _unread_in_small_run(self, state: Any, record: Any, simulate: Any) -> None:
            return None

    assert asyncio.run(RunChecksMixin._small_run_gate(_Engine(), {}, "py", {})) is None
    assert seen == [{"key": "k", "crashed": False}]


SIM = '''
def run_cell(cell):
    return {"loss": (cell["width"] - 3.0) ** 2, "cost": 1.0, "spare": cell["unused"]}
'''


class _Ex(VenvExecutor):
    def python_path(self, root: Path) -> Path:
        return Path(sys.executable)


def test_the_pass_runs_through_the_search_s_own_entry_and_leaves_the_quest_as_it_was(tmp_path: Path) -> None:
    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text(SIM, encoding="utf-8")
    before = sorted(p.relative_to(root).as_posix() for p in root.rglob("*"))
    block = _block(design_variables=[{"name": "width", "kind": "continuous", "low": 1.0, "high": 5.0},
                                     {"name": "unused", "kind": "continuous", "low": 0.0, "high": 1.0}])
    block["baseline"]["values"] = {"width": 2.0, "unused": 0.5}
    moves = asyncio.run(dm.run(_Ex(), sys.executable, root, "code/simulate.py", block, timeout_s=120, env=None,
                               thresholds=None, at_once=3))
    assert moves is not None
    got = _by(moves)
    assert got["width"].verdict == "moves" and got["unused"].verdict == "no_change"
    assert moves.evaluations == 5 and len(moves.elapsed) == 5
    assert sorted(p.relative_to(root).as_posix() for p in root.rglob("*")) == before


def _two_variable_block() -> dict[str, Any]:
    block = _block(design_variables=[{"name": "width", "kind": "continuous", "low": 1.0, "high": 5.0}])
    block["baseline"]["values"] = {"width": 2.0}
    return block


def test_a_missing_limit_never_hides_a_readable_objective() -> None:
    def no_limit_at_the_high(cell: dict[str, Any]) -> dict[str, Any]:
        return {"loss": 1.0} if cell["width"] == 5.0 else {"loss": 1.0, "cost": 2.0}

    width = _measure(_two_variable_block(), _Calls(no_limit_at_the_high)).variables[0]
    assert width.ends[1].status == "tested" and width.ends[1].objective == 1.0
    assert width.verdict == "no_change"  # the objective is 1.0 at every design: a missing limit is not a moved objective


def test_a_limit_is_judged_even_where_the_objective_is_missing() -> None:
    def no_objective_at_the_high(cell: dict[str, Any]) -> dict[str, Any]:
        return {"cost": 8.0} if cell["width"] == 5.0 else {"loss": 1.0, "cost": 2.0}

    width = _measure(_two_variable_block(), _Calls(no_objective_at_the_high)).variables[0]
    assert width.ends[1].status == "no_objective"
    assert width.moved_constraints == ["cost"]


def test_the_plan_line_names_only_designs_that_were_evaluated() -> None:
    moves = _measure(_block(), _Calls(reached=4))  # unused's high end not reached
    unused = _by(moves)["unused"]
    assert unused.verdict == "no_change" and unused.ends[1].status == "not_measured"
    assert "at the designs FI could evaluate" in moves.notice() and "two ends" not in moves.notice()


def test_a_plan_line_of_an_earlier_pass_is_taken_out_when_the_code_changed(tmp_path: Path, monkeypatch: Any) -> None:
    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text("def run_cell(cell):\n    return {}\n", encoding="utf-8")
    (root / "plan.md").write_text("# Plan\n\n## Checks already made\n\n- nothing yet\n", encoding="utf-8")
    model = [_model]

    async def fake_run(executor: Any, python: Any, quest_root: Path, module: str, block: dict[str, Any],
                       **kwargs: Any) -> dm.Moves:
        return await dm.measure(block, _Calls(model[0]), budget_s=kwargs["timeout_s"])

    monkeypatch.setattr(dm, "run", fake_run)

    class _Log:
        def __getattr__(self, name: str) -> Any:
            return lambda *a, **k: None

    class _Exec:
        timeout_s = 600
        settings_at_once = 2

    class _Cfg:
        execution = _Exec()

    class _Engine:
        quest_root = root
        executor = None
        config = _Cfg()
        _log = _Log()
        _small_run_note = ""

        def _protocol_block(self, state: Any) -> dict[str, Any]:
            return {"optimisation": _block()}

    asyncio.run(dm.disclose(_Engine(), {}, sys.executable, {}, {"key": "code-a", "crashed": False}))
    assert "`unused`" in (root / "plan.md").read_text(encoding="utf-8")
    model[0] = lambda cell: {**_model(cell), "loss": _model(cell)["loss"] + cell["unused"]}  # the repaired code reads it
    asyncio.run(dm.disclose(_Engine(), {}, sys.executable, {}, {"key": "code-b", "crashed": False}))
    text = (root / "plan.md").read_text(encoding="utf-8")
    assert "`unused`" not in text and "nothing yet" in text


def test_a_readable_baseline_objective_is_never_said_missing_and_a_refused_limit_is_said() -> None:
    block = _two_variable_block()
    block["constraints"] = [{"quantity": "cost", "limit": "about ten"}]  # a limit that cannot be read
    moves = _measure(block, _Calls(lambda cell: {"loss": (cell["width"] - 2.5) ** 2, "cost": 1.0}))
    assert moves.baseline_reason == ""
    width = moves.variables[0]
    assert width.verdict == "moves" and "cannot be read" in width.sentence("loss")


def _engine_for(root: Path, logged: list[str]) -> Any:
    class _Log:
        def __getattr__(self, name: str) -> Any:
            return lambda msg, *a, **k: logged.append(msg % a if a else msg)

    class _Exec:
        timeout_s = 600
        settings_at_once = 2

    class _Cfg:
        execution = _Exec()

    class _Engine:
        quest_root = root
        executor = None
        config = _Cfg()
        _log = _Log()
        _small_run_note = ""

        def _protocol_block(self, state: Any) -> dict[str, Any]:
            return {"optimisation": _block()}

    return _Engine()


def test_run_log_lists_each_variable_and_a_plan_line_not_written_is_tried_again(tmp_path: Path, monkeypatch: Any) -> None:
    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text("def run_cell(cell):\n    return {}\n", encoding="utf-8")
    (root / "plan.md").mkdir()  # plan.md cannot be read or written for now
    runs: list[int] = []

    async def fake_run(executor: Any, python: Any, quest_root: Path, module: str, block: dict[str, Any],
                       **kwargs: Any) -> dm.Moves:
        runs.append(1)
        return await dm.measure(block, _Calls(), budget_s=kwargs["timeout_s"])

    monkeypatch.setattr(dm, "run", fake_run)
    logged: list[str] = []
    asyncio.run(dm.disclose(_engine_for(root, logged), {}, sys.executable, {}, {"key": "code-a", "crashed": False}))
    assert any(line.startswith("[execute]   `unused`: no change") for line in logged)  # each variable's own line
    assert dm.read_record(root)["plan_pending"] is True
    (root / "plan.md").rmdir()
    (root / "plan.md").write_text("# Plan\n\n## Checks already made\n\n- nothing yet\n", encoding="utf-8")
    asyncio.run(dm.disclose(_engine_for(root, logged), {}, sys.executable, {}, {"key": "code-a", "crashed": False}))
    assert len(runs) == 1 and "`unused`" in (root / "plan.md").read_text(encoding="utf-8")
    assert dm.read_record(root)["plan_pending"] is False
