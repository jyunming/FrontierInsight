"""FI's table of the trials for the analysis it frames (core/frame_table.py, ``Table`` in core/skeleton.py's runtime): the numbers
equal FI's own (trial_table, paper_values) to 12 digits, a result the plan does not declare is refused with the declared list, and
the carried text runs with nothing of FI's on the path. Made-up topic (a colony); the recorded scratch quests are used when present."""

from __future__ import annotations

import json
import math
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from core import frame_table, paper_values, skeleton, stats, trial_runner, trial_table

SCRATCH = Path("C:/fi_runs/c1/skeleton/s4_inputs/scratch")
RECORDED = sorted(p for p in SCRATCH.iterdir() if (p / "raw" / "trials.json").is_file()) if SCRATCH.is_dir() else []
DIGITS = 12


def _cell(rate: float, dose: int, growth: list[float], *, failed: int = 0, **extra: Any) -> dict[str, Any]:
    cell = {"rate": rate, "dose": dose}
    return {"cell": cell, "key": trial_runner.cell_key(cell), "planned": len(growth) + failed, "ok": len(growth), "failed": failed,
            "metrics": {"growth": {"values": growth, "trials": list(range(len(growth)))},
                        "alive": {"values": [1.0 if v > 2 else 0.0 for v in growth]}}, **extra}


def _colony() -> list[dict[str, Any]]:
    return [_cell(0.1, 1, [1.0, 1.5, 2.5, 2.0]), _cell(0.1, 2, [3.0, 3.5, float("nan"), 4.5], failed=1),
            _cell(0.2, 1, [1.25, 1.75, 2.25, 2.75, 3.25]), _cell(0.2, 2, [5.0])]


def _table(cells: list[dict[str, Any]] | None = None, declared: tuple[str, ...] = ("growth", "alive"), **kw: Any) -> Any:
    ns: dict[str, Any] = {"fi_table": frame_table}
    sys.modules["fi_table"] = frame_table  # the runtime imports the carried module by this name
    exec(compile(skeleton.RUNTIME, "fi_frame.py", "exec"), ns)  # noqa: S102 -- FI's own runtime text
    return ns["Table"](cells if cells is not None else _colony(), list(declared), **kw)


def test_the_table_serves_the_settings_values_and_numbers_of_the_record() -> None:
    t = _table()
    assert t.keys() == ["rate=0.1,dose=1", "rate=0.1,dose=2", "rate=0.2,dose=1", "rate=0.2,dose=2"]
    assert t.setting("rate=0.2,dose=1") == {"rate": 0.2, "dose": 1}
    assert t.values("growth", "rate=0.1,dose=2") == [3.0, 3.5, 4.5], "a value that is not a finite number is not served"
    assert t.trials("growth", "rate=0.1,dose=2") == [0, 1, 3]
    assert t.n("growth", "rate=0.1,dose=2") == 3 and t.failed("rate=0.1,dose=2") == 1
    assert t.mean("growth", "rate=0.1,dose=1") == pytest.approx(1.75)
    lo, hi = t.interval("growth", "rate=0.1,dose=1")
    assert lo < 1.75 < hi
    assert t.interval("growth", "rate=0.2,dose=2") == (None, None), "one trial gives no interval"
    assert t.at(rate=0.2) == ["rate=0.2,dose=1", "rate=0.2,dose=2"]
    assert t.across("dose", rate=0.1) == [(1, "rate=0.1,dose=1"), (2, "rate=0.1,dose=2")]
    assert t.across("rate", dose=2) == [(0.1, "rate=0.1,dose=2"), (0.2, "rate=0.2,dose=2")]
    why = t.missing("no colony survived")
    assert why.why == "no colony survived"


def test_a_proportion_has_a_wilson_interval_and_a_mean_a_t_interval() -> None:
    t = _table()
    key = "rate=0.2,dose=1"
    assert t.mean("alive", key) == pytest.approx(0.6)
    lo, hi = t.interval("alive", key)
    assert (lo, hi) == stats.wilson_interval(3, 5)
    lo, hi = t.interval("growth", key)
    half = stats._t95(4) * stats.mean_std(t.values("growth", key))[1] / math.sqrt(5)
    assert lo == pytest.approx(2.25 - half, abs=0, rel=1e-12) and hi == pytest.approx(2.25 + half, abs=0, rel=1e-12)


def test_an_id_the_plan_does_not_declare_raises_with_the_declared_list() -> None:
    t = _table(declared=("growth",))
    for read in (lambda: t.mean("decay", "rate=0.1,dose=1"), lambda: t.values("alive", "rate=0.1,dose=1"),
                 lambda: t.interval("decay", "rate=0.1,dose=1"), lambda: t.levels("decay", "dose", "rate=0.1,dose=1"),
                 lambda: t.reference("decay", "rate=0.1,dose=1")):
        with pytest.raises(type(t).__init__.__globals__["UnknownResult"]) as e:
            read()
        assert "`" in str(e.value) and "growth" in str(e.value) and "declared" in str(e.value)
    assert t.mean("growth", "rate=0.1,dose=1") is not None
    assert t.served == ["growth"], "only an id that was read, and declared, is an observed input"


def test_a_setting_that_is_not_in_the_record_names_the_settings() -> None:
    with pytest.raises(KeyError) as e:
        _table().mean("growth", "rate=9,dose=9")
    assert "rate=0.1,dose=1" in str(e.value)


def test_the_means_of_what_each_run_compares_inside_it_come_from_the_value_records() -> None:
    cells = _colony()
    for level, vals in (("dry", [1.0, 3.0]), ("wet", [2.0, 4.0])):
        cell = {"rate": 0.1, "dose": 1, "soil": level}
        cells.append({"cell": cell, "key": trial_runner.cell_key(cell), "level": {"soil": level}, "planned": 4, "ok": 4, "failed": 0,
                      "metrics": {"growth": {"values": vals}}})
    t = _table(cells)
    assert t.levels("growth", "soil", "rate=0.1,dose=1") == {"dry": 2.0, "wet": 3.0}
    assert t.levels("growth", "soil", "rate=0.2,dose=1") == {}
    assert "rate=0.1,dose=1,soil=dry" not in t.keys(), "a value record is not a setting"


def test_parts_and_references_come_from_what_fi_gives_the_table() -> None:
    t = _table(references={"growth": {"rate=0.1,dose=1": 1.5}}, parts={"low dose": ["rate=0.1,dose=1"]})
    assert t.part("low dose") == ["rate=0.1,dose=1"]
    assert t.reference("growth", "rate=0.1,dose=1") == 1.5 and t.reference("growth", "rate=0.2,dose=1") is None
    with pytest.raises(Exception) as e:
        t.part("high dose")
    assert "low dose" in str(e.value)


# ---- the numbers are FI's own ---------------------------------------------------------------------------------------------

def _same(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return a is b
    return math.isclose(a, b, rel_tol=10.0 ** -DIGITS, abs_tol=0.0) or a == b


def _check_against_trial_table(cells: list[dict[str, Any]]) -> int:
    t = _table(cells, declared=tuple(sorted({m for c in cells for m in c["metrics"]})))
    plain = [c for c in cells if not c.get("level")]
    summarised = {c["key"]: c for c in trial_table.summaries(plain)}
    checked = 0
    for key in t.keys():
        for name, s in summarised[key]["metrics"].items():
            lo, hi, _how = trial_table.interval(s)
            assert t.n(name, key) == s["n"]
            assert _same(t.mean(name, key), s["mean"] if s["n"] else None), (key, name)
            assert _same(t.interval(name, key)[0], lo) and _same(t.interval(name, key)[1], hi), (key, name)
            checked += 1
    return checked


def test_the_table_equals_trial_table_on_the_made_up_record() -> None:
    assert _check_against_trial_table(_colony()) == 8


@pytest.mark.skipif(not RECORDED, reason="the recorded scratch quests are not here")
@pytest.mark.parametrize("quest", RECORDED, ids=lambda p: p.name)
def test_the_table_equals_trial_table_and_paper_values_on_a_recorded_raw(quest: Path) -> None:
    cells = frame_table.load(quest)
    assert cells and _check_against_trial_table(cells) > 0
    plain = [c for c in cells if not c.get("level")]
    assert all(c["key"] == trial_runner.cell_key(c["cell"]) for c in plain), "FI's names of the settings"
    t = _table(cells, declared=tuple(sorted({m for c in cells for m in c["metrics"]})))
    assert t.keys() == [c["key"] for c in plain if c["metrics"] or c.get("planned")]
    rec = paper_values.Records(cells=cells)
    compared = 0
    for c in plain[:6]:
        for name in list(c["metrics"])[:4]:
            found = rec.at_setting(name, c["cell"])
            if not found or len(found) != 1:
                continue
            try:
                mean, lo, hi, *_rest = paper_values._interval(rec, found, name)
            except paper_values._Unfilled:
                continue
            assert _same(t.mean(name, c["key"]), mean), (c["key"], name)
            assert _same(t.interval(name, c["key"])[0], lo) and _same(t.interval(name, c["key"])[1], hi), (c["key"], name)
            compared += 1
    assert compared, "at least one number was compared with paper_values"


def test_the_carried_numbers_are_the_ones_of_core_stats_and_trial_runner() -> None:
    assert frame_table._T95 == stats._T95 and frame_table._Z95 == stats._Z95
    for df in (0, 1, 7, 30, 35, 59, 120, 500):
        assert frame_table._t95(df) == stats._t95(df)
    for vals in ([], [2.0], [1.0, 2.5, 4.0], [0.1] * 7 + [3.5]):
        assert frame_table.mean_std(vals) == stats.mean_std(vals)
    for k, n in ((0, 5), (5, 5), (3, 10), (7, 0), (-1, 4), (6, 4)):
        assert frame_table.wilson_interval(k, n) == stats.wilson_interval(k, n)
    for cell in ({"a": 1, "b": 0.5}, {"x": 2.0, "y": [1, 2.0]}, {"name": "rk4"}, {}):
        assert frame_table.cell_key(cell) == trial_runner.cell_key(cell)


# ---- carried by text ------------------------------------------------------------------------------------------------------

def test_the_carried_text_imports_nothing_of_fis() -> None:
    for text in (skeleton.TABLE_SOURCE, skeleton.RUNTIME):
        assert "from core" not in text and "import core" not in text and "from ." not in text
    assert skeleton.TABLE_SOURCE == Path(frame_table.__file__).read_text(encoding="utf-8")


def test_the_carried_text_runs_with_only_the_quests_code_folder_on_the_path(tmp_path: Path) -> None:
    quest = tmp_path / "quest"
    code = quest / "code"
    (quest / "raw").mkdir(parents=True)
    code.mkdir()
    (quest / "raw" / "trials.json").write_text(json.dumps({"schema": "fi.trials/v1", "cells": [
        {**c, "metrics": {k: {"values": [v for v in m["values"] if v == v]} for k, m in c["metrics"].items()}} for c in _colony()]}),
        encoding="utf-8")
    assert skeleton.ensure_runtime(code, "") and skeleton.ensure_table(code, "")
    (code / "probe.py").write_text(textwrap.dedent("""
        import importlib.util, json, sys
        assert importlib.util.find_spec("core") is None, "FI's own modules must not be on the path"
        from fi_frame import Table, UnknownResult, NotYet, Figure
        t = Table.read(sys.argv[1], ["growth"])
        try:
            t.mean("alive", t.keys()[0])
            refused = ""
        except UnknownResult as e:
            refused = str(e)
        print(json.dumps({"keys": t.keys(), "mean": t.mean("growth", t.keys()[0]), "refused": refused,
                          "interval": t.interval("growth", t.keys()[0]), "version": sys.modules["fi_frame"].FRAME_RUNTIME_VERSION}))
    """), encoding="utf-8")
    run = subprocess.run([sys.executable, "-E", "-s", str(code / "probe.py"), str(quest)], cwd=tmp_path, capture_output=True, text=True,
                         timeout=120, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    assert run.returncode == 0, run.stderr
    out = json.loads(run.stdout.strip().splitlines()[-1])
    assert out["keys"][0] == "rate=0.1,dose=1" and out["mean"] == 1.75 and out["version"] == skeleton.VERSION
    assert "declared" in out["refused"] and "growth" in out["refused"]
    here = _table(declared=("growth",))
    assert tuple(out["interval"]) == here.interval("growth", "rate=0.1,dose=1"), "one implementation, the host's numbers"


def test_the_record_read_is_the_one_the_run_names_never_a_copy_left_from_an_earlier_run(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "trials.json").write_text(json.dumps({"cells": _colony()}), encoding="utf-8")
    (raw / "trials.analysis.json").write_text(json.dumps({"cells": _colony()[:2]}), encoding="utf-8")  # left by an earlier run
    assert len(frame_table.load(tmp_path)) == 4 and len(frame_table.load(raw / "trials.json")) == 4
    assert frame_table.record_path(tmp_path, {}) == str(tmp_path / "raw" / "trials.json"), "nothing named: FI's own record"
    named = frame_table.record_path(tmp_path, {"FI_TRIALS": "raw/trials.analysis.json"})
    assert len(frame_table.load(named)) == 2, "the file the harness names is the one read"
    assert frame_table.load(tmp_path / "nothing") == []
    assert frame_table.RESULTS_ENV == trial_runner.RESULTS_ENV


# ---- values compared inside each run ------------------------------------------------------------------------------------------

def _level_cell(base: dict[str, Any], level: dict[str, Any], values: list[float]) -> dict[str, Any]:
    cell = {**base, **level}
    return {"cell": cell, "key": trial_runner.cell_key(cell), "level": level, "planned": 4, "ok": 4, "failed": 0,
            "metrics": {"growth": {"values": values}}}


def test_levels_of_a_run_that_compares_two_factors_are_asked_for_one_factor_at_the_others_values() -> None:
    base = {"rate": 0.1, "dose": 1}
    cells = _colony()
    for soil, light, vals in (("dry", 1, [1.0, 3.0]), ("dry", 2, [5.0, 7.0]), ("wet", 1, [2.0, 4.0]), ("wet", 2, [6.0, 8.0])):
        cells.append(_level_cell(base, {"soil": soil, "light": light}, vals))
    pairs_only = list(cells)
    cells.append(_level_cell(base, {"soil": "dry"}, [10.0, 12.0]))  # a number that varies over soil alone
    cells.append(_level_cell(base, {"soil": "wet"}, [20.0, 22.0]))
    t = _table(cells)
    key = "rate=0.1,dose=1"
    assert t.levels("growth", "soil", key) == {"dry": 11.0, "wet": 21.0}, "the record kept per soil alone"
    assert t.levels("growth", "soil", key, light=2) == {"dry": 6.0, "wet": 7.0}
    assert t.levels("growth", "light", key, soil="wet") == {1: 3.0, 2: 7.0}
    assert t.levels("growth", "soil", "rate=0.2,dose=1") == {}
    assert t.levels("growth", "colour", key) == {}, "a factor the run does not compare"
    only_pairs = _table(pairs_only)
    with pytest.raises(ValueError) as e:
        only_pairs.levels("growth", "soil", key)
    assert "light" in str(e.value) and "growth" in str(e.value), "no silent empty answer"
    assert len(only_pairs.keys()) == 4, "value records are not settings"


# ---- edge cases of the numbers ----------------------------------------------------------------------------------------------

def test_edge_cases_give_what_trial_table_gives() -> None:
    nan = float("nan")
    cells = [_cell(0.1, 1, [nan, nan]), _cell(0.1, 2, [1.0, True, 0.0]), _cell(0.2, 1, [0.0, 1.0, 0.5, 1.0]), _cell(0.2, 2, [1.0]),
             _cell(0.3, 1, [0.0, 0.0, 0.0])]
    t = _table(cells)
    assert t.n("growth", "rate=0.1,dose=1") == 0 and t.mean("growth", "rate=0.1,dose=1") is None
    assert t.interval("growth", "rate=0.1,dose=1") == (None, None) and t.values("growth", "rate=0.1,dose=1") == []
    assert t.values("growth", "rate=0.1,dose=2") == [1.0, 0.0], "a bool is not a value"
    assert t.interval("growth", "rate=0.1,dose=2") == stats.wilson_interval(1, 2), "every value 0 or 1: Wilson"
    mixed = t.interval("growth", "rate=0.2,dose=1")
    assert mixed != stats.wilson_interval(3, 4) and mixed[0] < 0.625 < mixed[1], "a 0.5 among them: a t interval"
    assert t.interval("growth", "rate=0.2,dose=2") == stats.wilson_interval(1, 1), "one trial of a proportion, as trial_table"
    assert t.interval("growth", "rate=0.3,dose=1") == stats.wilson_interval(0, 3)
    assert _check_against_trial_table(cells) == 10


# ---- speed ------------------------------------------------------------------------------------------------------------------

@pytest.mark.skipif(not RECORDED, reason="the recorded scratch quests are not here")
def test_the_largest_recorded_record_is_served_quickly() -> None:
    import time

    biggest = max(RECORDED, key=lambda p: (p / "raw" / "trials.json").stat().st_size)
    cells = frame_table.load(biggest)
    names = tuple(sorted({m for c in cells for m in c["metrics"]}))
    started = time.monotonic()
    t = _table(cells, declared=names)
    factors = sorted({a for c in cells if c.get("level") for a in c["level"]})
    for key in t.keys():
        for name in names[:3]:
            t.mean(name, key)
            t.interval(name, key)
        for factor in factors:
            try:
                t.levels(names[0], factor, key)
            except ValueError:
                pass
    seconds = time.monotonic() - started
    print(f"{biggest.name}: {len(cells)} records, {len(t.keys())} settings, {seconds:.2f} s")
    assert seconds < 60


# ---- the runtime: version 3 for a new frame, a version 2 one kept ---------------------------------------------------------------

DESIGN: dict[str, Any] = {"protocol": {"grid": {"rate": [0.1, 0.2]}, "runs_per_setting": 3, "metrics": [{"id": "growth", "kind": "mean"}]}}


def _older(text: str) -> str:
    return text.replace(f"version {skeleton.VERSION}", "version 2").replace(f"FRAME_VERSION = {skeleton.VERSION}", "FRAME_VERSION = 2")


def test_a_new_frame_gets_the_version_3_runtime_and_the_table() -> None:
    assert skeleton.VERSION == 4 and skeleton.runtime_version(skeleton.RUNTIME) == 4
    for name in ("Table", "Figure", "NotYet", "UnknownResult"):
        assert f"class {name}" in skeleton.RUNTIME


def test_a_version_2_runtime_and_table_are_kept_on_resume_and_a_new_frame_writes_version_3(tmp_path: Path) -> None:
    frame = skeleton.frame_source(DESIGN, stochastic=True)
    old = _older(frame)
    assert skeleton.frame_version(old) == 2
    code = tmp_path / "code"
    code.mkdir()
    v2 = skeleton.RUNTIME.replace(f"FRAME_RUNTIME_VERSION = {skeleton.VERSION}", "FRAME_RUNTIME_VERSION = 2")
    (code / skeleton.RUNTIME_FILE).write_text(v2, encoding="utf-8")
    assert skeleton.ensure_runtime(code, old) is False and skeleton.ensure_table(code, old) is False
    assert skeleton.runtime_version((code / skeleton.RUNTIME_FILE).read_text(encoding="utf-8")) == 2
    assert not (code / skeleton.TABLE_FILE).exists(), "a version 2 frame never reads the table"
    assert skeleton.frame_version(skeleton.frame_for(DESIGN, stochastic=True, existing=old)) == 2, "a resume keeps the frame"
    assert skeleton.ensure_runtime(code, frame) is True and skeleton.ensure_table(code, frame) is True
    assert skeleton.runtime_version((code / skeleton.RUNTIME_FILE).read_text(encoding="utf-8")) == skeleton.VERSION
    assert (code / skeleton.TABLE_FILE).read_text(encoding="utf-8") == skeleton.TABLE_SOURCE
    assert skeleton.ensure_table(code, frame) is False, "already written"


# ---- run.py carries the runtime the quest has; fi_table.py is not part of the trials' key ---------------------------------------

def test_run_py_of_a_version_2_quest_carries_its_own_runtime_so_its_trials_are_not_run_again(tmp_path: Path) -> None:
    from core import code_project

    code = tmp_path / "code"
    code.mkdir()
    body = "def trial(cell, ctx):\n    return TrialResult()\n"
    (code / "simulate.py").write_text(_older(skeleton.fill(skeleton.frame_source(DESIGN, stochastic=True), body)[0] or ""), encoding="utf-8")
    v2 = skeleton.RUNTIME.replace(f"FRAME_RUNTIME_VERSION = {skeleton.VERSION}", "FRAME_RUNTIME_VERSION = 2")
    (code / skeleton.RUNTIME_FILE).write_text(v2, encoding="utf-8")
    text = code_project.run_source(code)
    assert repr(v2) in text, "the quest's runtime, not today's"
    assert code_project.run_source(code) == text, "writing it again changes nothing"
    # a quest whose runtime file is not FI's gets the current one
    (code / skeleton.RUNTIME_FILE).write_text("# mine\n", encoding="utf-8")
    assert repr(skeleton.RUNTIME) in code_project.run_source(code)
    (code / skeleton.RUNTIME_FILE).unlink()
    assert repr(skeleton.RUNTIME) in code_project.run_source(code)


def test_the_reader_of_the_record_is_not_an_input_of_the_trials(tmp_path: Path) -> None:
    from core import code_layout

    code = tmp_path / "code"
    code.mkdir()
    (code / "simulate.py").write_text("x = 1\n", encoding="utf-8")
    (code / skeleton.RUNTIME_FILE).write_text(skeleton.RUNTIME, encoding="utf-8")
    (code / skeleton.TABLE_FILE).write_text(skeleton.TABLE_SOURCE, encoding="utf-8")
    names = [p.name for p in code_layout.simulation_inputs(tmp_path)]
    assert skeleton.TABLE_FILE == code_layout.TABLE_SCRIPT
    assert skeleton.TABLE_FILE not in names and skeleton.RUNTIME_FILE in names and "simulate.py" in names


# ---- what a read gave and raised (FI's own record of the reads, for the figure check) ----------------------------------------------
def test_a_list_of_settings_is_not_a_key_and_the_message_says_what_to_do() -> None:
    t = _table()
    keys = t.at(rate=0.2)
    assert isinstance(keys, list)
    with pytest.raises(TypeError) as err:
        t.mean("growth", keys)
    assert "list of settings, not one setting" in str(err.value) and "table.at(...)[0]" in str(err.value)
    with pytest.raises(TypeError):
        t.interval("growth", tuple(keys))
    assert t.mean("growth", keys[0]) == pytest.approx(2.25), "one key of the list is a key"


def test_gave_values_holds_only_the_ids_a_read_returned_data_for_and_resets() -> None:
    t = _table(cells=[_cell(0.1, 1, [1.0, 2.0]), _cell(0.1, 2, [float("nan")])])
    assert t.gave_values == set()
    assert t.mean("growth", "rate=0.1,dose=2") is None and t.values("growth", "rate=0.1,dose=2") == []
    assert t.interval("growth", "rate=0.1,dose=2") == (None, None) and t.n("growth", "rate=0.1,dose=2") == 0
    assert "growth" in t.served and t.gave_values == set(), "served is not given: no read returned a value"
    assert t.mean("alive", "rate=0.1,dose=1") == pytest.approx(0.0) and t.gave_values == {"alive"}
    assert t.values("growth", "rate=0.1,dose=1") == [1.0, 2.0] and t.gave_values == {"alive", "growth"}
    t.reset_reads()
    assert t.trials("growth", "rate=0.1,dose=1") == [0, 1] and t.gave_values == set(), "trial numbers are not a result's values"
    assert t.gave_values == set() and t.last_error == {}


def test_the_last_error_of_a_read_is_kept_even_when_the_caller_swallows_it() -> None:
    t = _table()
    for key in (t.at(rate=0.1), "no such setting"):
        try:
            t.mean("growth", key)
        except Exception:  # the way a body hides it
            continue
    assert t.last_error["growth"].startswith("KeyError: ") and "no such setting" in t.last_error["growth"]
    try:
        t.mean("growth", t.at(rate=0.1))
    except Exception:
        pass
    assert t.last_error["growth"].startswith("TypeError: ") and "list of settings" in t.last_error["growth"]
    assert t.gave_values == set()


def test_the_last_error_in_the_dict_is_the_latest_raised_and_an_unhashable_id_keeps_its_own_error() -> None:
    t = _table()
    for name, key in (("growth", "no such setting"), ("alive", "no such setting"), ("growth", "again")):
        try:
            t.mean(name, key)
        except KeyError:
            pass
    assert list(t.last_error) == ["alive", "growth"] and "again" in t.last_error["growth"]
    with pytest.raises(type(t).__init__.__globals__["UnknownResult"]):
        t.mean(["growth"], "rate=0.1,dose=1")


def test_a_list_is_not_a_key_of_level_means_or_reference_either() -> None:
    t = _table()
    for read in (lambda: t.level_means("growth", ["rate=0.1,dose=1"]), lambda: t.reference("growth", ["rate=0.1,dose=1"])):
        with pytest.raises(TypeError, match="list of settings"):
            read()


def test_level_rows_give_the_trials_behind_each_compared_value_under_its_record_key_and_level_means_text() -> None:
    base = {"rate": 0.1, "dose": 1}
    cells = _colony() + [_level_cell(base, {"soil": "dry"}, [1.0, 3.0]), _level_cell(base, {"soil": "wet"}, [2.0, 4.0])]
    t = _table(cells)
    key = "rate=0.1,dose=1"
    assert t.level_rows("growth", key) == [("rate=0.1,dose=1,soil=dry", "dry", [1.0, 3.0]), ("rate=0.1,dose=1,soil=wet", "wet", [2.0, 4.0])]
    assert {text for _k, text, _v in t.level_rows("growth", key)} == set(t.level_means("growth", key))
    assert t.level_rows("growth", "rate=0.2,dose=1") == []
    with pytest.raises(TypeError, match="list of settings"):
        t.level_rows("growth", [key])
