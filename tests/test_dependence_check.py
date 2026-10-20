"""Whether the headline results are computed with the functions FI tested: FI changes each tested function's result
where the simulation calls it and sees whether a headline result changes (core/second_impl.py::dependence)."""
from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

from core import second_impl as si

MODEL = '''def edge(level):
    return 2.0 * level


def side_value(level):
    return level + 1.0


def unused(level):
    return 0.0
'''

# The headline result `width` comes from code of the simulation's own; the package's `edge` is called for it in one
# path, `side_value` only for a cross-check column whose failure is caught.
SIMULATE = '''from pkg import model


def own_edge(level):
    return 2.0 * level


def run_cell(cell):
    out = {"width": model.edge(cell["level"]) if cell["mode"] == "package" else own_edge(cell["level"])}
    try:
        out["side"] = model.side_value(cell["level"])
    except Exception:
        out["side"] = float("nan")
    return out
'''


def _quest(tmp_path: Path, simulate: str = SIMULATE) -> Path:
    (tmp_path / "code" / "pkg").mkdir(parents=True)
    (tmp_path / "code" / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "code" / "pkg" / "model.py").write_text(MODEL, encoding="utf-8")
    (tmp_path / "code" / "simulate.py").write_text(textwrap.dedent(simulate), encoding="utf-8")
    return tmp_path


def _run(root: Path):
    async def run(argv: list[str]) -> Any:
        done = subprocess.run([sys.executable, "-B", *argv], cwd=root, capture_output=True, text=True, timeout=120)
        return done.returncode, done.stdout, done.stderr
    return run


def _check(root: Path, cells: list[dict[str, Any]], names: list[str] | None = None,
           headline: set[str] | None = None, deterministic: bool = True) -> dict[str, Any]:
    names = names or ["edge", "side_value", "unused"]
    asyncio.run(si.capture_inputs(_run(root), root, "pkg", names, cells, deterministic=deterministic))
    return asyncio.run(si.dependence(_run(root), root, "pkg", names, headline or {"width"},
                                     deterministic=deterministic))


def test_a_result_computed_with_the_tested_function_depends_on_it(tmp_path: Path) -> None:
    got = _check(_quest(tmp_path), [{"level": 3.3, "mode": "package"}])
    assert got["edge"]["status"] == si.FEEDS
    assert got["side_value"]["status"] == si.NOT_FEEDING, "a side value feeds no headline"
    assert got["unused"]["status"] == si.NOT_CALLED


def test_a_result_the_simulation_computes_itself_does_not_depend_on_the_tested_function(tmp_path: Path) -> None:
    # 3.3, not 3.0: a whole-number result that does not move is not judged (a few percent need not move it).
    got = _check(_quest(tmp_path), [{"level": 3.3, "mode": "own"}])
    assert got["edge"]["status"] == si.NOT_CALLED
    assert any("side_value" in s and "do not use" in s for s in si.dependence_sentences(got))
    assert (tmp_path / si.DEPENDENCE).is_file()
    assert si.dependence_gaps(tmp_path), "a function the headline results do not use is an evidence gap"


def test_a_side_call_the_simulation_does_not_guard_is_not_taken_for_a_use(tmp_path: Path) -> None:
    # The physics computed by the simulation itself, the tested function called beside it and its result thrown away:
    # the case this check exists for. Making the function fail stopped the run and read as "feeds".
    root = _quest(tmp_path, '''
        from pkg import model
        def run_cell(cell):
            x = cell["level"]
            _ = model.edge(x)
            return {"width": 2.0 * x}
        ''')
    got = _check(root, [{"level": 3.3}], names=["edge"])
    assert got["edge"]["status"] == si.NOT_FEEDING


def test_a_simulation_that_stops_when_the_tested_function_disagrees_is_neither_use_nor_gap(tmp_path: Path) -> None:
    root = _quest(tmp_path, '''
        from pkg import model
        def run_cell(cell):
            x = cell["level"]
            own = 2.0 * x
            assert abs(model.edge(x) - own) < 1e-9
            return {"width": own}
        ''')
    got = _check(root, [{"level": 3.0}], names=["edge"])
    assert got["edge"]["status"] == si.STOPS
    assert not si.dependence_gaps(root)


def test_a_function_that_feeds_the_headline_only_at_a_later_setting_feeds(tmp_path: Path) -> None:
    root = _quest(tmp_path, '''
        from pkg import model
        def run_cell(cell):
            x = cell["level"]
            v = model.edge(x)
            return {"width": v if cell["mode"] == 1 else 2.0 * x}
        ''')
    got = _check(root, [{"level": 3.0, "mode": 0}, {"level": 3.0, "mode": 1}], names=["edge"])
    assert got["edge"]["status"] == si.FEEDS
    assert got["edge"]["headline_moved"] == ["width"]


def test_a_headline_with_no_number_is_not_shown_to_be_unused(tmp_path: Path) -> None:
    root = _quest(tmp_path, '''
        from pkg import model
        def run_cell(cell):
            model.edge(cell["level"])
            return {"width": float("nan")}
        ''')
    got = _check(root, [{"level": 3.0}], names=["edge"])
    assert got["edge"]["status"] == si.NOT_RUN


def test_results_that_change_between_two_runs_of_one_setting_are_not_taken_for_a_use(tmp_path: Path) -> None:
    root = _quest(tmp_path, '''
        import random
        from pkg import model
        def run_cell(cell):
            x = cell["level"]
            model.side_value(x)
            return {"width": 2.0 * x + random.random() * 1e-3}
        ''')
    got = _check(root, [{"level": 3.0}], names=["side_value"])
    assert got["side_value"]["status"] == si.NOT_RUN


def test_a_record_of_other_code_is_not_read(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    _check(root, [{"level": 3.0, "mode": "own"}])
    assert si.read_dependence(root)
    (root / "code" / "simulate.py").write_text(SIMULATE + "\n# changed\n", encoding="utf-8")
    assert si.read_dependence(root) == {} and si.dependence_gaps(root) == []
    assert json.loads((root / si.DEPENDENCE).read_text(encoding="utf-8"))["functions"]


def test_a_simulation_that_never_ran_is_not_a_function_never_called(tmp_path: Path) -> None:
    root = _quest(tmp_path, '''
        def run_cell(cell):
            raise ValueError("broken")
        ''')
    got = _check(root, [{"level": 3.0}], names=["edge"])
    assert got["edge"]["status"] == si.NOT_RUN
    assert si.dependence_sentences(got) == []


def test_a_result_that_uses_only_the_ratios_of_the_function_s_numbers_still_depends_on_it(tmp_path: Path) -> None:
    # Weights of a fit: the same share on every weight would leave the fit as it is.
    root = _quest(tmp_path, '''
        from pkg import model
        def run_cell(cell):
            xs = [1.0, 2.0, 3.0, 4.0]
            w = [model.side_value(x) for x in xs]
            return {"width": sum(wi * x for wi, x in zip(w, xs)) / sum(w)}
        ''')
    got = _check(root, [{"level": 3.0}], names=["side_value"])
    assert got["side_value"]["status"] == si.FEEDS


def test_a_headline_computed_after_the_run_is_held_against_everything_the_run_returns(tmp_path: Path) -> None:
    # The headline `bias` is an average the analysis takes over runs; the run returns the estimate it is made of.
    root = _quest(tmp_path, '''
        from pkg import model
        def run_cell(cell):
            return {"width": 2.0 * cell["level"], "estimate": model.side_value(cell["level"])}
        ''')
    got = _check(root, [{"level": 3.0}], names=["side_value"], headline={"width", "bias"})
    assert got["side_value"]["status"] == si.FEEDS
    assert got["side_value"]["all_results"] is True and got["side_value"]["headline_moved"] == [],         "the record says no headline result moved by name: only a value the analysis may use did"


def test_a_function_that_returns_zero_is_not_shown_to_be_unused(tmp_path: Path) -> None:
    # A zero changed by a share is still zero: nothing was changed, so nothing is shown.
    root = _quest(tmp_path, '''
        from pkg import model
        def run_cell(cell):
            return {"width": 2.0 * cell["level"] + model.unused(cell["level"])}
        ''')
    got = _check(root, [{"level": 3.0}], names=["unused"])
    assert got["unused"]["status"] == si.NOT_RUN


def test_a_time_the_run_reports_does_not_hide_that_the_headline_does_not_use_the_function(tmp_path: Path) -> None:
    # A run returns how long it took beside its results: that number differs between any two runs.
    root = _quest(tmp_path, '''
        import time
        from pkg import model
        def run_cell(cell):
            x = cell["level"]
            model.side_value(x)
            return {"width": 2.0 * x, "wall_s": time.perf_counter()}
        ''')
    got = _check(root, [{"level": 3.3}], names=["side_value"], headline={"width", "bias"})
    assert got["side_value"]["status"] == si.NOT_FEEDING
    assert got["side_value"]["not_repeating"] == ["wall_s"]


def test_a_whole_number_headline_beside_a_time_is_not_shown_to_be_unused_either(tmp_path: Path) -> None:
    """The same guard after the results that do not repeat are set aside: what is left is a whole number."""
    root = _quest(tmp_path, '''
        import time
        from pkg import model
        def run_cell(cell):
            x = cell["level"]
            model.side_value(x)
            return {"width": 2.0 * x, "wall_s": time.perf_counter()}
        ''')
    got = _check(root, [{"level": 3.0}], names=["side_value"], headline={"width", "bias"})
    assert got["side_value"]["status"] == si.NOT_RUN and "whole number" in got["side_value"]["why"], got


def test_a_whole_number_headline_that_does_not_move_is_not_shown_to_be_unused(tmp_path: Path) -> None:
    """A count or a yes/no built from the tested function need not move when its result moves by a few percent."""
    root = _quest(tmp_path, '''
        from pkg import model
        def run_cell(cell):
            return {"width": float(model.edge(cell["level"]) > 1.0)}
        ''')
    got = _check(root, [{"level": 3.0}], names=["edge"])
    assert got["edge"]["status"] == si.NOT_RUN and "whole number" in got["edge"]["why"], got["edge"]
    assert not si.dependence_gaps(root)
