"""A key function the plan gives no worked example for is still compared with its second version: FI runs the simulation
once and records what the function is called with (core/second_impl.py::capture_inputs), through its own trial harness."""
from __future__ import annotations

import asyncio
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

from core import function_steps as fs
from core import second_impl as si

MODEL = '''import numpy as np


def width_at(profile, level):
    """Where the profile crosses the level, by linear interpolation."""
    above = np.asarray(profile) >= level
    return float(above.sum())


def mask_bias(gap, scale=1.0):
    return gap * scale + width_at(np.linspace(0.0, 1.0, 11) * gap, 0.5)


def helper_object(thing):
    return 1.0
'''

SIMULATE = '''from pkg.model import mask_bias, helper_object
from pkg import model


def run_cell(cell):
    helper_object(object())  # an argument that is not plain data: not recorded
    return {"bias": mask_bias(cell["gap"]), "again": model.mask_bias(cell["gap"], scale=2.0)}
'''


def _quest(tmp_path: Path) -> Path:
    (tmp_path / "code" / "pkg").mkdir(parents=True)
    (tmp_path / "code" / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "code" / "pkg" / "model.py").write_text(MODEL, encoding="utf-8")
    (tmp_path / "code" / "simulate.py").write_text(SIMULATE, encoding="utf-8")
    return tmp_path


def _run(root: Path):
    async def run(argv: list[str]) -> Any:
        done = subprocess.run([sys.executable, "-B", *argv], cwd=root, capture_output=True, text=True, timeout=120)
        return done.returncode, done.stdout, done.stderr
    return run


def test_fi_records_what_each_function_is_called_with_when_the_simulation_runs(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    got = asyncio.run(si.capture_inputs(_run(root), root, "pkg", ["width_at", "mask_bias", "helper_object"],
                                        [{"gap": 3.0}], deterministic=True))
    assert got["mask_bias"][0] == {"gap": 3.0}
    assert {"gap": 3.0, "scale": 2.0} in got["mask_bias"], "a call through the module is recorded too"
    assert got["width_at"][0]["level"] == 0.5 and len(got["width_at"][0]["profile"]) == 11, "a real array, by its name"
    assert "helper_object" not in got, "a call with something that is not plain data is not kept"
    assert json.loads((root / si.CAPTURED).read_text(encoding="utf-8"))


def test_a_function_with_no_example_is_called_on_what_fi_recorded() -> None:
    spec = fs.Spec(name="mask_bias", signature="def mask_bias(gap, scale)", purpose="the bias", equation="E9")
    assert si.example_inputs(spec, {"model": {"equations": []}}) is None
    captured = {"mask_bias": [{"gap": 3.0, "scale": 1.0}, {"gap": 3.0, "scale": 2.0}]}
    assert si.example_inputs(spec, {"model": {"equations": []}}, captured) == {"gap": 3.0, "scale": 1.0}


def test_the_harness_without_a_capture_is_unchanged(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    from core import trial_runner as tr

    (root / tr.HARNESS_PATH).parent.mkdir(parents=True, exist_ok=True)
    (root / tr.HARNESS_PATH).write_text(tr.HARNESS_SOURCE, encoding="utf-8")
    (root / "x").mkdir()
    plan = tr._plan(root, "code/simulate.py", {"gap": [3.0]}, runs_per_setting=1, base_seed=0, deterministic=True,
                    folder=root / "x", out_name="o{index}.jsonl")
    subprocess.run([sys.executable, str(tr.HARNESS_PATH), plan[0]["spec"], plan[0]["out"]], cwd=root, check=True,
                   capture_output=True, timeout=120)
    row = json.loads((root / plan[0]["out"]).read_text(encoding="utf-8").splitlines()[-1])
    assert row["status"] == "ok" and not (root / si.CAPTURED).exists()


def test_the_scan_cells_of_a_search_are_given_the_values_every_evaluation_goes_through(tmp_path: Path,
                                                                                       monkeypatch) -> None:  # noqa: ANN001
    """The scan's cells leave out the values every evaluation goes through (the trial runner gives each cell all of
    them): without them a recording run of a scan cell is missing settings the simulation reads."""
    import asyncio
    from types import SimpleNamespace

    from core import second_impl as si
    from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig
    from core.engine import Engine
    from tests.test_optimise_runner import BLOCK

    eng = Engine(Config(topic="a neutral topic", title="capture", provider=ProviderConfig(name="openai"),
                        engine=EngineConfig(clarify_mode="off"), execution=ExecutionConfig(sandbox="venv"),
                        knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "out")))
    (eng.quest_root / "code").mkdir(parents=True, exist_ok=True)
    (eng.quest_root / "code" / "simulate.py").write_text("def run_cell(cell):\n    return {}\n", encoding="utf-8")
    seen: dict = {}

    async def capture(run, root, package, names, cells, **kw):  # noqa: ANN001, ANN202
        seen.update(cells=cells, **kw)
        return {}

    monkeypatch.setattr(si, "key_functions", lambda specs, protocol: ([SimpleNamespace(name="f")], []))
    monkeypatch.setattr(si, "capture_inputs", capture)
    protocol = {"optimisation": {**BLOCK, "in_each_run": {"case": [1, 2, 3]}, "grid": {"x": [-1.0, 1.0]}}}
    asyncio.run(eng._captured_calls({}, protocol, "pkg", [], {}))
    assert seen["in_each_run"] == {"case": [1, 2, 3]}
    assert len(seen["cells"]) > 1 and any("case" not in c for c in seen["cells"]), "the scan's cells, given the values"


def test_fi_records_every_key_a_function_returned(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    (root / 'code' / 'pkg' / 'model.py').write_text(
        "def fit(x):\n    return {'slope': 2.0 * x, 'spare': 1.0, 'cov': 3.0}\n\n\n"
        "def plain(x):\n    return x\n", encoding="utf-8")
    (root / 'code' / 'simulate.py').write_text(
        "from pkg.model import fit, plain\n\n\ndef run_cell(cell):\n"
        "    r = fit(cell['gap'])\n    plain(cell['gap'])\n    return {'v': r['slope']}\n", encoding="utf-8")
    asyncio.run(si.capture_inputs(_run(root), root, "pkg", ["fit", "plain"], [{"gap": 3.0}], deterministic=True))
    returned = si.read_captured(root)["returned"]
    assert returned["fit"] == ["cov", "slope", "spare"] and "plain" not in returned


GAPS_MODEL = '''import numpy as np


def filled(values, weights):
    return float(np.nansum(np.asarray(values))) + float(np.nansum(np.abs(np.asarray(weights))))
'''

GAPS_SIMULATE = '''import numpy as np
from pkg import model


def run_cell(cell):
    return {"v": model.filled(np.array([1.0, np.nan, np.inf]), np.array([1 + 1j, complex(np.nan, 0.0)]))}
'''


def test_an_array_with_a_missing_value_is_still_recorded(tmp_path: Path) -> None:
    """An array is kept whole, whatever its values: a real one as it always was, a complex one the same way."""
    (tmp_path / "code" / "pkg").mkdir(parents=True)
    (tmp_path / "code" / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "code" / "pkg" / "model.py").write_text(GAPS_MODEL, encoding="utf-8")
    (tmp_path / "code" / "simulate.py").write_text(GAPS_SIMULATE, encoding="utf-8")
    got = asyncio.run(si.capture_inputs(_run(tmp_path), tmp_path, "pkg", ["filled"], [{"gap": 3.0}], deterministic=True))
    assert "filled" in got, "a call whose array holds a NaN or an infinity is recorded, as it was"
    values, weights = got["filled"][0]["values"], got["filled"][0]["weights"]
    assert values[0] == 1.0 and values[1] != values[1] and values[2] == float("inf")
    assert weights[0] == {"__complex__": [1.0, 1.0]} and weights[1]["__complex__"][0] != weights[1]["__complex__"][0]


def test_a_recorded_value_that_is_not_numbers_all_the_way_down_is_handed_on_as_it_was() -> None:
    """A 2-D complex array and a list of dicts are lists of lists, not arrays of numbers: they are never rescaled."""
    base = {"table": [[{"__complex__": [1.0, 2.0]}, {"__complex__": [0.0, 1.0]}]], "rows": [{"a": 1.0}],
            "names": [["x", "y"]], "z": 2.0, "grid": [[1.0, 2.0], [3.0, 4.0]]}
    inputs = si.build_inputs(base)
    assert inputs[0] == base
    assert all(c["table"] == base["table"] and c["rows"] == base["rows"] and c["names"] == base["names"] for c in inputs)
    # The number and the array of numbers are still varied.
    assert any(c["z"] != 2.0 for c in inputs) and any(c["grid"] != base["grid"] for c in inputs)


def test_an_empty_array_or_one_with_an_infinity_does_not_stop_the_inputs_from_being_built() -> None:
    """The shift of an array is sized by its finite numbers; an array with none of them is shifted by a fixed step."""
    base = {"empty": [[]], "gaps": [1.0, float("nan"), float("inf")], "z": 2.0}
    inputs = si.build_inputs(base)
    assert len(inputs) > 1 and all(c["empty"] == [[]] for c in inputs)
    assert all(math.isfinite(c["gaps"][0]) for c in inputs), "an infinity in the array does not make the shift infinite"
    assert any(c["gaps"][0] != 1.0 for c in inputs)
    only = si.build_inputs({"gaps": [1.0, float("inf")]})
    assert any(abs(c["gaps"][0] - 1.1) < 1e-12 for c in only), "the shift is a tenth of the largest finite number"


def test_a_recorded_complex_input_is_shown_as_the_number_in_a_sentence() -> None:
    """What a person reads names the number, never FI's own spelling of it."""
    case = {"inputs": {"table": [{"__complex__": [1.0, 2.0]}, {"__complex__": [0.5, -1.0]}], "opts": {"n": 3}},
            "main": 1.0, "other": 2.0}
    text = si.cases_text([case])
    assert "__complex__" not in text and "1+2j" in text and "0.5-1j" in text
    assert "opts = {n: 3}" in text, "a dict that is not a complex number is shown as before"


def _dec_in():
    """The comparison script's own reader of a recorded input, taken from its text."""
    script = si.compare_script(code_dir="code", module="pkg.model", name="f", reference="r.py", inputs="i.json",
                               uses=[], own={}, study="", setup=None)
    ns: dict = {}
    exec("import numpy as np\n" + script[script.index("def dec_in"):script.index("def done")], ns)  # noqa: S102
    return ns["dec_in"]


def test_a_recorded_list_that_is_not_an_array_is_called_back_as_the_list_it_was() -> None:
    dec_in = _dec_in()
    ragged = dec_in([[1, 2], [3]])
    assert isinstance(ragged, list) and all(isinstance(r, list) for r in ragged) and ragged == [[1, 2], [3]]
    rows = dec_in([{"a": [1.0, 2.0]}, {"a": [3.0]}])
    assert isinstance(rows, list) and isinstance(rows[0], dict) and rows[0]["a"].tolist() == [1.0, 2.0]
    assert dec_in([[1.0, 2.0], [3.0, 4.0]]).shape == (2, 2), "an array of numbers is an array, as before"
    grid = dec_in([[{"__complex__": [1.0, 2.0]}], [{"__complex__": [0.0, 1.0]}]])
    assert grid.shape == (2, 1) and grid.dtype.kind == "c"
    odd = dec_in({"__complex__": ["a", "b"]})
    assert isinstance(odd, dict) and list(odd) == ["__complex__"], "only two numbers are read as a complex number"


KEYED_MODEL = '''def keyed(opts, z):
    return float(opts["__complex__"][0]) + z
'''

KEYED_SIMULATE = '''from pkg import model


def run_cell(cell):
    return {"v": model.keyed({"__complex__": [1.0, 2.0]}, cell["gap"])}
'''


def test_a_dict_spelled_like_fis_complex_number_is_not_recorded(tmp_path: Path) -> None:
    """A study's own dict with only the key FI writes a complex number under cannot be told from one: the call is not
    kept (the function is then reported as called but not compared), never called back with another value."""
    (tmp_path / "code" / "pkg").mkdir(parents=True)
    (tmp_path / "code" / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "code" / "pkg" / "model.py").write_text(KEYED_MODEL, encoding="utf-8")
    (tmp_path / "code" / "simulate.py").write_text(KEYED_SIMULATE, encoding="utf-8")
    got = asyncio.run(si.capture_inputs(_run(tmp_path), tmp_path, "pkg", ["keyed"], [{"gap": 3.0}], deterministic=True))
    assert "keyed" not in got
    assert si.read_captured(tmp_path)["called_at"].get("keyed") == [0], "it was called; FI says so"


PACKAGE_MODEL = '''import numpy as np


def combine(parts, weights, z):
    t = np.asarray(parts["t"])
    return np.real(t * np.conj(t)).sum() * float(np.sum(weights["w"])) + z


def never_called(a):
    return a
'''

PACKAGE_SIMULATE = '''import numpy as np
from pkg import model


def run_cell(cell):
    parts = {"t": np.array([1 + 2j, 0.5 - 1j]), "step": cell["gap"]}
    return {"v": model.combine(parts, {"w": np.array([0.25, 0.75])}, 1.0)}
'''


def test_calls_with_dicts_and_complex_arrays_are_recorded_and_called_back_as_they_were(tmp_path: Path) -> None:
    (tmp_path / "code" / "pkg").mkdir(parents=True)
    (tmp_path / "code" / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "code" / "pkg" / "model.py").write_text(PACKAGE_MODEL, encoding="utf-8")
    (tmp_path / "code" / "simulate.py").write_text(PACKAGE_SIMULATE, encoding="utf-8")
    got = asyncio.run(si.capture_inputs(_run(tmp_path), tmp_path, "pkg", ["combine", "never_called"], [{"gap": 3.0}],
                                        deterministic=True))
    assert "combine" in got, "a call through a package module with dict and complex arguments is recorded"
    assert "never_called" not in got, "a function the run never calls is still not recorded"
    record = si.read_captured(tmp_path)
    assert record["called_at"] == {"combine": [0]}
    case = got["combine"][0]
    assert case["parts"]["t"][0] == {"__complex__": [1.0, 2.0]} and case["weights"] == {"w": [0.25, 0.75]}
    # The comparison script decodes the case into what the function was given (its own helper, taken from its text).
    ns: dict = {}
    script = si.compare_script(code_dir="code", module="pkg.model", name="combine", reference="r.py", inputs="i.json",
                               uses=[], own={}, study="", setup=None)
    head = script[script.index("def dec_in"):script.index("def done")]
    exec("import numpy as np\n" + head, ns)  # noqa: S102
    decoded = {k: ns["dec_in"](v) for k, v in case.items()}
    assert decoded["parts"]["t"].dtype.kind == "c" and decoded["parts"]["t"][0] == 1 + 2j
    # Both versions are called on what FI recorded: a second version with the same values agrees, one 2% larger does not.
    inputs = si.build_inputs(case)
    (tmp_path / "i.json").write_text(json.dumps(inputs), encoding="utf-8")
    larger = PACKAGE_MODEL + "\n\n_same = combine\n\n\ndef combine(parts, weights, z):\n    return 1.02 * _same(parts, weights, z)\n"
    for second, want in ((PACKAGE_MODEL, "agree"), (larger, "disagree")):
        (tmp_path / "r.py").write_text(second, encoding="utf-8")
        p = subprocess.run([sys.executable, "-B", "-c", script], cwd=tmp_path, capture_output=True, text=True, timeout=120)
        rows = si.parse_output(p.stdout)
        assert rows and si.judge_rows(rows["rows"], inputs, si.REL)["status"] == want, (want, p.stdout[-300:], p.stderr[-300:])
    sys.path.insert(0, str(tmp_path / "code"))
    try:
        from pkg import model  # type: ignore[import-not-found]
        assert model.combine(**decoded) == model.combine({"t": [1 + 2j, 0.5 - 1j]}, {"w": [0.25, 0.75]}, 1.0)
    finally:
        sys.path.remove(str(tmp_path / "code"))
