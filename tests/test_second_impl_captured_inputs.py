"""A key function the plan gives no worked example for is still compared with its second version: FI runs the simulation
once and records what the function is called with (core/second_impl.py::capture_inputs), through its own trial harness."""
from __future__ import annotations

import asyncio
import json
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
