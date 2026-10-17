"""A setting the code reads must be a setting the study has (core/grid_keys.py, read before the run through
core/static_checks.py). Neutral toy code; the fake model returns the code it is told to."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from core import grid_keys as gk
from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig
from core.engine import Engine

PROTOCOL = {
    "grid": {"rate": [1, 2, 4], "size": [10, 20]},
    "oracles": [{"name": "c", "kind": "special_case", "case": {"rate": 1, "offset": 0.5}}],
    "fixed": {"gain": 3.0},
}

SIM_BAD = '''\
def run_cell(cell):
    rate = float(cell.get("rate_hz", cell.get("freq", 1.0)))
    size = cell["size"]
    return {"y": rate * size}
'''
SIM_OK = '''\
def run_cell(cell):
    rate = float(cell["rate"])
    size = cell["size"]
    offset = cell.get("offset", 0.0)
    gain = cell.get("gain", 3.0)
    return {"y": rate * size + offset * gain}
'''
ANALYSIS_BAD = '''\
import json, os
record = json.load(open(os.environ["FI_TRIALS"]))
for entry in record["cells"]:
    rate = entry["cell"].get("rate_hz", 1.0)
    n = entry["cell"]["count"]
'''
ANALYSIS_OK = '''\
import json, os
record = json.load(open(os.environ["FI_TRIALS"]))
for entry in record["cells"]:
    c = entry["cell"]
    rate = c["rate"]
    size = c.get("size")
    ok = entry["ok"]
    m = entry["metrics"]["y"]
'''


def _names(files: dict[str, str], protocol: dict[str, Any] | None = PROTOCOL) -> list[tuple[str, tuple[str, ...]]]:
    return [(r.script, r.keys) for r in gk.problems(files, protocol)]


def test_a_read_of_a_name_the_grid_does_not_have_is_caught_with_its_line_and_the_real_names() -> None:
    found = gk.problems({"simulate.py": SIM_BAD}, PROTOCOL)
    assert [(r.script, r.line, r.keys) for r in found] == [("simulate.py", 2, ("rate_hz", "freq"))]
    said = found[0].says(["rate", "size"])
    assert "simulate.py line 2" in said and "`rate_hz` or `freq`" in said and "`rate`, `size`" in said
    text = gk.directive(found, PROTOCOL)
    assert "did you mean `rate`?" in text and "`rate`, `size`" in text and text.rstrip().endswith("change nothing else.")
    assert "default" in text


def test_the_analysis_reads_a_cell_out_of_the_record_and_is_checked_the_same_way() -> None:
    got = _names({"experiment.py": ANALYSIS_BAD})
    assert ("experiment.py", ("rate_hz",)) in got and ("experiment.py", ("count",)) in got


def test_names_the_study_has_pass_in_every_form_it_gives_them() -> None:
    # the grid's axes, a check's case, a fixed setting, FI's own record fields, and a chain of spellings with one right
    assert _names({"simulate.py": SIM_OK, "experiment.py": ANALYSIS_OK}) == []
    chain = 'def run_cell(cell):\n    return {"y": cell.get("rate_hz", cell.get("rate", 1.0))}\n'
    assert _names({"simulate.py": chain}) == []
    either = 'def run_cell(cell):\n    return {"y": cell.get("rate_hz") or cell.get("size")}\n'
    assert _names({"simulate.py": either}) == []


def test_an_optional_setting_with_a_fallback_that_is_like_none_of_the_studys_passes() -> None:
    src = 'def run_cell(cell):\n    return {"y": cell["rate"] * cell.get("tolerance", 0.5)}\n'
    assert _names({"simulate.py": src}) == []
    # ... but without a fallback it can only fail, and a name spelt like a study setting is a fault either way
    assert _names({"simulate.py": 'def run_cell(cell):\n    return {"y": cell["tolerance"]}\n'}) == [
        ("simulate.py", ("tolerance",))]
    assert _names({"simulate.py": 'def run_cell(cell):\n    return {"y": cell.get("size_px", 5)}\n'}) == [
        ("simulate.py", ("size_px",))]


@pytest.mark.parametrize("src", [
    "def run_trial(cell, i, seed):\n    noise = lambda: cell['rate_hz']\n    return {'y': noise()}\n",
    "def run_trial(cell, i, seed):\n    def model(x):\n        return x * cell['rate_hz']\n    return {'y': model(1)}\n",
    "def run_trial(cell, i, seed):\n    sol = solve_ivp(lambda t, y: -cell['rate_hz'] * y, (0, 1), [1.0])\n"
    "    return {'y': sol}\n",
    "def run_trial(cell, i, seed):\n    c = cell\n    def model(x):\n        def inner():\n"
    "            return c.get('rate_hz')\n        return inner()\n    return {'y': model(1)}\n",
])
def test_a_setting_read_inside_a_lambda_or_a_closure_is_a_read(src: str) -> None:
    assert [r.keys for r in gk.reads(src, "simulate.py")] == [("rate_hz",)]
    assert _names({"simulate.py": src}) == [("simulate.py", ("rate_hz",))]


def test_a_nested_function_that_rebinds_the_name_does_not_read_a_setting() -> None:
    src = ("def run_trial(cell, i, seed):\n    c = cell\n    def model(c):\n        return c['rate_hz']\n"
           "    def other():\n        c = {'rate_hz': 1}\n        return c['rate_hz']\n"
           "    f = lambda c: c['rate_hz']\n    return {'y': model({}) + other() + f({}) + c['rate']}\n")
    assert [r.keys for r in gk.reads(src, "simulate.py")] == [("rate",)]


@pytest.mark.parametrize("src", [
    "def summarise(rows):\n    best = max(rows, key=lambda cell: cell['score'])\n    return best\n",
    "def rank(cells):\n    return sorted(cells, key=lambda setting: setting['mse'])\n",
    "best = min(results, key=lambda cell: cell['loss'])\n",
])
def test_a_lambda_parameter_named_like_a_setting_is_not_one_by_its_name(src: str) -> None:
    """A ``key=lambda cell: ...`` over other rows reads those rows, not the run's settings: no read, and no stop."""
    assert gk.reads(src, "simulate.py") == []
    assert _names({"simulate.py": src}) == []


@pytest.mark.parametrize("src", [
    "def run_trial(cell, i, seed):\n    f = lambda c: c['rate_hz']\n    return {'y': f(cell)}\n",
    "def run_trial(cell, i, seed):\n    f = lambda t, cell: cell['rate_hz'] * t\n    return {'y': f(1.0, cell)}\n",
    "def run_trial(cell, i, seed):\n    return {'y': (lambda c: c['rate_hz'])(cell)}\n",
    "def run_trial(cell, i, seed):\n    f = lambda t, c=cell: c['rate_hz'] * t\n    return {'y': f(1.0)}\n",
])
def test_a_lambda_called_with_the_run_s_settings_reads_them(src: str) -> None:
    assert [r.keys for r in gk.reads(src, "simulate.py")] == [("rate_hz",)]
    assert _names({"simulate.py": src}) == [("simulate.py", ("rate_hz",))]


def test_a_dict_that_is_not_a_setting_is_not_read_as_one() -> None:
    src = ('def run_cell(cell):\n    table = {"a": 1}\n    other = dict(table)\n'
           '    return {"y": table["a"] + other.get("b", 0) + cell["rate"]}\n')
    assert _names({"simulate.py": src}) == []


KINETICS = '''\
def main():
    record = load()
    cells = []
    for entry in record["cells"]:
        setting = entry["cell"]
        result = {"rate": float(setting["rate"]), "fit_failure_fraction": 0.0, "by_method": {"weighted": {}}}
        cells.append(result)
    supported = [
        cell for cell in cells
        if cell["fit_failure_fraction"] <= 0.01
        and abs(cell["by_method"]["weighted"]["ea_bias_j_mol"]) <= 0.05
    ]
    largest = max(cell["fit_failure_fraction"] for cell in supported)
    return supported, largest
'''
OVER_RECORD = '''\
def main(record):
    for cell in record["cells"]:
        cell["missing"]
    for entry in record["cells"]:
        c = entry["cell"]
        c["missing"]
'''
BUILT_FROM_SETTINGS = '''\
def main(record):
    cells = []
    for entry in record["cells"]:
        c = entry["cell"]
        cells.append(c)
    return [1 for cell in cells if cell["missing"]]
'''
BUILT_FROM_COPIES = '''\
def main(record):
    cells = []
    for entry in record["cells"]:
        cells.append(dict(entry["cell"]))
    return [1 for cell in cells if cell["missing"]]
'''
BUILT_AS_DICT = '''\
def main(record):
    cells = {}
    for entry in record["cells"]:
        cells[entry["key"]] = entry["cell"]
    return [1 for cell in cells.values() if cell["missing"]]
'''
UNTRUSTED_LISTS = (
    'def f(cells):\n    return [1 for cell in cells if cell["missing"]]\n',
    'def f(grid):\n    cells = []\n    cells.extend(grid)\n    cells.append(1)\n    return [1 for cell in cells if cell["missing"]]\n',
    'def f(grid):\n    cells = []\n    cells = grid\n    cells.append(1)\n    return [1 for cell in cells if cell["missing"]]\n',
)


def test_a_loop_variable_over_the_scripts_own_summaries_is_not_a_setting() -> None:
    assert _names({"experiment.py": KINETICS}) == []


def test_a_loop_over_the_record_or_a_parameter_called_cell_is_still_read_as_a_setting() -> None:
    assert _names({"a.py": OVER_RECORD}) == [("a.py", ("missing",)), ("a.py", ("missing",))]
    assert _names({"a.py": 'def f(cell):\n    return cell["missing"]\n'}) == [("a.py", ("missing",))]
    for src in UNTRUSTED_LISTS:  # a list not plainly built here (a parameter, an extend, a second assignment) is not trusted
        assert _names({"a.py": src}) == [("a.py", ("missing",))], src


def test_a_comprehension_over_a_list_built_from_settings_is_still_checked() -> None:
    for src in (BUILT_FROM_SETTINGS, BUILT_FROM_COPIES, BUILT_AS_DICT):
        assert _names({"a.py": src}) == [("a.py", ("missing",))], src


def test_no_grid_means_nothing_to_compare_with() -> None:
    assert _names({"simulate.py": SIM_BAD}, {}) == [] and _names({"simulate.py": SIM_BAD}, None) == []
    assert gk.reads("def broken(:\n", "x.py") == []


def test_tests_and_caches_of_the_code_folder_are_not_read(tmp_path: Path) -> None:
    code = tmp_path / "code"
    (code / "tests").mkdir(parents=True)
    (code / "simulate.py").write_text(SIM_OK, encoding="utf-8")
    (code / "tests" / "test_x.py").write_text(SIM_BAD, encoding="utf-8")
    (code / "test_y.py").write_text(SIM_BAD, encoding="utf-8")
    assert list(gk.code_files(code)) == ["simulate.py"]


# ---- through the engine: sent back, repaired, and an honest stop when it stays -------------------------------------------


class _Client:
    def __init__(self, replies: list[str], asked: list[str]) -> None:
        self.replies, self.asked = replies, asked
        self.last_usage = None
        self.last_model = "test-model"

    async def chat(self, messages: list[dict[str, str]], **kw: Any) -> str:
        self.asked.append(messages[-1]["content"])
        return self.replies[min(len(self.asked), len(self.replies)) - 1]


def _engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replies: list[str]) -> tuple[Engine, list[str]]:
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    asked: list[str] = []
    eng = Engine(Config(
        topic="a neutral topic", title="keys", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, oracle_check="off"),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60, split_analysis=True),
        knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "outputs"),
    ))
    eng._client = _Client(replies, asked)  # type: ignore[assignment]
    eng._protocol_block = lambda state: PROTOCOL  # type: ignore[method-assign]
    (eng.quest_root / "code").mkdir(parents=True, exist_ok=True)
    (eng.quest_root / "code" / "simulate.py").write_text(SIM_BAD, encoding="utf-8")
    return eng, asked


@pytest.mark.asyncio
async def test_a_misread_name_is_sent_back_and_a_repair_that_reads_the_right_one_is_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    eng, asked = _engine(tmp_path, monkeypatch, [json.dumps({"code": SIM_OK, "patch_summary": "renamed"})])
    assert await eng._static_gate({}) is None
    assert len(asked) == 1 and "reads a setting the study does not have" in asked[0] and "`rate_hz` or `freq`" in asked[0]
    assert (eng.quest_root / "code" / "simulate.py").read_text(encoding="utf-8") == SIM_OK
    assert not (eng.quest_root / "needs" / "grid_keys_check.json").exists()
    # The same text is not read twice.
    assert await eng._static_gate({}) is None and len(asked) == 1


@pytest.mark.asyncio
async def test_a_misread_name_that_stays_ends_in_the_honest_stop_without_asking_a_person(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    eng, asked = _engine(tmp_path, monkeypatch, [json.dumps({"code": SIM_BAD, "patch_summary": "no change"})])
    stop = await eng._static_gate({})
    assert len(asked) == 2, "asked twice, the code kept as written"
    assert stop is not None and stop["exec_give_up_reason"]
    assert "the study does not have" in stop["exec_result"]["cannot_repair"]
    assert "`rate_hz`" in stop["exec_result"]["cannot_repair"]
    record = json.loads((eng.quest_root / "needs" / "grid_keys_check.json").read_text(encoding="utf-8"))
    assert record["reads"][0]["names"] == ["rate_hz", "freq"] and record["settings_of_the_study"] == ["rate", "size"]
    # A resume of the same text asks nothing more and stops the same way.
    again = await eng._static_gate({})
    assert again is not None and len(asked) == 2


@pytest.mark.asyncio
async def test_a_repair_that_is_not_the_same_code_is_not_used(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for i, reply in enumerate(("{}", json.dumps({"code": "def broken(:\n"}), json.dumps({"code": SIM_BAD}))):
        eng, asked = _engine(tmp_path / str(i), monkeypatch, [reply])
        assert await eng._static_gate({}) is not None
        assert (eng.quest_root / "code" / "simulate.py").read_text(encoding="utf-8") == SIM_BAD, reply


@pytest.mark.asyncio
async def test_code_that_reads_only_the_studys_names_is_not_asked_about(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eng, asked = _engine(tmp_path, monkeypatch, ["{}"])
    (eng.quest_root / "code" / "simulate.py").write_text(SIM_OK, encoding="utf-8")
    assert await eng._static_gate({}) is None and asked == []


def test_a_list_picked_from_settings_by_a_comprehension_is_still_checked() -> None:
    src = ('def main(record):\n    cells = [entry["cell"] for entry in record["cells"]]\n'
           '    picked = [cell for cell in cells if cell["missing"]]\n    return picked\n')
    assert _names({"a.py": src}) == [("a.py", ("missing",))]


KINETICS_MODULE = '''\
import json, os
record = json.load(open(os.environ["FI_TRIALS"]))
cells = []
for entry in record["cells"]:
    setting = entry["cell"]
    result = {"rate": float(setting["rate"]), "fit_failure_fraction": 0.0, "by_method": {"weighted": {}}}
    cells.append(result)
supported = [cell for cell in cells if cell["fit_failure_fraction"] <= 0.01 and cell["by_method"]["weighted"]]
largest = max(cell["fit_failure_fraction"] for cell in supported)
for cell in cells:
    print(cell["by_method"])
'''
FLOW_TEMPLATE = '''\
def get_cell(entry):
    return entry["cell"]


def main(record, labels):
    cells = []
{build}
    return [1 for cell in cells if cell["missing"]]


def second(record, labels):
    cells = []
{build}
    for cell in cells:
        print(cell["missing"])
'''
# Each builds a list of settings in a way the check cannot see through, then reads a name the grid lacks.
FLOW_BUILDS = {
    "spread into a literal": ('    for entry in record["cells"]:\n        setting = entry["cell"]\n'
                              '        result = {**setting, "metric": 1}\n        cells.append(result)'),
    "dict(**setting)": ('    for entry in record["cells"]:\n        setting = entry["cell"]\n'
                        '        cells.append(dict(**setting))'),
    "helper that returns the cell": '    for entry in record["cells"]:\n        cells.append(get_cell(entry))',
    "zip target": '    for cell, _ in zip(record["cells"], labels):\n        cells.append(cell)',
    "enumerate target": '    for i, cell in enumerate(record["cells"]):\n        cells.append(cell)',
    "nested loop": '    for group in record["groups"]:\n        for c in group["cells"]:\n            cells.append(c)',
}


def test_every_way_a_list_of_settings_can_be_built_unseen_is_still_checked() -> None:
    for label, build in FLOW_BUILDS.items():
        src = FLOW_TEMPLATE.format(build=build)
        found = _names({"a.py": src})
        assert found == [("a.py", ("missing",)), ("a.py", ("missing",))], (label, found)


def test_the_scripts_own_summaries_are_not_settings_at_module_level_either() -> None:
    assert _names({"experiment.py": KINETICS_MODULE}) == []
    # ... but a comprehension over something else, in the same module, still is
    src = KINETICS_MODULE + 'other = [1 for cell in record["cells"] if cell["missing"]]\n'
    assert _names({"experiment.py": src}) == [("experiment.py", ("missing",))]


def test_a_summary_dict_that_is_changed_or_spread_is_not_trusted() -> None:
    for body in ('result = {"a": 1}\n        result.update(entry["cell"])\n        cells.append(result)',
                 'result = {"a": 1}\n        result |= entry["cell"]\n        cells.append(result)',
                 'result = {"a": 1, **entry["cell"]}\n        cells.append(result)',
                 'cells.append({"a": 1, **entry["cell"]})'):
        src = ('def main(record):\n    cells = []\n    for entry in record["cells"]:\n        ' + body
               + '\n    return [1 for cell in cells if cell["missing"]]\n')
        assert _names({"a.py": src}) == [("a.py", ("missing",))], body


def test_a_loop_over_settings_inside_a_loop_over_summaries_is_still_checked() -> None:
    head = ('def main(record):\n    cells = []\n    for entry in record["cells"]:\n        cells.append({"a": 1})\n')
    as_loop = head + ('    for cell in cells:\n        print(cell["by_method"])\n'
                      '        for cell in record["cells"]:\n            print(cell["missing"])\n')
    as_comprehension = head + ('    for cell in cells:\n        print(cell["by_method"])\n'
                               '        print([1 for cell in record["cells"] if cell["missing"]])\n')
    as_inner_comprehension = head + ('    print([[cell["missing"] for cell in record["cells"]] for cell in cells])\n')
    for src in (as_loop, as_comprehension, as_inner_comprehension):
        assert _names({"a.py": src}) == [("a.py", ("missing",))], src


def test_a_summary_dict_filled_by_item_assignment_is_not_trusted() -> None:
    for body in ('result = {}\n        for k, v in entry["cell"].items():\n            result[k] = v\n        cells.append(result)',
                 'result = {"a": 1}\n        result["b"] = entry["cell"]["rate"]\n        cells.append(result)',
                 'result = {"a": 1}\n        result["a"] += 1\n        cells.append(result)',
                 'result = {"a": 1}\n        fill(result, entry["cell"])\n        cells.append(result)'):
        base = 'def main(record):\n    cells = []\n    for entry in record["cells"]:\n        ' + body + '\n'
        for tail in ('    return [1 for cell in cells if cell["missing"]]\n',
                     '    for cell in cells:\n        print(cell["missing"])\n'):
            assert _names({"a.py": base + tail}) == [("a.py", ("missing",))], (body, tail)
