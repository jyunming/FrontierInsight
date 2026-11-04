"""A single failing function does not throw away the package; a fault of the environment never costs a function a repair
(core/function_steps.py). Fake models, neutral models (a damped spring)."""

from __future__ import annotations

import logging
import string
from pathlib import Path
from typing import Any

import pytest

from core import function_steps as fs
from core.experiment_deps import IMPORT_TO_PIP
from tests.test_function_steps import BAD, GOOD, OUTLINE, PROMPTS, PROTOCOL, _Model, _run, _specs, _which

AGENTS = Path(__file__).resolve().parents[1] / "agents"
WITH_MODULE = {**PROMPTS, "module": string.Template((AGENTS / "implement_module_repair.md").read_text(encoding="utf-8"))}


def _module(**functions: str) -> str:
    """A whole module from function texts (the imports of each once, at the top)."""
    texts = {**GOOD, **functions}
    return "import math\n\n\n" + "\n\n".join(t.replace("import math\n\n\n", "") for t in texts.values())


class _ModuleModel(_Model):
    """A fake model that also answers the repair of the whole module (a request with no single function in it)."""

    def __init__(self, replies: dict[str, list[str]], module_replies: list[str]) -> None:
        super().__init__(replies)
        self.module_replies, self.module_prompts = list(module_replies), []

    async def chat(self, prompt: str) -> str:
        if prompt.lstrip().startswith("You are repairing the model package"):
            self.prompts.append(prompt)
            self.module_prompts.append(prompt)
            n = len(self.module_prompts)
            return "```python\n" + self.module_replies[min(n, len(self.module_replies)) - 1] + "\n```"
        return await super().chat(prompt)


def _filler(root: Path, model: Any, *, run: Any = None, protocol: dict[str, Any] | None = None, **kw: Any) -> fs.FunctionFiller:
    async def default_run(argv: list[str]) -> tuple[int, str, str]:
        return await _run(argv, root)

    return fs.FunctionFiller(quest_root=root, package="spring", protocol=protocol or PROTOCOL,
                             summary="a mass on a damped spring", prompts=WITH_MODULE, chat=model.chat,
                             run=run or default_run, log=logging.getLogger("t"), **kw)


@pytest.mark.asyncio
async def test_function_two_of_three_keeps_failing_functions_one_and_three_stay_done_and_two_is_repaired_with_the_module(
        tmp_path: Path) -> None:
    model = _ModuleModel({"frequency": [GOOD["frequency"]], "quality": [BAD["quality"]], "decay": [GOOD["decay"]]},
                         module_replies=[_module()])
    outcome = await _filler(tmp_path, model).run_all(_specs())
    assert outcome.status == "done"
    kinds = [("module" if p.lstrip().startswith("You are repairing the model package") else _which(p)) for p in model.prompts]
    assert kinds == ["frequency", "quality", "quality", "quality", "decay", "module"], (
        "the failing function spends its own repairs, the next one is still written, and only then the module is repaired")
    prompt = model.module_prompts[0]
    passing, rest = prompt.split("# The functions that already pass (they must keep passing)", 1)[1].split(
        "# The functions to fix", 1)
    assert "`frequency`" in passing and "`decay`" in passing and "`quality`" not in passing
    failing, module_text = rest.split("# The module as it is now", 1)
    assert "`quality`" in failing and "Q = m * w / c" in failing and "`frequency`" not in failing
    assert "def quality" in module_text, "the whole module is shown"
    for secret in ("12.0", "'m': 2.0", '"m": 2'):
        assert secret not in prompt, "the worked example never reaches the request"
    record = fs.load(tmp_path)
    assert {e["status"] for e in record["functions"].values()} == {"done"} and record["module_repairs"] == 1
    assert fs.counts(record) == (3, 3) and record["functions_done"] == 3 and fs.is_ready(tmp_path, "spring")
    text = (tmp_path / "code" / "spring" / "model.py").read_text(encoding="utf-8")
    assert "return m * w / c" in text and "# E2" in text


@pytest.mark.asyncio
async def test_a_reply_of_the_module_that_loses_a_function_is_not_used(tmp_path: Path) -> None:
    only_two = "import math\n\n\n" + GOOD["frequency"].replace("import math\n\n\n", "")
    model = _ModuleModel({"frequency": [GOOD["frequency"]], "quality": [BAD["quality"]], "decay": [GOOD["decay"]]},
                         module_replies=[only_two])
    outcome = await _filler(tmp_path, model).run_all(_specs())
    assert outcome.status == "partial"
    assert "def decay" in (tmp_path / "code" / "spring" / "model.py").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_a_missing_package_is_installed_by_the_dependency_path_and_costs_no_repair(tmp_path: Path) -> None:
    uses = "import fakepkg_ok\n\n\n# E1\ndef frequency(k, m):\n    return fakepkg_ok.root(k / m)\n"
    asked: list[list[str]] = []

    async def install(packages: list[str]) -> list[tuple[str, str]]:
        asked.append(packages)
        (tmp_path / "code" / "fakepkg_ok.py").write_text("from math import sqrt as root\n", encoding="utf-8")
        return []

    model = _ModuleModel({"frequency": [uses], "quality": [GOOD["quality"]], "decay": [GOOD["decay"]]}, module_replies=[""])
    outcome = await _filler(tmp_path, model, install=install, declared=["fakepkg_ok>=1"]).run_all(_specs())
    assert outcome.status == "done" and asked == [["fakepkg_ok>=1"]]
    assert len(model.prompts) == 3, "one fill each: the missing package cost no repair"
    record = fs.load(tmp_path)
    assert record["functions"]["frequency"]["attempts"] == 1 and record["installed"] == ["fakepkg_ok>=1"]


@pytest.mark.asyncio
async def test_a_check_that_cannot_run_is_run_again_and_then_left_out_without_costing_a_repair(tmp_path: Path) -> None:
    calls = {"tests": 0}

    async def flaky(argv: list[str]) -> tuple[int, str, str]:
        if argv[:1] == ["-c"] and "EQUATION_MEASURED" in argv[1]:  # FI's measuring run
            calls["tests"] += 1
            if calls["tests"] == 1:
                return -1, "", "OSError('could not start')"
        return await _run(argv, tmp_path)

    model = _ModuleModel({k: [v] for k, v in GOOD.items()}, module_replies=[""])
    outcome = await _filler(tmp_path, model, run=flaky).run_all(_specs())
    assert outcome.status == "done" and len(model.prompts) == 3
    assert fs.load(tmp_path)["functions"]["frequency"]["attempts"] == 1, "one test run failed to start; none was blamed"

    silent_root = tmp_path / "silent"

    async def silent(argv: list[str]) -> tuple[int, str, str]:
        if argv[:1] == ["-c"] and "EQUATION_MEASURED" in argv[1]:  # FI's measuring run
            return 0, "", ""  # the run printed nothing, every time (an early exit, a hang)
        return await _run(argv, silent_root)

    # review pass 2 N2: a run that prints no result twice fails the test (the function is repaired); it is never left
    # out as "not tested" (before: done, 3 requests, "could not be run here").
    model = _ModuleModel({k: [v] * 4 for k, v in GOOD.items()}, module_replies=[""])
    await _filler(silent_root, model, run=silent).run_all(_specs())
    record = fs.load(silent_root)
    assert len(model.prompts) > 3, "the failed test asked for a repair"
    assert "frequency" not in (record.get("unchecked") or {}) and "E1" not in (record.get("untested") or {})


@pytest.mark.asyncio
async def test_an_example_whose_inputs_are_not_the_outline_s_parameters_is_fi_s_own_mismatch_not_the_function_s(
        tmp_path: Path) -> None:
    eqs = [{**PROTOCOL["model"]["equations"][0],
            "example": {"inputs": {"k": 8.0, "mass": 2.0}, "expected_formula": "sqrt(k / mass)"}},
           *PROTOCOL["model"]["equations"][1:]]
    protocol = {"model": {**PROTOCOL["model"], "equations": eqs}}
    model = _ModuleModel({k: [v] for k, v in GOOD.items()}, module_replies=[""])
    # FI's own reading of the outline now names the example's inputs, so a mismatch is possible only for a spec made before
    # the example changed (here: made from the earlier protocol); the filler still does not blame the function for it.
    assert fs.model_functions(OUTLINE, protocol)[0][0].params() == ["k", "mass"]
    specs, _ = fs.model_functions(OUTLINE, PROTOCOL)
    outcome = await _filler(tmp_path, model, protocol=protocol).run_all(specs)
    assert outcome.status == "done" and len(model.prompts) == 3
    assert "does not fit the outline's function" in fs.load(tmp_path)["unchecked"]["frequency"]


@pytest.mark.asyncio
async def test_a_function_left_as_its_placeholder_is_never_promoted_to_done_by_the_module_repair(tmp_path: Path) -> None:
    # decay (an equation with no worked example, so a placeholder passes its import check) never imports, by its own fault;
    # the module reply gives its placeholder back untouched.
    bad_import = "from math import not_there_at_all\n\n\n# E3\ndef decay(x0, g, t):\n    return 1.0\n"
    stub_back = ("import math\n\n\n" + GOOD["frequency"].replace("import math\n\n\n", "") + "\n\n" + GOOD["quality"]
                 + "\n\n\n# E3\ndef decay(x0, g, t):\n    raise NotImplementedError(\"not written yet\")\n")
    model = _ModuleModel({"frequency": [GOOD["frequency"]], "quality": [GOOD["quality"]], "decay": [bad_import]},
                         module_replies=[stub_back])
    outcome = await _filler(tmp_path, model).run_all(_specs())
    assert outcome.status == "fell_back" and outcome.failed == "decay", "a function with no code is not written"
    assert len(model.module_prompts) == fs.MODULE_REPAIRS
    assert fs.load(tmp_path)["functions"]["decay"]["status"] == "failed"


@pytest.mark.asyncio
async def test_only_a_known_or_declared_package_is_installed_an_unknown_import_never_is(tmp_path: Path) -> None:
    asked: list[list[str]] = []

    async def install(packages: list[str]) -> list[tuple[str, str]]:
        asked.append(packages)
        return [(p, "not installed here") for p in packages]

    uses = "import skfuzzy\n\n\n# E1\ndef frequency(k, m):\n    return 2.0\n"
    unknown = "import some_unlisted_module\n\n\n# E1\ndef frequency(k, m):\n    return 2.0\n"
    only = [s for s in _specs() if s.name == "frequency"]
    await _filler(tmp_path / "a", _ModuleModel({"frequency": [uses]}, [""]), install=install).run_all(only)
    assert asked and asked[0] == [IMPORT_TO_PIP["skfuzzy"]], "a name FI knows is installed under its pip name"
    asked.clear()
    await _filler(tmp_path / "b", _ModuleModel({"frequency": [unknown]}, [""]), install=install).run_all(only)
    assert asked == [], "an unknown name is not installed"
