"""The model's functions written and checked one at a time (core/function_steps.py), with fake models and neutral
models (a damped spring). The expected value of a worked example must never reach a request to the model."""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
import string
import sys
from pathlib import Path
from typing import Any

import pytest

from core import function_steps as fs

AGENTS = Path(__file__).resolve().parents[1] / "agents"
PROMPTS = {"fill": string.Template((AGENTS / "implement_function.md").read_text(encoding="utf-8")),
           "repair": string.Template((AGENTS / "implement_function_repair.md").read_text(encoding="utf-8"))}

PROTOCOL = {"model": {"summary": "a mass on a damped spring", "equations": [
    {"id": "E1", "formula": "w = sqrt(k / m)", "role": "generates", "source": "derivation", "derivation": "Newton's law",
     "example": {"inputs": {"k": 8.0, "m": 2.0}, "expected_formula": "sqrt(k / m)"}},   # 2.0
    {"id": "E2", "formula": "Q = m * w / c", "role": "generates", "source": "derivation",
     "example": {"inputs": {"m": 2.0, "w": 3.0, "c": 0.5}, "expected_formula": "m * w / c"}},   # 12.0
    {"id": "E3", "formula": "x(t) = x0 * exp(-g * t)", "role": "generates", "source": "derivation"},  # no example
]}}
OUTLINE = {"scaffold": "x = 1", "functions": [
    {"name": "quality", "signature": "def quality(m, w, c) -> float:", "one_line_purpose": "the quality factor",
     "implements": "E2", "depends_on": ["frequency"]},
    {"name": "frequency", "signature": "def frequency(k, m) -> float:", "one_line_purpose": "the natural frequency",
     "implements": "E1"},
    {"name": "decay", "signature": "def decay(x0, g, t) -> float:", "one_line_purpose": "the amplitude at t",
     "implements": "E3"},
    {"name": "run_cell", "signature": "def run_cell(cell) -> dict:", "one_line_purpose": "one setting"},
]}

GOOD = {
    "frequency": "import math\n\n\n# E1\ndef frequency(k, m):\n    return math.sqrt(k / m)\n",
    "quality": "# E2\ndef quality(m, w, c):\n    return m * w / c\n",
    "decay": "import math\n\n\n# E3\ndef decay(x0, g, t):\n    return x0 * math.exp(-g * t)\n",
}
BAD = {
    "frequency": "import math\n\n\n# E1\ndef frequency(k, m):\n    return math.sqrt(m / k)\n",
    "quality": "# E2\ndef quality(m, w, c):\n    return w / c\n",
    "decay": "import math\n\n\n# E3\ndef decay(x0, g, t):\n    return x0 * math.exp(g * t)\n",
}


def _which(prompt: str) -> str:
    """The function a request is about: the first one in its `The function` block (not one it depends on)."""
    found = re.search(r"# The function[^\n]*\n\n```python\n(?:#[^\n]*\n)*def (\w+)\(", prompt)
    assert found, prompt[-600:]
    return found.group(1)


class _Model:
    """A fake model: for each function a list of replies (the last is repeated), and a record of every request."""

    def __init__(self, replies: dict[str, list[str]]) -> None:
        self.replies, self.prompts, self.count = {k: list(v) for k, v in replies.items()}, [], {}

    async def chat(self, prompt: str) -> str:
        self.prompts.append(prompt)
        name = _which(prompt)
        n = self.count[name] = self.count.get(name, 0) + 1
        replies = self.replies[name]
        return "```python\n" + replies[min(n, len(replies)) - 1] + "```"


async def _run(argv: list[str], root: Path) -> tuple[int, str, str]:
    code = root / "code"
    for cache in code.glob("**/__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)
    proc = await asyncio.create_subprocess_exec(sys.executable, "-B", *argv, cwd=str(code),
                                                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    out, err = await proc.communicate()
    return proc.returncode or 0, out.decode(), err.decode()


def _filler(root: Path, model: _Model, **kw: Any) -> fs.FunctionFiller:
    async def run(argv: list[str]) -> tuple[int, str, str]:
        return await _run(argv, root)

    return fs.FunctionFiller(quest_root=root, package="spring", protocol=PROTOCOL, summary="a mass on a damped spring",
                             prompts=PROMPTS, chat=model.chat, run=run, log=logging.getLogger("t"), **kw)


def _specs() -> list[fs.Spec]:
    specs, _notes = fs.model_functions(OUTLINE, PROTOCOL)
    return specs


# --- what the outline names ----------------------------------------------------------------------------------------


def test_the_functions_come_in_dependency_order_and_only_for_equations_the_plan_lists() -> None:
    specs, notes = fs.model_functions(OUTLINE, PROTOCOL)
    assert [s.name for s in specs] == ["frequency", "quality", "decay"], "frequency is written before quality calls it"
    assert [s.equation for s in specs] == ["E1", "E2", "E3"] and notes == []
    plain = {"functions": [{k: v for k, v in f.items() if k != "depends_on"} for f in OUTLINE["functions"]]}
    assert [s.name for s in fs.model_functions(plain, PROTOCOL)[0]] == ["quality", "frequency", "decay"], (
        "without depends_on the outline's order")
    cyc = {"functions": [{**OUTLINE["functions"][0], "depends_on": ["frequency"]},
                         {**OUTLINE["functions"][1], "depends_on": ["quality"]}, OUTLINE["functions"][2]]}
    assert len(fs.model_functions(cyc, PROTOCOL)[0]) == 3, "a cycle falls back to the outline's order"


def test_the_step_is_not_used_when_an_equation_has_no_function_or_a_signature_cannot_be_read() -> None:
    without = {"functions": [f for f in OUTLINE["functions"] if f["name"] != "decay"]}
    specs, notes = fs.model_functions(without, PROTOCOL)
    assert specs == [] and "no function for equation E3" in notes[-1]
    broken = {"functions": [{**OUTLINE["functions"][1], "signature": "def frequency(k, m"}, *OUTLINE["functions"][2:]]}
    assert fs.model_functions(broken, PROTOCOL)[0] == []
    assert fs.model_functions({"functions": [OUTLINE["functions"][3]]}, PROTOCOL)[0] == []
    assert fs.model_functions(None, PROTOCOL)[0] == []


# --- putting a function into the module --------------------------------------------------------------------------------


def test_a_reply_replaces_only_its_own_function_keeps_the_label_and_adds_its_imports_and_helpers() -> None:
    specs = _specs()
    module = fs.initial_module(specs, "a spring")
    assert all(fs.is_stub(module, s.name) for s in specs)
    reply = "import math\n\nSCALE = 2.0\n\ndef _root(x):\n    return math.sqrt(x)\n\n\ndef frequency(k, m):\n    return _root(k / m)\n"
    new, why = fs.replace_function(module, "frequency", reply, signature=specs[0].signature, equation="E1")
    assert why == "" and new is not None
    assert not fs.is_stub(new, "frequency") and fs.is_stub(new, "quality") and fs.is_stub(new, "decay")
    assert "import math" in new and "def _root" in new and "SCALE = 2.0" in new
    assert "# E1\ndef frequency(k, m):" in new, "the label is put above the function when the reply left it out"
    assert new.index("import math") < new.index("def quality")


def test_a_reply_that_changes_the_parameters_or_holds_no_such_function_is_refused() -> None:
    specs = _specs()
    module = fs.initial_module(specs)
    new, why = fs.replace_function(module, "frequency", "def frequency(m, k):\n    return 1\n",
                                   signature=specs[0].signature, equation="E1")
    assert new is None and "parameters must be exactly (k, m)" in why
    assert fs.replace_function(module, "frequency", "def other(k, m):\n    return 1\n",
                               signature=specs[0].signature, equation="E1")[0] is None
    assert fs.replace_function(module, "frequency", "def frequency(k, m:\n", signature=specs[0].signature,
                               equation="E1")[0] is None
    assert fs.replace_function(module, "frequency", "", signature=specs[0].signature, equation="E1")[0] is None


# --- the step ---------------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_each_function_is_checked_before_the_next_and_the_call_count_is_one_per_function(tmp_path: Path) -> None:
    model = _Model({k: [v] for k, v in GOOD.items()})
    outcome = await _filler(tmp_path, model).run_all(_specs())
    assert outcome.status == "done" and outcome.calls == 3 and len(model.prompts) == 3
    assert [_which(p) for p in model.prompts] == ["frequency", "quality", "decay"]
    text = (tmp_path / "code" / "spring" / "model.py").read_text(encoding="utf-8")
    assert not any(fs.is_stub(text, n) for n in GOOD)
    record = fs.load(tmp_path)
    assert record["status"] == "done" and {e["status"] for e in record["functions"].values()} == {"done"}
    assert "E3" in record["untested"], "an equation with no worked example is written but not tested, and said so"
    assert fs.is_ready(tmp_path, "spring")
    # A request holds one equation and one function, not the study.
    assert "w = sqrt(k / m)" in model.prompts[0] and "Q = m * w" not in model.prompts[0]
    assert "def frequency(k, m)" in model.prompts[1], "the function quality depends on is shown by its signature"
    # The tested package is also a package the unit tests FI writes can import.
    assert (tmp_path / "code" / "spring" / "__init__.py").is_file()


@pytest.mark.asyncio
async def test_a_function_written_wrong_is_repaired_alone_and_the_next_function_goes_on(tmp_path: Path) -> None:
    model = _Model({"frequency": [BAD["frequency"], GOOD["frequency"]], "quality": [GOOD["quality"]],
                    "decay": [GOOD["decay"]]})
    outcome = await _filler(tmp_path, model).run_all(_specs())
    assert outcome.status == "done" and outcome.calls == 4
    assert [_which(p) for p in model.prompts] == ["frequency", "frequency", "quality", "decay"]
    repair = model.prompts[1]
    assert "repairing ONE function" in repair and "def frequency(k, m)" in repair
    assert "def quality" not in repair and "def decay" not in repair, "the repair sees that function, not the others"
    assert "w = sqrt(k / m)" in repair and "does not agree" in repair
    # The expected value, the inputs and the formula's result never reach a request to the model.
    for prompt in model.prompts:
        assert "12.0" not in prompt and "8.0" not in prompt and "'k': 8" not in prompt and '"k": 8' not in prompt
    assert fs.load(tmp_path)["functions"]["frequency"]["attempts"] == 2


@pytest.mark.asyncio
async def test_a_function_that_stays_wrong_ends_the_step_within_its_bound_and_the_code_is_written_whole(tmp_path: Path) -> None:
    model = _Model({"frequency": [BAD["frequency"]], "quality": [GOOD["quality"]], "decay": [GOOD["decay"]]})
    outcome = await _filler(tmp_path, model, repairs=2).run_all(_specs())
    assert outcome.status == "fell_back" and outcome.failed == "frequency"
    assert outcome.calls == 3 and len(model.prompts) == 3, "one fill and two repairs, then the step ends"
    assert not (tmp_path / "code" / "spring").exists(), "the half-written package is removed"
    assert fs.load(tmp_path)["status"] == "fell_back" and not fs.is_ready(tmp_path, "spring")
    again = _Model({k: [v] for k, v in GOOD.items()})
    assert (await _filler(tmp_path, again).run_all(_specs())).status == "skipped" and again.prompts == [], (
        "a step that fell back is not tried again for the same plan")


@pytest.mark.asyncio
async def test_the_most_the_step_may_ask_in_a_quest_is_a_fixed_number(tmp_path: Path) -> None:
    model = _Model({k: [BAD[k]] for k in GOOD})
    outcome = await _filler(tmp_path, model, repairs=5, max_calls=4).run_all(_specs())
    assert outcome.status == "fell_back" and len(model.prompts) == 4


@pytest.mark.asyncio
async def test_a_resume_does_not_write_a_function_that_is_done(tmp_path: Path) -> None:
    class _Dies(_Model):
        async def chat(self, prompt: str) -> str:
            if _which(prompt) == "quality":
                raise KeyboardInterrupt  # the process is stopped during the second function
            return await super().chat(prompt)

    first = _Dies({k: [v] for k, v in GOOD.items()})
    with pytest.raises(KeyboardInterrupt):
        await _filler(tmp_path, first).run_all(_specs())
    assert [_which(p) for p in first.prompts] == ["frequency"]
    record = fs.load(tmp_path)
    assert record["functions"]["frequency"]["status"] == "done" and record["functions"]["quality"]["status"] == "pending"
    second = _Model({k: [v] for k, v in GOOD.items()})
    outcome = await _filler(tmp_path, second).run_all(_specs())
    assert outcome.status == "done"
    assert [_which(p) for p in second.prompts] == ["quality", "decay"], "frequency is not written again"
    third = _Model({})
    assert (await _filler(tmp_path, third).run_all(_specs())).status == "done" and third.prompts == []


@pytest.mark.asyncio
async def test_a_different_plan_writes_the_functions_again(tmp_path: Path) -> None:
    await _filler(tmp_path, _Model({k: [v] for k, v in GOOD.items()})).run_all(_specs())
    changed = {"model": {"equations": [{**PROTOCOL["model"]["equations"][0], "formula": "w = sqrt(k / (2 * m))"},
                                       *PROTOCOL["model"]["equations"][1:]]}}
    assert fs.spec_hash(_specs(), PROTOCOL) != fs.spec_hash(_specs(), changed)


@pytest.mark.asyncio
async def test_a_missing_third_party_module_is_the_environment_s_problem_not_the_function_s(tmp_path: Path) -> None:
    uses = "import not_installed_anywhere\n\n\n# E1\ndef frequency(k, m):\n    return 2.0\n"
    only = [s for s in _specs() if s.name == "frequency"]
    model = _Model({"frequency": [uses]})
    outcome = await _filler(tmp_path, model).run_all(only)
    # The import check skips the missing module; the equation test (which imports it too) then reports the error.
    assert outcome.status == "fell_back" and len(model.prompts) == 3


def test_the_two_prompts_are_templates_with_their_parts() -> None:
    fill = PROMPTS["fill"].substitute(model_block="m", equation_block="E1: y", function_block="def f(a):", related_block="(none)",
                                      constants_block="(none)", package="p")
    assert "ONE equation" in fill and "Function Body" in fill
    repair = PROMPTS["repair"].substitute(equation_block="E1: y", function_block="def f(a): pass", problem="p",
                                          related_block="(none)", package="p")
    assert "repairing ONE function" in repair
