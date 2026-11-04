"""How the code request asks for the function ``trial`` when FI writes the frame of simulate.py (``execution.code_skeleton``):
the frame is shown first and ``trial`` is asked for beside experiment.py, with nothing that asks simulate.py for an entry point,
a seed or a loop. With the frame off the request is what it was. Made-up topic: a colony's growth. No real model is called."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pytest

from core import function_steps, skeleton
from core.config import Config, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig
from core.engine import Engine

DESIGN: dict[str, Any] = {"hypothesis": "growth rises with the rate", "protocol": {
    "grid": {"rate": [0.1, 0.2, 0.3], "dose": [1, 2, 3]}, "runs_per_setting": 5,
    "metrics": [{"id": "growth", "kind": "mean"}, {"id": "survival", "kind": "proportion"}]}}

OUTLINE: dict[str, Any] = {
    "scaffold": ("import json\n\n\ndef run_cell(cell):\n    raise NotImplementedError('filled')\n\n\n"
                 "def summarise(rows):\n    raise NotImplementedError('filled')\n\n\nif __name__ == '__main__':\n    pass\n"),
    "functions": [{"name": "run_cell", "signature": "def run_cell(cell) -> dict:", "one_line_purpose": "one setting"},
                  {"name": "summarise", "signature": "def summarise(rows) -> dict:", "one_line_purpose": "the table"}],
    "result_json_template": "{}", "deps": ["numpy"]}

STATE: dict[str, Any] = {"design": DESIGN, "implement_outline": OUTLINE}

TRIAL = '''
def trial(cell, ctx):
    out = TrialResult()
    out["growth"] = cell["rate"] * cell["dose"] + ctx.rng.random() * 0.01
    out["survival_count"] = 1
    out["survival_total"] = 1
    return out
'''
PACKAGE = ("```python\n# file: colony/__init__.py\nfrom . import model\n```\n\n"
           "```python\n# file: colony/model.py\ndef growth(rate, dose):\n    return rate * dose\n```\n\n")
EXPERIMENT = ("```python\n# file: experiment.py\nimport json\nprint('RESULT_JSON: ' + json.dumps({}))\n```\n\n" + PACKAGE
              + "DEPS: numpy\n")
BOTH = ("```python\n# file: simulate.py\n" + TRIAL + "```\n\n" + EXPERIMENT)

# What a request must not ask of simulate.py when FI writes its frame (outside the frame's own listing).
ASKS = re.compile(r"run_trial|run_cell|FI_REPLICATE_SEED|FI_PILOT|FI_ORACLE|complete script|Single Python file|single-file")


def _engine(tmp_path: Path, *, skeleton_on: bool) -> Engine:
    cfg = Config(topic="a colony's growth", title="colony", provider=ProviderConfig(name="openai"),
                 execution=ExecutionConfig(sandbox="venv", timeout_s=120, split_analysis=True, code_skeleton=skeleton_on),
                 knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "outputs"))
    engine = Engine(cfg)
    engine.quest_root.mkdir(parents=True, exist_ok=True)

    async def no_install(packages):  # noqa: ANN001
        return []

    engine._install_packages = no_install  # type: ignore[method-assign]
    return engine


class _Model:
    """A fake model: answers by what the request is (the trial request, or the code request)."""

    def __init__(self, trials: list[str], code: str) -> None:
        self.trials, self.code, self.prompts = list(trials), code, []

    async def chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        self.prompts.append(prompt)
        if prompt.startswith("You are the **Trial Function"):
            reply = self.trials[min(sum(p.startswith("You are the **Trial Function") for p in self.prompts), len(self.trials)) - 1]
            return "```python\n" + reply + "```"
        return self.code

    async def aclose(self) -> None:
        return None


async def _implement(tmp_path: Path, *, skeleton_on: bool, model: _Model | None = None) -> tuple[Engine, _Model]:
    engine = _engine(tmp_path, skeleton_on=skeleton_on)
    await engine.executor.setup(engine.quest_root)
    (engine.quest_root / "code").mkdir(parents=True, exist_ok=True)
    model = model or _Model([TRIAL], BOTH)
    engine._client = model
    await engine._node_implement(dict(STATE))
    return engine, model


def _sans_frame(prompt: str) -> str:
    return re.sub(r"```python\n.{0,200}?# >>> FI frame.*?# <<< FI frame\n```", "<FRAME>", prompt, flags=re.S)


@pytest.mark.asyncio
async def test_with_the_frame_off_the_code_request_is_what_it_was(tmp_path: Path) -> None:
    """The request hash is that of the commit before this change (6ad6c160)."""
    engine, model = await _implement(tmp_path, skeleton_on=False)
    text = model.prompts[0].replace(str(engine.quest_root), "<quest>").replace(str(tmp_path), "<tmp>")
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == "cb29b006a3c41edc2c87a9193062533bc94a569f4ddf13bf2b72ad4296636a6d"


@pytest.mark.asyncio
async def test_the_frame_is_shown_first_and_asks_simulate_py_for_nothing_but_trial(tmp_path: Path) -> None:
    engine, model = await _implement(tmp_path, skeleton_on=True)
    prompt = model.prompts[0]
    assert prompt.startswith("FI writes the frame of simulate.py"), "the frame instruction is the first thing in the request"
    assert "ONLY the function `trial(cell, ctx)`" in prompt[:400]
    assert "add no import of `fi_frame`" in prompt[:700], "a model shown the frame's import line must not copy it"
    assert "def trial(cell, ctx):" in prompt and "# >>> FI frame" in prompt, "the frame's own code is shown with the `trial` stub"
    text = _sans_frame(prompt)
    assert "<FRAME>" in text
    # the instructions (not the outline's description of its own program) hold nothing for run_trial / run_cell / the seed / main
    instructions = text.split("## Outline")[0] + text.split("## Wall-time limit")[-1]
    assert not ASKS.search(instructions), ASKS.search(instructions).group(0)
    assert "run_cell" not in json.dumps(engine._frame_outline(STATE, OUTLINE)), "the entry points are not in the outline shown"
    assert "def summarise" in json.dumps(engine._frame_outline(STATE, OUTLINE)), "the analysis side of the outline stays"
    assert "# file: simulate.py" in prompt and "# file: experiment.py" in prompt
    assert len(model.prompts) == 1, "one request: the function is asked for beside experiment.py"
    sim = (engine.quest_root / "code" / "simulate.py").read_text(encoding="utf-8")
    assert skeleton.has_frame(sim) and "def trial(cell, ctx)" in sim and "ctx.rng.random()" in sim
