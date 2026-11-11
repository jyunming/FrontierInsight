"""The functions of the study that already ran and passed are kept when the code step is entered again (core/kept_bodies.py).

Made-up topic: a colony's growth, with two figures (one over the rate, one over the dose). No real model is called."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

import pytest

from core import analysis_frame as af
from core import kept_bodies as kb
from core import skeleton, split_run
from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig
from core.engine import Engine

DESIGN: dict[str, Any] = {"hypothesis": "growth rises with the rate", "protocol": {
    "form": "contract", "grid": {"rate": [0.1, 0.2], "dose": [1, 2]}, "runs_per_setting": 3,
    "metrics": [{"id": "growth", "kind": "mean", "role": "result"}],
    "derived": [{"id": "peak", "inputs": ["growth"], "role": "result"}],
    "figures": [{"id": "over_rate", "file": "over_rate.png", "shows": ["growth"], "x": "rate"},
                {"id": "over_dose", "file": "over_dose.png", "shows": ["growth"], "x": "dose"}]}}

TRIAL = '''
def trial(cell, ctx):
    out = TrialResult()
    out["growth"] = cell["rate"] * cell["dose"] + ctx.rng.random() * 0.01
    return out
'''
GOOD = '''
HELPER = 1


def _scale(x):
    return x


def derive_peak(table):
    return max(table.mean("growth", k) for k in table.keys())


def figure_over_rate(fig, table):
    fig.axes[0].plot([1, 2], [1, 2])


def figure_over_dose(fig, table):
    fig.axes[0].plot([1, 2], [2, 1])
'''
WORSE = '''
def derive_peak(table):
    raise RuntimeError("rewritten peak")


def figure_over_rate(fig, table):
    raise RuntimeError("rewritten over_rate")


def figure_over_dose(fig, table):
    raise RuntimeError("rewritten over_dose")
'''


def _reply(derive: str, trial: str = TRIAL) -> str:
    return "```python\n# file: simulate.py\n" + trial + "```\n\n```python\n# file: derive.py\n" + derive + "```\n\nDEPS:\n"


class _Model:
    def __init__(self, code: str) -> None:
        self.code, self.prompts = code, []

    async def chat(self, messages, **kw):  # noqa: ANN001
        self.prompts.append(messages[-1]["content"])
        return self.code

    async def aclose(self) -> None:
        return None


def _engine(tmp_path: Path) -> Engine:
    cfg = Config(topic="a colony's growth", title="colony", provider=ProviderConfig(name="openai"),
                 engine=EngineConfig(max_iterations=1, review_loop=False, exec_reflect_max_iterations=3, ideate_reflect=False),
                 execution=ExecutionConfig(sandbox="venv", timeout_s=120, split_analysis=True, code_skeleton=True),
                 knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "out"))
    engine = Engine(cfg)
    engine.quest_root.mkdir(parents=True, exist_ok=True)

    async def no_install(packages):  # noqa: ANN001
        return []

    engine._install_packages = no_install  # type: ignore[method-assign]
    engine._code_layout = lambda state: None  # type: ignore[method-assign]  # the toy has no package of the model's equations
    return engine


def _later(path: Path, than: Path) -> None:
    t = than.stat().st_mtime + 5
    os.utime(path, (t, t))


def _ran(engine: Engine, status: dict[str, str]) -> None:
    """FI's record of a run (what experiment.py writes) and of its trials, as a run of the code on disk leaves them."""
    root = engine.quest_root
    record = {"schema": af.DERIVED_SCHEMA, "derived": {}, "figures": [], "fi_faults": [],
              "bodies": {n: {"status": s, "outputs": [f"figure:{n[7:]}"], "seconds": 0.1} for n, s in status.items()}}
    (root / "data" / "results").mkdir(parents=True, exist_ok=True)
    (root / af.DERIVED_PATH).write_text(json.dumps(record), encoding="utf-8")
    (root / ".fi").mkdir(exist_ok=True)
    (root / ".fi" / "run_facts.json").write_text(json.dumps({"trials_run": {"trials": 12}}), encoding="utf-8")
    assert kb.write_receipt(root, kb.receipt_inputs(root, None))  # what the run's own copy leaves (core/trial_runner.py)


async def _first(tmp_path: Path) -> tuple[Engine, dict[str, Any]]:
    engine = _engine(tmp_path)
    (engine.quest_root / "code").mkdir(parents=True, exist_ok=True)
    engine._client = _Model(_reply(GOOD))
    state: dict[str, Any] = {"design": DESIGN, "implement_outline": {}}
    state.update(await engine._node_implement(dict(state)))
    return engine, state


def _bodies(engine: Engine) -> dict[str, str]:
    return kb.sources((engine.quest_root / "code" / "derive.py").read_text(encoding="utf-8"))[0]


# ---- the record -----------------------------------------------------------------------------------------------------------------
def test_a_figure_reads_the_settings_it_shows_and_a_quantity_reads_every_setting() -> None:
    changed = copy.deepcopy(DESIGN)
    changed["protocol"]["grid"]["dose"] = [1, 2, 3]
    assert kb.reads(changed, "figure_over_rate") == kb.reads(DESIGN, "figure_over_rate")  # dose is not on its axis
    assert kb.reads(changed, "figure_over_dose") != kb.reads(DESIGN, "figure_over_dose")
    assert kb.reads(changed, "derive_peak") != kb.reads(DESIGN, "derive_peak")
    shown = copy.deepcopy(DESIGN)
    shown["protocol"]["figures"][0]["shows"] = ["growth", "peak"]
    assert kb.reads(shown, "figure_over_rate") != kb.reads(DESIGN, "figure_over_rate")
    assert kb.reads(shown, "figure_over_dose") == kb.reads(DESIGN, "figure_over_dose")


def test_a_function_is_put_back_where_it_was_or_added() -> None:
    text = "import os\n\n\ndef figure_a(fig, table):\n    return 1\n\n\ndef helper():\n    return 2\n"
    out = kb.put_back(text, "figure_a", "def figure_a(fig, table):\n    return 5\n")
    assert out == "import os\n\n\ndef figure_a(fig, table):\n    return 5\n\n\ndef helper():\n    return 2\n"
    assert kb.put_back(text, "figure_b", "def figure_b(fig, table):\n    pass\n").endswith("def figure_b(fig, table):\n    pass\n")
    funcs, rest = kb.sources(text)
    assert list(funcs) == ["figure_a"] and rest == kb.sources(out)[1]


@pytest.mark.asyncio
async def test_only_a_function_that_ran_and_passed_is_recorded(tmp_path: Path) -> None:
    engine, state = await _first(tmp_path)
    assert kb.load(engine.quest_root)["written_for"].keys() == {"trial", "derive_peak", "figure_over_rate", "figure_over_dose"} \
        or "trial" not in kb.load(engine.quest_root)["written_for"]
    _ran(engine, {"derive_peak": "ok", "figure_over_rate": "ok", "figure_over_dose": "failed"})
    got = kb.snapshot(engine.quest_root, DESIGN)
    assert sorted(got) == ["derive_peak", "figure_over_rate"]
    assert "source" in kb.load(engine.quest_root)["bodies"]["figure_over_rate"]


@pytest.mark.asyncio
async def test_code_changed_after_the_last_run_is_not_recorded_as_having_passed(tmp_path: Path) -> None:
    engine, state = await _first(tmp_path)
    _ran(engine, {"derive_peak": "ok", "figure_over_rate": "ok", "figure_over_dose": "ok"})
    derive = engine.quest_root / "code" / "derive.py"
    derive.write_text(derive.read_text(encoding="utf-8").replace("[1, 2], [2, 1]", "[1, 2], [9, 9]"), encoding="utf-8")
    t = (engine.quest_root / af.DERIVED_PATH).stat().st_mtime + 30
    os.utime(derive, (t, t))  # written after the run
    assert kb.snapshot(engine.quest_root, DESIGN) == []


@pytest.mark.asyncio
async def test_what_was_kept_follows_the_content_of_the_files_and_not_the_time_they_were_written(tmp_path: Path) -> None:
    engine, state = await _first(tmp_path)
    _ran(engine, {"derive_peak": "ok", "figure_over_rate": "ok", "figure_over_dose": "ok"})
    derive = engine.quest_root / "code" / "derive.py"
    # a checkout or a copy moves the time forward and leaves the text: still what ran
    t = (engine.quest_root / af.DERIVED_PATH).stat().st_mtime + 300
    os.utime(derive, (t, t))
    assert sorted(kb.snapshot(engine.quest_root, DESIGN)) == ["derive_peak", "figure_over_dose", "figure_over_rate"]
    # a text that changed, with a time before the run's: not what ran
    old = (engine.quest_root / af.DERIVED_PATH).stat().st_mtime - 300
    derive.write_text(derive.read_text(encoding="utf-8").replace("[1, 2], [2, 1]", "[1, 2], [9, 9]"), encoding="utf-8")
    os.utime(derive, (old, old))
    assert kb.snapshot(engine.quest_root, DESIGN) == []


@pytest.mark.asyncio
async def test_no_receipt_keeps_nothing(tmp_path: Path) -> None:
    engine, state = await _first(tmp_path)
    _ran(engine, {"derive_peak": "ok", "figure_over_rate": "ok", "figure_over_dose": "ok"})
    (engine.quest_root / kb.RECEIPT).unlink()
    assert kb.snapshot(engine.quest_root, DESIGN) == []


# ---- the code step entered again ------------------------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_a_second_pass_keeps_every_function_that_ran_and_asks_the_model_for_nothing(tmp_path: Path) -> None:
    engine, state = await _first(tmp_path)
    before = _bodies(engine)
    simulate = (engine.quest_root / "code" / "simulate.py").read_text(encoding="utf-8")
    model = _Model(_reply(WORSE, "def trial(cell, ctx):\n    raise RuntimeError('rewritten')\n"))
    engine._client = model
    engine._scripts_built_from = lambda s, o: {"p": "1"}  # type: ignore[method-assign]
    engine._bodies_kept(state, engine._analysis_bodies(state), {"scaffold": "x"})  # notes what the functions are written for
    _ran(engine, {"derive_peak": "ok", "figure_over_rate": "ok", "figure_over_dose": "ok"})
    kept = engine._bodies_kept(state, engine._analysis_bodies(state), {"scaffold": "x"})
    assert kept is not None and kept.trial and sorted(kept.names) == ["derive_peak", "figure_over_dose", "figure_over_rate"]
    state2 = {**state, "implement_outline": {"scaffold": "x", "functions": []}, "deps": ["numpy"]}
    out = await engine._node_implement(dict(state2))
    assert "numpy" in out["deps"]  # the packages of the kept simulation are still asked for
    assert model.prompts == []  # everything was kept: the model was not asked
    assert _bodies(engine) == before
    assert (engine.quest_root / "code" / "simulate.py").read_text(encoding="utf-8") == simulate


@pytest.mark.asyncio
async def test_a_function_the_reason_names_is_asked_again_and_the_others_stay(tmp_path: Path) -> None:
    engine, state = await _first(tmp_path)
    engine._scripts_built_from = lambda s, o: {"p": "1"}  # type: ignore[method-assign]
    outline = {"scaffold": "x", "functions": []}
    engine._bodies_kept(state, engine._analysis_bodies(state), outline)
    _ran(engine, {"derive_peak": "ok", "figure_over_rate": "ok", "figure_over_dose": "ok"})
    before = _bodies(engine)
    new_dose = "def figure_over_dose(fig, table):\n    fig.axes[0].plot([3], [3])\n"
    model = _Model(_reply(WORSE + "\n" + new_dose.replace("figure_over_dose", "figure_over_dose_x"), TRIAL))
    engine._client = model
    state2 = {**state, "implement_outline": outline, "exec_result": {"failed_body": "figure_over_dose"}}
    kept = engine._bodies_kept(state2, engine._analysis_bodies(state2), outline)
    assert kept is not None and kept.again == {"figure_over_dose": "the reason for this pass names it"}
    assert sorted(kept.names) == ["derive_peak", "figure_over_rate"] and kept.trial
    reply = _reply("def _scale(x):\n    return 5 * x\n\n\ndef figure_over_dose(fig, table):\n    fig.axes[0].plot([3], [_scale(3)])\n\n\n"
                   "def figure_over_rate(fig, table):\n    raise RuntimeError('rewritten')\n")
    model.code = reply
    await engine._node_implement(dict(state2))
    after = _bodies(engine)
    # a reply that gives a helper the file already has does not replace it: what a kept function calls is as it ran
    assert "def _scale(x):\n    return x\n" in (engine.quest_root / "code" / "derive.py").read_text(encoding="utf-8")
    assert after["figure_over_rate"] == before["figure_over_rate"] and after["derive_peak"] == before["derive_peak"]
    assert after["figure_over_dose"] != before["figure_over_dose"] and "plot([3], [_scale(3)])" in after["figure_over_dose"]
    assert len(model.prompts) == 1
    ask = model.prompts[0]
    assert ask.startswith("FI keeps") and "figure_over_rate" in ask.split("\n\n")[0] and "figure_over_dose" not in ask.split("\n\n")[0]


@pytest.mark.asyncio
async def test_known_positive_a_changed_axis_asks_again_for_the_figure_that_reads_it(tmp_path: Path) -> None:
    engine, state = await _first(tmp_path)
    engine._scripts_built_from = lambda s, o: {"p": "1"}  # type: ignore[method-assign]
    outline = {"scaffold": "x", "functions": []}
    engine._bodies_kept(state, engine._analysis_bodies(state), outline)
    _ran(engine, {"derive_peak": "ok", "figure_over_rate": "ok", "figure_over_dose": "ok"})
    changed = copy.deepcopy(DESIGN)
    changed["protocol"]["grid"]["dose"] = [1, 2, 3]
    state2 = {**state, "design": changed, "implement_outline": outline}
    kept = engine._bodies_kept(state2, engine._analysis_bodies(state2), outline)
    assert kept is not None and kept.names == ["figure_over_rate"]
    assert set(kept.again) == {"derive_peak", "figure_over_dose"}


@pytest.mark.asyncio
async def test_helpers_changed_after_a_function_ran_mean_it_is_not_kept(tmp_path: Path) -> None:
    engine, state = await _first(tmp_path)
    outline = {"scaffold": "x", "functions": []}
    engine._scripts_built_from = lambda s, o: {"p": "1"}  # type: ignore[method-assign]
    engine._bodies_kept(state, engine._analysis_bodies(state), outline)
    _ran(engine, {"derive_peak": "ok", "figure_over_rate": "ok", "figure_over_dose": "ok"})
    kb.snapshot(engine.quest_root, DESIGN)
    derive = engine.quest_root / "code" / "derive.py"
    derive.write_text(derive.read_text(encoding="utf-8").replace("HELPER = 1", "HELPER = 2"), encoding="utf-8")
    kept = kb.decide(engine.quest_root, DESIGN, ["derive_peak", "figure_over_rate", "figure_over_dose"], "", set())
    assert kept.names == [] and set(kept.again) == {"trial", "derive_peak", "figure_over_rate", "figure_over_dose"}


@pytest.mark.asyncio
async def test_a_quest_with_no_record_is_written_as_before(tmp_path: Path) -> None:
    engine, state = await _first(tmp_path)
    model = _Model(_reply(GOOD))
    engine._client = model
    await engine._node_implement(dict(state))
    assert len(model.prompts) == 1 and not model.prompts[0].startswith("FI keeps")


def test_a_reply_that_gives_a_kept_function_again_does_not_replace_it() -> None:
    frame = af.derive_source(DESIGN, "")
    text, _note = af.compose(DESIGN, WORSE, frame, keep=("figure_over_rate",))
    funcs = kb.sources(text)[0]
    assert "rewritten over_dose" in funcs["figure_over_dose"] and "NotYet" in funcs["figure_over_rate"]
    assert af.compose(DESIGN, "", frame, keep=tuple(af.body_names(af.plan_of(DESIGN))))[1] == ""


def test_the_simulation_kept_is_the_simulation_of_the_result() -> None:
    import re

    fence = re.compile(r"```(?:python)?\n(.*?)```", re.S)
    got = split_run.parse_split_response("```python\n# file: derive.py\ndef figure_a(fig, table):\n    pass\n```", fence,
                                         bodies=True, simulate_kept="KEPT")
    assert got is not None and got["simulate"] == "KEPT" and "figure_a" in got["derive"]
    assert split_run.parse_split_response("", fence, bodies=True) is None
    assert skeleton.RERUN_NOTE_ANALYSIS  # the sentence to the model stays for the functions that are written again


@pytest.mark.asyncio
async def test_a_change_in_the_raw_folder_beside_the_summary_or_a_write_under_code_gives_no_receipt(tmp_path: Path) -> None:
    engine, state = await _first(tmp_path)
    root = engine.quest_root
    (root / "raw").mkdir(exist_ok=True)
    (root / "raw" / "trials.json").write_text("{}", encoding="utf-8")
    (root / "raw" / "ledger.jsonl").write_text("1\n", encoding="utf-8")
    _ran(engine, {"derive_peak": "ok", "figure_over_rate": "ok", "figure_over_dose": "ok"})
    before = kb.receipt_inputs(root, "raw/trials.json")
    assert kb.write_receipt(root, before) and kb._receipt(root) == (True, True)
    (root / "raw" / "ledger.jsonl").write_text("2\n", encoding="utf-8")  # the analysis reads the ledger too
    assert kb._receipt(root) == (False, True)
    (root / "raw" / "ledger.jsonl").write_text("1\n", encoding="utf-8")
    assert not kb.write_receipt(root, before, ["code/helper.py"]), "the run wrote under code/ in its copy"
    assert not (root / kb.RECEIPT).exists()
