"""A person's refine that asks for a number the study lacks extends the existing script(s); one that asks for the figures to
be arranged again only redraws them from the saved data. Neither rewrites the experiment from the design (the quest
feedback of 2026-09-29: "add the missing number" rewrote the scripts and redid the experiment)."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig
from core.engine import Engine, _format_review_for_writer, _take_refine_points

NOTE = "I do not see the runtime for n=64; add it. The figures are fine, but put Figure 2 above Figure 1."
STATE: dict[str, Any] = {
    "topic": "t", "title": "t", "iteration": 1,
    "human_feedback": {"action": "refine", "feedback": NOTE},
    "feedback_history": [{"iteration": 1, "text": NOTE}],
}


def _engine(tmp_path: Path) -> Engine:
    return Engine(Config(
        topic="t", title="t", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(clarify_mode="off", review_loop=True, max_iterations=2),  # type: ignore[arg-type]
        execution=ExecutionConfig(sandbox="venv", timeout_s=60), knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "outputs"),
    ))


def test_the_writer_can_name_a_missing_number_and_a_layout_point() -> None:
    reply = ("# P\n\nBody.\n\nNEEDS_DATA: the runtime for n=64\n2. NEEDS_LAYOUT: Figure 2 above Figure 1\n"
             "NEEDS_EXPERIMENT: a different solver family\n")
    paper, points = _take_refine_points(reply)
    assert points == {"data": ["the runtime for n=64"], "layout": ["Figure 2 above Figure 1"],
                      "experiment": ["a different solver family"]}
    assert "NEEDS_" not in paper and "Body." in paper


def test_the_writer_is_told_about_all_three_kinds() -> None:
    text = _format_review_for_writer(STATE, refine_round=True)  # type: ignore[arg-type]
    assert "NEEDS_DATA:" in text and "NEEDS_LAYOUT:" in text and "NEEDS_EXPERIMENT:" in text


def test_routes(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    route = eng._route_after_write
    assert route({}) == "check"  # type: ignore[arg-type]
    assert route({"refine_layout": ["x"]}) == "replot"  # type: ignore[arg-type]
    assert route({"refine_extend": ["x"]}) == "extend"  # type: ignore[arg-type]
    assert route({"refine_extend": ["x"], "refine_layout": ["y"]}) == "extend"  # type: ignore[arg-type]
    assert route({"refine_needs_experiment": ["z"], "refine_extend": ["x"]}) == "redesign"  # type: ignore[arg-type]
    # A quest with no experiment has no script to extend: it collects the missing data instead.
    assert route({"refine_extend": ["x"], "no_simulation_resolved": True}) == "collect"  # type: ignore[arg-type]


def test_write_keeps_the_points_apart(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    eng.quest_root.mkdir(parents=True, exist_ok=True)
    (eng.quest_root / "paper").mkdir(exist_ok=True)

    async def chat(prompt: str, **kw: Any) -> str:
        return "# P\n\nBody.\n\nNEEDS_DATA: the runtime for n=64\nNEEDS_LAYOUT: Figure 2 above Figure 1\n"

    eng._chat = chat  # type: ignore[method-assign]
    out = asyncio.run(eng._node_write(dict(STATE)))  # type: ignore[arg-type]
    assert out["refine_extend"] == ["the runtime for n=64"] and out["refine_layout"] == ["Figure 2 above Figure 1"]
    assert out["refine_needs_experiment"] == [] and out["refine_scope"] == "data"
    assert "NEEDS_" not in Path(out["paper_md"]).read_text(encoding="utf-8")
    # A survey has nothing to extend: the missing number is looked for by going back to the design.
    survey = asyncio.run(eng._node_write({**STATE, "survey_mode_resolved": True}))  # type: ignore[arg-type]
    assert survey["refine_extend"] == [] and survey["refine_needs_experiment"] == ["the runtime for n=64"]
    # The write that comes after the extended run answers no refine, so it clears the points and the graph moves on.
    again = asyncio.run(eng._node_write({**STATE, **out}))  # type: ignore[arg-type]
    assert again["refine_extend"] == [] and again["refine_layout"] == ["Figure 2 above Figure 1"]
    assert eng._route_after_write({**STATE, **out, **again}) == "replot"  # type: ignore[arg-type]
    assert asyncio.run(eng._node_write({**STATE, **out, **again, "refine_layout": []}))["refine_layout"] == []  # type: ignore[arg-type]


def test_implement_extends_the_existing_script(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    code_dir = eng.quest_root / "code"
    code_dir.mkdir(parents=True)
    old = "import json\nfor n in (8, 16, 32):\n    pass\nprint('RESULT_JSON: {}')\n"
    (code_dir / "experiment.py").write_text(old, encoding="utf-8")
    seen: list[str] = []

    async def chat(prompt: str, *, node: str = "") -> str:
        seen.append(prompt)
        return "```python\nimport json\nfor n in (8, 16, 32, 64):\n    pass\nprint('RESULT_JSON: {}')\n```\nDEPS: numpy"

    eng._chat = chat  # type: ignore[method-assign]
    out = asyncio.run(eng._node_implement(  # type: ignore[arg-type]
        {**STATE, "design": {}, "refine_extend": ["the runtime for n=64"],
         "refine_scope": "data"}))
    prompt = seen[0]
    assert old in prompt, "the model is given the script that exists"
    assert "the runtime for n=64" in prompt and "change as little as you can" in prompt and "Keep the protocol" in prompt
    assert "(8, 16, 32, 64)" in (code_dir / "experiment.py").read_text(encoding="utf-8")
    assert "(8, 16, 32, 64)" in out["code"]


def _replot_quest(tmp_path: Path) -> Engine:
    eng = _engine(tmp_path)
    root = eng.quest_root
    (root / "code").mkdir(parents=True)
    (root / "data" / "results").mkdir(parents=True)
    (root / "figures").mkdir()
    (root / "code" / "experiment.py").write_text("print('the experiment')\n", encoding="utf-8")
    (root / "data" / "results" / "times.csv").write_text("n,t\n8,1\n", encoding="utf-8")
    (root / "figures" / "fig1.png").write_bytes(b"old-1")
    (root / "figures" / "fig2.png").write_bytes(b"old-2")
    return eng


def _run_replot(eng: Engine, script: str, *, timed_out: bool = False) -> tuple[Any, list[list[str]]]:
    """The layout step with a model that returns ``script``, run for real by this Python, so what the script writes is
    on disk and not just claimed by a fake result."""
    ran: list[list[str]] = []

    async def chat(prompt: str, *, node: str = "") -> str:
        return f"```python\n{script}\n```"

    async def execute(cmd: list[str], **kw: Any) -> Any:
        ran.append(cmd)
        proc = subprocess.run([sys.executable, cmd[-1]], cwd=str(kw["cwd"]), capture_output=True, text=True)
        return SimpleNamespace(returncode=0 if timed_out else proc.returncode, stdout=proc.stdout,
                               stderr=proc.stderr, duration_s=0.1, timed_out=timed_out)

    eng._chat = chat  # type: ignore[method-assign]
    eng.executor = SimpleNamespace(python_path=lambda q: Path(sys.executable), execute=execute)  # type: ignore[assignment]
    out = asyncio.run(eng._node_replot_layout({**STATE, "refine_layout": ["Figure 2 above Figure 1"]}))  # type: ignore[arg-type]
    return out, ran


def test_a_layout_refine_redraws_only_the_named_figure_from_saved_data(tmp_path: Path) -> None:
    eng = _replot_quest(tmp_path)
    root = eng.quest_root
    script = ("import csv\nrows = list(csv.DictReader(open('data/results/times.csv')))\n"
              "open('figures/fig2.png', 'wb').write(('new-' + rows[0]['t']).encode())\nprint('REPLOTTED: fig2.png')")
    out, ran = _run_replot(eng, script)
    assert len(ran) == 1 and ran[0][-1].endswith("replot_layout.py")
    assert (root / "figures" / "fig2.png").read_bytes() == b"new-1", "the figure was drawn from the saved numbers"
    assert (root / "figures" / "fig1.png").read_bytes() == b"old-1"
    assert (root / "code" / "experiment.py").read_text(encoding="utf-8") == "print('the experiment')\n"
    assert out["refine_layout"] == []


def test_a_failed_redraw_puts_the_figures_back(tmp_path: Path) -> None:
    eng = _replot_quest(tmp_path)
    _run_replot(eng, "open('figures/fig2.png', 'wb').write(b'half')\nraise SystemExit(3)")
    assert (eng.quest_root / "figures" / "fig2.png").read_bytes() == b"old-2"


def test_a_timed_out_redraw_puts_the_figures_back(tmp_path: Path) -> None:
    eng = _replot_quest(tmp_path)
    _run_replot(eng, "open('figures/fig2.png', 'wb').write(b'half')\nprint('REPLOTTED: fig2.png')", timed_out=True)
    assert (eng.quest_root / "figures" / "fig2.png").read_bytes() == b"old-2"


def test_a_redraw_that_changes_the_saved_numbers_or_an_unnamed_figure_is_undone(tmp_path: Path) -> None:
    eng = _replot_quest(tmp_path)
    root = eng.quest_root
    script = ("open('data/results/times.csv', 'w').write('n,t\\n8,999\\n')\n"
              "open('figures/fig1.png', 'wb').write(b'sneaky')\n"
              "open('figures/fig2.png', 'wb').write(b'new-2')\nprint('REPLOTTED: fig2.png')")
    _run_replot(eng, script)
    assert (root / "data" / "results" / "times.csv").read_text(encoding="utf-8") == "n,t\n8,1\n"
    assert (root / "figures" / "fig1.png").read_bytes() == b"old-1"
    assert (root / "figures" / "fig2.png").read_bytes() == b"new-2"


def test_an_extension_that_did_not_come_back_keeps_the_scripts_that_exist(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    code_dir = eng.quest_root / "code"
    code_dir.mkdir(parents=True)
    old = "for n in (8, 16, 32):\n    pass\nprint('RESULT_JSON: {}')\n"
    (code_dir / "experiment.py").write_text(old, encoding="utf-8")

    async def chat(prompt: str, *, node: str = "") -> str:
        return "I would add n=64 to the loop."  # no script in the reply

    eng._chat = chat  # type: ignore[method-assign]
    out = asyncio.run(eng._node_implement({**STATE, "design": {}, "refine_extend": ["the runtime for n=64"]}))  # type: ignore[arg-type]
    assert (code_dir / "experiment.py").read_text(encoding="utf-8") == old
    assert out["code"] == old and out["refine_extend"] == []


def test_a_one_script_reply_does_not_delete_the_simulation_of_a_two_script_quest(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    eng.config.execution.split_analysis = True  # type: ignore[assignment]
    eng.config.execution.split_failure = "warn"  # type: ignore[assignment]
    code_dir = eng.quest_root / "code"
    code_dir.mkdir(parents=True)
    (code_dir / "simulate.py").write_text("def run_trial(seed):\n    return {}\n", encoding="utf-8")
    (code_dir / "experiment.py").write_text("print('RESULT_JSON: {}')\n", encoding="utf-8")

    async def chat(prompt: str, *, node: str = "") -> str:
        return "```python\nprint('one script')\n```"

    eng._chat = chat  # type: ignore[method-assign]
    asyncio.run(eng._node_implement({**STATE, "design": {}, "refine_extend": ["n=64"]}))  # type: ignore[arg-type]
    assert (code_dir / "simulate.py").is_file()
    assert "one script" not in (code_dir / "experiment.py").read_text(encoding="utf-8")


def test_an_extension_beyond_the_frozen_protocol_is_recorded_not_repaired_or_paused(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    code_dir = eng.quest_root / "code"
    code_dir.mkdir(parents=True)
    (code_dir / "experiment.py").write_text("R0_LIST = [0.9, 1.5, 3.0]\nprint('RESULT_JSON: {}')\n", encoding="utf-8")
    protocol = {"grid": {"R0": [0.9, 1.5, 3.0]}}
    state = {**STATE, "design": {"hypothesis": "h", "protocol": protocol}, "refine_extend": ["R0 = 6.0"]}
    calls: list[str] = []

    async def chat(prompt: str, *, node: str = "") -> str:
        calls.append(node)
        return "```python\nR0_LIST = [0.9, 1.5, 3.0, 6.0]\nprint('RESULT_JSON: {}')\n```"

    eng._chat = chat  # type: ignore[method-assign]
    from core import protocol_check

    assert protocol_check.check(protocol, {"experiment.py": "R0_LIST = [0.9, 1.5, 3.0, 6.0]\n"}), \
        "the extended grid does differ from the protocol, so this test would otherwise prove nothing"
    out = asyncio.run(eng._node_implement(state))  # type: ignore[arg-type]
    assert "implement_protocol" not in calls, "the protocol was not used to repair the extension"
    assert "6.0" in (code_dir / "experiment.py").read_text(encoding="utf-8") and "6.0" in out["code"]
    assert "extended_by_person" in (eng.quest_root / "needs" / "PROTOCOL_CHECK.json").read_text(encoding="utf-8")


def test_the_graph_has_both_routes(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    graph = eng._build_graph().compile().get_graph()
    edges = {(e.source, e.target) for e in graph.edges}
    assert ("write", "implement") in edges and ("write", "replot_layout") in edges and ("write", "auto_collect_data") in edges
    assert ("replot_layout", "claim_check") in edges
