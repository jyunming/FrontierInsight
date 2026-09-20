"""Two prompts that carried text their stage did not need.

``implement_outline`` fixes a script's structure and received every selected skill's full text (about 17,000
tokens); the body stage that writes the calls gets that text again. ``analyze`` interprets an experiment's results
and received the whole reviewed-literature block (8,000 to 13,000 tokens) that the write step and the cross-check
read on their own. Replayed from three stored quests with the real model, outlines written from the summary named
the same functions and result template as ones written from the full text, and analyses written with only the
titles of the sources matched the ones written with the whole block.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig,
)
from core.engine import Engine

FULL = "FULL SKILL TEXT: the whole API, examples and caveats of every selected skill."
SUMMARY = "SKILL SUMMARY: what each is for, where it does not apply."


def _engine(tmp_path: Path, **execution: Any) -> Engine:
    eng = Engine(Config(
        topic="t", title="t", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60, **execution),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "out"),
    ))
    eng.quest_root = tmp_path  # type: ignore[attr-defined]
    eng.fi_dir = tmp_path / ".fi"  # type: ignore[attr-defined]
    return eng


# --- implement_outline --------------------------------------------------------------------------

def _outline_prompt(eng: Engine, monkeypatch, *, full: str = FULL, summary: str = SUMMARY) -> str:  # noqa: ANN001
    monkeypatch.setattr(eng, "_skills_block", lambda state=None: full)
    monkeypatch.setattr(eng, "_skills_summary_block", lambda state=None: summary)
    seen: list[str] = []

    async def chat(prompt, *, node=None):  # noqa: ANN001
        seen.append(prompt)
        return json.dumps({"scaffold": "def f():\n    pass\n", "functions": [], "constants": []})

    eng._chat = chat  # type: ignore[method-assign]
    asyncio.run(eng._node_implement_outline({"topic": "t", "design": {"method": "m"}}))  # type: ignore[arg-type]
    assert len(seen) == 1
    return seen[0]


def test_the_outline_reads_the_skill_summary_not_the_full_text(tmp_path: Path, monkeypatch) -> None:
    prompt = _outline_prompt(_engine(tmp_path), monkeypatch)
    assert SUMMARY in prompt and FULL not in prompt


def test_the_outline_keeps_the_full_text_for_a_background_job(tmp_path: Path, monkeypatch) -> None:
    """The skill says how the job is submitted and read back, which is structure."""
    prompt = _outline_prompt(_engine(tmp_path, background_jobs=True), monkeypatch)
    assert FULL in prompt and SUMMARY not in prompt


def test_the_outline_keeps_the_full_text_when_the_user_supplied_example_files(tmp_path: Path, monkeypatch) -> None:
    """The examples are combined with the skills into a new simulation."""
    monkeypatch.setattr("core.example_inputs.render_block", lambda root: "a.cfg: a simulation setup")
    prompt = _outline_prompt(_engine(tmp_path), monkeypatch)
    assert FULL in prompt and SUMMARY not in prompt


def test_the_outline_says_so_when_no_skill_is_selected(tmp_path: Path, monkeypatch) -> None:
    prompt = _outline_prompt(_engine(tmp_path), monkeypatch, full="", summary="")
    assert "(no skills selected for this quest)" in prompt


def test_the_body_stage_still_receives_the_full_text(tmp_path: Path, monkeypatch) -> None:
    eng = _engine(tmp_path)
    monkeypatch.setattr(eng, "_skills_block", lambda state=None: FULL)
    monkeypatch.setattr(eng, "_skills_summary_block", lambda state=None: SUMMARY)
    seen: list[str] = []

    async def chat(prompt, *, node=None):  # noqa: ANN001
        seen.append(prompt)
        return "```python\nprint('RESULT_JSON: {}')\n```\n```requirements\n```"

    eng._chat = chat  # type: ignore[method-assign]
    state = {"topic": "t", "design": {"method": "m"}, "implement_outline": {"scaffold": "def f():\n    pass\n"}}
    try:
        asyncio.run(eng._node_implement(state))  # type: ignore[arg-type]
    except Exception:  # noqa: BLE001 — the reply's shape is not what is under test
        pass
    assert seen and FULL in seen[0] and SUMMARY not in seen[0]


# --- analyze --------------------------------------------------------------------------------------

def _lit(title: str, year: int, text: str) -> dict[str, Any]:
    return {"content": f"{title}. {text}", "metadata": {
        "title": title, "year": year, "authors": ["A. Author"], "doi": f"10.1/{year}", "source": "openalex",
        "venue": "V", "work_type": "article",
    }}


LITERATURE = [
    _lit("A contribution to the mathematical theory of epidemics", 1927, "SENTINEL-ABSTRACT-ONE " * 12),
    _lit("The outcome of a stochastic epidemic", 1955, "SENTINEL-ABSTRACT-TWO " * 12),
]
RESULTS = {"outbreak_probability": 0.42, "final_size": 0.58}


def _analyze_state(**extra: Any) -> dict[str, Any]:
    return {"topic": "t", "design": {}, "figures": [], "exec_result": {"returncode": 0},
            "result_json": RESULTS, "literature": LITERATURE, **extra}


def test_analysis_of_an_experiments_results_gets_the_titles_of_the_sources_only(tmp_path: Path) -> None:
    block = _engine(tmp_path)._analyze_literature_block(_analyze_state())  # type: ignore[arg-type]
    assert "Titles of the sources retrieved for this study" in block
    assert "- [1] A contribution to the mathematical theory of epidemics (1927)" in block
    assert "- [2] The outcome of a stochastic epidemic (1955)" in block
    assert "SENTINEL" not in block  # none of the sources' text


@pytest.mark.parametrize("extra", [
    {"no_simulation_resolved": True},   # a study with no experiment
    {"survey_mode_resolved": True},     # a survey: the literature is what is interpreted
    {"result_json": {}},                # nothing to interpret but the literature
])
def test_the_whole_literature_stays_where_it_is_what_the_analysis_interprets(tmp_path: Path, extra: dict) -> None:
    block = _engine(tmp_path)._analyze_literature_block(_analyze_state(**extra))  # type: ignore[arg-type]
    assert "SENTINEL-ABSTRACT-ONE" in block and "SENTINEL-ABSTRACT-TWO" in block
    assert "Titles of the sources retrieved" not in block


def test_the_whole_literature_stays_for_a_run_on_the_users_own_data_and_a_degenerate_result(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    eng.config.engine.analyze_local_first = True
    assert "SENTINEL-ABSTRACT-ONE" in eng._analyze_literature_block(_analyze_state())  # type: ignore[arg-type]
    eng.config.engine.analyze_local_first = False
    assert "SENTINEL-ABSTRACT-ONE" in eng._analyze_literature_block(_analyze_state(), degenerate=True)  # type: ignore[arg-type]


def test_an_analysis_with_no_sources_gets_what_it_always_got(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    block = eng._analyze_literature_block(_analyze_state(literature=[]))  # type: ignore[arg-type]
    assert "Titles of the sources retrieved" not in block


def test_the_analyze_prompt_carries_the_titles_and_not_the_text(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    seen: list[str] = []

    async def chat(prompt, *, node=None):  # noqa: ANN001
        seen.append(prompt)
        return json.dumps({"summary": "s", "key_findings": ["f"], "next_step": "publish"})

    eng._chat = chat  # type: ignore[method-assign]
    patch = asyncio.run(eng._node_analyze(_analyze_state()))  # type: ignore[arg-type]
    assert patch["analysis"]["next_step"] == "publish"
    prompt = seen[0]
    assert "### Reviewed literature (published sources)\nTitles of the sources retrieved" in prompt
    assert "The outcome of a stochastic epidemic (1955)" in prompt and "SENTINEL" not in prompt
    # A survey study's analyze prompt is as it was.
    seen.clear()
    asyncio.run(eng._node_analyze(_analyze_state(no_simulation_resolved=True, result_json={})))  # type: ignore[arg-type]
    assert "SENTINEL-ABSTRACT-ONE" in seen[0]
