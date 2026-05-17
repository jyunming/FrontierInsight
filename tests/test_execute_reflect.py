"""Execute-repair loop tests.

Validates the `_node_execute_reflect` node's three branches:
  - script succeeded → pass-through, no LLM call
  - script failed + budget left → patch code, route back to execute
  - script failed + LLM gives up → record reason, route to analyze
  - script failed + budget exhausted → route to analyze with broken state

Plus the routing function `_route_after_execute_reflect`.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig,
    OutputConfig, ProviderConfig,
)
from core.engine import Engine, _format_reflect_history


def _mk_cfg(tmp_path: Path, *, max_iter: int = 3) -> Config:
    return Config(
        topic="execute-reflect tests",
        title="exec-reflect",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(
            max_iterations=1, review_loop=False,
            exec_reflect_max_iterations=max_iter,
            ideate_reflect=False,  # keep tests focused
        ),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "out"),
    )


# --- _format_reflect_history helper ----------------------------------------


def test_format_reflect_history_empty() -> None:
    assert "no prior repair" in _format_reflect_history([])


def test_format_reflect_history_renders_each_iter() -> None:
    out = _format_reflect_history([
        {"iter": 1, "returncode": 2, "stderr_tail": "ImportError: matplotlib",
         "patch_summary": "added matplotlib to deps"},
        {"iter": 2, "returncode": 0, "stderr_tail": "",
         "patch_summary": "(success)"},
    ])
    assert "iter 1" in out and "ImportError" in out
    assert "iter 2" in out
    assert "added matplotlib" in out


# --- _node_execute_reflect ---------------------------------------------------


@pytest.mark.asyncio
async def test_reflect_passes_through_on_success(tmp_path: Path) -> None:
    """rc=0 AND RESULT_JSON parsed → reflect is a no-op. No LLM call."""
    eng = Engine(_mk_cfg(tmp_path))
    chat_mock = AsyncMock(return_value='{}')
    eng._client = type("Stub", (), {"chat": chat_mock})()

    state = {
        "exec_result": {"returncode": 0, "stdout_tail": "ok", "stderr_tail": ""},
        "result_json": {"score": 1},
        "code": "print('ok')",
        "design": {"hypothesis": "h"},
    }
    patch = await eng._node_execute_reflect(state)

    assert patch == {}
    chat_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_reflect_patches_code_and_advances_iter(tmp_path: Path) -> None:
    """Script failed → LLM returns a patched `code` field → write it to
    disk and bump `exec_reflect_iter`. Routing function (tested
    separately) will then send the graph back to execute."""
    eng = Engine(_mk_cfg(tmp_path))
    eng.quest_root.mkdir(parents=True, exist_ok=True)
    (eng.quest_root / "code").mkdir(parents=True, exist_ok=True)

    patched_code = "import matplotlib\nprint('RESULT_JSON: {}')\n"
    chat_mock = AsyncMock(return_value=json.dumps({
        "code": patched_code,
        "deps": ["matplotlib"],
        "patch_summary": "added matplotlib import + RESULT_JSON line",
        "give_up_reason": "",
    }))
    eng._client = type("Stub", (), {"chat": chat_mock})()

    state = {
        "exec_result": {
            "returncode": 1,
            "stdout_tail": "",
            "stderr_tail": "NameError: name 'matplotlib' is not defined",
        },
        "result_json": None,
        "code": "matplotlib.figure()",
        "design": {"hypothesis": "h"},
        "deps": ["numpy"],
    }
    patch = await eng._node_execute_reflect(state)

    assert "code" in patch
    assert patch["exec_reflect_iter"] == 1
    assert patch["exec_reflect_history"][0]["patch_summary"].startswith("added matplotlib")
    # deps merged, no duplicate
    assert patch["deps"] == ["numpy", "matplotlib"]
    # Code was written to disk so the next `execute` picks it up.
    on_disk = (eng.quest_root / "code" / "experiment.py").read_text(encoding="utf-8")
    assert on_disk == patched_code


@pytest.mark.asyncio
async def test_reflect_records_give_up_reason(tmp_path: Path) -> None:
    """LLM emits `give_up_reason` → record it; the routing function
    will then send the graph to analyze with the broken state."""
    eng = Engine(_mk_cfg(tmp_path))
    eng.quest_root.mkdir(parents=True, exist_ok=True)
    chat_mock = AsyncMock(return_value=json.dumps({
        "code": "",
        "deps": [],
        "patch_summary": "",
        "give_up_reason": "design requires a GPU; cannot run on CPU venv",
    }))
    eng._client = type("Stub", (), {"chat": chat_mock})()

    state = {
        "exec_result": {"returncode": 1, "stdout_tail": "", "stderr_tail": "CUDA missing"},
        "result_json": None,
        "code": "import torch\ntorch.cuda.is_available()",
        "design": {"hypothesis": "h"},
    }
    patch = await eng._node_execute_reflect(state)

    assert patch["exec_give_up_reason"].startswith("design requires a GPU")
    assert patch["exec_reflect_iter"] == 1
    assert "code" not in patch  # no patched code when giving up


@pytest.mark.asyncio
async def test_reflect_stops_when_iterations_exhausted(tmp_path: Path) -> None:
    """When the budget is already at the cap, the node passes through
    without calling the LLM — bounded so we don't burn calls forever."""
    eng = Engine(_mk_cfg(tmp_path, max_iter=2))
    chat_mock = AsyncMock(return_value="{}")
    eng._client = type("Stub", (), {"chat": chat_mock})()

    state = {
        "exec_result": {"returncode": 1, "stdout_tail": "", "stderr_tail": "boom"},
        "result_json": None,
        "code": "boom",
        "design": {},
        "exec_reflect_iter": 2,  # already at max
    }
    patch = await eng._node_execute_reflect(state)

    assert patch == {}
    chat_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_reflect_handles_unparseable_llm_output(tmp_path: Path) -> None:
    """LLM returns garbage → we record a give-up reason and advance the
    iteration counter so we don't loop forever on a broken provider."""
    eng = Engine(_mk_cfg(tmp_path))
    eng.quest_root.mkdir(parents=True, exist_ok=True)
    chat_mock = AsyncMock(return_value="no json at all, sorry")
    eng._client = type("Stub", (), {"chat": chat_mock})()

    state = {
        "exec_result": {"returncode": 1, "stdout_tail": "", "stderr_tail": "boom"},
        "result_json": None,
        "code": "boom",
        "design": {},
    }
    patch = await eng._node_execute_reflect(state)

    assert "no patched code" in patch["exec_give_up_reason"]
    assert patch["exec_reflect_iter"] == 1


# --- routing function -------------------------------------------------------


def test_route_after_execute_reflect_proceeds_on_success(tmp_path: Path) -> None:
    eng = Engine(_mk_cfg(tmp_path))
    assert eng._route_after_execute_reflect({
        "exec_result": {"returncode": 0},
        "result_json": {"score": 1},
    }) == "proceed"


def test_route_after_execute_reflect_retries_on_failure_with_budget(tmp_path: Path) -> None:
    eng = Engine(_mk_cfg(tmp_path, max_iter=3))
    # rc!=0, iter 1 of 3, no give-up reason → retry.
    assert eng._route_after_execute_reflect({
        "exec_result": {"returncode": 1},
        "result_json": None,
        "exec_reflect_iter": 1,
    }) == "retry"


def test_route_after_execute_reflect_proceeds_on_give_up(tmp_path: Path) -> None:
    eng = Engine(_mk_cfg(tmp_path))
    assert eng._route_after_execute_reflect({
        "exec_result": {"returncode": 1},
        "result_json": None,
        "exec_give_up_reason": "unfixable",
        "exec_reflect_iter": 1,
    }) == "proceed"


def test_route_after_execute_reflect_proceeds_on_budget_exhaustion(tmp_path: Path) -> None:
    eng = Engine(_mk_cfg(tmp_path, max_iter=2))
    assert eng._route_after_execute_reflect({
        "exec_result": {"returncode": 1},
        "result_json": None,
        "exec_reflect_iter": 2,
    }) == "proceed"
