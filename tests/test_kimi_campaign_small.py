"""Four faults a real-model campaign (Kimi K3) found that the fake-model tests never met.

- The papers pause offered "go on without them", but resuming asked for the same papers again, forever.
- A ``node_http_timeout_s`` (or ``node_cli_timeout_s``) map with one key replaced the whole default table.
- Under ``rigor_profile: research`` a pause suggested a setting the profile refuses (``run_manifest_check: warn``).
- A failed model call and its retry were not in the quest's run.log.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import tenacity

from core import todo
from core.config import Config, ProviderConfig
from core.engine import _papers_to_ask
from core.knowledge import RetrievedDoc
from core.provider import LLMClient, ResolvedEndpoint

# --- the papers pause ------------------------------------------------------------------------------------------------


def _doc(doi: str) -> RetrievedDoc:
    return RetrievedDoc(content="abstract", metadata={"doi": doi, "title": doi})


def test_going_on_without_the_papers_is_not_asked_again(tmp_path: Path) -> None:
    log = logging.getLogger("test.papers")
    first = [_doc("10.1/a"), _doc("10.1/b")]
    assert _papers_to_ask(first, tmp_path, log) == first, "asked the first time"
    assert _papers_to_ask(list(first), tmp_path, log) == [], "resumed without adding files: going on without them"
    assert _papers_to_ask([_doc("10.1/b")], tmp_path, log) == [], "a subset asked before is not asked again"
    new = [_doc("10.1/b"), _doc("10.1/c")]
    assert _papers_to_ask(new, tmp_path, log) == new, "a paper not asked for before is asked for"
    assert _papers_to_ask([], tmp_path, log) == []


def test_the_pause_says_how_to_go_on_without_them() -> None:
    import inspect

    from core import engine

    assert "Or go on without them" in inspect.getsource(engine.Engine)


# --- timeouts over the defaults ---------------------------------------------------------------------------------------


def test_one_http_timeout_keeps_the_other_steps_budgets() -> None:
    default = ProviderConfig().node_http_timeout_s
    p = ProviderConfig.model_validate({"name": "openai", "node_http_timeout_s": {"implement": 1800}})
    assert p.node_http_timeout_s["implement"] == 1800
    assert {k: v for k, v in p.node_http_timeout_s.items() if k != "implement"} == \
        {k: v for k, v in default.items() if k != "implement"}
    assert ProviderConfig.model_validate({"node_http_timeout_s": None}).node_http_timeout_s == default


def test_one_cli_timeout_keeps_the_others_and_the_floor_still_raises_them() -> None:
    p = ProviderConfig.model_validate({"name": "claude_cli", "node_cli_timeout_s": {"write": 50}})
    assert p.node_cli_timeout_s["write"] == 50, "a tighter budget written on purpose is kept"
    assert p.node_cli_timeout_s["implement"] == ProviderConfig().node_cli_timeout_s["implement"]
    floored = ProviderConfig.model_validate(
        {"name": "claude_cli", "cli_timeout_s": 900, "node_cli_timeout_s": {"write": 50}})
    assert floored.node_cli_timeout_s["write"] == 50
    assert floored.node_cli_timeout_s["implement_outline"] == 900, "a built-in budget below the floor is raised"


def test_the_whole_config_merges_too() -> None:
    cfg = Config.model_validate({"topic": "t", "provider": {"name": "openai", "node_http_timeout_s": {"implement": 1800}}})
    assert cfg.provider.node_http_timeout_s["write"] == ProviderConfig().node_http_timeout_s["write"]


# --- research pauses never suggest a refused setting ------------------------------------------------------------------


@pytest.mark.parametrize("kind", sorted(todo._ADVICE))
def test_no_research_pause_card_suggests_a_refused_setting(kind: str) -> None:
    steps = ["The run differs from the frozen protocol. Set `engine.run_manifest_check: warn` to go on with the "
             "difference recorded.", "Set `engine.oracle_check: warn` and go on."]
    item = todo.pause_item(kind, "stopped", steps, profile="research")
    for line in [item.recommended, *item.alternatives, *item.steps]:
        assert todo.refused_settings(line, "research") == [], (kind, line)
    assert todo.RESEARCH_INSTEAD in [item.recommended, *item.alternatives]
    assert item.steps == ["The run differs from the frozen protocol."]


def test_outside_research_the_suggestions_stay() -> None:
    item = todo.pause_item("manifest", "stopped", ["Set `engine.run_manifest_check: warn` to go on."])
    assert "`engine.run_manifest_check: warn`" in " ".join([*item.steps, *item.alternatives])


def test_a_setting_the_profile_keeps_is_not_taken_for_a_refused_one() -> None:
    assert todo.refused_settings("keep `engine.run_manifest_check: block`", "research") == []
    assert todo.refused_settings("`execution.split_failure: warn`", "research") == ["`execution.split_failure: warn`"]
    assert todo.refused_settings("`knowledge.read_figures: false`", "research") == []


# --- a failed call and its retry are in run.log -----------------------------------------------------------------------


class _Collect(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(record.getMessage())


async def test_a_failed_http_call_and_its_retry_are_in_the_quest_log(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("core.provider.wait_random_exponential", lambda **kw: tenacity.wait_none())
    ok = MagicMock()
    ok.status_code = 200
    ok.json = MagicMock(return_value={"choices": [{"message": {"content": "hello"}}]})
    ok.raise_for_status = MagicMock()
    fake_http = MagicMock()
    fake_http.post = AsyncMock(side_effect=[httpx.ReadTimeout("read timed out, Bearer sk-abcdefghijkl"), ok])
    quest_log = logging.getLogger("frontier_insight.test-retry-quest")
    handler = _Collect()
    quest_log.addHandler(handler)
    try:
        client = LLMClient(ResolvedEndpoint(base_url="https://api.example.com/v1", model="m1", api_key="k",
                                            provider_name="kimi"), http=fake_http, run_log=quest_log)
        assert await client.chat([{"role": "user", "content": "hi"}], node="implement") == "hello"
    finally:
        quest_log.removeHandler(handler)
    (line,) = handler.lines
    assert line.startswith("[implement] the model call failed (kimi over HTTP, model m1), attempt 1 of 4: ReadTimeout")
    assert "trying again" in line and "sk-abcdefghijkl" not in line and "[redacted]" in line
