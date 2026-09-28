"""Four faults a real-model campaign (Kimi K3) found that the fake-model tests never met.

- The papers pause offered "go on without them", but resuming asked for the same papers again, forever.
- A ``node_http_timeout_s`` (or ``node_cli_timeout_s``) map with one key replaced the whole default table.
- Under ``rigor_profile: research`` a pause suggested a setting the profile refuses (``run_manifest_check: warn``).
- A failed model call and its retry were not in the quest's run.log.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import tenacity

from core import todo
from core.config import Config, ProviderConfig
from core.knowledge import RetrievedDoc
from core.provider import LLMClient, ResolvedEndpoint

# --- the papers pause ------------------------------------------------------------------------------------------------


def _doc(doi: str) -> RetrievedDoc:
    return RetrievedDoc(content="abstract", metadata={"doi": doi, "title": doi})


class _Paused(Exception):
    pass


def _paper_engine(tmp_path: Path, docs: list[RetrievedDoc]):
    from core.config import EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig
    from core.engine import Engine

    eng = Engine(Config(
        topic="papers pause", title="pp", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, ideate_reflect=False),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60), knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "out"), pauses={"papers": True},
    ))

    async def fake_search(query, **kw):  # noqa: ANN001
        return list(docs)

    eng.knowledge.asearch = fake_search  # type: ignore[method-assign]
    asked: list[dict] = []

    def fake_pause(**kw):  # noqa: ANN003 -- stands in for interrupt(): records the pause the way the real one does
        asked.append(kw)
        eng.fi_dir.mkdir(parents=True, exist_ok=True)
        (eng.fi_dir / "pause.json").write_text(json.dumps({"kind": kw["kind"]}), encoding="utf-8")
        raise _Paused

    eng._pause_for_human = fake_pause  # type: ignore[method-assign]
    return eng, asked


def _paywalled(doi: str) -> RetrievedDoc:
    return RetrievedDoc(content="an abstract", metadata={"doi": doi, "title": f"Paper {doi}", "source": "crossref",
                                                         "abstract_only": True})


async def test_the_literature_step_pauses_then_goes_on_without_the_papers(tmp_path: Path) -> None:
    from core import audit_log

    docs = [_paywalled("10.1/a"), _paywalled("10.1/b")]
    eng, asked = _paper_engine(tmp_path, docs)
    state = {"topic": "papers pause", "chosen_idea": {"title": "T"}}
    with pytest.raises(_Paused):
        await eng._node_literature(dict(state))
    assert len(asked) == 1 and asked[0]["kind"] == "papers"
    # Resumed without adding any file. run() clears the pause markers before any node runs, as a real resume does.
    eng._clear_stale_pause_markers()
    assert not (eng.fi_dir / "pause.json").exists()
    patch = await eng._node_literature(dict(state))
    assert patch.get("literature"), "the literature step finished"
    assert len(asked) == 1, "not asked again"
    events = [e for e in audit_log.read(eng.fi_dir / "audit.jsonl") if e.get("kind") == "papers_declined"]
    assert events and events[-1]["count"] == 2
    assert any(i.kind == "papers_declined" and "2 paper(s)" in i.why for i in todo.waiting(eng.quest_root))
    # A new paper later: only it is asked for, and the card says how many were already declined.
    eng.knowledge.asearch = lambda q, **kw: _async([*docs, _paywalled("10.1/c")])  # type: ignore[method-assign]
    eng._resumed_from_pause = None  # a later literature pass in a run that did not resume from the papers pause
    with pytest.raises(_Paused):
        await eng._node_literature(dict(state))
    assert asked[-1]["headline"] == "download 1 paywalled paper(s)"
    assert "2 other(s) you went on without" in asked[-1]["steps"][0]


async def _async(value):
    return value


def test_an_update_or_another_pause_does_not_count_as_going_on_without_them(tmp_path: Path) -> None:
    from core.engine import _papers_to_ask, _take_papers_declined, forget_papers_asked

    fi = tmp_path / ".fi"
    new, already = _papers_to_ask([_paywalled("10.1/a")], fi)
    assert len(new) == 1 and already == 0
    assert _take_papers_declined(fi, tmp_path, "plan") == set(), "the quest is going on from another pause"
    assert _take_papers_declined(fi, tmp_path, None) == set(), "a fresh run is not an answer"
    forget_papers_asked(fi)  # --update
    assert _take_papers_declined(fi, tmp_path, "papers") == set()
    assert _papers_to_ask([_paywalled("10.1/a")], fi)[0], "asked again after the quest was changed"
    (tmp_path / "inputs" / "papers").mkdir(parents=True)
    (tmp_path / "inputs" / "papers" / "a.pdf").write_bytes(b"%PDF-1.4")
    assert _take_papers_declined(fi, tmp_path, "papers") == set(), "a paper was added: nothing was declined"


def test_a_run_from_a_step_keeps_the_papers_already_declined(tmp_path: Path) -> None:
    from core.engine import _papers_to_ask, _take_papers_declined, forget_papers_asked, papers_declined_count

    fi = tmp_path / ".fi"
    _papers_to_ask([_paywalled("10.1/a")], fi)
    assert _take_papers_declined(fi, tmp_path, "papers") == {"doi:10.1/a"}
    _papers_to_ask([_paywalled("10.1/b")], fi)  # asked again later, not answered
    forget_papers_asked(fi, declined=False)  # --from write / re-open: the literature is not searched again
    assert papers_declined_count(fi) == 1 and not (fi / "papers_asked.json").exists()


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


def test_raising_the_http_base_never_shortens_a_step() -> None:
    raised = ProviderConfig.model_validate({"http_timeout_s": 1800}).node_http_timeout_s
    assert all(v >= 1800 for v in raised.values())
    with_empty = ProviderConfig.model_validate({"http_timeout_s": 1800, "node_http_timeout_s": {}}).node_http_timeout_s
    assert with_empty == raised, "an empty map changes no step"
    tight = ProviderConfig.model_validate({"http_timeout_s": 1800, "node_http_timeout_s": {"write": 60}})
    assert tight.node_http_timeout_s["write"] == 60, "a budget written on purpose is kept"


def test_the_whole_config_merges_too() -> None:
    cfg = Config.model_validate({"topic": "t", "provider": {"name": "openai", "node_http_timeout_s": {"implement": 1800}}})
    assert cfg.provider.node_http_timeout_s["write"] == ProviderConfig().node_http_timeout_s["write"]


# --- research pauses never suggest a refused setting ------------------------------------------------------------------


_ACTIONS = ("resume", "fix", "start a new quest", "go on", "choose", "put ", "read ", "download", "approve", "--")


@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize("kind", sorted(todo._ADVICE))
def test_every_research_card_keeps_an_action_and_contradicts_nothing(kind: str, frozen: bool) -> None:
    steps = ["The run differs from the frozen protocol. Set `engine.run_manifest_check: warn` to go on with the "
             "difference recorded.",
             "Resume to ask for the scripts again; set `execution.split_failure: warn` to let the quest run as one "
             "script instead.",
             "Set `engine.oracle_check: warn` and go on."]
    item = todo.pause_item(kind, "stopped", steps, profile="research", frozen=frozen)
    lines = [item.recommended, *item.alternatives, *item.steps]
    for line in lines:
        assert todo.refused_settings(line, "research") == [], (kind, line)
        if frozen:
            assert "--revise-plan" not in line, (kind, "the plan cannot change a frozen protocol", line)
    assert any(a in line.lower() for line in [item.recommended, *item.alternatives] for a in _ACTIONS), (kind, lines)
    assert "Resume to ask for the scripts again." in item.steps, "a clause after ';' is kept, not the whole line lost"
    assert item.steps[0] == "The run differs from the frozen protocol."


def test_the_frozen_oracle_card_says_the_oracle_cannot_change_inside_the_quest() -> None:
    card = todo.pause_item("oracle", "stopped", ["Set `engine.oracle_check: warn` and go on."], profile="research",
                           frozen=True)
    text = " ".join([card.recommended, *card.alternatives])
    assert "start a new quest" in text and "--revise-plan" not in text


def test_a_quoted_value_the_profile_keeps_is_not_refused() -> None:
    assert todo.refused_settings('`engine.run_manifest_check: "block"`', "research") == []
    assert todo.refused_settings("`engine.run_manifest_check: 'warn'`", "research")


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


def test_research_cards_offer_only_what_the_engine_allows() -> None:
    card = todo.pause_item("numeric", "stopped", [], profile="research", frozen=True)
    assert any("Go on unchanged" in a and "does not count the run as holding to its protocol" in a
               for a in card.alternatives), "research lets the warnings be accepted, and the card says what it costs"


def test_a_key_in_a_url_is_not_written_to_the_log() -> None:
    from types import SimpleNamespace

    from core.provider import _retry_line

    rs = SimpleNamespace(attempt_number=1, next_action=None, outcome=SimpleNamespace(
        exception=lambda: httpx.ConnectError(
            "GET https://api.x/v1?key=AIzaSyABCDEFG123&token=tok_123456789 (header token: hdr_abcdefgh99)")))
    line = _retry_line("write", "gemini over HTTP", rs, 4)
    assert "AIzaSyABCDEFG123" not in line and "tok_123456789" not in line and "hdr_abcdefgh99" not in line


def test_a_call_outside_the_engine_reaches_the_quest_run_log(tmp_path: Path) -> None:
    from core.provider import quest_run_log

    assert quest_run_log(tmp_path) is None, "no quest folder"
    (tmp_path / ".fi").mkdir()
    log = quest_run_log(tmp_path)
    log.warning("[slides] the model call failed")
    log.info("not a warning")
    text = (tmp_path / ".fi" / "run.log").read_text(encoding="utf-8")
    assert "[slides] the model call failed" in text and "not a warning" not in text
