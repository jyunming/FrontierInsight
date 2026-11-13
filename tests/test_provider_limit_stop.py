"""A provider limit (or refused sign-in) that does not clear within the run stops the quest; it is never an empty answer.

Real failure: the codex CLI answered every call with "You've hit your usage limit ... try again at Nov 4th, 2026 10:49
AM." and the repair of code-review findings logged "a request could not be made", took each as "the reply held no
test", and carried on."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from core import finding_repair as fr
from core.engine import ModelUnavailable
from core.provider import ProviderLimitError, ProviderUnusable, _CliTransientError, _retry_cli_error, limit_resumes_at
from tests.test_model_unavailable import _engine

LIMIT = ("C:\\Users\\x\\AppData\\Roaming\\npm\\codex.CMD exited rc=1: You\u2019ve hit your usage limit. Upgrade to Pro "
         "(https://chatgpt.com/explore/pro), visit https://chatgpt.com/settings/usage to purchase more credits or try "
         "again at Nov 4th, 2026 10:49 AM.")


def test_the_resume_time_is_read_from_the_providers_text() -> None:
    assert limit_resumes_at(LIMIT) == "Nov 4th, 2026 10:49 AM"
    assert limit_resumes_at("You've hit your usage limit. Try again at 4:51 AM.") == "4:51 AM"
    assert limit_resumes_at("rate limit exceeded \u00b7 resets 5pm (UTC)") == "5pm"
    assert limit_resumes_at("stream disconnected") is None
    assert limit_resumes_at("rate limit exceeded · resets in 20s") is None, "a wait is not a time"
    assert limit_resumes_at("You've hit your usage limit. Try again in 20 seconds.") is None


def _client(monkeypatch: pytest.MonkeyPatch, text: str = LIMIT) -> tuple[Any, list[int]]:
    from unittest.mock import MagicMock

    from core.provider import LLMClient, ResolvedEndpoint

    calls: list[int] = []
    client = LLMClient(ResolvedEndpoint(base_url="", model="gpt-x", api_key="", transport="cli", provider_name="codex_cli"),
                       http=MagicMock())

    async def impl(messages: Any, **kw: Any) -> str:
        calls.append(1)
        raise _CliTransientError(text)

    monkeypatch.setattr(client, "_chat_impl", impl)
    return client, calls


@pytest.mark.asyncio
async def test_a_cli_limit_raises_its_own_error_with_the_resume_time(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _calls = _client(monkeypatch)
    with pytest.raises(ProviderLimitError) as caught:
        await client.chat([{"role": "user", "content": "hi"}], node="plan")
    err = caught.value
    assert (err.provider, err.model, err.resumes_at) == ("codex_cli", "gpt-x", "Nov 4th, 2026 10:49 AM")
    assert isinstance(err, _CliTransientError) and not _retry_cli_error(err), "still a CLI failure, still not retried"


@pytest.mark.asyncio
async def test_after_a_limit_every_later_call_on_the_model_raises_at_once(monkeypatch: pytest.MonkeyPatch) -> None:
    client, calls = _client(monkeypatch)
    for _ in range(3):
        with pytest.raises(ProviderLimitError):
            await client.chat([{"role": "user", "content": "hi"}])
    assert len(calls) == 1, "the CLI was started once; the later calls did not start it"
    # another model of the same client is not blocked by it
    with pytest.raises(ProviderLimitError):
        await client.chat([{"role": "user", "content": "hi"}], model="other")
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_a_passing_failure_is_not_remembered(monkeypatch: pytest.MonkeyPatch) -> None:
    client, calls = _client(monkeypatch, "codex.CMD exited rc=1: stream disconnected")
    for _ in range(2):
        with pytest.raises(_CliTransientError) as caught:
            await client.chat([{"role": "user", "content": "hi"}])
        assert not isinstance(caught.value, ProviderLimitError)
    assert len(calls) == 2


class _Env:
    last_spend = None

    def __init__(self, eng: Any) -> None:
        self.log = eng._log
        self._eng = eng

    async def ask(self, prompt: str, who: str, kind: str) -> str:
        return await self._eng._chat(prompt, node="implement.finding_test")


class _Limited:
    async def chat(self, messages: Any, **kw: Any) -> str:
        raise ProviderLimitError(LIMIT, provider="codex_cli", model="gpt-x", resumes_at="Nov 4th, 2026 10:49 AM")


@pytest.mark.asyncio
async def test_finding_repair_does_not_turn_a_limit_into_an_empty_reply(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eng = _engine(tmp_path, monkeypatch, _Limited())
    with pytest.raises(ProviderUnusable):
        await fr._ask(_Env(eng), "p", "code", "test")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_the_step_stops_with_one_plain_sentence_and_is_not_kept(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eng = _engine(tmp_path, monkeypatch, _Limited())

    async def step(state: Any) -> dict[str, Any]:
        await fr._ask(_Env(eng), "p", "code", "test")  # type: ignore[arg-type]
        return {"decided": "went on"}

    with pytest.raises(ModelUnavailable) as caught:
        await eng._audited("execute", step)({})
    say = str(caught.value)
    assert "usage limit was reached" in say and "codex_cli" in say and "gpt-x" in say
    assert "Nothing the current step decided was kept" in say
    assert "https" not in say and "implement.finding" not in say and "execute" not in say, "no internal id, no provider text"
    assert "after Nov 4th, 2026 10:49 AM" in say and "--resume" in say and eng.quest_id in say


@pytest.mark.asyncio
async def test_a_limit_escaping_a_step_is_the_same_stop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eng = _engine(tmp_path, monkeypatch, _Limited())

    async def step(state: Any) -> dict[str, Any]:
        raise ProviderUnusable(ProviderLimitError(LIMIT, provider="codex_cli", model="gpt-x", resumes_at="Nov 4th, 2026 10:49 AM"))

    with pytest.raises(ModelUnavailable) as caught:
        await eng._audited("execute", step)({})
    assert "after Nov 4th, 2026 10:49 AM" in str(caught.value) and isinstance(caught.value.__cause__, ProviderLimitError)


def test_the_failure_file_says_the_same_in_plain_words() -> None:
    from core import crash_kind

    err = ProviderLimitError(LIMIT, provider="codex_cli", model="gpt-x", resumes_at="Nov 4th, 2026 10:49 AM")
    stop = ModelUnavailable("x")
    stop.__cause__ = err
    got = crash_kind.classify(stop, provider="codex_cli", model="gpt-x")
    assert got.kind == "setup" and "used up its allowance" in got.say and "Nov 4th, 2026 10:49 AM" in got.do


# --- the limit is not an ordinary failure anywhere ------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_except_exception_between_the_call_and_the_step_can_take_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eng = _engine(tmp_path, monkeypatch, _Limited())
    assert not issubclass(ProviderUnusable, Exception)
    with pytest.raises(ProviderUnusable):
        try:
            await eng._chat("p", node="anything")
        except Exception:  # noqa: BLE001 -- the catch-all every caller has
            pytest.fail("a limit was taken for an ordinary failure")


@pytest.mark.asyncio
async def test_a_gathered_call_that_met_the_limit_is_not_a_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """asyncio.gather(return_exceptions=True) hands a BaseException back as a value: the callers that gather model calls
    (the idea tournament, the review panel) raise it again."""
    import asyncio

    eng = _engine(tmp_path, monkeypatch, _Limited())
    got = await asyncio.gather(eng._chat("p", node="ideate"), return_exceptions=True)
    assert isinstance(got[0], ProviderUnusable) and not isinstance(got[0], Exception)


@pytest.mark.asyncio
async def test_a_tolerated_call_is_still_left_out_not_a_stop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eng = _engine(tmp_path, monkeypatch, _Limited())
    with pytest.raises(_CliTransientError):  # an ensemble member: the original error, the step goes on without it
        await eng._recorded_call("cross_check", [], lambda: _raise_limit(), tolerated=True)


async def _raise_limit() -> str:
    raise ProviderLimitError(LIMIT, provider="codex_cli", model="gpt-x")


# --- the original failure shape: the code model hits the limit in the middle of the repair step ---------------------------


class _LimitAfter:
    """The code model answers its first request and then hits the limit; every other model keeps answering."""

    def __init__(self, answers: list[str]) -> None:
        self.answers, self.asked = list(answers), 0

    async def chat(self, messages: Any, **kw: Any) -> str:
        if str(kw.get("node", "")).startswith("implement.finding"):
            self.asked += 1
            if not self.answers:
                raise _CliTransientError(LIMIT)
            return self.answers.pop(0)
        return "another model answers"


@pytest.mark.asyncio
async def test_a_limit_in_the_middle_of_the_repair_leaves_the_rest_open_and_the_budget_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.test_finding_repair import FAILING, Run, _block

    run = Run(tmp_path / "q", [{"id": "F-1"}, {"id": "F-2"}, {"id": "F-3"}], {})
    client = _LimitAfter([_block(FAILING)])  # F-1's test is made; the request for its repair meets the limit
    eng = _engine(tmp_path / "e", monkeypatch, client)

    async def ask(prompt: str, who: str, kind: str) -> str:
        return await eng._chat(prompt, node=f"implement.finding_{'test' if kind == 'test' else 'repair'}")

    run.env.ask = ask  # the engine's own path to the model, as Engine._repair_review_findings builds it

    async def step(state: Any) -> dict[str, Any]:
        await fr.repair_issues(run.env)
        return {"committed": True}

    with pytest.raises(ModelUnavailable) as caught:
        await eng._audited("execute", step)({})
    say = str(caught.value)
    assert "usage limit was reached" in say and "Nothing the current step decided was kept" in say and "--resume" in say
    issues = run.record["issues"]
    assert client.asked == 2, "after the hit nothing more was asked"
    assert issues["F-1"]["outcome"] != "gave_up" and issues["F-1"].get("repairs") == []
    assert issues["F-2"]["outcome"] == "open" and issues["F-3"]["outcome"] == "open", "no 'not demonstrated' was written for them"
    assert not any("reply held no" in str(i.get("reason", "")) for i in issues.values())
    # the request for the repair was counted before it was made and never answered: it is given back
    assert issues["F-1"]["calls"]["repair"] == 0 and issues["F-1"]["calls"]["test"] == 1
    assert run.record["finding_calls"] == 1 and run.record.get("repair_calls", 0) == 0 and run.record["test_calls"] == 1


@pytest.mark.asyncio
async def test_a_second_quest_in_the_same_process_is_not_stopped_by_the_first(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    limited = _engine(tmp_path / "a", monkeypatch, _Limited())

    class _Fine:
        async def chat(self, messages: Any, **kw: Any) -> str:
            return "an answer"

    fine = _engine(tmp_path / "b", monkeypatch, _Fine())

    async def asks(eng: Any) -> dict[str, Any]:
        try:
            return {"reply": await eng._chat("p", node="plan")}
        except Exception:  # noqa: BLE001 -- the "could not ask" path
            return {"reply": ""}

    a, b = await asyncio.gather(eng_step(limited, asks), eng_step(fine, asks), return_exceptions=True)
    assert isinstance(a, ModelUnavailable) and b["reply"] == "an answer"
    # and the other quest goes on being served afterwards
    assert (await eng_step(fine, asks))["reply"] == "an answer"


async def eng_step(eng: Any, asks: Any) -> Any:
    async def step(state: Any) -> dict[str, Any]:
        return await asks(eng)

    return await eng._audited("plan", step)({})


# --- through the whole run --------------------------------------------------------------------------------------------------

from tests.test_engine_smoke import smoke_config  # noqa: E402,F401 -- the engine smoke test's config


@pytest.mark.asyncio
async def test_a_whole_run_ends_with_the_resume_sentence_and_commits_nothing_of_the_stopped_step(
    smoke_config: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.engine import Engine

    async def limited(self: Any, messages: Any, **kw: Any) -> str:
        raise _CliTransientError(LIMIT)

    monkeypatch.setattr("core.engine.LLMClient.chat", limited)
    engine = Engine(smoke_config)
    with pytest.raises(ModelUnavailable) as caught:
        await engine.run()
    say = str(caught.value)
    assert "usage limit was reached" in say and "after Nov 4th, 2026 10:49 AM" in say and f"--resume {engine.quest_id}" in say
    failed = (engine.quest_root / "quest_failed.md").read_text(encoding="utf-8")
    assert "Nov 4th, 2026 10:49 AM" in failed
    assert not (engine.quest_root / "paper.md").exists()


# --- round 3: refunds at the plan and design asks, the failure card, the one-shot command, what is remembered ---------------


def _unusable() -> ProviderUnusable:
    return ProviderUnusable(ProviderLimitError(LIMIT, provider="codex_cli", model="gpt-x", resumes_at="Nov 4th, 2026 10:49 AM"))


def test_the_plan_step_gives_its_ask_back_when_the_model_cannot_be_used(tmp_path: Path) -> None:
    import asyncio
    import json

    from core import plan
    from tests.test_code_review import _engine as _review_engine

    engine = _review_engine(tmp_path)
    plan.plan_path(engine.quest_root).parent.mkdir(parents=True, exist_ok=True)
    plan.plan_path(engine.quest_root).write_text("# plan\n", encoding="utf-8")

    async def complete(ask: Any, path: Any, **kwargs: Any) -> None:
        raise _unusable()

    engine._complete_plan_parts = complete  # type: ignore[method-assign]
    engine._progress = lambda *a, **k: None  # type: ignore[method-assign]
    with pytest.raises(ProviderUnusable):
        asyncio.run(engine._ask_plan_to_complete(
            {}, "sha", lacks=lambda design: ["uncertainty is missing"], request=lambda design, text: "ask",
            keep=lambda before, revised: revised, file="part_asked.json", what="settings", short="it", note="n"))
    record = json.loads((engine.fi_dir / "part_asked.json").read_text(encoding="utf-8"))
    assert "model could not be used" in json.dumps(record)
    assert not any(isinstance(v, int) and v > 0 for k, v in record.items() if k in ("asks", "count", "calls"))


def test_the_design_step_gives_its_ask_back_when_the_model_cannot_be_used(tmp_path: Path) -> None:
    import asyncio
    import json

    from tests.test_code_review import _engine as _review_engine

    engine = _review_engine(tmp_path)

    async def chat(prompt: str, **kwargs: Any) -> str:
        raise _unusable()

    engine._chat = chat  # type: ignore[method-assign]
    engine._progress = lambda *a, **k: None  # type: ignore[method-assign]
    with pytest.raises(ProviderUnusable):
        asyncio.run(engine._ask_design_for_the_topic({}, {"protocol": {}}, {}, [{"sentence": "the topic asks for power"}], iteration=1))
    record = json.loads((engine.fi_dir / "topic_design_asked.json").read_text(encoding="utf-8"))
    assert record["asks"] == 0


@pytest.mark.asyncio
async def test_a_call_outside_any_step_gets_the_failure_card_of_a_used_up_allowance(
    smoke_config: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.engine import Engine

    engine = Engine(smoke_config)

    async def raises() -> None:
        raise _unusable()

    monkeypatch.setattr(engine, "_preflight_required_skills", raises)
    with pytest.raises(ModelUnavailable) as caught:
        await engine.run()
    assert isinstance(caught.value.__cause__, ProviderLimitError)
    failed = (engine.quest_root / "quest_failed.md").read_text(encoding="utf-8")
    assert "used up its allowance" in failed and "Nov 4th, 2026 10:49 AM" in failed


@pytest.mark.asyncio
async def test_a_stop_after_a_swallowing_step_also_names_the_limit_as_its_cause(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eng = _engine(tmp_path, monkeypatch, _Limited())
    async def step(state: Any) -> dict[str, Any]:
        eng.__dict__["_provider_unavailable"] = "x: the ensemble"
        eng.__dict__["_provider_limit"] = ProviderLimitError(LIMIT, provider="codex_cli", model="gpt-x")
        return {}

    with pytest.raises(ModelUnavailable) as caught:
        await eng._audited("plan", step)({})
    assert isinstance(caught.value.__cause__, ProviderLimitError)


@pytest.mark.asyncio
async def test_the_one_shot_plan_rewrite_stops_with_the_plain_sentence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    import launch
    from core import plan
    from core.provider import ProxySupervisor
    from tests.test_plan import DESIGN, EXTRA, _cfg

    cfg = _cfg(tmp_path)
    quest_id = "1700000000-probe-abcdef"
    quest = cfg.output.output_dir / quest_id
    quest.mkdir(parents=True)
    text = plan.render("PROBE", EXTRA, DESIGN)
    plan.plan_path(quest).write_text(text, encoding="utf-8")
    plan.record_version(quest, text, by="model")

    async def limited(self: Any, messages: Any, **kw: Any) -> str:
        raise _CliTransientError(LIMIT)

    monkeypatch.setattr("core.engine.LLMClient.chat", limited)
    rc = await launch._revise_plan_once(cfg, quest_id, "change the method", supervisor=ProxySupervisor())
    err = capsys.readouterr().err
    assert rc == 1 and "usage limit was reached" in err and "--resume" in err and "Traceback" not in err


def _http_error(status: int) -> Any:
    import httpx

    req = httpx.Request("POST", "http://x/v1/chat")
    return httpx.HTTPStatusError("refused", request=req, response=httpx.Response(status, request=req, json={"error": "no"}))


@pytest.mark.asyncio
@pytest.mark.parametrize("status,remembered", [(403, False), (401, True), (402, True)])
async def test_only_a_limit_or_a_refused_sign_in_is_remembered(
    monkeypatch: pytest.MonkeyPatch, status: int, remembered: bool,
) -> None:
    from unittest.mock import MagicMock

    from core.provider import LLMClient, ResolvedEndpoint

    client = LLMClient(ResolvedEndpoint(base_url="http://x", model="m", api_key="k"), http=MagicMock())
    calls: list[int] = []

    async def impl(messages: Any, **kw: Any) -> str:
        calls.append(1)
        raise _http_error(status)

    monkeypatch.setattr(client, "_chat_impl", impl)
    for _ in range(2):
        with pytest.raises(Exception):  # noqa: B017 -- the original HTTP error either way
            await client.chat([{"role": "user", "content": "hi"}])
    assert len(calls) == (1 if remembered else 2)


@pytest.mark.asyncio
async def test_a_relative_wait_says_later_and_a_repeated_call_carries_its_own_note(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eng = _engine(tmp_path, monkeypatch, _Limited())
    err = ProviderLimitError("You have hit your usage limit. Try again in 20 seconds.", provider="codex_cli", model="gpt-x")
    assert "once the limit has reset" in eng._unavailable_message("", err) and "Resume with: python launch.py" in eng._unavailable_message("", err)
    client, _calls = _client(monkeypatch)
    for node in ("a", "b"):
        with pytest.raises(ProviderLimitError) as caught:
            await client.chat([{"role": "user", "content": "hi"}], node=node)
    notes = getattr(caught.value, "__notes__", [])
    assert len(notes) == 1 and "node=b" in notes[0]


@pytest.mark.asyncio
async def test_a_panelist_that_meets_the_limit_stops_the_review_at_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from core.config import ProviderConfig
    from core.engine import Engine
    from core.provider import LLMClient, resolve_endpoint
    from tests.test_review_panel import _mk_cfg

    eng = Engine(_mk_cfg(tmp_path, panel=["methodologist", "statistician", "devil_advocate"]))
    paper = eng.quest_root / "paper" / "paper.md"
    paper.parent.mkdir(parents=True, exist_ok=True)
    paper.write_text("# title\n\nbody.\n", encoding="utf-8")

    async def fake_chat(self: Any, messages: Any, **kw: Any) -> str:
        if "Statistician." in messages[-1]["content"]:
            raise _CliTransientError(LIMIT)
        return '{"verdict": "accept", "score": 4}'

    monkeypatch.setattr("core.provider.LLMClient.chat", fake_chat)
    eng._client = LLMClient(resolve_endpoint(ProviderConfig(name="openai", model="x")))
    with pytest.raises(ProviderUnusable):
        await eng._node_review({"topic": "t", "design": {}, "analysis": {}, "paper_md": str(paper), "iteration": 0})


@pytest.mark.asyncio
async def test_an_ensemble_whose_members_all_meet_the_limit_stops_with_the_limit_as_its_cause(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every member of an ensemble meets the usage limit: the step goes on without an answer, and its end stops the quest
    with the members' own error as the cause, so the failure card names the limit and its time."""
    from core import crash_kind
    from core.config import NodeEnsembleConfig

    eng = _engine(tmp_path, monkeypatch, _Limited())

    async def step(state: Any) -> dict[str, Any]:
        try:
            await eng._ensemble_chat("p", node="cross_check", ensemble_cfg=NodeEnsembleConfig(models=["a", "b"], merge="vote"))
        except Exception:
            pass  # a step that goes on without the ensemble's answer
        return {}

    with pytest.raises(ModelUnavailable) as caught:
        await eng._audited("cross_check", step)({})
    assert isinstance(caught.value.__cause__, ProviderLimitError)
    got = crash_kind.classify(caught.value, provider="codex_cli", model="gpt-x")
    assert got.kind == "setup" and "Nov 4th, 2026 10:49 AM" in got.do


@pytest.mark.asyncio
async def test_a_block_that_outlasts_the_retries_is_a_stop_that_says_so_and_keeps_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A call refused before the service, still refused after the client's own long retries: the step is not kept (no
    check is skipped on a call never answered) and the sentence names neither a usage limit nor a sign-in to renew."""
    from core.provider import _cli_failure

    page = "Failed to authenticate. API Error: 403 Attention Required! | A Gateway"

    class _Blocked:
        async def chat(self, messages: Any, **kw: Any) -> str:
            raise _cli_failure(f"claude exited rc=1 (stream error: {page}): ", "authentication_failed", page)

    eng = _engine(tmp_path, monkeypatch, _Blocked())

    async def step(state: Any) -> dict[str, Any]:
        try:
            await fr._ask(_Env(eng), "p", "code", "test")  # type: ignore[arg-type]
        except Exception:  # noqa: BLE001 -- a step that would go on without its check
            pass
        return {"decided": "went on"}

    with pytest.raises(ModelUnavailable) as caught:
        await eng._audited("execute", step)({})
    say = str(caught.value)
    assert "refused on the way to its service" in say and "Nothing the current step decided was kept" in say
    assert "usage limit was reached" not in say and "/login" not in say and "--resume" in say and eng.quest_id in say
