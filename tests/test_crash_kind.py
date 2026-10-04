"""A quest that stops on an error says what happened and the one thing to do, in words a scientist can act on.

``core/crash_kind.py`` sorts a failure: a passing problem (tried again by the engine itself before it stops), a setup
problem (one action named), a problem in FI's own code (said plainly to be FI's, not the person's), or an unexpected
one. ``quest_failed.md``, ``.fi/failure.json``, the to-do card, the web quest page and the CLI all lead with that; the
exception and the log tail are details for a bug report. No real model is called.
"""

from __future__ import annotations

import errno
import json
from pathlib import Path

import httpx
import pytest

from core import crash_kind as ck


def _raised(exc: BaseException) -> BaseException:
    """``exc`` with a traceback through this file (a file in FI's tree)."""
    try:
        raise exc
    except BaseException as e:  # noqa: BLE001
        return e


def _http(status: int, body: dict | None = None) -> httpx.HTTPStatusError:
    req = httpx.Request("POST", "https://api.example.com/v1/chat/completions")
    resp = httpx.Response(status, request=req, json=body or {"error": {"message": "x"}})
    return httpx.HTTPStatusError(f"HTTP {status}", request=req, response=resp)


# ---- the sorting ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("exc", [
    httpx.ReadTimeout("read timed out"),
    httpx.ConnectError("connection refused"),
    TimeoutError("the call timed out"),
    ConnectionResetError("connection reset by peer"),
    _http(429, {"error": {"message": "rate limit, retry in 20s"}}),
    _http(503),
])
def test_a_passing_problem_is_transient(exc: BaseException) -> None:
    f = ck.classify(exc, node="design")
    assert f.kind == "transient", f
    assert "passing problem" in f.say and "`design`" in f.say
    assert "continue the quest" in f.do


def test_a_file_another_program_holds_is_transient() -> None:
    err = PermissionError(13, "The process cannot access the file because it is being used by another process")
    err.winerror = 32  # type: ignore[attr-defined]
    assert ck.classify(err).kind == "transient"


@pytest.mark.parametrize("exc, words", [
    (_http(401), "sign in"),
    (_http(403), "sign in"),
    (RuntimeError("[FI] HTTP 401: openai did not accept the API key FI sent."), "sign in"),
    (RuntimeError("You've hit your weekly limit · resets Mon 9am"), "allowance"),
    (_http(429, {"error": {"message": "you have reached your monthly usage limit"}}), "allowance"),
    (_http(404, {"error": {"message": "model 'gemma9' not found"}}), "does not know the model"),
    (ModuleNotFoundError("No module named 'scipy'", name="scipy"), "not installed (scipy)"),
    (FileNotFoundError(errno.ENOENT, "No such file", "pandoc"), "not found on this machine (pandoc)"),
    (OSError(errno.ENOSPC, "No space left on device"), "disk is full"),
])
def test_a_setup_problem_names_one_action(exc: BaseException, words: str) -> None:
    f = ck.classify(exc, provider="openai", model="gemma9")
    assert f.kind == "setup", f
    assert words in f.say
    assert f.do and "then continue the quest" in f.do or "Try again later" in f.do


def test_a_used_up_quota_is_not_tried_again_as_a_passing_problem() -> None:
    """A 429 that says the allowance is used up cannot clear in minutes: setup, not transient."""
    exc = _http(429, {"error": {"type": "insufficient_quota", "message": "You exceeded your current quota"}})
    assert ck.classify(exc).kind == "setup"


def test_a_programming_error_in_fi_code_is_fi_s_problem() -> None:
    def fi_code() -> None:
        {}["missing"]  # noqa: B018 -- a KeyError raised from a file in FI's tree

    try:
        fi_code()
    except KeyError as e:
        f = ck.classify(e, node="analyze")
    assert f.kind == "fi"
    assert "problem in FI itself" in f.say and "not in your settings or your research" in f.say
    assert "report it" in f.do and "quest_failed.md" in f.do
    assert "KeyError" in f.detail


def test_a_programming_error_from_outside_fi_is_not_blamed_on_fi() -> None:
    """No traceback through FI's files (here: never raised at all): FI cannot say it is its own bug."""
    f = ck.classify(KeyError("x"))
    assert f.kind == "unknown"


def test_a_frame_in_an_installed_package_under_fi_is_not_fi_code(tmp_path: Path) -> None:
    fake = ck._FI_ROOT / ".venv" / "Lib" / "site-packages" / "pkg.py"
    code = compile("def f():\n    raise TypeError('in a package')\n", str(fake), "exec")
    ns: dict = {}
    exec(code, ns)  # noqa: S102 -- a frame whose file name is a package inside FI's folder
    try:
        ns["f"]()
    except TypeError as e:
        # This test's own frame is FI's: only the package frame is excluded, so the error is still FI's here.
        assert ck._fi_frame(e) is True
    e2 = TypeError("bare")
    assert ck._fi_frame(e2) is False


def test_an_unexpected_error_says_so_and_how_to_continue() -> None:
    f = ck.classify(_raised(RuntimeError("pandoc not on PATH — preflight refused to continue")), node="write")
    assert f.kind == "unknown"
    assert "did not expect" in f.say and "pandoc not on PATH" in f.say and "`write`" in f.say
    assert "picks up at the step that stopped" in f.do


def test_the_cause_of_an_error_is_read_too() -> None:
    try:
        try:
            raise httpx.ReadTimeout("timed out")
        except httpx.ReadTimeout as inner:
            raise RuntimeError("the design step failed") from inner
    except RuntimeError as outer:
        assert ck.classify(outer).kind == "transient"


def test_a_pre_graph_label_is_not_shown_as_a_step() -> None:
    f = ck.classify(_raised(RuntimeError("x")), node="(pre-graph stage — preflight / endpoint resolution / setup)")
    assert "(pre-graph" not in f.say


def test_the_record_round_trips_and_is_cleared(tmp_path: Path) -> None:
    f = ck.classify(httpx.ReadTimeout("t"), node="design", retries=2)
    ck.write(tmp_path, f)
    back = ck.read(tmp_path)
    assert back and back["kind"] == "transient" and back["retries"] == 2 and back["title"]
    ck.clear(tmp_path)
    assert ck.read(tmp_path) is None


# ---- what the person sees ------------------------------------------------------------------------------------------

def test_the_to_do_card_leads_with_what_happened(tmp_path: Path) -> None:
    from core import todo

    (tmp_path / ".fi").mkdir()
    (tmp_path / "quest_failed.md").write_text("x", encoding="utf-8")
    ck.write(tmp_path / ".fi", ck.classify(_http(401), provider="openai"))
    items = [i for i in todo.waiting(tmp_path) if i.kind == "failed"]
    assert items and "FI could not sign in to openai" in items[0].why
    assert "Sign in to it again" in items[0].recommended and "quest_failed.md" in items[0].recommended


def test_an_older_failure_file_without_a_record_keeps_the_old_card(tmp_path: Path) -> None:
    from core import todo

    (tmp_path / "quest_failed.md").write_text("x", encoding="utf-8")
    items = [i for i in todo.waiting(tmp_path) if i.kind == "failed"]
    assert items and "quest_failed.md says where" in items[0].why


def test_the_web_page_reads_the_plain_words(tmp_path: Path) -> None:
    from web.server import _read_quest_failed_md

    (tmp_path / ".fi").mkdir()
    (tmp_path / "quest_failed.md").write_text(
        "# The quest stopped\n\n## Details (for a bug report)\n\n**Failing node:** `design`\n\n## What broke\n\n"
        "```\nReadTimeout: t\n```\n", encoding="utf-8")
    ck.write(tmp_path / ".fi", ck.classify(httpx.ReadTimeout("t"), node="design"))
    info = _read_quest_failed_md(tmp_path)
    assert info and info["kind"] == "transient" and "passing problem" in info["say"] and info["do"]
    assert info["failing_node"] == "design" and info["what_broke"] == "ReadTimeout: t"


def test_the_terminal_prints_the_plain_words(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import launch
    from types import SimpleNamespace

    eng = SimpleNamespace(fi_dir=tmp_path / ".fi", quest_id="q1", quest_root=tmp_path)
    assert launch._print_quest_failure(eng) is False  # nothing recorded: the caller keeps the traceback
    ck.write(eng.fi_dir, ck.classify(_http(401), provider="openai"))
    assert launch._print_quest_failure(eng) is True
    err = capsys.readouterr().err
    assert "FI could not sign in to openai" in err and "What you can do:" in err and "fi --resume q1" in err


# ---- the engine ----------------------------------------------------------------------------------------------------

def _quest_config(tmp_path: Path, title: str):
    from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig

    return Config(
        topic=f"a topic for the {title} test", title=title,
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, auto_accept_on_pass=True),
        execution=ExecutionConfig(sandbox="venv", timeout_s=120),
        knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "outputs"),
    )


@pytest.fixture
def _no_waits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ck, "RETRY_WAITS_S", (0.0, 0.0))
    monkeypatch.setenv("OPENAI_API_KEY", "k")


@pytest.mark.asyncio
async def test_a_step_that_fails_on_a_passing_problem_is_tried_again(tmp_path, monkeypatch, _no_waits) -> None:
    """The design step times out once (past the provider's own retries): the engine waits and runs the step again by
    itself, and the quest goes on (here to the next stop, a filtered answer) with no failure written."""
    from core.engine import Engine
    from core.provider import ModelAnswerFiltered
    from tests.test_engine_smoke import _classify, _fake_response_for

    calls = {"design": 0}

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        if _classify(prompt) == "Experiment Design":
            calls["design"] += 1
            if calls["design"] == 1:
                raise httpx.ReadTimeout("read timed out")
            raise ModelAnswerFiltered("withheld by the content filter", model="m")
        return _fake_response_for(prompt)

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)
    eng = Engine(_quest_config(tmp_path, "transient-once"))
    await eng.run()
    assert calls["design"] >= 2
    assert not (eng.quest_root / "quest_failed.md").exists()
    assert ck.read(eng.fi_dir) is None
    assert json.loads((eng.fi_dir / "pause.json").read_text(encoding="utf-8"))["kind"] == "model_output"
    assert "trying it again" in (eng.fi_dir / "run.log").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_a_passing_problem_that_does_not_pass_stops_with_plain_words(tmp_path, monkeypatch, _no_waits) -> None:
    from core.engine import Engine
    from tests.test_engine_smoke import _classify, _fake_response_for

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        if _classify(prompt) == "Experiment Design":
            raise httpx.ReadTimeout("read timed out")
        return _fake_response_for(prompt)

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)
    eng = Engine(_quest_config(tmp_path, "transient-always"))
    with pytest.raises(httpx.ReadTimeout):
        await eng.run()
    record = ck.read(eng.fi_dir)
    assert record and record["kind"] == "transient" and record["retries"] == len(ck.RETRY_WAITS_S)
    body = (eng.quest_root / "quest_failed.md").read_text(encoding="utf-8")
    # The plain words come first; the exception is under the details heading.
    assert body.index("**What happened:**") < body.index("**What you can do:**") < body.index("## Details")
    assert body.index("## Details") < body.index("ReadTimeout")
    assert "tried the step again 2 time(s)" in body and "--resume" in body
    assert "Traceback" in (eng.fi_dir / "run.log").read_text(encoding="utf-8")  # the full trace is kept for a report


@pytest.mark.asyncio
async def test_an_fi_bug_is_not_tried_again(tmp_path, monkeypatch, _no_waits) -> None:
    from core.engine import Engine
    from tests.test_engine_smoke import _classify, _fake_response_for

    calls = {"design": 0}

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        if _classify(prompt) == "Experiment Design":
            calls["design"] += 1
            return {}["no such key"]  # a programming error raised in FI's tree
        return _fake_response_for(prompt)

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)
    eng = Engine(_quest_config(tmp_path, "fi-bug"))
    with pytest.raises(KeyError):
        await eng.run()
    assert calls["design"] == 1
    record = ck.read(eng.fi_dir)
    assert record and record["kind"] == "fi"
    body = (eng.quest_root / "quest_failed.md").read_text(encoding="utf-8")
    assert "This is a problem in FI itself" in body
