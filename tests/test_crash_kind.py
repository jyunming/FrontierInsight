"""A quest that stops on an error says what happened in words a scientist can act on, and asks nothing of them but an
action on their own machine or account.

``core/crash_kind.py`` sorts a failure: a passing problem (run again by the engine itself before it stops), a setup
problem (one action named), a problem in FI's own code (said plainly to be FI's, nothing for the person to fix), or an
unexpected one (nothing to debug either). ``quest_failed.md``, ``.fi/failure.json``, the to-do card, the web quest
page and the CLI all lead with that; the exception and the log tail are details. No real model is called.
"""

from __future__ import annotations

import errno
import json
import time
from pathlib import Path

import httpx
import pytest

from core import crash_kind as ck


def _raised(exc: BaseException) -> BaseException:
    """``exc`` with a traceback (raised in this test file, which is not FI's own code)."""
    try:
        raise exc
    except BaseException as e:  # noqa: BLE001
        return e


def _fi_code(src: str):
    """A function whose frames are in FI's own code (compiled under ``core/``), for a bug raised inside FI."""
    ns: dict = {}
    exec(compile(src, str(ck._CORE / "_planted_bug.py"), "exec"), ns)  # noqa: S102
    return ns["f"]


def _http(status: int, body: dict | None = None, headers: dict | None = None) -> httpx.HTTPStatusError:
    req = httpx.Request("POST", "https://api.example.com/v1/chat/completions")
    resp = httpx.Response(status, request=req, json=body or {"error": {"message": "x"}}, headers=headers)
    return httpx.HTTPStatusError(f"HTTP {status}", request=req, response=resp)


# ---- passing problems ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("exc", [
    httpx.ReadTimeout("read timed out"),
    TimeoutError("the call timed out"),
    ConnectionResetError("connection reset by peer"),
    _http(429, {"error": {"message": "rate limit, retry in 20s"}}),
    _http(503),
])
def test_a_passing_problem_is_transient_and_asks_nothing(exc: BaseException) -> None:
    f = ck.classify(exc, node="design")
    assert f.kind == "transient", f
    assert "passing problem" in f.say and "`design`" in f.say
    assert "Nothing for you to fix" in f.do


def test_a_per_minute_limit_that_mentions_quota_is_still_a_passing_problem() -> None:
    """Gemini's per-minute limit says "Quota exceeded ... per minute" with a Retry-After: the provider's rule reads it
    as a rate limit (tried again), and so must the sorting (review finding 1)."""
    exc = _http(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
                                "message": "Quota exceeded for metric: requests, limit: 15 per minute. Please retry "
                                           "in 20s. See https://ai.google.dev/billing"}},
                headers={"Retry-After": "20"})
    assert ck.classify(exc, provider="gemini").kind == "transient"


def test_a_cli_that_hangs_without_answering_is_not_run_again() -> None:
    from core.provider import _CliWedgeError

    assert ck.classify(_CliWedgeError("the CLI closed its output without an answer")).kind != "transient"


# ---- setup problems: one action, in one sentence ---------------------------------------------------------------------

def _winerror(cls, winerror: int, msg: str, filename: str | None = None):
    err = cls(errno.ENOENT if winerror == 2 else errno.EACCES, msg, filename)
    err.winerror = winerror  # type: ignore[attr-defined]
    return err


@pytest.mark.parametrize("exc, words", [
    (_http(401), "sign in"),
    (_http(403), "sign in"),
    (RuntimeError("[FI] HTTP 401: openai did not accept the API key FI sent."), "sign in"),
    (RuntimeError("You've hit your weekly limit - resets Mon 9am"), "allowance"),
    (_http(429, {"error": {"type": "insufficient_quota", "message": "You exceeded your current quota"}}), "allowance"),
    (_http(404, {"error": {"message": "model 'gemma9' not found"}}), "does not know the model"),
    (ModuleNotFoundError("No module named 'scipy'", name="scipy"), "not installed (scipy)"),
    (FileNotFoundError(errno.ENOENT, "No such file", "pandoc"), "not found on this machine (pandoc)"),
    (_winerror(FileNotFoundError, 2, "The system cannot find the file specified"), "not found on this machine"),
    (OSError(errno.ENOSPC, "No space left on device"), "disk is full"),
    (httpx.UnsupportedProtocol("Request URL is missing an 'http://' or 'https://' protocol."), "not a valid web address"),
    (httpx.ConnectError("[WinError 10061] No connection could be made because the target machine actively refused it"),
     "is not running"),
    (RuntimeError("bridge connection dropped"), "VS Code's connection"),
    (_winerror(PermissionError, 32, "being used by another process", "C:/q/data/results.csv"), "(results.csv)"),
])
def test_a_setup_problem_names_one_action(exc: BaseException, words: str) -> None:
    f = ck.classify(exc, provider="openai", model="gemma9")
    assert f.kind == "setup", f
    assert words in f.say
    assert f.do.count(".") <= 2 and "continue the quest" in f.do.lower() or "@fi /resume" in f.do


def test_a_missing_input_file_is_not_a_missing_model_or_program() -> None:
    """``execution.inputs`` naming a file that is not there says "does not exist": not a model problem, and not a
    program to install (review finding 2)."""
    f = ck.classify(_raised(FileNotFoundError("execution.inputs: 'data/x.csv' does not exist.")), model="gpt-5")
    assert f.kind == "unknown" and "data/x.csv" in f.say and "does not know the model" not in f.say


def test_a_404_that_is_not_a_model_call_is_not_a_missing_model() -> None:
    class APIError(Exception):  # docker-py's shape: an exception with a .response carrying a 404
        def __init__(self) -> None:
            super().__init__("404 Client Error: No such image: python:3.11")
            self.response = httpx.Response(404, request=httpx.Request("GET", "http://docker/images"))

    assert "does not know the model" not in ck.classify(_raised(APIError())).say


# ---- FI's own problems and the rest ----------------------------------------------------------------------------------

def test_a_programming_error_raised_in_fi_code_is_fi_s_problem() -> None:
    f_bug = _fi_code("def f():\n    return {}['missing']\n")
    try:
        f_bug()
    except KeyError as e:
        f = ck.classify(e, node="analyze")
    assert f.kind == "fi"
    assert "problem in FI itself" in f.say and "not in your settings or your research" in f.say
    assert "Nothing for you to fix" in f.do and "report" not in f.do.lower()
    assert f.detail.startswith("KeyError")


def test_a_programming_error_raised_in_another_package_is_not_blamed_on_fi() -> None:
    """Every escaping error has FI's engine further out; only where it was raised counts (review finding 7)."""
    import textwrap

    try:
        textwrap.dedent(5)  # type: ignore[arg-type]
    except TypeError as e:
        f = ck.classify(e)
    assert f.kind == "unknown"


def test_a_bug_raised_while_handling_a_timeout_is_still_the_bug() -> None:
    """A KeyError raised inside an ``except ReadTimeout`` is FI's bug, not a passing problem (review finding 8)."""
    f_bug = _fi_code("import httpx\ndef f():\n    try:\n        raise httpx.ReadTimeout('t')\n"
                     "    except httpx.ReadTimeout:\n        return {}['k']\n")
    try:
        f_bug()
    except KeyError as e:
        assert ck.classify(e).kind == "fi"


def test_an_explicit_cause_is_read() -> None:
    try:
        try:
            raise httpx.ReadTimeout("timed out")
        except httpx.ReadTimeout as inner:
            raise RuntimeError("the design step failed") from inner
    except RuntimeError as outer:
        assert ck.classify(outer).kind == "transient"


def test_an_unexpected_error_asks_nothing_to_debug() -> None:
    f = ck.classify(_raised(RuntimeError("pandoc not on PATH — preflight refused to continue")), node="write")
    assert f.kind == "unknown"
    assert "did not expect" in f.say and "pandoc not on PATH" in f.say and "`write`" in f.say
    assert "Nothing for you to fix" in f.do and "picks up at the step that stopped" in f.do


def test_the_detail_is_the_error_not_its_note() -> None:
    exc = RuntimeError("boom happened")
    exc.add_note("[FI] provider=openai, transport=http, model=m, node=design")
    assert ck.classify(exc).detail == "RuntimeError: boom happened"


def test_sorting_never_raises() -> None:
    class Odd(Exception):
        def __str__(self) -> str:
            raise ValueError("no text")

    f = ck.classify(Odd())
    assert f.kind == "unknown" and f.say


def test_a_pre_graph_label_is_not_shown_as_a_step() -> None:
    f = ck.classify(_raised(RuntimeError("x")), node="(pre-graph stage — preflight / endpoint resolution / setup)")
    assert "(pre-graph" not in f.say


def test_the_record_round_trips_is_dated_and_is_cleared(tmp_path: Path) -> None:
    before = time.time()
    ck.write(tmp_path, ck.classify(httpx.ReadTimeout("t"), node="design", retries=2))
    back = ck.read(tmp_path)
    assert back and back["kind"] == "transient" and back["retries"] == 2 and back["title"] and back["at"] >= before
    assert ck.read(tmp_path, since=time.time() + 60) is None  # an earlier run's record is not this run's
    ck.clear(tmp_path)
    assert ck.read(tmp_path) is None


# ---- what the person sees --------------------------------------------------------------------------------------------

def test_the_to_do_card_leads_with_what_happened(tmp_path: Path) -> None:
    from core import todo

    (tmp_path / ".fi").mkdir()
    (tmp_path / "quest_failed.md").write_text("x", encoding="utf-8")
    ck.write(tmp_path / ".fi", ck.classify(_http(401), provider="openai"))
    items = [i for i in todo.waiting(tmp_path) if i.kind == "failed"]
    assert items and "FI could not sign in to openai" in items[0].why
    assert "Sign in to it again" in items[0].recommended and "--resume" in items[0].recommended


def test_an_older_failure_file_without_a_record_keeps_the_old_card(tmp_path: Path) -> None:
    from core import todo

    (tmp_path / "quest_failed.md").write_text("x", encoding="utf-8")
    items = [i for i in todo.waiting(tmp_path) if i.kind == "failed"]
    assert items and "quest_failed.md says where" in items[0].why


def test_the_web_page_reads_the_plain_words(tmp_path: Path) -> None:
    from web.server import _read_quest_failed_md

    (tmp_path / ".fi").mkdir()
    (tmp_path / "quest_failed.md").write_text(
        "# The quest stopped\n\n```bash\npython launch.py --resume q\n```\n\n## Details (for a bug report)\n\n"
        "**Failing node:** `design`\n\n## What broke\n\n```\nReadTimeout: t\n```\n", encoding="utf-8")
    ck.write(tmp_path / ".fi", ck.classify(httpx.ReadTimeout("t"), node="design"))
    info = _read_quest_failed_md(tmp_path)
    assert info and info["kind"] == "transient" and "passing problem" in info["say"] and info["do"]
    assert info["failing_node"] == "design" and info["what_broke"] == "ReadTimeout: t"


def test_the_terminal_prints_only_this_run_s_plain_words(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import launch
    from types import SimpleNamespace

    eng = SimpleNamespace(fi_dir=tmp_path / ".fi", quest_id="q1", quest_root=tmp_path)
    assert launch._print_quest_failure(eng) is False  # the engine never started a run: the traceback stays
    eng._run_started_at = time.time()
    assert launch._print_quest_failure(eng) is False  # nothing recorded
    ck.write(eng.fi_dir, ck.classify(_http(401), provider="openai"))
    assert launch._print_quest_failure(eng) is True
    err = capsys.readouterr().err
    assert "FI could not sign in to openai" in err and "python launch.py --resume q1" in err
    eng._run_started_at = time.time() + 120  # a record from before this run (a resume that failed earlier on)
    assert launch._print_quest_failure(eng) is False


# ---- the engine ------------------------------------------------------------------------------------------------------

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
async def test_a_step_that_fails_on_a_passing_problem_is_run_again(tmp_path, monkeypatch, _no_waits) -> None:
    """The design step times out once (past the provider's own retries): the engine waits, says so on screen, runs
    the step again by itself, and the quest goes on (here to the next stop, a filtered answer) with no failure."""
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
    log = (eng.fi_dir / "run.log").read_text(encoding="utf-8")
    assert "trying it again" in log and "FI tries the step again in 1 minute" in log


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
    assert body.index("**What happened:**") < body.index("**What to do:**") < body.index("## Details")
    assert body.index("## Details") < body.index("ReadTimeout")
    assert "tried the step again 2 time(s)" in body and "--resume" in body
    assert "Traceback" in (eng.fi_dir / "run.log").read_text(encoding="utf-8")  # the full trace is kept


@pytest.mark.asyncio
async def test_a_long_step_is_not_run_again(tmp_path, monkeypatch, _no_waits) -> None:
    """A step that had been running longer than RETRY_ONLY_UNDER_S is not repeated by itself (review finding 4)."""
    from core.engine import Engine
    from tests.test_engine_smoke import _classify, _fake_response_for

    monkeypatch.setattr(ck, "RETRY_ONLY_UNDER_S", -1.0)
    calls = {"design": 0}

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        if _classify(prompt) == "Experiment Design":
            calls["design"] += 1
            raise httpx.ReadTimeout("read timed out")
        return _fake_response_for(prompt)

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)
    eng = Engine(_quest_config(tmp_path, "transient-long"))
    with pytest.raises(httpx.ReadTimeout):
        await eng.run()
    assert calls["design"] == 1
    assert ck.read(eng.fi_dir)["retries"] == 0


@pytest.mark.asyncio
async def test_an_fi_bug_is_not_run_again(tmp_path, monkeypatch, _no_waits) -> None:
    from core.engine import Engine
    from tests.test_engine_smoke import _classify, _fake_response_for

    calls = {"design": 0}
    planted = _fi_code("def f():\n    return {}['no such key']\n")

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        if _classify(prompt) == "Experiment Design":
            calls["design"] += 1
            return planted()  # a programming error raised in FI's own code
        return _fake_response_for(prompt)

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)
    eng = Engine(_quest_config(tmp_path, "fi-bug"))
    with pytest.raises(KeyError):
        await eng.run()
    assert calls["design"] == 1
    record = ck.read(eng.fi_dir)
    assert record and record["kind"] == "fi"
    body = (eng.quest_root / "quest_failed.md").read_text(encoding="utf-8")
    assert "This is a problem in FI itself" in body and "Nothing for you to fix" in body
