"""A quest that stops on an error says what happened in words a scientist can act on, and asks nothing of them but an
action on their own machine or account.

``core/crash_kind.py`` sorts a failure: a passing problem (run again by the engine itself before it stops), a setup
problem (one action named), a problem in FI's own code (said plainly to be FI's, nothing for the person to fix), or an
unexpected one (nothing to debug either). ``quest_failed.md``, ``.fi/failure.json``, the to-do card, the web quest
page and the CLI all lead with that; the exception and the log tail are details. No real model is called.
"""

from __future__ import annotations

import errno
import logging
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


def _http(status: int, body: dict | None = None, headers: dict | None = None, *,
          model_call: bool = True) -> httpx.HTTPStatusError:
    """An HTTP error; a model call carries the note the provider attaches to every call it makes."""
    req = httpx.Request("POST", "https://api.example.com/v1/chat/completions")
    resp = httpx.Response(status, request=req, json=body or {"error": {"message": "x"}}, headers=headers)
    exc = httpx.HTTPStatusError(f"HTTP {status}", request=req, response=resp)
    if model_call:
        exc.add_note("[FI] provider=openai, transport=http, model=gemma9, node=design")
    return exc


def _noted(exc: BaseException) -> BaseException:
    """A provider error as the provider raises it: with its ``[FI] provider=`` note."""
    exc.add_note("[FI] provider=claude_cli, transport=cli, model=m, node=design")
    return exc


def _spawn_missing() -> FileNotFoundError:
    """The error a real start of a program that is not installed raises on this platform (Windows: WinError 2 with
    no file name; POSIX: the program's name)."""
    import subprocess

    try:
        subprocess.run(["fi_no_such_program_xyz"], check=False)
    except FileNotFoundError as e:
        return e
    raise AssertionError("the program unexpectedly exists")


def _local_refused(model_call: bool) -> httpx.ConnectError:
    """What a real httpx call to a closed port on this machine raises: the message names no host."""
    exc = httpx.ConnectError("All connection attempts failed",
                             request=httpx.Request("POST", "http://127.0.0.1:11434/v1/chat/completions"))
    if model_call:
        exc.add_note("[FI] provider=ollama, transport=http, model=gemma4, node=design")
    return exc


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


def test_a_passing_problem_is_said_without_the_raw_error() -> None:
    f = ck.classify(httpx.ReadTimeout("All connection attempts failed"), node="design")
    assert "All connection" not in f.say and "All connection" in f.detail


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
    (_noted(RuntimeError("[FI] HTTP 401: openai did not accept the API key FI sent.")), "sign in"),
    (_noted(RuntimeError("You've hit your weekly limit - resets Mon 9am")), "allowance"),
    (_http(429, {"error": {"type": "insufficient_quota", "message": "You exceeded your current quota"}}), "allowance"),
    (_http(404, {"error": {"message": "model 'gemma9' not found"}}), "does not know the model"),
    (ModuleNotFoundError("No module named 'scipy'", name="scipy"), "not installed (scipy)"),
    (_spawn_missing(), "not found on this machine"),
    (OSError(errno.ENOSPC, "No space left on device"), "disk is full"),
    (httpx.UnsupportedProtocol("Request URL is missing an 'http://' or 'https://' protocol."), "not a valid web address"),
    (_local_refused(model_call=True), "The model service on this machine (openai) is not running"),
    (_local_refused(model_call=False), "A service on this machine that FI uses (127.0.0.1:11434) is not running"),
    (RuntimeError("bridge connection dropped"), "VS Code's connection"),
    (_winerror(PermissionError, 32, "being used by another process", "C:/q/data/results.csv"), "(results.csv)"),
])
def test_a_setup_problem_names_one_action(exc: BaseException, words: str) -> None:
    f = ck.classify(exc, provider="openai", model="gemma9")
    assert f.kind == "setup", f
    assert words in f.say
    assert ("continue the quest" in f.do.lower()) or ("@fi /resume" in f.do), f.do


def test_a_missing_input_file_is_the_person_s_one_action() -> None:
    """``execution.inputs`` naming a file that is not there: not a model problem and not a program to install, and not
    "nothing to fix" either: the person put it in the settings and puts it right (second review, P1)."""
    from core import example_inputs

    try:
        example_inputs.stage_inputs(["no/such/input.csv"], Path("."), logging.getLogger("t"))
    except FileNotFoundError as e:
        f = ck.classify(e, model="gpt-5")
    assert f.kind == "setup" and "no/such/input.csv" in f.say and "does not know the model" not in f.say
    assert "Put the file in place" in f.do


def test_a_data_file_fi_reads_is_not_a_missing_program() -> None:
    try:
        open("LICENSE_no_such_file")  # noqa: SIM115 -- the error of a missing data file
    except FileNotFoundError as e:
        assert "not found on this machine" not in ck.classify(e).say


def test_an_http_error_that_is_not_a_model_call_is_not_a_model_or_sign_in_problem() -> None:
    """A literature service's 404 or the knowledge store's 401 is not "the model" or "sign in to the provider"."""
    assert "does not know the model" not in ck.classify(_http(404, model_call=False), provider="openai").say
    assert "could not sign in" not in ck.classify(_http(401, model_call=False), provider="openai").say


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


# ---- settings the person chose, FI's own helper, and the retry timing (third review) ----------------------------------

@pytest.mark.parametrize("message, words", [
    ("Docker daemon not reachable. Install Docker Desktop (Windows/macOS) or run `dockerd` (Linux), then retry.",
     "Docker, which is not installed or not running"),
    ("execution.sandbox=docker requires `pip install docker`", "the Python package FI uses to reach Docker"),
    ("[preflight] paper_pdf requested with output.require_pdf=True but pandoc not found on this host", "PDF"),
    ("venv creation failed via py (python_version=3.9): rc=1", "(3.9)"),
    ("engine.skills_required names skill(s) that cannot be used: foo (no skill by that name was found)",
     "engine.skills_required"),
])
def test_a_refused_setting_is_the_person_s_one_action(message: str, words: str) -> None:
    f = ck.classify(_raised(RuntimeError(message)))
    assert f.kind == "setup" and words in f.say, f
    assert "Nothing for you to fix" not in f.do and "continue the quest" in f.do


def test_a_missing_docker_package_is_named_as_such_with_its_real_cause_chain() -> None:
    """The real error is ``raise RuntimeError(...) from ModuleNotFoundError(name="docker")``: the card names the
    package FI uses to reach Docker, not the generic missing package and not Docker being down."""
    def raise_it():
        try:
            raise ModuleNotFoundError("No module named 'docker'", name="docker")
        except ModuleNotFoundError as inner:
            raise RuntimeError("execution.sandbox=docker requires `pip install docker`") from inner

    try:
        raise_it()
    except RuntimeError as exc:
        f = ck.classify(exc)
    assert f.kind == "setup" and "the Python package FI uses to reach Docker" in f.say, f
    assert "not running" not in f.say and "A Python package FI needs" not in f.say


@pytest.mark.parametrize("exc, kind", [
    (FileNotFoundError(errno.ENOENT, "No such file", "C:/q/.fi/engine.log"), "unknown"),
    (RuntimeError("knowledge.asearch returned no list"), "unknown"),
    (RuntimeError("the writer answered with something that is not a paper (output.md ...)"), "unknown"),
    (ConnectionResetError("connection reset while writing output.bin"), "transient"),
])
def test_a_message_that_only_mentions_a_settings_like_name_is_not_a_settings_refusal(exc, kind) -> None:
    """Only FI's own refusals of a setting, by their opening words, count (fourth check)."""
    f = ck.classify(_raised(exc))
    assert f.kind == kind and "Change that setting" not in f.do, f


def test_a_file_held_or_a_full_disk_keeps_its_own_words_even_with_a_settings_like_file_name() -> None:
    held = _winerror(PermissionError, 32, "being used by another process", "C:/q/execution.log")
    assert "Close that file" in ck.classify(_raised(held)).do
    full = OSError(errno.ENOSPC, "No space left on device", "C:/q/output.pdf")
    assert "disk is full" in ck.classify(_raised(full)).say


def test_a_bug_in_the_inputs_module_is_not_blamed_on_the_person() -> None:
    planted = compile("def f():\n    return None.size\n", str(ck._CORE / "example_inputs.py"), "exec")
    ns: dict = {}
    exec(planted, ns)  # noqa: S102
    try:
        ns["f"]()
    except AttributeError as e:
        f = ck.classify(e)
    assert f.kind == "fi" and "file named in the quest's settings" not in f.say


def test_fi_s_own_connection_helper_is_not_the_person_s_to_start() -> None:
    exc = httpx.ConnectError("All connection attempts failed",
                             request=httpx.Request("POST", "http://127.0.0.1:41234/v1/chat/completions"))
    exc.add_note("[FI] provider=claude_code, transport=http, model=m, node=design")
    f = ck.classify(exc, provider="claude_code")
    assert "Start it" not in f.do and f.kind != "setup"


def test_an_ipv6_address_is_written_with_brackets() -> None:
    exc = httpx.ConnectError("All connection attempts failed",
                             request=httpx.Request("GET", "http://[::1]:8000/health"))
    assert "([::1]:8000)" in ck.classify(exc).say


class _Snap:
    def __init__(self, step: str, checkpoint: str, created_at: str | None) -> None:
        self.next = (step,)
        self.config = {"configurable": {"checkpoint_id": checkpoint}}
        self.created_at = created_at


class _Graph:
    def __init__(self, snap: _Snap) -> None:
        self.snap = snap

    async def aget_state(self, _config):  # noqa: ANN001
        return self.snap


def _bare_engine():
    import logging
    from types import SimpleNamespace

    from core.engine import Engine

    eng = Engine.__new__(Engine)
    eng._step_retries = {}
    eng._log = logging.getLogger("test-crash-kind")
    eng._last_progress = ""
    eng._progress = lambda text: None  # type: ignore[method-assign]
    return eng, SimpleNamespace


@pytest.mark.asyncio
async def test_a_resumed_quest_s_first_step_can_be_run_again(monkeypatch) -> None:
    """The last checkpoint is an earlier run's (hours old): the step's working time counts from its start in this run
    (second review, finding 4)."""
    monkeypatch.setattr(ck, "RETRY_WAITS_S", (0.0, 0.0))
    eng, _ = _bare_engine()
    eng._invoke_started_at = time.time() - 5
    graph = _Graph(_Snap("design", "c1", "2026-01-01T00:00:00+00:00"))
    assert await eng._try_step_again(graph, {}, httpx.ReadTimeout("t")) is True


@pytest.mark.asyncio
async def test_a_step_that_worked_long_in_this_run_is_not_run_again(monkeypatch) -> None:
    monkeypatch.setattr(ck, "RETRY_WAITS_S", (0.0, 0.0))
    eng, _ = _bare_engine()
    eng._invoke_started_at = time.time() - ck.RETRY_ONLY_UNDER_S - 60
    graph = _Graph(_Snap("execute", "c1", None))
    assert await eng._try_step_again(graph, {}, httpx.ReadTimeout("t")) is False


@pytest.mark.asyncio
async def test_retries_are_counted_per_entry_into_a_step(monkeypatch) -> None:
    monkeypatch.setattr(ck, "RETRY_WAITS_S", (0.0, 0.0))
    eng, _ = _bare_engine()
    eng._invoke_started_at = time.time()
    first = _Graph(_Snap("design", "c1", None))
    assert await eng._try_step_again(first, {}, httpx.ReadTimeout("t")) is True
    assert await eng._try_step_again(first, {}, httpx.ReadTimeout("t")) is True
    assert await eng._try_step_again(first, {}, httpx.ReadTimeout("t")) is False  # this entry's two are used
    later = _Graph(_Snap("design", "c2", None))  # the quest came back to the design from another checkpoint
    assert await eng._try_step_again(later, {}, KeyError("x")) is False  # not a passing problem: no retry...
    assert eng._step_retries["design"] == ("c2", 0)  # ...and the card counts this entry's retries, none


def test_the_fleet_summary_does_not_repeat_a_failure_already_said(tmp_path, capsys) -> None:
    import launch

    source = Path(launch.__file__).read_text(encoding="utf-8")
    assert 'if not getattr(r, "_fi_failure_shown", False):' in source


# ---- after the combined review ------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_retry_is_sorted_with_the_same_provider_and_model_as_the_card(monkeypatch) -> None:
    from types import SimpleNamespace

    monkeypatch.setattr(ck, "RETRY_WAITS_S", (0.0, 0.0))
    seen: list[dict] = []
    real = ck.classify

    def spy(exc, **kw):  # noqa: ANN001, ANN003
        seen.append(kw)
        return real(exc, **kw)

    monkeypatch.setattr(ck, "classify", spy)
    eng, _ = _bare_engine()
    eng.config = SimpleNamespace(provider=SimpleNamespace(name="claude_code", model="sonnet"),
                                 execution=SimpleNamespace(background_jobs=False))
    eng._invoke_started_at = time.time()
    await eng._try_step_again(_Graph(_Snap("design", "c1", None)), {}, httpx.ReadTimeout("t"))
    assert seen and seen[0]["provider"] == "claude_code" and seen[0]["model"] == "sonnet"


@pytest.mark.asyncio
async def test_a_step_that_hands_work_to_background_jobs_is_never_run_again_by_itself(monkeypatch) -> None:
    from types import SimpleNamespace

    monkeypatch.setattr(ck, "RETRY_WAITS_S", (0.0, 0.0))
    eng, _ = _bare_engine()
    eng.config = SimpleNamespace(provider=SimpleNamespace(name="openai", model="m"),
                                 execution=SimpleNamespace(background_jobs=True))
    eng._invoke_started_at = time.time()
    assert await eng._try_step_again(_Graph(_Snap("execute", "c1", None)), {}, httpx.ReadTimeout("t")) is False
    assert await eng._try_step_again(_Graph(_Snap("design", "c2", None)), {}, httpx.ReadTimeout("t")) is True


def test_words_alone_never_make_an_error_passing_or_a_missing_model() -> None:
    # FI's own message with "timeout" in it, not from a model call: not a passing problem.
    assert ck.classify(_raised(RuntimeError("the step's timeout setting is too small"))).kind != "transient"
    # A model call whose text says a FILE does not exist, with no HTTP answer and no model named: not "unknown model".
    err = RuntimeError("claude CLI failed: the file C:/x/prompt.txt does not exist")
    err.add_note("[FI] provider=claude_cli, node=design, model=sonnet")
    f = ck.classify(_raised(err), provider="claude_cli", model="sonnet")
    assert "does not know the model" not in f.say
    # The same words with the model named are still read as a missing model.
    err2 = RuntimeError("model sonnet-9 does not exist")
    err2.add_note("[FI] provider=claude_cli, node=design, model=sonnet-9")
    assert "does not know the model" in ck.classify(_raised(err2), provider="claude_cli", model="sonnet-9").say
