"""Every connection that can hand a model's reasoning back is asked for it (when ``output.save_thinking`` is on) and what
comes back reaches ``.fi/thinking.jsonl``; with the setting off nothing is asked for or kept.

No real model is called: ``httpx.MockTransport`` for the HTTP providers, a fake binary for the CLIs and a mock bridge
server for VS Code.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest

from core import attempt_records as ar
from core import thinking_capture as tc
from core.config import ProviderConfig
from core.engine import Engine
from core.provider import _CLI_SPECS, LLMClient, ResolvedEndpoint, _CliSpec, _run_cli, resolve_endpoint
from tests.test_engine_smoke import smoke_config  # noqa: F401 -- the fixture
from tests.test_vscode_bridge import _MockBridgeServer

MESSAGES = [{"role": "user", "content": "hi"}]


def _ndjson(chunks: list[dict]) -> bytes:
    return "".join(json.dumps(c) + "\n" for c in chunks).encode()


def _ollama_handler(seen: list[httpx.Request], *, thinking: str = "weighing 91 = 7 x 13", status: int = 200,
                    error: str = ""):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/api/chat":
            if status != 200:
                return httpx.Response(status, json={"error": error})
            return httpx.Response(200, content=_ndjson([
                {"model": "gemma4:31b-cloud", "message": {"role": "assistant", "content": "", "thinking": thinking}},
                {"model": "gemma4:31b-cloud", "message": {"role": "assistant", "content": "No."}},
                {"model": "gemma4:31b-cloud", "message": {"role": "assistant", "content": ""}, "done": True,
                 "done_reason": "stop", "prompt_eval_count": 7, "eval_count": 11}]))
        return httpx.Response(200, json={
            "model": "gemma4:31b-cloud",
            "choices": [{"message": {"role": "assistant", "content": "No."}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})

    return handler


def _ollama_call(handler, *, want: bool = True, with_holder: bool = True, provider: dict | None = None,
                 calls: int = 1) -> tuple[list[str], str]:
    async def go() -> tuple[list[str], str]:
        holder, token = tc.open_holder(want=want) if with_holder else ({"text": ""}, None)
        http = httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=30.0)
        try:
            client = LLMClient(resolve_endpoint(ProviderConfig(name="ollama", model="gemma4:31b-cloud",
                                                               **(provider or {}))), http=http)
            answers = [await client.chat(MESSAGES, max_tokens=300) for _ in range(calls)]
            return answers, holder["text"]
        finally:
            await http.aclose()
            if token is not None:
                tc.close_holder(token)

    return asyncio.run(go())


def test_ollama_is_asked_for_its_reasoning_on_its_own_api_and_it_comes_back() -> None:
    seen: list[httpx.Request] = []
    answers, thinking = _ollama_call(_ollama_handler(seen))
    assert answers == ["No."] and thinking == "weighing 91 = 7 x 13"
    (req,) = seen
    assert req.url.path == "/api/chat"
    body = json.loads(req.content)
    assert body["think"] is True and body["stream"] is True and body["options"]["num_predict"] == 300
    assert body["model"] == "gemma4:31b-cloud" and body["messages"] == MESSAGES


def test_ollama_with_saving_off_is_not_asked_and_uses_the_compatible_call() -> None:
    seen: list[httpx.Request] = []
    answers, thinking = _ollama_call(_ollama_handler(seen), want=False)
    assert answers == ["No."] and thinking == ""
    assert [r.url.path for r in seen] == ["/v1/chat/completions"]
    assert "think" not in json.loads(seen[0].content)


def test_ollama_called_outside_a_step_is_not_asked() -> None:
    seen: list[httpx.Request] = []
    _ollama_call(_ollama_handler(seen), with_holder=False)
    assert [r.url.path for r in seen] == ["/v1/chat/completions"]


def test_ollama_keeps_the_users_reasoning_level() -> None:
    seen: list[httpx.Request] = []
    _ollama_call(_ollama_handler(seen), provider={"reasoning_effort": "high"})
    assert json.loads(seen[0].content)["think"] == "high"


def test_a_model_that_cannot_think_falls_back_once_and_is_not_asked_again() -> None:
    seen: list[httpx.Request] = []
    handler = _ollama_handler(seen, status=400, error='"gemma4:31b-cloud" does not support thinking')
    answers, thinking = _ollama_call(handler, calls=2)
    assert answers == ["No.", "No."] and thinking == ""
    assert [r.url.path for r in seen] == ["/api/chat", "/v1/chat/completions", "/v1/chat/completions"]


def test_a_request_the_native_form_would_change_stays_on_the_compatible_call() -> None:
    seen: list[httpx.Request] = []
    _ollama_call(_ollama_handler(seen), provider={"extra_body": {"seed": 3}})
    assert [r.url.path for r in seen] == ["/v1/chat/completions"]


def test_a_reply_cut_at_the_limit_still_reads_as_cut_off() -> None:
    from core.provider import _ollama_as_openai

    out = _ollama_as_openai({"done_reason": "length", "message": {"content": "abc", "thinking": "t"}})
    assert out["choices"][0]["finish_reason"] == "length"
    assert out["choices"][0]["message"]["reasoning_content"] == "t"


def test_an_openai_compatible_server_reasoning_text_is_read_too() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok", "reasoning_text": "because"},
                                                      "finish_reason": "stop"}]})

    async def go() -> str:
        holder, token = tc.open_holder()
        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            await LLMClient(resolve_endpoint(ProviderConfig(name="vllm")), http=http).chat(MESSAGES)
            return holder["text"]
        finally:
            await http.aclose()
            tc.close_holder(token)

    assert asyncio.run(go()) == "because"


# --- CLIs -------------------------------------------------------------------------------------------------------

def _fake_cli(tmp_path: Path, spec_kind: str, thinking_args: tuple[str, ...]) -> tuple[_CliSpec, Path]:
    """A binary that records the arguments it got and prints what the real one prints (stream-json for claude, JSONL
    plus the last-message file for codex)."""
    argv_file = tmp_path / "argv.json"
    script = tmp_path / "fake_cli.py"
    if spec_kind == "claude":
        body = (
            "for ev in [{'type':'stream_event','event':{'type':'content_block_delta','delta':{'type':'thinking_delta',"
            "'thinking':'summary of the thinking'}}},{'type':'stream_event','event':{'type':'content_block_delta',"
            "'delta':{'type':'text_delta','text':'Hi'}}},{'type':'result','result':'Hi'}]:\n"
            "    print(json.dumps(ev), flush=True)\n")
        spec = _CliSpec(argv=(sys.executable, str(script)), pass_prompt_via="stdin", output_via="stream_json",
                        model_flag=None, thinking_args=thinking_args)
    else:
        body = (
            "out = sys.argv[sys.argv.index('--output-last-message') + 1]\n"
            "open(out, 'w').write('Hi')\n"
            "print(json.dumps({'type':'item.completed','item':{'id':'i0','type':'reasoning','text':'**Checking**'}}))\n"
            "print(json.dumps({'type':'item.completed','item':{'id':'i1','type':'agent_message','text':'Hi'}}))\n")
        from core.provider import _extract_codex_reasoning

        spec = _CliSpec(argv=(sys.executable, str(script)), pass_prompt_via="stdin", output_via="last_message_file",
                        model_flag=None, reasoning_extractor=_extract_codex_reasoning, thinking_args=thinking_args,
                        usage_extractor=lambda raw: None)  # codex reads its stdout (the real spec has an extractor)
    script.write_text("import json, sys\nopen(" + repr(str(argv_file)) + ", 'w').write(json.dumps(sys.argv[1:]))\n"
                      + body, encoding="utf-8")
    return spec, argv_file


@pytest.mark.parametrize("kind", ["claude", "codex"])
@pytest.mark.parametrize("want", [True, False])
@pytest.mark.asyncio
async def test_a_cli_is_asked_for_its_reasoning_only_when_it_will_be_kept(tmp_path: Path, kind: str, want: bool) -> None:
    ask = ("--settings", '{"showThinkingSummaries":true}') if kind == "claude" else ("-c", 'model_reasoning_summary="detailed"')
    spec, argv_file = _fake_cli(tmp_path, kind, ask)
    holder, token = tc.open_holder(want=want)
    try:
        assert await _run_cli(spec, "p", timeout_s=20.0, inactivity_timeout_s=10.0) == "Hi"
    finally:
        tc.close_holder(token)
    argv = json.loads(argv_file.read_text(encoding="utf-8"))
    assert (all(a in argv for a in ask)) is want
    # What a CLI sent is kept either way; asking is what ``want`` decides.
    assert holder["text"].strip() == ("summary of the thinking" if kind == "claude" else "**Checking**")


@pytest.mark.asyncio
async def test_a_cli_called_outside_a_step_is_not_asked(tmp_path: Path) -> None:
    ask = ("-c", 'model_reasoning_summary="detailed"')
    spec, argv_file = _fake_cli(tmp_path, "codex", ask)
    assert await _run_cli(spec, "p", timeout_s=20.0, inactivity_timeout_s=10.0) == "Hi"
    assert "model_reasoning_summary" not in argv_file.read_text(encoding="utf-8")


def test_the_real_claude_and_codex_commands_ask_for_their_reasoning_text() -> None:
    assert _CLI_SPECS["claude_cli"].thinking_args == ("--settings", '{"showThinkingSummaries":true}')
    assert _CLI_SPECS["codex_cli"].thinking_args == ("-c", 'model_reasoning_summary="detailed"')
    # The CLIs that cannot return reasoning text are not given an option that would do nothing.
    for name in ("gemini_cli", "antigravity_cli", "copilot_cli"):
        assert _CLI_SPECS[name].thinking_args == () and _CLI_SPECS[name].reasoning_extractor is None


def test_the_real_claude_command_line_has_the_setting_as_one_json_argument(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    spec = _CLI_SPECS["claude_cli"]
    fake, argv_file = _fake_cli(tmp_path, "claude", spec.thinking_args)

    async def go() -> None:
        holder, token = tc.open_holder()
        try:
            await _run_cli(fake, "p", timeout_s=20.0, inactivity_timeout_s=10.0)
        finally:
            tc.close_holder(token)

    asyncio.run(go())
    argv = json.loads(argv_file.read_text(encoding="utf-8"))
    assert json.loads(argv[argv.index("--settings") + 1]) == {"showThinkingSummaries": True}


# --- the file ---------------------------------------------------------------------------------------------------

def _engine_with(cfg, client) -> Engine:  # noqa: ANN001
    engine = Engine(cfg)
    engine.fi_dir.mkdir(parents=True, exist_ok=True)
    engine._client = client  # type: ignore[assignment]
    return engine


@pytest.mark.parametrize("save", [True, False])
@pytest.mark.asyncio
async def test_ollama_reasoning_reaches_the_file_only_when_saving_is_on(smoke_config, save: bool) -> None:  # noqa: ANN001, F811
    smoke_config.output.save_thinking = save
    seen: list[httpx.Request] = []
    http = httpx.AsyncClient(transport=httpx.MockTransport(_ollama_handler(seen)), timeout=30.0)
    client = LLMClient(resolve_endpoint(ProviderConfig(name="ollama", model="gemma4:31b-cloud")), http=http)
    engine = _engine_with(smoke_config, client)
    try:
        assert await engine._chat("prompt", node="design") == "No."
    finally:
        await http.aclose()
    if save:
        (line,) = ar.read(engine.fi_dir, tc.THINKING_FILE)
        assert line["thinking"] == "weighing 91 = 7 x 13" and line["node"] == "design"
        assert [r.url.path for r in seen] == ["/api/chat"]
    else:
        assert not (engine.fi_dir / tc.THINKING_FILE).exists()
        assert [r.url.path for r in seen] == ["/v1/chat/completions"]


@pytest.mark.parametrize("kind", ["claude", "codex"])
@pytest.mark.asyncio
async def test_a_cli_s_reasoning_reaches_the_file(smoke_config, tmp_path: Path, kind: str) -> None:  # noqa: ANN001, F811
    ask = ("--settings", "{}") if kind == "claude" else ("-c", "x=1")
    spec, _ = _fake_cli(tmp_path, kind, ask)
    client = LLMClient(ResolvedEndpoint(base_url="", model="m", api_key="not-needed", transport="cli", cli_spec=spec,
                                        provider_name=f"{kind}_cli"))
    engine = _engine_with(smoke_config, client)
    try:
        assert await engine._chat("prompt", node="design") == "Hi"
    finally:
        await client.aclose()
    (line,) = ar.read(engine.fi_dir, tc.THINKING_FILE)
    assert line["thinking"].strip() == ("summary of the thinking" if kind == "claude" else "**Checking**")


@pytest.mark.parametrize("save", [True, False])
@pytest.mark.asyncio
async def test_the_vscode_bridge_is_asked_only_when_saving_is_on(smoke_config, save: bool) -> None:  # noqa: ANN001, F811
    smoke_config.output.save_thinking = save
    server = _MockBridgeServer()
    asked: list[object] = []

    def handler(msg: dict, w) -> list[dict]:  # noqa: ANN001
        if msg["type"] != "lm_request":
            return []
        asked.append(msg.get("ask_thinking"))
        return [{"type": "lm_done", "id": msg["id"], "content": "Answer", "thinking": "reasoned over the bridge"}]

    port = await server.start(handler)
    client = LLMClient(ResolvedEndpoint(base_url="", model="(VSCode chat default)", api_key="not-needed",
                                        transport="vscode_bridge", vscode_bridge_port=port))
    engine = _engine_with(smoke_config, client)
    try:
        assert await engine._chat("prompt", node="design") == "Answer"
    finally:
        await client.aclose()
        await server.stop()
    assert asked == [save]
    assert (engine.fi_dir / tc.THINKING_FILE).exists() is save


@pytest.mark.asyncio
async def test_a_connection_that_cannot_return_reasoning_says_so_once(smoke_config) -> None:  # noqa: ANN001, F811
    class _Gemini:
        async def chat(self, messages, **kw):  # noqa: ANN001
            from core.provider import LAST_CALL

            LAST_CALL.set({"provider": "gemini_cli", "model": "m", "reported": False})
            return "answer"

    engine = _engine_with(smoke_config, _Gemini())
    for node in ("design", "analyze", "write"):
        await engine._chat("prompt", node=node)
    for h in engine._log.handlers:
        h.flush()
    text = (engine.fi_dir / "run.log").read_text(encoding="utf-8")
    assert text.count("gemini_cli cannot return a model's reasoning") == 1
    assert "returned no reasoning for this step" not in text
    assert not (engine.fi_dir / tc.THINKING_FILE).exists()


# ---- the native call is streamed: assembled, classified and timed as a stream ----

def _stream_call(chunks, *, timeout: float = 30.0, inactivity: float | None = None, gaps: float = 0.0,
                 raw: bytes | None = None, final_hang: float = 0.0):
    """``_post_ollama_streamed`` against an in-memory stream that sends ``chunks`` one line at a time, ``gaps`` seconds
    apart (and, with ``final_hang``, stays silent that long before ending)."""
    from core.provider import _post_ollama_streamed

    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            if raw is not None:
                yield raw
            for c in chunks:
                await asyncio.sleep(gaps)
                yield (json.dumps(c) + "\n").encode()
            if final_hang:
                await asyncio.sleep(final_hang)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=Body())

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            return await _post_ollama_streamed(http, "http://x/api/chat", {"stream": True}, {}, timeout, inactivity)

    return asyncio.run(go())


def _done(reason: str = "stop") -> dict:
    return {"model": "m", "message": {"content": ""}, "done": True, "done_reason": reason,
            "prompt_eval_count": 5, "eval_count": 9}


def test_a_streamed_native_reply_is_assembled_from_its_chunks() -> None:
    from core.provider import _ollama_as_openai

    out = _ollama_as_openai(_stream_call([
        {"message": {"thinking": "step one, "}}, {"message": {"thinking": "step two"}},
        {"message": {"content": "He"}}, {"message": {"content": "llo"}}, _done()]))
    choice = out["choices"][0]
    assert choice["message"]["content"] == "Hello" and choice["message"]["reasoning_content"] == "step one, step two"
    assert choice["finish_reason"] == "stop" and out["model"] == "m"
    assert out["usage"] == {"prompt_tokens": 5, "completion_tokens": 9, "total_tokens": 14}


def test_a_streamed_reply_cut_at_the_limit_reads_as_cut_off() -> None:
    from core.provider import _ollama_as_openai

    out = _ollama_as_openai(_stream_call([{"message": {"content": "abc"}}, _done("length")]))
    assert out["choices"][0]["finish_reason"] == "length" and out["choices"][0]["message"]["content"] == "abc"


def test_an_error_inside_the_stream_is_classified_like_any_stream_error() -> None:
    with pytest.raises(httpx.RemoteProtocolError):
        _stream_call([{"message": {"content": "a"}}, {"error": "server is overloaded, try again"}])
    with pytest.raises(httpx.HTTPStatusError):
        _stream_call([{"error": {"type": "invalid_request_error", "message": "bad"}}])


def test_a_stream_that_ends_without_done_is_an_error_that_is_tried_again() -> None:
    from core.provider import _retry_http_error

    with pytest.raises(httpx.RemoteProtocolError) as e:
        _stream_call([{"message": {"content": "a"}}])
    assert _retry_http_error(e.value)


def test_a_line_that_is_not_json_is_a_protocol_error() -> None:
    with pytest.raises(httpx.RemoteProtocolError):
        _stream_call([], raw=b"not json\n")


def test_thinking_that_keeps_arriving_is_not_cut_off_by_the_inactivity_timeout() -> None:
    # 8 chunks 0.15 s apart = 1.2 s in all, each gap well under the 0.5 s limit.
    chunks = [{"message": {"thinking": "t"}}] * 7 + [_done()]
    out = _stream_call(chunks, timeout=10.0, inactivity=0.5, gaps=0.15)
    assert out["message"]["thinking"] == "t" * 7


def test_a_silent_stream_is_cut_off_by_the_inactivity_timeout_and_retried() -> None:
    from core.provider import _retry_http_error

    with pytest.raises(httpx.ReadTimeout) as e:
        _stream_call([{"message": {"thinking": "t"}}], timeout=10.0, inactivity=0.3, final_hang=3.0)
    assert _retry_http_error(e.value)


def test_the_whole_call_budget_still_bounds_a_stream_that_never_goes_silent() -> None:
    chunks = [{"message": {"thinking": "t"}}] * 50 + [_done()]
    with pytest.raises(httpx.ReadTimeout):
        _stream_call(chunks, timeout=0.5, inactivity=0.4, gaps=0.1)


def test_a_refusal_of_thinking_sent_inside_the_stream_falls_back_and_is_remembered() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/api/chat":
            return httpx.Response(200, content=_ndjson([
                {"error": {"type": "invalid_request_error", "message": '"gemma4:31b-cloud" does not support thinking'}}]))
        return httpx.Response(200, json={"model": "m", "choices": [{"message": {"content": "No."}, "finish_reason": "stop"}]})

    answers, thinking = _ollama_call(handler, calls=2)
    assert answers == ["No.", "No."] and thinking == ""
    assert [r.url.path for r in seen] == ["/api/chat", "/v1/chat/completions", "/v1/chat/completions"]
