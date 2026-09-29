"""What a model says about its own reasoning is kept in ``.fi/thinking.jsonl`` when a connection hands it back.

No real model is called: ``httpx.MockTransport``, a fake stream-json binary, a mock bridge server and a stand-in client.
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
from core.provider import LLMClient, _CliSpec, _run_cli, resolve_endpoint
from core.vscode_bridge import VSCodeBridgeClient
from tests.test_engine_smoke import smoke_config  # noqa: F401 -- the fixture
from tests.test_provider_kimi_streaming import KIMI, _chunk, _sse
from tests.test_vscode_bridge import _MockBridgeServer

MESSAGES = [{"role": "user", "content": "hi"}]


def _http_call(endpoint, handler) -> tuple[str, str]:
    async def go() -> tuple[str, str]:
        holder, token = tc.open_holder()
        http = httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=30.0)
        try:
            text = await LLMClient(endpoint, http=http).chat(MESSAGES)
            return text, holder["text"]
        finally:
            await http.aclose()
            tc.close_holder(token)

    return asyncio.run(go())


@pytest.mark.parametrize("key", ["reasoning_content", "reasoning"])
def test_a_plain_reply_with_reasoning_is_kept(key: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "model": "deepseek-reasoner",
            "choices": [{"message": {"role": "assistant", "content": "42", key: "first I add, then I check"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 9, "total_tokens": 12},
        })

    text, thinking = _http_call(resolve_endpoint(ProviderConfig(name="openai", model="deepseek-reasoner")), handler)
    assert text == "42" and thinking == "first I add, then I check"


def test_a_reply_without_reasoning_keeps_nothing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]})

    text, thinking = _http_call(resolve_endpoint(ProviderConfig(name="openai")), handler)
    assert text == "ok" and thinking == ""


def test_a_streamed_reply_puts_its_reasoning_back_together() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        think = lambda t: {"choices": [{"delta": {"reasoning_content": t}, "finish_reason": None}]}  # noqa: E731
        return httpx.Response(200, content=_sse(
            think("let me "), think("think"), _chunk("Hello"), _chunk(finish="stop"),
            {"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}))

    text, thinking = _http_call(resolve_endpoint(ProviderConfig(**KIMI)), handler)
    assert text == "Hello" and thinking == "let me think"


def test_without_a_holder_nothing_happens() -> None:
    tc.note_thinking("no one is collecting")
    tc.add_thinking("no one is collecting")
    holder, token = tc.open_holder()
    tc.close_holder(token)
    tc.note_thinking("still no one")
    assert holder["text"] == ""


def test_a_call_inside_another_leaves_the_outer_holder_as_it_was() -> None:
    outer, outer_token = tc.open_holder()
    tc.note_thinking("outer")
    inner, inner_token = tc.open_holder()
    tc.note_thinking("inner")
    tc.close_holder(inner_token)
    tc.add_thinking(" more outer")
    tc.close_holder(outer_token)
    assert inner["text"] == "inner" and outer["text"] == "outer more outer"


def _stream_binary(tmp_path: Path, events: list[dict]) -> _CliSpec:
    script = tmp_path / "fake_claude.py"
    script.write_text(
        "import json\n"
        f"for ev in json.loads({json.dumps(json.dumps(events))}):\n"
        "    print(json.dumps(ev), flush=True)\n", encoding="utf-8")
    return _CliSpec(argv=(sys.executable, str(script)), pass_prompt_via="stdin", output_via="stream_json",
                    model_flag=None)


def _delta(kind: str, text: str) -> dict:
    return {"type": "stream_event", "event": {"type": "content_block_delta",
                                              "delta": {"type": f"{kind}_delta", kind if kind == "text" else "thinking": text}}}


@pytest.mark.asyncio
async def test_the_claude_cli_s_thinking_deltas_are_kept(tmp_path: Path) -> None:
    spec = _stream_binary(tmp_path, [_delta("thinking", "step one; "), _delta("thinking", "step two"),
                                     _delta("text", "Hello"), {"type": "result", "result": "Hello"}])
    holder, token = tc.open_holder()
    try:
        assert await _run_cli(spec, "p", timeout_s=20.0, inactivity_timeout_s=10.0) == "Hello"
        assert holder["text"] == "step one; step two"
    finally:
        tc.close_holder(token)


def _block_start() -> dict:
    return {"type": "stream_event", "event": {"type": "content_block_start", "content_block": {"type": "thinking"}}}


@pytest.mark.asyncio
async def test_streamed_thinking_is_not_repeated_by_the_assistant_message_that_follows(tmp_path: Path) -> None:
    spec = _stream_binary(tmp_path, [
        _block_start(), _delta("thinking", "first block."), _block_start(), _delta("thinking", "second block."),
        _delta("text", "Hi"),
        {"type": "assistant", "message": {"content": [{"type": "thinking", "thinking": "first block."},
                                                       {"type": "thinking", "thinking": "second block."},
                                                       {"type": "text", "text": "Hi"}]}},
        {"type": "result", "result": "Hi"}])
    holder, token = tc.open_holder()
    try:
        assert await _run_cli(spec, "p", timeout_s=20.0, inactivity_timeout_s=10.0) == "Hi"
        assert holder["text"] == "first block.\n\nsecond block."
    finally:
        tc.close_holder(token)


@pytest.mark.asyncio
async def test_the_claude_cli_s_whole_message_thinking_is_kept_once_when_nothing_streamed(tmp_path: Path) -> None:
    spec = _stream_binary(tmp_path, [
        {"type": "assistant", "message": {"content": [{"type": "thinking", "thinking": "weighing it"},
                                                       {"type": "text", "text": "Hi"}]}},
        {"type": "result", "result": "Hi"}])
    holder, token = tc.open_holder()
    try:
        assert await _run_cli(spec, "p", timeout_s=20.0, inactivity_timeout_s=10.0) == "Hi"
        assert holder["text"].strip() == "weighing it"
    finally:
        tc.close_holder(token)


@pytest.mark.asyncio
async def test_the_vscode_bridge_hands_the_thinking_on() -> None:
    server = _MockBridgeServer()

    def handler(msg: dict, w) -> list[dict]:  # noqa: ANN001
        if msg["type"] != "lm_request":
            return []
        return [{"type": "lm_done", "id": msg["id"], "content": "Answer", "thinking": "the model's reasoning"}]

    port = await server.start(handler)
    client = VSCodeBridgeClient(port=port)
    try:
        await client.connect()
        from core.vscode_bridge import LAST_BRIDGE_THINKING

        assert await client.chat(MESSAGES, node="design", model_hint="") == "Answer"
        assert LAST_BRIDGE_THINKING.get() == "the model's reasoning"
    finally:
        await client.aclose()
        await server.stop()


class _Client:
    def __init__(self, thinking: str, answer: str = "answer") -> None:
        self.thinking, self.answer = thinking, answer

    async def chat(self, messages, **kw):  # noqa: ANN001
        tc.note_thinking(self.thinking)
        return self.answer


def _engine(cfg, thinking: str) -> Engine:  # noqa: ANN001
    engine = Engine(cfg)
    engine.fi_dir.mkdir(parents=True, exist_ok=True)
    engine._client = _Client(thinking)  # type: ignore[assignment]
    return engine


@pytest.mark.asyncio
async def test_a_call_s_thinking_is_written_beside_its_call_record(smoke_config, monkeypatch) -> None:  # noqa: ANN001, F811
    monkeypatch.setenv("FI_TEST_SECRET_KEY", "sk-secret-value-123456")
    engine = _engine(smoke_config, "I will use sk-abcdefghijklmnopqrstuvwxyz0123 to look it up")
    assert await engine._chat("prompt", node="design") == "answer"
    (line,) = ar.read(engine.fi_dir, tc.THINKING_FILE)
    (call,) = [r for r in ar.read(engine.fi_dir, ar.MODEL_CALLS) if r["node"] == "design"]
    assert line["node"] == "design" and line["call_id"] == call["call_id"]
    assert line["note"] == tc.THINKING_NOTE and "not evidence" in line["note"]
    assert "sk-abcdefghijklmnopqrstuvwxyz0123" not in line["thinking"] and "I will use" in line["thinking"]
    # The claim events of this step name the same call, the model asked for and why the answer ended.
    prov = engine._chat_provenance("design")
    assert prov["call_id"] == call["call_id"]


@pytest.mark.asyncio
async def test_the_thinking_is_neither_sealed_nor_kept_when_turned_off(smoke_config) -> None:  # noqa: ANN001, F811
    from core import evidence

    assert not any("thinking" in rel for rel in evidence.SEALED_FILES)
    smoke_config.output.save_thinking = False
    engine = _engine(smoke_config, "some reasoning")
    await engine._chat("prompt", node="design")
    assert not (engine.fi_dir / tc.THINKING_FILE).exists()


@pytest.mark.asyncio
async def test_a_call_with_no_reasoning_writes_no_line(smoke_config) -> None:  # noqa: ANN001, F811
    engine = _engine(smoke_config, "")
    await engine._chat("prompt", node="design")
    assert not (engine.fi_dir / tc.THINKING_FILE).exists()


def test_the_setting_is_on_by_default() -> None:
    from core.config import OutputConfig

    assert OutputConfig().save_thinking is True
    assert json.loads(json.dumps({"k": OutputConfig().save_thinking}))["k"] is True


# --- one attempt's reasoning is never filed under another's -----------------------------------------------------------

@pytest.mark.asyncio
async def test_a_failed_rung_s_reasoning_is_not_filed_under_the_rung_that_answers() -> None:
    from core.provider import LAST_CALL, FallbackLLMClient, _CliTransientError

    class _Reasoning:
        last_model = "reasoner"
        last_usage = None

        async def chat(self, messages, **kw):  # noqa: ANN001
            tc.note_thinking("the reasoning of a rung that then failed")
            raise _CliTransientError("timed out")

    class _Plain:
        last_model = "plain"
        last_usage = None

        async def chat(self, messages, **kw):  # noqa: ANN001
            LAST_CALL.set({"provider": "plain", "model": "plain", "reported": True})
            return "answer"  # a transport that never touches the holder (a stdout CLI)

    async def make():  # noqa: ANN202
        return _Plain()

    client = FallbackLLMClient(_Reasoning(), [("plain", make)])  # type: ignore[arg-type]
    holder, token = tc.open_holder()
    try:
        assert await client.chat(MESSAGES) == "answer"
        assert holder["text"] == ""
    finally:
        tc.close_holder(token)


@pytest.mark.asyncio
async def test_the_line_names_who_answered_and_how_the_call_ended(smoke_config) -> None:  # noqa: ANN001, F811
    from core.provider import LAST_CALL

    class _Answering:
        async def chat(self, messages, **kw):  # noqa: ANN001
            tc.note_thinking("a thought")
            LAST_CALL.set({"provider": "codex", "model": "gpt-x", "reported": True, "finish_reason": "stop"})
            return "answer"

    engine = Engine(smoke_config)
    engine.fi_dir.mkdir(parents=True, exist_ok=True)
    engine._client = _Answering()  # type: ignore[assignment]
    await engine._chat("prompt", node="design")
    (line,) = ar.read(engine.fi_dir, tc.THINKING_FILE)
    assert (line["provider"], line["model"], line["outcome"]) == ("codex", "gpt-x", "ok")
    assert engine._chat_provenance("design")["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_calls_made_at_the_same_time_keep_their_own_reasoning(smoke_config) -> None:  # noqa: ANN001, F811
    class _Slow:
        async def chat(self, messages, **kw):  # noqa: ANN001
            node = kw["node"]
            tc.note_thinking(f"{node} starts")
            await asyncio.sleep(0.05 if node == "design" else 0.01)
            tc.add_thinking(f" and {node} ends")
            return node

    engine = Engine(smoke_config)
    engine.fi_dir.mkdir(parents=True, exist_ok=True)
    engine._client = _Slow()  # type: ignore[assignment]
    await asyncio.gather(engine._chat("p", node="design"), engine._chat("p", node="analyze"))
    kept = {r["node"]: r["thinking"] for r in ar.read(engine.fi_dir, tc.THINKING_FILE)}
    assert kept == {"design": "design starts and design ends", "analyze": "analyze starts and analyze ends"}


@pytest.mark.asyncio
async def test_a_failed_call_keeps_its_reasoning_and_says_it_failed(smoke_config) -> None:  # noqa: ANN001, F811
    class _Failing:
        async def chat(self, messages, **kw):  # noqa: ANN001
            tc.note_thinking("thought before the error")
            raise RuntimeError("boom")

    engine = Engine(smoke_config)
    engine.fi_dir.mkdir(parents=True, exist_ok=True)
    engine._client = _Failing()  # type: ignore[assignment]
    with pytest.raises(RuntimeError):
        await engine._chat("prompt", node="design")
    (line,) = ar.read(engine.fi_dir, tc.THINKING_FILE)
    assert line["thinking"] == "thought before the error" and line["outcome"] != "ok"


# --- what is removed, and how much is kept -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_credential_values_and_the_home_folder_are_removed(smoke_config, monkeypatch) -> None:  # noqa: ANN001, F811
    monkeypatch.setenv("FI_TEST_SECRET_KEY", "hunter2-plain-secret-value")
    home = str(Path.home())
    engine = _engine(smoke_config, f"the key is hunter2-plain-secret-value, my data is under {home}/private/data.csv")
    await engine._chat("prompt", node="design")
    (line,) = ar.read(engine.fi_dir, tc.THINKING_FILE)
    assert "the key is" in line["thinking"]  # the text was really there
    assert "hunter2-plain-secret-value" not in line["thinking"] and home not in line["thinking"]
    assert "~" in line["thinking"]


@pytest.mark.asyncio
async def test_one_line_is_cut_and_says_how_much_was_left_out(smoke_config, monkeypatch) -> None:  # noqa: ANN001, F811
    monkeypatch.setattr(tc, "THINKING_LINE_CHARS", 1000)
    engine = _engine(smoke_config, "x" * 5000)
    await engine._chat("prompt", node="design")
    (line,) = ar.read(engine.fi_dir, tc.THINKING_FILE)
    assert line["thinking"].startswith("x" * 1000) and line["thinking"].endswith("[4000 more characters not kept]")


@pytest.mark.asyncio
async def test_the_file_stops_growing_at_its_limit(smoke_config, monkeypatch) -> None:  # noqa: ANN001, F811
    monkeypatch.setattr(tc, "THINKING_FILE_BYTES", 1500)
    engine = _engine(smoke_config, "y" * 400)
    for _ in range(8):
        await engine._chat("prompt", node="design")
    assert 1 <= len(ar.read(engine.fi_dir, tc.THINKING_FILE)) <= 4


@pytest.mark.asyncio
async def test_a_call_that_left_no_record_does_not_lend_its_id_to_the_next_claim(smoke_config, monkeypatch) -> None:  # noqa: ANN001, F811
    engine = _engine(smoke_config, "")
    await engine._chat("first", node="design")
    first = engine._chat_provenance("design")["call_id"]
    monkeypatch.setattr(engine, "_record_model_call", lambda *a, **k: None)
    await engine._chat("second", node="design")
    assert engine._chat_provenance("design").get("call_id") != first


@pytest.mark.asyncio
async def test_no_model_asked_for_means_no_requested_model_in_the_claim(smoke_config) -> None:  # noqa: ANN001, F811
    smoke_config.provider.model = None
    smoke_config.provider.node_models = {}
    engine = _engine(smoke_config, "")
    await engine._chat("prompt", node="design")
    assert "requested_model" not in engine._chat_provenance("design")


def test_a_file_written_after_the_seal_does_not_break_it(tmp_path: Path) -> None:
    from core import evidence
    from tests.test_seal_identity_context import _finished

    root, _record = _finished(tmp_path)
    before = evidence.read(root)["trace_seal"]
    (root / ".fi").mkdir(exist_ok=True)
    (root / ".fi" / tc.THINKING_FILE).write_text('{"thinking": "written after the seal"}\n', encoding="utf-8")
    assert evidence.read(root)["trace_seal"] == before


# --- a long reasoning must not break the VS Code connection ------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("size", [1_000, 200_000, 3_000_000])
async def test_a_long_thinking_on_the_bridge_still_returns_the_answer(size: int) -> None:
    from core.vscode_bridge import LAST_BRIDGE_THINKING

    server = _MockBridgeServer()

    def handler(msg: dict, w) -> list[dict]:  # noqa: ANN001
        if msg["type"] != "lm_request":
            return []
        return [{"type": "lm_done", "id": msg["id"], "content": "Answer", "thinking": "t" * size}]

    port = await server.start(handler)
    client = VSCodeBridgeClient(port=port)
    try:
        await client.connect()
        assert await client.chat(MESSAGES, node="design", model_hint="") == "Answer"
        assert len(LAST_BRIDGE_THINKING.get() or "") == size
    finally:
        await client.aclose()
        await server.stop()


@pytest.mark.asyncio
async def test_a_real_bridge_call_reaches_the_file(smoke_config) -> None:  # noqa: ANN001, F811
    from core.provider import ResolvedEndpoint

    server = _MockBridgeServer()

    def handler(msg: dict, w) -> list[dict]:  # noqa: ANN001
        if msg["type"] != "lm_request":
            return []
        return [{"type": "lm_done", "id": msg["id"], "content": "Answer", "thinking": "reasoned over the bridge"}]

    port = await server.start(handler)
    client = LLMClient(ResolvedEndpoint(base_url="", model="(VSCode chat default)", api_key="not-needed",
                                        transport="vscode_bridge", vscode_bridge_port=port))
    engine = Engine(smoke_config)
    engine.fi_dir.mkdir(parents=True, exist_ok=True)
    engine._client = client  # type: ignore[assignment]
    try:
        assert await engine._chat("prompt", node="design") == "Answer"
    finally:
        await client.aclose()
        await server.stop()
    (line,) = ar.read(engine.fi_dir, tc.THINKING_FILE)
    assert line["thinking"] == "reasoned over the bridge" and line["node"] == "design"


# --- the extension's message builder (compiled TypeScript, run under node) --------------------------------------------

_NODE_DONE = """
const { lmDoneMessage, LM_DONE_MAX_BYTES } = require(%s);
const bytes = (m) => Buffer.byteLength(JSON.stringify(m), "utf8") + 1;
const base = { type: "lm_done", id: 1, content: "Answer" };
const huge = lmDoneMessage(base, "t".repeat(2000000));
process.stdout.write(JSON.stringify({
  small: lmDoneMessage(base, "short reasoning").thinking,
  none: "thinking" in lmDoneMessage(base, ""),
  hugeBytes: bytes(huge),
  hugeTail: huge.thinking.split("\\n").pop(),
  cjkBytes: bytes(lmDoneMessage(base, "\\u63a8".repeat(100000))),
  bigAnswer: "thinking" in lmDoneMessage({ ...base, content: "a".repeat(LM_DONE_MAX_BYTES) }, "reasoning"),
  max: LM_DONE_MAX_BYTES,
}));
"""


def test_the_extension_cuts_the_thinking_so_an_lm_done_line_stays_small(tmp_path: Path) -> None:
    import shutil
    import subprocess

    ext = Path(__file__).resolve().parent.parent / "vscode-frontier-insight"
    compiled = ext / "out" / "lm-messages.js"
    node = shutil.which("node")
    if node is None or not compiled.exists():
        pytest.skip("node or the compiled extension (npm run compile) is missing")
    driver = tmp_path / "driver.js"
    driver.write_text(_NODE_DONE % json.dumps(str(compiled).replace("\\", "/")), encoding="utf-8")
    run = subprocess.run([node, str(driver)], capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stderr
    out = json.loads(run.stdout)
    assert out["small"] == "short reasoning" and out["none"] is False
    assert out["hugeBytes"] <= out["max"] and out["cjkBytes"] <= out["max"] < 65536
    assert out["hugeTail"].endswith("more characters not sent]")
    assert out["bigAnswer"] is False
