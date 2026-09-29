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
        holder = tc.open_holder()
        http = httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=30.0)
        try:
            text = await LLMClient(endpoint, http=http).chat(MESSAGES)
            return text, holder["text"]
        finally:
            await http.aclose()
            tc.close_holder()

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
    tc.open_holder()
    tc.close_holder()
    tc.note_thinking("still no one")


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
    holder = tc.open_holder()
    try:
        assert await _run_cli(spec, "p", timeout_s=20.0, inactivity_timeout_s=10.0) == "Hello"
        assert holder["text"] == "step one; step two"
    finally:
        tc.close_holder()


@pytest.mark.asyncio
async def test_the_claude_cli_s_whole_message_thinking_is_kept_once_when_nothing_streamed(tmp_path: Path) -> None:
    spec = _stream_binary(tmp_path, [
        {"type": "assistant", "message": {"content": [{"type": "thinking", "thinking": "weighing it"},
                                                       {"type": "text", "text": "Hi"}]}},
        {"type": "result", "result": "Hi"}])
    holder = tc.open_holder()
    try:
        assert await _run_cli(spec, "p", timeout_s=20.0, inactivity_timeout_s=10.0) == "Hi"
        assert holder["text"].strip() == "weighing it"
    finally:
        tc.close_holder()


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
    assert prov["call_id"] == call["call_id"] and "requested_model" in prov


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
