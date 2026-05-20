"""Python side of the VSCode-extension bridge.

Tests run against an in-process mock TCP server that speaks the wire
protocol described in `core/vscode_bridge.py`. No real VSCode required.

Covered:
  - Round-trip: send lm_request, receive lm_chunk+lm_done, get content.
  - Concurrent calls demux correctly by id (out-of-order responses).
  - lm_error from the extension surfaces as BridgeError on the awaiter.
  - Bridge connection drop mid-request resolves pending awaiters with
    BridgeError rather than hanging.
  - Provider integration: a `vscode_extension` ResolvedEndpoint with
    vscode_bridge_port set routes LLMClient.chat through the bridge.
  - Engine-side per-call model routing flows model_hint into the wire
    payload (per-node routing works through this transport).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from core.config import ProviderConfig
from core.provider import LLMClient, ResolvedEndpoint, resolve_endpoint
from core.vscode_bridge import VSCodeBridgeClient, BridgeError


class _MockBridgeServer:
    """In-process TCP server that plays the role of the FI VSCode
    extension. Records every Python→server message and lets each test
    script the responses."""

    def __init__(self) -> None:
        self._server: asyncio.AbstractServer | None = None
        self.port: int = 0
        self.received: list[dict[str, Any]] = []
        # handler(msg) → list[response_dict] to send back, OR a single
        # dict, OR None to send nothing. Allows scripted responses.
        self._handler = None
        # Track active connections so tests can force-close them.
        self._writers: list[asyncio.StreamWriter] = []

    async def start(self, handler) -> int:
        self._handler = handler
        self._server = await asyncio.start_server(
            self._client_connected, "127.0.0.1", 0,
        )
        self.port = self._server.sockets[0].getsockname()[1]
        return self.port

    async def stop(self) -> None:
        for w in list(self._writers):
            if not w.is_closing():
                w.close()
                try:
                    await w.wait_closed()
                except Exception:
                    pass
        if self._server is not None:
            self._server.close()
            try:
                await self._server.wait_closed()
            except Exception:
                pass

    async def _client_connected(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
    ) -> None:
        self._writers.append(writer)
        try:
            while True:
                line = await reader.readline()
                if not line:
                    return
                msg = json.loads(line.decode("utf-8"))
                self.received.append(msg)
                if self._handler is None:
                    continue
                responses = self._handler(msg, writer) or []
                if isinstance(responses, dict):
                    responses = [responses]
                for resp in responses:
                    writer.write((json.dumps(resp) + "\n").encode("utf-8"))
                await writer.drain()
        except (ConnectionError, asyncio.CancelledError):
            return

    def force_drop_all(self) -> None:
        """Close every active connection mid-request. Used to test that
        pending awaiters fail gracefully instead of hanging."""
        for w in list(self._writers):
            try:
                w.close()
            except Exception:
                pass


@pytest.mark.asyncio
async def test_bridge_round_trip_chunked_then_done() -> None:
    """Standard happy path: extension streams 3 chunks then an
    `lm_done` with the final content. Client returns the content."""
    server = _MockBridgeServer()

    def handler(msg: dict[str, Any], w) -> list[dict[str, Any]]:
        if msg["type"] != "lm_request":
            return []
        rid = msg["id"]
        return [
            {"type": "lm_chunk", "id": rid, "delta": "Hello, "},
            {"type": "lm_chunk", "id": rid, "delta": "world"},
            {"type": "lm_done", "id": rid, "content": "Hello, world!",
             "total_tokens": 8},
        ]

    port = await server.start(handler)
    client = VSCodeBridgeClient(port=port)
    try:
        await client.connect()
        out = await client.chat(
            [{"role": "user", "content": "hi"}],
            node="ideate", model_hint="gpt-5",
        )
        assert out == "Hello, world!"
        # The Python side actually sent what the extension expects.
        sent = server.received[0]
        assert sent["type"] == "lm_request"
        assert sent["node"] == "ideate"
        assert sent["model_hint"] == "gpt-5"
        assert sent["messages"][0]["content"] == "hi"
    finally:
        await client.aclose()
        await server.stop()


@pytest.mark.asyncio
async def test_bridge_falls_back_to_chunk_reassembly_when_done_omits_content() -> None:
    """Some extension implementations stream chunks and send an
    `lm_done` with no final `content` field. The client reassembles
    the chunks rather than returning an empty string."""
    server = _MockBridgeServer()

    def handler(msg: dict[str, Any], w) -> list[dict[str, Any]]:
        rid = msg["id"]
        return [
            {"type": "lm_chunk", "id": rid, "delta": "alpha "},
            {"type": "lm_chunk", "id": rid, "delta": "beta"},
            {"type": "lm_done", "id": rid},  # no `content`
        ]

    port = await server.start(handler)
    client = VSCodeBridgeClient(port=port)
    try:
        await client.connect()
        out = await client.chat([{"role": "user", "content": "x"}])
        assert out == "alpha beta"
    finally:
        await client.aclose()
        await server.stop()


@pytest.mark.asyncio
async def test_bridge_concurrent_calls_demux_by_id() -> None:
    """Two chat calls in flight at the same time. The extension
    answers in REVERSE order. The client must route each response to
    the matching awaiter — id correlation is the whole point of the
    protocol."""
    server = _MockBridgeServer()

    pending_responses: list[tuple[asyncio.StreamWriter, dict[str, Any]]] = []

    def handler(msg: dict[str, Any], w) -> list[dict[str, Any]]:
        # Park the response in the queue; we drain it manually below.
        if msg["type"] == "lm_request":
            rid = msg["id"]
            content = f"reply-{rid}"
            pending_responses.append((w, {
                "type": "lm_done", "id": rid, "content": content,
            }))
        return []

    port = await server.start(handler)
    client = VSCodeBridgeClient(port=port)
    try:
        await client.connect()

        t1 = asyncio.create_task(
            client.chat([{"role": "user", "content": "first"}])
        )
        t2 = asyncio.create_task(
            client.chat([{"role": "user", "content": "second"}])
        )
        # Wait until both requests have been received before flushing.
        for _ in range(50):
            await asyncio.sleep(0.01)
            if len(pending_responses) == 2:
                break
        assert len(pending_responses) == 2

        # Flush IN REVERSE order — the second request finishes first.
        for w, resp in reversed(pending_responses):
            w.write((json.dumps(resp) + "\n").encode("utf-8"))
            await w.drain()

        r1, r2 = await asyncio.gather(t1, t2)
        # Each call got its own correctly-correlated reply.
        assert r1.startswith("reply-")
        assert r2.startswith("reply-")
        assert r1 != r2
    finally:
        await client.aclose()
        await server.stop()


@pytest.mark.asyncio
async def test_bridge_lm_error_raises_bridge_error() -> None:
    """When the extension can't satisfy a request (no model selected,
    user revoked consent, rate-limited, etc.) it sends `lm_error` and
    the awaiting Python call raises BridgeError, not RuntimeError."""
    server = _MockBridgeServer()

    def handler(msg: dict[str, Any], w) -> list[dict[str, Any]]:
        return [{"type": "lm_error", "id": msg["id"],
                 "error": "no model selected in Chat picker"}]

    port = await server.start(handler)
    client = VSCodeBridgeClient(port=port)
    try:
        await client.connect()
        with pytest.raises(BridgeError, match="no model selected"):
            await client.chat([{"role": "user", "content": "x"}])
    finally:
        await client.aclose()
        await server.stop()


@pytest.mark.asyncio
async def test_bridge_connection_drop_mid_request_surfaces_error() -> None:
    """If the extension crashes or the user closes VSCode while FI is
    awaiting a response, the bridge connection drops. Pending awaiters
    must fail with BridgeError rather than hanging forever."""
    server = _MockBridgeServer()

    def handler(msg: dict[str, Any], w) -> list[dict[str, Any]]:
        # Receive the request but never reply.
        return []

    port = await server.start(handler)
    client = VSCodeBridgeClient(port=port)
    try:
        await client.connect()
        task = asyncio.create_task(
            client.chat([{"role": "user", "content": "x"}])
        )
        # Wait until the request lands on the server, then force-drop.
        for _ in range(50):
            await asyncio.sleep(0.01)
            if server.received:
                break
        server.force_drop_all()
        with pytest.raises(BridgeError, match="dropped|closed"):
            await asyncio.wait_for(task, timeout=2.0)
    finally:
        await client.aclose()
        await server.stop()


@pytest.mark.asyncio
async def test_bridge_connect_to_dead_port_raises_immediately() -> None:
    """Trying to talk to a port nobody's listening on should raise
    quickly with a clear BridgeError — not hang."""
    client = VSCodeBridgeClient(port=1)  # privileged port, almost certainly closed
    with pytest.raises(BridgeError):
        await client.connect()


@pytest.mark.asyncio
async def test_bridge_chat_raises_when_response_never_arrives() -> None:
    """The TS-side bridge has its own 180 s inactivity timer, but a
    TCP/IPC blip that drops the ``lm_error`` itself — or an older
    .vsix without the timer — would still hang Python's ``await fut``
    forever. ``VSCodeBridgeClient.cli_timeout_s`` is the belt-and-braces
    ceiling: a bridge that never resolves the future must surface
    ``BridgeError("bridge stalled ...")`` so tenacity can retry, not
    wedge the engine indefinitely.

    Audit Slice 4 BLOCK #1 / Slice 7 HIGH #1: ``--serve`` and
    ``--tools`` quests hung forever on Copilot HTTP/2 stalls because
    the persistent-bridge had no equivalent of ``bridge.ts``'s
    inactivity timer."""
    server = _MockBridgeServer()

    def handler(msg: dict[str, Any], w) -> list[dict[str, Any]]:
        # Receive the request but never reply — simulates a TS bridge
        # whose inactivity timer also failed to fire (or an older .vsix
        # without the timer).
        return []

    port = await server.start(handler)
    # 1 s budget keeps the test fast; production default is 300 s.
    client = VSCodeBridgeClient(port=port, cli_timeout_s=1.0)
    try:
        await client.connect()
        # Outer ``wait_for`` is a regression guard. If a future change
        # to ``VSCodeBridgeClient.chat`` lost the inner ``wait_for``
        # (or the inner timeout ever stopped firing), the inner future
        # would await forever and CI would hang for the full test
        # timeout instead of failing fast. ``cli_timeout_s=1.0`` means
        # ``chat`` should raise within ~1 s; we give ourselves 5 s of
        # slack before forcing a failure.
        with pytest.raises(BridgeError, match="bridge stalled"):
            await asyncio.wait_for(
                client.chat([{"role": "user", "content": "x"}]),
                timeout=5.0,
            )
        # The "bridge stalled" phrasing is load-bearing — Python's
        # ``_TRANSIENT_BRIDGE_MARKERS`` matches on it so tenacity
        # retries instead of crashing the quest.
        from core.provider import _is_bridge_error_transient
        assert _is_bridge_error_transient(
            "bridge stalled (no response within 1s)"
        ), "stall timeout error must be classified as transient"
    finally:
        await client.aclose()
        await server.stop()


def test_provider_config_exposes_cli_timeout_s() -> None:
    """``ProviderConfig.cli_timeout_s`` is the user-facing knob for
    the bridge stall ceiling (and the CLI-provider wall-clock cap).
    Audit Slice 4 HIGH #3 flagged this missing; the bridge-stall fix
    depends on a configurable override."""
    cfg = ProviderConfig(name="vscode_extension", extra={"bridge_port": 1})
    # Default matches the CLI-transport budget in LLMClient.
    assert cfg.cli_timeout_s == 300.0
    cfg2 = ProviderConfig(
        name="vscode_extension", extra={"bridge_port": 1}, cli_timeout_s=45.0,
    )
    assert cfg2.cli_timeout_s == 45.0


@pytest.mark.asyncio
async def test_llmclient_threads_cli_timeout_into_bridge_client() -> None:
    """``LLMClient(cli_timeout_s=...)`` must reach the
    ``VSCodeBridgeClient`` it constructs lazily — otherwise the
    user-facing ``provider.cli_timeout_s`` knob silently does nothing
    on the bridge path and a stall still wedges the engine."""
    server = _MockBridgeServer()

    def handler(msg: dict[str, Any], w) -> list[dict[str, Any]]:
        return [{"type": "lm_done", "id": msg["id"], "content": "ok"}]

    port = await server.start(handler)
    ep = ResolvedEndpoint(
        base_url="", model="(VSCode chat default)", api_key="not-needed",
        transport="vscode_bridge", vscode_bridge_port=port,
    )
    client = LLMClient(ep, cli_timeout_s=17.5)
    try:
        await client.chat([{"role": "user", "content": "hi"}])
        # The lazily-built bridge must have inherited the timeout.
        assert client._bridge is not None
        assert client._bridge.cli_timeout_s == 17.5
    finally:
        await client.aclose()
        await server.stop()


# --- Provider integration ---------------------------------------------------


def test_resolve_endpoint_vscode_extension_requires_bridge_port() -> None:
    """`provider.name = vscode_extension` without `extra['bridge_port']`
    must raise with a clear hint about who's supposed to provide it
    (the FI VSCode extension via --vscode-bridge-port)."""
    cfg = ProviderConfig(name="vscode_extension")
    with pytest.raises(RuntimeError, match="bridge_port"):
        resolve_endpoint(cfg)


def test_resolve_endpoint_vscode_extension_returns_bridge_transport() -> None:
    cfg = ProviderConfig(name="vscode_extension", extra={"bridge_port": 8721})
    ep = resolve_endpoint(cfg)
    assert ep.transport == "vscode_bridge"
    assert ep.vscode_bridge_port == 8721
    # Display string never goes blank in Engine logs.
    assert ep.model and ep.model != ""
    # CRITICAL: `vscode_model_override` is empty when the YAML did
    # NOT pin a model. The display string above is for logs only —
    # using it as a real model_hint would crash selectChatModels
    # with "no Copilot model for hint '(VSCode chat default)'".
    assert ep.vscode_model_override == "", (
        "When provider.model is unset, the bridge MUST send an empty "
        "model_hint over the wire — anything else (including the "
        "display string) will be treated as a literal model family "
        "name by vscode.lm.selectChatModels and fail."
    )


def test_resolve_endpoint_vscode_extension_with_explicit_model() -> None:
    """When the YAML pins `provider.model`, that value flows out as
    `vscode_model_override` (the wire-level field) AND as the display
    `model`. They're the same string here because the user picked it
    intentionally."""
    cfg = ProviderConfig(
        name="vscode_extension", model="gpt-4o",
        extra={"bridge_port": 8721},
    )
    ep = resolve_endpoint(cfg)
    assert ep.vscode_model_override == "gpt-4o"
    assert ep.model == "gpt-4o"


@pytest.mark.asyncio
async def test_bridge_chat_sends_empty_hint_when_no_model_pinned() -> None:
    """End-to-end regression: a `vscode_extension` provider with no
    explicit `provider.model` must send an EMPTY `model_hint` over
    the wire, not the human-readable display fallback. The latter
    crashed a real user's quest with
    'no Copilot model available for hint (VSCode chat default)'."""
    server = _MockBridgeServer()
    captured: list[dict[str, Any]] = []

    def handler(msg, w):
        captured.append(msg)
        return [{"type": "lm_done", "id": msg["id"], "content": "ok"}]

    port = await server.start(handler)
    cfg = ProviderConfig(name="vscode_extension", extra={"bridge_port": port})
    ep = resolve_endpoint(cfg)
    client = LLMClient(ep)
    try:
        await client.chat([{"role": "user", "content": "hi"}])
        wire = captured[0]
        assert wire["model_hint"] == "", (
            f"unset provider.model leaked the display string {wire['model_hint']!r} "
            f"into the wire — selectChatModels would reject it"
        )
    finally:
        await client.aclose()
        await server.stop()


@pytest.mark.asyncio
async def test_bridge_clarify_round_trip() -> None:
    """Engine fires `interactive` clarify_mode → extension shows
    modals → user answers → engine resumes. The bridge protocol
    carries the questions out and the answers back, keyed by id."""
    server = _MockBridgeServer()
    received: list[dict[str, Any]] = []

    def handler(msg: dict[str, Any], w) -> list[dict[str, Any]]:
        received.append(msg)
        if msg["type"] == "clarify_request":
            return [{
                "type": "clarify_response",
                "id": msg["id"],
                "answers": {
                    "comparative_baseline": "user-chosen X",
                    "success_metric": "user-chosen Y",
                },
            }]
        return []

    port = await server.start(handler)
    client = VSCodeBridgeClient(port=port)
    try:
        await client.connect()
        answers = await client.clarify({
            "comparative_baseline": {"question": "what baseline?", "default": "trivial"},
            "success_metric": {"question": "what metric?", "default": "RMSE"},
        })
        assert answers["comparative_baseline"] == "user-chosen X"
        assert answers["success_metric"] == "user-chosen Y"
        # The wire payload carried the questions out under the
        # expected key.
        out = received[0]
        assert out["type"] == "clarify_request"
        assert "comparative_baseline" in out["questions"]
    finally:
        await client.aclose()
        await server.stop()


@pytest.mark.asyncio
async def test_bridge_clarify_cancel_raises_bridge_error() -> None:
    """User pressed Esc → extension sends clarify_cancelled → Python's
    `clarify()` raises BridgeError. The engine's caller can catch
    this and decide how to recover (typically: fail the quest with a
    clear message, or fall back to per-question defaults)."""
    server = _MockBridgeServer()

    def handler(msg: dict[str, Any], w) -> list[dict[str, Any]]:
        if msg["type"] == "clarify_request":
            return [{"type": "clarify_cancelled", "id": msg["id"]}]
        return []

    port = await server.start(handler)
    client = VSCodeBridgeClient(port=port)
    try:
        await client.connect()
        with pytest.raises(BridgeError, match="cancelled"):
            await client.clarify({"x": {"question": "?", "default": "a"}})
    finally:
        await client.aclose()
        await server.stop()


@pytest.mark.asyncio
async def test_bridge_clarify_rejects_malformed_response() -> None:
    """If the extension sends `clarify_response` without an `answers`
    dict, `clarify()` raises rather than returning a malformed result."""
    server = _MockBridgeServer()

    def handler(msg: dict[str, Any], w) -> list[dict[str, Any]]:
        if msg["type"] == "clarify_request":
            # `answers` is the wrong type — should be a dict.
            return [{"type": "clarify_response", "id": msg["id"], "answers": "wrong"}]
        return []

    port = await server.start(handler)
    client = VSCodeBridgeClient(port=port)
    try:
        await client.connect()
        with pytest.raises(BridgeError, match="answers"):
            await client.clarify({"x": {"question": "?", "default": "a"}})
    finally:
        await client.aclose()
        await server.stop()


@pytest.mark.asyncio
async def test_launch_picks_bridge_clarify_callback_for_vscode_extension(
    tmp_path,
) -> None:
    """When the config says ``provider.name = vscode_extension`` with a
    bridge port, ``launch._pick_clarify_callback`` returns a callable
    that round-trips through the bridge. Without this wiring (the bug
    that prompted these tests), the engine would crash with a
    "no clarify_callback supplied" RuntimeError when the YAML asked
    for ``engine.clarify_mode = interactive``.

    This test also pins the SINGLE-bridge invariant: the callback
    must re-use ``engine._client._bridge`` rather than opening a
    second connection. The extension's bridge server only tracks
    one socket; two concurrent clients would have their responses
    cross-routed."""
    from launch import _pick_clarify_callback
    from core.config import (
        Config, EngineConfig, ExecutionConfig, KnowledgeConfig,
        OutputConfig, ProviderConfig,
    )
    from core.engine import Engine
    from core.provider import LLMClient, ResolvedEndpoint

    server = _MockBridgeServer()

    def handler(msg, w):
        if msg["type"] == "clarify_request":
            return [{"type": "clarify_response", "id": msg["id"],
                     "answers": {"comparative_baseline": "bridge-routed"}}]
        return []

    port = await server.start(handler)
    try:
        cfg = Config(
            topic="bridge clarify wiring",
            title="bcw",
            provider=ProviderConfig(
                name="vscode_extension",
                extra={"bridge_port": port},
            ),
            engine=EngineConfig(
                max_iterations=1, review_loop=False,
                clarify_mode="interactive", ideate_reflect=False,
            ),
            execution=ExecutionConfig(sandbox="venv", timeout_s=60),
            knowledge=KnowledgeConfig(enabled=False),
            output=OutputConfig(output_dir=tmp_path / "out"),
        )
        engine = Engine(cfg)
        # Simulate the state Engine.run() leaves things in just before
        # the clarify-interrupt fires: an LLMClient exists with a
        # connected bridge. The callback under test must re-use this
        # bridge rather than open its own.
        ep = ResolvedEndpoint(
            base_url="", model="(VSCode chat default)", api_key="not-needed",
            transport="vscode_bridge", vscode_bridge_port=port,
        )
        engine._client = LLMClient(ep)
        existing_bridge = VSCodeBridgeClient(host="127.0.0.1", port=port)
        await existing_bridge.connect()
        engine._client._bridge = existing_bridge

        cb = _pick_clarify_callback(cfg, engine, interactive=False)
        assert cb is not None, (
            "no clarify callback returned for vscode_extension — the "
            "interactive clarify mode would crash without one"
        )
        answers = await cb({
            "comparative_baseline": {"question": "?", "default": "default"},
        })
        assert answers == {"comparative_baseline": "bridge-routed"}
        # The callback must NOT have opened a second bridge — the
        # client's bridge attribute is the same object we created.
        assert engine._client._bridge is existing_bridge, (
            "clarify callback opened a second bridge; this will crash "
            "the live extension because its TCP server only tracks "
            "one active socket"
        )
        await existing_bridge.aclose()
    finally:
        await server.stop()


def test_is_bridge_error_transient_classification() -> None:
    """The transient-marker list must catch every well-known Copilot
    backend failure mode so we don't crash a 5-minute quest on a
    one-second network blip — but it must NOT match auth or model-
    selection errors, which won't get better with retry."""
    from core.provider import _is_bridge_error_transient

    # Transient — should retry.
    assert _is_bridge_error_transient(
        "Please check your firewall rules and network connection then "
        "try again. Error Code: net::ERR_HTTP2_PROTOCOL_ERROR."
    )
    assert _is_bridge_error_transient("socket hang up")
    assert _is_bridge_error_transient("Server returned 503 Service Unavailable")
    assert _is_bridge_error_transient("rate limit exceeded")
    assert _is_bridge_error_transient("bridge connection dropped")
    assert _is_bridge_error_transient("ECONNRESET on chunked body")
    # TS-side stall-detection (no chunk for 180 s) surfaces this
    # error and must be retried, not propagated immediately —
    # otherwise the bridge can hang for many minutes on a silent
    # model.
    assert _is_bridge_error_transient(
        "bridge stalled: no chunk for 180 s (received 0 chunks / 0 chars before stall)"
    )

    # Non-transient — should propagate immediately, no retry.
    assert not _is_bridge_error_transient(
        "no Copilot model available for hint 'gpt-5.4-mini'"
    )
    assert not _is_bridge_error_transient("user cancelled the clarify panel")
    assert not _is_bridge_error_transient("authentication failed: invalid token")
    assert not _is_bridge_error_transient("")


@pytest.mark.asyncio
async def test_llmclient_chat_retries_transient_bridge_error() -> None:
    """Regression: a real user hit `net::ERR_HTTP2_PROTOCOL_ERROR` on
    the implement-node call and lost a 5-minute quest. The bridge
    transport's chat path must retry transient errors with backoff
    (same pattern HTTP and CLI transports already use)."""
    server = _MockBridgeServer()

    call_count = {"n": 0}

    def handler(msg, w):
        call_count["n"] += 1
        if msg["type"] == "lm_request":
            if call_count["n"] <= 2:
                return [{
                    "type": "lm_error", "id": msg["id"],
                    "error": "Please check your firewall rules and network "
                             "connection then try again. Error Code: "
                             "net::ERR_HTTP2_PROTOCOL_ERROR.",
                }]
            return [{"type": "lm_done", "id": msg["id"], "content": "ok"}]
        return []

    port = await server.start(handler)
    ep = ResolvedEndpoint(
        base_url="", model="(VSCode chat default)", api_key="not-needed",
        transport="vscode_bridge", vscode_bridge_port=port,
    )
    client = LLMClient(ep)
    try:
        # Patch the tenacity backoff so the test doesn't actually sleep
        # 2s + 4s between attempts.
        from unittest.mock import patch
        with patch("core.provider.wait_exponential",
                   return_value=lambda *a, **kw: 0):
            out = await client.chat([{"role": "user", "content": "hi"}])
        assert out == "ok"
        assert call_count["n"] == 3, (
            f"expected 3 attempts (2 transient + 1 success); got {call_count['n']}"
        )
    finally:
        await client.aclose()
        await server.stop()


@pytest.mark.asyncio
async def test_llmclient_chat_retries_up_to_six_then_friendly_error() -> None:
    """Regression: the prior 3-attempt budget was too tight for
    sustained Copilot HTTP/2 outages (observed 30-90s). Budget is now
    6 attempts (~2 min cumulative backoff); on full exhaustion, the
    surfaced error must clearly say the issue is upstream so users
    don't chase phantom config bugs."""
    from core.vscode_bridge import BridgeError

    server = _MockBridgeServer()
    call_count = {"n": 0}

    def handler(msg, w):
        call_count["n"] += 1
        if msg["type"] == "lm_request":
            return [{
                "type": "lm_error", "id": msg["id"],
                "error": "Please check your firewall rules and network "
                         "connection then try again. Error Code: "
                         "net::ERR_HTTP2_PROTOCOL_ERROR.",
            }]
        return []

    port = await server.start(handler)
    ep = ResolvedEndpoint(
        base_url="", model="(VSCode chat default)", api_key="not-needed",
        transport="vscode_bridge", vscode_bridge_port=port,
    )
    client = LLMClient(ep)
    try:
        from unittest.mock import patch
        with patch("core.provider.wait_exponential",
                   return_value=lambda *a, **kw: 0):
            with pytest.raises(BridgeError) as excinfo:
                await client.chat([{"role": "user", "content": "hi"}])
        # 6 attempts, all failing transient.
        assert call_count["n"] == 6, (
            f"expected 6 attempts before exhaustion; got {call_count['n']}"
        )
        msg = str(excinfo.value).lower()
        assert "upstream" in msg, f"final error must name upstream: {excinfo.value!r}"
    finally:
        await client.aclose()
        await server.stop()


@pytest.mark.asyncio
async def test_llmclient_chat_routes_vscode_bridge_transport() -> None:
    """End-to-end: a ResolvedEndpoint with transport=vscode_bridge
    sends the chat over the bridge, and the per-call `model` kwarg
    surfaces as `model_hint` on the wire — per-node model routing
    composes with the bridge transport."""
    server = _MockBridgeServer()
    captured: list[dict[str, Any]] = []

    def handler(msg: dict[str, Any], w) -> list[dict[str, Any]]:
        captured.append(msg)
        return [{"type": "lm_done", "id": msg["id"], "content": "from-bridge"}]

    port = await server.start(handler)
    ep = ResolvedEndpoint(
        base_url="", model="(VSCode chat default)", api_key="not-needed",
        transport="vscode_bridge", vscode_bridge_port=port,
    )
    client = LLMClient(ep)
    try:
        out = await client.chat(
            [{"role": "user", "content": "hi"}],
            model="claude-opus-4-7",  # per-call override
        )
        assert out == "from-bridge"
        wire = captured[0]
        assert wire["type"] == "lm_request"
        assert wire["model_hint"] == "claude-opus-4-7"
    finally:
        await client.aclose()
        await server.stop()
