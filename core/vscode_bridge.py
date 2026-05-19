"""Python side of the VSCode-extension bridge.

The FI VSCode extension (TypeScript, `vscode-frontier-insight/`) spawns
the Python engine with ``--vscode-bridge-port <N>`` and listens on a
TCP socket at ``127.0.0.1:<N>``. Every LLM call from the engine becomes
a newline-delimited JSON message on that socket; the extension routes
the request through ``vscode.lm.selectChatModels`` / ``model.sendRequest``
on the user's authenticated Copilot session and streams the response
back as chunks.

Wire protocol (one JSON object per line; both directions):

Python → extension::

    {"type":"lm_request","id":<int>,"node":"<name>","messages":[...],
     "model_hint":"<optional>","temperature":<float>}
    {"type":"clarify_request","id":<int>,"questions":{...}}
    {"type":"quest_event","event":"node_started","node":"<name>",
     "quest_id":"<id>"}
    {"type":"quest_event","event":"quest_done","quest_id":"<id>",
     "paper_path":"<path>"}

Extension → Python::

    {"type":"lm_chunk","id":<int>,"delta":"<text>"}
    {"type":"lm_done","id":<int>,"content":"<text>","total_tokens":<int>}
    {"type":"lm_error","id":<int>,"error":"<message>"}
    {"type":"clarify_response","id":<int>,"answers":{...}}
    {"type":"clarify_cancelled","id":<int>}

Why a TCP socket (and not stdio): logging, print(), tenacity, and
pip-install subprocess noise all leak into stdout of a Python process
in the middle of a quest. Reserving stdio for the bridge would force
us to muzzle every observability channel. A dedicated localhost socket
keeps stdout free for normal logs and the bridge separate.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

_log = logging.getLogger("frontier_insight.vscode_bridge")


class BridgeError(RuntimeError):
    """Surfaced when the extension reports an `lm_error` or the
    bridge connection drops mid-call. Distinct from RuntimeError so
    LLMClient's retry layer can ignore it (re-trying without VSCode
    being clicked again would be pointless)."""


class VSCodeBridgeClient:
    """Owns one persistent TCP connection to the FI VSCode extension.

    A single client instance is created per FI run and shared across
    every node (and across every quest if the extension launched FI
    in fleet mode). Concurrent calls are demultiplexed by request id.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        *,
        socket_path: str | None = None,
    ) -> None:
        """Construct an idle client. ``socket_path`` takes precedence
        over ``host``/``port`` when set — that's the per-user IPC
        path the extension's PersistentBridge listens on (Unix domain
        socket on POSIX, named pipe on Windows). Falling back to
        ``host``/``port`` preserves the per-chat-command TCP path the
        extension's per-spawn :class:`Bridge` still uses."""
        self.host = host
        self.port = port
        self.socket_path = socket_path
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        # Per-request futures keyed by id. lm requests resolve to the
        # response text; clarify requests resolve to the answers dict.
        # We mix kinds in one map keyed by id because each id is unique
        # across the lifetime of the client.
        self._pending: dict[int, asyncio.Future[Any]] = {}
        # Streaming-chunk buffers per request, in case the extension
        # streams in chunks before sending the final lm_done.
        self._chunks: dict[int, list[str]] = {}
        self._next_id = 1
        self._reader_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()  # serialize writes

    async def connect(self) -> None:
        """Open the connection to the extension. Idempotent.

        Two transports today:
        * ``socket_path`` set → connect to the extension's
          PersistentBridge over a Unix-domain socket (POSIX) or named
          pipe (Windows).
        * Otherwise → TCP connect to ``host``/``port`` (the
          per-chat-command transport the extension's per-spawn Bridge
          class uses).
        """
        if self._writer is not None and not self._writer.is_closing():
            return
        if self.socket_path:
            self._reader, self._writer = await _open_ipc_connection(
                self.socket_path,
            )
            self._reader_task = asyncio.create_task(self._read_loop())
            _log.info(
                "connected to VSCode persistent bridge at %s",
                self.socket_path,
            )
            return
        try:
            self._reader, self._writer = await asyncio.open_connection(
                self.host, self.port,
            )
        except OSError as e:
            raise BridgeError(
                f"could not connect to VSCode bridge at "
                f"{self.host}:{self.port}: {e}"
            ) from e
        self._reader_task = asyncio.create_task(self._read_loop())
        _log.info("connected to VSCode bridge at %s:%d", self.host, self.port)

    async def aclose(self) -> None:
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
            try:
                await self._reader_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._writer is not None and not self._writer.is_closing():
            self._writer.close()
            try:
                await self._writer.wait_closed()
            except Exception:
                pass
        # Resolve any pending futures so awaiters don't hang on shutdown.
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(BridgeError("bridge closed"))
        self._pending.clear()
        self._chunks.clear()

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        node: str = "",
        model_hint: str = "",
        temperature: float = 0.2,
    ) -> str:
        """Send one ``lm_request`` to the extension and await its
        ``lm_done``. The extension does the actual ``vscode.lm`` call
        on the user's behalf; we just shuttle JSON.

        ``node`` is the FI engine node name (``"ideate"``,
        ``"review_panel.statistician"``, …) so the extension can render
        progress in the chat panel with the right tag.

        ``model_hint`` is the resolved per-node model name; the
        extension translates it into a ``selectChatModels`` filter.
        Empty string means "use whatever model the user has selected
        in the Chat picker."
        """
        if self._writer is None:
            await self.connect()
        assert self._writer is not None

        req_id = self._next_id
        self._next_id += 1
        fut: asyncio.Future[str] = asyncio.get_event_loop().create_future()
        self._pending[req_id] = fut
        self._chunks[req_id] = []

        payload = {
            "type": "lm_request",
            "id": req_id,
            "node": node,
            "messages": messages,
            "model_hint": model_hint,
            "temperature": temperature,
        }
        await self._send(payload)
        try:
            return await fut
        finally:
            self._pending.pop(req_id, None)
            self._chunks.pop(req_id, None)

    async def clarify(self, questions: dict[str, Any]) -> dict[str, Any]:
        """Pause for human-in-the-loop clarify answers. The extension
        renders one input modal per question (pre-filled with the
        agent's suggested default) and POSTs the answers back.

        Returns the answers dict on success. Raises ``BridgeError`` if
        the user cancelled (e.g. pressed Esc) or the bridge dropped —
        the engine's caller should catch that and fall back to using
        the per-question defaults rather than crashing the whole quest.
        """
        if self._writer is None:
            await self.connect()
        req_id = self._next_id
        self._next_id += 1
        fut: asyncio.Future[Any] = asyncio.get_event_loop().create_future()
        self._pending[req_id] = fut
        await self._send({
            "type": "clarify_request",
            "id": req_id,
            "questions": questions,
        })
        try:
            result = await fut
        finally:
            self._pending.pop(req_id, None)
        if not isinstance(result, dict):
            raise BridgeError(f"clarify response was not a dict: {result!r}")
        return result

    async def emit_event(self, event: str, **fields: Any) -> None:
        """Push a one-way ``quest_event`` to the extension for UI
        rendering. Best-effort: bridge failures here are logged but
        not raised (an offline UI shouldn't crash the engine)."""
        try:
            await self._send({"type": "quest_event", "event": event, **fields})
        except BridgeError as e:
            _log.info("emit_event(%s) failed: %s", event, e)

    async def _send(self, obj: dict[str, Any]) -> None:
        assert self._writer is not None
        line = (json.dumps(obj) + "\n").encode("utf-8")
        async with self._lock:
            try:
                self._writer.write(line)
                await self._writer.drain()
            except (ConnectionError, OSError) as e:
                raise BridgeError(f"bridge write failed: {e}") from e

    async def _read_loop(self) -> None:
        """Background task: read one JSON line at a time and route
        each ``lm_chunk`` / ``lm_done`` / ``lm_error`` to the matching
        pending future."""
        assert self._reader is not None
        try:
            while True:
                line = await self._reader.readline()
                if not line:  # EOF — extension closed the socket
                    break
                try:
                    msg = json.loads(line.decode("utf-8"))
                except json.JSONDecodeError as e:
                    _log.warning("bridge: malformed message: %s", e)
                    continue
                self._dispatch(msg)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            _log.warning("bridge read loop error: %s", e)
        finally:
            # Reader stopped — drop any in-flight requests.
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(BridgeError("bridge connection dropped"))
            self._pending.clear()

    def _dispatch(self, msg: dict[str, Any]) -> None:
        mtype = msg.get("type")
        if mtype == "lm_chunk":
            req_id = int(msg.get("id", 0))
            self._chunks.setdefault(req_id, []).append(str(msg.get("delta", "")))
        elif mtype == "lm_done":
            req_id = int(msg.get("id", 0))
            fut = self._pending.get(req_id)
            if fut is None or fut.done():
                return
            content = msg.get("content")
            if not isinstance(content, str):
                # Fall back to chunk reassembly when the extension
                # streamed and didn't repeat the final content.
                content = "".join(self._chunks.get(req_id, []))
            fut.set_result(content)
        elif mtype == "lm_error":
            req_id = int(msg.get("id", 0))
            fut = self._pending.get(req_id)
            if fut is None or fut.done():
                return
            err = str(msg.get("error", "unknown bridge error"))
            fut.set_exception(BridgeError(err))
        elif mtype == "clarify_response":
            req_id = int(msg.get("id", 0))
            fut = self._pending.get(req_id)
            if fut is None or fut.done():
                return
            answers = msg.get("answers")
            if not isinstance(answers, dict):
                fut.set_exception(BridgeError(
                    f"clarify_response missing 'answers' dict: {msg!r}"
                ))
                return
            fut.set_result(answers)
        elif mtype == "clarify_cancelled":
            req_id = int(msg.get("id", 0))
            fut = self._pending.get(req_id)
            if fut is None or fut.done():
                return
            fut.set_exception(BridgeError(
                "user cancelled the clarify panel"
            ))
        else:
            _log.info("bridge: ignoring message type=%r", mtype)


async def _open_ipc_connection(
    socket_path: str,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """Connect to the extension's PersistentBridge over the platform's
    IPC channel and return the same ``(reader, writer)`` shape
    :func:`asyncio.open_connection` does, so :class:`VSCodeBridgeClient`
    doesn't have to special-case the transport elsewhere.

    POSIX → Unix-domain socket via :func:`asyncio.open_unix_connection`.

    Windows → named pipe via :meth:`ProactorEventLoop.create_pipe_connection`
    wrapped in a :class:`StreamReaderProtocol` adapter so the caller
    still sees the high-level Stream API. Requires the proactor event
    loop (the default on Python 3.8+ Windows).
    """
    import sys

    if not sys.platform.startswith("win"):
        try:
            return await asyncio.open_unix_connection(socket_path)
        except OSError as e:
            raise BridgeError(
                f"could not connect to VSCode persistent bridge at "
                f"{socket_path}: {e}"
            ) from e

    # Windows named pipe — ProactorEventLoop's create_pipe_connection
    # returns a (transport, protocol) tuple; wrap it as a StreamReader
    # / StreamWriter pair manually because there's no
    # ``open_pipe_connection`` high-level helper in stdlib.
    loop = asyncio.get_event_loop()
    if not hasattr(loop, "create_pipe_connection"):
        raise BridgeError(
            "Windows named-pipe transport requires asyncio.ProactorEventLoop "
            "(the default on Python 3.8+). Got "
            f"{type(loop).__name__}.",
        )
    reader = asyncio.StreamReader(loop=loop)
    protocol = asyncio.StreamReaderProtocol(reader, loop=loop)
    try:
        transport, _ = await loop.create_pipe_connection(  # type: ignore[attr-defined]
            lambda: protocol, socket_path,
        )
    except OSError as e:
        raise BridgeError(
            f"could not connect to VSCode persistent bridge at "
            f"{socket_path}: {e}"
        ) from e
    writer = asyncio.StreamWriter(transport, protocol, reader, loop)
    return reader, writer
