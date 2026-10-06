"""A VS Code extension older than this FI is said once (never stopping the quest); a GPT model's reasoning that reaches VS
Code with no text in it is said once; a silent model is named in the stall error.

No real model is called: a mock bridge server stands in for the extension; the TypeScript helpers are compiled with node.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from core import vscode_bridge as vb
from core.provider import LLMClient, ResolvedEndpoint
from tests.test_engine_smoke import smoke_config  # noqa: F401 -- the fixture
from tests.test_thinking_capture import _compile_lm_messages
from tests.test_thinking_capture_transports import _engine_with
from tests.test_vscode_bridge import _MockBridgeServer

EXT = Path(__file__).resolve().parent.parent / "vscode-frontier-insight"


def test_the_protocol_the_extension_sends_is_the_one_this_fi_expects() -> None:
    src = (EXT / "src" / "lm-messages.ts").read_text(encoding="utf-8")
    sent = int(re.search(r"export const BRIDGE_PROTOCOL = (\d+);", src).group(1))
    assert sent == vb.REQUIRED_BRIDGE_PROTOCOL
    for name in ("bridge.ts", "persistent-bridge.ts"):
        text = (EXT / "src" / name).read_text(encoding="utf-8")
        assert "protocol: BRIDGE_PROTOCOL" in text or "BRIDGE_PROTOCOL)" in text, name


def test_older_and_current_extensions_are_told_apart() -> None:
    assert vb.extension_is_older(0)
    assert not vb.extension_is_older(vb.REQUIRED_BRIDGE_PROTOCOL)
    assert not vb.extension_is_older(None)  # nothing answered yet: nothing known
    assert "reload the window" in vb.EXTENSION_OLDER_NOTE and ".vsix" in vb.EXTENSION_OLDER_NOTE


async def _run(smoke_config, reply: dict, *, save: bool = True, nodes=("design", "analyze")) -> str:  # noqa: ANN001, F811
    smoke_config.output.save_thinking = save
    server = _MockBridgeServer()

    def handler(msg: dict, w) -> list[dict]:  # noqa: ANN001
        if msg["type"] != "lm_request":
            return []
        return [{"type": "lm_done", "id": msg["id"], "content": "Answer", **reply}]

    port = await server.start(handler)
    client = LLMClient(ResolvedEndpoint(base_url="", model="(VSCode chat default)", api_key="not-needed",
                                        transport="vscode_bridge", vscode_bridge_port=port,
                                        provider_name="vscode_extension"))
    engine = _engine_with(smoke_config, client)
    try:
        for node in nodes:
            assert await engine._chat("prompt", node=node) == "Answer"
    finally:
        await client.aclose()
        await server.stop()
    for h in engine._log.handlers:
        h.flush()
    return (engine.fi_dir / "run.log").read_text(encoding="utf-8")


@pytest.mark.parametrize("save", [True, False])
@pytest.mark.asyncio
async def test_an_older_extension_is_said_once_and_the_quest_goes_on(smoke_config, save: bool) -> None:  # noqa: ANN001, F811
    text = await _run(smoke_config, {}, save=save, nodes=("design", "analyze", "write"))
    assert text.count(vb.EXTENSION_OLDER_NOTE) == 1


@pytest.mark.asyncio
async def test_a_current_extension_says_nothing(smoke_config) -> None:  # noqa: ANN001, F811
    text = await _run(smoke_config, {"protocol": vb.REQUIRED_BRIDGE_PROTOCOL, "thinking": "reasoned"})
    assert "older than this FI" not in text


@pytest.mark.asyncio
async def test_reasoning_that_came_with_no_text_is_said_once_and_names_the_route(smoke_config) -> None:  # noqa: ANN001, F811
    text = await _run(smoke_config, {"protocol": 1, "thinking_parts_empty": 3, "served_model": {"id": "gpt-5.4"}},
                      nodes=("design", "analyze", "write"))
    assert text.count("returned no reasoning through VS Code") == 1
    assert "encrypted only" in text and "codex_cli" in text
    assert "returned no reasoning for this step" not in text


@pytest.mark.asyncio
async def test_a_vscode_model_with_no_reasoning_is_said_once_for_the_model(smoke_config) -> None:  # noqa: ANN001, F811
    text = await _run(smoke_config, {"protocol": 1, "served_model": {"id": "some-model"}},
                      nodes=("design", "analyze", "write"))
    assert text.count("returned no reasoning through VS Code") == 1
    assert "not every model in it returns its reasoning" in text


@pytest.mark.asyncio
async def test_the_client_keeps_the_protocol_and_empty_parts_of_its_own_call() -> None:
    server = _MockBridgeServer()

    def handler(msg: dict, w) -> list[dict]:  # noqa: ANN001
        if msg["type"] != "lm_request":
            return []
        return [{"type": "lm_done", "id": msg["id"], "content": "x", "protocol": 7, "thinking_parts_empty": 2}]

    port = await server.start(handler)
    client = vb.VSCodeBridgeClient(port=port)
    try:
        await client.chat([{"role": "user", "content": "hi"}])
        assert vb.LAST_BRIDGE_PROTOCOL.get() == 7 and vb.LAST_BRIDGE_EMPTY_PARTS.get() == 2
        assert client.extension_protocol == 7
    finally:
        await client.aclose()
        await server.stop()


def test_the_stall_error_names_who_went_silent_and_the_extension_counts_empty_parts(tmp_path: Path) -> None:
    node = shutil.which("node")
    compiled = _compile_lm_messages(tmp_path)
    script = tmp_path / "t.js"
    script.write_text(
        "const m = require(%s);\n"
        "const c = new m.ThinkingCollector(); c.add(''); c.add('a'); c.add('');\n"
        "console.log(JSON.stringify({stall: m.stallMessage({vendor:'ollama-models', id:'gemma4'}, 180, 'received 0 chunks'),"
        " bare: m.stallMessage(undefined, 180, 'x'), empty: c.emptyParts, total: c.total, proto: m.BRIDGE_PROTOCOL}));\n"
        % json.dumps(str(compiled)), encoding="utf-8")
    out = json.loads(subprocess.run([node, str(script)], capture_output=True, text=True, timeout=60).stdout)
    assert out["stall"].startswith("bridge stalled: no part from ollama-models/gemma4 for 180 s")
    assert out["bare"] == "bridge stalled: no part for 180 s (x)"
    assert out["empty"] == 2 and out["total"] == 1 and out["proto"] == vb.REQUIRED_BRIDGE_PROTOCOL
