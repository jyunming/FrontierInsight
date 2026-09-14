"""Images in chat messages.

The visual check sends page screenshots as OpenAI content parts. Every
transport either delivers them to the model or says plainly that it cannot,
so the check can fall back to its measurements instead of asking a model
that never saw the page.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.config import ProviderConfig
from core.provider import (
    _CLI_SPECS,
    ImageInputUnsupported,
    LLMClient,
    ResolvedEndpoint,
    _messages_to_text,
    _run_cli,
    image_part,
    resolve_endpoint,
)

PNG = b"\x89PNG\r\n\x1a\n-not-a-real-image"


def _message(text: str, *images: bytes) -> dict:
    return {"role": "user", "content": [{"type": "text", "text": text}, *(image_part(i) for i in images)]}


def _http_returning(content: str) -> MagicMock:
    response = MagicMock()
    response.json = MagicMock(return_value={"choices": [{"message": {"content": content}}]})
    response.status_code = 200
    response.raise_for_status = MagicMock()
    http = MagicMock()
    http.post = AsyncMock(return_value=response)
    return http


def test_flattening_for_a_cli_keeps_the_text_and_leaves_the_images_out() -> None:
    assert _messages_to_text([_message("Check this page.", PNG)]) == "Check this page."


@pytest.mark.asyncio
async def test_http_sends_the_image_parts_unchanged() -> None:
    http = _http_returning("fine")
    client = LLMClient(ResolvedEndpoint(base_url="https://x/v1", model="m", api_key="k"), http=http)
    message = _message("Check this page.", PNG)
    assert await client.chat([message], node="visual_check") == "fine"
    assert http.post.call_args.kwargs["json"]["messages"][0]["content"] == message["content"]


@pytest.mark.asyncio
async def test_the_prompt_cap_trims_text_and_keeps_every_image() -> None:
    http = _http_returning("fine")
    client = LLMClient(
        ResolvedEndpoint(base_url="https://x/v1", model="m", api_key="k"), http=http, max_prompt_chars=1000,
    )
    message = _message("HEAD" + "m" * 50_000 + "TAIL", PNG, PNG)
    await client.chat([message], node="visual_check")
    sent = http.post.call_args.kwargs["json"]["messages"][0]["content"]
    text = [part["text"] for part in sent if part["type"] == "text"]
    assert len(text) == 1 and len(text[0]) <= 1000 and text[0].startswith("HEAD") and text[0].endswith("TAIL")
    assert [part for part in sent if part["type"] == "image_url"] == message["content"][1:]


def test_the_usage_estimate_counts_text_and_images() -> None:
    client = LLMClient(ResolvedEndpoint(base_url="https://x/v1", model="m", api_key="k"), http=MagicMock())
    client.last_usage = None
    client._fill_usage_estimate_if_missing([_message("x" * 400, PNG, PNG)], "ok")
    assert client.last_usage["prompt_tokens"] == 400 // client._CHARS_PER_TOKEN + 2 * 1000


@pytest.mark.asyncio
async def test_claude_gets_the_images_inline_in_a_stream_json_turn() -> None:
    written: list[bytes] = []
    proc = AsyncMock()
    lines = iter([
        b'{"type":"stream_event","event":{"type":"content_block_delta","delta":{"type":"text_delta","text":"Red square"}}}\n',
        b"",
    ])

    async def readline() -> bytes:
        return next(lines, b"")

    proc.stdout = AsyncMock()
    proc.stdout.readline = readline
    proc.stdin = AsyncMock()
    proc.stdin.write = written.append
    proc.stdin.drain = AsyncMock(return_value=None)
    proc.stdin.close = lambda: None
    proc.stderr = AsyncMock()
    proc.stderr.read = AsyncMock(return_value=b"")
    proc.wait = AsyncMock(return_value=0)
    proc.returncode = 0
    proc.kill = lambda: None
    client = LLMClient(resolve_endpoint(ProviderConfig(name="claude_cli")))
    try:
        with patch("core.provider.shutil.which", return_value="/usr/bin/claude"), \
             patch("core.provider.asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc)) as spawn:
            answer = await client.chat([_message("Name the shapes.", PNG)])
    finally:
        await client.aclose()
    assert answer == "Red square"
    args = spawn.call_args.args
    assert args[args.index("--input-format") + 1] == "stream-json"
    turn = json.loads(b"".join(written).decode("utf-8"))
    content = turn["message"]["content"]
    assert content[0]["type"] == "image" and content[0]["source"]["media_type"] == "image/png"
    assert content[0]["source"]["data"] == image_part(PNG)["image_url"]["url"].split(",", 1)[1]
    assert content[-1] == {"type": "text", "text": "Name the shapes."}


@pytest.mark.asyncio
async def test_a_cli_with_an_image_flag_reads_temp_files_that_are_removed_afterwards() -> None:
    spec = dataclasses.replace(_CLI_SPECS["codex_cli"], image_input="file_flag", image_flag="-i")
    seen: dict[str, bytes] = {}
    paths: list[Path] = []

    async def spawn(*argv, **kwargs):  # type: ignore[no-untyped-def]
        image = Path(argv[list(argv).index("-i") + 1])
        paths.append(image)
        seen["image"] = image.read_bytes()
        Path(argv[list(argv).index("--output-last-message") + 1]).write_text("done", encoding="utf-8")
        proc = AsyncMock()
        proc.communicate = AsyncMock(return_value=(b"", b""))
        proc.returncode = 0
        proc.wait = AsyncMock(return_value=0)
        proc.kill = lambda: None
        return proc

    with patch("core.provider.shutil.which", return_value="/usr/bin/codex"), \
         patch("core.provider.asyncio.create_subprocess_exec", new=spawn):
        answer = await _run_cli(spec, "Name the shapes.", images=[("image/png", PNG)])
    assert answer == "done"
    assert seen["image"] == PNG
    assert paths and not paths[0].exists()


@pytest.mark.asyncio
async def test_a_cli_that_cannot_take_images_says_so_without_running() -> None:
    client = LLMClient(resolve_endpoint(ProviderConfig(name="gemini_cli")))
    try:
        with patch("core.provider.shutil.which", return_value="/usr/bin/gemini"), \
             patch("core.provider.asyncio.create_subprocess_exec", new=AsyncMock()) as spawn:
            with pytest.raises(ImageInputUnsupported):
                await client.chat([_message("Name the shapes.", PNG)])
    finally:
        await client.aclose()
    spawn.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_vscode_bridge_that_cannot_send_images_reports_it_as_unsupported() -> None:
    from core.vscode_bridge import BridgeError

    endpoint = dataclasses.replace(
        ResolvedEndpoint(base_url="http://127.0.0.1:1", model="m", api_key="k"), transport="vscode_bridge",
    )
    client = LLMClient(endpoint)
    client._bridge = MagicMock()
    client._bridge.chat = AsyncMock(side_effect=BridgeError(
        "this VS Code version cannot send images to a language model (LanguageModelDataPart.image is missing)",
    ))
    with pytest.raises(ImageInputUnsupported):
        await client.chat([_message("Name the shapes.", PNG)])
    client._bridge.chat = AsyncMock(side_effect=BridgeError("no language model is available in this VSCode window"))
    with pytest.raises(BridgeError):
        await client.chat([{"role": "user", "content": "text only"}])
