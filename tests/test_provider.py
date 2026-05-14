"""Direct unit tests for `core.provider`.

These do **not** spawn real proxies, do **not** call real LLM APIs, and do
**not** open real sockets to a model server. `subprocess.Popen`,
`httpx.AsyncClient`, and `_wait_for_openai_endpoint` are mocked.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.config import ProviderConfig
from core.provider import (
    _DIRECT_DEFAULTS,
    _PROXY_PROVIDERS,
    LLMClient,
    ProxySupervisor,
    ResolvedEndpoint,
    resolve_endpoint,
    resolve_endpoint_async,
)


# ---------------------------------------------------------------------------
# resolve_endpoint / resolve_endpoint_async — direct providers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(_DIRECT_DEFAULTS.keys()))
def test_direct_provider_resolves_with_env_fallback(name: str, monkeypatch):
    """Every direct provider resolves; missing env-var key falls back to
    `"not-needed"` (so keyless servers like ollama/vllm work transparently)."""
    defaults = _DIRECT_DEFAULTS[name]
    if defaults["api_key_env"]:
        monkeypatch.delenv(defaults["api_key_env"], raising=False)

    cfg = ProviderConfig(name=name)  # type: ignore[arg-type]
    ep = resolve_endpoint(cfg)

    assert ep.base_url == defaults["base_url"]
    assert ep.model == defaults["model"]
    assert ep.api_key == "not-needed"


def test_direct_provider_resolves_with_env_var_present(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
    ep = resolve_endpoint(ProviderConfig(name="openai"))
    assert ep.api_key == "sk-test-123"


def test_unknown_provider_raises():
    # Bypass pydantic Literal validation by constructing the object then
    # patching the `name` attribute directly.
    cfg = ProviderConfig(name="openai")
    object.__setattr__(cfg, "name", "made_up_provider")
    with pytest.raises(ValueError, match="unknown provider"):
        resolve_endpoint(cfg)


def test_resolve_endpoint_rejects_proxy_synchronously():
    cfg = ProviderConfig(name="claude_code")
    with pytest.raises(RuntimeError, match="async resolution"):
        resolve_endpoint(cfg)


async def test_resolve_endpoint_async_dispatches_direct_without_supervisor(monkeypatch):
    """For direct providers, async dispatch must not touch the supervisor."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    sup = MagicMock(spec=ProxySupervisor)
    sup.acquire = AsyncMock(side_effect=AssertionError("supervisor must not be touched"))

    ep = await resolve_endpoint_async(ProviderConfig(name="openai"), sup)

    assert ep.base_url == _DIRECT_DEFAULTS["openai"]["base_url"]
    sup.acquire.assert_not_called()


# ---------------------------------------------------------------------------
# ProxySupervisor — _spawn error path and ref-counting
# ---------------------------------------------------------------------------


async def test_proxy_provider_raises_clean_error_when_cli_missing():
    """If `npx` / `poetry` is not on PATH, the user gets a `RuntimeError`
    pointing them at the install instructions — not a bare `FileNotFoundError`."""
    sup = ProxySupervisor()
    with patch("core.provider.subprocess.Popen", side_effect=FileNotFoundError("no npx")):
        with pytest.raises(RuntimeError, match="not found on PATH"):
            await sup.acquire("github_copilot_cli")


async def test_proxy_supervisor_rejects_non_proxy_provider():
    sup = ProxySupervisor()
    with pytest.raises(ValueError, match="not a proxy provider"):
        await sup.acquire("openai")


async def test_proxy_refcount_under_concurrent_acquire_release():
    """Hammer `acquire`/`release` under `asyncio.gather` and verify the
    handle goes 0 -> 1 -> 2 -> 1 -> 0 with the process spawned exactly once
    and terminated exactly once."""
    fake_proc = MagicMock()
    fake_proc.terminate = MagicMock()
    fake_proc.wait = MagicMock(return_value=0)
    fake_proc.kill = MagicMock()

    sup = ProxySupervisor()
    with patch("core.provider.subprocess.Popen", return_value=fake_proc) as popen, \
         patch("core.provider._wait_for_openai_endpoint") as wait_endpoint, \
         patch("core.provider._free_port", return_value=54321):

        # 0 -> 1 (spawn)
        h1 = await sup.acquire("github_copilot_cli")
        assert h1.refcount == 1
        assert sup._handles["github_copilot_cli"] is h1

        # 1 -> 2 (no respawn)
        h2 = await sup.acquire("github_copilot_cli")
        assert h2 is h1
        assert h1.refcount == 2

        # 2 -> 1 (still alive)
        await sup.release("github_copilot_cli")
        assert h1.refcount == 1
        assert "github_copilot_cli" in sup._handles
        fake_proc.terminate.assert_not_called()

        # 1 -> 0 (terminate)
        await sup.release("github_copilot_cli")
        assert "github_copilot_cli" not in sup._handles
        fake_proc.terminate.assert_called_once()

        popen.assert_called_once()
        wait_endpoint.assert_called_once_with(54321, timeout_s=60)


async def test_proxy_refcount_concurrent_gather(tmp_path, monkeypatch):
    """Many concurrent `acquire` calls must spawn exactly one process; the
    final ref-count after matching releases must be zero."""
    fake_proc = MagicMock()
    fake_proc.wait = MagicMock(return_value=0)
    monkeypatch.setenv("FI_CLAUDE_CODE_WRAPPER_DIR", str(tmp_path))

    sup = ProxySupervisor()
    with patch("core.provider.subprocess.Popen", return_value=fake_proc) as popen, \
         patch("core.provider._wait_for_openai_endpoint"), \
         patch("core.provider._free_port", return_value=54322):
        handles = await asyncio.gather(*(sup.acquire("claude_code") for _ in range(8)))

        assert popen.call_count == 1
        assert all(h is handles[0] for h in handles)
        assert handles[0].refcount == 8

        await asyncio.gather(*(sup.release("claude_code") for _ in range(8)))
        assert sup._handles == {}
        fake_proc.terminate.assert_called_once()


async def test_proxy_release_unknown_provider_is_noop():
    """Releasing a provider that was never acquired must not raise."""
    sup = ProxySupervisor()
    await sup.release("claude_code")  # should not raise


async def test_proxy_spawn_uses_correct_cli_for_claude_code(tmp_path, monkeypatch):
    """`claude_code` spawns via poetry inside FI_CLAUDE_CODE_WRAPPER_DIR."""
    fake_proc = MagicMock()
    fake_proc.wait = MagicMock(return_value=0)
    monkeypatch.setenv("FI_CLAUDE_CODE_WRAPPER_DIR", str(tmp_path))

    sup = ProxySupervisor()
    with patch("core.provider.subprocess.Popen", return_value=fake_proc) as popen, \
         patch("core.provider._wait_for_openai_endpoint"), \
         patch("core.provider._free_port", return_value=55555):
        await sup.acquire("claude_code")

    args, kwargs = popen.call_args
    cmd = args[0]
    assert cmd[:4] == ["poetry", "run", "python", "main.py"]
    assert cmd[4] == "55555"
    assert kwargs["cwd"] == str(tmp_path)
    assert kwargs["env"]["PORT"] == "55555"


async def test_proxy_spawn_uses_correct_cli_for_copilot():
    """`github_copilot_*` spawns via npx with the rate-limit/wait flags."""
    fake_proc = MagicMock()
    fake_proc.wait = MagicMock(return_value=0)

    sup = ProxySupervisor()
    with patch("core.provider.subprocess.Popen", return_value=fake_proc) as popen, \
         patch("core.provider._wait_for_openai_endpoint"), \
         patch("core.provider._free_port", return_value=55556):
        await sup.acquire("github_copilot_vscode")

    args, kwargs = popen.call_args
    cmd = args[0]
    assert cmd[0] == "npx"
    assert "copilot-api@latest" in cmd
    assert "start" in cmd
    assert "--port" in cmd and "55556" in cmd
    assert "--rate-limit" in cmd and "60" in cmd
    assert "--wait" in cmd
    assert kwargs["cwd"] is None


async def test_resolve_endpoint_async_proxy_uses_supervisor_port(tmp_path, monkeypatch):
    """For a proxy provider, the resolved base_url must point at the local
    port the supervisor allocated, not the configured one."""
    fake_proc = MagicMock()
    fake_proc.wait = MagicMock(return_value=0)
    monkeypatch.setenv("FI_CLAUDE_CODE_WRAPPER_DIR", str(tmp_path))

    sup = ProxySupervisor()
    with patch("core.provider.subprocess.Popen", return_value=fake_proc), \
         patch("core.provider._wait_for_openai_endpoint"), \
         patch("core.provider._free_port", return_value=61234):
        ep = await resolve_endpoint_async(ProviderConfig(name="claude_code"), sup)

    assert ep.base_url == "http://127.0.0.1:61234/v1"
    assert ep.api_key == "not-needed"


# ---------------------------------------------------------------------------
# LLMClient.chat — URL, body, header behavior
# ---------------------------------------------------------------------------


def _fake_response(payload: dict, status_code: int = 200) -> MagicMock:
    r = MagicMock()
    r.status_code = status_code
    r.json = MagicMock(return_value=payload)
    r.raise_for_status = MagicMock()
    return r


async def test_chat_posts_correct_url_and_parses_content():
    ep = ResolvedEndpoint(
        base_url="https://api.example.com/v1/",
        model="m1",
        api_key="sk-real",
    )
    fake_http = MagicMock()
    fake_http.post = AsyncMock(
        return_value=_fake_response({"choices": [{"message": {"content": "hello world"}}]})
    )

    client = LLMClient(ep, http=fake_http)
    out = await client.chat([{"role": "user", "content": "hi"}], temperature=0.7, max_tokens=42)

    assert out == "hello world"
    fake_http.post.assert_awaited_once()
    args, kwargs = fake_http.post.call_args

    # Trailing slash on base_url must be stripped before appending the path.
    assert args[0] == "https://api.example.com/v1/chat/completions"
    body = kwargs["json"]
    assert body["model"] == "m1"
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert body["temperature"] == 0.7
    assert body["max_tokens"] == 42
    headers = kwargs["headers"]
    assert headers["Content-Type"] == "application/json"
    assert headers["Authorization"] == "Bearer sk-real"


async def test_chat_omits_authorization_for_keyless_endpoint():
    """For Ollama/vLLM the api_key resolves to `"not-needed"`. We must NOT
    send a `Bearer not-needed` header — some strict reverse proxies reject it."""
    ep = ResolvedEndpoint(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen2.5-coder:32b",
        api_key="not-needed",
    )
    fake_http = MagicMock()
    fake_http.post = AsyncMock(
        return_value=_fake_response({"choices": [{"message": {"content": "ok"}}]})
    )

    client = LLMClient(ep, http=fake_http)
    await client.chat([{"role": "user", "content": "ping"}])

    headers = fake_http.post.call_args.kwargs["headers"]
    assert "Authorization" not in headers
    assert headers["Content-Type"] == "application/json"


async def test_chat_passes_through_extra_body_fields():
    ep = ResolvedEndpoint(base_url="https://x/v1", model="m", api_key="k")
    fake_http = MagicMock()
    fake_http.post = AsyncMock(
        return_value=_fake_response({"choices": [{"message": {"content": "y"}}]})
    )

    client = LLMClient(ep, http=fake_http)
    await client.chat(
        [{"role": "user", "content": "x"}],
        extra={"response_format": {"type": "json_object"}, "seed": 7},
    )

    body = fake_http.post.call_args.kwargs["json"]
    assert body["response_format"] == {"type": "json_object"}
    assert body["seed"] == 7


async def test_chat_does_not_add_max_tokens_when_unset():
    ep = ResolvedEndpoint(base_url="https://x/v1", model="m", api_key="k")
    fake_http = MagicMock()
    fake_http.post = AsyncMock(
        return_value=_fake_response({"choices": [{"message": {"content": "y"}}]})
    )

    client = LLMClient(ep, http=fake_http)
    await client.chat([{"role": "user", "content": "x"}])

    body = fake_http.post.call_args.kwargs["json"]
    assert "max_tokens" not in body


async def test_chat_propagates_http_error():
    """No try/except around `raise_for_status` is intentional — errors must
    surface to the caller, which retries with tenacity."""
    import httpx

    ep = ResolvedEndpoint(base_url="https://x/v1", model="m", api_key="k")
    bad = MagicMock()
    bad.status_code = 500
    bad.raise_for_status = MagicMock(
        side_effect=httpx.HTTPStatusError("boom", request=MagicMock(), response=MagicMock())
    )
    fake_http = MagicMock()
    fake_http.post = AsyncMock(return_value=bad)

    client = LLMClient(ep, http=fake_http)
    with pytest.raises(httpx.HTTPStatusError):
        await client.chat([{"role": "user", "content": "x"}])


async def test_chat_propagates_cancellation_promptly():
    """When the user Ctrl-C's a quest mid-LLM-call, asyncio cancellation
    must propagate through tenacity's retry wrapper and unwind the
    awaiting ``httpx.AsyncClient.post`` in well under a second — NOT
    get swallowed.

    Scope this test actually covers (limited to asyncio await-chain
    unwinding):
      - ``asyncio.CancelledError`` (BaseException, not Exception) is NOT
        caught by ``retry_if_exception_type((HTTPStatusError, ...))``,
        so tenacity doesn't swallow it.
      - The awaiting code in ``LLMClient.chat`` re-raises cleanly.

    Scope it does NOT cover (would need a real network socket):
      - Whether httpx/anyio actually closes the OS-level TCP socket on
        cancel. ``httpx.MockTransport`` is in-memory; no socket is ever
        opened. Real-socket cancellation is a downstream httpx/anyio
        contract we trust the upstream test suites to enforce.

    Regression modes this guards:
      - Adding ``asyncio.CancelledError`` to the retry predicate.
      - Wrapping the retry block in ``except BaseException`` (NOT
        ``except Exception`` — that's safe because CancelledError is
        BaseException, not Exception, on Python 3.8+)."""
    import httpx

    handler_entered = asyncio.Event()

    async def hanging_handler(request: httpx.Request) -> httpx.Response:
        # Signal that the handler is actually running — i.e., the
        # ``post()`` await is in flight — BEFORE the test cancels.
        # Without this sync point a sleep(0.05) before cancel races
        # the task scheduler on a busy loop and could cancel before
        # post() is even entered, making the test pass for the wrong
        # reason.
        handler_entered.set()
        await asyncio.sleep(60.0)
        return httpx.Response(  # pragma: no cover — should never reach here
            200, json={"choices": [{"message": {"content": "x"}}]},
        )

    transport = httpx.MockTransport(hanging_handler)
    real_http = httpx.AsyncClient(transport=transport, timeout=120.0)
    try:
        ep = ResolvedEndpoint(base_url="http://example.invalid/v1", model="m", api_key="x")
        client = LLMClient(ep, http=real_http)

        task = asyncio.create_task(
            client.chat([{"role": "user", "content": "hi"}]),
        )
        # Wait until the handler is actually running, with a bounded
        # timeout so a regression doesn't make the test wait forever.
        await asyncio.wait_for(handler_entered.wait(), timeout=2.0)

        task.cancel()
        # Bound the cancellation-await too: if a regression swallows
        # the cancel and the handler's 60 s sleep runs to completion,
        # the test would otherwise hang for a full minute on each
        # failure. 2 s is generous for slow CI; in practice this
        # finishes in <10 ms on a developer laptop.
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2.0)
    finally:
        await real_http.aclose()


async def test_chat_error_includes_provider_and_node_in_note():
    """When an LLM call fails, the exception should carry an
    ``add_note`` line tagging the provider, transport, model, and
    engine node so a user reading the traceback knows what was running
    when it blew up — instead of seeing a bare httpx stack trace."""
    import httpx

    ep = ResolvedEndpoint(
        base_url="https://api.openai.com/v1", model="gpt-5",
        api_key="sk-test", provider_name="openai",
    )
    bad = MagicMock()
    bad.status_code = 502
    bad.raise_for_status = MagicMock(
        side_effect=httpx.HTTPStatusError(
            "502 Bad Gateway", request=MagicMock(), response=MagicMock(),
        )
    )
    fake_http = MagicMock()
    fake_http.post = AsyncMock(return_value=bad)
    client = LLMClient(ep, http=fake_http)

    with pytest.raises(httpx.HTTPStatusError) as ei:
        await client.chat(
            [{"role": "user", "content": "x"}], node="implement",
        )

    # ``add_note`` is a Python 3.11+ feature; the project floor is 3.11
    # so the note must be present.
    notes = getattr(ei.value, "__notes__", [])
    assert notes, "exception should have at least one FI note attached"
    full = " ".join(notes)
    assert "[FI]" in full
    assert "provider=openai" in full
    assert "transport=http" in full
    assert "model=gpt-5" in full
    assert "node=implement" in full


async def test_chat_error_note_uses_model_override_when_passed():
    """When the caller passes per-call ``model=...`` (Phase O: per-node
    model routing), the error note should reflect the EFFECTIVE model
    used for the call, not the endpoint default."""
    import httpx

    ep = ResolvedEndpoint(
        base_url="https://api.openai.com/v1", model="gpt-5",
        api_key="sk-test", provider_name="openai",
    )
    fake_http = MagicMock()
    fake_http.post = AsyncMock(side_effect=httpx.TransportError("boom"))
    client = LLMClient(ep, http=fake_http)

    with pytest.raises(httpx.TransportError) as ei:
        await client.chat(
            [{"role": "user", "content": "x"}],
            node="write", model="claude-3-5-sonnet",
        )
    notes = " ".join(getattr(ei.value, "__notes__", []))
    assert "model=claude-3-5-sonnet" in notes, notes
    assert "node=write" in notes


async def test_chat_error_note_omits_node_when_unset():
    """``node`` defaults to ``""``. Empty node shouldn't appear in the
    note as ``node=`` — it should just be omitted."""
    import httpx

    ep = ResolvedEndpoint(
        base_url="https://api.openai.com/v1", model="gpt-5",
        api_key="sk-test", provider_name="openai",
    )
    fake_http = MagicMock()
    fake_http.post = AsyncMock(side_effect=httpx.TransportError("boom"))
    client = LLMClient(ep, http=fake_http)

    with pytest.raises(httpx.TransportError) as ei:
        await client.chat([{"role": "user", "content": "x"}])
    notes = " ".join(getattr(ei.value, "__notes__", []))
    assert "node=" not in notes, f"node= snuck into note: {notes!r}"
    assert "provider=openai" in notes


async def test_chat_cancellation_does_not_get_a_note():
    """The error-context wrapper uses ``except Exception``, NOT
    ``except BaseException``, specifically so ``CancelledError``
    propagates clean and un-noted. This is required for the
    cancellation contract to hold (see comment in
    LLMClient._chat_impl). Regression guard."""
    post_entered = asyncio.Event()

    async def hang(*a, **kw):
        post_entered.set()
        await asyncio.sleep(60.0)
        raise AssertionError("unreachable")  # pragma: no cover

    fake_http = MagicMock()
    fake_http.post = AsyncMock(side_effect=hang)
    ep = ResolvedEndpoint(
        base_url="http://x/v1", model="m", api_key="k", provider_name="openai",
    )
    client = LLMClient(ep, http=fake_http)

    task = asyncio.create_task(
        client.chat([{"role": "user", "content": "hi"}], node="ideate"),
    )
    await asyncio.wait_for(post_entered.wait(), timeout=2.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError) as ei:
        await asyncio.wait_for(task, timeout=2.0)
    # CancelledError must NOT have been augmented with an FI note —
    # the wrapper only catches Exception, and CancelledError is a
    # BaseException.
    notes = getattr(ei.value, "__notes__", [])
    fi_notes = [n for n in notes if "[FI]" in n]
    assert not fi_notes, (
        f"CancelledError was given an FI note ({fi_notes!r}) — the "
        f"chat() try/except must NOT catch BaseException, only Exception."
    )


async def test_chat_does_not_retry_on_cancellation():
    """Tenacity is configured with ``retry_if_exception_type((HTTPStatusError,
    TransportError, ReadTimeout))``. ``asyncio.CancelledError`` is a
    BaseException and intentionally not in that set — so on cancel we
    must hit the underlying ``post`` exactly ONCE, never multiple
    retry attempts. Regression test."""

    post_entered = asyncio.Event()
    call_count = {"n": 0}

    async def hang_then_count(*a, **kw):
        call_count["n"] += 1
        post_entered.set()       # cancel only AFTER post() is in flight
        await asyncio.sleep(60.0)
        raise AssertionError("unreachable")  # pragma: no cover

    fake_http = MagicMock()
    fake_http.post = AsyncMock(side_effect=hang_then_count)

    ep = ResolvedEndpoint(base_url="http://x/v1", model="m", api_key="k")
    client = LLMClient(ep, http=fake_http)

    task = asyncio.create_task(client.chat([{"role": "user", "content": "hi"}]))
    await asyncio.wait_for(post_entered.wait(), timeout=2.0)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=2.0)

    assert call_count["n"] == 1, (
        f"post was called {call_count['n']} times — tenacity is "
        f"retrying CancelledError, which it must not"
    )


async def test_llm_client_does_not_close_external_http():
    """When the caller passes in an `httpx.AsyncClient`, the client owns
    its lifecycle — `LLMClient.aclose` must NOT close it. This is what lets
    N Engines share one connection pool."""
    fake_http = MagicMock()
    fake_http.aclose = AsyncMock()

    client = LLMClient(
        ResolvedEndpoint(base_url="x", model="m", api_key="k"),
        http=fake_http,
    )
    await client.aclose()
    fake_http.aclose.assert_not_called()


async def test_llm_client_closes_owned_http():
    """When LLMClient created the AsyncClient itself, aclose() must close it."""
    with patch("core.provider.httpx.AsyncClient") as ac:
        instance = MagicMock()
        instance.aclose = AsyncMock()
        ac.return_value = instance

        client = LLMClient(ResolvedEndpoint(base_url="x", model="m", api_key="k"))
        await client.aclose()

        instance.aclose.assert_awaited_once()


# ---------------------------------------------------------------------------
# Sanity check on the constants
# ---------------------------------------------------------------------------


def test_proxy_set_matches_known_proxies():
    assert _PROXY_PROVIDERS == {
        "claude_code",
        "github_copilot_cli",
        "github_copilot_vscode",
    }


def test_direct_defaults_have_required_keys():
    for name, d in _DIRECT_DEFAULTS.items():
        assert "base_url" in d, name
        assert "model" in d, name
        assert "api_key_env" in d, name
