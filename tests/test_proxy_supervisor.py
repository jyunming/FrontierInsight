"""ProxySupervisor concurrency: per-key locking (PR-12).

The supervisor reference-counts localhost proxy subprocesses. It used a single
global lock held across the up-to-60s ``_spawn`` warmup, so in a --fleet one
provider's proxy startup blocked every other provider's acquire. Per-key locks
let different providers warm up concurrently while still preventing a
double-spawn of the same proxy.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from core.provider import ProxySupervisor, _ProxyHandle


class _FakeProc:
    def __init__(self) -> None:
        self.terminated = False
        self.killed = False

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout=None):  # noqa: ANN001
        return 0

    def kill(self) -> None:
        self.killed = True


@pytest.mark.asyncio
async def test_different_providers_spawn_concurrently():
    """Two DIFFERENT providers must be able to spawn at the same time. The
    barrier requires both _spawn calls to be in flight simultaneously — if the
    old global lock still serialized them, the first would block forever at the
    barrier and this test would fail with BrokenBarrierError."""
    sup = ProxySupervisor()
    barrier = threading.Barrier(2, timeout=5)

    def fake_spawn(key: str) -> _ProxyHandle:
        barrier.wait()  # proves both spawns overlap
        return _ProxyHandle(name="p", port=1234, proc=_FakeProc())

    sup._spawn = fake_spawn  # type: ignore[method-assign]

    a, b = await asyncio.gather(
        sup.acquire("claude_code"),
        sup.acquire("github_copilot_cli"),
    )
    assert a is not b
    assert a.refcount == 1 and b.refcount == 1


@pytest.mark.asyncio
async def test_same_provider_spawns_once_and_shares_handle():
    """Concurrent acquires of the SAME provider must spawn exactly one proxy
    and share the handle (refcount 2) — no double-spawn."""
    sup = ProxySupervisor()
    spawns: list[str] = []

    def fake_spawn(key: str) -> _ProxyHandle:
        spawns.append(key)
        return _ProxyHandle(name="p", port=4321, proc=_FakeProc())

    sup._spawn = fake_spawn  # type: ignore[method-assign]

    a, b = await asyncio.gather(
        sup.acquire("claude_code"),
        sup.acquire("claude_code"),
    )
    assert a is b
    assert a.refcount == 2
    assert spawns == ["claude_code"], "the proxy must be spawned exactly once"


@pytest.mark.asyncio
async def test_release_terminates_at_zero_and_respawns_after():
    sup = ProxySupervisor()
    procs: list[_FakeProc] = []

    def fake_spawn(key: str) -> _ProxyHandle:
        proc = _FakeProc()
        procs.append(proc)
        return _ProxyHandle(name="p", port=5555, proc=proc)

    sup._spawn = fake_spawn  # type: ignore[method-assign]

    await sup.acquire("claude_code")
    await sup.release("claude_code")
    assert procs[0].terminated is True

    # Handle was removed at refcount 0, so the next acquire respawns.
    await sup.acquire("claude_code")
    assert len(procs) == 2


@pytest.mark.asyncio
async def test_refcount_holds_proxy_until_last_release():
    sup = ProxySupervisor()
    procs: list[_FakeProc] = []
    sup._spawn = lambda key: (  # type: ignore[method-assign]
        procs.append(_FakeProc()) or _ProxyHandle(name="p", port=6, proc=procs[-1])
    )

    await sup.acquire("claude_code")
    await sup.acquire("claude_code")  # refcount 2, still one proc
    await sup.release("claude_code")
    assert procs[0].terminated is False, "still one holder — must not terminate"
    await sup.release("claude_code")
    assert procs[0].terminated is True


@pytest.mark.asyncio
async def test_alias_shares_canonical_handle():
    """github_copilot_vscode canonicalizes to github_copilot_cli — the two
    aliases must share one proxy, not spawn two."""
    sup = ProxySupervisor()
    spawns: list[str] = []
    sup._spawn = lambda key: (  # type: ignore[method-assign]
        spawns.append(key) or _ProxyHandle(name="p", port=7, proc=_FakeProc())
    )

    a = await sup.acquire("github_copilot_cli")
    b = await sup.acquire("github_copilot_vscode")
    assert a is b
    assert a.refcount == 2
    assert len(spawns) == 1
