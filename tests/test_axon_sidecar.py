"""Tests for ``core.axon_sidecar`` — the helper that keeps an Axon
API process warm across FI quests.

The sidecar is supposed to be **idempotent**: when an Axon API is
already listening on the configured host/port, ``ensure_axon_up()``
must NOT spawn a second subprocess. We pin that contract here with
a stub for ``urllib.request.urlopen`` plus a spy on
``subprocess.Popen``.
"""
from __future__ import annotations

import io
import socket
from unittest import mock

import pytest

from core import axon_sidecar


class _FakeResponse:
    """Minimal context-manager stand-in for ``urlopen`` return value."""
    def __init__(self, status: int = 200) -> None:
        self.status = status
    def __enter__(self) -> "_FakeResponse":
        return self
    def __exit__(self, *_args: object) -> None:
        pass


def test_axon_status_reports_running_and_ready_on_200(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both /health/live and /health/ready return 200 → running=True,
    ready=True, error=None."""
    monkeypatch.setattr(
        axon_sidecar.urllib.request,
        "urlopen",
        lambda url, timeout=None: _FakeResponse(status=200),
    )
    status = axon_sidecar.axon_status(host="127.0.0.1", port=8765)
    assert status["running"] is True
    assert status["ready"] is True
    assert status["url"] == "http://127.0.0.1:8765"
    assert status["error"] is None


def test_axon_status_running_but_not_ready_when_ready_probe_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sidecar that's accepting connections but hasn't finished
    loading its embedding model returns 200 on /health/live and
    something non-2xx on /health/ready. axon_status must split
    those signals."""
    calls: list[str] = []

    def fake_urlopen(url, timeout=None):
        calls.append(url)
        if url.endswith("/health/live"):
            return _FakeResponse(status=200)
        # ready probe — simulate "still loading" as a 503-like raise
        raise axon_sidecar.urllib.error.URLError("brain not ready")

    monkeypatch.setattr(axon_sidecar.urllib.request, "urlopen", fake_urlopen)
    status = axon_sidecar.axon_status()
    assert status["running"] is True, "live probe succeeded"
    assert status["ready"] is False, "ready probe failed"
    # We probed BOTH endpoints, not just the live one.
    assert any("live" in u for u in calls)
    assert any("ready" in u for u in calls)


def test_axon_status_reports_not_running_on_refused_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No Axon API listening → ConnectionRefusedError-equivalent →
    running=False with an error message."""
    def fake_urlopen(_url, timeout=None):
        raise axon_sidecar.urllib.error.URLError("Connection refused")
    monkeypatch.setattr(axon_sidecar.urllib.request, "urlopen", fake_urlopen)

    status = axon_sidecar.axon_status()
    assert status["running"] is False
    assert status["ready"] is False
    assert status["error"] is not None
    assert "refused" in status["error"].lower() or "Connection" in status["error"]


def test_axon_status_handles_socket_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A slow / hung process gets caught as a socket.timeout — the
    handler classifies the entry as not-running rather than letting
    the exception escape and crash whatever code probed it."""
    def fake_urlopen(_url, timeout=None):
        raise socket.timeout("read timed out")
    monkeypatch.setattr(axon_sidecar.urllib.request, "urlopen", fake_urlopen)

    status = axon_sidecar.axon_status()
    assert status["running"] is False
    # ``error`` aggregates every candidate we tried, so the message
    # names both the endpoint and the reason rather than the bare
    # reason alone — that's what makes a wrong-port setup diagnosable.
    assert "timeout" in (status["error"] or "")


def test_ensure_axon_up_short_circuits_when_already_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Idempotence contract: when ``axon_status`` reports running=True,
    ``ensure_axon_up`` must NOT call ``subprocess.Popen``. Multiple
    FI invocations (CLI quest, web serve, CLI quest again) would
    otherwise stack duplicate Axon processes on the same port and
    fail."""
    monkeypatch.setattr(
        axon_sidecar.urllib.request,
        "urlopen",
        lambda url, timeout=None: _FakeResponse(status=200),
    )
    popen_called: list[object] = []

    def spy_popen(*args, **kwargs):  # type: ignore[no-untyped-def]
        popen_called.append((args, kwargs))
        raise AssertionError("Popen must NOT be called when sidecar is up")

    monkeypatch.setattr(axon_sidecar.subprocess, "Popen", spy_popen)
    log_lines: list[str] = []
    status = axon_sidecar.ensure_axon_up(log=log_lines.append)

    assert status["running"] is True
    assert popen_called == [], "subprocess.Popen was called despite warm sidecar"
    assert any("already up" in line for line in log_lines)


def test_ensure_axon_up_spawns_when_sidecar_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When axon_status reports running=False, ensure_axon_up spawns
    ``python -m axon.api`` and forwards the host/port via env vars
    so the spawned process binds where we expect it to."""
    probe_calls = [0]

    def fake_urlopen(_url, timeout=None):
        probe_calls[0] += 1
        # Never become reachable — exercises the timeout path.
        raise axon_sidecar.urllib.error.URLError("refused")

    monkeypatch.setattr(axon_sidecar.urllib.request, "urlopen", fake_urlopen)

    spawned: list[dict] = []

    class _FakeProc:
        pid = 12345

    def fake_popen(argv, env=None, **kwargs):  # type: ignore[no-untyped-def]
        spawned.append({"argv": list(argv), "env": dict(env or {}), "kwargs": kwargs})
        return _FakeProc()

    monkeypatch.setattr(axon_sidecar.subprocess, "Popen", fake_popen)
    log_lines: list[str] = []
    status = axon_sidecar.ensure_axon_up(
        host="127.0.0.1", port=8765,
        boot_timeout=0.05, poll_interval=0.01,
        log=log_lines.append,
    )

    # Spawned exactly once with the python -m axon.api invocation.
    assert len(spawned) == 1, f"expected one spawn, got {len(spawned)}"
    argv = spawned[0]["argv"]
    assert argv[-2:] == ["-m", "axon.api"], f"argv: {argv}"
    env = spawned[0]["env"]
    assert env["AXON_HOST"] == "127.0.0.1"
    assert env["AXON_PORT"] == "8765"

    # boot_timeout passed without the sidecar coming up → returned a
    # not-running status with a timeout error, not a hard exception.
    assert status["running"] is False
    assert "boot timeout" in (status["error"] or "")


def _spawn_env_with(
    monkeypatch: pytest.MonkeyPatch, **kwargs: object,
) -> dict[str, str]:
    """Drive ``ensure_axon_up`` with a down sidecar + fake Popen and
    return the env dict the spawned process would have received."""
    monkeypatch.setattr(
        axon_sidecar.urllib.request, "urlopen",
        lambda _url, timeout=None: (_ for _ in ()).throw(
            axon_sidecar.urllib.error.URLError("refused"),
        ),
    )
    captured: dict[str, dict] = {}

    class _FakeProc:
        pid = 1

    def fake_popen(argv, env=None, **kw):  # type: ignore[no-untyped-def]
        captured["env"] = dict(env or {})
        return _FakeProc()

    monkeypatch.setattr(axon_sidecar.subprocess, "Popen", fake_popen)
    axon_sidecar.ensure_axon_up(
        host="127.0.0.1", port=8765,
        boot_timeout=0.02, poll_interval=0.01,
        log=lambda _m: None, **kwargs,  # type: ignore[arg-type]
    )
    return captured["env"]


def test_ensure_axon_up_injects_offline_env_from_explicit_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``offline=True`` + ``models_dir`` → the spawned sidecar gets the HF
    offline env vars + HF_HOME so it loads embedding weights locally with
    no network (the air-gapped path)."""
    monkeypatch.delenv("FI_OFFLINE", raising=False)
    monkeypatch.delenv("FI_MODELS_DIR", raising=False)
    env = _spawn_env_with(monkeypatch, offline=True, models_dir=r"D:\fi-models")
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["TRANSFORMERS_OFFLINE"] == "1"
    assert env["HF_HOME"] == r"D:\fi-models"


def test_ensure_axon_up_resolves_offline_env_from_FI_envvars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No explicit args, but ``FI_OFFLINE`` / ``FI_MODELS_DIR`` set →
    resolved from env (the cross-process knobs that also seed
    KnowledgeConfig). The sidecar is launched before per-quest YAML, so
    env is the source of truth for it."""
    monkeypatch.setenv("FI_OFFLINE", "1")
    monkeypatch.setenv("FI_MODELS_DIR", r"E:\models")
    env = _spawn_env_with(monkeypatch)
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["TRANSFORMERS_OFFLINE"] == "1"
    assert env["HF_HOME"] == r"E:\models"


def test_ensure_axon_up_no_offline_leaves_hf_env_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default (online) path must NOT inject HF offline vars — a normal
    machine keeps fetching from huggingface.co as before."""
    for _k in ("FI_OFFLINE", "FI_MODELS_DIR", "HF_HUB_OFFLINE",
               "TRANSFORMERS_OFFLINE", "HF_HOME"):
        monkeypatch.delenv(_k, raising=False)
    env = _spawn_env_with(monkeypatch)
    assert "HF_HUB_OFFLINE" not in env
    assert "TRANSFORMERS_OFFLINE" not in env


def test_ensure_axon_up_expands_tilde_in_models_dir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If models_dir starts with ~, ensure_axon_up expands it to an absolute
    path before injecting HF_HOME."""
    monkeypatch.delenv("FI_OFFLINE", raising=False)
    monkeypatch.delenv("FI_MODELS_DIR", raising=False)
    env = _spawn_env_with(monkeypatch, offline=True, models_dir="~/fi-models")
    import os
    expected = os.path.expanduser("~/fi-models")
    assert env["HF_HOME"] == expected

