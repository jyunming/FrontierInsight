"""Tests for ``core.axon_endpoint`` — finding a sidecar whose port moves.

Axon's API port is resolved at server start (``--port`` > ``AXON_PORT`` >
``config.yaml``'s ``api.port`` > ``8420``), so FI cannot assume one. These
tests pin the discovery contract: what we look at, in what order, and
which inputs are instructions rather than hints.

The regression behind them: FI probed a hardcoded ``127.0.0.1:8000`` (the
pre-8420 default) and reported a healthy sidecar as "not running".
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from core import axon_endpoint as ep
from core import axon_sidecar


class _StubConfig:
    """Minimal stand-in for ``AxonConfig`` — only the fields we read."""

    def __init__(
        self,
        projects_root: str | None = None,
        api_host: str | None = None,
        api_port: int | None = None,
    ) -> None:
        self.projects_root = projects_root
        self.api_host = api_host
        self.api_port = api_port


def _use_config(monkeypatch: pytest.MonkeyPatch, config: Any) -> None:
    """Pin the Axon config ``candidates()`` sees, so these tests don't
    depend on whether Axon is installed on the machine running them."""
    monkeypatch.setattr(ep, "_axon_config", lambda: config)


def _sources(cands: list[ep.Candidate]) -> list[str]:
    return [c.source for c in cands]


def _urls(cands: list[ep.Candidate]) -> list[str]:
    return [c.base_url for c in cands]


# ---------------------------------------------------------------------------
# Bind address vs. connect address
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("wildcard", ["0.0.0.0", "::", "[::]", "*", "", "  "])
def test_wildcard_bind_addresses_become_loopback(wildcard: str) -> None:
    """The lock file records the address the server *bound*, which is
    routinely ``0.0.0.0`` — an address you cannot connect to. Probing it
    verbatim fails against a perfectly healthy sidecar."""
    assert ep.probe_host(wildcard) == "127.0.0.1"


def test_real_host_is_left_alone() -> None:
    assert ep.probe_host("192.168.1.50") == "192.168.1.50"
    assert ep.probe_host(None) == "127.0.0.1"


# ---------------------------------------------------------------------------
# The store lock file
# ---------------------------------------------------------------------------


def test_lock_file_supplies_host_and_port(tmp_path: Path) -> None:
    """The whole point: a running server tells us where it is."""
    (tmp_path / ".axon-api.lock").write_text(
        json.dumps({"host": "0.0.0.0", "port": 9137, "pid": 4242}),
        encoding="utf-8",
    )
    assert ep.lock_file_endpoint(_StubConfig(projects_root=str(tmp_path))) == (
        "127.0.0.1", 9137,
    )


def test_missing_lock_file_is_not_an_error(tmp_path: Path) -> None:
    assert ep.lock_file_endpoint(_StubConfig(projects_root=str(tmp_path))) is None


@pytest.mark.parametrize(
    "content",
    ['{"host": "127.0.0.1"}', "not json at all", '{"host": "x", "port": "abc"}'],
)
def test_malformed_lock_file_is_ignored(tmp_path: Path, content: str) -> None:
    """A half-written or corrupt lock must degrade to "no answer" rather
    than raising out of a health check."""
    (tmp_path / ".axon-api.lock").write_text(content, encoding="utf-8")
    assert ep.lock_file_endpoint(_StubConfig(projects_root=str(tmp_path))) is None


# ---------------------------------------------------------------------------
# Candidate assembly
# ---------------------------------------------------------------------------


def test_candidates_prefer_lock_file_over_config_and_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / ".axon-api.lock").write_text(
        json.dumps({"host": "0.0.0.0", "port": 9137}), encoding="utf-8",
    )
    _use_config(
        monkeypatch,
        _StubConfig(projects_root=str(tmp_path), api_host="127.0.0.1", api_port=8420),
    )

    cands = ep.candidates(env={})
    assert _urls(cands) == [
        "http://127.0.0.1:9137",   # the running server
        "http://127.0.0.1:8420",   # config.yaml
        "http://127.0.0.1:8000",   # pre-8420 Axon
    ]
    assert _sources(cands)[0] == ep.SOURCE_LOCK


def test_stale_axon_port_does_not_hide_a_live_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact shape of the bug this module exists to fix: an
    environment left pointing at the old 8000 must cost one dead probe,
    not the whole discovery."""
    (tmp_path / ".axon-api.lock").write_text(
        json.dumps({"host": "0.0.0.0", "port": 8420}), encoding="utf-8",
    )
    _use_config(monkeypatch, _StubConfig(projects_root=str(tmp_path)))

    cands = ep.candidates(env={"AXON_PORT": "8000"})
    assert _urls(cands)[0] == "http://127.0.0.1:8000"
    assert "http://127.0.0.1:8420" in _urls(cands), "the live sidecar is still reachable"


def test_duplicate_endpoints_are_probed_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lock file and config.yaml normally agree. Probing that endpoint
    twice would double the cost of the common case."""
    (tmp_path / ".axon-api.lock").write_text(
        json.dumps({"host": "127.0.0.1", "port": 8420}), encoding="utf-8",
    )
    _use_config(
        monkeypatch,
        _StubConfig(projects_root=str(tmp_path), api_host="127.0.0.1", api_port=8420),
    )
    urls = _urls(ep.candidates(env={}))
    assert urls.count("http://127.0.0.1:8420") == 1


def test_current_default_is_probed_before_the_legacy_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_config(monkeypatch, None)
    urls = _urls(ep.candidates(env={}))
    assert urls == ["http://127.0.0.1:8420", "http://127.0.0.1:8000"]


def test_no_axon_installed_still_yields_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Axon is an optional dependency; discovery must not need it."""
    _use_config(monkeypatch, None)
    assert len(ep.candidates(env={})) == 2


# ---------------------------------------------------------------------------
# Instructions vs. hints
# ---------------------------------------------------------------------------


def test_api_base_env_replaces_the_candidate_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``AXON_API_BASE`` names one specific Axon. Two instances hold
    different corpora, so falling through to a local one when the named
    one is down would silently change what a quest reads."""
    (tmp_path / ".axon-api.lock").write_text(
        json.dumps({"host": "127.0.0.1", "port": 8420}), encoding="utf-8",
    )
    _use_config(monkeypatch, _StubConfig(projects_root=str(tmp_path)))

    cands = ep.candidates(env={"AXON_API_BASE": "http://remote.box:9000"})
    assert _urls(cands) == ["http://remote.box:9000"]


def test_explicit_arguments_replace_the_candidate_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_config(monkeypatch, None)
    assert _urls(ep.candidates("127.0.0.1", 9999, env={})) == ["http://127.0.0.1:9999"]


def test_api_base_without_a_scheme_or_port_is_still_usable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_config(monkeypatch, None)
    assert _urls(ep.candidates(env={"AXON_API_BASE": "axon.internal"})) == [
        "http://axon.internal:8420",
    ]


def test_unparseable_api_base_falls_back_to_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A typo'd override shouldn't leave the user with zero candidates."""
    _use_config(monkeypatch, None)
    urls = _urls(ep.candidates(env={"AXON_API_BASE": "   "}))
    assert urls == ["http://127.0.0.1:8420", "http://127.0.0.1:8000"]


# ---------------------------------------------------------------------------
# Where we'd start one
# ---------------------------------------------------------------------------


def test_preferred_endpoint_never_suggests_the_legacy_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spawning on 8000 would plant a sidecar current Axon has moved off."""
    _use_config(monkeypatch, None)
    assert ep.preferred_endpoint(env={}) == ("127.0.0.1", ep.DEFAULT_PORT)


def test_preferred_endpoint_ignores_the_lock_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lock describes a server that already exists — there is nothing
    to start there. Prefer the address Axon itself would bind."""
    (tmp_path / ".axon-api.lock").write_text(
        json.dumps({"host": "127.0.0.1", "port": 9137}), encoding="utf-8",
    )
    _use_config(
        monkeypatch,
        _StubConfig(projects_root=str(tmp_path), api_host="127.0.0.1", api_port=8500),
    )
    assert ep.preferred_endpoint(env={}) == ("127.0.0.1", 8500)


def test_preferred_endpoint_honours_config_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_config(monkeypatch, _StubConfig(api_host="127.0.0.1", api_port=8500))
    assert ep.preferred_endpoint(env={}) == ("127.0.0.1", 8500)


# ---------------------------------------------------------------------------
# Telling Axon apart from whatever else is on the port
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [b'{"status": "alive"}', b'{"status": "ok", "project": "default"}'],
)
def test_axon_health_bodies_are_accepted(body: bytes) -> None:
    assert axon_sidecar._looks_like_axon(body) is True


def test_another_service_answering_200_is_rejected() -> None:
    """Axon's own config defaults ``vllm_base_url`` to
    ``localhost:8000/v1`` — the very port FI used to probe. A vLLM server
    answering 200 must not be reported as a warm Axon sidecar."""
    body = b'{"object": "list", "data": [{"id": "gemma4-26b"}]}'
    assert axon_sidecar._looks_like_axon(body) is False


@pytest.mark.parametrize("body", [None, b"", b"not json", b"[1, 2, 3]"])
def test_unreadable_bodies_stay_accepted(body: bytes | None) -> None:
    """An empty or non-JSON body is inconclusive, not wrong. The status
    code already did the real filtering; refusing here would break
    against an older Axon whose payload we haven't seen."""
    assert axon_sidecar._looks_like_axon(body) is True
