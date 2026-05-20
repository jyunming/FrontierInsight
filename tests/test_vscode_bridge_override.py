"""``_apply_vscode_bridge_override`` / ``_apply_vscode_bridge_socket_override`` tests.

Pin the provider-respecting semantics introduced after the OPC quest
that pinned ``claude_cli`` in YAML, then hit the VSCode bridge anyway
because the auto-resolved socket silently clobbered ``provider.name``
back to ``vscode_extension``.

The contract:

  - YAML ``provider.name == "vscode_extension"`` → bridge wiring
    lands in ``provider.extra``, provider confirmed as
    ``vscode_extension``.
  - YAML ``provider.name`` is anything else (every other value in
    the ``ProviderName`` Literal — ``claude_cli``, ``openai``,
    ``gemini``, ``vllm``, ``codex_cli``, ``github_copilot_cli``,
    ``claude_code``, …) → respected verbatim. No clobber. The
    bridge being available is irrelevant to a quest that explicitly
    picked a different transport.
"""

from __future__ import annotations

from pathlib import Path
from typing import get_args

import pytest

from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig,
    OutputConfig, ProviderConfig, ProviderName,
)
from launch import (
    _apply_vscode_bridge_override,
    _apply_vscode_bridge_socket_override,
    _set_vscode_bridge_extras,
)


# Derived from the ``ProviderName`` Literal so this list can't drift
# when a new provider is added to ``core/config.py``. The
# ``vscode_extension`` value is the one this override IS allowed to
# coerce / confirm; everything else must be respected.
_NON_VSCODE_PROVIDERS = tuple(
    p for p in get_args(ProviderName) if p != "vscode_extension"
)


def _mk_cfg(tmp_path: Path, *, provider_name: str = "") -> Config:
    return Config(
        topic="x",
        title="x",
        provider=ProviderConfig(name=provider_name) if provider_name else ProviderConfig(),
        engine=EngineConfig(max_iterations=1, review_loop=False, clarify_mode="off"),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "out"),
    )


# --- port override -----------------------------------------------------------


def test_port_override_no_op_when_port_zero(tmp_path: Path) -> None:
    cfg = _mk_cfg(tmp_path, provider_name="claude_cli")
    _apply_vscode_bridge_override(cfg, port=0)
    assert cfg.provider.name == "claude_cli"
    assert "bridge_port" not in (cfg.provider.extra or {})


def test_port_override_wires_when_provider_is_vscode_extension(
    tmp_path: Path,
) -> None:
    """The chat-spawn / YAML-picked-vscode case — bridge wiring lands."""
    cfg = _mk_cfg(tmp_path, provider_name="vscode_extension")
    _apply_vscode_bridge_override(cfg, port=12345)
    assert cfg.provider.name == "vscode_extension"
    assert cfg.provider.extra["bridge_port"] == 12345


def test_port_override_respects_explicit_claude_cli_yaml(tmp_path: Path) -> None:
    """The bug this PR fixes. YAML says claude_cli; bridge port is
    available; the override must NOT silently rewrite the provider.
    The quest stays on claude_cli's CLI transport."""
    cfg = _mk_cfg(tmp_path, provider_name="claude_cli")
    _apply_vscode_bridge_override(cfg, port=12345)
    assert cfg.provider.name == "claude_cli"
    assert "bridge_port" not in (cfg.provider.extra or {}), (
        "bridge_port leaked into provider.extra for a non-vscode "
        "provider; the CLI transport doesn't read it but writing it "
        "is dead weight and confuses log readers."
    )


@pytest.mark.parametrize("provider_name", _NON_VSCODE_PROVIDERS)
def test_port_override_respects_every_non_vscode_provider(
    tmp_path: Path, provider_name: str,
) -> None:
    """Pin the rule across every non-vscode provider the engine
    accepts (derived from ``ProviderName`` Literal so adding a new
    provider to ``core/config.py`` auto-extends coverage). A typo on
    the name check or a partial rewrite that handles only some
    providers gets caught here."""
    cfg = _mk_cfg(tmp_path, provider_name=provider_name)
    _apply_vscode_bridge_override(cfg, port=999)
    assert cfg.provider.name == provider_name


# --- socket override ---------------------------------------------------------


def test_socket_override_no_op_when_socket_empty(tmp_path: Path) -> None:
    cfg = _mk_cfg(tmp_path, provider_name="claude_cli")
    _apply_vscode_bridge_socket_override(cfg, socket_path="")
    assert cfg.provider.name == "claude_cli"
    assert "bridge_socket" not in (cfg.provider.extra or {})


def test_socket_override_wires_when_provider_is_vscode_extension(
    tmp_path: Path,
) -> None:
    cfg = _mk_cfg(tmp_path, provider_name="vscode_extension")
    _apply_vscode_bridge_socket_override(cfg, socket_path="/tmp/fi.sock")
    assert cfg.provider.name == "vscode_extension"
    assert cfg.provider.extra["bridge_socket"] == "/tmp/fi.sock"


def test_socket_override_respects_explicit_claude_cli_yaml(tmp_path: Path) -> None:
    """The actual production bug — auto-resolved socket from --serve
    silently re-routed a claude_cli quest through the VSCode bridge."""
    cfg = _mk_cfg(tmp_path, provider_name="claude_cli")
    _apply_vscode_bridge_socket_override(cfg, socket_path="/tmp/fi.sock")
    assert cfg.provider.name == "claude_cli"
    assert "bridge_socket" not in (cfg.provider.extra or {})


@pytest.mark.parametrize("provider_name", _NON_VSCODE_PROVIDERS)
def test_socket_override_respects_every_non_vscode_provider(
    tmp_path: Path, provider_name: str,
) -> None:
    """Derived from ``ProviderName`` — see the port variant for why."""
    cfg = _mk_cfg(tmp_path, provider_name=provider_name)
    _apply_vscode_bridge_socket_override(cfg, socket_path="/tmp/fi.sock")
    assert cfg.provider.name == provider_name


# --- PM-command extras helper -----------------------------------------------
#
# ``_set_vscode_bridge_extras`` is the helper that the five PM-command
# runners (``_run_summarize`` / ``_run_digest`` / ``_run_portfolio`` /
# ``_run_critique`` / ``_run_proposal``) call to wire the bridge port +
# socket into their freshly-built ``ProviderConfig``. The guard added in
# PR #134 to ``_apply_vscode_bridge_override`` /
# ``_apply_vscode_bridge_socket_override`` was missed here, so a user
# running e.g. ``python launch.py --critique <id> --critique-provider
# claude_cli`` while ``--serve``'s auto-resolved bridge socket was set
# had their critique silently re-routed through Copilot. Same contract
# now applies; same Literal-derived sweep guards it.


def test_extras_no_op_when_port_zero_and_socket_empty(tmp_path: Path) -> None:
    cfg = _mk_cfg(tmp_path, provider_name="claude_cli")
    _set_vscode_bridge_extras(cfg.provider, bridge_port=0, bridge_socket="")
    assert cfg.provider.name == "claude_cli"
    assert "bridge_port" not in (cfg.provider.extra or {})
    assert "bridge_socket" not in (cfg.provider.extra or {})


def test_extras_wires_port_when_provider_is_vscode_extension(
    tmp_path: Path,
) -> None:
    cfg = _mk_cfg(tmp_path, provider_name="vscode_extension")
    _set_vscode_bridge_extras(cfg.provider, bridge_port=12345, bridge_socket="")
    assert cfg.provider.name == "vscode_extension"
    assert cfg.provider.extra["bridge_port"] == 12345
    assert "bridge_socket" not in cfg.provider.extra


def test_extras_wires_socket_when_provider_is_vscode_extension(
    tmp_path: Path,
) -> None:
    cfg = _mk_cfg(tmp_path, provider_name="vscode_extension")
    _set_vscode_bridge_extras(
        cfg.provider, bridge_port=0, bridge_socket="/tmp/fi.sock",
    )
    assert cfg.provider.name == "vscode_extension"
    assert cfg.provider.extra["bridge_socket"] == "/tmp/fi.sock"
    assert "bridge_port" not in cfg.provider.extra


def test_extras_wires_both_port_and_socket_when_vscode_extension(
    tmp_path: Path,
) -> None:
    """``core.provider`` prefers the socket but both are accepted; the
    helper writes whichever are non-empty so callers can hand it both
    and let the resolver pick."""
    cfg = _mk_cfg(tmp_path, provider_name="vscode_extension")
    _set_vscode_bridge_extras(
        cfg.provider, bridge_port=12345, bridge_socket="/tmp/fi.sock",
    )
    assert cfg.provider.name == "vscode_extension"
    assert cfg.provider.extra["bridge_port"] == 12345
    assert cfg.provider.extra["bridge_socket"] == "/tmp/fi.sock"


def test_extras_respects_explicit_claude_cli_provider(tmp_path: Path) -> None:
    """The Slice 7 HIGH #3 bug. ``--critique-provider claude_cli``
    while the persistent VSCode bridge socket is auto-resolved must
    NOT re-route the critique through Copilot."""
    cfg = _mk_cfg(tmp_path, provider_name="claude_cli")
    _set_vscode_bridge_extras(
        cfg.provider, bridge_port=12345, bridge_socket="/tmp/fi.sock",
    )
    assert cfg.provider.name == "claude_cli"
    assert "bridge_port" not in (cfg.provider.extra or {})
    assert "bridge_socket" not in (cfg.provider.extra or {})


@pytest.mark.parametrize("provider_name", _NON_VSCODE_PROVIDERS)
def test_extras_respects_every_non_vscode_provider_port(
    tmp_path: Path, provider_name: str,
) -> None:
    """``ProviderName``-derived sweep — bridge port alone must not
    clobber any non-vscode provider chosen on a PM-command flag."""
    cfg = _mk_cfg(tmp_path, provider_name=provider_name)
    _set_vscode_bridge_extras(cfg.provider, bridge_port=999, bridge_socket="")
    assert cfg.provider.name == provider_name
    assert "bridge_port" not in (cfg.provider.extra or {})


@pytest.mark.parametrize("provider_name", _NON_VSCODE_PROVIDERS)
def test_extras_respects_every_non_vscode_provider_socket(
    tmp_path: Path, provider_name: str,
) -> None:
    """``ProviderName``-derived sweep — bridge socket alone must not
    clobber any non-vscode provider chosen on a PM-command flag."""
    cfg = _mk_cfg(tmp_path, provider_name=provider_name)
    _set_vscode_bridge_extras(
        cfg.provider, bridge_port=0, bridge_socket="/tmp/fi.sock",
    )
    assert cfg.provider.name == provider_name
    assert "bridge_socket" not in (cfg.provider.extra or {})


@pytest.mark.parametrize("provider_name", _NON_VSCODE_PROVIDERS)
def test_extras_respects_every_non_vscode_provider_both(
    tmp_path: Path, provider_name: str,
) -> None:
    """The realistic ``--serve`` case: both port AND socket auto-
    resolved; explicit non-vscode PM-command provider still wins."""
    cfg = _mk_cfg(tmp_path, provider_name=provider_name)
    _set_vscode_bridge_extras(
        cfg.provider, bridge_port=12345, bridge_socket="/tmp/fi.sock",
    )
    assert cfg.provider.name == provider_name
    extras = cfg.provider.extra or {}
    assert "bridge_port" not in extras
    assert "bridge_socket" not in extras
