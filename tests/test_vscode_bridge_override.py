"""``_apply_vscode_bridge_override`` / ``_apply_vscode_bridge_socket_override`` tests.

Pin the provider-respecting semantics introduced after the OPC quest
that pinned ``claude_cli`` in YAML, then hit the VSCode bridge anyway
because the auto-resolved socket silently clobbered ``provider.name``
back to ``vscode_extension``.

The contract:

  - YAML ``provider.name == "vscode_extension"`` (or empty/unset) →
    bridge wiring lands in ``provider.extra``, provider coerced /
    confirmed as ``vscode_extension``.
  - YAML ``provider.name`` is anything else (``claude_cli``,
    ``openai``, ...) → respected verbatim. No clobber. The bridge
    being available is irrelevant to a quest that explicitly picked
    a different transport.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig,
    OutputConfig, ProviderConfig,
)
from launch import (
    _apply_vscode_bridge_override,
    _apply_vscode_bridge_socket_override,
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


@pytest.mark.parametrize("provider_name", [
    "claude_cli", "openai", "codex", "codex_cli", "copilot_cli",
    "gemini_cli", "ollama",
])
def test_port_override_respects_every_non_vscode_provider(
    tmp_path: Path, provider_name: str,
) -> None:
    """Pin the rule across every non-vscode provider the engine
    accepts so a future addition to the provider list (or a typo on
    the name check) doesn't silently start clobbering."""
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


@pytest.mark.parametrize("provider_name", [
    "claude_cli", "openai", "codex", "codex_cli", "copilot_cli",
    "gemini_cli", "ollama",
])
def test_socket_override_respects_every_non_vscode_provider(
    tmp_path: Path, provider_name: str,
) -> None:
    cfg = _mk_cfg(tmp_path, provider_name=provider_name)
    _apply_vscode_bridge_socket_override(cfg, socket_path="/tmp/fi.sock")
    assert cfg.provider.name == provider_name
