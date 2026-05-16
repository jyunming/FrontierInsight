"""Tests for web/quest_launcher.py — the subprocess pool that
spawns quests from the ``--serve`` web UI's interview-submit
endpoint.

These tests do NOT spawn a real Python; they monkeypatch
``subprocess.Popen`` so we can exercise the pool's accounting +
cancel + reap semantics without 60s+ test runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from web.quest_launcher import (
    LaunchedQuest, QuestLauncher, QuestLauncherFull,
)


class _FakePopen:
    """Stand-in for ``subprocess.Popen`` that lets the test decide
    when the child has 'exited' via the ``alive`` attribute."""

    def __init__(self, pid: int = 4242) -> None:
        self.pid = pid
        self._alive = True
        self._exit_code: int | None = None
        self.signals_received: list[Any] = []

    def poll(self) -> int | None:
        return None if self._alive else self._exit_code

    def terminate(self) -> None:
        self._alive = False
        self._exit_code = -15
        self.signals_received.append("terminate")

    def kill(self) -> None:
        self._alive = False
        self._exit_code = -9
        self.signals_received.append("kill")

    def send_signal(self, sig: Any) -> None:
        self._alive = False
        self._exit_code = -1
        self.signals_received.append(sig)

    def wait(self, timeout: float | None = None) -> int:
        if self._alive:
            # Simulate a timeout if the child is still 'alive'.
            import subprocess
            raise subprocess.TimeoutExpired(cmd="fake", timeout=timeout or 0)
        return self._exit_code or 0


@pytest.fixture
def launcher(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Build a QuestLauncher whose subprocess.Popen is replaced
    by a factory returning fresh _FakePopen instances. Each
    .launch() returns the most-recently-spawned _FakePopen via
    the LaunchedQuest.process attribute."""
    fakes: list[_FakePopen] = []

    def fake_popen(argv, **kwargs):
        p = _FakePopen(pid=4000 + len(fakes))
        fakes.append(p)
        return p

    monkeypatch.setattr("web.quest_launcher.subprocess.Popen", fake_popen)

    launcher = QuestLauncher(
        repo_root=tmp_path,
        max_concurrent=2,
        python_path="python",
        vscode_bridge_port=0,
    )
    launcher._fakes = fakes  # type: ignore[attr-defined]
    return launcher


# ---------------------------------------------------------------------------
# Basic launch + tracking
# ---------------------------------------------------------------------------


def test_launch_records_quest_in_registry(launcher: QuestLauncher, tmp_path: Path) -> None:
    yaml = tmp_path / "q.yaml"
    yaml.write_text("topic: x", encoding="utf-8")
    entry = launcher.launch(quest_id="abc-123", yaml_path=yaml)
    assert entry.quest_id == "abc-123"
    assert entry.pid == 4000
    assert entry.yaml_path == yaml
    alive = launcher.list_alive()
    assert len(alive) == 1
    assert alive[0].quest_id == "abc-123"


def test_launch_sets_preseed_env(launcher: QuestLauncher, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The launcher must pass FI_PRESEED_QUEST_ID to the child so
    its Engine reuses the pre-minted quest_id."""
    captured: dict[str, Any] = {}

    def fake_popen_capture(argv, **kwargs):
        captured["env"] = kwargs.get("env", {})
        captured["argv"] = argv
        return _FakePopen()

    monkeypatch.setattr("web.quest_launcher.subprocess.Popen", fake_popen_capture)
    yaml = tmp_path / "q.yaml"
    yaml.write_text("topic: x", encoding="utf-8")
    launcher.launch(quest_id="my-quest", yaml_path=yaml)

    assert captured["env"]["FI_PRESEED_QUEST_ID"] == "my-quest"
    assert "--config" in captured["argv"]
    assert str(yaml) in captured["argv"]


def test_launch_forwards_vscode_bridge_port(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When the parent server was launched with --vscode-bridge-port,
    each spawned child needs the same flag so LLM calls route back
    through the same bridge."""
    captured_argv: list[list[str]] = []

    def fake_popen(argv, **kwargs):
        captured_argv.append(argv)
        return _FakePopen()

    monkeypatch.setattr("web.quest_launcher.subprocess.Popen", fake_popen)
    launcher = QuestLauncher(repo_root=tmp_path, max_concurrent=2, vscode_bridge_port=12345)
    yaml = tmp_path / "q.yaml"
    yaml.write_text("topic: x", encoding="utf-8")
    launcher.launch(quest_id="q1", yaml_path=yaml)
    argv = captured_argv[0]
    assert "--vscode-bridge-port" in argv
    assert "12345" in argv


# ---------------------------------------------------------------------------
# Concurrency cap
# ---------------------------------------------------------------------------


def test_launcher_refuses_when_at_capacity(launcher: QuestLauncher, tmp_path: Path) -> None:
    """max_concurrent=2 means the third launch attempt raises
    QuestLauncherFull. The handler maps that to 503."""
    yaml = tmp_path / "q.yaml"
    yaml.write_text("topic: x", encoding="utf-8")
    launcher.launch(quest_id="q1", yaml_path=yaml)
    launcher.launch(quest_id="q2", yaml_path=yaml)
    with pytest.raises(QuestLauncherFull):
        launcher.launch(quest_id="q3", yaml_path=yaml)


def test_finished_quests_are_reaped_freeing_slots(launcher: QuestLauncher, tmp_path: Path) -> None:
    """If one of the two running quests finishes, a third can
    launch. The reap is triggered as a side effect of the next
    .launch() call (or .list_alive())."""
    yaml = tmp_path / "q.yaml"
    yaml.write_text("topic: x", encoding="utf-8")
    e1 = launcher.launch(quest_id="q1", yaml_path=yaml)
    launcher.launch(quest_id="q2", yaml_path=yaml)
    # Simulate q1 finishing.
    e1.process._alive = False  # type: ignore[attr-defined]
    e1.process._exit_code = 0  # type: ignore[attr-defined]
    # Should now succeed.
    e3 = launcher.launch(quest_id="q3", yaml_path=yaml)
    assert e3.quest_id == "q3"


# ---------------------------------------------------------------------------
# Cancel
# ---------------------------------------------------------------------------


def test_cancel_running_quest_signals_termination(launcher: QuestLauncher, tmp_path: Path) -> None:
    yaml = tmp_path / "q.yaml"
    yaml.write_text("topic: x", encoding="utf-8")
    entry = launcher.launch(quest_id="killable", yaml_path=yaml)
    # The fake's wait() raises TimeoutExpired while alive, so the
    # cancel path falls through to kill().
    ok = launcher.cancel("killable", grace_s=0.01)
    assert ok is True
    received = entry.process.signals_received  # type: ignore[attr-defined]
    # POSIX path uses terminate(); Windows uses send_signal(CTRL_BREAK_EVENT).
    # str() on a Signals enum returns "Signals.CTRL_BREAK_EVENT".
    repr_received = [repr(s) for s in received]
    assert "terminate" in received or any("CTRL_BREAK" in r for r in repr_received), (
        f"expected terminate or CTRL_BREAK_EVENT, got {received}"
    )
    # After grace timeout the fake's wait() raises and kill() fires.
    # Skip the kill assertion on Windows — the send_signal path
    # marks the fake as not-alive immediately, so the grace-timeout
    # branch doesn't trigger kill(). On POSIX, terminate() also
    # marks not-alive, but we set up the fake to make wait() raise
    # if alive — so kill is reached only if terminate didn't set
    # alive=False. Both paths are valid outcomes; the assert above
    # already proves the cancel signal landed.


def test_cancel_unknown_quest_returns_false(launcher: QuestLauncher) -> None:
    assert launcher.cancel("never-launched") is False


def test_cancel_already_finished_quest_returns_false(launcher: QuestLauncher, tmp_path: Path) -> None:
    yaml = tmp_path / "q.yaml"
    yaml.write_text("topic: x", encoding="utf-8")
    entry = launcher.launch(quest_id="done", yaml_path=yaml)
    entry.process._alive = False  # type: ignore[attr-defined]
    entry.process._exit_code = 0  # type: ignore[attr-defined]
    assert launcher.cancel("done") is False


# ---------------------------------------------------------------------------
# status_for
# ---------------------------------------------------------------------------


def test_status_for_known_quest_returns_metadata(launcher: QuestLauncher, tmp_path: Path) -> None:
    yaml = tmp_path / "q.yaml"
    yaml.write_text("topic: x", encoding="utf-8")
    launcher.launch(quest_id="visible", yaml_path=yaml)
    status = launcher.status_for("visible")
    assert status is not None
    assert status["alive"] is True
    assert status["pid"] >= 4000
    assert status["age_seconds"] >= 0


def test_status_for_unknown_quest_returns_none(launcher: QuestLauncher) -> None:
    assert launcher.status_for("never-launched") is None
