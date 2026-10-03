"""The CLI interview a first-time person meets: each provider listed with its state (not a READY), the chosen one
checked at no cost before the quest is launched, and no "(default) (default)". No model is called."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import launch
from core import provider_readiness as pr


def test_a_label_that_says_default_is_not_marked_twice(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    from core.interview import QUESTIONS

    q = next(q for q in QUESTIONS if q.id == "result_use")
    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    assert launch._cli_prompt_for(q, {}, {}) == "research"
    out = capsys.readouterr().out
    assert "Research (default)" in out
    assert "(default) (default)" not in out


def test_the_provider_list_shows_each_state(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from core.interview import QUESTIONS

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(pr.shutil, "which", lambda name: "/bin/claude" if name == "claude" else None)
    monkeypatch.setattr(pr, "_run_command", lambda argv, timeout_s: pytest.fail("the list runs no command"))
    q = next(q for q in QUESTIONS if q.id == "provider")
    monkeypatch.setattr("builtins.input", lambda prompt="": "1")
    launch._cli_prompt_for(q, {}, {})
    out = capsys.readouterr().out
    assert "No key — OPENAI_API_KEY is not set." in out
    assert "Installed — The `claude` command is here" in out
    assert "Not installed — No `codex` command" in out
    assert "READY" not in out and "✓ ready" not in out


@pytest.mark.asyncio
async def test_launch_is_stopped_when_the_chosen_model_is_not_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    from core.provider import ProxySupervisor

    monkeypatch.setenv("FI_HOME", str(tmp_path / "fi_home"))

    async def fake_clarify(**_kw: Any) -> dict[str, str]:
        return {}

    monkeypatch.setattr("core.interview.preflight_clarify", fake_clarify)
    checked: list[tuple[str, Any]] = []

    async def not_signed_in(provider, *, model=None, base_url="", api_key_env="", timeout_s=5.0):  # noqa: ANN001, ANN202
        checked.append((provider, model))
        return pr.Readiness(provider, "not_signed_in", "`claude auth status` says you are not signed in.",
                            "Run `claude auth login` in a terminal.")

    monkeypatch.setattr(pr, "preflight", not_signed_in)
    ran: list[str] = []

    async def fake_run_one(*a: Any, **k: Any) -> None:
        ran.append("run")

    monkeypatch.setattr(launch, "run_one", fake_run_one)
    # topic, result use (explore: no reviewer model), provider 3 (claude_cli), model 1, then Enter on the review
    # screen to launch.
    typed = iter(["A topic", "3", "3", "1", ""])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(typed, ""))
    rc = await launch._run_new(
        output_root=tmp_path / "outputs", draft_only=False, vscode_bridge_port=0,
        interactive=False, supervisor=ProxySupervisor(),
    )
    out = capsys.readouterr().out
    assert rc == 1 and ran == []
    assert checked and checked[0][0] == "claude_cli"
    assert "Not signed in — `claude auth status` says you are not signed in." in out
    assert "To fix: Run `claude auth login` in a terminal." in out
    # The answers are kept, and the command that runs them once fixed is printed.
    (draft,) = list((tmp_path / "outputs" / "_drafts").glob("*.yaml"))
    assert f"--config {draft}" in out
