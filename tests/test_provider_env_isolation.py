"""A CLI provider must not be hijacked by an ambient credential."""
import os

import pytest

from core.provider import _CliSpec, _child_env


def _spec(**kw) -> _CliSpec:
    base = dict(argv=("copilot", "-p"), pass_prompt_via="arg", output_via="stdout")
    base.update(kw)
    return _CliSpec(**base)


def test_ambient_github_token_is_cleared_for_copilot(monkeypatch) -> None:
    """The token that is there for git must not outrank `copilot /login`.

    Copilot resolves COPILOT_GITHUB_TOKEN > GH_TOKEN > GITHUB_TOKEN > its
    own stored login. A developer machine almost always has GITHUB_TOKEN
    set for git and gh, holding a repo-scoped fine-grained PAT with no
    "Copilot Requests" permission -- so the call 401s with a message about
    an invalid or expired token while a working login sits unused.
    """
    monkeypatch.setenv("GITHUB_TOKEN", "github_pat_repo_scoped")
    monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
    env = _child_env(_spec(env_unset=("GITHUB_TOKEN", "GH_TOKEN"),
                           env_unset_override="COPILOT_GITHUB_TOKEN"))
    assert env is not None, "the subprocess inherited the ambient token"
    assert "GITHUB_TOKEN" not in env
    assert env.get("PATH") == os.environ.get("PATH"), "the rest of the env must survive"


def test_an_explicit_copilot_token_is_left_alone(monkeypatch) -> None:
    """Setting the override is how a user says "authenticate via the
    environment". Then we touch nothing -- including GITHUB_TOKEN, which
    they may well have set as the fallback on purpose."""
    monkeypatch.setenv("GITHUB_TOKEN", "github_pat_x")
    monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "gho_deliberate")
    assert _child_env(_spec(env_unset=("GITHUB_TOKEN",),
                            env_unset_override="COPILOT_GITHUB_TOKEN")) is None


def test_nothing_to_clear_inherits_unchanged(monkeypatch) -> None:
    """None means "inherit", which is cheaper than copying the environment
    and keeps the common case identical to the old behaviour."""
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    assert _child_env(_spec(env_unset=("GITHUB_TOKEN", "GH_TOKEN"),
                            env_unset_override="COPILOT_GITHUB_TOKEN")) is None
    assert _child_env(_spec()) is None


def test_other_clis_are_untouched() -> None:
    """Only a spec that opts in gets a modified environment; codex and
    claude authenticate their own way and must keep inheriting."""
    from core.provider import _CLI_SPECS

    for name, spec in _CLI_SPECS.items():
        if name == "copilot_cli":
            assert spec.env_unset, "copilot_cli must opt in"
        else:
            assert not spec.env_unset, f"{name} unexpectedly clears env vars"
