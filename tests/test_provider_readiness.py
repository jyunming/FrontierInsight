"""Provider readiness: a ladder of plain states (not installed, installed, signed in, reachable, model available)
instead of a READY that only meant "the command is on PATH" or "a key is set".

No real CLI and no network: the status-command runner, ``shutil.which`` and the HTTP getter are replaced.
"""

from __future__ import annotations

import asyncio

import pytest

from core import provider_readiness as pr


@pytest.fixture
def no_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    for env in ("OPENAI_API_KEY", "GEMINI_API_KEY", "OLLAMA_HOST"):
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setattr(pr.shutil, "which", lambda _name: None)

    async def nothing_answers(url, headers, timeout_s):  # noqa: ANN001, ANN202
        return None, None

    monkeypatch.setattr(pr, "_get_json", nothing_answers)

    def no_command(argv, timeout_s):  # noqa: ANN001, ANN202
        raise AssertionError(f"no status command should run: {argv}")

    monkeypatch.setattr(pr, "_run_command", no_command)


def _which(*present: str):  # noqa: ANN202
    return lambda name: f"/bin/{name}" if name in present else None


def test_cli_binaries_match_what_the_provider_layer_runs() -> None:
    from core.provider import _CLI_SPECS

    assert pr.CLI_BINARIES == {name: spec.argv[0] for name, spec in _CLI_SPECS.items()}


def test_a_command_on_path_is_installed_never_ready(no_tools: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """The audit's finding: a CLI on PATH was shown as READY. It is "installed" and says what is not checked yet."""
    monkeypatch.setattr(pr.shutil, "which", _which("claude"))
    r = pr.check_local("claude_cli")
    assert r.state == "installed" and r.label == "Installed"
    assert "signed in" in r.sentence
    assert "READY" not in pr.one_line(r).upper().split()


def test_a_key_is_only_key_present(no_tools: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    r = asyncio.run(pr.check("openai", sign_in=True))  # the Settings page: no call to the service
    assert r.state == "key_present" and not r.blocked
    assert "OPENAI_API_KEY is set" in r.sentence


def test_no_key_says_which_variable(no_tools: None) -> None:
    r = pr.check_local("openai")
    assert r.state == "no_key" and r.blocked
    assert "OPENAI_API_KEY" in r.sentence and "OPENAI_API_KEY" in r.fix


def test_not_installed_cli_says_how_to_install(no_tools: None) -> None:
    r = asyncio.run(pr.check("codex_cli"))
    assert r.state == "not_installed" and r.blocked
    assert "codex login" in r.fix


def test_claude_signed_in_reads_only_logged_in(no_tools: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pr.shutil, "which", _which("claude"))
    seen: list[list[str]] = []

    def status(argv, timeout_s):  # noqa: ANN001, ANN202
        seen.append(list(argv))
        assert timeout_s <= 10
        return 0, '{"loggedIn": true, "email": "someone@example.org", "orgId": "x"}'

    monkeypatch.setattr(pr, "_run_command", status)
    r = asyncio.run(pr.check("claude_cli"))
    assert seen == [["/bin/claude", "auth", "status"]]
    assert r.state == "signed_in" and not r.blocked
    # What the status command said beyond loggedIn is never shown or kept.
    assert "someone@example.org" not in repr(r.as_dict())


def test_claude_signed_out_says_how_to_sign_in(no_tools: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pr.shutil, "which", _which("claude"))
    monkeypatch.setattr(pr, "_run_command", lambda argv, timeout_s: (1, '{"loggedIn": false}'))
    r = asyncio.run(pr.check("claude_cli"))
    assert r.state == "not_signed_in" and r.blocked
    assert r.fix == "Run `claude auth login` in a terminal."


@pytest.mark.parametrize(("rc", "out", "state"), [
    (0, "Logged in using ChatGPT", "signed_in"),
    (1, "Not logged in", "not_signed_in"),
    (None, "", "sign_in_unknown"),  # timed out
    (2, "error: unknown command", "sign_in_unknown"),
])
def test_codex_status(no_tools: None, monkeypatch: pytest.MonkeyPatch, rc, out, state) -> None:  # noqa: ANN001
    monkeypatch.setattr(pr.shutil, "which", _which("codex"))
    monkeypatch.setattr(pr, "_run_command", lambda argv, timeout_s: (rc, out))
    assert asyncio.run(pr.check("codex_cli")).state == state


def test_a_cli_with_no_status_command_says_it_cannot_tell(no_tools: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pr.shutil, "which", _which("gemini", "copilot"))
    for name in ("gemini_cli", "copilot_cli"):
        r = asyncio.run(pr.check(name))
        assert r.state == "sign_in_unknown" and not r.blocked
        assert r.label == "Signed in: unknown"


def test_a_status_command_that_hangs_is_stopped() -> None:
    """The real runner: a command that never answers is stopped at the time limit (with what it started) and
    reported as not answering, so a check never hangs the page or the interview."""
    import sys
    import time

    started = time.monotonic()
    rc, out = pr._run_command([sys.executable, "-c", "import time; time.sleep(30)"], 1.0)
    assert rc is None and out == ""
    assert time.monotonic() - started < 15


def test_ollama_server_and_model(no_tools: None, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    async def tags(url, headers, timeout_s):  # noqa: ANN001, ANN202
        calls.append(url)
        return 200, {"models": [{"name": "llama3.2:latest"}, {"name": "qwen2.5:7b"}]}

    monkeypatch.setattr(pr, "_get_json", tags)
    assert asyncio.run(pr.check("ollama", model="llama3.2")).state == "model_available"
    missing = asyncio.run(pr.check("ollama", model="mistral"))
    assert missing.state == "model_missing" and missing.blocked
    assert "ollama pull mistral" in missing.fix
    assert calls[0] == "http://127.0.0.1:11434/api/tags"


def test_ollama_in_a_container_needs_no_command(no_tools: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """The server is what counts, not the `ollama` command: one elsewhere (a container), named by provider.base_url,
    is reachable. OLLAMA_HOST is not read: the engine calls provider.base_url or 127.0.0.1:11434, never it."""
    monkeypatch.setenv("OLLAMA_HOST", "0.0.0.0")
    seen: list[str] = []

    async def tags(url, headers, timeout_s):  # noqa: ANN001, ANN202
        seen.append(url)
        return 200, {"models": [{"name": "gemma3:4b"}]}

    monkeypatch.setattr(pr, "_get_json", tags)
    assert asyncio.run(pr.check("ollama", base_url="http://10.0.0.5:11434/v1")).state == "reachable"
    assert asyncio.run(pr.check("ollama")).state == "reachable"
    assert seen == ["http://10.0.0.5:11434/api/tags", "http://127.0.0.1:11434/api/tags"]


def test_ollama_launch_check_names_the_default_model(no_tools: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """No model named: the quest calls the provider's default, so the launch check looks for that one."""
    from core.provider import _DIRECT_DEFAULTS

    async def tags(url, headers, timeout_s):  # noqa: ANN001, ANN202
        return 200, {"models": [{"name": "nomic-embed-text:latest"}]}

    monkeypatch.setattr(pr, "_get_json", tags)
    default = _DIRECT_DEFAULTS["ollama"]["model"]
    r = asyncio.run(pr.preflight("ollama"))
    assert r.state == "model_missing" and r.blocked and f"ollama pull {default}" in r.fix
    assert asyncio.run(pr.check("ollama")).state == "reachable"  # Settings: no model chosen, none assumed


def test_defaults_match_the_provider_layer() -> None:
    from core.provider import _DIRECT_DEFAULTS

    for name, env in pr.KEY_ENVS.items():
        assert _DIRECT_DEFAULTS[name]["api_key_env"] == env
    for name, url in pr._BASE_URLS.items():
        assert _DIRECT_DEFAULTS[name]["base_url"].rstrip("/") == url


def test_own_gateway_needs_no_key(no_tools: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """A base_url of your own with no key named starts without a key (core.provider.missing_api_key): not blocked."""
    from core.config import ProviderConfig
    from core.provider import missing_api_key

    seen: list[tuple[str, dict]] = []

    async def models(url, headers, timeout_s):  # noqa: ANN001, ANN202
        seen.append((url, headers))
        return 200, {"data": [{"id": "my-model"}]}

    monkeypatch.setattr(pr, "_get_json", models)
    url = "http://127.0.0.1:1234/v1"
    assert missing_api_key(ProviderConfig(name="openai", base_url=url, model="my-model")) is None
    assert not pr.check_local("openai", base_url=url).blocked
    r = asyncio.run(pr.preflight("openai", model="my-model", base_url=url))
    assert r.state == "model_available" and seen == [(f"{url}/models", {})]
    assert "answers" in r.sentence and "key" not in r.sentence  # no key was sent, so none was "accepted"
    # Naming a key for it makes the key needed again.
    assert pr.check_local("openai", base_url=url, api_key_env="MY_KEY").state == "no_key"


def test_ollama_installed_but_not_running(no_tools: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pr.shutil, "which", _which("ollama"))
    r = asyncio.run(pr.check("ollama", model="llama3.2"))
    assert r.state == "not_running" and r.blocked
    assert "ollama serve" in r.fix


def test_preflight_asks_the_free_model_list(no_tools: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    seen: list[tuple[str, dict]] = []

    async def models(url, headers, timeout_s):  # noqa: ANN001, ANN202
        seen.append((url, headers))
        return 200, {"data": [{"id": "gpt-5"}, {"id": "gpt-5-mini"}]}

    monkeypatch.setattr(pr, "_get_json", models)
    ok = asyncio.run(pr.preflight("openai", model="gpt-5-mini"))
    assert ok.state == "model_available" and not ok.blocked
    assert seen[0] == ("https://api.openai.com/v1/models", {"Authorization": "Bearer sk-test"})
    # A model the list does not show is not blocked (a list can be incomplete); the sentence says so.
    unlisted = asyncio.run(pr.preflight("openai", model="gpt-9"))
    assert unlisted.state == "model_not_listed" and not unlisted.blocked
    assert "first call" in unlisted.sentence


def test_a_key_that_may_not_list_models_is_not_refused(no_tools: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-restricted")

    async def forbidden(url, headers, timeout_s):  # noqa: ANN001, ANN202
        return 403, {"error": "missing scope"}

    monkeypatch.setattr(pr, "_get_json", forbidden)
    r = asyncio.run(pr.preflight("openai", model="gpt-5"))
    assert r.state == "reachable" and not r.blocked


def test_a_cli_without_a_status_command_does_not_promise_a_check(
    no_tools: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pr.shutil, "which", _which("gemini", "claude"))
    assert "cannot check the sign-in" in pr.check_local("gemini_cli").sentence
    assert "is checked when you launch" in pr.check_local("claude_cli").sentence


def test_preflight_rejected_key_and_no_network(no_tools: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-bad")

    async def refused(url, headers, timeout_s):  # noqa: ANN001, ANN202
        return 401, {"error": "invalid key"}

    monkeypatch.setattr(pr, "_get_json", refused)
    r = asyncio.run(pr.preflight("openai", model="gpt-5"))
    assert r.state == "key_rejected" and r.blocked and "OPENAI_API_KEY" in r.fix
    monkeypatch.setattr(pr, "_get_json", lambda *a, **k: _none())
    r = asyncio.run(pr.preflight("openai", model="gpt-5"))
    assert r.state == "unreachable" and r.blocked


async def _none():  # noqa: ANN202
    return None, None


def test_settings_check_never_calls_an_api_key_service(no_tools: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    async def boom(url, headers, timeout_s):  # noqa: ANN001, ANN202
        raise AssertionError(f"no request expected: {url}")

    monkeypatch.setattr(pr, "_get_json", boom)
    assert asyncio.run(pr.check("openai")).state == "key_present"


def test_available_providers_is_installed_or_key_set(no_tools: None, monkeypatch: pytest.MonkeyPatch) -> None:
    from core.interview import available_providers

    monkeypatch.setattr(pr.shutil, "which", _which("copilot"))
    # copilot_cli runs `copilot`, not `gh`: the old probe looked for `gh`.
    assert available_providers() == ["copilot_cli"]
