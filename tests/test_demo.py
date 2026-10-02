"""`fi demo`: the first step after installing. It writes a small example quest in the folder it runs in (never over a
file that is there), checks the model at no cost, and asks before running it; with no terminal to ask in, it prints
the command and stops. Nothing here calls a model or the network."""

from __future__ import annotations

from pathlib import Path

import pytest

import launch
from core import demo
from core import provider_readiness as pr
from core.config import Config


@pytest.fixture
def ready_openai(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """openai set up and passing its check; records which providers were checked."""
    checked: list[str] = []
    monkeypatch.setattr(demo, "set_up_providers", lambda: ["openai"])

    async def fake_preflight(provider, **kwargs):  # noqa: ANN001, ANN202
        checked.append(provider)
        return pr.Readiness(provider, "model_available", "The service accepts the key and lists gpt-5.")

    monkeypatch.setattr(pr, "preflight", fake_preflight)
    return checked


def _no_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(launch, "_stdin_is_terminal", lambda: False)


def test_demo_writes_an_example_that_loads_and_names_no_missing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], ready_openai: list[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    _no_terminal(monkeypatch)
    assert launch._demo([]) == 0
    out = capsys.readouterr().out
    path = tmp_path / "fi-demo.yaml"
    assert path.is_file()
    cfg = Config.from_yaml(path)
    assert cfg.provider.name == "openai"
    # The pip-installed FI has no examples/ folder: nothing the demo writes or prints may point into one.
    assert "examples/" not in out and "examples/" not in path.read_text(encoding="utf-8")
    # No terminal to ask in: the exact command, and nothing is run.
    assert "--config fi-demo.yaml" in out
    assert ready_openai == ["openai"]


def test_demo_never_overwrites(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ready_openai: list[str]) -> None:
    monkeypatch.chdir(tmp_path)
    _no_terminal(monkeypatch)
    (tmp_path / "fi-demo.yaml").write_text("mine", encoding="utf-8")
    assert launch._demo([]) == 0
    assert (tmp_path / "fi-demo.yaml").read_text(encoding="utf-8") == "mine"
    assert (tmp_path / "fi-demo-2.yaml").is_file()
    assert launch._demo([]) == 0
    assert (tmp_path / "fi-demo-3.yaml").is_file()


def test_demo_asks_before_running_and_runs_on_yes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ready_openai: list[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(launch, "_stdin_is_terminal", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True, raising=False)
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")
    assert launch._demo([]) == 0  # no: nothing is run
    monkeypatch.setattr("builtins.input", lambda prompt="": "y")
    went_on = launch._demo([])
    assert went_on == ["--config", str(tmp_path / "fi-demo-2.yaml")]


def test_demo_passes_over_a_provider_that_fails_its_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """A key that is set but refused is passed over for the next provider set up on this computer."""
    monkeypatch.chdir(tmp_path)
    _no_terminal(monkeypatch)
    monkeypatch.setattr(demo, "set_up_providers", lambda: ["openai", "claude_cli"])

    async def fake_preflight(provider, **kwargs):  # noqa: ANN001, ANN202
        if provider == "openai":
            return pr.Readiness(provider, "key_rejected", "The service refused the key.", "Check OPENAI_API_KEY.")
        return pr.Readiness(provider, "signed_in", "`claude auth status` says you are signed in.")

    monkeypatch.setattr(pr, "preflight", fake_preflight)
    assert launch._demo([]) == 0
    assert Config.from_yaml(tmp_path / "fi-demo.yaml").provider.name == "claude_cli"


def test_demo_with_nothing_set_up_says_what_to_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    _no_terminal(monkeypatch)
    monkeypatch.setattr(demo, "set_up_providers", lambda: [])

    async def no_key(provider, **kwargs):  # noqa: ANN001, ANN202
        return pr.check_local("openai")

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(pr, "preflight", no_key)
    # Exit 1: `fi demo && fi --config fi-demo.yaml` must not go on to a model just found not ready.
    assert launch._demo([]) == 1
    out = capsys.readouterr().out
    assert "OPENAI_API_KEY" in out and "Fix the model first" in out
    assert "openai" in (tmp_path / "fi-demo.yaml").read_text(encoding="utf-8")


def test_demo_rejects_arguments(capsys: pytest.CaptureFixture[str]) -> None:
    assert launch._demo(["--bogus"]) == 2
    assert launch._demo(["--help"]) == 0
    assert "fi-demo.yaml" in capsys.readouterr().out


def test_fi_demo_is_dispatched_before_argument_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    """`fi demo` is a word, not a flag: main() hands it to _demo, and argparse never sees it."""
    seen: list[list[str]] = []
    monkeypatch.setattr(launch, "_demo", lambda argv: seen.append(argv) or 0)
    monkeypatch.setattr(launch, "_load_dotenvs", lambda: None)  # a .env must not leak keys into later tests
    monkeypatch.setattr(launch, "_force_utf8_streams", lambda: None)
    monkeypatch.setattr(launch, "_quiet_network_logs", lambda: None)  # it changes logging for the whole process
    monkeypatch.setattr("sys.argv", ["fi", "demo"])
    assert launch.main() == 0
    assert seen == [[]]


def test_demo_yaml_text_is_the_written_file(tmp_path: Path) -> None:
    path = demo.write_demo_config(tmp_path, "ollama", "llama3.2")
    text = path.read_text(encoding="utf-8")
    assert "name: ollama" in text and "model: llama3.2" in text
    assert Config.from_yaml(path).provider.model == "llama3.2"
    with pytest.raises(FileExistsError):
        with open(path, "x", encoding="utf-8"):
            pass


def test_set_up_providers_follows_the_local_check(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(pr.shutil, "which", lambda name: "/bin/codex" if name == "codex" else None)
    assert demo.set_up_providers() == ["codex_cli"]
