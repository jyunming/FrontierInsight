"""What a person meets in their first run: the command the messages tell them to type, the key they forgot, the bundled example."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.config import Config, ProviderConfig
from core.provider import missing_api_key
from launch import parse_args

ROOT = Path(__file__).resolve().parent.parent


# --- `fi --resume <id>` is enough ---------------------------------------------------------------------------------------


def _saved_quest(tmp_path: Path, quest_id: str = "q-1") -> Path:
    quest = tmp_path / "outputs" / quest_id
    quest.mkdir(parents=True)
    (quest / "config.yaml").write_text("topic: t\n", encoding="utf-8")
    return quest


def test_resume_by_id_alone_uses_the_config_the_quest_saved(tmp_path: Path) -> None:
    quest = _saved_quest(tmp_path)
    args = parse_args(["--resume", "q-1", "--output-root", str(tmp_path / "outputs")])
    assert Path(args.config) == quest / "config.yaml" and args.resume == "q-1"


def test_resume_accepts_the_quest_folder_and_watch_gets_the_same_help(tmp_path: Path) -> None:
    quest = _saved_quest(tmp_path)
    by_folder = parse_args(["--resume", str(quest)])
    assert Path(by_folder.config) == quest / "config.yaml" and by_folder.resume == "q-1"
    watched = parse_args(["--watch", "q-1", "--output-root", str(tmp_path / "outputs")])
    assert Path(watched.config) == quest / "config.yaml"


def test_an_explicit_config_still_wins(tmp_path: Path) -> None:
    _saved_quest(tmp_path)
    mine = tmp_path / "mine.yaml"
    mine.write_text("topic: other\n", encoding="utf-8")
    args = parse_args(["--config", str(mine), "--resume", "q-1", "--output-root", str(tmp_path / "outputs")])
    assert Path(args.config) == mine


def test_an_unknown_quest_says_where_it_looked_instead_of_asking_for_a_mode(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as stop:
        parse_args(["--resume", "nope", "--output-root", str(tmp_path / "outputs")])
    assert stop.value.code == 2
    err = capsys.readouterr().err
    assert "no quest 'nope'" in err and "config.yaml" in err and "--output-root" in err and "is required" not in err


# --- the key ------------------------------------------------------------------------------------------------------------


def test_a_missing_key_is_named_before_anything_starts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    message = missing_api_key(ProviderConfig(name="openai"))
    assert message and "OPENAI_API_KEY" in message and "not set" in message and "Nothing was started" in message and ".env" in message
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert missing_api_key(ProviderConfig(name="openai")) is None


def test_the_key_a_yaml_names_is_the_one_asked_for(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MY_KEY", raising=False)
    message = missing_api_key(ProviderConfig(name="openai", base_url="http://127.0.0.1:9/v1", api_key_env="MY_KEY"))
    assert message and "MY_KEY" in message


@pytest.mark.parametrize("provider", [
    ProviderConfig(name="ollama"), ProviderConfig(name="claude_cli"), ProviderConfig(name="vscode_extension"),
    ProviderConfig(name="openai", base_url="http://127.0.0.1:8772/v1"),     # a gateway of your own that never named a key
])
def test_nothing_is_asked_of_a_provider_that_needs_no_key(monkeypatch: pytest.MonkeyPatch, provider: ProviderConfig) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert missing_api_key(provider) is None


@pytest.mark.asyncio
async def test_the_command_line_stops_with_that_message_and_exit_2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import launch

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = tmp_path / "quest.yaml"
    cfg.write_text(f"topic: t\ntitle: t\nprovider:\n  name: openai\noutput:\n  output_dir: {tmp_path.as_posix()}/outputs\n", encoding="utf-8")
    assert await launch.main_async(parse_args(["--config", str(cfg), "--no-axon-sidecar"])) == 2
    assert "OPENAI_API_KEY" in capsys.readouterr().err
    assert not (tmp_path / "outputs").exists() or not list((tmp_path / "outputs").iterdir()), "no quest was started"


# --- the bundled example ------------------------------------------------------------------------------------------------


def test_the_bundled_example_finishes_on_its_own_and_uses_a_provider_the_readme_lists() -> None:
    cfg = Config.from_yaml(ROOT / "examples" / "integrator_bakeoff" / "config.yaml")
    assert cfg.pauses.papers is False and cfg.pauses.review == "off", "the README promises a paper from this one command; neither stop may interrupt it"
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"`{cfg.provider.name}`" in readme


def test_the_console_is_not_a_wall_of_request_urls() -> None:
    import logging

    import launch

    logging.getLogger("httpx").setLevel(logging.INFO)
    launch._quiet_network_logs()
    assert logging.getLogger("httpx").level == logging.WARNING and logging.getLogger("httpcore").level == logging.WARNING
