"""FI runs from any folder, not only from its own checkout: the quest, the files it
is given and its outputs live relative to where it is run.

The user's project is not the FI folder. A relative ``output.output_dir`` and a
relative ``execution.inputs`` mean "in the folder I ran this from"."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig,
    ProviderConfig,
)
from core.engine import Engine
from core.state_dump import dump_state
from tests.test_engine_smoke import _classify, _fake_response_for

REPO = Path(__file__).resolve().parent.parent

_READS_THE_EXAMPLE = """\
import json, os
d = os.environ["FI_INPUT_DIR"]
text = open(os.path.join(d, "example", "setup.in"), encoding="utf-8").read().strip()
print("RESULT_JSON: " + json.dumps({"first_line": text.splitlines()[0]}))
"""


def _project(tmp_path: Path) -> Path:
    project = tmp_path / "my_project"
    (project / "example").mkdir(parents=True)
    (project / "example" / "setup.in").write_text("temperature 300\n", encoding="utf-8")
    return project


@pytest.mark.asyncio
async def test_a_quest_run_from_another_folder_keeps_everything_relative_to_it(
    tmp_path: Path, monkeypatch,
) -> None:
    project = _project(tmp_path)
    monkeypatch.chdir(project)
    before = {p.name for p in REPO.iterdir()}

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        if _classify(prompt) == "Implementation":
            return json.dumps({"code": _READS_THE_EXAMPLE, "deps": []})
        return _fake_response_for(prompt)

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)
    # Written the way a user's YAML is: relative paths, nothing absolute.
    cfg = Config(
        topic="run from anywhere", title="anywhere",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, auto_accept_on_pass=True),
        execution=ExecutionConfig(sandbox="venv", timeout_s=120, inputs=["example"]),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=Path("outputs")),
    )

    engine = Engine(cfg)
    artifacts = await engine.run()

    quest = project / "outputs" / engine.quest_id
    assert engine.quest_root == quest.resolve()
    assert artifacts.paper_md == quest.resolve() / "paper" / "paper.md" and artifacts.paper_md.is_file()
    assert (quest / ".fi" / "state.sqlite").is_file() and (quest / ".fi" / "run.log").is_file()
    assert (quest / "inputs" / "examples" / "example" / "setup.in").is_file()
    assert artifacts.raw_state["result_json"] == {"first_line": "temperature 300"}
    assert f"quest: {engine.quest_id}" in dump_state(quest)
    assert {p.name for p in REPO.iterdir()} == before, "nothing was written into the FI checkout"


def test_resume_looks_where_the_relative_output_is_and_says_where(
    tmp_path: Path, monkeypatch,
) -> None:
    """Run from a folder that is not the project, a relative output_dir points at
    the wrong place. The message names the place it looked."""
    import launch

    project = _project(tmp_path)
    (project / "outputs").mkdir()
    monkeypatch.chdir(tmp_path)
    err = launch._validate_resume_quest_id("1700000000-x-abcdef", Path("outputs"))
    assert err is not None and str((tmp_path / "outputs").resolve()) in err.replace("\\\\", "\\")
