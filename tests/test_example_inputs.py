"""Example files the user supplies (`execution.inputs`): copied in whatever their
type, shown to the design and the code-writing steps, and found by the experiment
through FI_INPUT_DIR. Before this only tabular data files reached only the analysis,
so a `.in` or `.yaml` setup never reached the code that had to use it."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from core import example_inputs as ei

LOG = logging.getLogger("test-inputs")


def _deck(root: Path) -> Path:
    deck = root / "deck"
    deck.mkdir(parents=True)
    (deck / "setup.in").write_text("temperature 300\nsteps 100\n", encoding="utf-8")
    (deck / "run.yaml").write_text("solver: fast\n", encoding="utf-8")
    (deck / "mesh.bin").write_bytes(b"\x00\x01\x02" * 200)
    (deck / "notes.txt").write_text("y" * 9000, encoding="utf-8")
    return deck


def test_any_file_type_is_copied_and_a_folder_keeps_its_name(tmp_path: Path) -> None:
    deck = _deck(tmp_path / "src")
    single = tmp_path / "src" / "solo.cfg"
    single.write_text("a=1\n", encoding="utf-8")
    quest = tmp_path / "quest"

    files = ei.stage_inputs([str(deck), str(single)], quest, LOG)

    assert files == [
        "inputs/examples/deck/mesh.bin", "inputs/examples/deck/notes.txt",
        "inputs/examples/deck/run.yaml", "inputs/examples/deck/setup.in",
        "inputs/examples/solo.cfg",
    ]
    assert (quest / "inputs/examples/deck/setup.in").read_text(encoding="utf-8").startswith("temperature")


def test_staging_twice_does_not_rewrite_a_file_you_edited_while_paused(tmp_path: Path) -> None:
    deck = _deck(tmp_path / "src")
    quest = tmp_path / "quest"
    ei.stage_inputs([str(deck)], quest, LOG)
    edited = quest / "inputs/examples/deck/run.yaml"
    edited.write_text("solver: fast\n# edited by hand, and longer\n", encoding="utf-8")

    ei.stage_inputs([str(deck)], quest, LOG)

    assert "edited by hand" in edited.read_text(encoding="utf-8")


def test_a_source_that_changed_is_copied_again(tmp_path: Path) -> None:
    deck = _deck(tmp_path / "src")
    quest = tmp_path / "quest"
    ei.stage_inputs([str(deck)], quest, LOG)

    (deck / "setup.in").write_text("temperature 500\nsteps 100\nextra 1\n", encoding="utf-8")
    ei.stage_inputs([str(deck)], quest, LOG)

    assert "temperature 500" in (quest / "inputs/examples/deck/setup.in").read_text(encoding="utf-8")


def test_a_missing_path_stops_with_its_name(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="execution.inputs: 'nope.in' does not exist"):
        ei.stage_inputs(["nope.in"], tmp_path / "quest", LOG)


def test_a_path_far_too_large_is_refused(tmp_path: Path, monkeypatch) -> None:
    deck = _deck(tmp_path / "src")
    monkeypatch.setattr(ei, "MAX_COPY_BYTES", 100)
    with pytest.raises(ValueError, match="the quest will copy"):
        ei.stage_inputs([str(deck)], tmp_path / "quest", LOG)
    assert not (tmp_path / "quest" / "inputs").exists(), "nothing is copied before the check"


def test_the_block_shows_small_text_files_and_says_what_it_cannot(tmp_path: Path) -> None:
    quest = tmp_path / "quest"
    ei.stage_inputs([str(_deck(tmp_path / "src"))], quest, LOG)

    block = ei.render_block(quest)

    assert "temperature 300" in block and "solver: fast" in block
    assert "deck/mesh.bin (600 bytes)" in block and "(binary file, 600 bytes; not shown)" in block
    assert "(truncated)" in block, "a 9000-character file is cut, and says so"
    assert "FI_INPUT_DIR" in block and "NEW simulation" in block


def test_no_files_no_block(tmp_path: Path) -> None:
    assert ei.render_block(tmp_path / "quest") == ""


def test_files_dropped_in_by_hand_count(tmp_path: Path) -> None:
    quest = tmp_path / "quest"
    drop = quest / "inputs" / "examples"
    drop.mkdir(parents=True)
    (drop / "mine.in").write_text("dropped\n", encoding="utf-8")
    assert ei.list_inputs(quest) == ["inputs/examples/mine.in"]
    assert "dropped" in ei.render_block(quest)


# --- through a real quest ---------------------------------------------------

_READS_THE_DECK = """\
import glob, json, os
d = os.environ["FI_INPUT_DIR"]
first = open(os.path.join(d, "deck", "setup.in"), encoding="utf-8").read().strip()
print("RESULT_JSON: " + json.dumps({"first_line": first.splitlines()[0], "n_files": len(glob.glob(d + "/**/*", recursive=True))}))
"""


def _config(tmp_path: Path, inputs: list[str]):
    from core.config import (
        Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig,
        ProviderConfig,
    )

    return Config(
        topic="example inputs probe", title="example-inputs",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, auto_accept_on_pass=True),
        execution=ExecutionConfig(sandbox="venv", timeout_s=120, inputs=inputs),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "outputs"),
    )


@pytest.mark.asyncio
async def test_the_design_and_the_code_are_written_from_the_examples_and_the_run_finds_them(
    tmp_path: Path, monkeypatch,
) -> None:
    from core.engine import Engine
    from tests.test_engine_smoke import _classify, _fake_response_for

    deck = _deck(tmp_path / "src")
    prompts: dict[str, str] = {}

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        tag = _classify(prompt)
        prompts.setdefault(tag, prompt)
        if tag == "Implementation":
            return json.dumps({"code": _READS_THE_DECK, "deps": []})
        return _fake_response_for(prompt)

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)

    artifacts = await Engine(_config(tmp_path, [str(deck)])).run()

    assert "temperature 300" in prompts["Experiment Design"]
    assert "temperature 300" in prompts["Implementation"]
    result = artifacts.raw_state["result_json"]
    assert result["first_line"] == "temperature 300", "the experiment could not read FI_INPUT_DIR"
    assert result["n_files"] >= 4


@pytest.mark.asyncio
async def test_without_inputs_the_prompts_say_none_were_supplied(tmp_path: Path, monkeypatch) -> None:
    from core.engine import Engine
    from tests.test_engine_smoke import _classify, _fake_response_for

    seen: list[str] = []

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        if _classify(prompt) == "Experiment Design":
            seen.append(prompt)
        return _fake_response_for(prompt)

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)
    await Engine(_config(tmp_path, [])).run()
    assert "(none supplied)" in seen[0]


@pytest.mark.asyncio
async def test_a_missing_input_stops_the_quest_before_any_llm_call(tmp_path: Path, monkeypatch) -> None:
    from core.engine import Engine
    from tests.test_engine_smoke import _fake_response_for

    calls: list[str] = []

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        calls.append("call")
        return _fake_response_for(messages[-1]["content"])

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)

    with pytest.raises(FileNotFoundError, match="execution.inputs"):
        await Engine(_config(tmp_path, [str(tmp_path / "no-such-deck")])).run()
    assert calls == []
