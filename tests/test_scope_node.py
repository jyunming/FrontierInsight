"""The ``scope`` node: before the plan, a few studies the topic could mean, for a person to pick one."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from core import scope_cards as sc
from core.config import (Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, PausesConfig,
                         ProviderConfig)
from core.engine import Engine

CARDS = {"cards": [
    {"question": "How fast does a cup of tea cool in still air?", "study_type": "measure",
     "main_results": ["cooling time"], "compared_settings": ["cup material"], "literature": [], "run_time": "minutes"},
    {"question": "Which lid keeps the tea warm longest?", "study_type": "find_best_design",
     "main_results": ["heat loss"], "compared_settings": [], "literature": [], "run_time": "an hour", "widest": True},
    {"question": "How does the starting temperature change the cooling curve?", "study_type": "measure",
     "main_results": ["cooling time", "final temperature"], "compared_settings": ["starting temperature"],
     "literature": [], "run_time": "minutes"},
]}


def _engine(tmp_path: Path, scope: str = "ask") -> Engine:
    cfg = Config(
        topic="a neutral topic about a cooling cup of tea", title="scope-node", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, clarify_mode="off", oracle_check="off"),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60), pauses=PausesConfig(scope=scope),
        knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "out"),
    )
    eng = Engine(cfg)
    eng.quest_root.mkdir(parents=True, exist_ok=True)
    eng.fi_dir.mkdir(parents=True, exist_ok=True)
    return eng


def _replies(eng: Engine, *replies: Any) -> list[str]:
    prompts: list[str] = []
    queue = list(replies)

    async def chat(prompt: str, *, node: str, **_kw: Any) -> str:
        assert node == "scope"
        prompts.append(prompt)
        return json.dumps(queue.pop(0) if len(queue) > 1 else queue[0])

    eng._chat = chat  # type: ignore[method-assign]
    return prompts


STATE: dict[str, Any] = {"topic": "a neutral topic about a cooling cup of tea", "iteration": 0}


def test_cards_are_recorded_and_nobody_chose_when_no_one_can_answer(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    _replies(eng, CARDS)
    out = asyncio.run(eng._node_scope(dict(STATE)))
    assert out == {"scope_card": None}
    record = sc.read_record(eng.quest_root)
    assert record and record["chosen_by"] == "nobody" and len(record["cards"]) == 3 and record["widest_offered"]


def test_a_staged_answer_picks_the_card_and_the_plan_prompt_names_it(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    _replies(eng, CARDS)
    (eng.quest_root / sc.ANSWER).write_text(json.dumps({"study": 3}), encoding="utf-8")
    out = asyncio.run(eng._node_scope(dict(STATE)))
    card = out["scope_card"]
    assert card["question"] == CARDS["cards"][2]["question"]
    assert sc.read_record(eng.quest_root)["chosen_by"] == "person"
    note = sc.plan_note(card)
    assert card["question"] in note  # what _node_plan appends to its prompt for a chosen card


def test_the_plan_prompt_carries_the_chosen_card(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    card = sc.read_answer(2, sc.normalize(CARDS, set())[0])
    seen: list[str] = []

    async def reply(prompt: str, *, node: str) -> dict[str, Any]:
        seen.append(prompt)
        raise RuntimeError("stop after the prompt")

    async def nothing(*_a: Any, **_k: Any) -> None:
        return None

    eng._design_reply = reply  # type: ignore[method-assign]
    eng._read_topic_before_the_prompt = nothing  # type: ignore[method-assign]
    try:
        asyncio.run(eng._node_plan({**STATE, "scope_card": card}))
    except RuntimeError:
        pass
    assert seen and card["question"] in seen[0] and "The person chose this study to run" in seen[0]


def test_the_setup_answer_file_is_not_read_as_the_study_answer(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    _replies(eng, CARDS)
    (eng.fi_dir / "clarify_answer.json").write_text(json.dumps({"study": 1, "study_type": "measure"}),
                                                    encoding="utf-8")
    out = asyncio.run(eng._node_scope(dict(STATE)))
    assert out == {"scope_card": None}


def test_a_resume_after_the_cards_does_not_ask_again(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    prompts = _replies(eng, CARDS)
    (eng.quest_root / sc.ANSWER).write_text(json.dumps({"study": 1}), encoding="utf-8")
    first = asyncio.run(eng._node_scope(dict(STATE)))
    again = asyncio.run(eng._node_scope(dict(STATE)))
    assert len(prompts) == 1 and again == first


def test_a_chosen_find_best_design_card_drives_the_study_type(tmp_path: Path) -> None:
    card = sc.normalize(CARDS, set())[0][1]
    state = {**STATE, "scope_card": card, "clarify_answers": {"study_type": "measure"}}
    assert Engine._study_type_asked(state) == "find_best_design"
    assert Engine._study_type_asked({**STATE, "clarify_answers": {"study_type": "measure"}}) == "measure"


def test_scope_off_writes_nothing(tmp_path: Path) -> None:
    eng = _engine(tmp_path, scope="off")
    prompts = _replies(eng, CARDS)
    assert asyncio.run(eng._node_scope(dict(STATE))) == {}
    assert not prompts and sc.read_record(eng.quest_root) is None


def test_a_reply_with_two_alike_cards_is_asked_again_once(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    alike = {"cards": [CARDS["cards"][0], {**CARDS["cards"][0], "question": "The same study, worded again"}]}
    prompts = _replies(eng, alike, CARDS)
    asyncio.run(eng._node_scope(dict(STATE)))
    assert len(prompts) == 2 and "gave 1 different studies" in prompts[1]
    assert len(sc.read_record(eng.quest_root)["cards"]) == 3


def test_the_answer_channel_is_asked_and_a_timeout_is_nobody(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    _replies(eng, CARDS)
    asked: list[dict[str, Any]] = []

    async def answer(questions: dict[str, Any]) -> dict[str, Any]:
        asked.append(questions)
        return {"study": "2"}

    eng._scope_answerer = answer
    out = asyncio.run(eng._node_scope(dict(STATE)))
    assert out["scope_card"]["question"] == CARDS["cards"][1]["question"]
    assert "Pick the study to run" in asked[0]["study"]["question"]

    eng2 = _engine(tmp_path / "b")
    _replies(eng2, CARDS)

    async def never(_q: dict[str, Any]) -> dict[str, Any]:
        await asyncio.sleep(30)
        return {}

    eng2._scope_answerer = never
    eng2.config.pauses.timeout_s = 0.05
    assert asyncio.run(eng2._node_scope(dict(STATE))) == {"scope_card": None}


def test_the_setup_answer_channel_is_not_the_study_answer_channel(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    _replies(eng, CARDS)
    asked: list[Any] = []

    async def setup(questions: dict[str, Any]) -> dict[str, Any]:
        asked.append(questions)
        return {"study": 1}

    eng._clarify_answerable = True
    eng._scope_answerer = None  # what Engine.run sets when it is given a setup callback and no study callback
    assert asyncio.run(eng._node_scope(dict(STATE))) == {"scope_card": None} and not asked


def test_the_quest_map_shows_the_study_cards() -> None:
    from core import rerun_from

    assert "scope" in rerun_from.NODES and rerun_from.resolve("scope") == "studies"
