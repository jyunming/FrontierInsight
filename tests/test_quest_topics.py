"""The general-science test topics in dev/quest-topics/ must stay loadable and field-neutral."""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from core.config import Config, page_limit_from_text

TOPICS = sorted((Path(__file__).resolve().parents[1] / "dev" / "quest-topics").glob("*.yaml"))


def test_there_are_several_topics_across_fields() -> None:
    assert len(TOPICS) >= 8


@pytest.mark.parametrize("path", TOPICS, ids=lambda p: p.stem)
def test_each_topic_loads_and_states_a_four_page_limit(path: Path) -> None:
    Config.from_yaml(path)
    topic = yaml.safe_load(path.read_text(encoding="utf-8"))["topic"]
    assert page_limit_from_text(topic) == 4


@pytest.mark.parametrize("path", TOPICS, ids=lambda p: p.stem)
def test_no_topic_is_tied_to_one_industry(path: Path) -> None:
    text = path.read_text(encoding="utf-8").lower()
    for word in ("lithograph", "photoresist", "euv", "duv", "mosfet"):
        assert not re.search(rf"{word}", text)


@pytest.mark.parametrize("path", TOPICS, ids=lambda p: p.stem)
def test_each_topic_runs_unattended_and_stays_independent(path: Path) -> None:
    cfg = Config.from_yaml(path)
    assert cfg.pauses.papers is False
    assert cfg.pauses.review == "off"
    assert cfg.knowledge.write_back_quests is False
