"""The interview recommends the research profile, and the answer reaches the config on every surface (CLI, web, VSCode)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from core.config import Config
from core.interview import QUESTIONS, InterviewAnswers, answers_to_yaml, export_schema_json
from tests.test_interview_roundtrip import _full_answers
from tests.test_vscode_interview_yaml import EXT, _emit
from tests.test_web_interview import _client, _ok_answers_payload

PANEL = ["methodologist", "statistician", "reproducibility", "devil_advocate"]


def _cfg(tmp_path: Path, answers: InterviewAnswers) -> tuple[str, Config]:
    text = answers_to_yaml(answers, frontend="cli")
    path = tmp_path / "quest.yaml"
    path.write_text(text, encoding="utf-8")
    return text, Config.from_yaml(path)


def test_the_question_is_an_advanced_one_that_recommends_research_and_defaults_to_the_default_profile() -> None:
    (q,) = [q for q in QUESTIONS if q.id == "rigor_profile"]
    assert q.tier == 3 and q.default == "default" and [c.value for c in q.choices] == ["default", "research"]
    assert "recommended for a simulation study" in q.prompt and "recommended for a simulation study" in q.choices[1].label
    schema_q = next(x for x in export_schema_json()["questions"] if x["id"] == "rigor_profile")
    assert [c["value"] for c in schema_q["choices"]] == ["default", "research"]


def test_the_cli_writes_the_profile_only_when_asked_and_the_config_then_applies_it(tmp_path: Path) -> None:
    answers = replace(_full_answers(), review_panel=list(PANEL), pause_for_plan=False)
    (tmp_path / "a").mkdir()
    text, cfg = _cfg(tmp_path / "a", answers)
    assert "rigor_profile" not in text and cfg.rigor_profile == "default" and cfg.pauses.plan == "off"
    (tmp_path / "b").mkdir()
    text, cfg = _cfg(tmp_path / "b", replace(answers, rigor_profile="research"))
    assert 'rigor_profile: "research"' in text.splitlines()
    assert cfg.rigor_profile == "research" and cfg.pauses.plan == "ask" and cfg.execution.split_failure == "block"
    assert cfg.engine.oracle_check == "block" and cfg.engine.cross_check_verify is True


def test_an_empty_review_panel_does_not_contradict_the_profile(tmp_path: Path) -> None:
    """The interview writes review_panel: [] for a single reviewer; beside the research profile that would be refused."""
    answers = replace(_full_answers(), review_panel=[], pause_for_plan=False)
    (tmp_path / "d").mkdir()
    text, cfg = _cfg(tmp_path / "d", answers)
    assert "review_panel: []" in text and cfg.engine.review_panel == []
    (tmp_path / "r").mkdir()
    text, cfg = _cfg(tmp_path / "r", replace(answers, rigor_profile="research"))
    assert "review_panel" not in text and cfg.engine.review_panel == PANEL


def test_the_profile_survives_an_update_of_the_other_answers(tmp_path: Path) -> None:
    from core.interview_update import load_current_answers, rewrite_yaml_with_new_answers

    quest = tmp_path / "quest"
    quest.mkdir()
    answers = replace(_full_answers(), review_panel=list(PANEL), rigor_profile="research")
    (quest / "config.yaml").write_text(answers_to_yaml(answers, frontend="cli"), encoding="utf-8")
    current, _path, raw = load_current_answers(quest)
    assert current.rigor_profile == "research"
    (quest / "config.yaml").write_text(rewrite_yaml_with_new_answers(raw, replace(current, budget="3 hours")), encoding="utf-8")
    assert Config.from_yaml(quest / "config.yaml").rigor_profile == "research"


def test_the_web_form_carries_the_profile_and_refuses_an_unknown_one(tmp_path: Path) -> None:
    client = _client(tmp_path)
    body = _ok_answers_payload()
    body["rigor_profile"] = "research"
    res = client.post("/api/interview/submit", json=body)
    assert res.status_code == 200, res.text
    cfg = Config.from_yaml(Path(res.json()["yaml_path"]))
    assert cfg.rigor_profile == "research" and cfg.pauses.plan == "ask"

    res = client.post("/api/interview/submit", json=_ok_answers_payload())
    assert Config.from_yaml(Path(res.json()["yaml_path"])).rigor_profile == "default"

    body["rigor_profile"] = "strict"
    assert client.post("/api/interview/submit", json=body).status_code == 400
    page = (Path(__file__).resolve().parent.parent / "web" / "static" / "interview.html").read_text(encoding="utf-8")
    assert "rigor_profile: 'default'" in page


def test_vscode_emits_the_profile_and_loads(tmp_path: Path) -> None:
    yaml_text, cfg = _emit(tmp_path, rigor_profile="research", review_panel=list(PANEL))
    assert 'rigor_profile: "research"' in yaml_text.splitlines()
    assert cfg.rigor_profile == "research" and cfg.pauses.plan == "ask" and cfg.engine.review_panel == PANEL
    yaml_text, cfg = _emit(tmp_path, rigor_profile="research", review_panel=[])
    assert "review_panel" not in yaml_text and cfg.engine.review_panel == PANEL, "an empty panel would contradict the profile"
    yaml_text, cfg = _emit(tmp_path, review_panel=[])
    assert "rigor_profile" not in yaml_text and "review_panel: []" in yaml_text and cfg.rigor_profile == "default"
    ts = (EXT / "src" / "interview.ts").read_text(encoding="utf-8")
    assert '{ label: "Rigor profile", value: "rigor_profile" }' in ts and 'case "rigor_profile"' in ts
    assert "recommended for a simulation study" in ts
    assert 'rigor_profile?: "default" | "research";' in (EXT / "src" / "interview-core.ts").read_text(encoding="utf-8")


@pytest.mark.parametrize("surface", ["cli", "vscode"])
def test_the_two_emitters_write_the_same_profile_line(tmp_path: Path, surface: str) -> None:
    if surface == "cli":
        (tmp_path / "c").mkdir()
        text, _cfg_ = _cfg(tmp_path / "c", replace(_full_answers(), review_panel=list(PANEL), rigor_profile="research"))
    else:
        text, _cfg_ = _emit(tmp_path, rigor_profile="research", review_panel=list(PANEL))
    lines = text.splitlines()
    assert lines.index('rigor_profile: "research"') > lines.index([l for l in lines if l.startswith("title:")][0])
    assert lines.index('rigor_profile: "research"') < lines.index("provider:")
