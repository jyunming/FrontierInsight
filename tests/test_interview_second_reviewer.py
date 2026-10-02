"""The interview asks for the second reviewer's model when the result is for research or a decision.

``rigor_profile: research`` needs the review panel's reviewers not all on one model. A quest made by the interview used
to stop once on its first run to ask for one (``Engine._review_models_stop``). Now every interface asks right after
"What is the result for?", and offers "I only have one model": the quest then runs, and its result says the review was
one model's view and is never publication-ready.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from core import evidence
from core.config import Config
from core.engine import Engine
from core.interview import (
    ONE_MODEL_ANSWER, QUESTIONS, InterviewAnswers, answers_to_yaml, export_schema_json, question_applies,
    questions_for_tier, second_reviewer_choices,
)
from tests.test_evidence import ON, _quest, _state
from tests.test_interview_roundtrip import _full_answers
from tests.test_vscode_interview_yaml import EXT, _emit
from tests.test_web_interview import _client, _ok_answers_payload

PANEL = ["methodologist", "statistician", "reproducibility", "devil_advocate"]
(QUESTION,) = [q for q in QUESTIONS if q.id == "second_reviewer_model"]


def _answers(**over: object) -> InterviewAnswers:
    base = replace(_full_answers(), review_panel=list(PANEL), result_use="research", survey_mode=False,
                   ensemble_profile="off", ensemble_models="")
    return replace(base, **over)  # type: ignore[arg-type]


def _cfg(tmp_path: Path, answers: InterviewAnswers) -> tuple[str, Config]:
    text = answers_to_yaml(answers, frontend="cli")
    path = tmp_path / "quest.yaml"
    path.write_text(text, encoding="utf-8")
    return text, Config.from_yaml(path)


# ---- the question ----


def test_the_question_is_asked_on_every_interface_right_after_the_model() -> None:
    assert QUESTION.tier == 1 and set(QUESTION.frontends) == {"cli", "serve", "vscode"}
    assert QUESTION.kind == "single" and QUESTION.allow_other and QUESTION.default is None
    assert not QUESTION.mid_quest_editable
    assert QUESTION.label == "A different model for one reviewer"
    assert "Four AI reviewers" in QUESTION.prompt and "I only have one model" in QUESTION.prompt
    assert "statistician" not in QUESTION.prompt, "plain words: the statistics reviewer"
    assert [c.value for c in QUESTION.choices] == [ONE_MODEL_ANSWER]
    assert QUESTION.choices[0].label == "I only have one model"
    ids = [q.id for q in questions_for_tier(1, "cli")]
    assert ids.index("second_reviewer_model") == ids.index("provider_model") + 1
    schema_q = next(q for q in export_schema_json()["questions"] if q["id"] == "second_reviewer_model")
    assert schema_q["ask_if"] == {"question": "result_use", "one_of": ["research", "decision"]}
    assert all(q["ask_if"] is None for q in export_schema_json()["questions"] if q["id"] != "second_reviewer_model")


@pytest.mark.parametrize("use, asked", [("research", True), ("decision", True), ("explore", False), (None, True)])
def test_asked_only_when_the_result_is_for_research_or_a_decision(use: str | None, asked: bool) -> None:
    """Not answered yet counts as the default answer (research)."""
    answers = {} if use is None else {"result_use": use}
    assert question_applies(QUESTION, answers) is asked
    assert all(question_applies(q, answers) for q in QUESTIONS if q.id != "second_reviewer_model")


def test_the_choices_are_the_providers_other_models_then_one_model() -> None:
    values = [c.value for c in second_reviewer_choices("openai", "gpt-5")]
    assert "gpt-5" not in values and values[0] == "gpt-5-mini" and values[-1] == ONE_MODEL_ANSWER
    assert [c.value for c in second_reviewer_choices("unknown-provider", None)] == [ONE_MODEL_ANSWER]


@pytest.mark.asyncio
@pytest.mark.parametrize("use, typed, statistician, one", [
    ("", ["1"], "gpt-5-mini", False),            # research (default): the first model other than the quest's own
    ("2", ["I only"], None, True),               # a decision: "I only have one model", by its label
    ("3", [], None, False),                      # exploring: not asked
])
async def test_the_cli_asks_it_and_writes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    use: str, typed: list[str], statistician: str | None, one: bool,
) -> None:
    from tests.test_interview_e2e_cli import _new

    cfg = await _new(tmp_path, monkeypatch, ["Second reviewer probe", use, "1", "1", *typed, ""])
    shown = capsys.readouterr().out
    assert ("A different model for one reviewer" in shown) is bool(typed)
    assert cfg.provider.model == "gpt-5"
    assert (cfg.provider.node_models or {}).get("review_panel.statistician") == statistician
    assert cfg.engine.one_model_review is one
    if typed:
        plan = shown.split("Review before launch", 1)[1]
        assert "A different model for one reviewer" in plan, "the review screen's model card shows the answer"
        assert ("gpt-5-mini" in plan) is (statistician is not None)
        assert ("I only have one model" in plan) is one


@pytest.mark.asyncio
async def test_the_cli_does_not_take_the_quests_own_model_typed_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """Typed in through "Other", the quest's own model would leave every reviewer on it: asked again."""
    from tests.test_interview_e2e_cli import _new

    other = str(len(second_reviewer_choices("openai", "gpt-5")) + 1)
    cfg = await _new(tmp_path, monkeypatch, ["Same model probe", "", "1", "1", other, "gpt-5", "2", ""])
    assert "the model the quest runs on" in capsys.readouterr().out
    assert (cfg.provider.node_models or {}).get("review_panel.statistician") == "gpt-4o"


# ---- what it writes (CLI / web: answers_to_yaml; VS Code: its TypeScript mirror) ----


def test_a_model_goes_on_the_statistician_and_the_quest_does_not_stop(tmp_path: Path) -> None:
    text, cfg = _cfg(tmp_path, _answers(second_reviewer_model="gpt-5-mini"))
    assert cfg.provider.node_models == {"review_panel.statistician": "gpt-5-mini"}
    assert "one_model_review" not in text and cfg.engine.one_model_review is False
    engine = Engine(cfg.model_copy(update={"output": cfg.output.model_copy(update={"output_dir": tmp_path / "out"})}))
    assert engine._reviewer_model("statistician") == "gpt-5-mini"
    assert engine._review_models_stop() is None


def test_one_model_is_written_and_the_quest_runs(tmp_path: Path) -> None:
    text, cfg = _cfg(tmp_path, _answers(second_reviewer_model=ONE_MODEL_ANSWER))
    assert "  one_model_review: true" in text.splitlines() and ONE_MODEL_ANSWER not in text
    assert cfg.engine.one_model_review is True and not cfg.provider.node_models
    engine = Engine(cfg.model_copy(update={"output": cfg.output.model_copy(update={"output_dir": tmp_path / "out"})}))
    assert engine._review_models_stop() is None
    assert not (engine.quest_root / "NEXT_STEP.md").exists()


def test_the_answer_merges_into_the_per_node_models_and_a_named_reviewer_model_wins(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    _text, cfg = _cfg(tmp_path / "a", _answers(second_reviewer_model="m-two", node_models="poster:m-cheap"))
    assert cfg.provider.node_models == {"poster": "m-cheap", "review_panel.statistician": "m-two"}
    (tmp_path / "b").mkdir()
    _text, cfg = _cfg(tmp_path / "b", _answers(second_reviewer_model="m-two",
                                               node_models="review_panel.methodologist:m-own"))
    assert cfg.provider.node_models == {"review_panel.methodologist": "m-own"}
    # A named reviewer on the quest's own model (gpt-5 here) leaves every reviewer on it: the answer is merged in too.
    (tmp_path / "c").mkdir()
    _text, cfg = _cfg(tmp_path / "c", _answers(second_reviewer_model="m-two",
                                               node_models="review_panel.methodologist:gpt-5"))
    assert cfg.provider.node_models == {"review_panel.methodologist": "gpt-5", "review_panel.statistician": "m-two"}


def test_one_model_is_not_written_when_a_reviewer_is_on_another_model_after_all(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    text, cfg = _cfg(tmp_path / "a", _answers(second_reviewer_model=ONE_MODEL_ANSWER,
                                              node_models="review_panel.methodologist:m2"))
    assert "one_model_review" not in text and cfg.engine.one_model_review is False
    # ... but a named reviewer on the quest's own model does not count as another model.
    (tmp_path / "b").mkdir()
    text, cfg = _cfg(tmp_path / "b", _answers(second_reviewer_model=ONE_MODEL_ANSWER,
                                              node_models="review_panel.methodologist:gpt-5"))
    assert cfg.engine.one_model_review is True
    for node_models, one in (("review_panel.methodologist:m2", False), ("review_panel.methodologist:m-main", True)):
        ts_text, ts_cfg = _emit(tmp_path, result_use="research", review_panel=list(PANEL), provider_model="m-main",
                                node_models=node_models, second_reviewer_model=ONE_MODEL_ANSWER)
        assert ts_cfg.engine.one_model_review is one, ts_text


def test_exploring_or_not_asked_writes_neither(tmp_path: Path) -> None:
    for i, answers in enumerate((_answers(result_use="explore", review_panel=[], second_reviewer_model="m-two"),
                                 _answers(result_use="explore", review_panel=[], second_reviewer_model=ONE_MODEL_ANSWER),
                                 _answers(second_reviewer_model=""))):
        (tmp_path / str(i)).mkdir()
        text, cfg = _cfg(tmp_path / str(i), answers)
        assert "one_model_review" not in text and "review_panel.statistician" not in text
        assert cfg.engine.one_model_review is False


@pytest.mark.parametrize("answer", ["m-two", ONE_MODEL_ANSWER])
def test_vscode_writes_the_same_as_the_cli(tmp_path: Path, answer: str) -> None:
    ts_text, ts_cfg = _emit(tmp_path, result_use="decision", review_panel=list(PANEL), provider_model="m-main",
                            node_models="poster:m-cheap", second_reviewer_model=answer)
    (tmp_path / "py").mkdir()
    py_text, py_cfg = _cfg(tmp_path / "py", replace(
        _answers(), provider="vscode_extension", provider_model="m-main", result_use="decision",
        node_models="poster:m-cheap", second_reviewer_model=answer,
    ))
    assert ts_cfg.provider.node_models == py_cfg.provider.node_models
    assert ts_cfg.engine.one_model_review == py_cfg.engine.one_model_review == (answer == ONE_MODEL_ANSWER)

    def block(text: str, start: str) -> list[str]:
        lines = text.splitlines()
        i = lines.index(start)
        out = []
        for line in lines[i + 1:]:
            if not line.startswith("    "):
                break
            out.append(line)
        return out

    assert block(ts_text, "  node_models:") == block(py_text, "  node_models:")
    # Exploring: nothing, from VS Code as from the CLI.
    ts_text, ts_cfg = _emit(tmp_path, result_use="explore", review_panel=[], second_reviewer_model=answer)
    assert "one_model_review" not in ts_text and "review_panel.statistician" not in ts_text


def test_vscode_asks_it_after_what_the_result_is_for_and_leaves_the_chat_model_out() -> None:
    ts = (EXT / "src" / "interview.ts").read_text(encoding="utf-8")
    run = ts[ts.index("export async function runInterview("):]
    run = run[:run.index("\n}\n")]
    assert run.index("what is the result for?") < run.index("pickSecondReviewerModel(primaryModel)")
    assert 'resultUse !== "explore"' in run
    assert "I only have one model" in ts and "m.family === primary.family" in ts
    assert "[primary.id, primary.family].includes(s.trim())" in ts, "a typed name equal to the chat model is refused"
    # Changing the answer to research on the review screen asks it then.
    case = ts[ts.index('case "result_use"'):]
    assert "pickSecondReviewerModel(primaryModel)" in case[:case.index("return;\n        }")]
    ext = (EXT / "src" / "extension.ts").read_text(encoding="utf-8")
    assert "runInterview(stream, userPickedModel)" in ext
    core_ts = (EXT / "src" / "interview-core.ts").read_text(encoding="utf-8")
    assert f'export const ONE_MODEL_ANSWER = "{ONE_MODEL_ANSWER}";' in core_ts


def test_vscode_uses_the_same_words_checks_typed_names_and_will_not_launch_without_an_answer() -> None:
    import re

    ts = (EXT / "src" / "interview.ts").read_text(encoding="utf-8")
    m = re.search(r"export const SECOND_REVIEWER_PROMPT =\s*((?:\"[^\"]*\"\s*\+?\s*)+);", ts)
    assert m, "SECOND_REVIEWER_PROMPT not found"
    assert "".join(re.findall(r'"([^"]*)"', m.group(1))) == QUESTION.prompt
    assert f'export const SECOND_REVIEWER_LABEL = "{QUESTION.label}";' in ts
    assert "placeHolder: SECOND_REVIEWER_PROMPT" in ts
    # A typed name must be one VS Code offers (by id or family), or the bridge would use the chat model instead.
    assert "selectChatModels({ id: name })" in ts and "selectChatModels({ family: name })" in ts
    assert "await vscodeOffersModel(typed)" in ts
    # Launch with research or a decision and no answer asks it, and does not launch without one.
    launch = ts[ts.index('if (action.value === "launch") {'):]
    launch = launch[:launch.index("return answers;")]
    assert 'answers.result_use !== "explore" && !answers.second_reviewer_model' in launch
    assert "pickSecondReviewerModel(primaryModel)" in launch and "continue;" in launch


# ---- the web form ----


def test_the_web_form_carries_the_answer(tmp_path: Path) -> None:
    client = _client(tmp_path)
    for answer, statistician, one in (("m-two", "m-two", False), (ONE_MODEL_ANSWER, None, True), ("", None, False)):
        res = client.post("/api/interview/submit",
                          json={**_ok_answers_payload(), "result_use": "research", "second_reviewer_model": answer})
        assert res.status_code == 200, res.text
        cfg = Config.from_yaml(Path(res.json()["yaml_path"]))
        assert (cfg.provider.node_models or {}).get("review_panel.statistician") == statistician
        assert cfg.engine.one_model_review is one
    bad = client.post("/api/interview/submit", json={**_ok_answers_payload(), "second_reviewer_model": ["a"]})
    assert bad.status_code == 400
    same = client.post("/api/interview/submit", json={**_ok_answers_payload(), "provider_model": "m-main",
                                                      "second_reviewer_model": "m-main"})
    assert same.status_code == 400 and "the model the quest runs on" in same.text
    page = (Path(__file__).resolve().parent.parent / "web" / "static" / "interview.html").read_text(encoding="utf-8")
    assert "function questionApplies" in page and "renderSecondReviewerOptions" in page
    assert "String(m.value) !== primary" in page, "the quest's own model is left out of the list"
    assert "out.second_reviewer_model === out.provider_model" in page, "nor can it be typed in"
    # Research or a decision needs the quest's model named, so "the same model" can always be checked.
    for use in ("research", "decision"):
        blank = client.post("/api/interview/submit", json={**_ok_answers_payload(), "result_use": use,
                                                           "provider_model": "", "second_reviewer_model": "gpt-5"})
        assert blank.status_code == 400 and "provider_model" in blank.text
    assert client.post("/api/interview/submit", json={**_ok_answers_payload(), "result_use": "explore",
                                                      "provider_model": ""}).status_code == 200
    assert "!String(out.provider_model || '').trim()" in page
    # The update form hides a question by the schema's own editability, not by having a condition.
    assert "!(q.ask_if && !schema.editable_fields.includes(q.id))" in page
    # A draft's model that only the live list has is kept until it arrives.
    assert "pendingModelValues[qid] = target" in page and "applyPendingModel('provider_model')" in page


# ---- a later --update keeps it ----


@pytest.mark.parametrize("answer", ["m-two", ONE_MODEL_ANSWER])
def test_an_update_keeps_the_answer(tmp_path: Path, answer: str) -> None:
    from core.interview_update import load_current_answers, rewrite_yaml_with_new_answers

    quest = tmp_path / "quest"
    quest.mkdir()
    (quest / "config.yaml").write_text(answers_to_yaml(_answers(second_reviewer_model=answer)), encoding="utf-8")
    current, _path, raw = load_current_answers(quest)
    (quest / "config.yaml").write_text(rewrite_yaml_with_new_answers(raw, replace(current, budget="3 hours")),
                                       encoding="utf-8")
    cfg = Config.from_yaml(quest / "config.yaml")
    assert cfg.engine.one_model_review is (answer == ONE_MODEL_ANSWER)
    assert (cfg.provider.node_models or {}).get("review_panel.statistician") == (None if answer == ONE_MODEL_ANSWER else "m-two")


@pytest.mark.parametrize("node_models, kept", [
    ({"review_panel.statistician": "m-two"}, False),   # a reviewer on another model: the flag no longer holds
    ({"review_panel.statistician": "gpt-5"}, True),    # on the quest's own model: still one model
    ({"poster": "m-cheap"}, True),                     # not a reviewer
])
def test_an_update_that_names_a_reviewer_model_drops_one_model_review(
    tmp_path: Path, node_models: dict[str, str], kept: bool,
) -> None:
    import yaml

    from core.interview_update import load_current_answers, rewrite_yaml_with_new_answers

    quest = tmp_path / "quest"
    quest.mkdir()
    (quest / "config.yaml").write_text(answers_to_yaml(_answers(second_reviewer_model=ONE_MODEL_ANSWER)),
                                       encoding="utf-8")
    current, _path, raw = load_current_answers(quest)
    node_models_answer = ", ".join(f"{k}:{v}" for k, v in node_models.items())
    (quest / "config.yaml").write_text(
        rewrite_yaml_with_new_answers(raw, replace(current, node_models=node_models_answer)), encoding="utf-8")
    cfg = Config.from_yaml(quest / "config.yaml")
    assert cfg.engine.one_model_review is kept
    assert (cfg.provider.node_models or {}) == node_models
    # A hand-edited config (the per-node models added straight to the YAML) is handled the same way.
    raw2 = yaml.safe_load(answers_to_yaml(_answers(second_reviewer_model=ONE_MODEL_ANSWER)))
    raw2["provider"]["node_models"] = dict(node_models)
    current2 = replace(current, node_models="")
    assert ("one_model_review" in (yaml.safe_load(rewrite_yaml_with_new_answers(raw2, current2))["engine"])) is kept


@pytest.mark.asyncio
async def test_the_cli_asks_again_when_other_is_left_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    from tests.test_interview_e2e_cli import _new

    other = str(len(second_reviewer_choices("openai", "gpt-5")) + 1)
    cfg = await _new(tmp_path, monkeypatch, ["Empty other probe", "", "1", "1", other, "", "1", ""])
    assert "nothing typed" in capsys.readouterr().out
    assert (cfg.provider.node_models or {}).get("review_panel.statistician") == "gpt-5-mini"


# ---- the result ----


def test_one_model_review_is_a_publication_ready_gap_under_research(tmp_path: Path) -> None:
    root = _quest(tmp_path, protocol_status="ok", oracle_status="ok")
    research = {**ON, "rigor_profile": "research"}
    one = evidence.assess(root, _state(), settings={**research, "one_model_review": True})
    assert one["status"] != "publication_ready"
    assert any("every reviewer used one model" in g and "one model's view" in g for g in one["gaps"])
    assert not any("every reviewer used one model" in g for g in evidence.assess(root, _state(), settings=research)["gaps"])
    # Outside research there is no requirement, so no such gap either.
    default = evidence.assess(root, _state(), settings={**ON, "one_model_review": True})
    assert not any("every reviewer used one model" in g for g in default["gaps"])


def test_one_model_review_is_allowed_beside_the_research_profile() -> None:
    cfg = Config.model_validate({"topic": "t", "rigor_profile": "research", "engine": {"one_model_review": True}})
    assert cfg.engine.one_model_review is True and cfg.rigor_profile == "research"
