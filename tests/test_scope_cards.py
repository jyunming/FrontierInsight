"""The study cards (core/scope_cards.py), the card prompt and the pauses.scope setting."""

from __future__ import annotations

import json
import string
from pathlib import Path

import pytest

from core import scope_cards as sc
from core.config import Config


def _card(question="How fast does a cup of tea cool?", study_type="measure", results=("cool rate",),
          settings=("lid on", "lid off"), literature=(), **extra):
    card = {"question": question, "study_type": study_type, "main_results": list(results),
            "compared_settings": list(settings), "literature": list(literature), "run_time": "a few minutes"}
    card.update(extra)
    return card


def test_normalize_keeps_three_distinct_cards_and_maps_compare_to_measure():
    reply = {"cards": [
        _card(study_type="compare", results=("cool rate",)),
        _card(question="Which cup shape keeps tea warm longest?", study_type="find_best_design",
              results=("final temperature",), widest=True),
        _card(question="How does the room temperature matter?", results=("half time", "final temperature")),
        {"question": "bad type", "study_type": "essay", "main_results": ["x"]},
    ]}
    cards, removed = sc.normalize(reply, {"src1"})
    assert len(cards) == 3
    assert cards[0]["study_type"] == "measure"
    assert cards[1]["widest"] is True
    assert removed == []
    assert sc.distinct_enough(cards) and sc.has_widest(cards)


def test_a_literature_line_without_a_read_source_is_removed_and_recorded():
    reply = {"cards": [_card(literature=[{"says": "tea cools by Newton's law", "source": "src1"},
                                         {"says": "lids stop all loss", "source": "src9"}])]}
    cards, removed = sc.normalize(reply, {"src1", "src2"})
    assert cards[0]["literature"] == [{"says": "tea cools by Newton's law", "source": "src1"}]
    assert "src9" not in json.dumps(cards)
    assert len(removed) == 1 and "src9" in removed[0]


def test_two_cards_with_the_same_type_and_results_are_one():
    reply = {"cards": [_card(results=("Cool  Rate",)), _card(question="Another wording?", results=("cool rate",)),
                       _card(question="Other", results=("final temperature",))]}
    cards, _ = sc.normalize(reply, set())
    assert len(cards) == 2
    assert not sc.distinct_enough(cards)


def test_normalize_keeps_at_most_four_cards():
    reply = {"cards": [_card(question=f"q{i}", results=(f"r{i}",)) for i in range(6)]}
    cards, _ = sc.normalize(reply, set())
    assert len(cards) == 4


def test_read_answer_number_sentence_and_edited_card():
    cards = [_card(results=("a",)), _card(question="second", results=("b",)), _card(question="third", results=("c",))]
    assert sc.read_answer("2", cards) is cards[1] or sc.read_answer("2", cards) == cards[1]
    assert sc.read_answer(3, cards) == cards[2]
    assert sc.read_answer({"study": "1"}, cards) == cards[0]
    own = sc.read_answer("Measure how a lid changes the cooling of tea", cards)
    assert own["question"] == "Measure how a lid changes the cooling of tea"
    assert own["study_type"] is None and own["written_by_person"] is True
    assert own["main_results"] == [] and own["literature"] == []
    edited = sc.read_answer({"study": dict(cards[1], question="second, edited")}, cards)
    assert edited["question"] == "second, edited" and edited["main_results"] == ["b"]


@pytest.mark.parametrize("answer", ["", "0", "9", "e", None, "   ", {"study": ""}, 0, 9])
def test_an_answer_that_names_no_card_is_nobody_chose(answer):
    cards = [_card(results=("a",)), _card(question="second", results=("b",)), _card(question="third", results=("c",))]
    assert sc.read_answer(answer, cards) is None


def test_write_record_keeps_the_earlier_choice_in_history(tmp_path: Path):
    cards = [_card(), _card(question="second", results=("b",))]
    sc.write_record(tmp_path, cards, cards[0], ["a line was removed"], widest=True)
    first = sc.read_record(tmp_path)
    assert first["chosen_by"] == "person" and first["chosen"] == cards[0]
    assert first["widest_offered"] is True and first["history"] == []
    assert first["removed"] == ["a line was removed"]
    sc.write_record(tmp_path, cards, cards[1], [], widest=False)
    second = sc.read_record(tmp_path)
    assert second["chosen"] == cards[1] and second["history"] == [cards[0]]
    sc.write_record(tmp_path, cards, None, [], widest=False)
    third = sc.read_record(tmp_path)
    assert third["chosen_by"] == "nobody" and third["chosen"] is None
    assert third["history"] == [cards[0], cards[1]]
    assert (tmp_path / sc.RECORD).is_file()


def test_read_record_is_none_without_a_file(tmp_path: Path):
    assert sc.read_record(tmp_path) is None


def test_gaps_names_a_main_result_the_plan_has_no_metric_for():
    card = _card(results=("cool rate", "final temperature"), settings=("lid_on",))
    design = {"metrics": [{"id": "cool_rate"}], "grid": {"lid on": [0, 1]}}
    found = sc.gaps(card, design)
    assert len(found) == 1 and "final temperature" in found[0]
    assert sc.gaps(card, {"metrics": [{"label": "Cool rate"}, {"id": "final_temperature"}],
                          "protocol": {"grid": {"lid_on": [0, 1]}}}) == []
    missing_setting = sc.gaps(_card(results=("cool rate",), settings=("cup shape",)),
                              {"metrics": [{"id": "cool_rate"}],
                               "optimisation": {"design_variables": [{"name": "wall_width"}]}})
    assert len(missing_setting) == 1 and "cup shape" in missing_setting[0]
    assert sc.gaps(_card(results=(), settings=("wall width",)),
                   {"optimisation": {"design_variables": [{"name": "wall_width"}]}}) == []


def test_question_text_numbers_each_card_and_says_pick_the_study_to_run():
    cards = [_card(), _card(question="Which cup shape keeps tea warm longest?", study_type="find_best_design",
                            results=("final temperature",))]
    payload = sc.question(cards)
    entry = payload["study"]
    assert entry["default"] == "" and entry["cards"] == cards
    text = entry["question"]
    assert text.startswith("Pick the study to run (a number), or write the study you want in a sentence:")
    assert "1." in text and "2." in text and "Which cup shape keeps tea warm longest?" in text


def test_plan_note_and_nobody_line():
    note = sc.plan_note(_card())
    assert "How fast does a cup of tea cool?" in note and "chose" in note
    assert sc.nobody_line().startswith("Nobody picked the study to run;")
def test_unquoted_off_turns_the_study_cards_off():
    assert Config.model_validate({"topic": "t", "pauses": {"scope": False}}).pauses.scope == "off"
    assert Config.model_validate({"topic": "t", "pauses": {"scope": "off"}}).pauses.scope == "off"


def test_study_cards_are_on_by_default():
    assert Config.model_validate({"topic": "t"}).pauses.scope == "ask"


def test_the_study_card_prompt_has_its_placeholders():
    text = (Path(__file__).resolve().parent.parent / "agents" / "scope.md").read_text(encoding="utf-8")
    names = {"topic", "answers", "direction", "sources", "again"}
    assert names <= set(string.Template(text).get_identifiers())
    out = string.Template(text).substitute({n: f"<<{n}>>" for n in names})
    assert "$" not in out.replace("$$", "")


def test_a_literature_field_that_is_not_a_list_is_dropped_not_a_crash():
    reply = {"cards": [{"question": "q", "study_type": "measure", "literature": 1}]}
    cards, removed = sc.normalize(reply, {"1"})
    assert cards[0]["literature"] == [] and removed == []
    edited = sc.read_answer({"study": {"question": "my own study", "literature": 1}}, [])
    assert edited is not None and edited["literature"] == []


def test_e_and_a_number_whose_edit_never_came_picks_that_card():
    cards = [_card(), _card(question="Which cup keeps tea warm longest?")]
    assert sc.read_answer("e 2", cards) == cards[1]
    assert sc.read_answer("E1", cards) == cards[0]
    assert sc.read_answer("e 9", cards) is None


def test_plan_section_names_the_pick_or_says_nobody_picked():
    cards = [_card(), _card(question="Which cup keeps tea warm longest?", study_type="find_best_design")]
    picked = sc.plan_section({"cards": cards, "chosen": cards[1], "chosen_by": "person"})
    assert picked.startswith("## The study to run") and "Which cup keeps tea warm longest?" in picked
    nobody = sc.plan_section({"cards": cards, "chosen": None, "chosen_by": "nobody"})
    assert sc.nobody_line() in nobody and "- How fast does a cup of tea cool?" in nobody
    assert sc.plan_section(None) == "" and sc.plan_section({"cards": []}) == ""


def test_gaps_read_the_readers_names_the_protocol_gives():
    design = {"protocol": {"grid": {"m": ["a", "b"]}, "metrics": [{"id": "t_end"}],
                           "labels": {"m": "Cup material $m$ (-)", "t_end": "Final temperature $T$ (K)"}}}
    assert sc.gaps(_card(results=("final temperature",), settings=("cup material",)), design) == []
    # a reader's name for something the plan does not vary is not a setting it varies
    found = sc.gaps(_card(results=(), settings=("lid colour",)),
                    {"protocol": {"grid": {"m": [1]}, "labels": {"lid": "Lid colour"}}})
    assert len(found) == 1 and "lid colour" in found[0]
