"""The plan's model is asked which result answers each main quantity of the topic, and at which value of a setting where the topic
asks for one (the ``answers`` part of core/contract_parts.py), and the paper's answers block is built from that map: one line
for each value of a setting the study compares, never one range across them; a quantity pinned to a value is answered at it; a
quantity nothing answers stays "not answered". The fake model is a function. Neutral made-up study: a pond's algae."""

from __future__ import annotations

import asyncio
import json
import string
from pathlib import Path
from typing import Any

from core import contract_parts as cp
from core import paper_facts as pf
from core import paper_values as pv
from core import topic_requirements as tr

TEMPLATES = {"plan_answers": string.Template((Path(__file__).parent.parent / "agents" / "plan_answers.md").read_text(encoding="utf-8"))}
PROTOCOL: dict[str, Any] = {
    "grid": {"nutrient": [1, 2], "temp": [10, 20]},
    "runs_per_setting": 5,
    "metrics": [{"id": "algae_peak", "kind": "mean", "estimand": "peak algae count", "unit": "one pond run"},
                {"id": "bloom", "kind": "proportion", "estimand": "P(bloom)", "unit": "one pond run"}],
    "comparisons": [{"id": "peak_by_temp", "result": "algae_peak", "across": "temp", "values": [10, 20], "how": "difference"}],
}
DESIGN = {"hypothesis": "h", "method": "Simulate algae in a pond.", "protocol": PROTOCOL}
REQ = {"quantities": [
    {"name": "peak algae", "stated": "report the peak", "main": True},
    {"name": "chance of a bloom (at nutrient 1)", "stated": "the bloom chance at the lowest nutrient", "main": True},
    {"name": "water clarity", "stated": "report the clarity", "main": True},
    {"name": "a side quantity", "stated": "x", "main": False},
]}


def _ask(design: dict[str, Any], replies: list[Any], asked: dict[str, dict[str, str]]):
    seen: list[str] = []

    async def ask(prompt: str) -> str:
        seen.append(prompt)
        reply = replies.pop(0)
        return reply if isinstance(reply, str) else json.dumps(reply)

    def parse(text: str) -> Any:
        try:
            return json.loads(text)
        except ValueError:
            return None

    out = asyncio.run(cp.ask_parts(design, ask, parse, TEMPLATES, only=("answers",), topic_quantities=asked))
    return out, seen


def test_only_main_quantities_without_a_placed_map_are_asked() -> None:
    asked = tr.quantities_to_place(REQ, PROTOCOL)
    assert list(asked) == ["Q1", "Q2", "Q3"] and asked["Q1"]["by"] == ""
    placed = {**PROTOCOL, "topic_map": [{"item": "Q1", "by": "metric:algae_peak", "at": {}}]}
    got = tr.quantities_to_place(REQ, placed)
    assert list(got) == ["Q2", "Q3"]
    unplaced = {**PROTOCOL, "topic_map": [{"item": "Q1", "by": "metric:algae_peak"}]}
    assert tr.quantities_to_place(REQ, unplaced)["Q1"]["by"] == "metric:algae_peak"  # the plan's own map is kept, `at` is asked


def test_a_valid_answer_is_written_into_the_map_and_an_unmappable_quantity_adds_nothing() -> None:
    asked = tr.quantities_to_place(REQ, PROTOCOL)
    reply = {"answers": {"Q1": {"by": "metric:algae_peak"}, "Q2": {"by": "metric:bloom", "at": {"nutrient": 1}},
                         "Q3": {"by": "none"}}}
    out, seen = _ask(DESIGN, [reply], asked)
    assert out.parts["answers"]["valid"] and len(seen) == 1
    rows = out.design["protocol"]["topic_map"]
    assert {"item": "Q2", "by": "metric:bloom", "at": {"nutrient": 1}} in rows
    assert {"item": "Q1", "by": "metric:algae_peak", "at": {}} in rows
    assert not [r for r in rows if r["item"] == "Q3"]  # nothing answers it: it stays unmapped


def test_an_answer_naming_what_the_study_does_not_have_is_asked_again_with_the_reason() -> None:
    asked = tr.quantities_to_place(REQ, PROTOCOL)
    bad = {"answers": {"Q1": {"by": "metric:salinity"}, "Q2": {"by": "metric:bloom", "at": {"nutrient": 7}},
                       "Q3": {"by": "none", "at": {"depth": 1}}}}
    good = {"answers": {"Q1": {"by": "metric:algae_peak"}, "Q2": {"by": "metric:bloom"}, "Q3": {"by": "none"}}}
    out, seen = _ask(DESIGN, [bad, good], asked)
    assert out.parts["answers"]["valid"] and out.parts["answers"]["asks"] == 2
    assert "metric:salinity" in seen[1] or "`by` must be" in seen[1]
    assert "no value 7" in seen[1] and "not a setting of the study" in seen[1]
    # a plan that never gives a usable answer is left as it was
    out, _seen = _ask(DESIGN, [bad] * cp.MAX_ASKS, asked)
    assert not out.parts["answers"]["valid"] and "topic_map" not in out.design["protocol"]


def _quest(tmp_path: Path, monkeypatch, topic_map: list[dict[str, Any]]) -> Path:
    (tmp_path / "raw").mkdir(parents=True)
    cells = []
    for n in (1, 2):
        for t in (10, 20):
            vals = [n * 10.0 + t + k for k in range(5)]
            cells.append({"cell": {"nutrient": n, "temp": t}, "key": f"nutrient={n},temp={t}", "planned": 5, "ok": 5, "failed": 0,
                          "metrics": {"algae_peak": {"values": vals, "trials": list(range(5)), "count": 5, "total": sum(vals),
                                                     "non_finite": 0},
                                      "bloom": {"values": [0.0, 1.0, 0.0, 1.0, 1.0], "trials": list(range(5)), "count": 5,
                                                "total": 3.0, "non_finite": 0}}})
    (tmp_path / "raw" / "trials.json").write_text(json.dumps({"cells": cells}), encoding="utf-8")
    (tmp_path / ".fi").mkdir()
    (tmp_path / ".fi" / "topic_requirements.json").write_text(json.dumps(REQ), encoding="utf-8")
    protocol = {**PROTOCOL, "topic_map": topic_map}
    monkeypatch.setattr(pf, "plan_protocol", lambda *a, **k: protocol)
    return tmp_path


def test_a_compared_setting_is_answered_for_each_value_and_an_at_pins_the_quantity(tmp_path, monkeypatch) -> None:
    root = _quest(tmp_path, monkeypatch, [{"item": "Q1", "by": "metric:algae_peak", "at": {}},
                                          {"item": "Q2", "by": "metric:bloom", "at": {"nutrient": 1}}])
    text = pf.answers_text(root)
    assert "- peak algae (temp = 10) (the plan's map: `algae_peak`): {{v:algae_peak @ nutrient=each, temp=10 | range}}" in text
    assert "- peak algae (temp = 20)" in text
    assert "{{v:algae_peak @ nutrient=each, temp=each | range}}" not in text  # no one range across the compared values
    assert "(nutrient = 1)" in text and "{{v:bloom:mean_ci @ nutrient=1, temp=10}}" in text
    assert "- water clarity: not answered (the plan maps it to no result of the study)" in text  # the known positive
    out = pv.fill_values(text, pv.records_of(root, pf.plan_protocol(root)))[0]
    assert "not recorded" not in out


def test_a_map_without_at_or_comparisons_is_read_as_before(tmp_path, monkeypatch) -> None:
    root = _quest(tmp_path, monkeypatch, [{"item": "Q1", "by": "metric:algae_peak"}])
    monkeypatch.setattr(pf, "plan_protocol", lambda *a, **k: {**PROTOCOL, "comparisons": [],
                                                              "topic_map": [{"item": "Q1", "by": "metric:algae_peak"}]})
    assert "- peak algae (the plan's map: `algae_peak`): {{v:algae_peak @ nutrient=each, temp=each | range}}" in pf.answers_text(root)


def test_a_study_run_through_the_deterministic_entry_says_no_seed_was_passed_otherwise_the_recorded_base_seed(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(pf._run_facts, "seed_verdict", lambda root: {"verdict": pf._run_facts.UNKNOWN})
    monkeypatch.setattr(pf._disclosure, "recorded_seeds", lambda root: (0, frozenset()))
    monkeypatch.setattr(pf, "plan_protocol", lambda *a, **k: {"seed_policy": "A constant base_seed of 99 is used."})
    (tmp_path / "code").mkdir()
    study = tmp_path / "code" / "study.json"
    study.write_text(json.dumps({"entry": "run_trial", "base_seed": 0}), encoding="utf-8")
    assert pf._seed_text(tmp_path) == ("a seed passed to each trial, derived from base seed 0 "
                                       "(FI could not tell from the code whether the simulation uses it)")  # the plan's prose is not quoted
    study.write_text(json.dumps({"entry": "run_cell", "base_seed": 0}), encoding="utf-8")
    assert pf._seed_text(tmp_path) == "none (each setting ran once with no random numbers; FI passed no seed)"
