"""The model behind the numbers, and where each oracle's expected value comes from.

A real quest's plan expected an RK4 error of 1.637e-08 where the true one is 3.33241e-07, and nothing said where that
number came from. The plan now states the model that produces the numbers (its equations, each with its role and its
source), and the engine reads each oracle's ``kind`` and ``reference``: an expected value must come from a derivation
written out, a source this quest retrieved, an equation of the model, or a second implementation that says what code it
does not share.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from core import evidence, oracle_check, plan
from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, PausesConfig, ProviderConfig,
)
from core.engine import Engine

SOURCES = [
    {"label": "1", "title": "Numerical Methods for Ordinary Differential Equations", "doi": "10.1002/9781119121534", "url": ""},
    {"label": "2", "title": "Solving Ordinary Differential Equations I: Nonstiff Problems", "doi": "", "url": ""},
]

MODEL = {
    "summary": "classical RK4 on the linear ODE y' = -y",
    "assumptions": ["f is smooth", "fixed step h"],
    "holds_for": "h small enough that h*|lambda| < 2.78 (the stability limit)",
    "equations": [
        {"id": "E1", "formula": "y_{n+1} = y_n + h/6 (k1 + 2 k2 + 2 k3 + k4)", "role": "generates", "source": "[1]"},
        {"id": "E2", "formula": "p = log2(e(h) / e(h/2))", "role": "analyses",
         "source": "derivation", "derivation": "e(h) ~ C h^p, so e(h)/e(h/2) = 2^p and p = log2 of the ratio"},
    ],
}

GOOD_ORACLE = {"name": "rk4 error", "kind": "closed_form", "check": "error at t=1 of y'=-y, h=0.1",
               "expected": 3.33241e-07, "tolerance": 1e-8,
               "reference": "derivation: the exact solution is exp(-1); one RK4 step multiplies by the 4th-order Taylor "
                            "polynomial of exp(-h), so after 10 steps y = T(0.1)^10 and the error is |T(0.1)^10 - exp(-1)|"}


def _protocol(**over: Any) -> dict[str, Any]:
    return {"grid": {"h": [0.1, 0.05]}, "oracles": [dict(GOOD_ORACLE)], "model": json.loads(json.dumps(MODEL)), **over}


# --- the model section: normalised, validated, rendered ------------------------------------------------------------


def test_the_model_is_kept_in_the_protocol_and_placed_last() -> None:
    raw = {"model": MODEL, "grid": {"h": [0.1, 0.05]}, "oracles": [GOOD_ORACLE]}
    fixed, why = plan.normalize_protocol(raw)
    assert why is None
    assert list(fixed)[-1] == "model", "the model goes last, so a cut of the protocol keeps the grid and the checks"
    assert fixed["model"]["equations"][0]["id"] == "E1"
    assert fixed["model"]["equations"][1]["role"] == "analyses"
    assert fixed["model"]["assumptions"] == ["f is smooth", "fixed step h"]


@pytest.mark.parametrize("bad, needle", [
    ("just a sentence", "protocol.model"),
    ({"summary": "x", "equations": "E1: y = x"}, "protocol.model"),
    ({"summary": "x", "equations": [{"id": "E1"}]}, "formula"),
    ({"summary": ["x"]}, "protocol.model"),
])
def test_a_model_block_that_cannot_be_read_is_refused_with_the_reason(bad: Any, needle: str) -> None:
    fixed, why = plan.normalize_protocol({"model": bad})
    assert fixed is None and needle in (why or ""), why


def test_a_role_written_in_other_words_is_read_as_one_of_the_two() -> None:
    model = {"summary": "x", "equations": [{"id": "E1", "formula": "a = b", "role": "Generate", "source": "[1]"},
                                          {"id": "E2", "formula": "c = d", "role": "analysis", "source": "[1]"}]}
    fixed, why = plan.normalize_protocol({"model": model})
    assert why is None
    assert [e["role"] for e in fixed["model"]["equations"]] == ["generates", "analyses"]


def test_a_draft_loses_only_the_equation_that_cannot_be_read_and_says_so() -> None:
    model = {"summary": "x", "equations": [{"id": "E1", "formula": "a = b", "role": "generates", "source": "[1]"},
                                          {"id": "E2", "role": "analyses"}]}
    fixed, notes = plan.repair_protocol({"grid": {"h": [0.1]}, "model": model})
    assert fixed is not None
    assert [e["id"] for e in fixed["model"]["equations"]] == ["E1"]
    assert any("E2" in n and "left out" in n for n in notes), notes


def test_an_equation_without_an_id_gets_the_next_free_one_in_a_draft_and_the_plan_says_so() -> None:
    model = {"summary": "x", "equations": [{"formula": "a = b", "role": "generates", "source": "[1]"}]}
    fixed, notes = plan.repair_protocol({"model": model})
    assert fixed["model"]["equations"][0]["id"] == "E1"
    assert any("E1" in n for n in notes), notes


def test_the_plan_shows_the_model_as_a_readable_section_and_the_block_round_trips() -> None:
    design = {"hypothesis": "RK4 converges at order 4", "protocol": plan.normalize_protocol(_protocol())[0]}
    text = plan.render("RK4 convergence", {}, design)
    assert "## The model behind the numbers" in text
    section = text.split("## The model behind the numbers", 1)[1].split("## ", 1)[0]
    assert "classical RK4 on the linear ODE" in section
    assert "E1" in section and "produces the data" in section and "[1]" in section
    assert "E2" in section and "analyses the results" in section and "derivation" in section
    assert "rk4 error" in section and "a special or limiting case" in section
    assert "```" not in section, "a fence above the design block would be read as the design"
    assert text.index("## The model behind the numbers") < text.index(f"## {plan.DESIGN_HEADING}")
    assert plan.parse(text).design == plan.normalize_design(design)[0]


def test_a_plan_without_a_model_says_so_where_a_person_reads_it() -> None:
    design = {"hypothesis": "h", "protocol": {"grid": {"h": [0.1]}, "oracles": [GOOD_ORACLE]}}
    text = plan.render("t", {}, design)
    assert "## The model behind the numbers" in text
    assert "does not say what model produces the numbers" in text


# --- the kinds of oracle ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("written, kind", [
    ("closed_form", "special_case"), ("limiting_case", "special_case"), ("exact_small_case", "special_case"),
    ("invariant", "invariant"), ("independent_implementation", "second_implementation"),
    ("special case", "special_case"), ("Conservation", "invariant"), ("scaling", "symmetry"),
    ("convergence-rate", "convergence_rate"), ("published benchmark", "published_value"),
    ("second_implementation", "second_implementation"), ("vibes", None), ("", None),
])
def test_the_older_names_of_a_kind_are_read_as_the_six(written: str, kind: str | None) -> None:
    assert oracle_check.kind_of({"name": "x", "kind": written}) == kind
    assert set(oracle_check.KINDS) == {"special_case", "invariant", "symmetry", "second_implementation",
                                       "convergence_rate", "published_value"}


def test_a_kind_that_is_none_of_the_six_is_noted_not_refused() -> None:
    fixed, why = plan.normalize_protocol({"oracles": [{**GOOD_ORACLE, "kind": "vibes"}]})
    assert why is None and fixed["oracles"][0]["kind"] == "vibes", "the plan keeps what was written"
    notes = oracle_check.model_notes({"oracles": [{**GOOD_ORACLE, "kind": "vibes"}], "model": MODEL}, SOURCES)
    assert any("vibes" in n for n in notes), notes


# --- where an expected value comes from -----------------------------------------------------------------------------


def _gaps(reference: str, *, kind: str = "closed_form", model: dict[str, Any] | None = MODEL,
          sources: list[dict[str, str]] = SOURCES) -> list[str]:
    protocol = {"oracles": [{**GOOD_ORACLE, "kind": kind, "reference": reference}], **({"model": model} if model else {})}
    return oracle_check.source_gaps(protocol, sources)


def test_a_derivation_with_its_steps_written_is_a_source() -> None:
    assert _gaps(GOOD_ORACLE["reference"]) == []


def test_a_derivation_named_without_its_steps_is_not() -> None:
    (gap,) = _gaps("derivation")
    assert "rk4 error" in gap and "steps" in gap


def test_an_empty_reference_is_a_gap() -> None:
    (gap,) = _gaps("")
    assert "does not say where its expected value comes from" in gap


def test_a_retrieved_source_by_its_number_title_or_doi_is_a_source() -> None:
    assert _gaps("[2], table 3") == []
    assert _gaps("Hairer et al., Solving Ordinary Differential Equations I: Nonstiff Problems, p. 140") == []
    assert _gaps("the error constant in doi 10.1002/9781119121534") == []


def test_a_source_the_quest_did_not_retrieve_is_a_gap() -> None:
    (gap,) = _gaps("[7]")
    assert "[7]" in gap and "did not retrieve" in gap
    (gap,) = _gaps("Butcher 2008, from memory")
    assert "did not retrieve" in gap
    (gap,) = _gaps("10.1000/not-retrieved")
    assert "did not retrieve" in gap


def test_nothing_retrieved_means_only_a_derivation_counts() -> None:
    assert _gaps("[1]", sources=[]) != []
    assert _gaps(GOOD_ORACLE["reference"], sources=[]) == []


def test_an_equation_of_the_model_counts_when_its_own_source_does() -> None:
    assert _gaps("E1 applied ten times at h = 0.1") == []
    assert _gaps("E2") == []  # a derivation with its steps
    unsourced = {**MODEL, "equations": [{**MODEL["equations"][0], "source": "[9]"}]}
    (gap,) = _gaps("E1", model=unsourced)
    assert "E1" in gap and "[9]" in gap
    (gap,) = _gaps("E5")
    assert "E5" in gap and "not" in gap


def test_a_second_implementation_must_say_what_code_it_does_not_share() -> None:
    assert _gaps("a second solver, scipy's solve_ivp (RK45), shares no code with the simulation",
                 kind="independent_implementation") == []
    (gap,) = _gaps("a second implementation of the same stepper", kind="second_implementation")
    assert "does not share" in gap


def test_the_model_notes_name_an_equation_without_a_source_or_role() -> None:
    model = {"summary": "x", "equations": [{"id": "E1", "formula": "a = b", "role": "", "source": ""},
                                          {"id": "E2", "formula": "c = d", "role": "generates", "source": "[8]"}]}
    notes = oracle_check.model_notes({"oracles": [GOOD_ORACLE], "model": model}, SOURCES)
    joined = " ".join(notes)
    assert "E1" in joined and "role" in joined and "source" in joined
    assert "E2" in joined and "[8]" in joined
    assert oracle_check.model_notes({"oracles": [GOOD_ORACLE]}, SOURCES), "no model at all is noted"
    assert oracle_check.model_notes(_protocol(), SOURCES) == []


# --- the evidence ladder -----------------------------------------------------------------------------------------


def test_an_unsourced_expected_value_is_a_gap_at_the_independently_validated_level(tmp_path: Path) -> None:
    state = {"design": {"protocol": {"oracles": [GOOD_ORACLE]}}}
    record = evidence.assess(tmp_path, state, oracle_source_gaps=["the check 'x' cites a source this quest did not retrieve"])
    assert any("did not retrieve" in g for g in record["all_gaps"].get("independently_validated", [])), record["all_gaps"]
    clean = evidence.assess(tmp_path, state, oracle_source_gaps=[])
    assert not any("did not retrieve" in g for g in clean["all_gaps"].get("independently_validated", []))


# --- the engine ------------------------------------------------------------------------------------------------------


def _cfg(tmp_path: Path, **over: Any) -> Config:
    return Config(
        topic="RK4 convergence on y' = -y", title="rk4", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, clarify_mode="off"),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60), knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "out"), pauses=PausesConfig(plan="off"), **over,
    )


class Paused(Exception):
    pass


def _engine(tmp_path: Path, protocol: dict[str, Any], *, research: bool = False) -> tuple[Engine, list[dict[str, Any]]]:
    eng = Engine(_cfg(tmp_path))
    if research:
        eng.config = eng.config.model_copy(update={"rigor_profile": "research"})
    design = {"hypothesis": "RK4 converges at order 4", "method": "sweep h", "protocol": protocol,
              "plan": {"in_short": "x", "literature": [{"source": "[1]", "says": "RK4 is order 4"}]}}
    audit = json.dumps({"objections_addressed": []})
    eng._client = type("Stub", (), {"chat": AsyncMock(side_effect=[json.dumps(design), audit])})()
    seen: list[dict[str, Any]] = []

    def fake(**kwargs: Any) -> None:
        seen.append(kwargs)
        raise Paused(kwargs["kind"])

    eng._pause_for_human = fake  # type: ignore[method-assign]
    return eng, seen


def _literature() -> list[dict[str, Any]]:
    return [{"content": "RK4 ...", "metadata": {"title": s["title"], "doi": s["doi"], "source": "openalex"}} for s in SOURCES]


@pytest.mark.asyncio
async def test_a_plan_with_a_model_round_trips_through_plan_md_and_the_freeze(tmp_path: Path) -> None:
    from core import frozen_protocol

    eng, seen = _engine(tmp_path, _protocol())
    state = {"topic": "RK4", "iteration": 0, "literature": _literature()}
    await eng._node_plan(state)
    assert seen == []
    text = plan.plan_path(eng.quest_root).read_text(encoding="utf-8")
    assert "## The model behind the numbers" in text and "classical RK4" in text
    eng._freeze_protocol_if_due(state)
    frozen = frozen_protocol.protocol_of(eng.quest_root)
    assert frozen["model"]["equations"][0]["formula"].startswith("y_{n+1}")
    assert frozen["model"]["summary"] == MODEL["summary"]


@pytest.mark.asyncio
async def test_by_default_an_unsourced_expected_value_is_warned_about_and_the_quest_goes_on(tmp_path: Path) -> None:
    eng, seen = _engine(tmp_path, _protocol(oracles=[{**GOOD_ORACLE, "reference": "Butcher 2008"}]))
    await eng._node_plan({"topic": "RK4", "iteration": 0, "literature": _literature()})
    assert seen == []
    text = plan.plan_path(eng.quest_root).read_text(encoding="utf-8")
    checks = text.split("## Checks already made", 1)[1]
    assert "did not retrieve" in checks
    log = (eng.fi_dir / "run.log").read_text(encoding="utf-8")
    assert "did not retrieve" in log


@pytest.mark.asyncio
async def test_under_research_an_unsourced_expected_value_stops_the_quest_at_the_plan_until_it_is_fixed(
        tmp_path: Path) -> None:
    eng, seen = _engine(tmp_path, _protocol(oracles=[{**GOOD_ORACLE, "reference": ""}]), research=True)
    state = {"topic": "RK4", "iteration": 0, "literature": _literature()}
    with pytest.raises(Paused):
        await eng._node_plan(state)
    assert seen[-1]["kind"] == "plan"
    assert "rk4 error" in " ".join(seen[-1]["steps"])
    # A resume that fixed nothing stops again (the plan-pause marker does not let it through).
    with pytest.raises(Paused):
        await eng._node_plan(state)
    # Fixed in plan.md: the resume goes on without another stop.
    path = plan.plan_path(eng.quest_root)
    fixed = path.read_text(encoding="utf-8").replace("reference: ''", "reference: '[1], the error constant of RK4'")
    assert fixed != path.read_text(encoding="utf-8")
    path.write_text(fixed, encoding="utf-8")
    seen.clear()
    await eng._node_plan(state)
    assert seen == []


@pytest.mark.asyncio
async def test_under_research_an_oracle_the_engine_added_without_a_source_stops_before_the_freeze(
        tmp_path: Path) -> None:
    from core import frozen_protocol

    eng, seen = _engine(tmp_path, _protocol(), research=True)
    state = {"topic": "RK4", "iteration": 0, "literature": _literature()}
    await eng._node_plan(state)
    path = plan.plan_path(eng.quest_root)
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace(GOOD_ORACLE["reference"][:40], "from memory, roughly "), encoding="utf-8")
    with pytest.raises(Paused):
        eng._check_plan_sources(state, stop=True)
    assert frozen_protocol.load(eng.quest_root) is None
