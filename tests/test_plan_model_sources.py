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
    (42, "protocol.model"),
    ({"summary": "x", "equations": 7}, "protocol.model"),
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
    assert "rk4 error" in gap and "at least one equation (with `=`" in gap


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


def _engine(tmp_path: Path, protocol: dict[str, Any], *, research: bool = False,
            ask: bool = False) -> tuple[Engine, list[dict[str, Any]]]:
    eng = Engine(_cfg(tmp_path))
    if research:
        # As the profile has it: the plan is held for the person (pauses.plan: ask).
        eng.config = eng.config.model_copy(update={
            "rigor_profile": "research", "pauses": eng.config.pauses.model_copy(update={"plan": "ask" if ask else "off"})})
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
    return [{"content": "RK4 ...", "metadata": {"title": s["title"], "doi": s["doi"], "source": "openalex",
                                              "url": f"https://example.org/{s['label']}"}} for s in SOURCES]


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
async def test_under_research_an_unsourced_expected_value_is_marked_not_confirmed_and_never_stops(
        tmp_path: Path) -> None:
    from core import accepted_checks

    eng, seen = _engine(tmp_path, _protocol(oracles=[{**GOOD_ORACLE, "reference": ""}]), research=True)
    state = {"topic": "RK4", "iteration": 0, "literature": _literature()}
    await eng._node_plan(state)
    assert seen == [], "where a value comes from is not a question a person is asked"
    assert eng._not_confirmed_names(state, eng._draft_protocol(state)) == ["rk4 error"]
    assert accepted_checks.accepted(eng.quest_root)["chosen"]["rk4 error"]["by"] == accepted_checks.AUTOMATIC
    # Given a source later, the check is no longer marked.
    path = plan.plan_path(eng.quest_root)
    fixed = path.read_text(encoding="utf-8").replace("reference: ''", "reference: '[1], the error constant of RK4'")
    assert fixed != path.read_text(encoding="utf-8")
    path.write_text(fixed, encoding="utf-8")
    await eng._node_plan(state)
    assert seen == [] and eng._not_confirmed_names(state, eng._draft_protocol(state)) == []


@pytest.mark.asyncio
async def test_under_research_an_oracle_the_engine_added_without_a_source_is_frozen_marked_not_confirmed(
        tmp_path: Path) -> None:
    from core import frozen_protocol

    eng, seen = _engine(tmp_path, _protocol(), research=True)
    state = {"topic": "RK4", "iteration": 0, "literature": _literature()}
    await eng._node_plan(state)
    path = plan.plan_path(eng.quest_root)
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace(GOOD_ORACLE["reference"][:40], "from memory, roughly "), encoding="utf-8")
    eng._check_plan_sources(state, stop=True)
    assert seen == []
    eng._freeze_protocol_if_due(state)
    approved = frozen_protocol.load(eng.quest_root)["approved_by"]
    assert "source not confirmed" in approved and "'rk4 error'" in approved, "the freeze says the value is unconfirmed"


# --- the edge cases a review found -------------------------------------------------------------------------------


def test_a_grouped_or_ranged_citation_of_retrieved_sources_is_a_source() -> None:
    three = SOURCES + [{"label": "3", "title": "x", "doi": "", "url": ""}]
    assert _gaps("[1, 2]", sources=three) == []
    assert _gaps("[1–3]", sources=three) == []
    (gap,) = _gaps("[1, 7]", sources=three)
    named = gap.split(" which ")[0]  # what is named as missing; the sentence then lists the real ones, [1]–[3]
    assert "[7]" in named and "[1]" not in named
    assert "(they are [1]–[3])" in gap


def test_an_array_index_is_not_a_citation() -> None:
    assert _gaps("the value y[10] of E1 after 10 steps") == []


def test_a_doi_with_parentheses_or_a_prefix_is_matched_as_written() -> None:
    sources = [{"label": "1", "title": "t", "doi": "https://doi.org/10.1016/0021-9991(76)90041-3", "url": ""}]
    from core.oracle_check import retrieved_sources
    assert _gaps("doi 10.1016/0021-9991(76)90041-3, eq. 12", sources=sources) == []
    assert _gaps("see https://doi.org/10.1016/0021-9991(76)90041-3.", sources=sources) == []
    stored = retrieved_sources([("1", {"title": "t", "doi": "doi:10.1016/0021-9991(76)90041-3"})])
    assert stored[0]["doi"] == "10.1016/0021-9991(76)90041-3"


def test_a_derivation_that_only_points_to_a_recalled_source_is_not_a_derivation() -> None:
    (gap,) = _gaps("derived from Butcher (2008), Numerical Methods, table 5.2")
    assert "Butcher" in gap or "equation" in gap
    (gap,) = _gaps("Derivation: see Butcher 2008 p. 99 for the constant")
    assert "equation" in gap or "Butcher" in gap
    (gap,) = _gaps("derivation: the constant is in the textbook we all know well")
    assert "equation" in gap
    eq = {"id": "E1", "formula": "a = b", "role": "generates", "source": "derived", "derivation": "from Hairer's book, recalled"}
    assert oracle_check.equation_problem(eq, SOURCES)


def test_an_equation_is_cited_by_its_id_in_any_case_and_a_physics_symbol_is_not_an_equation() -> None:
    assert _gaps("e1 at h = 0.1") == []
    assert _gaps("[2], the ground-state energy E0") == []
    fixed, why = plan.normalize_protocol({"model": {"summary": "x", "equations": [
        {"id": 1, "formula": "a = b", "role": "generates", "source": "[1]"}]}})
    assert why is None and fixed["model"]["equations"][0]["id"] == "E1"


def test_a_short_generic_title_is_not_a_citation() -> None:
    sources = [{"label": "1", "title": "Monte Carlo Methods", "doi": "", "url": ""}]
    assert _gaps("a standard result for monte carlo methods, from memory", sources=sources) != []


def test_a_second_implementation_that_reuses_the_simulations_code_is_not_independent() -> None:
    assert _gaps("second implementation using a different method: the same RK4 function",
                 kind="second_implementation") != []
    assert _gaps("a second implementation that doesn’t use the simulation’s code: scipy's solve_ivp",
                 kind="second_implementation") == []


def test_a_model_written_as_one_sentence_before_the_block_had_parts_still_parses() -> None:
    fixed, why = plan.normalize_protocol({"grid": {"h": [0.1]}, "model": "SIR ODE with frequency-dependent transmission"})
    assert why is None and fixed["model"] == "SIR ODE with frequency-dependent transmission"
    notes = oracle_check.model_notes(fixed, SOURCES)
    assert any("one sentence" in n for n in notes), notes
    text = plan.render("t", {}, {"hypothesis": "h", "protocol": fixed})
    assert "SIR ODE with frequency-dependent transmission" in text.split("## Success criteria")[0]


def test_a_model_without_equations_hashes_as_it_was_written() -> None:
    from core import frozen_protocol

    raw = {"grid": {"h": [0.1]}, "model": {"summary": "x"}}
    fixed, _ = plan.normalize_protocol(raw)
    assert frozen_protocol.sha256(fixed) == frozen_protocol.sha256(plan.normalize_protocol(fixed)[0])
    assert "equations" not in fixed["model"]


def test_the_plan_lists_the_sources_the_quest_found_by_number() -> None:
    text = plan.render("t", {}, {"hypothesis": "h", "protocol": plan.normalize_protocol(_protocol())[0]}, sources=SOURCES)
    assert "## The sources this quest found" in text
    assert "- [1] Numerical Methods for Ordinary Differential Equations (DOI 10.1002/9781119121534)" in text
    assert plan.parse(text).error is None


def test_the_malformed_parts_of_a_model_or_a_source_list_never_crash_the_check() -> None:
    assert oracle_check.source_gaps({"oracles": [{**GOOD_ORACLE, "reference": "E1"}], "model": {"equations": 5}},
                                    [1, None, "x"]) != []
    assert oracle_check.model_notes({"model": {"equations": 5}}, [])


# --- the engine, again ---------------------------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("path_key", ["no_simulation_resolved", "survey_mode_resolved"])
async def test_a_quest_that_runs_no_experiment_is_not_checked(tmp_path: Path, path_key: str) -> None:
    eng, seen = _engine(tmp_path, _protocol(oracles=[{**GOOD_ORACLE, "reference": ""}]), research=True)
    await eng._node_plan({"topic": "RK4", "iteration": 0, "literature": _literature(), path_key: True})
    assert seen == []
    assert "did not" not in plan.plan_path(eng.quest_root).read_text(encoding="utf-8").split("## Checks already made")[1]


@pytest.mark.asyncio
async def test_under_research_with_the_plan_held_the_plain_plan_stop_is_the_only_one(tmp_path: Path) -> None:
    from core import frozen_protocol

    eng, seen = _engine(tmp_path, _protocol(oracles=[{**GOOD_ORACLE, "reference": ""}]), research=True, ask=True)
    state = {"topic": "RK4", "iteration": 0, "literature": _literature()}
    with pytest.raises(Paused):
        await eng._node_plan(state)
    assert seen[-1]["headline"] == "read and edit the plan", "the stop the person asked for, nothing about sources"
    path = plan.plan_path(eng.quest_root)
    path.write_text(path.read_text(encoding="utf-8").replace("reference: ''", "reference: '[1], the error of RK4'"),
                    encoding="utf-8")
    seen.clear()
    await eng._node_plan(state)
    assert seen == [], "the person saw the plan at the first stop: it does not stop again"
    eng._freeze_protocol_if_due(state)
    record = frozen_protocol.load(eng.quest_root)
    assert record["approved_by"].startswith("human")
    assert [s["label"] for s in record["sources"]] == ["1", "2"]


def test_the_freeze_keeps_the_sources_outside_the_hash_and_an_amendment_carries_them(tmp_path: Path) -> None:
    from core import frozen_protocol

    protocol = {"grid": {"h": [0.1]}}
    record = frozen_protocol.freeze(tmp_path, protocol, approved_by="t", source="plan.md", sources=SOURCES)
    assert record["sha256"] == frozen_protocol.sha256(protocol)
    assert frozen_protocol.load(tmp_path).get("sources_problem") is None
    pending = {"n": 1, "proposed_protocol": {"grid": {"h": [0.2]}}, "proposed_sha256": frozen_protocol.sha256({"grid": {"h": [0.2]}})}
    frozen_protocol.apply(tmp_path, pending, {"approved_by": "p"}, raw_root=None)
    after = frozen_protocol.load(tmp_path)
    assert after["sources"] == SOURCES and after.get("sources_problem") is None


def test_an_edited_list_of_sources_in_the_frozen_record_is_not_believed(tmp_path: Path) -> None:
    from core import frozen_protocol

    frozen_protocol.freeze(tmp_path, {"grid": {"h": [0.1]}}, approved_by="t", source="plan.md", sources=SOURCES)
    path = frozen_protocol.frozen_path(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["sources"].append({"label": "7", "title": "added later", "doi": "", "url": ""})
    path.write_text(json.dumps(data), encoding="utf-8")
    assert "edited after the freeze" in frozen_protocol.load(tmp_path)["sources_problem"]


@pytest.mark.asyncio
async def test_the_evidence_judges_citations_against_the_sources_as_numbered_at_the_freeze(tmp_path: Path) -> None:
    eng, _seen = _engine(tmp_path, _protocol(oracles=[{**GOOD_ORACLE, "reference": "[2], table 3"}]))
    state = {"topic": "RK4", "iteration": 0, "literature": _literature()}
    await eng._node_plan(state)
    eng._freeze_protocol_if_due(state)
    protocol = eng._protocol_block(state)
    # The paper later cites one source only, renumbered: [2] no longer exists in the state's list.
    after_write = {**state, "literature": _literature()[:1]}
    assert eng._oracle_source_gaps(after_write, protocol) == []


def test_a_quest_frozen_before_the_sources_were_kept_is_judged_only_on_empty_references(tmp_path: Path) -> None:
    from core import frozen_protocol

    eng, _seen = _engine(tmp_path, _protocol())
    protocol = {"oracles": [{**GOOD_ORACLE, "reference": "Butcher 2008"}, {**GOOD_ORACLE, "name": "b", "reference": ""}]}
    frozen_protocol.freeze(eng.quest_root, protocol, approved_by="t", source="plan.md")
    gaps = eng._oracle_source_gaps({"literature": []}, protocol)
    assert len(gaps) == 1 and "'b'" in gaps[0] and "empty" in gaps[0]


def test_a_derivation_that_starts_with_from_or_in_but_writes_a_relation_is_a_derivation() -> None:
    for ref in ("derivation: from y' = -y, y(1) = e^-1 = 0.3679", "derivation: In the limit h -> 0, error = C h^4",
                "derivation: Per unit mass, E = v^2/2 = 0.5", "derivation: Re 2000 flow, u = 1 m/s gives Cd = 0.4",
                "derivation: with the June 2020 data, N = 120 + 30 = 150"):
        assert _gaps(ref) == [], ref


def test_a_recalled_value_written_as_a_derivation_is_not_one() -> None:
    for ref in ("derivation: value 1.637e-08 from Butcher's textbook",
                "derivation: recalled from Hairer's book; value is 1.637e-08",
                "derivation: e = 1.637e-08, as the handbook gives it"):
        assert _gaps(ref) != [], ref


def test_a_second_label_right_after_the_first_is_read_too() -> None:
    (gap,) = _gaps("[1][7]")
    assert "[7]" in gap


def test_going_on_without_a_source_never_counts_as_showing_the_checks_fi_added(tmp_path: Path) -> None:
    eng, seen = _engine(tmp_path, _protocol(), research=True)
    eng.quest_root.mkdir(parents=True, exist_ok=True)
    plan.plan_path(eng.quest_root).write_text("x", encoding="utf-8")
    eng._oracles_added_write({"oracles": ["added one"], "shown": False})
    protocol = {"oracles": [{**GOOD_ORACLE, "name": "added one", "reference": ""}]}
    eng._check_plan_sources({"literature": _literature()}, stop=True, protocol=protocol)
    assert seen == []
    assert eng._oracles_added_read()["shown"] is False, "nobody was shown it, so the freeze never says they approved it"


@pytest.mark.asyncio
async def test_an_analysis_of_existing_data_is_not_checked(tmp_path: Path) -> None:
    eng, seen = _engine(tmp_path, _protocol(oracles=[{**GOOD_ORACLE, "reference": ""}]), research=True)
    eng.config.engine.analyze_local_first = True
    assert eng._check_plan_sources({"literature": _literature()}, stop=True, protocol=_protocol(
        oracles=[{**GOOD_ORACLE, "reference": ""}])) == []
