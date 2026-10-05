"""A quest stopped because its checks do not say where their expected values come from has a way on.

A real quest (rigor_profile: research, a search for the best design, a Copilot model) wrote a plan whose model behind
the numbers showed only "Where it holds", and whose three checks had no kind and no reference. It stopped at the plan;
`@fi /resume <id> --revise-plan "..."` resumed it into the same stop (the words were dropped), and the only ways on were
to edit plan.md by hand. These are that quest's failures, one test each. Where a check's expected value comes from is
not a question a person is asked: FI fills it in once itself, and when it finds none the quest goes on by itself with each such check marked "source not confirmed" in the
evidence, the freeze and the paper. A person can still change the plan in words, or go on under their own name.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from core import accepted_checks, oracle_check, plan
from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, PausesConfig, ProviderConfig,
)
from core.engine import Engine, _plan_checks_note

SOURCES = [
    {"label": "1", "title": "Illumination optics for lithography and the shape of the source", "doi": "", "url": ""},
    {"label": "2", "title": "Partially coherent imaging and the transmission cross coefficient", "doi": "", "url": ""},
]

GOOD_MODEL = {
    "summary": "Hopkins imaging of a partially coherent source",
    "assumptions": ["scalar optics", "thin mask"],
    "holds_for": "NA below 0.9",
    "equations": [{"id": "E1", "formula": "I(x) = sum_s S(s) |H * M|^2", "role": "generates", "source": "[2]"}],
}


def _oracles(reference: str = "") -> list[dict[str, Any]]:
    return [
        {"name": "source_normalization", "check": "sum of the source weights", "expected": 1.0, "tolerance": 1e-9,
         "reference": reference},
        {"name": "horizontal_symmetry", "check": "x-centroid of the source", "expected": 0.0, "tolerance": 1e-9,
         "reference": reference},
    ]


# --- 1. what a model writes is read, not silently lost --------------------------------------------------------------


def test_equations_written_as_a_mapping_of_ids_or_as_lines_keep_every_equation() -> None:
    by_id, notes = plan.repair_protocol({"model": {"summary": "x", "equations": {"E1": "I = |E|^2", "E2": "c = I_max - I_min"}}})
    assert [e["id"] for e in by_id["model"]["equations"]] == ["E1", "E2"] and notes == []
    lines, notes = plan.repair_protocol({"model": {"summary": "x", "equations": ["E1: I = |E|^2", "c = I_max - I_min"]}})
    assert [(e["id"], e["formula"]) for e in lines["model"]["equations"]] == [("E1", "I = |E|^2"), ("E2", "c = I_max - I_min")]
    # A plan read back hashes as it was written.
    assert plan.normalize_protocol(lines)[0] == lines


def test_one_check_that_cannot_be_read_is_left_out_not_every_check() -> None:
    fixed, notes = plan.repair_protocol({"oracles": [*_oracles("[1]"), {"name": "bad", "expected": {"x": 1}}]})
    assert [o["name"] for o in fixed["oracles"]] == ["source_normalization", "horizontal_symmetry"]
    assert any("'bad'" in n and "left out" in n for n in notes), notes


def test_a_kind_or_reference_written_as_a_list_is_read_as_text() -> None:
    fixed, why = plan.normalize_protocol({"oracles": [{**_oracles()[0], "kind": ["invariant"], "reference": ["[1]", "eq. 3"]}]})
    assert why is None
    assert fixed["oracles"][0]["kind"] == "invariant" and fixed["oracles"][0]["reference"] == "[1]; eq. 3"


@pytest.mark.parametrize("written, kind", [
    ("normalization", "invariant"), ("Normalisation", "invariant"), ("horizontal_symmetry", "symmetry"),
    ("mirror symmetry", "symmetry"), ("mass conservation check", "invariant"), ("exact solution", "special_case"),
    ("baseline", None), ("independent_of_grid", None), ("zeroth order limit", "special_case"), ("closed-form limit", "special_case"),
])
def test_the_kinds_a_real_plan_wrote_are_read_as_one_of_the_six(written: str, kind: str | None) -> None:
    assert oracle_check.kind_of({"name": "x", "kind": written}) == kind
    assert oracle_check.kind_of({"name": "x", "type": written}) == kind, "`type` is read as the kind"


def test_a_reference_under_another_key_or_inside_the_check_is_read() -> None:
    base = _oracles()[0]
    assert oracle_check.source_gaps({"oracles": [{**base, "source": "[1], section 2"}]}, SOURCES) == []
    assert oracle_check.source_gaps({"oracles": [{**base, "basis": "derivation: sum of S over the pupil = 1 by definition"}]},
                                    SOURCES) == []
    in_check = {**base, "check": "sum of the source weights (derivation: sum of S over the pupil = 1 by definition)"}
    assert oracle_check.source_gaps({"oracles": [in_check]}, SOURCES) == []
    assert oracle_check.source_gaps({"oracles": [base]}, SOURCES), "an empty reference is still a gap"


def test_a_key_or_a_word_that_means_something_else_is_not_read_as_a_reference() -> None:
    base = _oracles()[0]
    # In optics the `source` is the thing simulated, and a `basis` is a set of functions.
    (gap,) = oracle_check.source_gaps({"oracles": [{**base, "source": "annular, sigma 0.5-0.8"}]}, SOURCES)
    assert "`reference` is empty" in gap
    assert oracle_check.source_gaps({"oracles": [{**base, "basis": "Legendre polynomials"}]}, SOURCES)
    assert oracle_check.source_gaps({"oracles": [{**base, "check": "x-centroid of the source: S(x) from E1 is symmetric"}]},
                                    SOURCES), "'source:' in a check's words is not a reference"
    # A frozen quest judged without its sources is judged on `reference` alone, as it always was.
    assert oracle_check.empty_references({"oracles": [{**base, "citation": "[1]"}]})
    # A label is not a sentence saying what produces the numbers.
    assert "what produces the numbers" in oracle_check.model_missing({"model": {"name": "Hopkins"}})
    # A `;` inside a formula is not the end of an equation.
    eqs = oracle_check.equation_items("E1: f(x; t) = a; E2: g = b")
    assert [(e["id"], e["formula"]) for e in eqs] == [("E1", "f(x; t) = a"), ("E2", "g = b")]


def test_a_value_alone_is_not_a_derivation_but_a_fact_by_definition_is() -> None:
    assert oracle_check.source_gaps({"oracles": _oracles("derivation: error = 1.637e-08")}, SOURCES)
    assert oracle_check.source_gaps({"oracles": _oracles("derivation: y(1) = 0.3679")}, SOURCES)
    assert oracle_check.source_gaps({"oracles": _oracles("derivation: S = 1 by definition")}, SOURCES) == []


def test_a_model_with_its_parts_under_other_names_is_shown_and_not_reported_missing() -> None:
    model = {"holds_for": "NA below 0.9", "description": "Hopkins imaging", "assumes": ["scalar optics"],
             "governing_equations": {"E1": {"formula": "I = |E|^2", "role": "generation", "reference": "[2]"}}}
    protocol = plan.normalize_protocol({"oracles": _oracles("[1]"), "model": model})[0]
    assert oracle_check.model_missing(protocol) == []
    section = plan.render("t", {}, {"hypothesis": "h", "protocol": protocol}).split(f"## {plan.MODEL_HEADING}")[1]
    section = section.split("\n## ")[0]
    assert "Hopkins imaging" in section and "scalar optics" in section and "I = |E|^2" in section
    assert "(not written)" not in section and "(none written)" not in section
    assert oracle_check.generating_equations(protocol) == ["E1"]
    assert not [n for n in oracle_check.model_notes(protocol, SOURCES) if "E1" in n], "its source [2] was read"


def test_a_derivation_true_by_definition_passes_and_a_plain_sentence_is_told_to_write_an_equation() -> None:
    assert oracle_check.source_gaps({"oracles": _oracles("derivation: sum S = 1 by definition")}, SOURCES) == []
    (gap, _) = oracle_check.source_gaps({"oracles": _oracles("derivation: the source is normalised so it integrates to one")},
                                        SOURCES)
    assert "at least one equation (with `=`" in gap
    assert "exp(-1)" not in gap, "the example is not tied to one field"
    assert oracle_check.source_gaps({"oracles": _oracles("derivation: x = 1")}, SOURCES), "a value alone is not a derivation"


# --- the prompts --------------------------------------------------------------------------------------------------


def test_a_rewrite_of_the_plan_is_told_what_fi_reads_and_what_is_missing() -> None:
    text = plan.render("t", {}, {"hypothesis": "h", "protocol": {"oracles": _oracles(), "model": {"holds_for": "NA < 0.9"}}},
                       sources=SOURCES)
    note = _plan_checks_note(text)
    assert "never read" in note and "`reference`" in note and "at least one equation" in note
    assert "'source_normalization'" in note and "what produces the numbers" in note
    assert _plan_checks_note(plan.render("t", {}, {"hypothesis": "h"})) == "", "a plan with no checks gets no note"


def test_the_listed_sources_are_read_back_from_the_plan() -> None:
    text = plan.render("t", {}, {"hypothesis": "h", "protocol": {"oracles": _oracles("[2]")}}, sources=SOURCES)
    assert [s["label"] for s in plan.listed_sources(text)] == ["1", "2"]


# --- the engine ------------------------------------------------------------------------------------------------------


def _cfg(tmp_path: Path) -> Config:
    return Config(
        topic="the best illumination source for a line pattern", title="src", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, clarify_mode="off"),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60), knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "out"), pauses=PausesConfig(plan="off"),
    )


class Paused(Exception):
    pass


def _literature() -> list[dict[str, Any]]:
    return [{"content": "x", "metadata": {"title": s["title"], "source": "openalex", "url": f"https://e.org/{s['label']}"}}
            for s in SOURCES]


STATE = {"topic": "source", "iteration": 0, "literature": _literature()}


def _engine(tmp_path: Path, replies: list[str], *, oracles: list[dict[str, Any]] | None = None,
            model: Any = GOOD_MODEL) -> tuple[Engine, list[dict[str, Any]]]:
    eng = Engine(_cfg(tmp_path))
    # The profile's own stop (as test_plan_model_sources does it), without the plain plan stop that comes with it.
    eng.config = eng.config.model_copy(update={"rigor_profile": "research"})
    protocol = {"grid": {"sigma": [0.3, 0.5]}, "oracles": oracles if oracles is not None else _oracles(),
                **({"model": model} if model is not None else {})}
    design = {"hypothesis": "an annular source prints the line best", "method": "sweep sigma", "protocol": protocol,
              "plan": {"in_short": "x", "literature": [{"source": "[1]", "says": "sources are normalised"}]}}
    eng._client = type("Stub", (), {"chat": AsyncMock(side_effect=[json.dumps(design), json.dumps({"objections_addressed": []}),
                                                                   *replies])})()
    seen: list[dict[str, Any]] = []

    def fake(**kwargs: Any) -> None:
        seen.append(kwargs)
        raise Paused(kwargs["kind"])

    eng._pause_for_human = fake  # type: ignore[method-assign]
    return eng, seen


def _with(text: str, reference: str) -> str:
    return text.replace("reference: ''", f"reference: '{reference}'")


@pytest.mark.asyncio
async def test_the_plan_prompt_the_model_gets_carries_the_model_and_reference_rules(tmp_path: Path) -> None:
    eng, _seen = _engine(tmp_path, [], oracles=_oracles("[1]"))
    await eng._node_plan(dict(STATE))
    prompt = _content(eng._client.chat.await_args_list[0])
    assert "`model` says what produces the numbers" in prompt and "Its `reference` says where `expected` comes from" in prompt


def _content(call: Any) -> str:
    """The text a recorded chat call sent."""
    return "\n".join(str(m.get("content")) for m in call.args[0])


def _chat_by_step(eng: Engine, revise: Any) -> list[tuple[str, str]]:
    """A model that answers each step of the plan node as a model would, and a rewrite of the plan with ``revise``."""
    calls: list[tuple[str, str]] = []

    async def chat(messages: Any, **kw: Any) -> str:
        node = str(kw.get("node") or "")
        calls.append((node, "\n".join(str(m.get("content")) for m in messages)))
        if node == "plan":
            return json.dumps({"hypothesis": "an annular source prints the line best", "method": "sweep sigma",
                               "protocol": {"grid": {"sigma": [0.3, 0.5]}, "oracles": _oracles(), "model": GOOD_MODEL},
                               "plan": {"in_short": "x"}})
        if node == "plan_revise":
            return revise(plan.plan_path(eng.quest_root).read_text(encoding="utf-8"))
        if node == "plan_criteria":
            return json.dumps({"criteria": []})
        return json.dumps({"objections_addressed": []})

    eng._client = type("Stub", (), {"chat": staticmethod(chat)})()
    return calls


@pytest.mark.asyncio
async def test_under_research_fi_fills_the_sources_in_once_before_it_would_stop(tmp_path: Path) -> None:
    eng, seen = _engine(tmp_path, [])
    # The fill's reply: the plan as written, with each check's reference filled in (and nothing else changed).
    calls = _chat_by_step(eng, lambda text: _with(text, "derivation: sum of S over the pupil = 1 by definition"))
    await eng._node_plan(dict(STATE))
    assert seen == [], "the model filled it in, so the quest does not stop"
    fills = [text for node, text in calls if node == "plan_revise"]
    assert len(fills) == 1 and "Fill in where each check's expected value comes from" in fills[0]
    assert "at least one equation (with `=`" in fills[0]
    rows = plan.history(eng.quest_root)
    assert rows[0]["by"] == "model" and rows[-1]["by"] == "engine" and rows[-1]["note"].startswith("Fill in where")
    text = plan.plan_path(eng.quest_root).read_text(encoding="utf-8")
    assert "from: derivation: sum of S over the pupil = 1 by definition" in text, "the section is shown again from the block"


@pytest.mark.asyncio
async def test_a_fill_that_changes_a_checks_number_is_put_back_and_the_quest_goes_on_marked(tmp_path: Path) -> None:
    eng, seen = _engine(tmp_path, [])
    calls = _chat_by_step(eng, lambda text: _with(text, "derivation: sum S = 1 by definition").replace(
        "expected: 1.0", "expected: 2.0"))
    await eng._node_plan(dict(STATE))
    assert seen == [], "no stop: FI goes on by itself"
    assert plan.parse(plan.plan_path(eng.quest_root).read_text(encoding="utf-8")).design["protocol"]["oracles"][0][
        "expected"] == 1.0, "the check's number is as the plan had it"
    assert "put back" in plan.history(eng.quest_root)[-1]["note"]
    record = accepted_checks.accepted(eng.quest_root)
    assert record and {c["by"] for c in record["chosen"].values()} == {accepted_checks.AUTOMATIC}
    # Asked once: a resume asks the model no second time.
    await eng._node_plan(dict(STATE))
    assert [node for node, _text in calls].count("plan_revise") == 1


@pytest.mark.asyncio
async def test_when_fi_finds_no_source_the_quest_goes_on_marked_and_asks_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    from core import audit_log

    eng, seen = _engine(tmp_path, [])  # the fill call fails (no reply)
    await eng._node_plan(dict(STATE))
    assert seen == [], "no question about where a value comes from"
    out = capsys.readouterr().out
    assert "do not say where their expected value comes from, and FI could not find it" in out
    assert "Nothing to do" in out
    assert accepted_checks.pending(eng.quest_root) is None, "no surface offers a choice"
    chosen = accepted_checks.accepted(eng.quest_root)["chosen"]
    assert sorted(chosen) == ["horizontal_symmetry", "source_normalization"]
    assert all(c["by"] == accepted_checks.AUTOMATIC for c in chosen.values())
    events = [e for e in audit_log.read(eng.audit.path) if e.get("check") == "oracle_sources"]
    assert events and "FI went on by itself" in events[-1]["summary"]


@pytest.mark.asyncio
async def test_a_change_in_words_that_fills_one_source_leaves_only_the_other_marked(tmp_path: Path) -> None:
    eng, seen = _engine(tmp_path, [])
    _chat_by_step(eng, lambda text: "I could not do that.")
    await eng._node_plan(dict(STATE))
    path = plan.plan_path(eng.quest_root)
    # The person asks for a change; the model fills in one check properly and the other in words with no equation.
    revised = path.read_text(encoding="utf-8").replace(
        "reference: ''", "reference: 'derivation: sum S = 1 by definition'", 1).replace(
        "reference: ''", "reference: 'derivation: the source is symmetric so its centroid is at zero'", 1)
    eng._client = type("Stub", (), {"chat": AsyncMock(return_value=revised)})()
    await eng.revise_plan("change the checks by looking at literature and formulas")
    prompt = _content(eng._client.chat.await_args_list[0])
    assert "never read" in prompt and "'source_normalization'" in prompt, "the rewrite was told what FI reads"
    await eng._node_plan(dict(STATE))
    assert seen == []
    assert eng._not_confirmed_names(dict(STATE), eng._draft_protocol(dict(STATE))) == ["horizontal_symmetry"]


@pytest.mark.asyncio
async def test_a_revision_that_fills_the_references_lets_the_quest_go_on(tmp_path: Path) -> None:
    from core.engine import _REFERENCE_FORMS

    eng, seen = _engine(tmp_path, [])
    await eng._node_plan(dict(STATE))
    path = plan.plan_path(eng.quest_root)
    revised = _with(path.read_text(encoding="utf-8"), "derivation: sum of S over the pupil = 1 by definition")
    eng._client = type("Stub", (), {"chat": AsyncMock(return_value=revised)})()
    await eng.revise_plan("say where the checks' values come from")
    assert _REFERENCE_FORMS in _content(eng._client.chat.await_args_list[0]), "the rule the check reads, in the prompt"
    assert eng._check_plan_sources(dict(STATE), stop=True) == []
    assert seen == [] and accepted_checks.pending(eng.quest_root) is None
    assert eng._not_confirmed_names(dict(STATE), eng._draft_protocol(dict(STATE))) == []


@pytest.mark.asyncio
async def test_going_on_by_itself_is_recorded_everywhere_and_relaxes_nothing(tmp_path: Path) -> None:
    from core import frozen_protocol

    eng, seen = _engine(tmp_path, [])
    await eng._node_plan(dict(STATE))
    assert seen == []
    # The evidence keeps the gap, marked; the freeze and the paper say so, and that FI went on by itself.
    gaps = eng._oracle_source_gaps(dict(STATE), eng._draft_protocol(dict(STATE)))
    assert len(gaps) == 2 and all("source not confirmed: FI (no source could be found) went on without one" in g
                                  for g in gaps)
    names = eng._not_confirmed_names(dict(STATE), eng._draft_protocol(dict(STATE)))
    assert names == ["source_normalization", "horizontal_symmetry"]
    disclosure = accepted_checks.disclosure(eng.quest_root, names)
    assert "'source_normalization'" in disclosure and "FI (no source could be found) went on without one" in disclosure
    assert "Do not describe these checks as validated against an independent source" in disclosure
    # A check given a source afterwards is no longer said to have none (not in the freeze, not in the paper).
    path = plan.plan_path(eng.quest_root)
    path.write_text(path.read_text(encoding="utf-8").replace("reference: ''", "reference: '[1], section 2'", 1),
                    encoding="utf-8")
    assert eng._not_confirmed_names(dict(STATE), eng._draft_protocol(dict(STATE))) == ["horizontal_symmetry"]
    eng._freeze_protocol_if_due(dict(STATE))
    approved = frozen_protocol.load(eng.quest_root)["approved_by"]
    assert "source not confirmed" in approved and "'horizontal_symmetry'" in approved
    assert "'source_normalization'" not in approved


@pytest.mark.asyncio
async def test_a_persons_own_choice_keeps_their_name(tmp_path: Path) -> None:
    eng, seen = _engine(tmp_path, [])
    by_name = {o["name"]: o for o in _oracles()}
    accepted_checks.write_pending(eng.quest_root, [
        {"name": n, "why": f"the check '{n}' ...", "expected": o["expected"], "fingerprint": accepted_checks.fingerprint(o)}
        for n, o in by_name.items()], plan_version=1)
    assert accepted_checks.accept(eng.quest_root, "  ", via="cli")[0] is False, "a person's choice needs a name"
    assert accepted_checks.accept(eng.quest_root, "Jun", via="cli")[0]
    await eng._node_plan(dict(STATE))
    assert seen == []
    chosen = accepted_checks.accepted(eng.quest_root)["chosen"]
    assert {c["by"] for c in chosen.values()} == {"Jun"}, "FI going on by itself never overwrites a person's choice"


@pytest.mark.asyncio
async def test_a_check_whose_numbers_change_after_going_on_is_marked_again(tmp_path: Path) -> None:
    eng, seen = _engine(tmp_path, [])
    await eng._node_plan(dict(STATE))
    before = accepted_checks.accepted(eng.quest_root)["chosen"]["source_normalization"]["fingerprint"]
    path = plan.plan_path(eng.quest_root)
    path.write_text(path.read_text(encoding="utf-8").replace("expected: 1.0", "expected: 5.0"), encoding="utf-8")
    eng._went_on_unsourced = False
    await eng._node_plan(dict(STATE))
    assert seen == []
    after = accepted_checks.accepted(eng.quest_root)["chosen"]["source_normalization"]
    assert after["fingerprint"] != before and after["expected"] == 5.0, "the record names the numbers as they are now"


def test_a_pending_record_without_names_cannot_be_gone_on_with(tmp_path: Path) -> None:
    accepted_checks.write_pending(tmp_path, [{"name": "", "why": "the check ..."}], plan_version=1)
    ok, message = accepted_checks.accept(tmp_path, "Jun", via="cli")
    assert not ok and "resume the quest" in message


@pytest.mark.asyncio
async def test_an_incomplete_answer_to_the_fill_does_not_use_up_the_one_ask(tmp_path: Path) -> None:
    from core.provider import ModelAnswerTruncated

    eng, seen = _engine(tmp_path, [])
    calls: list[str] = []

    async def chat(messages: Any, **kw: Any) -> str:
        node = str(kw.get("node") or "")
        calls.append(node)
        if node == "plan":
            return json.dumps({"hypothesis": "h", "method": "m", "plan": {"in_short": "x"},
                               "protocol": {"grid": {"sigma": [0.3]}, "oracles": _oracles(), "model": GOOD_MODEL}})
        if node == "plan_revise":
            raise ModelAnswerTruncated("cut off", node=node)
        if node == "plan_criteria":
            return json.dumps({"criteria": []})
        return json.dumps({"objections_addressed": []})

    eng._client = type("Stub", (), {"chat": staticmethod(chat)})()
    await eng._node_plan(dict(STATE))
    assert seen == [], "no stop"
    assert not (eng.fi_dir / "plan_sources_asked.json").exists(), "a cut-off answer does not use up the one ask"


# A real plan (claude haiku, 2026-10-03) named scipy as a second implementation and wrote its call into the reference:
# `t_span=[0, 10], y0=[1, 0]` was read as citing sources [0], [10] and [1], and the quest stopped for a "[0]" it never
# cited, even after the fill.
_SCIPY_REFERENCE = ("Independent second implementation: scipy.integrate.solve_ivp(f, t_span=[0, 10], y0=[1, 0], "
                    "rtol=1e-12); shares no code with the simulation.")


def test_a_list_written_as_code_is_not_read_as_citing_sources() -> None:
    oracle = {"name": "scipy_vs_exact", "kind": "second_implementation", "check": "x(10) by scipy", "expected": 0.0,
              "tolerance": 1e-8, "reference": _SCIPY_REFERENCE}
    assert oracle_check.reference_problem(oracle, None, SOURCES) is None
    spaced = {**oracle, "reference": _SCIPY_REFERENCE.replace("y0=[1, 0]", "y0 = [1, 0]")}
    assert oracle_check.reference_problem(spaced, None, SOURCES) is None
    # A derivation over an interval that starts at 0 cites nothing either; a real citation in it still counts.
    over = {**oracle, "kind": "analytic", "reference": "derivation: the integral of 2x over [0, 1] = 1, as in [2]"}
    assert oracle_check.reference_problem(over, None, SOURCES) is None
    assert oracle_check._cites(over["reference"], SOURCES) == (["[2]"], [])
    # Each rule on its own: a list assigned to a name, and one passed to a call, with no 0 in either.
    assert oracle_check._cites("y0=[1, 7] and t_span = [2, 9]", SOURCES) == ([], [])
    assert oracle_check._cites("solve_ivp(f, [1, 10], y)", SOURCES) == ([], [])
    # Still citations: one number alone, also after `=`; a group in prose, also in parentheses; a comparison.
    assert oracle_check._cites("as in ref=[2]", SOURCES) == (["[2]"], [])
    assert oracle_check._cites("the result (see [1, 2])", SOURCES) == (["[1]", "[2]"], [])
    assert oracle_check._cites("x == [1, 2]", SOURCES) == (["[1]", "[2]"], [])


def test_a_source_number_that_does_not_exist_is_named_with_the_ones_that_do() -> None:
    oracle = {"name": "x_exact", "check": "x(10)", "expected": 0.3, "tolerance": 1e-6, "reference": "[0]"}
    why = oracle_check.reference_problem(oracle, None, SOURCES + [{"label": "W1", "title": "a web page"}])
    assert why is not None and "did not retrieve" in why
    assert "[0] is not one of the sources this quest found (they are [1]–[2] and [W1])" in why
    gap = SOURCES + [{"label": "5", "title": "t"}]
    assert oracle_check.labels_note(gap) == "[1]–[2], [5]"


@pytest.mark.asyncio
async def test_the_fill_request_says_plainly_that_a_source_number_does_not_exist(tmp_path: Path) -> None:
    eng, seen = _engine(tmp_path, [], oracles=[{**o, "reference": "[0]"} for o in _oracles()])
    calls: list[tuple[str, str]] = []

    async def chat(messages: Any, **kw: Any) -> str:
        node = str(kw.get("node") or "")
        calls.append((node, "\n".join(str(m.get("content")) for m in messages)))
        if node == "plan":
            return json.dumps({"hypothesis": "h", "method": "m", "plan": {"in_short": "x"},
                               "protocol": {"grid": {"sigma": [0.3]}, "model": GOOD_MODEL,
                                            "oracles": [{**o, "reference": "[0]"} for o in _oracles()]}})
        if node == "plan_revise":
            return _with(plan.plan_path(eng.quest_root).read_text(encoding="utf-8").replace("reference: '[0]'",
                                                                                            "reference: ''"),
                         "derivation: sum of S over the pupil = 1 by definition")
        if node == "plan_criteria":
            return json.dumps({"criteria": []})
        return json.dumps({"objections_addressed": []})

    eng._client = type("Stub", (), {"chat": staticmethod(chat)})()
    await eng._node_plan(dict(STATE))
    fills = [text for node, text in calls if node == "plan_revise"]
    assert len(fills) == 1
    assert "[0] is not one of the sources this quest found (they are [1]–[2])" in fills[0]


@pytest.mark.asyncio
async def test_a_fill_whose_design_block_breaks_is_asked_for_the_block_once_and_then_kept(tmp_path: Path) -> None:
    eng, seen = _engine(tmp_path, [])
    calls: list[tuple[str, str]] = []
    filled: dict[str, str] = {}

    async def chat(messages: Any, **kw: Any) -> str:
        node = str(kw.get("node") or "")
        text = "\n".join(str(m.get("content")) for m in messages)
        calls.append((node, text))
        if node == "plan":
            return json.dumps({"hypothesis": "an annular source prints the line best", "method": "sweep sigma",
                               "protocol": {"grid": {"sigma": [0.3, 0.5]}, "oracles": _oracles(), "model": GOOD_MODEL},
                               "plan": {"in_short": "x"}})
        if node == "plan_revise" and "Return ONLY the corrected block" not in text:
            good = _with(plan.plan_path(eng.quest_root).read_text(encoding="utf-8"),
                         "derivation: sum of S over the pupil = 1 by definition")
            filled["block"] = plan.design_block(good)[2]
            # An unclosed quote: no repair can say where the value was meant to end.
            return good.replace("method: sweep sigma", "method: 'sweep sigma")
        if node == "plan_revise":
            return "```yaml\n" + filled["block"] + "\n```"
        if node == "plan_criteria":
            return json.dumps({"criteria": []})
        return json.dumps({"objections_addressed": []})

    eng._client = type("Stub", (), {"chat": staticmethod(chat)})()
    await eng._node_plan(dict(STATE))
    assert seen == [], "the corrected block filled the sources in, so the quest does not stop"
    revise = [text for node, text in calls if node == "plan_revise"]
    assert len(revise) == 2 and "Where: line" in revise[1] and "Fill in where" not in revise[1]
    assert "from: derivation: sum of S over the pupil = 1 by definition" in plan.plan_path(eng.quest_root).read_text(
        encoding="utf-8")


def test_a_fill_may_only_add_sources_kinds_and_missing_parts_of_the_model() -> None:
    from core.engine import _fill_changed_more

    base = {"hypothesis": "h", "protocol": {"grid": {"s": [1]}, "oracles": _oracles(), "model": GOOD_MODEL}}

    def with_oracle(**over: Any) -> dict[str, Any]:
        return {**base, "protocol": {**base["protocol"], "oracles": [{**_oracles()[0], **over}, _oracles()[1]]}}

    assert _fill_changed_more(base, with_oracle(reference="derivation: sum S = 1 by definition", kind="invariant")) == ""
    assert _fill_changed_more(base, with_oracle(check="sum of the source weights (derivation: sum S = 1 by definition)")) == ""
    assert "rewrote what the check" in _fill_changed_more(base, with_oracle(check="sum of three weights"))
    assert _fill_changed_more(base, with_oracle(case={"sigma": 0.3}))
    assert _fill_changed_more(base, with_oracle(measure="w_sum"))
    assert _fill_changed_more(base, {**base, "protocol": {**base["protocol"], "grid": {"s": [2]}}})
    changed_eq = {**GOOD_MODEL, "equations": [{**GOOD_MODEL["equations"][0], "formula": "I = 0"}]}
    assert "equation E1" in _fill_changed_more(base, {**base, "protocol": {**base["protocol"], "model": changed_eq}})
    assert "could not be read" in _fill_changed_more(None, base)
    # `source` may be filled with a reference where it was empty, but the illumination source it named is not the fill's.
    assert _fill_changed_more(base, with_oracle(source="[1], section 2")) == ""
    physical = {**base, "protocol": {**base["protocol"], "oracles": [{**_oracles()[0], "source": "annular, sigma 0.5"}]}}
    changed = {**base, "protocol": {**base["protocol"], "oracles": [{**_oracles()[0], "source": "dipole, sigma 0.2"}]}}
    assert _fill_changed_more(physical, changed)
    # Nor are the model's other keys.
    with_params = {**base, "protocol": {**base["protocol"], "model": {**GOOD_MODEL, "parameters": {"NA": 0.6}}}}
    other_params = {**base, "protocol": {**base["protocol"], "model": {**GOOD_MODEL, "parameters": {"NA": 0.9}}}}
    assert "parameters" in _fill_changed_more(with_params, other_params)


@pytest.mark.asyncio
async def test_a_check_added_after_going_on_is_marked_too(tmp_path: Path) -> None:
    eng, seen = _engine(tmp_path, [])
    await eng._node_plan(dict(STATE))
    path = plan.plan_path(eng.quest_root)
    design = plan.parse(path.read_text(encoding="utf-8")).design
    extra = {"name": "baseline_design_check", "check": "baseline", "expected": 3.0, "tolerance": 0.1, "reference": ""}
    design["protocol"]["oracles"].append(extra)
    path.write_text(plan.render("t", {}, design, sources=SOURCES), encoding="utf-8")
    eng._went_on_unsourced = False
    await eng._node_plan(dict(STATE))
    assert seen == []
    assert "baseline_design_check" in accepted_checks.accepted(eng.quest_root)["chosen"]


# --- the three surfaces ------------------------------------------------------------------------------------------------


def _stopped_quest(root: Path, name: str = "q-1") -> Path:
    quest = root / name
    (quest / ".fi").mkdir(parents=True)
    accepted_checks.write_pending(quest, [{"name": "source_normalization", "why": "the check 'source_normalization' ...",
                                           "expected": 1.0, "fingerprint": accepted_checks.fingerprint(_oracles()[0])}],
                                  plan_version=1)
    return quest


def test_the_cli_records_going_on_with_a_name(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import launch

    _stopped_quest(tmp_path)
    args = launch.parse_args(["--accept-checks", "q-1", "--approve-as", "Jun", "--output-root", str(tmp_path)])
    assert args.accept_checks == "q-1" and args.approve_as == "Jun"
    assert launch._accept_checks("q-1", "", tmp_path) == 2
    assert launch._accept_checks("nope", "Jun", tmp_path) == 1
    assert launch._accept_checks("q-1", "Jun", tmp_path) == 0
    assert "--resume q-1" in capsys.readouterr().out
    assert accepted_checks.accepted(tmp_path / "q-1")["by"] == "Jun"
    empty = tmp_path / "q-2"
    (empty / ".fi").mkdir(parents=True)
    assert launch._accept_checks("q-2", "Jun", tmp_path) == 1, "a quest not stopped for it has nothing to go on with"


def test_the_one_shot_rewrite_says_per_check_what_is_still_missing(tmp_path: Path) -> None:
    import launch

    text = plan.render("t", {}, {"hypothesis": "h", "protocol": {"oracles": [
        {**_oracles()[0], "reference": "derivation: sum S = 1 by definition"}, _oracles()[1]]}}, sources=SOURCES)
    path = tmp_path / "plan.md"
    path.write_text(text, encoding="utf-8")
    lines = launch._plan_sources_status(path)
    assert lines[0].startswith("1 of 2 check(s) still do not say")
    assert any("still missing" in line and "'horizontal_symmetry'" in line for line in lines)
    assert any("'source_normalization' says where" in line for line in lines)


def test_the_web_page_offers_go_on_as_it_is(tmp_path: Path) -> None:
    try:
        from fastapi.testclient import TestClient
    except Exception:  # pragma: no cover
        pytest.skip("fastapi not installed")
    from web.server import make_app

    root = tmp_path / "outputs"
    root.mkdir()
    client = TestClient(make_app(root))
    _stopped_quest(root, "p1")
    assert client.get("/api/quests/p1/plan/unsourced").json()["pending"]["checks"][0]["name"] == "source_normalization"
    refused = client.post("/api/quests/p1/plan/accept-checks", json={"who": ""})
    assert refused.status_code == 400 and "--approve-as" in refused.json()["detail"]
    ok = client.post("/api/quests/p1/plan/accept-checks", json={"who": "Jun"})
    assert ok.status_code == 200 and ok.json()["accepted"] is True
    page = (Path(__file__).resolve().parents[1] / "web" / "static" / "quest.html").read_text(encoding="utf-8")
    assert "Go on as it is" in page and "/plan/accept-checks" in page and "renderUnsourcedChecks(data.unsourced_checks, data.failed_checks)" in page


def test_vs_code_reads_revise_plan_after_resume_and_says_what_it_does_not_understand() -> None:
    """Runs the compiled argument parser of `@fi /resume` (vscode-frontier-insight/src/resume-args.ts) with node."""
    import shutil
    import subprocess

    ext = Path(__file__).resolve().parents[1] / "vscode-frontier-insight"
    compiled, source = ext / "out" / "resume-args.js", ext / "src" / "resume-args.ts"
    node = shutil.which("node")
    if node is None or not compiled.is_file() or compiled.stat().st_mtime < source.stat().st_mtime:
        pytest.skip("node, or a compiled vscode-frontier-insight/out/resume-args.js as new as its source, is not here "
                    "(npm run compile)")
    cases = [
        'q-1 --revise-plan "change the checks by looking at literature and formulas"',
        "q-1 --revise-plan “fill it in”",
        "q-1 --revise-plan=make it so",
        "q-1 --revise-plan",
        "q-1",
        'q-1 --watchme --from code --revise-plan "a --x b"',
    ]
    script = ("const m = require(process.argv[1]); const cases = JSON.parse(require('fs').readFileSync(0, 'utf8'));"
              "console.log(JSON.stringify(cases.map(c => [m.splitRevisePlan(c), m.unknownResumeFlags(c)])));")
    done = subprocess.run([node, "-e", script, str(compiled)], input=json.dumps(cases), capture_output=True,
                          text=True, encoding="utf-8", timeout=60)
    assert done.returncode == 0, done.stderr
    got = json.loads(done.stdout)
    assert got[0][0] == {"questId": "q-1", "request": "change the checks by looking at literature and formulas"}
    assert got[1][0] == {"questId": "q-1", "request": "fill it in"}
    assert got[2][0] == {"questId": "q-1", "request": "make it so"}
    assert got[3][0] == {"questId": "q-1", "request": ""}
    assert got[4] == [None, []]
    assert got[5][1] == ["--watchme"], "a flag inside the quoted request is not a flag of /resume"


def test_vs_code_has_go_on_as_it_is_and_reads_revise_plan_after_resume() -> None:
    ext = Path(__file__).resolve().parents[1] / "vscode-frontier-insight"
    source = (ext / "src" / "extension.ts").read_text(encoding="utf-8")
    skills = (ext / "src" / "skills.ts").read_text(encoding="utf-8")
    assert 'cmd === "accept-checks"' in source and "runAcceptChecks" in source
    assert "export async function runAcceptChecks" in skills and '"--accept-checks"' in skills and "showInputBox" in skills
    assert '"name": "accept-checks"' in (ext / "package.json").read_text(encoding="utf-8")
    # `/resume <id> --revise-plan "..."` rewrites the plan instead of resuming into the same stop.
    assert "splitRevisePlan(promptArgs)" in source and "unknownResumeFlags(promptArgs)" in source
