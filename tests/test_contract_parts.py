"""The plan's model is asked the contract's parts one at a time (core/contract_parts.py); the fake model here is a function,
never a real one. Neutral made-up study: a pond's algae count against nutrient level and temperature."""

from __future__ import annotations

import json
import string
from pathlib import Path
from typing import Any

import pytest

from core import contract, contract_parts as cp, metric_spec, numeric_warnings, plan, quality_report, rerun_from, run_checks

TEMPLATES = {n: string.Template((Path(__file__).parent.parent / "agents" / f"{n}.md").read_text(encoding="utf-8"))
             for n in ("plan_roles", "plan_names", "plan_comparisons")}

DESIGN: dict[str, Any] = {
    "hypothesis": "h", "method": "Simulate algae growth; compare with `growth_theory` computed by the closed form.",
    "protocol": {
        "grid": {"nutrient": [1, 2, 4], "temp": [10, 20]},
        "runs_per_setting": 50,
        "metrics": [
            {"id": "algae_peak", "kind": "mean", "estimand": "peak algae count", "unit": "one pond run"},
            {"id": "bloom", "kind": "proportion", "estimand": "P(bloom)", "unit": "one pond run"},
        ],
        "oracles": [{"name": "closed form", "kind": "special_case", "check": "c", "expected": 0.0, "tolerance": 0.1,
                     "case": {"nutrient": 1, "temp": 10}, "measure": "abs(algae_peak - growth_theory)",
                     "reference": "derivation: logistic growth"}],
        "model": {"summary": "logistic growth", "equations": [
            {"id": "E1", "formula": "N(t) = K / (1 + exp(-r t))", "role": "generates", "source": "derivation",
             "derivation": "solve dN/dt = rN(1-N/K)"}]},
    },
}

ROLES = {"results": {"algae_peak": {"role": "result", "depends_on": ["nutrient", "temp"], "returned": "each_run"},
                     "bloom": {"role": "physical_bound", "depends_on": ["nutrient"], "returned": "each_run"}}}
NAMES = {"names": {"growth_theory": {"answer": "reference", "what": "the closed-form peak", "equation": "E1"}}}
COMPARE = {"comparisons": [{"result": "algae_peak", "across": "temp", "values": [10, 20], "how": "difference"}]}


class Fake:
    """Answers each part from a queue of replies (a dict is sent as JSON, a string as it is, an exception is raised)."""

    def __init__(self, **replies: list[Any]) -> None:
        self.replies, self.prompts = replies, []

    async def __call__(self, prompt: str) -> str:
        part = next(k for k, marker in (("roles", "What kind of number it is"), ("names", "Say what each one is"),
                                        ("comparisons", "compares between values")) if marker in prompt)
        self.prompts.append((part, prompt))
        reply = self.replies[part].pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply if isinstance(reply, str) else json.dumps(reply)


def parse(text: str) -> Any:
    try:
        return json.loads(text)
    except ValueError:
        return None


async def run(fake: Fake, design: dict[str, Any] = DESIGN) -> cp.Outcome:
    return await cp.ask_parts(design, fake, parse, TEMPLATES)


def test_check_part_roles_hold_the_answer_to_declared_ids_and_varied_settings() -> None:
    assert contract.check_part("roles", ROLES["results"], DESIGN) == []
    bad = {"algae_peak": {"role": "result", "depends_on": ["salinity"], "returned": "sometimes"},
           "extra": {"role": "result", "depends_on": [], "returned": "each_run"}}
    problems = contract.check_part("roles", bad, DESIGN)
    assert any("salinity" in p for p in problems) and any("`bloom` is not answered" in p for p in problems)
    assert any("`extra` is not a declared result" in p for p in problems)
    assert any("returned must be each_run or afterwards" in p for p in problems)
    worse = {**ROLES["results"], "bloom": {"role": "great", "depends_on": [], "returned": "each_run"}}
    assert any("one of" in p for p in contract.check_part("roles", worse, DESIGN))


def test_a_result_another_is_given_a_subset_of_is_returned_by_each_run() -> None:
    d = json.loads(json.dumps(DESIGN))
    d["protocol"]["metrics"][0]["given"] = "bloom"
    later = {**ROLES["results"], "bloom": {**ROLES["results"]["bloom"], "returned": "afterwards"}}
    assert any("each run returns it" in p for p in contract.check_part("roles", later, d))
    assert contract.check_part("roles", ROLES["results"], d) == []


def test_the_undeclared_names_come_from_the_plan_by_shape_and_not_from_declared_ids_or_settings() -> None:
    found = contract.undeclared_names(DESIGN)
    assert list(found) == ["growth_theory"]
    assert "check" in found["growth_theory"]


def test_check_part_names_needs_a_formula_or_a_model_equation_for_a_reference() -> None:
    assert contract.check_part("names", NAMES["names"], DESIGN) == []
    no_source = {"growth_theory": {"answer": "reference", "equation": "E9"}}
    assert any("reference needs" in p for p in contract.check_part("names", no_source, DESIGN))
    assert any("inputs" in p for p in contract.check_part("names", {"growth_theory": {"answer": "derived"}}, DESIGN))
    assert contract.check_part("names", {"growth_theory": {"answer": "diagnostic"}}, DESIGN) == []


def test_check_part_comparisons_hold_to_declared_results_and_varied_settings() -> None:
    assert contract.check_part("comparisons", COMPARE, DESIGN) == []
    bad = {"comparisons": [{"result": "nope", "across": "pond", "values": [3], "how": "ratio"}]}
    problems = contract.check_part("comparisons", bad, DESIGN)
    # The values cannot be held to an axis that is not a setting, so the unknown axis is the one problem about them.
    assert len(problems) == 2
    assert any("`result`" in p for p in problems) and any("`across`" in p for p in problems)
    assert contract.check_part("comparisons", {"comparisons": []}, DESIGN) == []


@pytest.mark.asyncio
async def test_every_part_valid_first_try_makes_a_contract_form_plan_with_three_asks() -> None:
    fake = Fake(roles=[ROLES], names=[NAMES], comparisons=[COMPARE])
    out = await run(fake)
    assert out.form == "contract" and len(fake.prompts) == 3
    p = out.design["protocol"]
    assert p["form"] == "contract" and "parts_left_out" not in p
    c = contract.from_design(out.design)
    assert c.form == "contract"
    assert c.role_of("bloom") == "physical_bound" and c.result("algae_peak").depends_on == ("nutrient", "temp")
    assert c.result("algae_peak").returned == "each_run"
    assert [x.id for x in c.comparisons] == ["algae_peak_by_temp"]
    assert not any("could not be read" in g for g in c.gaps)


@pytest.mark.asyncio
async def test_what_each_run_returns_is_asked_and_kept_for_every_declared_result() -> None:
    answer = {"results": {"algae_peak": {"role": "result", "depends_on": [], "returned": "afterwards"},
                          "bloom": {"role": "result", "depends_on": [], "returned": "each_run"}}}
    out = await run(Fake(roles=[answer], names=[NAMES], comparisons=[COMPARE]))
    c = contract.from_design(out.design)
    assert (c.result("algae_peak").returned, c.result("bloom").returned) == ("afterwards", "each_run")
    first = out.parts["roles"]["answer"]["algae_peak"]
    assert first["returned"] == "afterwards"


@pytest.mark.asyncio
async def test_a_failed_part_is_asked_again_alone_with_the_reason_and_then_used() -> None:
    wrong = {"results": {"algae_peak": {"role": "result", "depends_on": ["salinity"], "returned": "each_run"},
                         "bloom": {"role": "result", "depends_on": [], "returned": "each_run"}}}
    fake = Fake(roles=[wrong, ROLES], names=[NAMES], comparisons=[COMPARE])
    out = await run(fake)
    assert out.form == "contract" and out.parts["roles"]["asks"] == 2 and out.parts["names"]["asks"] == 1
    again = [p for part, p in fake.prompts if part == "roles"][1]
    assert "salinity" in again and "Answer again" in again
    assert not any(part != "roles" and "salinity" in p for part, p in fake.prompts)


@pytest.mark.asyncio
async def test_a_part_that_fails_three_times_is_left_out_named_and_the_plan_stays_older() -> None:
    fake = Fake(roles=[ROLES], names=["not json", "still not", {"names": {}}], comparisons=[COMPARE])
    out = await run(fake)
    assert out.form == "older" and out.parts["names"]["asks"] == cp.MAX_ASKS and not out.parts["names"]["valid"]
    p = out.design["protocol"]
    assert "form" not in p and p["parts_left_out"] == [cp.PART_WORDS["names"]]
    assert p["metrics"][0]["role"] == "result"  # the valid parts are kept
    c = contract.from_design(out.design)
    assert c.form == "older" and any("did not give" in g and "names the plan uses" in g for g in c.gaps)
    assert out.notes and "older form" in out.notes[0]


@pytest.mark.asyncio
async def test_a_part_whose_call_raises_is_left_out_alone_and_the_valid_parts_are_kept() -> None:
    fake = Fake(roles=[ROLES], names=[RuntimeError("the model is down")], comparisons=[COMPARE])
    out = await run(fake)
    assert out.form == "older" and set(out.errors) == {"names"} and "the model is down" in out.errors["names"]
    assert out.parts["roles"]["valid"] and out.parts["comparisons"]["valid"] and not out.parts["names"]["valid"]
    assert out.design["protocol"]["metrics"][0]["role"] == "result" and out.design["protocol"]["comparisons"]


@pytest.mark.asyncio
async def test_a_model_written_form_is_never_believed() -> None:
    d = json.loads(json.dumps(DESIGN))
    d["protocol"]["form"] = "contract"
    fake = Fake(roles=[ROLES], names=["x", "y", "z"], comparisons=[COMPARE])
    out = await run(fake, d)
    assert "form" not in out.design["protocol"] and contract.from_design(out.design).form == "older"
    assert "form" not in cp.strip_fi_keys(d)["protocol"]


@pytest.mark.asyncio
async def test_a_structural_role_wins_over_the_models_word() -> None:
    d = json.loads(json.dumps(DESIGN))
    d["protocol"]["metrics"][0]["reference"] = {"formula": "nutrient * 2", "kind": "value"}
    out = await run(Fake(roles=[ROLES], names=[NAMES], comparisons=[COMPARE]), d)
    assert contract.from_design(out.design).role_of("algae_peak") == "reference"


@pytest.mark.asyncio
async def test_a_name_answered_derived_is_a_derived_quantity_and_can_be_compared() -> None:
    names = {"names": {"growth_theory": {"answer": "derived", "what": "a width", "inputs": ["algae_peak"]}}}
    cmp_ = {"comparisons": [{"result": "growth_theory", "across": "temp", "values": [], "how": "ratio"}]}
    out = await run(Fake(roles=[ROLES], names=[names], comparisons=[cmp_]))
    c = contract.from_design(out.design)
    assert out.form == "contract" and [d.id for d in c.derived] == ["growth_theory"] and c.comparisons[0].how == "ratio"


@pytest.mark.asyncio
async def test_a_reference_formula_that_the_plan_cannot_compute_is_asked_again_and_a_good_one_is_a_derived_reference() -> None:
    bad = {"names": {"growth_theory": {"answer": "reference", "what": "x", "formula": "salinity * 2"}}}
    good = {"names": {"growth_theory": {"answer": "reference", "what": "x", "formula": "nutrient * 2"}}}
    fake = Fake(roles=[ROLES], names=[bad, good], comparisons=[COMPARE])
    out = await run(fake)
    assert out.form == "contract" and out.parts["names"]["asks"] == 2
    d = contract.from_design(out.design).derived[0]
    assert (d.id, d.role, d.formula) == ("growth_theory", "reference", "nutrient * 2")


@pytest.mark.asyncio
async def test_a_name_called_a_result_or_a_reference_never_becomes_a_number_every_run_must_return() -> None:
    for answer in ({"answer": "result", "what": "w"}, {"answer": "reference", "what": "w", "equation": "E1"}):
        out = await run(Fake(roles=[ROLES], names=[{"names": {"growth_theory": answer}}], comparisons=[COMPARE]))
        p = out.design["protocol"]
        assert [m["id"] for m in p["metrics"]] == ["algae_peak", "bloom"]
        assert [m["id"] for m in metric_spec.declared(p)] == ["algae_peak", "bloom"]
        assert numeric_warnings.declared_results(p) == {"algae_peak", "bloom"}
        assert [n["id"] for n in p["named_results"]] == ["growth_theory"]
        c = contract.from_design(out.design)
        assert [r.id for r in c.results] == ["algae_peak", "bloom"] and c.notes["named_results"][0]["id"] == "growth_theory"
        assert out.form == "contract"


@pytest.mark.asyncio
async def test_the_plans_own_comparisons_are_kept_and_the_models_are_added() -> None:
    d = json.loads(json.dumps(DESIGN))
    d["protocol"]["comparisons"] = [{"id": "mine", "result": "algae_peak", "across": "temp", "how": "ratio"}]
    more = {"comparisons": [{"result": "algae_peak", "across": "temp", "values": [], "how": "difference"},
                            {"result": "bloom", "across": "nutrient", "values": [], "how": "difference"}]}
    out = await run(Fake(roles=[ROLES], names=[NAMES], comparisons=[more]), d)
    rows = out.design["protocol"]["comparisons"]
    assert rows[0] == {"id": "mine", "result": "algae_peak", "across": "temp", "how": "ratio"}
    assert [(r["result"], r["across"]) for r in rows] == [("algae_peak", "temp"), ("bloom", "nutrient")]


@pytest.mark.asyncio
async def test_names_past_the_cap_are_not_asked_and_are_said() -> None:
    d = json.loads(json.dumps(DESIGN))
    d["protocol"]["labels"] = {f"name_{i:02d}": f"Label {i}" for i in range(contract.MAX_NAMES + 3)}
    many = {"names": {n: {"answer": "diagnostic"} for n in contract.undeclared_names(d)}}
    out = await run(Fake(roles=[ROLES], names=[many], comparisons=[COMPARE]), d)
    p = out.design["protocol"]
    assert len(p["names_not_asked"]) == 4 and out.form == "contract"
    assert any("more name(s) than FI asked" in n for n in out.notes)
    assert any("not asked" in g for g in contract.from_design(out.design).gaps)


@pytest.mark.asyncio
async def test_no_metrics_asks_nothing_and_changes_nothing() -> None:
    d = {"hypothesis": "h", "protocol": {"grid": {"a": [1, 2]}}}
    out = await cp.ask_parts(d, Fake(), parse, TEMPLATES)
    assert out.design == d and out.form == "older" and out.parts == {}


def test_a_models_guess_of_what_a_result_may_change_with_decides_no_check() -> None:
    """The reviewer's input: ``depends_on = ["temp"]`` made ``temp`` a setting the study compares, and a code that never
    reads ``temp`` was then a fault to repair. The answer is a superset ("may change with"): no check reads it."""
    protocol = json.loads(json.dumps(DESIGN["protocol"]))
    axes = ["nutrient", "temp"]
    protocol["oracles"] = []
    before = run_checks.settings_the_study_compares(Path("no-such-fi-dir"), DESIGN, protocol, axes)
    protocol["metrics"][0].update({"role": "result", "depends_on": ["temp"], "returned": "each_run"})
    after = run_checks.settings_the_study_compares(Path("no-such-fi-dir"), DESIGN, protocol, axes)
    assert before == {} and after == {}


def test_an_asked_comparison_decides_no_check_but_the_plans_own_does() -> None:
    """The second review's input: a comparison the model gave in answer to FI's question made ``temp`` a setting the
    study compares, so a code that never reads ``temp`` became a fault to repair. The plan's own comparison still does."""
    protocol = json.loads(json.dumps(DESIGN["protocol"]))
    protocol["oracles"] = []
    axes = ["nutrient", "temp"]
    asked = cp._apply_comparisons({**protocol}, COMPARE["comparisons"])
    assert asked["comparisons"][0]["asked"] is True
    assert run_checks.settings_the_study_compares(Path("no-such-fi-dir"), DESIGN, asked, axes) == {}
    own = {**protocol, "comparisons": [{"id": "c", "result": "algae_peak", "across": "temp"}]}
    assert set(run_checks.settings_the_study_compares(Path("no-such-fi-dir"), DESIGN, own, axes)) == {"temp"}


def test_the_audits_verdict_still_covers_the_design_after_the_answers() -> None:
    """The second review's input: the answers changed the design after the audit's receipt was carried, so the design
    that runs no longer matched it and a second audit ran on every quest. What FI writes from the answers is not what
    the audit judged; a change to the study itself still breaks the match."""
    from core import receipts

    before = json.loads(json.dumps(DESIGN))
    p = cp._apply_roles(before["protocol"], ROLES["results"])
    p, _ = cp._apply_names(p, NAMES["names"])
    p = cp._apply_comparisons(p, COMPARE["comparisons"])
    after = {**before, "protocol": cp.settle_form(p, before["protocol"])}
    assert after != before
    assert receipts.design_core(after) == receipts.design_core(before)
    changed = {**after, "protocol": {**after["protocol"], "grid": {"nutrient": [1, 2], "temp": [10, 20]}}}
    assert receipts.design_core(changed) != receipts.design_core(before)


def test_a_models_own_asked_mark_is_dropped() -> None:
    d = json.loads(json.dumps(DESIGN))
    d["protocol"]["comparisons"] = [{"id": "c", "result": "algae_peak", "across": "temp", "asked": True}]
    assert "asked" not in cp.strip_fi_keys(d)["protocol"]["comparisons"][0]


def test_the_plan_shows_what_the_study_returns_in_plain_words() -> None:
    d = json.loads(json.dumps(DESIGN))
    d["protocol"]["metrics"][0].update({"role": "reference", "depends_on": ["nutrient"], "returned": "afterwards"})
    d["protocol"]["metrics"][1].update({"role": "physical_bound", "depends_on": []})
    d["protocol"]["comparisons"] = [{"id": "c", "result": "algae_peak", "across": "temp", "how": "ratio"}]
    d["protocol"]["derived"] = [{"id": "gt", "inputs": [], "role": "reference", "formula": "nutrient * 2"}]
    d["protocol"]["named_results"] = [{"id": "other", "answer": "reference", "equation": "E1"}]
    text = "\n".join(plan._returns_lines(d["protocol"]))
    assert "What the study returns" in text and "a theory or reference value; may change with `nutrient`; computed afterwards" in text
    assert "changes with none of the settings" in text and "compared: `algae_peak` between values of `temp` (ratio)" in text
    assert "FI computes from `nutrient * 2`" in text and "equation E1 of the model (not checked)" in text
    assert plan._returns_lines(DESIGN["protocol"]) == []


def test_the_returns_section_survives_the_plan_file_and_goes_when_the_roles_go() -> None:
    d = json.loads(json.dumps(DESIGN))
    d["protocol"].update({"form": "contract", "parts_left_out": []})
    d["protocol"]["metrics"][0].update({"role": "result", "depends_on": ["temp"], "returned": "each_run"})
    text = plan.render("Algae", {}, {**d, "plan": {}}, [])
    parsed = plan.parse(text).design
    assert parsed["protocol"]["form"] == "contract" and parsed["protocol"]["metrics"][0]["returned"] == "each_run"
    assert "## What the study returns" in text
    bare = plan.render("Algae", {}, {**DESIGN, "plan": {}}, [])
    assert "What the study returns" not in plan.refresh_model_section(bare)
    # a section whose roles are gone is taken out
    assert "What the study returns" not in plan.refresh_model_section(
        text.replace("role: result", "role_gone: result").replace('"role": "result"', '"role_gone": "result"'))


def test_a_contract_form_protocol_normalizes_and_freezes_the_same_way_as_an_older_one() -> None:
    from core import frozen_protocol as fp

    out = json.loads(json.dumps(DESIGN["protocol"]))
    fixed, why = plan.normalize_protocol({**out, "form": "contract", "comparisons": [], "derived": [], "named_results": []})
    assert why is None and fixed["form"] == "contract"
    assert fp.sha256(fixed) == fp.sha256(json.loads(fp.canonical(fixed)))
    older, _ = plan.normalize_protocol(out)
    assert "form" not in older and contract.from_design({"protocol": older}).form == "older"


def test_the_questions_are_filed_with_the_plan_stage_and_a_new_plan_forgets_the_old_answers() -> None:
    assert quality_report._NODE_STAGE["plan_contract"] == "plan"
    assert ".fi/contract_parts.json" in rerun_from._PLAN_LOOK


def test_a_part_left_out_is_a_disclosed_row_of_the_quality_report(tmp_path: Path) -> None:
    fi = tmp_path / ".fi"
    fi.mkdir()
    (fi / "contract_parts.json").write_text(json.dumps({"form": "older", "parts": {
        "roles": {"asks": 1, "valid": True, "problems": []},
        "names": {"asks": 1, "valid": False, "problems": ["x"], "error": "Traceback ..."}}}), encoding="utf-8")
    rows = quality_report._plan_problems(quality_report._Quest(tmp_path))
    mine = [r for r in rows if r["source"] == ".fi/contract_parts.json"]
    assert len(mine) == 1 and mine[0]["status"] == "disclosed" and "older form" in mine[0]["text"]


# --- the engine ----------------------------------------------------------------------------------------------------------


def _engine(tmp_path: Path) -> Any:
    from tests.test_criteria_engine import _config
    from core.engine import Engine

    return Engine(_config(tmp_path))


def _model(raise_in: str | None):  # noqa: ANN202
    from tests.test_criteria_engine import CASE_ORACLE, PROTOCOL, _Model, _design_reply
    from tests.test_engine_smoke import _fake_response_for

    protocol = {**PROTOCOL, "metrics": [{"id": "error_rate", "estimand": "e", "kind": "mean", "unit": "run"}],
                "oracles": [{**CASE_ORACLE, "measure": "error"}]}
    roles = {"results": {"error_rate": {"role": "result", "depends_on": ["dt"], "returned": "each_run"}}}

    class Model(_Model):
        def __init__(self) -> None:
            super().__init__(protocol, None)
            self.parts: list[str] = []

        async def chat(self, messages, **kw):  # noqa: ANN001, ANN003
            prompt = messages[-1]["content"]
            for part, marker, reply in (("roles", "What kind of number it is", roles),
                                        ("names", "Say what each one is", {"names": {"error": {"answer": "diagnostic"}}}),
                                        ("comparisons", "compares between values", {"comparisons": []})):
                if marker in prompt:
                    self.parts.append(part)
                    if part == raise_in:
                        raise RuntimeError(f"the {part} call failed")
                    return json.dumps(reply)
            return await super().chat(messages, **kw)

    return Model()


@pytest.mark.asyncio
async def test_inside_the_plan_step_one_part_raising_leaves_that_part_out_alone_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    from tests.test_criteria_engine import _Search

    monkeypatch.setattr("core.knowledge.Knowledge.asearch", _Search().fn)
    engine = _engine(tmp_path)
    model = _model("names")
    engine._client = model
    await engine._node_plan({"topic": engine.config.topic, "literature": []})
    assert model.parts[:2] == ["roles", "names"] and "comparisons" in model.parts
    text = plan.plan_path(engine.quest_root).read_text(encoding="utf-8")
    protocol = plan.parse(text).design["protocol"]
    assert protocol["metrics"][0]["role"] == "result" and "form" not in protocol  # roles kept, the plan stays older
    assert protocol["parts_left_out"] == [cp.PART_WORDS["names"]]
    assert "FI could not ask the plan about what the names the plan uses but does not declare are" in text
    assert "the study goes on with the plan as written" in capsys.readouterr().out
    log = (engine.fi_dir / "run.log").read_text(encoding="utf-8")
    assert "Traceback" in log and "the names call failed" in log
    record = json.loads((engine.fi_dir / "contract_parts.json").read_text(encoding="utf-8"))
    assert record["parts"]["roles"]["valid"] and not record["parts"]["names"]["valid"]
    assert "## What the study returns" in text


@pytest.mark.asyncio
async def test_inside_the_plan_step_every_part_valid_writes_the_contract_form_after_the_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core import receipts as _receipts
    from tests.test_criteria_engine import _Search

    monkeypatch.setattr("core.knowledge.Knowledge.asearch", _Search().fn)
    carried: list[Any] = []
    real = _receipts.carry_output

    def spy(root, kind, before, after, why):  # noqa: ANN001, ANN202
        carried.append(json.dumps(after, default=str))
        return real(root, kind, before, after, why)

    monkeypatch.setattr("core.engine._receipts.carry_output", spy)
    engine = _engine(tmp_path)
    engine._client = _model(None)
    await engine._node_plan({"topic": engine.config.topic, "literature": []})
    protocol = plan.parse(plan.plan_path(engine.quest_root).read_text(encoding="utf-8")).design["protocol"]
    assert protocol["form"] == "contract" and protocol["metrics"][0]["returned"] == "each_run"
    assert carried and '"form"' not in carried[0] and '"depends_on"' not in carried[0], "the audit's receipt covers what it saw"


def _redesign_engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, planned: dict[str, Any]) -> Any:
    engine = _engine(tmp_path)
    monkeypatch.setattr("core.plan.load_design", lambda root: ({"hypothesis": "h", "protocol": planned}, None))
    return engine


def test_a_redesign_before_the_freeze_works_out_the_form_again_from_its_own_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    planned = json.loads(json.dumps(DESIGN["protocol"]))
    planned.update({"form": "contract", "comparisons": [{"id": "c", "result": "algae_peak", "across": "temp"}]})
    for m, r in zip(planned["metrics"], ROLES["results"].values()):
        m.update(r)
    engine = _redesign_engine(tmp_path, monkeypatch, planned)
    plain = {"grid": {"nutrient": [1, 2]}, "metrics": [{"id": "algae_peak", "kind": "mean", "estimand": "e", "unit": "u"}]}
    design = {"hypothesis": "h", "protocol": plain}
    kept = engine._inherit_plan_parts({"design": design, "iteration": 1}, design, iteration=1)["protocol"]
    assert "form" not in kept and kept["parts_left_out"] == [cp.PART_WORDS["roles"]]
    assert contract.from_design({"protocol": kept}).form == "older"
    assert any("did not give" in g for g in contract.from_design({"protocol": kept}).gaps)
    # a redesign whose results all carry a validated role is a contract plan, and nothing was asked
    done = {**plain, "metrics": [{**plain["metrics"][0], **ROLES["results"]["algae_peak"]}]}
    again = engine._inherit_plan_parts({"design": {"protocol": done}, "iteration": 1}, {"hypothesis": "h", "protocol": done},
                                       iteration=1)["protocol"]
    assert again["form"] == "contract" and "parts_left_out" not in again
    # a part the plan itself left out stays left out
    planned["parts_left_out"] = [cp.PART_WORDS["names"]]
    kept2 = engine._inherit_plan_parts({"design": {"protocol": done}, "iteration": 1}, {"hypothesis": "h", "protocol": done},
                                       iteration=1)["protocol"]
    assert "form" not in kept2
