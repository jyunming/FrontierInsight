"""The plan's model is asked the contract's parts one at a time (core/contract_parts.py); the fake model here is a function,
never a real one. Neutral made-up study: a pond's algae count against nutrient level and temperature."""

from __future__ import annotations

import json
import string
from pathlib import Path
from typing import Any

import pytest

from core import contract, contract_parts as cp, plan

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

ROLES = {"results": {"algae_peak": {"role": "result", "depends_on": ["nutrient", "temp"]},
                     "bloom": {"role": "physical_bound", "depends_on": ["nutrient"]}}}
NAMES = {"names": {"growth_theory": {"answer": "reference", "what": "the closed-form peak", "equation": "E1",
                                     "depends_on": ["nutrient"]}}}
COMPARE = {"comparisons": [{"result": "algae_peak", "across": "temp", "values": [10, 20], "how": "difference"}]}


class Fake:
    """Answers each part from a queue of replies (a dict is sent as JSON, a string as it is); records the prompts."""

    def __init__(self, **replies: list[Any]) -> None:
        self.replies, self.prompts = replies, []

    async def __call__(self, prompt: str) -> str:
        part = next(k for k, marker in (("roles", "What kind of number it is"), ("names", "Say what each one is"),
                                        ("comparisons", "compares between values")) if marker in prompt)
        self.prompts.append((part, prompt))
        reply = self.replies[part].pop(0)
        return reply if isinstance(reply, str) else json.dumps(reply)


def parse(text: str) -> Any:
    try:
        return json.loads(text)
    except ValueError:
        return None


async def run(fake: Fake) -> cp.Outcome:
    return await cp.ask_parts(DESIGN, fake, parse, TEMPLATES)


def test_check_part_roles_hold_the_answer_to_declared_ids_and_varied_settings() -> None:
    assert contract.check_part("roles", ROLES["results"], DESIGN) == []
    bad = {"algae_peak": {"role": "result", "depends_on": ["salinity"]}, "extra": {"role": "result", "depends_on": []}}
    problems = contract.check_part("roles", bad, DESIGN)
    assert any("salinity" in p for p in problems) and any("`bloom` is not answered" in p for p in problems)
    assert any("`extra` is not a declared result" in p for p in problems)
    assert any("one of" in p for p in contract.check_part("roles", {**ROLES["results"], "bloom": {"role": "great", "depends_on": []}}, DESIGN))


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
    assert len(problems) == 2 + 0 or len(problems) >= 2
    assert contract.check_part("comparisons", {"comparisons": []}, DESIGN) == []


@pytest.mark.asyncio
async def test_every_part_valid_first_try_makes_a_contract_form_plan_with_three_asks() -> None:
    fake = Fake(roles=[ROLES], names=[NAMES], comparisons=[COMPARE])
    out = await run(fake)
    assert out.form == "contract" and [len(v) for v in [fake.prompts]] == [3]
    p = out.design["protocol"]
    assert p["form"] == "contract" and "parts_left_out" not in p
    c = contract.from_design(out.design)
    assert c.form == "contract"
    assert c.role_of("bloom") == "physical_bound" and c.result("algae_peak").depends_on == ("nutrient", "temp")
    assert c.role_of("growth_theory") == "reference"
    assert [x.id for x in c.comparisons] == ["algae_peak_by_temp"]
    assert not any("could not be read" in g for g in c.gaps)


@pytest.mark.asyncio
async def test_a_failed_part_is_asked_again_alone_with_the_reason_and_then_used() -> None:
    wrong = {"results": {"algae_peak": {"role": "result", "depends_on": ["salinity"]},
                         "bloom": {"role": "result", "depends_on": []}}}
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
async def test_a_structural_role_wins_over_the_models_word() -> None:
    d = json.loads(json.dumps(DESIGN))
    d["protocol"]["metrics"][0]["reference"] = {"formula": "nutrient * 2", "kind": "value"}
    asked = {"results": {"algae_peak": {"role": "result", "depends_on": ["nutrient"]},
                         "bloom": {"role": "result", "depends_on": []}}}
    fake = Fake(roles=[asked], names=[NAMES], comparisons=[COMPARE])
    out = await cp.ask_parts(d, fake, parse, TEMPLATES)
    assert contract.from_design(out.design).role_of("algae_peak") == "reference"


@pytest.mark.asyncio
async def test_a_name_answered_derived_is_a_derived_quantity_and_can_be_compared() -> None:
    names = {"names": {"growth_theory": {"answer": "derived", "what": "a width", "inputs": ["algae_peak"]}}}
    cmp_ = {"comparisons": [{"result": "growth_theory", "across": "temp", "values": [], "how": "ratio"}]}
    out = await run(Fake(roles=[ROLES], names=[names], comparisons=[cmp_]))
    c = contract.from_design(out.design)
    assert out.form == "contract" and [d.id for d in c.derived] == ["growth_theory"] and c.comparisons[0].how == "ratio"


@pytest.mark.asyncio
async def test_a_reference_formula_that_the_plan_cannot_compute_is_asked_again() -> None:
    names = {"names": {"growth_theory": {"answer": "reference", "what": "x", "formula": "salinity * 2"}}}
    fake = Fake(roles=[ROLES], names=[names, NAMES], comparisons=[COMPARE])
    out = await run(fake)
    assert out.form == "contract" and out.parts["names"]["asks"] == 2


@pytest.mark.asyncio
async def test_no_metrics_asks_nothing_and_changes_nothing() -> None:
    d = {"hypothesis": "h", "protocol": {"grid": {"a": [1, 2]}}}
    out = await cp.ask_parts(d, Fake(), parse, TEMPLATES)
    assert out.design == d and out.form == "older" and out.parts == {}


def test_the_plan_shows_what_the_study_returns_in_plain_words() -> None:
    d = json.loads(json.dumps(DESIGN))
    d["protocol"]["metrics"][0].update({"role": "reference", "depends_on": ["nutrient"]})
    d["protocol"]["metrics"][1].update({"role": "physical_bound", "depends_on": []})
    d["protocol"]["comparisons"] = [{"id": "c", "result": "algae_peak", "across": "temp", "how": "ratio"}]
    text = "\n".join(plan._returns_lines(d["protocol"]))
    assert "What the study returns" in text and "a theory or reference value; changes with `nutrient`" in text
    assert "changes with none of the settings" in text and "compared: `algae_peak` between values of `temp` (ratio)" in text
    assert plan._returns_lines(DESIGN["protocol"]) == []


def test_a_contract_form_protocol_normalizes_and_freezes_the_same_way_as_an_older_one() -> None:
    from core import frozen_protocol as fp

    out = json.loads(json.dumps(DESIGN["protocol"]))
    fixed, why = plan.normalize_protocol({**out, "form": "contract", "comparisons": [], "derived": []})
    assert why is None and fixed["form"] == "contract"
    assert fp.sha256(fixed) == fp.sha256(json.loads(fp.canonical(fixed)))
    older, _ = plan.normalize_protocol(out)
    assert "form" not in older and contract.from_design({"protocol": older}).form == "older"


@pytest.mark.asyncio
async def test_the_engine_asks_the_parts_keeps_the_record_and_never_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, PausesConfig, ProviderConfig
    from core.engine import Engine

    cfg = Config(topic="Algae in a pond", title="algae", provider=ProviderConfig(name="openai"),
                 engine=EngineConfig(max_iterations=1), execution=ExecutionConfig(sandbox="venv", timeout_s=60),
                 knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "outputs"),
                 pauses=PausesConfig(plan="off", papers=False))
    engine = Engine(cfg)
    fake = Fake(roles=[ROLES], names=[NAMES], comparisons=[COMPARE])
    nodes: list[str] = []

    async def chat(self, prompt, *, node=None, temperature=None):  # noqa: ANN001
        nodes.append(node)
        return await fake(prompt)

    monkeypatch.setattr(Engine, "_chat", chat)
    design, notes = await engine._ask_contract_parts(DESIGN)
    assert design["protocol"]["form"] == "contract" and notes == [] and set(nodes) == {"plan_contract"}
    record = json.loads((engine.fi_dir / "contract_parts.json").read_text(encoding="utf-8"))
    assert record["form"] == "contract" and record["parts"]["roles"]["asks"] == 1

    async def broken(self, prompt, *, node=None, temperature=None):  # noqa: ANN001
        raise RuntimeError("the model is down")

    monkeypatch.setattr(Engine, "_chat", broken)
    same, notes = await engine._ask_contract_parts(DESIGN)
    assert same is DESIGN and notes == []
