"""A reviewer finds problems (each with where, what, why, how to fix, how serious) and lists what it checked, so silence is
never approval (core/review_pairs.py); the plan's, the design's and the code's reviews read that way. Fake models, neutral
topics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from core import review_pairs as pairs
from core.engine import Engine
from tests.test_write_review_pairs import DESIGN, _Seen, _critique_entry, _engine


def test_a_finding_says_where_what_why_how_to_fix_and_how_serious() -> None:
    got = pairs.findings([
        {"where": "simulate.py:42", "what": "the sum is not a mean", "why": "the plan defines a mean", "fix": "divide by n",
         "severity": "Critical"},
        {"problem": "no unit conversion", "location": "method", "reason": "the plan wants SI", "suggestion": "convert",
         "severity": "minor"},
        {"what": "looks fine"}, {"what": ""}, "a bare sentence", 7, {"where": "x"}])
    assert [(f["where"], f["severity"]) for f in got] == [("simulate.py:42", "high"), ("method", "low"), ("", "medium")]
    assert got[0]["fix"] == "divide by n" and got[1]["what"] == "no unit conversion" and got[1]["fix"] == "convert"
    assert pairs.sentence(got[0]).rstrip(".") == "simulate.py:42: the sum is not a mean (the plan defines a mean). Fix: divide by n"
    assert pairs.checked(["E1: normalised by the area", {"item": "E2", "note": "same factors"}, "",
                          "E1: normalised by the area"]) == ["E1: normalised by the area", "E2: same factors"]
    assert pairs.said_nothing([], []) and not pairs.said_nothing(got, []) and not pairs.said_nothing([], ["E1"])
    assert not pairs.said_nothing([], [], evidence=True) and not pairs.said_nothing([], [], other_problem=True)


def test_only_the_plans_own_values_are_taken_out_of_what_a_reviewer_says() -> None:
    values = [0.25, 3.0e-6, 42.0]
    f = pairs.without_values(
        {"where": "simulate.py:9", "what": "E2 divides by the wrong length and uses 1 sigma instead of 3 sigma",
         "why": "the expected value 0.25 (that is 25%, or 2.5e-1) is not reached; see lines 12-14 of R2",
         "fix": "divide by the cell area, a factor of 2 on line 9; the tolerance 3e-6 is fine"}, values)
    assert f["what"] == "E2 divides by the wrong length and uses 1 sigma instead of 3 sigma"
    assert f["why"] == f"the expected value {pairs.PLACEHOLDER} (that is {pairs.PLACEHOLDER}, or {pairs.PLACEHOLDER}) is not reached; see lines 12-14 of R2"
    assert f["fix"] == f"divide by the cell area, a factor of 2 on line 9; the tolerance {pairs.PLACEHOLDER} is fine"
    assert pairs.scrub("a result of 0.2500001 and of 4.2e1", values) == f"a result of {pairs.PLACEHOLDER} and of {pairs.PLACEHOLDER}"
    assert pairs.scrub("E42 on line 42 stays", values) == "E42 on line 42 stays"
    assert pairs.scrub("anything 0.25", None) == "anything 0.25"


def test_the_small_run_note_is_scrubbed_at_the_precision_it_was_printed() -> None:
    # r6t: the small run's note printed k_true_2=0.1036 while the plan's check expects 0.1035560610989359
    values = [0.1035560610989359, 0.0125]
    note = "Setting noise=0.01: k_true_2=0.1036, k_true_3=0.10356, k_hat=0.1041, a share 1.25%, n=12, T=350"
    assert "k_true_2=0.1036," in pairs.scrub(note, values)  # the plain scrub leaves a rounded value
    got = pairs.scrub(note, values, printed=True)
    assert got == (f"Setting noise=0.01: k_true_2={pairs.PLACEHOLDER}, k_true_3={pairs.PLACEHOLDER}, k_hat=0.1041, "
                   f"a share {pairs.PLACEHOLDER}, n=12, T=350")
    # two significant digits are too few to tell a plan value from any other number
    assert pairs.scrub("about 0.10", values, printed=True) == "about 0.10"


def test_the_plans_values_are_its_checks_expected_values_and_tolerances_and_its_examples() -> None:
    from tests.test_oracle_review import PROTOCOL

    protocol = {**PROTOCOL, "oracles": [{**PROTOCOL["oracles"][0], "expected": 0.123456789, "tolerance": 3e-6,
                                         "expected_formula": "7.5 * 2"}]}
    values = pairs.plan_values(protocol)
    assert 0.123456789 in values and 3e-6 in values and 15.0 in values       # a check and its formula's result
    assert 2.0 in values and 4.0 in values or 2.0 in values                   # an equation example's input
    assert 0 not in values and 1 not in values and pairs.plan_values(None) == []


# --- the plan's checks -------------------------------------------------------------------------------------------------


def _plan_reply(**over: Any) -> dict[str, Any]:
    from tests.test_oracle_review import REVIEW

    base = {**REVIEW, "equations_not_tested": [], "add": []}
    base.update(over)
    return base


def test_the_plan_reviews_structured_findings_and_checked_items_are_read_and_said() -> None:
    from core import oracle_review as orv
    from tests.test_oracle_review import PROTOCOL

    reply = _plan_reply(findings=[{"where": "power_conservation", "what": "the case mixes two regimes",
                                   "why": "the stated condition does not hold everywhere", "fix": "run it in one regime",
                                   "severity": "high"}], checked=["E1: the standard definition"])
    review = orv.parse(reply, PROTOCOL)
    assert review.found[0]["severity"] == "high" and review.checked == ["E1: the standard definition"]
    assert not review.said_nothing
    assert any("the case mixes two regimes" in f and "Fix: run it in one regime" in f for f in orv.findings(review))
    text = "\n".join(orv.plan_lines(review, reviewer="b", planner="a", same_model=False, research=True))
    assert "Found (high): power_conservation: the case mixes two regimes" in text
    assert "Checked, and no problem found: E1: the standard definition" in text


def test_a_plan_reply_that_agrees_and_says_nothing_it_checked_is_no_review() -> None:
    from core import oracle_review as orv
    from tests.test_oracle_review import PROTOCOL

    bare = {"checks": [{"name": "power_conservation", "appropriate": "yes", "discriminating": "yes", "well_defined": "yes"}]}
    assert orv.parse(bare, PROTOCOL).said_nothing
    assert not orv.parse({**bare, "checked": ["power_conservation: traced against E1"]}, PROTOCOL).said_nothing
    assert not orv.parse({"checks": [{**bare["checks"][0], "why": "it tests E1"}]}, PROTOCOL).said_nothing
    assert not orv.parse({"checks": [{**bare["checks"][0], "appropriate": "no"}]}, PROTOCOL).said_nothing


class _Sequence:
    """A stand-in model: the plan's reviewer answers ``replies`` in turn; every other step is answered as in the plan tests."""

    def __init__(self, replies: list[Any]) -> None:
        from tests.test_oracle_review import PROTOCOL, _Model

        self.inner, self.replies, self.reviews, self.last_review = _Model(PROTOCOL, None), replies, 0, ""

    async def chat(self, messages, **kw):  # noqa: ANN001, ANN201
        if messages[-1]["content"].lstrip().startswith("# Second Opinion on the Checks"):
            self.reviews += 1
            self.last_review = messages[-1]["content"]
            reply = self.replies[min(self.reviews, len(self.replies)) - 1]
            return reply if isinstance(reply, str) else json.dumps(reply)
        return await self.inner.chat(messages, **kw)

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_an_agree_only_plan_review_is_asked_again_and_then_recorded_as_not_reviewed(tmp_path: Path) -> None:
    from core import oracle_review as orv
    from tests.test_oracle_review import _config, _section

    bare = {"checks": [{"name": "power_conservation", "appropriate": "yes", "discriminating": "yes", "well_defined": "yes"}]}
    engine = Engine(_config(tmp_path / "a", node_models={"oracle_review": "reader-model"}))
    model = _Sequence([bare, bare])
    engine._client = model
    await engine._node_plan({"topic": engine.config.topic, "literature": []})
    assert model.reviews == 2 and "Your previous reply was no review" in model.last_review   # once more, never a third time
    record = json.loads((engine.fi_dir / "oracle_review.json").read_text(encoding="utf-8"))
    assert record["error"] == pairs.NOT_A_REVIEW and "verdicts" not in record
    assert "could not be used (its reply named no problem and did not say what it checked" in _section(engine)
    assert orv.independence_gaps(record, [], configured=True)[0].startswith(orv.NOT_REVIEWED)

    again = Engine(_config(tmp_path / "b", node_models={"oracle_review": "reader-model"}))
    model = _Sequence([bare, {**bare, "checked": ["power_conservation: traced against E1"]}])
    again._client = model
    await again._node_plan({"topic": again.config.topic, "literature": []})
    assert model.reviews == 2
    assert "Checked, and no problem found: power_conservation: traced against E1" in _section(again)
    assert "error" not in json.loads((again.fi_dir / "oracle_review.json").read_text(encoding="utf-8"))


@pytest.mark.asyncio
async def test_a_plan_finding_goes_back_to_the_plan_in_the_one_request(tmp_path: Path) -> None:
    from tests.test_oracle_review import PROTOCOL, _config, _Model

    engine = Engine(_config(tmp_path))
    model = _Model(PROTOCOL, _plan_reply(findings=[{"where": "power_conservation", "what": "the tolerance is unreachable",
                                                     "why": "a correct method cannot get there",
                                                     "fix": "loosen nothing; compare at the step", "severity": "medium"}]))
    engine._client = model
    await engine._node_plan({"topic": engine.config.topic, "literature": []})
    assert len(model.revisions) == 1
    assert "the tolerance is unreachable" in model.revisions[0] and "Fix: loosen nothing" in model.revisions[0]


def test_the_prompts_of_the_three_reviews_ask_for_problems_and_for_what_was_checked() -> None:
    root = Path(__file__).resolve().parents[1] / "agents"
    for name, keys in (("oracle_review", ("findings", "checked")),
                       ("design_self_critique", ("objections_addressed", "checked_no_problem")),
                       ("code_review", ("findings", "checked"))):
        text = (root / f"{name}.md").read_text(encoding="utf-8")
        assert "FIND PROBLEMS" in text and all(f'"{k}"' in text for k in keys), name
        assert "is no review" in text, name


# --- the design: a reviewer that is another model must find problems or say what it checked ------------------------------


@pytest.mark.asyncio
async def test_an_agree_only_design_review_by_another_model_is_asked_again_then_not_reviewed(tmp_path: Path) -> None:
    engine = _engine(tmp_path, node_models={"design_review": "reviewer-model"})
    client = _Seen([{"objections_addressed": []}, {"objections_addressed": []}])
    engine._client = client
    design, objections = await engine._audit_design({"topic": "t", "iteration": 0}, dict(DESIGN))
    assert len(client.prompts) == 2 and "listed no objection and did not say what it checked" in client.prompts[1]
    assert design["method"] == DESIGN["method"] and objections is None
    entry = _critique_entry(engine)
    assert entry["status"] == "failed" and pairs.NOT_A_REVIEW in entry["failure"] and entry["review"]["kind"] == "not_reviewed"
    receipt = json.loads((engine.quest_root / "needs" / "receipts" / "design_audit.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "unknown"
    # Asked again and it said what it checked: a review (the design is read, nothing found).
    second = _engine(tmp_path / "b", node_models={"design_review": "reviewer-model"})
    second._client = _Seen([{"objections_addressed": []},
                            {"objections_addressed": [], "checked_no_problem": [{"check": "oracle", "note": "present"}]}])
    await second._audit_design({"topic": "t", "iteration": 0}, dict(DESIGN))
    assert _critique_entry(second)["status"] == "ok" and _critique_entry(second)["review"]["kind"] == "independent_review"


@pytest.mark.asyncio
async def test_the_main_models_own_empty_audit_is_accepted_as_it_always_was(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    client = _Seen({"objections_addressed": []})
    engine._client = client
    await engine._audit_design({"topic": "t", "iteration": 0}, dict(DESIGN))
    assert len(client.prompts) == 1 and _critique_entry(engine)["status"] == "ok"
    assert _critique_entry(engine)["review"]["kind"] == "self_review"
