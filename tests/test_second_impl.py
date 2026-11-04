"""core/second_impl.py: key function choice, inputs, comparison (pure), and the whole step with a fake model and a real
interpreter (the comparison really imports both versions in a subprocess)."""

from __future__ import annotations

import asyncio
import json
import math
import os
import string
import subprocess
import sys
from pathlib import Path

import pytest

from core import function_steps as fs
from core import plan_steps
from core import second_impl as si

ROOT = Path(__file__).resolve().parent.parent
PROMPTS = {"write": string.Template((ROOT / "agents" / "second_impl_write.md").read_text(encoding="utf-8")),
           "adjudicate": string.Template((ROOT / "agents" / "second_impl_adjudicate.md").read_text(encoding="utf-8"))}

GOOD_MAIN = '''import numpy as np


# S1
def activation_energy(slope):
    """Activation energy in J/mol from the slope of ln k against 1/T."""
    return -slope * 8.314
'''
MUTANT_MAIN = GOOD_MAIN.replace("-slope * 8.314", "slope * 8.314")  # sign error
UNITS_MAIN = GOOD_MAIN.replace("-slope * 8.314", "-slope * 8.314 / 1000.0")  # J/mol vs kJ/mol
GOOD_REF = "def activation_energy(slope):\n    R = 8.314\n    return -R * slope\n"
BAD_REF = "def activation_energy(slope):\n    R = 8.314\n    return R * slope\n"
SIG = "def activation_energy(slope):"


def protocol(example: bool = True) -> dict:
    step = {"id": "S1", "name": "activation energy", "does": "Ea = -R times the slope of ln k against 1/T",
            "inputs": {"slope": "K, slope of ln k against 1/T"}, "output": "Ea in J/mol", "uses": [],
            "source": "Arrhenius 1889, textbook"}
    if example:
        step["example"] = {"inputs": {"slope": -5000.0}, "expected_formula": "-slope * 8.314"}
    return {"model": {"summary": "kinetics", "equations": [
        {"id": "E1", "formula": "k = A * exp(-Ea / (R * T))", "role": "analyses"}], "steps": [step]}}


def spec_for(proto: dict, name: str = "activation_energy", signature: str = SIG) -> fs.Spec:
    card = plan_steps.parse(proto)[0][0]
    return fs.Spec(name=name, signature=signature, purpose="activation energy", equation="S1", kind=fs.STEP, card=card)


class Fake:
    """A fake model: replies per node, in order; every prompt is kept."""

    def __init__(self, write=(), adjudicate=()):
        self.replies = {"second_impl": list(write), "second_impl.adjudicate": list(adjudicate)}
        self.prompts: list[tuple[str, str]] = []

    async def chat(self, node, messages):
        self.prompts.append((node, messages[0]["content"]))
        queue = self.replies[node]
        if not queue:
            raise AssertionError(f"unexpected request to {node}")
        return queue.pop(0)

    def count(self, node=None):
        return sum(1 for n, _ in self.prompts if node in (None, n))


def fenced(code: str) -> str:
    return "```python\n" + code + "\n```"


def verdict(v: str, reason: str = "r") -> str:
    return json.dumps({"verdict": v, "reason": reason})


def make_quest(tmp_path: Path, main: str = GOOD_MAIN) -> Path:
    root = tmp_path / "quest"
    pkg = root / "code" / "pkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "model.py").write_text(main, encoding="utf-8")
    return root


def make_run(root: Path):
    async def run(argv):
        def go():
            p = subprocess.run([sys.executable, "-B", *argv], cwd=root, capture_output=True, text=True, timeout=120)
            return p.returncode, p.stdout, p.stderr
        return await asyncio.to_thread(go)
    return run


def build(root: Path, fake: Fake, specs, proto, *, repair_main=None, still_passes=None, **kw) -> si.SecondImpl:
    async def no_repair(name, problem, cases):
        raise AssertionError("repair_main must not be called")

    async def passes(name):
        return True

    return si.SecondImpl(quest_root=root, package="pkg", specs=specs, protocol=proto, prompts=PROMPTS, chat=fake.chat,
                         run=make_run(root), repair_main=repair_main or no_repair, still_passes=still_passes or passes, **kw)


def run_it(tmp_path, fake, *, main=GOOD_MAIN, proto=None, **kw):
    proto = proto or protocol()
    root = make_quest(tmp_path, main)
    s = build(root, fake, [spec_for(proto)], proto, **kw)
    rec = asyncio.run(s.run_all())
    return root, s, rec


# --- pure helpers -----------------------------------------------------------------------------------------------------


def test_build_inputs_scalars_arrays_and_edges():
    cases = si.build_inputs({"x": 2.0, "n": 3, "label": "a", "v": [1.0, 2.0]}, {"x": [0.1, 10]})
    assert 4 <= len(cases) <= si.MAX_INPUTS
    assert cases[0] == {"x": 2.0, "n": 3, "label": "a", "v": [1.0, 2.0]}
    assert {"x": 1.0, "n": 3, "label": "a", "v": [1.0, 2.0]} in cases  # x0.5, an int is not halved
    assert any(c["x"] == 4.0 and c["n"] == 6 for c in cases)
    assert any(c["x"] == 0.1 for c in cases) and any(c["x"] == 10 for c in cases)
    assert any(c["v"] == [2.0, 4.0] for c in cases)
    assert si.build_inputs({"x": 2.0}, {"x": [1, 5]}) == si.build_inputs({"x": 2.0}, {"x": [1, 5]})  # deterministic
    assert len({json.dumps(c, sort_keys=True) for c in cases}) == len(cases)


def test_build_inputs_zero_and_strings_only():
    assert len(si.build_inputs({"x": 0.0})) >= 4
    assert si.build_inputs({"mode": "fast"}) == [{"mode": "fast"}]


def test_same_numbers_arrays_nan_inf_strings():
    assert si.same(1.0, 1.0 + 1e-9) and not si.same(1.0, 1.0 + 1e-5)
    assert si.same(float("nan"), float("nan")) and not si.same(float("nan"), 1.0)
    assert si.same(float("inf"), float("inf")) and not si.same(float("inf"), float("-inf"))
    assert si.same(0.0, 1e-15) and not si.same(0.0, 1e-3)
    assert si.same([1.0, [2.0, 3.0]], [1.0, [2.0, 3.0]]) and not si.same([1.0, 2.0], [1.0, 2.0, 3.0])
    assert si.same(3, 3) and not si.same(3, 4) and si.same("a", "a") and not si.same("a", "b")
    assert not si.same(True, 1) and si.same(True, True)
    assert si.same({"a": 1.0}, {"a": 1.0 + 1e-9}) and not si.same({"a": 1.0}, {"b": 1.0})
    assert si.same(1 + 2j, 1 + 2j) and not si.same(1 + 2j, 1 + 3j)
    assert si.same(si.decode({"f": "nan"}), float("nan")) and si.decode({"f": "inf"}) == math.inf


def test_judge_rows_exceptions_and_excluded():
    inputs = [{"x": 1.0}, {"x": 2.0}, {"x": 3.0}, {"x": 4.0}]
    rows = [{"main": 1.0, "ref": 1.0}, {"main_err": "ValueError: x", "ref_err": "ZeroDivisionError: y"},
            {"main": 3.0, "ref_err": "ValueError: z"}, {"main": [1.0, 2.0], "ref": [1.0, 2.5]}]
    v = si.judge_rows(rows, inputs, si.REL)
    assert v["status"] == "disagree" and v["compared"] == 3 and v["excluded"] == 1
    assert [d["kind"] for d in v["disagreements"]] == ["domain", "value"]
    assert v["disagreements"][0]["inputs"] == {"x": 3.0}
    assert si.judge_rows(rows[:2], inputs[:2], si.REL)["status"] == "agree"
    assert si.judge_rows([rows[1]], inputs[1:2], si.REL)["status"] == "unconfirmed"


def test_a_different_return_shape_with_equal_shared_values_is_not_a_disagreement():
    inputs = [{"x": 1.0}, {"x": 2.0}]
    main = {"Ea": 49.6, "lnA": 15.3, "se_Ea": 0.69, "se_lnA": 0.26}
    ref = {**main, "covariance": [[0.48, 0.18], [0.18, 0.07]]}
    v = si.judge_rows([{"main": main, "ref": ref}, {"main": main, "ref": ref}], inputs, si.REL)
    assert v["status"] == "agree" and not v["disagreements"] and v["compared"] == 2
    assert "only the second version returns covariance" in v["shape_notes"][0]
    assert "those were not compared" in v["shape_notes"][0] and "description" not in v["shape_notes"][0]


def test_a_returned_key_the_second_version_lacks_is_unconfirmed_never_agree():
    main, ref = {"answer": 999, "count": 1}, {"count": 1}
    v = si.judge_rows([{"main": main, "ref": ref}], [{"x": 1.0}], si.REL)
    assert v["status"] == "unconfirmed" and "answer" in v["why"] and not v["disagreements"]
    # the s8 shape: the second version lacks one of the first version's keys
    main = {"Ea": 49.6, "lnA": 15.3, "se_Ea": 0.69, "se_lnA": 0.26}
    ref = {"Ea": 49.6, "lnA": 15.3, "covariance": [[1.0]], "se_Ea": 0.69}
    assert si.judge_rows([{"main": main, "ref": ref}], [{"x": 1.0}], si.REL)["status"] == "unconfirmed"
    assert si.judge_rows([{"main": main, "ref": {**main, "answer": 1}}], [{"x": 1.0}], si.REL)["status"] == "agree"


def test_nothing_shared_is_not_compared_and_never_a_disagreement():
    v = si.judge_rows([{"main": {"a": 1.0}, "ref": {"b": 2.0}}], [{"x": 1.0}], si.REL)
    assert v["status"] == "unconfirmed" and "was not compared" in v["why"] and not v["disagreements"]


def test_a_transformed_first_element_of_a_list_is_still_a_disagreement():
    v = si.judge_rows([{"main": [-4.609, 0.000844], "ref": [0.00996, 0.000844]}], [{"x": 1.0}], si.REL)
    assert v["status"] == "disagree"
    assert si.compare_shared([1.0, 2.0], [1.0, 2.0, 3.0])[1]  # lists of another length still differ


def test_a_step_whose_versions_return_different_shapes_with_equal_values_is_resolved_agree(tmp_path):
    main = GOOD_MAIN.replace("return -slope * 8.314", 'return {"Ea": -slope * 8.314}')
    ref = ("def activation_energy(slope):\n"
           "    return {'Ea': -8.314 * slope, 'covariance': [[1.0]]}\n")
    fake = Fake(write=[fenced(ref)])
    root, s, rec = run_it(tmp_path, fake, main=main)
    e = rec["functions"]["activation_energy"]
    assert e["status"] == "agree" and rec["calls"] == 1
    assert "only the second version returns covariance" in e["shape_note"] and "limitation" not in e


def test_key_functions_order_and_random_skip():
    proto = protocol(example=True)
    proto["model"]["equations"] = [
        {"id": "E1", "formula": "y = a * x", "role": "analyses"}, {"id": "E2", "formula": "z = b * x", "role": "analyses"}]
    proto["model"]["steps"] = [
        {"id": "S1", "name": "slope fit", "does": "fit", "inputs": {"x": "m"}, "output": "o", "uses": [],
         "example": {"inputs": {"x": 2.0}, "expected_formula": "x * 2"}},
        {"id": "S2", "name": "no example", "does": "d", "inputs": {"x": "m"}, "output": "o", "uses": []},
        {"id": "S3", "name": "draw", "does": "noise", "inputs": {"x": "m"}, "output": "o", "uses": [], "random": True}]
    proto["metrics"] = [{"id": "m1", "estimand": "uses z from equation E2"}]
    proto["precision"] = {"metric": "m1"}
    cards = {c["id"]: c for c in plan_steps.parse(proto)[0]}
    specs = [
        fs.Spec("fn_e1", "def fn_e1(x):", "p", "E1"),
        fs.Spec("fn_e2", "def fn_e2(x):", "p", "E2"),
        fs.Spec("fn_s1", "def fn_s1(x):", "p", "S1", kind=fs.STEP, card=cards["S1"]),
        fs.Spec("fn_s2", "def fn_s2(x):", "p", "S2", kind=fs.STEP, card=cards["S2"]),
        fs.Spec("fn_s3", "def fn_s3(x, rng):", "p", "S3", kind=fs.STEP, card=cards["S3"]),
    ]
    chosen, skipped = si.key_functions(specs, proto)
    assert [s.name for s in chosen] == ["fn_s2", "fn_e2", "fn_e1", "fn_s1"]  # no example, feeds the metric, equation, other step
    assert "fn_s3" in skipped and "random" in skipped["fn_s3"]
    assert [s.name for s in si.key_functions(specs, proto, cap=2)[0]] == ["fn_s2", "fn_e2"]


def test_reference_problem():
    spec = spec_for(protocol())
    assert si.reference_problem(GOOD_REF, spec, "pkg") == ""
    assert "does not define" in si.reference_problem("def other(slope):\n    return 1\n", spec, "pkg")
    assert "parameters" in si.reference_problem("def activation_energy(s):\n    return 1\n", spec, "pkg")
    assert "package" in si.reference_problem("from pkg.model import activation_energy\ndef activation_energy(slope):\n    return 1\n", spec, "pkg")
    assert si.reference_problem("", spec, "pkg")


# --- the prompts never carry the main version or an expected number ----------------------------------------------------------


def test_reference_prompt_has_no_main_source_and_no_expected_value(tmp_path):
    fake = Fake(write=[fenced(GOOD_REF)])
    root, s, rec = run_it(tmp_path, fake)
    node, prompt = fake.prompts[0]
    assert node == "second_impl"
    assert "return -slope * 8.314" not in prompt and "8.314 * " not in prompt
    assert "41570" not in prompt  # -(-5000) * 8.314, the expected output FI computed
    assert "expected_formula" not in prompt
    assert "slope = -5000" in prompt  # the example INPUTS are given
    assert "activation_energy(slope)" in prompt
    assert rec["functions"]["activation_energy"]["status"] == "agree"


def test_write_prompt_refuses_a_main_line_in_the_request():
    spec = spec_for(protocol())
    leak = "    return the_secret_main_implementation_line(slope)\n"
    with pytest.raises(ValueError):
        si.write_prompt(spec, template=PROMPTS["write"], protocol=protocol(), specs=[spec],
                        passages="passage\n" + leak, example=None, main_source="def activation_energy(slope):\n" + leak)


def test_reference_lives_in_fi_not_code(tmp_path):
    fake = Fake(write=[fenced(GOOD_REF)])
    root, _s, _rec = run_it(tmp_path, fake)
    assert (root / ".fi" / "second_impl" / "activation_energy.py").is_file()
    assert not list((root / "code").rglob("activation_energy*.py"))


# --- the whole step ---------------------------------------------------------------------------------------------------------


def test_correct_reference_agrees(tmp_path):
    fake = Fake(write=[fenced(GOOD_REF)])
    root, s, rec = run_it(tmp_path, fake)
    e = rec["functions"]["activation_energy"]
    assert e["status"] == "agree" and e["rounds"] == 0 and e["inputs_tried"] >= 4 and e["disagreements"] == []
    assert rec["calls"] == 1 and fake.count() == 1
    assert si.limitations(rec) == []
    assert json.loads((root / si.RECORD).read_text(encoding="utf-8"))["functions"]["activation_energy"]["status"] == "agree"
    assert rec["code_sha"]


def test_faulty_reference_is_rewritten_once(tmp_path):
    fake = Fake(write=[fenced(BAD_REF), fenced(GOOD_REF)], adjudicate=[verdict("reference_wrong", "the sign is missing")])
    root, s, rec = run_it(tmp_path, fake)
    e = rec["functions"]["activation_energy"]
    assert e["status"] == "agree" and e["rounds"] == 1 and rec["calls"] == 3
    rewrite_prompt = fake.prompts[2][1]
    assert "the sign is missing" in rewrite_prompt and "return R * slope" in rewrite_prompt  # its own earlier version
    assert "return -slope * 8.314" not in rewrite_prompt  # never the main version


def test_reference_wrong_twice_is_unresolved(tmp_path):
    fake = Fake(write=[fenced(BAD_REF), fenced(BAD_REF)],
                adjudicate=[verdict("reference_wrong"), verdict("reference_wrong")])
    root, s, rec = run_it(tmp_path, fake)
    e = rec["functions"]["activation_energy"]
    assert e["status"] == "unresolved" and rec["calls"] == 4  # write, judge, rewrite, judge; no second rewrite
    assert e["rounds"] == 2


def test_faulty_main_is_repaired_and_kept_when_it_still_passes(tmp_path):
    fake = Fake(write=[fenced(GOOD_REF)], adjudicate=[verdict("main_wrong", "the sign of the slope is lost")])
    seen = {}
    root = make_quest(tmp_path, MUTANT_MAIN)
    proto = protocol()

    async def repair_main(name, problem, cases):
        seen.update(name=name, problem=problem, cases=cases)
        (root / "code" / "pkg" / "model.py").write_text(GOOD_MAIN, encoding="utf-8")
        return True

    async def still_passes(name):
        return True

    s = build(root, fake, [spec_for(proto)], proto, repair_main=repair_main, still_passes=still_passes)
    rec = asyncio.run(s.run_all())
    e = rec["functions"]["activation_energy"]
    assert e["status"] == "agree" and e["rounds"] == 1 and rec["calls"] == 3
    assert seen["name"] == "activation_energy" and seen["cases"] and "expected" not in seen["problem"].lower()
    assert "neither is known to be right" in seen["problem"] and "the sign of the slope is lost" in seen["problem"]
    assert (root / "code" / "pkg" / "model.py").read_text(encoding="utf-8") == GOOD_MAIN
    assert rec["code_sha"] == si._sha(root / "code" / "pkg" / "model.py")


def test_repair_is_dropped_when_the_function_no_longer_passes_its_tests(tmp_path):
    fake = Fake(write=[fenced(GOOD_REF)], adjudicate=[verdict("main_wrong")])
    root = make_quest(tmp_path, MUTANT_MAIN)
    proto = protocol()

    async def repair_main(name, problem, cases):
        (root / "code" / "pkg" / "model.py").write_text(UNITS_MAIN, encoding="utf-8")
        return True

    async def still_passes(name):
        return False

    s = build(root, fake, [spec_for(proto)], proto, repair_main=repair_main, still_passes=still_passes)
    rec = asyncio.run(s.run_all())
    e = rec["functions"]["activation_energy"]
    assert e["status"] == "unresolved"
    assert (root / "code" / "pkg" / "model.py").read_text(encoding="utf-8") == MUTANT_MAIN  # restored
    assert e["limitation"].startswith("Two independently written versions of step S1")
    assert "gave different results" in e["limitation"] and "slope = " in e["limitation"]
    assert e["limitation"].endswith("the number depends on which is right.")
    assert si.limitations(rec) == [e["limitation"]]


def test_card_ambiguous_is_unresolved_with_the_plain_sentence(tmp_path):
    fake = Fake(write=[fenced(BAD_REF)], adjudicate=[verdict("card_ambiguous", "units are not stated")])
    root, s, rec = run_it(tmp_path, fake)
    e = rec["functions"]["activation_energy"]
    assert e["status"] == "unresolved" and rec["calls"] == 2
    assert "Two independently written versions of step S1 (activation energy) gave different results" in e["limitation"]


@pytest.mark.parametrize("main", [MUTANT_MAIN, UNITS_MAIN])
def test_a_mutant_of_the_kinetics_function_disagrees(tmp_path, main):
    fake = Fake(write=[fenced(GOOD_REF)])
    root = make_quest(tmp_path, main)
    proto = protocol()
    spec = spec_for(proto)
    s = build(root, fake, [spec], proto)
    (root / ".fi" / "second_impl").mkdir(parents=True)
    (root / ".fi" / "second_impl" / "activation_energy.py").write_text(GOOD_REF, encoding="utf-8")
    entry: dict = {}
    res = asyncio.run(s._compare(spec, entry))
    assert res["status"] == "disagree" and res["compared"] >= 4
    assert res["disagreements"] and res["disagreements"][0]["kind"] == "value"
    # the unmutated main agrees with the same reference
    (root / "code" / "pkg" / "model.py").write_text(GOOD_MAIN, encoding="utf-8")
    assert asyncio.run(s._compare(spec, {}))["status"] == "agree"


def test_arrays_nan_and_exceptions_in_the_real_comparison(tmp_path):
    main = ("import numpy as np\n\ndef f(x, v):\n    if x < 0:\n        raise ValueError('negative')\n"
            "    return np.log(v) * x\n")
    ref = "import numpy as np\n\ndef f(x, v):\n    if x < 0:\n        raise ValueError('neg')\n    return x * np.log(v)\n"
    proto = {"model": {"summary": "s", "equations": [], "steps": []}}
    spec = fs.Spec("f", "def f(x, v):", "p", "E9")
    root = make_quest(tmp_path, main)
    fake = Fake()
    s = build(root, fake, [spec], proto)
    (root / ".fi" / "second_impl").mkdir(parents=True)
    (root / ".fi" / "second_impl" / "f.py").write_text(ref, encoding="utf-8")
    s.max_calls = 0
    # no example and no defaults for x, v: unconfirmed, with the plain reason
    assert asyncio.run(s._compare(spec, {}))["status"] == "unconfirmed"
    spec2 = fs.Spec("f", "def f(x=1.5, v=[1.0, 2.0, 0.0]):", "p", "E9")
    res = asyncio.run(s._compare(spec2, {}))  # log(0) = -inf and a warning: both give -inf; arrays are compared whole
    assert res["status"] == "agree"
    (root / "code" / "pkg" / "model.py").write_text(main.replace("np.log(v) * x", "np.log(v) * x + 1"), encoding="utf-8")
    assert asyncio.run(s._compare(spec2, {}))["status"] == "disagree"


def test_one_version_raising_is_a_domain_disagreement(tmp_path):
    main = "def f(x):\n    return x ** 0.5\n"
    ref = "import math\n\ndef f(x):\n    return math.sqrt(x)\n"
    spec = fs.Spec("f", "def f(x=-1.0):", "p", "E9")
    root = make_quest(tmp_path, main)
    s = build(root, Fake(), [spec], {"model": {"summary": "s", "equations": []}})
    (root / ".fi" / "second_impl").mkdir(parents=True)
    (root / ".fi" / "second_impl" / "f.py").write_text(ref, encoding="utf-8")
    res = asyncio.run(s._compare(spec, {}))
    # x ** 0.5 of a negative float is complex in Python 3, math.sqrt raises
    assert res["status"] == "disagree" and res["disagreements"][0]["kind"] == "domain"


def test_unreadable_reply_is_unconfirmed_not_a_stop(tmp_path):
    fake = Fake(write=["I cannot do that."])
    root, s, rec = run_it(tmp_path, fake)
    e = rec["functions"]["activation_energy"]
    assert e["status"] == "unconfirmed" and "could not be written" in e["limitation"] and rec["calls"] == 1


def test_no_example_inputs_is_unconfirmed(tmp_path):
    proto = protocol(example=False)
    fake = Fake(write=[fenced(GOOD_REF)])
    root, s, rec = run_it(tmp_path, fake, proto=proto)
    e = rec["functions"]["activation_energy"]
    assert e["status"] == "unconfirmed" and "example inputs" in e["limitation"]


# --- resume, cap, identity ---------------------------------------------------------------------------------------------------


def test_a_finished_function_is_not_asked_again(tmp_path):
    fake = Fake(write=[fenced(GOOD_REF)])
    root, s, rec = run_it(tmp_path, fake)
    proto = protocol()
    again = build(root, Fake(), [spec_for(proto)], proto)  # any request would raise
    rec2 = asyncio.run(again.run_all())
    assert rec2["functions"]["activation_energy"]["status"] == "agree" and rec2["calls"] == 1


def test_resume_mid_adjudication_keeps_counts_and_does_not_ask_the_judge_again(tmp_path):
    proto = protocol()
    root = make_quest(tmp_path, MUTANT_MAIN)
    (root / ".fi" / "second_impl").mkdir(parents=True)
    (root / ".fi" / "second_impl" / "activation_energy.py").write_text(GOOD_REF, encoding="utf-8")
    disagreement = {"inputs": {"slope": -5000.0}, "main": "-41570", "other": "41570", "kind": "value"}
    si.save(root, {"code_sha": si._sha(root / "code" / "pkg" / "model.py"), "calls": 2, "functions": {"activation_energy": {
        "status": "working", "rounds": 1, "inputs_tried": 5, "disagreements": [disagreement], "same_family": None,
        "adjudication": {"round": 1, "verdict": "main_wrong", "reason": "sign", "acted": False}}}})

    async def repair_main(name, problem, cases):
        (root / "code" / "pkg" / "model.py").write_text(GOOD_MAIN, encoding="utf-8")
        return True

    async def passes(name):
        return True

    fake = Fake()  # the judge and the writer must not be asked
    s = build(root, fake, [spec_for(proto)], proto, repair_main=repair_main, still_passes=passes)
    rec = asyncio.run(s.run_all())
    e = rec["functions"]["activation_energy"]
    assert e["status"] == "agree" and fake.count() == 0
    assert e["rounds"] == 1 and rec["calls"] == 3  # the two counted before the crash, plus the repair


def test_a_request_sent_but_not_saved_is_not_sent_again(tmp_path):
    proto = protocol()
    root = make_quest(tmp_path)
    si.save(root, {"calls": 2, "code_sha": si._sha(root / "code" / "pkg" / "model.py"), "functions": {"activation_energy": {
        "status": "working", "rounds": 1, "inputs_tried": 5, "disagreements": [], "pending": {"kind": "adjudicate", "round": 1}}}})
    (root / ".fi" / "second_impl").mkdir(parents=True, exist_ok=True)
    (root / ".fi" / "second_impl" / "activation_energy.py").write_text(GOOD_REF, encoding="utf-8")
    fake = Fake()
    rec = asyncio.run(build(root, fake, [spec_for(proto)], proto).run_all())
    e = rec["functions"]["activation_energy"]
    assert e["status"] == "unconfirmed" and fake.count() == 0 and rec["calls"] == 2


def test_calls_are_counted_before_the_request_and_capped(tmp_path):
    proto = protocol()
    root = make_quest(tmp_path)
    seen = []

    class Peek(Fake):
        async def chat(self, node, messages):
            seen.append(si.load(root)["calls"])  # already written when the request goes out
            return await super().chat(node, messages)

    fake = Peek(write=[fenced(GOOD_REF)])
    asyncio.run(build(root, fake, [spec_for(proto)], proto).run_all())
    assert seen == [1]
    # at the limit nothing is asked
    root2 = make_quest(tmp_path / "two")
    s = build(root2, Fake(), [spec_for(proto)], proto)
    s.max_calls = 0
    rec = asyncio.run(s.run_all())
    e = rec["functions"]["activation_energy"]
    assert e["status"] == "unconfirmed" and "limit" in e["why"] and rec["calls"] == 0
    assert si.max_calls_for(6) == 24 and si.max_calls_for(4) == 18


def test_cap_limits_how_many_functions_are_written(tmp_path):
    proto = {"model": {"summary": "s", "equations": [], "steps": [
        {"id": f"S{i}", "name": f"step {i}", "does": "d", "inputs": {"x": "m"}, "output": "o", "uses": [],
         "example": {"inputs": {"x": 2.0}, "expected_formula": "x * 2"}} for i in (1, 2, 3)]}}
    cards = plan_steps.parse(proto)[0]
    specs = [fs.Spec(f"f{i}", f"def f{i}(x):", "p", f"S{i}", kind=fs.STEP, card=cards[i - 1]) for i in (1, 2, 3)]
    root = make_quest(tmp_path)
    (root / "code" / "pkg" / "model.py").write_text("".join(f"def f{i}(x):\n    return x * 2\n\n" for i in (1, 2, 3)), encoding="utf-8")
    fake = Fake(write=[fenced(f"def f{i}(x):\n    return 2 * x\n") for i in (1, 2)])
    s = si.SecondImpl(quest_root=root, package="pkg", specs=specs, protocol=proto, prompts=PROMPTS, chat=fake.chat,
                      run=make_run(root), repair_main=lambda *a: None, still_passes=lambda n: None, cap=2)
    rec = asyncio.run(s.run_all())
    assert sorted(rec["functions"]) == ["f1", "f2"] and fake.count() == 2
    assert all(e["status"] == "agree" for e in rec["functions"].values())
    assert s.max_calls == 12


def test_same_family_is_recorded(tmp_path):
    fake = Fake(write=[fenced(GOOD_REF)])
    _root, _s, rec = run_it(tmp_path, fake, family_of=lambda node: "gpt")
    assert rec["functions"]["activation_energy"]["same_family"] is True
    fake = Fake(write=[fenced(GOOD_REF)])
    _root, _s, rec = run_it(tmp_path / "b", fake, family_of=lambda node: "claude" if node == "second_impl" else "gpt")
    assert rec["functions"]["activation_energy"]["same_family"] is False
    fake = Fake(write=[fenced(GOOD_REF)])
    _root, _s, rec = run_it(tmp_path / "c", fake)
    assert rec["functions"]["activation_energy"]["same_family"] is None


def test_random_functions_are_skipped_and_said(tmp_path):
    proto = {"model": {"summary": "s", "equations": [], "steps": []}}
    spec = fs.Spec("draw", "def draw(x, rng):", "p", "E9")
    root = make_quest(tmp_path)
    rec = asyncio.run(build(root, Fake(), [spec], proto).run_all())
    e = rec["functions"]["draw"]
    assert e["status"] == "skipped" and "random" in e["why"] and rec["calls"] == 0


# --- equation functions (the description of an equation used to raise TypeError, so none was ever checked) -------------------


def equation_protocol() -> dict:
    return {"model": {"summary": "x", "equations": [
        {"id": "E1", "formula": "Ea = -R * slope", "role": "analyses", "source": "Arrhenius 1889", "units": {"slope": "K"}}]}}


def test_spec_block_describes_an_equation_function():
    spec = fs.Spec(name="activation_energy", signature=SIG, purpose="activation energy", equation="E1")
    text = si._spec_block(spec, equation_protocol())
    assert "E1: Ea = -R * slope" in text and "source: Arrhenius 1889" in text and "purpose: activation energy" in text


def test_an_equation_function_is_checked_not_left_unconfirmed(tmp_path):
    proto = equation_protocol()
    proto["model"]["equations"][0]["example"] = {"inputs": {"slope": -5000.0}, "expected_formula": "-slope * 8.314"}
    root = make_quest(tmp_path)
    spec = fs.Spec(name="activation_energy", signature=SIG, purpose="activation energy", equation="E1")
    fake = Fake(write=[fenced(GOOD_REF)])
    rec = asyncio.run(build(root, fake, [spec], proto).run_all())
    assert rec["functions"]["activation_energy"]["status"] == "agree" and fake.count("second_impl") == 1


# --- second pass of the external review -------------------------------------------------------------------------------------


def test_no_second_version_is_asked_for_a_function_with_no_usable_example_inputs(tmp_path):
    fake = Fake()  # any request would raise
    root, s, rec = run_it(tmp_path, fake, proto=protocol(example=False))
    e = rec["functions"]["activation_energy"]
    assert e["status"] == "unconfirmed" and "example inputs" in e["limitation"]
    assert rec["calls"] == 0 and fake.count() == 0


def _repair_setup(tmp_path):
    proto = protocol()
    root = make_quest(tmp_path, MUTANT_MAIN)
    model = root / "code" / "pkg" / "model.py"
    return proto, root, model


def test_a_repair_that_breaks_another_check_of_the_package_is_undone(tmp_path):
    proto, root, model = _repair_setup(tmp_path)

    async def repair(name, problem, cases):
        model.write_text(GOOD_MAIN, encoding="utf-8")
        return True

    async def broke():
        return "E1"

    fake = Fake(write=[fenced(GOOD_REF)], adjudicate=[verdict("main_wrong")])
    s = build(root, fake, [spec_for(proto)], proto, repair_main=repair, verify_package=broke)
    rec = asyncio.run(s.run_all())
    e = rec["functions"]["activation_energy"]
    assert e["status"] == "unresolved" and "broke another check" in e["why"]
    assert model.read_text(encoding="utf-8") == MUTANT_MAIN, "the repair is not kept"
    assert not (root / si.DIR / "activation_energy.orig.py").exists()


def test_the_package_before_a_repair_is_saved_before_the_repair_writes(tmp_path):
    proto, root, model = _repair_setup(tmp_path)
    seen = {}

    async def repair(name, problem, cases):
        saved = root / si.DIR / "activation_energy.orig.py"
        seen["orig"] = saved.read_text(encoding="utf-8") if saved.is_file() else None
        model.write_text(GOOD_MAIN, encoding="utf-8")
        return True

    fake = Fake(write=[fenced(GOOD_REF)], adjudicate=[verdict("main_wrong")])
    rec = asyncio.run(build(root, fake, [spec_for(proto)], proto, repair_main=repair).run_all())
    assert seen["orig"] == MUTANT_MAIN
    assert rec["functions"]["activation_energy"]["status"] == "agree"
    assert not (root / si.DIR / "activation_energy.orig.py").exists(), "kept repair: the saved copy is dropped"


def test_a_run_that_ended_in_the_middle_of_a_repair_gets_the_unverified_code_taken_out(tmp_path):
    proto, root, model = _repair_setup(tmp_path)
    ref = root / si.DIR / "activation_energy.py"
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_text(GOOD_REF, encoding="utf-8")
    (root / si.DIR / "activation_energy.orig.py").write_text(MUTANT_MAIN, encoding="utf-8")
    model.write_text(GOOD_MAIN.replace("8.314", "99.0"), encoding="utf-8")  # written by the repair that was never judged
    si.save(root, {"calls": 3, "functions": {"activation_energy": {
        "status": "working", "rounds": 1, "inputs_tried": 5, "disagreements": [], "same_family": None,
        "repair_pending": True, "pending": {"kind": "repair_main", "round": 1}}}})
    rec = asyncio.run(build(root, Fake(), [spec_for(proto)], proto).run_all())
    assert model.read_text(encoding="utf-8") == MUTANT_MAIN, "the pre-repair file is back"
    assert "repair_pending" not in rec["functions"]["activation_energy"]


DEP_MAIN = "def a(x=2.0):\n    return x + 1.0\n\n\ndef b(x=2.0):\n    return a(x) * 3.0\n"  # a is wrong: it should double
DEP_REF_A = "def a(x=2.0):\n    return 2.0 * x\n"
DEP_REF_B = "def b(x=2.0):\n    return a(x) * 3.0\n"


def _dep_specs():
    return [fs.Spec(name="a", signature="def a(x=2.0):", purpose="double", equation="E1"),
            fs.Spec(name="b", signature="def b(x=2.0):", purpose="triple the double", equation="E2", depends_on=["a"])]


def _dep_protocol():
    return {"model": {"summary": "x", "equations": [{"id": "E1", "formula": "a = 2 * x", "role": "analyses"},
                                                    {"id": "E2", "formula": "b = 3 * a", "role": "analyses"}]}}


def test_a_dependency_with_a_second_version_of_its_own_is_called_through_it(tmp_path):
    root = make_quest(tmp_path, DEP_MAIN)
    proto = _dep_protocol()
    ref_dir = root / si.DIR
    ref_dir.mkdir(parents=True)
    (ref_dir / "a.py").write_text(DEP_REF_A, encoding="utf-8")
    fake = Fake(write=[fenced(DEP_REF_B)], adjudicate=[verdict("card_ambiguous")])
    si.save(root, {"calls": 0, "functions": {"a": {"status": "agree", "rounds": 0, "inputs_tried": 5, "disagreements": [],
                                                   "same_family": None}}})
    rec = asyncio.run(build(root, fake, _dep_specs()[1:], proto).run_all() if False else
                      build(root, fake, _dep_specs(), proto, cap=2).run_all())
    e = rec["functions"]["b"]
    assert e["status"] == "unresolved", "a fault in the dependency is not shared when it has a version of its own"
    assert "deps_from_first_version" not in e


def test_a_dependency_without_one_is_called_from_the_first_version_and_the_record_says_so(tmp_path):
    root = make_quest(tmp_path, DEP_MAIN)
    proto = _dep_protocol()
    fake = Fake(write=[fenced(DEP_REF_B)])
    specs = _dep_specs()[1:]
    rec = asyncio.run(build(root, fake, specs, proto).run_all())
    e = rec["functions"]["b"]
    assert e["status"] == "agree" and e["deps_from_first_version"] == ["a"]


def test_a_stale_entry_checked_again_starts_its_rounds_over(tmp_path):
    proto = protocol()
    root = make_quest(tmp_path, MUTANT_MAIN)
    ref = root / si.DIR / "activation_energy.py"
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_text(GOOD_REF, encoding="utf-8")
    si.save(root, {"calls": 2, "code_sha": "old", "functions": {"activation_energy": {
        "status": "unresolved", "rounds": 2, "inputs_tried": 5, "disagreements": [], "same_family": None}}})
    fake = Fake(adjudicate=[verdict("card_ambiguous")])
    rec = asyncio.run(build(root, fake, [spec_for(proto)], proto).run_all())
    e = rec["functions"]["activation_energy"]
    assert fake.count("second_impl.adjudicate") == 1, "the rounds allowed start again for the changed code"
    assert e["rounds"] == 1 and e["status"] == "unresolved"


def _write_request(tmp_path, record):
    proto = protocol()
    root = make_quest(tmp_path)
    if record is not None:
        (root / ".fi" / "second_impl").mkdir(parents=True)
        (root / si.CAPTURED).write_text(json.dumps(record), encoding="utf-8")
    s = build(root, Fake(), [spec_for(proto)], proto)
    return si.write_prompt(spec_for(proto), template=PROMPTS["write"], protocol=proto, specs=s.specs, passages="",
                           example=None, return_keys=s._return_keys(spec_for(proto)))


def test_the_second_writer_is_told_every_key_the_first_version_returned(tmp_path):
    text = _write_request(tmp_path, {"cases": {}, "returned": {"activation_energy": ["Ea", "ci_Ea", "cov"]}})
    assert "the first version's dict results carried these keys: Ea, ci_Ea, cov. A key may appear only for some inputs" in text


def test_the_second_writer_gets_no_key_list_without_a_record(tmp_path):
    assert "What it must return" not in _write_request(tmp_path, None)


def test_a_key_the_first_version_has_at_any_depth_that_the_second_lacks_is_unconfirmed_with_its_path():
    one = [{"x": 1.0}]
    nested = si.judge_rows([{"main": {"result": {"a": 1, "b": 2}}, "ref": {"result": {"a": 1}}}], one, si.REL)
    assert nested["status"] == "unconfirmed" and "result.b" in nested["why"]
    in_list = si.judge_rows([{"main": {"rows": [{"a": 1, "b": 2}]}, "ref": {"rows": [{"a": 1}]}}], one, si.REL)
    assert in_list["status"] == "unconfirmed" and "rows[0].b" in in_list["why"]
    # the second version may carry more at depth; a real difference at depth is still a disagreement
    assert si.judge_rows([{"main": {"result": {"a": 1}}, "ref": {"result": {"a": 1, "c": 3}}}], one, si.REL)["status"] == "agree"
    assert si.judge_rows([{"main": {"result": {"a": 1.0}}, "ref": {"result": {"a": 1.5}}}], one, si.REL)["status"] == "disagree"


def test_the_comparison_calls_the_function_after_the_study_file_has_set_the_package_constants(tmp_path):
    """The package's constant is set by simulate.py when the study runs; the comparison loads simulate.py the same way
    before calling, so the first version returns a value instead of raising NameError."""
    pkg = tmp_path / "code" / "pkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "model.py").write_text("def scaled(x):\n    return x * LIMIT\n", encoding="utf-8")
    (tmp_path / "code" / "simulate.py").write_text("from pkg import model\nmodel.LIMIT = 4\n", encoding="utf-8")
    (tmp_path / "ref.py").write_text("def scaled(x):\n    return x * 4\n", encoding="utf-8")
    (tmp_path / "inputs.json").write_text(json.dumps([{"x": 1.0}, {"x": 2.0}]), encoding="utf-8")

    def rows(study: str) -> dict:
        script = si.compare_script(code_dir="code", module="pkg.model", name="scaled", reference="ref.py",
                                   inputs="inputs.json", uses=[], study=study)
        p = subprocess.run([sys.executable, "-B", "-c", script], cwd=tmp_path, capture_output=True, text=True, timeout=120)
        return si.parse_output(p.stdout)

    without = rows("")
    assert "NameError" in without["rows"][0]["main_err"]  # what the comparison did before: it raised, the other did not
    got = rows("code/simulate.py")
    assert si.judge_rows(got["rows"], [{"x": 1.0}, {"x": 2.0}], si.REL)["status"] == si.AGREE
    assert [r["main"] for r in got["rows"]] == [4.0, 8.0]


def test_both_versions_raising_alike_is_a_comparison_that_could_not_run():
    rows = [{"main_err": "NameError: LIMIT", "ref_err": "NameError: LIMIT"}]
    res = si.judge_rows(rows, [{"x": 1.0}], si.REL)
    assert res["status"] == si.UNCONFIRMED and not res["disagreements"]


def test_the_study_file_is_loaded_with_the_environment_the_trial_harness_gives_it(tmp_path):
    """A study file that reads FI's thresholds when it loads gets them in the comparison too, as in a trial."""
    pkg = tmp_path / "code" / "pkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "model.py").write_text("def scaled(x):\n    return x * LIMIT\n", encoding="utf-8")
    (tmp_path / "code" / "simulate.py").write_text(
        "import json, os\nfrom pkg import model\nTHRESHOLDS = json.loads(os.environ['FI_THRESHOLDS'])\n"
        "KIND = os.environ['FI_CELL_KIND']\nmodel.LIMIT = 4\n", encoding="utf-8")
    (tmp_path / "ref.py").write_text("def scaled(x):\n    return x * 4\n", encoding="utf-8")
    (tmp_path / "inputs.json").write_text(json.dumps([{"x": 1.0}]), encoding="utf-8")
    script = si.compare_script(code_dir="code", module="pkg.model", name="scaled", reference="ref.py",
                               inputs="inputs.json", uses=[], study="code/simulate.py")
    for inherited in ({}, {"FI_THRESHOLDS": '{"x": 1}', "FI_CELL_KIND": "search"}):
        env = {k: v for k, v in os.environ.items() if k not in ("FI_THRESHOLDS", "FI_CELL_KIND")}
        p = subprocess.run([sys.executable, "-B", "-c", script], cwd=tmp_path, capture_output=True, text=True,
                           timeout=120, env={**env, **inherited})
        got = si.parse_output(p.stdout)
        assert "study_load_error" not in got, got
        assert [r["main"] for r in got["rows"]] == [4.0]


def test_the_study_file_never_sees_a_threshold_or_kind_inherited_from_the_process_that_started_the_comparison(tmp_path):
    pkg = tmp_path / "code" / "pkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "model.py").write_text("def seen(x):\n    return [THRESHOLDS, KIND]\n", encoding="utf-8")
    (tmp_path / "code" / "simulate.py").write_text(
        "import json, os\nfrom pkg import model\nmodel.THRESHOLDS = json.loads(os.environ['FI_THRESHOLDS'])\n"
        "model.KIND = os.environ['FI_CELL_KIND']\n", encoding="utf-8")
    (tmp_path / "ref.py").write_text("def seen(x):\n    return [{}, 'check']\n", encoding="utf-8")
    (tmp_path / "inputs.json").write_text(json.dumps([{"x": 1.0}]), encoding="utf-8")
    script = si.compare_script(code_dir="code", module="pkg.model", name="seen", reference="ref.py",
                               inputs="inputs.json", uses=[], study="code/simulate.py")
    env = {**os.environ, "FI_THRESHOLDS": '{"x": 1}', "FI_CELL_KIND": "search"}
    p = subprocess.run([sys.executable, "-B", "-c", script], cwd=tmp_path, capture_output=True, text=True, timeout=120,
                       env=env)
    got = si.parse_output(p.stdout)
    assert si.judge_rows(got["rows"], [{"x": 1.0}], si.REL)["status"] == si.AGREE, got


# --- a version that could not run is not a disagreement; the study is set up as a trial sets it up -------------------------


def test_a_version_that_raised_on_every_input_was_not_compared():
    inputs = [{"x": 1.0}, {"x": 2.0}, {"x": 3.0}]
    first = [{"main_err": "NameError: name 'LIMIT' is not defined", "ref": 4.0} for _ in inputs]
    res = si.judge_rows(first, inputs, si.REL)
    assert res["status"] == si.UNCONFIRMED and not res["disagreements"] and res["compared"] == 0
    assert res["why"] == "the first version could not be run in the comparison: NameError (the study's setup trial was not run)"
    assert "did not run to the end" in si.judge_rows(first, inputs, si.REL, setup="failed")["why"]
    assert "ran to the end" in si.judge_rows(first, inputs, si.REL, setup="ran")["why"]
    second = [{"main": 4.0, "ref_err": "ModuleNotFoundError: no module named q"} for _ in inputs]
    assert si.judge_rows(second, inputs, si.REL)["why"].startswith("the second version could not be run in the comparison: Mod")
    # an error that can be the version's own fault, on every input, is a disagreement the judge can settle
    for fault in ("AttributeError: 'float' object has no attribute 'sum'", "UnboundLocalError: local variable 'a'"):
        own = [{"main_err": fault, "ref": 4.0} for _ in inputs]
        assert si.judge_rows(own, inputs, si.REL)["status"] == si.DISAGREE
    # every error must be a missing name: a missing name on one input and another error on the rest is a disagreement
    mixed = [{"main_err": "NameError: x", "ref": 4.0}, {"main_err": "ZeroDivisionError: y", "ref": 4.0},
             {"main_err": "ZeroDivisionError: y", "ref": 4.0}]
    assert si.judge_rows(mixed, inputs, si.REL)["status"] == si.DISAGREE
    # a domain error on every input is a result of the function, not a version that could not run
    domain = [{"main": 4.0, "ref_err": "ValueError: math domain error"} for _ in inputs]
    assert si.judge_rows(domain, inputs, si.REL)["status"] == si.DISAGREE
    # raising on only some inputs is still a disagreement about where the versions differ (a known positive)
    some = [{"main_err": "NameError: x", "ref": 4.0}, {"main": 4.0, "ref": 4.0}, {"main": 4.0, "ref": 4.0}]
    assert si.judge_rows(some, inputs, si.REL)["status"] == si.DISAGREE


def _study_quest(tmp_path: Path, main: str, simulate: str) -> Path:
    pkg = tmp_path / "code" / "pkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "model.py").write_text(main, encoding="utf-8")
    (tmp_path / "code" / "simulate.py").write_text(simulate, encoding="utf-8")
    (tmp_path / "ref.py").write_text("def scaled(x):\n    return x * 4\n", encoding="utf-8")
    (tmp_path / "inputs.json").write_text(json.dumps([{"x": 1.0}, {"x": 2.0}]), encoding="utf-8")
    return tmp_path


CONFIGURED_BY_THE_ENTRY = (
    "from pkg import model\n\n\ndef _configure(limit):\n    model.LIMIT = limit\n\n\n"
    "def run_trial(cell, trial_id, seed):\n    _configure(cell['limit'])\n    return {'y': 1.0}\n")


def _compare_rows(root: Path, setup) -> dict:
    script = si.compare_script(code_dir="code", module="pkg.model", name="scaled", reference="ref.py",
                               inputs="inputs.json", uses=[], study="code/simulate.py", setup=setup)
    p = subprocess.run([sys.executable, "-B", "-c", script], cwd=root, capture_output=True, text=True, timeout=120)
    return si.parse_output(p.stdout)


def test_a_constant_the_entry_sets_is_set_before_the_versions_are_called(tmp_path):
    """The study's entry sets the package's constant when a trial runs (not when the file loads). Calling the entry once on
    a setting that ran puts the comparison in the same state; without it the first version raises NameError."""
    root = _study_quest(tmp_path, "def scaled(x):\n    return x * LIMIT\n", CONFIGURED_BY_THE_ENTRY)
    inputs = [{"x": 1.0}, {"x": 2.0}]
    bare = _compare_rows(root, None)
    assert "NameError" in bare["rows"][0]["main_err"]
    assert si.judge_rows(bare["rows"], inputs, si.REL)["status"] == si.UNCONFIRMED
    got = _compare_rows(root, {"entry": "run_trial", "cell": {"limit": 4}, "kind": "scan"})
    assert [r["main"] for r in got["rows"]] == [4.0, 8.0]
    assert si.judge_rows(got["rows"], inputs, si.REL)["status"] == si.AGREE
    # a known positive: with the constant set, a first version that really differs is still a disagreement
    (root / "code" / "pkg" / "model.py").write_text("def scaled(x):\n    return x * LIMIT * 2\n", encoding="utf-8")
    off = _compare_rows(root, {"entry": "run_trial", "cell": {"limit": 4}, "kind": "scan"})
    res = si.judge_rows(off["rows"], inputs, si.REL)
    assert res["status"] == si.DISAGREE and res["disagreements"][0]["kind"] == "value"


def test_a_setup_trial_that_fails_leaves_the_comparison_to_say_which_version_could_not_run(tmp_path):
    root = _study_quest(tmp_path, "def scaled(x):\n    return x * LIMIT\n",
                        "def run_trial(cell, trial_id, seed):\n    raise RuntimeError('no')\n")
    got = _compare_rows(root, {"entry": "run_trial", "cell": {}, "kind": "scan"})
    assert "study_load_error" not in got
    assert si.judge_rows(got["rows"], [{"x": 1.0}, {"x": 2.0}], si.REL)["why"].startswith("the first version could not be run")


def test_the_run_fi_made_gives_the_setting_the_comparison_sets_the_study_up_with(tmp_path):
    root = _study_quest(tmp_path, "def scaled(x):\n    return x * LIMIT\n",
                        "from pkg import model\n\n\ndef run_cell(cell):\n    model.LIMIT = cell['limit']\n    "
                        "return {'y': model.scaled(1.0)}\n")
    asyncio.run(si.capture_inputs(make_run(root), root, "pkg", ["scaled"], [{"limit": 4}, {"limit": 5}], deterministic=True,
                                  in_each_run={"t": [1, 2]}))
    setup = si.read_captured(root)["setup"]
    assert setup["entry"] == "run_cell" and setup["kind"] == "scan" and setup["cell"]["t"] == [1, 2]
    assert setup["cell"]["limit"] in (4, 5) and setup["thresholds"] == {} and setup["duration_s"] > 0


def test_a_comparison_where_the_first_version_cannot_run_is_not_sent_to_the_judge(tmp_path):
    """The recorded fault: the first version raised on every input, the judge was asked, and the paper said the two
    versions disagree. Now the step is unconfirmed, nothing is asked and no sentence says they disagree."""
    main = GOOD_MAIN.replace("-slope * 8.314", "-slope * MISSING_CONSTANT")
    fake = Fake(write=[fenced(GOOD_REF)])
    root, s, rec = run_it(tmp_path, fake, main=main)
    e = rec["functions"]["activation_energy"]
    assert e["status"] == "unconfirmed" and not e["disagreements"]
    assert fake.count("second_impl.adjudicate") == 0
    assert "the first version could not be run in the comparison: NameError" in e["limitation"]
    assert "setup trial was not run" in e["limitation"]  # no study file here: nothing was set up
    assert "different results" not in e["limitation"]


def test_the_judge_is_shown_the_values_the_plan_fixes():
    proto = protocol()
    proto["constants"] = {"gas_constant": 8.314, "true_energy": 60}
    proto["numerical_settings"] = {"tol": 1e-10}
    block = si._spec_block(spec_for(proto), proto)
    assert "gas_constant = 8.314" in block and "true_energy = 60" in block and "tol = " in block
    assert "not results" in block
    assert "the plan fixes" not in si._spec_block(spec_for(protocol()), protocol())


def test_fixed_values_are_never_cut_inside_an_entry_and_never_hold_a_result_field():
    proto = protocol()
    proto["constants"] = {f"c{i}": 0.123456789012 + i for i in range(80)}
    proto["metrics"] = [{"id": "m", "known_value": 987654.321, "reference": "R-REF"}]
    proto["oracles"] = [{"name": "o", "expected": 424242.5}]
    proto["acceptance"] = {"rule": "ACCEPT-RULE"}
    text = si._fixed_values(proto)
    assert "more not shown" in text and len(text) < si.FIXED_VALUES_CHARS + 300
    shown = [e for e in text.split(": ", 1)[1].split("; ")]
    assert all(e.rsplit(" = ", 1)[-1].replace(" (", " ").split(" ")[0].count(".") <= 1 for e in shown)  # a number, whole
    assert shown[0].startswith("constant c0 = 0.123456789012")
    for held_result in ("987654", "R-REF", "424242", "ACCEPT-RULE"):
        assert held_result not in text


def test_the_sentence_gives_each_versions_result_and_labels_the_inputs():
    cases = [{"inputs": {"slope": -5000.0}, "kind": "value", "main": "41570", "other": "20785"}]
    sentence = si.limitation_unresolved(spec_for(protocol()), cases)
    inputs_at, results_at = sentence.index("slope = -5000"), sentence.index("41570")
    assert "inputs" in sentence[:inputs_at] and inputs_at < results_at < sentence.index("20785")
    assert "41570" not in sentence[:inputs_at] and "slope" not in sentence[results_at:]
    assert "41570" in si.cases_text(cases) and "20785" in si.cases_text(cases)


def test_the_setup_is_the_fastest_setting_that_ran_with_its_thresholds_and_seed(tmp_path):
    root = _study_quest(tmp_path, "def scaled(x):\n    return x * LIMIT\n",
                        "import os, time, json\nfrom pkg import model\n\n\ndef run_cell(cell):\n"
                        "    model.LIMIT = cell['limit']\n    time.sleep(cell['wait'])\n    return {'y': 1.0}\n")
    cells = [{"limit": 4, "wait": 1.5}, {"limit": 5, "wait": 0.0}, {"limit": 6, "wait": 0.7}]
    asyncio.run(si.capture_inputs(make_run(root), root, "pkg", ["scaled"], cells, deterministic=True,
                                  thresholds={"cut": 3}))
    rec = si.read_captured(root)
    setup = rec["setup"]
    assert setup["cell"]["limit"] == 5 and setup["thresholds"] == {"cut": 3}
    assert setup["duration_s"] == rec["took"]["1"] == min(rec["took"].values())
    assert set(rec["took"]) == {"0", "1", "2"}


def test_a_setup_trial_gets_the_thresholds_of_the_trials_and_the_versions_calls_do_not(tmp_path):
    study = ("import os, json\nfrom pkg import model\n\n\ndef run_trial(cell, trial_id, seed):\n"
             "    model.CUT = json.loads(os.environ['FI_THRESHOLDS'])['cut']\n    model.SEED = seed\n    return {'y': 1.0}\n")
    root = _study_quest(tmp_path, "import os\n\n\ndef scaled(x):\n    return [x * CUT, SEED, os.environ['FI_THRESHOLDS'], "
                                  "os.environ['FI_CELL_KIND']]\n", study)
    got = _compare_rows(root, {"entry": "run_trial", "cell": {}, "kind": "scan", "thresholds": {"cut": 3}, "seed": 17})
    assert got["setup"] == "ran"
    assert got["rows"][0]["main"] == [3.0, 17, "{}", "check"]


def test_a_comparison_with_a_setup_trial_runs_apart_from_the_quest_and_leaves_its_outputs_alone(tmp_path):
    """The comparison now runs a real trial: it must run from a copy (as FI's capture run does), neither reading nor
    leaving the study's kept values, and the study's outputs stay as they were."""
    from core.execution import ExecutionResult
    from core.outside_run import OutsideExecutor

    class Inner:
        async def execute(self, cmd, *, cwd, timeout_s, env=None):
            done = await asyncio.to_thread(subprocess.run, cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout_s,
                                           env=env)
            return ExecutionResult(done.returncode, done.stdout, done.stderr, 0.0)

    study = ("import os\nfrom pkg import model\n\n\ndef run_trial(cell, trial_id, seed):\n"
             "    os.makedirs('data/results/cache', exist_ok=True)\n"
             "    model.KEPT = os.path.isfile('data/results/cache/kept.txt')\n"
             "    open('data/results/cache/kept.txt', 'w').write('x')\n    open('data/results/x.txt', 'w').write('x')\n"
             "    open('figure.png', 'w').write('x')\n    model.LIMIT = 4\n    return {'y': 1.0}\n")
    proto = protocol(example=False)
    root = make_quest(tmp_path, "def activation_energy(slope):\n    return [slope * LIMIT, KEPT]\n")
    (root / "code" / "simulate.py").write_text(study, encoding="utf-8")
    (root / "data" / "results" / "cache").mkdir(parents=True)
    (root / "data" / "results" / "cache" / "kept.txt").write_text("the study's own", encoding="utf-8")
    record = {"cases": {"activation_energy": [{"slope": 2.0}]}, "ran": [0],
              "setup": {"entry": "run_trial", "cell": {}, "kind": "scan", "thresholds": {}, "seed": 1, "duration_s": 0.1}}
    (root / ".fi" / "second_impl").mkdir(parents=True)
    (root / si.CAPTURED).write_text(json.dumps(record), encoding="utf-8")
    before = {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file() and ".fi" not in p.parts
              and "__pycache__" not in p.parts}
    executor = OutsideExecutor(Inner(), root, base=tmp_path / "scratch", share=False, only_named=True)
    seen_extra = []

    async def run(argv, extra_s=0.0):
        seen_extra.append(extra_s)
        ran = await executor.execute([sys.executable, "-B", *argv], cwd=root, timeout_s=300)
        return ran.returncode, ran.stdout, ran.stderr

    fake = Fake(write=[fenced("def activation_energy(slope):\n    return [slope * 4, False]\n")])
    s = si.SecondImpl(quest_root=root, package="pkg", specs=[spec_for(proto)], protocol=proto, prompts=PROMPTS,
                      chat=fake.chat, run=run, repair_main=None, still_passes=None,
                      captured=record["cases"])
    rec = asyncio.run(s.run_all())
    e = rec["functions"]["activation_energy"]
    assert e["status"] == "agree", e  # the cached value the study kept was not read: KEPT is False in the trial
    assert seen_extra and seen_extra[0] == 0.1  # the time FI measured for the setup trial is added to the limit
    after = {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file() and ".fi" not in p.parts
             and "__pycache__" not in p.parts}
    assert after == before, set(after) ^ set(before)


def _apart_runner(exec_env, *, comparison, timeout_s=600):
    from types import SimpleNamespace

    from core.engine import Engine

    seen = {}

    async def run_in_env(argv, **kw):
        seen.update(kw)
        return 0, "", ""

    me = SimpleNamespace(_run_in_env=run_in_env, config=SimpleNamespace(execution=SimpleNamespace(timeout_s=timeout_s)))
    return Engine._run_apart_from_study(me, exec_env, comparison=comparison), seen


def test_the_comparison_keeps_an_unset_environment_and_the_captures_runs_keep_an_empty_one():
    run, seen = _apart_runner(None, comparison=True)
    asyncio.run(run(["-c", "x"]))
    assert seen["env"] is None and seen["apart"] is True
    run, seen = _apart_runner(None, comparison=False)
    asyncio.run(run(["x"]))
    assert seen["env"] == {} and seen["apart"] is True
    run, seen = _apart_runner({"A": "1"}, comparison=True)
    asyncio.run(run(["x"]))
    assert seen["env"] == {"A": "1"}


def test_the_comparisons_base_limit_is_never_under_what_it_had_before_and_adds_the_setup_time():
    run, seen = _apart_runner({}, comparison=True, timeout_s=600)  # the capture budget of a 600 s limit is under 180 s
    asyncio.run(run(["x"]))
    assert seen["timeout_s"] == si.COMPARE_BASE_S
    asyncio.run(run(["x"], extra_s=41.2))
    assert seen["timeout_s"] == si.COMPARE_BASE_S + 42
    run, seen = _apart_runner({}, comparison=False, timeout_s=600)
    asyncio.run(run(["x"]))
    assert seen["timeout_s"] < si.COMPARE_BASE_S  # the capture's own limit is unchanged


def test_the_setup_state_is_in_the_record_and_an_agreement_on_an_unset_study_says_so(tmp_path):
    study = "def run_trial(cell, trial_id, seed):\n    raise RuntimeError('no')\n"
    proto = protocol(example=False)
    root = make_quest(tmp_path, "def activation_energy(slope):\n    return -slope * 8.314\n")
    (root / "code" / "simulate.py").write_text(study, encoding="utf-8")
    (root / ".fi" / "second_impl").mkdir(parents=True)
    record = {"cases": {"activation_energy": [{"slope": 2.0}]}, "ran": [0],
              "setup": {"entry": "run_trial", "cell": {}, "kind": "scan", "thresholds": {}, "seed": 1, "duration_s": 0.1}}
    (root / si.CAPTURED).write_text(json.dumps(record), encoding="utf-8")
    fake = Fake(write=[fenced(GOOD_REF)])
    s = build(root, fake, [spec_for(proto)], proto, captured=record["cases"])
    e = asyncio.run(s.run_all())["functions"]["activation_energy"]
    assert e["status"] == "agree" and e["setup"] == "failed"  # the verdict is unchanged
    assert "setup trial did not run to the end" in e["setup_note"] and e["setup_note"] in e["why"]


def test_a_run_that_ran_out_of_time_is_not_run_again(tmp_path):
    proto = protocol()
    root = make_quest(tmp_path, GOOD_MAIN)
    (root / "code" / "simulate.py").write_text("def run_cell(cell):\n    return {}\n", encoding="utf-8")
    (root / ".fi" / "second_impl").mkdir(parents=True)
    record = {"cases": {}, "ran": [0], "setup": {"entry": "run_cell", "cell": {}, "kind": "scan", "thresholds": {},
                                                  "duration_s": 5.0}}
    (root / si.CAPTURED).write_text(json.dumps(record), encoding="utf-8")
    calls = []

    async def run(argv, extra_s=0.0):
        calls.append(extra_s)
        return -1, "", "partial\n" + si.TIMED_OUT

    s = build(root, Fake(write=[fenced(GOOD_REF)]), [spec_for(proto)], proto)
    s.run = run
    e = asyncio.run(s.run_all())["functions"]["activation_energy"]
    assert calls == [5.0], calls
    assert e["status"] == "unconfirmed" and "did not finish in time" in e["why"]


# --- what is compared is what both versions return ----------------------------------------------------------------------


def _script_rows(tmp_path, main_src: str, ref_src: str, inputs: list) -> list:
    pkg = tmp_path / "code" / "pkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "model.py").write_text(main_src, encoding="utf-8")
    (tmp_path / "ref.py").write_text(ref_src, encoding="utf-8")
    (tmp_path / "inputs.json").write_text(json.dumps(inputs), encoding="utf-8")
    script = si.compare_script(code_dir="code", module="pkg.model", name="level", reference="ref.py", inputs="inputs.json",
                               uses=[])
    p = subprocess.run([sys.executable, "-B", "-c", script], cwd=tmp_path, capture_output=True, text=True, timeout=120)
    return si.parse_output(p.stdout)["rows"]


MAIN_FOUR = ("def level(x):\n    if x <= 0:\n        return (float('nan'), float('nan'), False, 'x<=0')\n"
             "    return (2.0 * x, 0.5 * x, True, '')\n")
REF_TWO = "def level(x):\n    if x <= 0:\n        return (float('nan'), 0.25)\n    return (2.0 * x, 0.5 * x)\n"


def test_extra_trailing_items_are_named_not_a_disagreement(tmp_path):
    inputs = [{"x": 1.0}, {"x": 2.0}]
    rows = _script_rows(tmp_path, MAIN_FOUR, REF_TWO, inputs)
    v = si.judge_rows(rows, inputs, si.REL)
    assert v["status"] == si.AGREE and not v["disagreements"] and v["compared"] == 2
    assert "only the first version returns items 3 and 4; not checked by the other version" in v["shape_notes"][0]


def test_a_shared_item_that_differs_is_still_a_disagreement_beside_extra_items(tmp_path):
    ref = "def level(x):\n    return (2.5 * x, 0.5 * x)\n"
    inputs = [{"x": 1.0}]
    rows = _script_rows(tmp_path, MAIN_FOUR, ref, inputs)
    v = si.judge_rows(rows, inputs, si.REL)
    assert v["status"] == si.DISAGREE and v["disagreements"][0]["kind"] == "value"


def test_a_series_of_another_length_is_still_a_disagreement(tmp_path):
    main = "def level(x):\n    return [x, x, x]\n"
    ref = "def level(x):\n    return [x, x]\n"
    inputs = [{"x": 1.0}]
    v = si.judge_rows(_script_rows(tmp_path, main, ref, inputs), inputs, si.REL)
    assert v["status"] == si.DISAGREE


def test_a_failure_one_version_reports_where_the_other_returns_a_number_is_a_disagreement(tmp_path):
    # NaN from one version where the other returns a number is a different answer on that input, whatever else agrees
    inputs = [{"x": 1.0}, {"x": -1.0}]
    v = si.judge_rows(_script_rows(tmp_path, MAIN_FOUR, REF_TWO, inputs), inputs, si.REL)
    assert v["status"] == si.DISAGREE and [d["inputs"] for d in v["disagreements"]] == [{"x": -1.0}]

# --- the tolerance of the comparison: relative 1e-3 -------------------------------------------------------------------------


def _spec_with(example: dict | None = None) -> fs.Spec:
    card = {"id": "S1", "name": "fit", "example": example} if example is not None else {"id": "S1", "name": "fit"}
    return fs.Spec(name="fit_rate", signature="def fit_rate(t, c):", purpose="fit", equation="S1", kind="step", card=card)


def test_tolerance_is_relative_1e3_for_every_function_and_a_method_example_may_loosen_it():
    assert si.tolerance_for(_spec_with({"untestable": "arrays"}), {}) == 1e-3
    assert si.tolerance_for(_spec_with(), None) == 1e-3
    method = {"method": "bisection", "tolerance": 1e-2, "tolerance_mode": "relative"}
    assert si.tolerance_for(_spec_with(method), {}) == 1e-2
    assert si.tolerance_for(_spec_with({**method, "tolerance": 1e-9}), {}) == 1e-3  # never tighter than the tolerance


def test_a_solver_level_gap_agrees_and_a_one_percent_difference_does_not():
    tol = si.tolerance_for(_spec_with(), {})
    inputs = [{"t": [1.0]}]
    benign = [{"main": [0.00246142, 0.000955612], "ref": [0.00246148, 0.000955512]}]  # gaps of 2e-5 and 1e-4
    assert si.judge_rows(benign, inputs, si.REL)["status"] == si.DISAGREE  # the failing input, on the old floor
    assert si.judge_rows(benign, inputs, tol)["status"] == si.AGREE
    planted = [{"main": [1.0, 2.0], "ref": [1.0, 2.0 * 1.01]}]
    assert si.judge_rows(planted, inputs, tol)["status"] == si.DISAGREE


def test_nan_or_infinity_against_a_number_is_a_disagreement_at_any_tolerance():
    tol = si.tolerance_for(_spec_with(), {})
    inputs = [{"t": [1.0]}]
    assert si.judge_rows([{"main": float("nan"), "ref": 0.5}], inputs, tol)["status"] == si.DISAGREE
    assert si.judge_rows([{"main": {"f": "nan"}, "ref": 0.5}], inputs, tol)["status"] == si.DISAGREE
    assert si.judge_rows([{"main": 0.5, "ref": {"f": "inf"}}], inputs, tol)["status"] == si.DISAGREE
