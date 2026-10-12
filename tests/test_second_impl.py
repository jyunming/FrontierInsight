"""core/second_impl.py: key function choice, inputs, comparison (pure), and the whole step with a fake model and a real
interpreter (the comparison really imports both versions in a subprocess)."""

from __future__ import annotations

import asyncio
import json
import math
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
    assert "gave different results on slope = " in e["limitation"] and e["limitation"].endswith("the number depends on which is right.")
    assert si.limitations(rec) == [e["limitation"]]


def test_card_ambiguous_is_unresolved_with_the_plain_sentence(tmp_path):
    fake = Fake(write=[fenced(BAD_REF)], adjudicate=[verdict("card_ambiguous", "units are not stated")])
    root, s, rec = run_it(tmp_path, fake)
    e = rec["functions"]["activation_energy"]
    assert e["status"] == "unresolved" and rec["calls"] == 2
    assert "Two independently written versions of step S1 (activation energy) gave different results on" in e["limitation"]


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
