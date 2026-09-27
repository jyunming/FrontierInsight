"""A cap counts inside the range only for the value it is proven to cap (the 2026-09-27 delta re-audit, F6).

The check used to match the last segment of the path: ``{"train": {"rmse": min(train_rmse, 10.0)}}`` made a test RMSE
of exactly 10 look capped, though nothing capped it. Now a cap blocks only when it sits at the value's whole path in
the result the script prints; a cap that may be the value's but whose place cannot be shown (built in a loop, appended,
assigned under another key) is a warning (``clamped_unproven``), never a block.
"""

from __future__ import annotations

from core.plausibility import Assertion, direct_caps, violations

RMSE = [Assertion(path="rmse", min=0.0, max=100.0)]
TEST_RMSE = [Assertion(path="test.rmse", min=0.0, max=100.0)]
P = "print(json.dumps(out))\n"


def _kinds(found: list) -> list[str]:
    return [v.kind for v in found]


def test_a_cap_on_train_is_not_a_cap_on_test() -> None:
    code = 'out = {"train": {"rmse": min(train_rmse, 10.0)}, "test": {"rmse": test_rmse}}\n' + P
    result = {"train": {"rmse": 3.0}, "test": {"rmse": 10.0}}
    assert violations(result, TEST_RMSE, code=code) == [], "the re-audit's counterexample"
    assert violations(result, RMSE, code=code) == [], "the assertion names only the leaf: still test's own path counts"
    (found,) = violations({"train": {"rmse": 10.0}, "test": {"rmse": 2.0}}, RMSE, code=code)
    assert found.kind == "clamped" and found.path == "train.rmse"


def test_a_cap_whose_place_cannot_be_shown_is_a_warning() -> None:
    for code in (
        'out = to_native({"test": {"rmse": min(r, 10.0)}})\n' + P,   # wrapped in a call the check cannot follow
        'rows.append({"rmse": min(e, 10.0)})\nout = {"runs": rows}\n' + P,  # a list of records built by append
        'unused = {"test": {"rmse": min(x, 10.0)}}\nout = {"test": {"rmse": r}}\n' + P,  # a literal never printed
    ):
        result = {"runs": [{"rmse": 10.0}]} if "rows" in code else {"test": {"rmse": 10.0}}
        assert _kinds(violations(result, RMSE, code=code)) == ["clamped_unproven"], code
    # A train cap written in a loop is still not a test cap (its chain ends in train.rmse).
    code = 'for k in ks:\n    res[k] = {"train": {"rmse": min(t, 10.0)}}\nout = {"by": res}\n' + P
    assert violations({"by": {"a": {"test": {"rmse": 10.0}}}}, RMSE, code=code) == []


def test_sweep_shapes_are_proven_when_printed_whole() -> None:
    # A computed key matches any one key, a comprehension position any position.
    code = 'out = {"by_R0": {f"R0={r}": {"rmse": min(e, 10.0)} for r, e in zip(R0S, E)}}\n' + P
    (found,) = violations({"by_R0": {"R0=0.9": {"rmse": 10.0}}}, RMSE, code=code)
    assert found.kind == "clamped"
    code = 'out = {"runs": [{"rmse": min(r, 10.0)} for r in rs]}\n' + P
    (found,) = violations({"runs": [{"rmse": 10.0}]}, RMSE, code=code)
    assert found.kind == "clamped"
    code = 'out = {"test": {"rmse": min(r, 10.0)} if ok else None}\n' + P
    (found,) = violations({"test": {"rmse": 10.0}}, RMSE, code=code)
    assert found.kind == "clamped"


def test_the_path_can_be_given_as_tokens_or_a_string() -> None:
    code = 'out = {"test": {"rmse": min(x, 7.5)}}\n' + P
    assert direct_caps(code, "test.rmse") == {7.5}
    assert direct_caps(code, (("k", "test"), ("k", "rmse"))) == {7.5}
    assert direct_caps(code, "train.rmse") == set()
    assert direct_caps("not python (", "rmse") == set()
    assert direct_caps('out = {"rmse": r}\n', "rmse") == set(), "no cap at all"


def test_the_engine_sends_back_only_a_proven_cap() -> None:
    from core.engine import _assertion_violations, _assertion_warnings

    design = {"result_assertions": [{"path": "rmse", "min": 0, "max": 100}]}
    unproven = {"result_json": {"test": {"rmse": 10.0}}, "design": design,
                "code": 'out = to_native({"test": {"rmse": min(r, 10.0)}})\nprint(json.dumps(out))\n'}
    assert _assertion_violations(unproven) == [] and len(_assertion_warnings(unproven)) == 1
    proven = {**unproven, "code": 'out = {"test": {"rmse": min(r, 10.0)}}\nprint(json.dumps(out))\n'}
    assert len(_assertion_violations(proven)) == 1 and _assertion_warnings(proven) == []


def test_a_name_or_a_function_s_return_inside_the_printed_result_is_followed() -> None:
    # The inline alias: test_metrics is placed under "test" by the printed literal, so its cap is proven there.
    code = 'test_metrics = {"rmse": min(e, 10.0)}\nprint("RESULT_JSON:", json.dumps({"test": test_metrics}))\n'
    (found,) = violations({"test": {"rmse": 10.0}}, RMSE, code=code)
    assert found.kind == "clamped"
    # ... and a train literal placed under "train" is not a top-level rmse cap.
    code = 'train = {"rmse": min(t, 10.0)}\nprint(json.dumps({"train": train, "rmse": test_rmse}))\n'
    assert violations({"train": {"rmse": 3.0}, "rmse": 10.0}, RMSE, code=code) == []
    # A result returned by a function the script prints.
    code = 'def run():\n    return {"test": {"rmse": min(r, 10.0)}}\nprint("RESULT_JSON:", json.dumps(run()))\n'
    (found,) = violations({"test": {"rmse": 10.0}}, RMSE, code=code)
    assert found.kind == "clamped"
    # A name bound inside a function is looked up there, not in another function.
    code = ('def fit():\n    out = {"rmse": min(e, 10.0)}\n    return out\n'
            'def main():\n    out = {"rmse": final}\n    print(json.dumps(out))\n')
    assert _kinds(violations({"rmse": 10.0}, RMSE, code=code)) == ["clamped_unproven"], "fit's out is never printed"
    # An annotated binding and a conditional at the root.
    code = 'out: dict = {"rmse": min(r, 10.0)}\nprint(json.dumps(out))\n'
    assert _kinds(violations({"rmse": 10.0}, RMSE, code=code)) == ["clamped"]
