from core import plan_steps as ps


def _card(**kw):
    c = {"id": "S1", "name": "decay", "does": "decays x", "inputs": {"k": "1/s", "t": "s"}, "output": "x",
         "uses": ["E1"], "source": "[1]"}
    c.update(kw)
    return c


def _proto(*cards):
    return {"model": {"equations": [{"id": "E1", "formula": "x = a", "role": "generates"}], "steps": list(cards)}}


def test_absent_steps_is_fine():
    assert ps.parse({}) == ([], [])
    assert ps.parse({"model": {"equations": []}}) == ([], [])


def test_good_card_with_example_computed_by_fi():
    ex = {"inputs": {"k": 2.0, "t": 0.5}, "expected_formula": "exp(-k*t)"}
    cards, problems = ps.parse(_proto(_card(example=ex)))
    assert problems == []
    assert cards[0]["example_state"] == "ok" and abs(cards[0]["expected"] - 0.36787944) < 1e-6
    assert not cards[0]["untested_by_example"]


def test_untested_by_example_and_relations():
    cards, problems = ps.parse(_proto(_card()))
    assert problems == [] and cards[0]["untested_by_example"]
    cards, _ = ps.parse(_proto(_card(relations=["doubles when t doubles"])))
    assert not cards[0]["untested_by_example"]


def test_problems():
    bad = {"id": "step1", "name": "", "does": "x", "inputs": {}, "output": ""}
    _cards, problems = ps.parse(_proto(bad, _card(), _card(), _card(id="S2", uses=["E9"]), "oops"))
    text = " | ".join(problems)
    for needle in ("needs an id like S1", "no `name`", "no `output`", "needs `inputs`", "used twice", "E9", "must be a mapping"):
        assert needle in text


def test_example_rules_shared_with_equations():
    unset = {"inputs": {"k": 1.0, "t": 1.0}, "expected_formula": "z*k"}
    _c, problems = ps.parse(_proto(_card(example=unset)))
    assert any("not among its `inputs`" in p for p in problems)
    mismatch = {"inputs": {"k": 1.0}, "expected_formula": "k"}
    c, problems = ps.parse(_proto(_card(example=mismatch)))
    assert any("must be the card's inputs" in p for p in problems) and c[0]["example_state"] == "unusable"
    c, problems = ps.parse(_proto(_card(example={"untestable": "no closed form"})))
    assert problems == [] and c[0]["example_state"] == "untestable"
    method = {"inputs": {"k": 1.0, "t": 1.0}, "expected_formula": "k*t", "method": "rk4"}
    assert any("tolerance" in p for p in ps.parse(_proto(_card(example=method)))[1])


def test_not_a_list():
    assert ps.parse({"model": {"steps": "x"}})[1]


def test_directive():
    assert "steps" in ps.DIRECTIVE_TEXT and "example" in ps.DIRECTIVE_TEXT
