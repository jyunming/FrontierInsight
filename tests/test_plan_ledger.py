import json
from pathlib import Path

from core import plan_ledger as pl
from core import topic_requirements as tr

FIX = Path(__file__).parent / "fixtures" / "topic_requirements"
TOPICS = ["duv", "euv", "first_order_kinetics", "false_discovery"]


def _good_sim(i="X1"):
    return {"id": i, "replaces": "temperature", "because": "fixed", "effect": "small", "probe": "vary it"}


def _defn(**kw):
    d = {"metric": "m1", "defined_as": "mean", "source": "[1]", "quote": "the mean is", "differs": ""}
    d.update(kw)
    return d


def test_checklist_empty_and_no_results():
    assert tr.checklist_block(None) == ""
    assert tr.checklist_block({}) == ""
    req = {"factors": [{"name": "dose", "levels": [1.0, 2.0], "range": []}], "quantities": [{"name": "power"}],
           "formulas": [], "constraints": ["n >= 3"], "stated_results": ["power is 0.8"]}
    out = tr.checklist_block(req)
    assert "dose: 1, 2" in out and "report power" in out and "constraint: n >= 3" in out
    assert "0.8" not in out


def test_checklist_truncates():
    req = {"factors": [{"name": f"factor number {i}" * 3, "levels": [], "range": [0, 1]} for i in range(20)]}
    out = tr.checklist_block(req, limit=800)
    assert len(out) < 800 and "more)" in out


def test_checklist_replay_stable_and_small():
    for name in TOPICS:
        req = json.loads((FIX / f"{name}.json").read_text(encoding="utf-8"))
        out = tr.checklist_block(req)
        assert 0 < len(out) <= tr.CHECKLIST_LIMIT
        assert out == tr.checklist_block(json.loads(json.dumps(req)))
        for result in req.get("stated_results") or []:
            assert result not in out


def test_simplifications_ok_absent_and_problems():
    assert pl.parse_simplifications({}) == ([], [])
    items, problems = pl.parse_simplifications({"simplifications": [_good_sim(), _good_sim(),
                                                                    {"id": "X2", "replaces": "a"}, "bad"]})
    assert [i["id"] for i in items] == ["X1", "X2"]
    assert any("used twice" in p for p in problems)
    assert any("X2 has no `because`" in p for p in problems)
    assert any("must be a mapping" in p for p in problems)
    assert pl.parse_simplifications({"simplifications": "x"})[1]


def test_definitions_metric_without_definition_and_source_rules():
    proto = {"metrics": [{"id": "m1"}, {"id": "m2"}], "simplifications": [_good_sim()],
             "definitions": [_defn(), _defn(metric="m3", source="", quote=""), {"quantity": "q", "defined_as": "d"}]}
    items, problems = pl.parse_definitions(proto)
    assert "the metric m2 has no definition" in problems
    assert not any("m1 has no definition" in p for p in problems)
    assert sum("definition not taken from a source" in p for p in problems) == 2
    assert items[2]["metric"] == "q"


def test_differs_must_name_a_simplification():
    proto = {"metrics": [{"id": "m1"}], "simplifications": [_good_sim()], "definitions": [_defn(differs="X1")]}
    assert pl.parse_definitions(proto)[1] == []
    proto["definitions"] = [_defn(differs="X9")]
    assert any("X9" in p for p in pl.parse_definitions(proto)[1])


def test_quote_found_normalised():
    text = "In this paper,  the Mean\nis   defined as sum/n."
    assert pl.quote_found("the mean is defined as sum/n", text)
    assert pl.quote_found("THE MEAN IS DEFINED AS SUM N", [text])
    assert not pl.quote_found("the median", text)
    assert not pl.quote_found("", text)
    ds = [_defn(quote="the mean is defined"), _defn(metric="m2", quote="invented")]
    assert pl.unfound_quotes(ds, text) == ["m2"]


def test_directive_mentions_keys():
    for word in ("simplifications", "definitions", "quote", "differs"):
        assert word in pl.DIRECTIVE_TEXT


def test_uncovered_and_named():
    req = {"factors": [{"name": "temperature", "levels": [], "range": [], "main": True}],
           "quantities": [{"name": "yield", "main": True}], "formulas": []}
    proto = {"grid": {"pressure": [1, 2]}, "metrics": [{"id": "rate", "estimand": "reaction rate"}]}
    ids = [u["id"] for u in pl.uncovered(req, proto)]
    assert ids == ["F1", "Q1"]
    proto["simplifications"] = [_good_sim()]  # replaces "temperature"
    assert [u["id"] for u in pl.uncovered(req, proto)] == ["Q1"]
    proto["simplifications"].append({**_good_sim("X2"), "replaces": "Q1 yield is not measured"})
    assert pl.uncovered(req, proto) == []
    assert pl.uncovered({}, proto) == []


def test_ledger_roundtrip(tmp_path):
    rec = pl.ledger_record("abc", [_good_sim(), _good_sim("X2")], {"X1": ["R1"]})
    assert rec["items"][0]["status"] == "tested" and rec["items"][1]["status"] == pl.STATUS_UNTESTED
    path = tmp_path / ".fi" / "ledger.json"
    pl.write_ledger(path, rec)
    assert pl.read_ledger(path) == rec
    assert [i["id"] for i in pl.untested(rec)] == ["X2"]
    assert pl.read_ledger(tmp_path / "none.json") is None
    path.write_text("{}", encoding="utf-8")
    assert pl.read_ledger(path) is None
