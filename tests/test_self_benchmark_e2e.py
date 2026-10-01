"""The self-benchmark's tools end to end, on fake-model quests (no model, no network): a clean research quest is
recorded, five errors are planted in it (R1, R2, N3, S1, L1), each is run against its control, and FI must catch every
one; then a planted run is replayed with no model at all, and a replay that runs out of recording says so. The report
is written and its numbers checked.

Slow: the recording and every copy run in their own virtual environment.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from core import retractions
from core.config import Config
from core.provider import LAST_CALL
from core.replay import CrossrefReplay, read_events
from dev.evaluation.bench import answers, cli, plant, report, runner, validity
from tests.test_engine_smoke import _FAKE_RESPONSES, _classify, _fake_response_for
from tests.test_research_acceptance import METRIC, _with_package
from tests.test_retractions import RETRACTED_ITEM
from tests.test_run_manifest import ANALYSIS_TRIAL, PROTOCOL, SIM_TRIAL, _reply

# The module's campaign (a recording and eleven runs) is set up by the first test: it gets the e2e files' longer limit.
pytestmark = [pytest.mark.slow, pytest.mark.timeout(3600)]

PROTO = {
    **PROTOCOL,
    "oracles": [
        {"name": "no_spread", "kind": "special_case", "check": "with R0 = 0 nobody is infected", "expected": 0.0,
         "tolerance": 0.01, "case": {"R0": 0.0}, "measure": "outbreak",
         "reference": "derivation: with R0 = 0 the infection rate beta = R0 * gamma is 0, so no susceptible is ever "
                      "infected and the outbreak indicator is 0 in every run"},
        # The check a missing factor changes: without it a planted factor would pass every check (N3).
        {"name": "size_at_one", "kind": "special_case", "check": "at R0 = 1 the final size is 100 plus a uniform draw",
         "expected": 100.5, "tolerance": 0.6, "case": {"R0": 1.0}, "measure": "final_size",
         "reference": "derivation: the model's final size is 100 R0 plus one uniform draw on [0, 1), so at R0 = 1 it "
                      "lies in [100, 101), within 0.5 of 100.5"},
    ],
    "metrics": [METRIC],
    "contrasts": [{"metric": "final_size", "a": "R0=0.9", "b": "R0=3.0"}],
    "criteria": [{"name": "nobody infected at R0 = 0", "oracle": "no_spread", "direction": "lower", "target": 0.01,
                  "tolerance": 0.001}],
}
SIM = SIM_TRIAL + '\n\ndef oracle():\n    return {"half": 0.5004}\n'
PAPER = ("# Final size of a toy epidemic\n\nWe ran 300 outbreaks per setting.\n\n## Results\n\n"
         "The headline score was 0.987 across the settings. ![r](figures/result.png)\n\n## References\n1. Smith 2020.\n")
CITING = ("# What earlier work says\n\nWe ran 300 outbreaks per setting.\n\n## Results\n\n"
          "The headline score was 0.987 across the settings. ![r](figures/result.png)\n\n"
          "Earlier work reports on the exposure and the outcome in children [1].\n\n## References\n1. The source.\n")
TASK = {
    "schema": answers.SCHEMA, "task": "Q99", "title": "the fake benchmark task", "kind": "simulation",
    "ask": "BENCHMARK. Declare the metric `final_size` and include R0 = 3.0 (axis `R0`) in the grid.",
    "answers": [{"metric": "final_size", "cell": {"R0": 3.0}, "statistic": "mean", "expected": 300.5,
                 "tolerance": 0.2, "tolerance_kind": "absolute", "source": "the fake model: 100 R0 + U(0, 1)"}],
    "errors": ["R1", "R2", "N3", "S1", "L1"],
}


def _claim_for(prompt: str) -> str:
    """The fake claim check grounds the citing sentence in source [1], quoting it."""
    src = plant.WAKEFIELD if "Ileal-lymphoid" in prompt else plant.MADSEN
    return json.dumps({"claims": [{"claim": "Earlier work reports on the exposure and the outcome in children",
                                   "basis": "citation", "citation_index": 1, "quote": src["content"][:60],
                                   "evidence": "the source"}], "summary": "one claim"})


def _fake(calls: list[str]):
    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        LAST_CALL.set({"provider": "openai", "model": kw.get("model") or "m-default", "reported": True})
        prompt = messages[-1]["content"]
        kind = _classify(prompt)
        calls.append(kind)
        if kind == "Experiment Design":
            body = json.loads(_FAKE_RESPONSES["design"])
            body["protocol"] = PROTO
            return json.dumps(body)
        if kind == "Implementation":
            return _with_package(_reply(SIM, ANALYSIS_TRIAL), prompt)
        if kind == "Writing":
            return CITING if ("Ileal-lymphoid" in prompt or "population-based study" in prompt) else PAPER
        if kind == "ClaimCheck" and ("Ileal-lymphoid" in prompt or "population-based study" in prompt):
            return _claim_for(prompt)
        if prompt.lstrip().startswith("**Persona:"):
            return _FAKE_RESPONSES["review"]
        return _fake_response_for(prompt)

    return fake_chat


def _base(root: Path, **knowledge: Any) -> Config:
    return Config.model_validate({
        "topic": "smoke topic for the self-benchmark", "title": "self-benchmark", "rigor_profile": "research",
        "provider": {"name": "openai", "node_models": {"review_panel.statistician": "m-other"}},
        "engine": {"max_iterations": 1, "review_loop": False, "execute_replicates": 3, "pilot_run": False},
        "execution": {"sandbox": "venv", "timeout_s": 300},
        "knowledge": {"enabled": False, **knowledge},
        "output": {"output_dir": str(root)},
    })


@pytest.fixture(scope="module")
def bench(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """The recorded clean quest, every planted and control run, and their scores: one module-wide campaign."""
    root = tmp_path_factory.mktemp("bench")
    adir = root / "answers"
    adir.mkdir()
    cfg_path = root / "task.yaml"
    cfg_path.write_text(yaml.safe_dump(_base(root).model_dump(mode="json")), encoding="utf-8")
    (adir / "fake.answer.json").write_text(json.dumps({**TASK, "config": str(cfg_path)}), encoding="utf-8")
    calls: list[str] = []
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(answers, "ANSWERS_DIR", adir)
        mp.setattr("core.engine.LLMClient.chat", _fake(calls))
        # L1's quests search the literature: no Axon store is opened and no network is reached.
        mp.setattr("core.knowledge._AXON_AVAILABLE", False)
        mp.setattr("core.knowledge._AXON_IMPORT_ERROR", "left out by the test", raising=False)

        async def no_search(self, *a, **kw):  # noqa: ANN001, ARG001
            return []

        async def same(self, docs, *a, **kw):  # noqa: ANN001, ARG001
            return docs

        mp.setattr("core.knowledge.Knowledge.asearch", no_search)
        mp.setattr("core.knowledge.Knowledge.fetch_full_text", same)
        mp.setattr("core.passages._embed_scores", lambda blobs, q: None)
        task = answers.load_task("Q99")

        clean = root / "clean"
        asyncio.run(cli.record(task, clean, base=_base(runner.paths(clean)[0])))

        runs = {
            "R1": cli.plant("R1", out=root / "R1", from_run=clean),
            "R2": cli.plant("R2", out=root / "R2", from_run=clean),
            "N3": cli.plant("N3", out=root / "N3", from_run=clean, file="code/simulate.py",
                            find="100.0 * cell", replace="50.0 * cell"),
            "S1": cli.plant("S1", out=root / "S1", from_run=clean, file="code/experiment.py",
                            find='m["values"]', replace='m["values"][::5]'),
            "control-claims": cli.plant("control", out=root / "control-claims", from_run=clean, step="claims"),
            "control-writing": cli.plant("control", out=root / "control-writing", from_run=clean, step="writing"),
            "control-run": cli.plant("control", out=root / "control-run", from_run=clean, step="run"),
            "L1": cli.plant("L1", out=root / "L1", task=task),
            "control-L1": cli.plant("L1", out=root / "control-L1", task=task, control=True),
        }
        filtered = {name: asyncio.run(validity.check(clean, root / name, task)) for name in ("N3", "S1")}
        # Crossref's answers, recorded once (the real shape) and replayed: no lookup reaches the network.
        recorded = CrossrefReplay(mode="replay", out_dir=root / "crossref", recorded={
            retractions.normalize_doi(plant.WAKEFIELD["doi"]): retractions._verdict(RETRACTED_ITEM),
            retractions.normalize_doi(plant.MADSEN["doi"]): {"status": "not_retracted", "why": "Crossref lists no retraction",
                                                             "notices": []},
        })
        lit = {"knowledge": {"enabled": True, "source_routing": "manual", "literature_screen": False,
                             "foundational_works": False, "web_search": False, "try_fetch_full_text": False}}

        async def campaign() -> list[dict[str, Any]]:
            jobs = [cli.run_planted(root / n, mode="partial") for n in runs if "L1" not in n]
            jobs += [cli.run_planted(root / n, mode="partial", settings=lit, base=_base(runner.paths(root / n)[0]),
                                     crossref=recorded) for n in ("L1", "control-L1")]
            return await asyncio.gather(*jobs)

        lookup = retractions.check_dois
        meta = asyncio.run(campaign())
        assert retractions.check_dois is lookup, "the retraction lookup is put back after the runs"
        # Mode A: the R2 run replayed from its own recording; nothing may reach the model.
        calls_before = len(calls)
        cli.plant("R2", out=root / "R2-replay", from_run=clean)
        asyncio.run(cli.run_planted(root / "R2-replay", mode="replay", recording=root / "R2"))
        replay_calls = len(calls) - calls_before
        # A replay whose recording lacks the reviewers' moderator: the run asks for a call it does not have.
        short = root / "short-recording"
        short.mkdir()
        lines = (runner.paths(root / "R2")[1] / "calls.jsonl").read_text(encoding="utf-8").splitlines()
        (short / "calls.jsonl").write_text("\n".join(l for l in lines if json.loads(l)["node"] != "review_moderator"),
                                           encoding="utf-8")
        cli.plant("R2", out=root / "R2-short", from_run=clean)
        asyncio.run(cli.run_planted(root / "R2-short", mode="replay", recording=short / "calls.jsonl"))

        outcomes = cli.score_dir(root)
        js, md = report.write(outcomes, root / "report")
    return {"root": root, "clean": clean, "runs": runs, "meta": meta, "filtered": filtered, "replay_calls": replay_calls,
            "outcomes": {o["run"]: o for o in outcomes}, "report": (js, md), "calls": calls}


def test_the_clean_recording_is_right_and_would_be_published(bench: dict[str, Any]) -> None:
    clean = bench["outcomes"]["clean"]
    assert clean["answer"]["correct"] is True, clean["answer"]
    assert clean["would_publish"] and clean["flagged"] == [], clean
    calls = (runner.paths(bench["clean"])[1] / "calls.jsonl").read_text(encoding="utf-8").splitlines()
    assert {json.loads(c)["source"] for c in calls} == {"real"} and len(calls) >= 15


def test_every_control_would_still_be_published(bench: dict[str, Any]) -> None:
    for name in ("control-claims", "control-writing", "control-run", "control-L1"):
        o = bench["outcomes"][name]
        assert o["role"] == "control" and o["would_publish"], (name, o["evidence_level"], o["flag_reasons"], o["stopped"])
    # L1's control rests on the source added in place of the retracted one: the plant was exercised.
    assert bench["outcomes"]["control-L1"]["cites_planted"] is True


def test_the_valid_error_filter_keeps_n3_and_s1(bench: dict[str, Any]) -> None:
    n3, s1 = bench["filtered"]["N3"], bench["filtered"]["S1"]
    assert n3["valid"] is True, n3
    (row,) = n3["answers"]
    assert row["clean_ok"] is True and abs(row["planted"] - 150.5) < 0.6, row
    assert s1["valid"] is True, s1


@pytest.mark.parametrize("error,gate", [("R1", "numbers"), ("R2", "numbers"), ("N3", "oracle"),
                                        ("S1", "run_record"), ("L1", "retractions")])
def test_fi_catches_each_planted_error(bench: dict[str, Any], error: str, gate: str) -> None:
    o = bench["outcomes"][error]
    assert o["role"] == "planted" and o["valid"] is True, o
    assert not o["would_publish"], (error, o["evidence_level"], o["flag_reasons"])
    assert o["first_gate"] == gate, (error, o["flagged"], o["flag_reasons"])


def test_a_full_replay_calls_no_model_and_reaches_the_same_outcome(bench: dict[str, Any]) -> None:
    assert bench["replay_calls"] == 0
    replayed, original = bench["outcomes"]["R2-replay"], bench["outcomes"]["R2"]
    assert replayed["divergences"] == 0 and replayed["mode"] == "replay"
    assert (replayed["would_publish"], replayed["first_gate"]) == (original["would_publish"], original["first_gate"])


def test_a_replay_that_runs_out_of_recording_says_which_step_asked(bench: dict[str, Any]) -> None:
    o = bench["outcomes"]["R2-short"]
    assert o["divergences"] >= 1
    # It did not follow its recording, so it is not counted.
    assert o["valid"] is None and "not in the recording" in o["not_counted_because"]
    events = read_events(runner.paths(bench["root"] / "R2-short")[1])
    assert any(e["event"] == "divergence" and e["node"] == "review_moderator" for e in events), events


def test_the_report_has_the_numbers(bench: dict[str, Any]) -> None:
    js, md = bench["report"]
    data = json.loads(js.read_text(encoding="utf-8"))
    total = data["summary"]["false_pass"]["total"]
    # R1, R2, N3, S1, L1 and the full replay of R2 (the replay short of a call is not counted): none let through.
    assert total["n"] == 6 and total["k"] == 0, total
    assert data["summary"]["false_pass"]["not_valid"] == 1
    assert data["summary"]["false_block"]["k"] == 0 and data["summary"]["false_block"]["n"] == 1
    text = md.read_text(encoding="utf-8")
    assert "Wrong results let through" in text and "0/6" in text
    assert "| N3 |" in text and "oracle" in text
