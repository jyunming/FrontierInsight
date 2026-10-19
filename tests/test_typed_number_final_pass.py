"""When no round is left, a whole-paper draft gets one edit of its sentences with numbers FI did not fill before any is
taken out; ``{{calc:EXPR}}`` lets that edit keep a number of the method FI computes from the plan's numbers.

The engine tests run write -> review -> (edit) -> review with a fake model on a real trial record of an earlier round, as
tests/test_paper_keys_wiring.py does."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from core import paper_facts, paper_values
from tests.test_paper_keys_wiring import KEY_SD, PROTOCOL, TYPED, _engine, _quest

LEVEL = "0.975"  # the two-sided level of the plan's 95% interval: not a number of the plan, but computed from one
CALC = "{{calc:1 - (1 - 0.95) / 2}}"


def _paper() -> str:
    return (f"# Parameter bias\n\n## Abstract\nWe study the bias.\n\n## Methods\nThe run took {{{{run_time}}}}. Intervals used "
            f"the t quantile at level {LEVEL}.\n\n## Results\nThe SD of the bias was {KEY_SD} at noise 0.01, and the mean "
            f"bias was {TYPED} in the main grid.\n\n## Discussion\nIt is small.\n")


def _run(eng: Any, patch_reply: str) -> tuple[list[tuple[str, str]], dict[str, Any]]:
    calls: list[tuple[str, str]] = []

    async def chat(prompt: str, *, node: Any = None) -> str:
        calls.append((str(node), prompt))
        if node == "write":
            return _paper()
        if node == "write.patch":
            return patch_reply
        return json.dumps({"verdict": "accept", "score": 5, "suggestions": [], "must_flag_hits": []})

    eng._chat = chat  # type: ignore[assignment]
    state: dict[str, Any] = {"topic": eng.config.topic, "title": "t", "iteration": 0,
                             "exec_result": {"returncode": 0, "duration_s": 3.0}}
    return calls, state


def test_a_whole_draft_in_the_last_round_is_edited_once_before_anything_is_taken_out(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    eng = _engine(root, max_iterations=1)
    calls, state = _run(eng, json.dumps([{"find": f"at level {LEVEL}", "replace": f"at level {CALC}"}]))
    state.update(asyncio.run(eng._node_write(state)))  # type: ignore[arg-type]
    assert state["paper_keys"] is True and state["draft_edited"] is False
    review = asyncio.run(eng._node_review(state))  # type: ignore[arg-type]
    paper = Path(state["paper_md"]).read_text(encoding="utf-8")
    # No round is left, but nothing is taken out of a whole-paper draft yet: its sentences are edited first.
    assert "[number removed]" not in paper and TYPED in paper and LEVEL in paper
    passes = review["review"]["typed_pass_hits"]
    assert any(TYPED in h for h in passes) and any(LEVEL in h for h in passes)
    state.update(review)
    assert eng._route_after_review(state) == "rewrite"  # type: ignore[arg-type]
    before = len(calls)
    state.update(asyncio.run(eng._node_write(state)))  # type: ignore[arg-type]
    edits = [(n, p) for n, p in calls[before:] if n.startswith("write")]
    assert [n for n, _p in edits] == ["write.patch"]  # the edit only: no whole paper, which would type new numbers
    prompt = edits[0][1]
    assert "FI'S RECORD OF THIS RUN" in prompt and "{{calc:EXPR}}" in prompt and TYPED in prompt
    assert state["draft_edited"] is True
    shown = paper_facts.without_marks(Path(state["paper_md"]).read_text(encoding="utf-8"))
    assert f"at level {LEVEL}" in shown  # FI computed it from the plan's 0.95
    review = asyncio.run(eng._node_review(state))  # type: ignore[arg-type]
    final = paper_facts.without_marks(Path(state["paper_md"]).read_text(encoding="utf-8"))
    # Known positive: the typed result the edit left in is taken out now; the computed level stays.
    assert TYPED not in final and "[number removed]" in final and f"at level {LEVEL}" in final
    assert "typed_pass_hits" not in review["review"]
    state.update(review)
    assert eng._route_after_review(state) != "rewrite"  # type: ignore[arg-type]
    removed = [r["number"] for r in paper_facts.kept_check(root)["removed_numbers"]]
    assert removed == [TYPED]


def test_a_draft_that_cannot_be_edited_is_kept_and_the_review_takes_the_numbers_out(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    eng = _engine(root, max_iterations=1)
    calls, state = _run(eng, "not edits at all")
    state.update(asyncio.run(eng._node_write(state)))  # type: ignore[arg-type]
    state.update(asyncio.run(eng._node_review(state)))  # type: ignore[arg-type]
    assert eng._route_after_review(state) == "rewrite"  # type: ignore[arg-type]
    before = len(calls)
    state.update(asyncio.run(eng._node_write(state)))  # type: ignore[arg-type]
    assert [n for n, _p in calls[before:] if n.startswith("write")] == ["write.patch"]
    assert state["draft_edited"] is True and TYPED in Path(state["paper_md"]).read_text(encoding="utf-8")
    review = asyncio.run(eng._node_review(state))  # type: ignore[arg-type]
    final = paper_facts.without_marks(Path(state["paper_md"]).read_text(encoding="utf-8"))
    assert TYPED not in final and LEVEL not in final and "[number removed]" in final
    assert "typed_pass_hits" not in review["review"]


def test_a_round_left_sends_the_numbers_back_as_before(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    eng = _engine(root, max_iterations=3)
    _calls, state = _run(eng, "[]")
    state.update(asyncio.run(eng._node_write(state)))  # type: ignore[arg-type]
    review = asyncio.run(eng._node_review(state))  # type: ignore[arg-type]
    assert "typed_pass_hits" not in review["review"]
    assert any(TYPED in h for h in review["review"]["must_flag_hits"])


def test_the_edit_goes_before_every_route_that_would_deliver_the_draft(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    eng = _engine(root, max_iterations=1, review_loop=False)
    hit = f"paper_record_mismatch: you typed {TYPED}, which is not a number of the plan or the topic. — “x”"
    state = {"iteration": 5, "review": {"verdict": "accept", "must_flag_hits": [hit], "typed_pass_hits": [hit]}}
    assert eng._route_after_review(state) == "rewrite"  # type: ignore[arg-type]
    state["review"].pop("typed_pass_hits")
    assert eng._route_after_review(state) in ("done", "human_feedback")  # type: ignore[arg-type]


def test_a_route_that_runs_the_experiment_again_or_answers_a_record_contradiction_comes_first(
        tmp_path: Path, monkeypatch: Any) -> None:
    from core import engine as engine_mod

    root = _quest(tmp_path)
    eng = _engine(root, max_iterations=2)
    hit = f"paper_record_mismatch: you typed {TYPED}, which is not a number of the plan or the topic. — “x”"
    monkeypatch.setattr(engine_mod, "_review_sends_the_experiment_back", lambda review, state: "k_hat")
    state = {"iteration": 1, "review": {"verdict": "accept", "must_flag_hits": [hit], "typed_pass_hits": [hit]}}
    assert eng._route_after_review(state) == "re_execute"  # type: ignore[arg-type]
    monkeypatch.setattr(engine_mod, "_review_sends_the_experiment_back", lambda review, state: "")
    record = "record_contradiction: the paper says the run met the limit; FI's record says it did not. — “y”"
    state = {"iteration": 9, "record_rewrites": 1,
             "review": {"verdict": "accept", "must_flag_hits": [record, hit], "typed_pass_hits": [hit]}}
    assert eng._route_after_review(state) == "rewrite"  # type: ignore[arg-type]


def test_the_write_after_a_record_contradiction_or_on_an_edited_draft_is_not_the_edit(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    eng = _engine(root, max_iterations=1)
    calls, state = _run(eng, json.dumps([{"find": "in the main grid", "replace": "in the grid"}]))
    state.update(asyncio.run(eng._node_write(state)))  # type: ignore[arg-type]
    state.update(asyncio.run(eng._node_review(state)))  # type: ignore[arg-type]
    passes = state["review"]["typed_pass_hits"]
    record = "record_contradiction: the paper says the run met the limit; FI's record says it did not. — “y”"
    logged: list[str] = []
    info = eng._log.info
    eng._log.info = lambda msg, *a, **k: (logged.append(msg % a if a else msg), info(msg, *a, **k))  # type: ignore[assignment]
    before = len(calls)
    out = asyncio.run(eng._node_write({**state, "review": {**state["review"], "must_flag_hits": [record, *passes]}}))
    assert "write" in [n for n, _p in calls[before:]]  # the whole paper, as the contradiction's rewrite asks for
    assert out["draft_edited"] is False
    assert not [m for m in logged if "numbers FI did not fill, once" in m]
    # Nor when a request the run could not carry out is to be told to the whole-paper writer.
    before = len(calls)
    out = asyncio.run(eng._node_write({**state, "layout_missed": ["a panel per noise level"]}))  # type: ignore[arg-type]
    assert "write" in [n for n, _p in calls[before:]] and out["draft_edited"] is False
    assert not [m for m in logged if "numbers FI did not fill, once" in m]
    # A review's hits left in the state are not the edit again on a draft that was itself edited.
    asyncio.run(eng._node_write({**state, "draft_edited": True}))  # type: ignore[arg-type]
    assert not [m for m in logged if "numbers FI did not fill, once" in m]


def test_only_hits_that_quote_their_sentence_are_edited(tmp_path: Path, monkeypatch: Any) -> None:
    root = _quest(tmp_path)
    eng = _engine(root, max_iterations=1)
    quoted = f"paper_record_mismatch: you typed {TYPED}, which is not a number of the plan or the topic. — “x”"
    rest = "paper_record_mismatch: a further 4 typed number(s) in other sentences are not numbers of the plan or the topic."
    monkeypatch.setattr(paper_facts, "typed_hits", lambda *a, **k: [quoted, rest])
    monkeypatch.setattr(paper_facts, "value_hits", lambda *a, **k: [])
    state = {"paper_keys": True, "iteration": 1}
    assert eng._typed_pass_hits("p", state) == [quoted]  # type: ignore[arg-type]
    monkeypatch.setattr(paper_facts, "typed_hits", lambda *a, **k: [rest])
    assert eng._typed_pass_hits("p", state) == []  # type: ignore[arg-type]


def test_a_draft_the_extension_reask_wrote_whole_is_not_an_edited_draft(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    eng = _engine(root, max_iterations=3)
    _calls, state = _run(eng, json.dumps([{"find": "in the main grid", "replace": "in the grid"}]))
    state.update(asyncio.run(eng._node_write(state)))  # type: ignore[arg-type]
    state.update(asyncio.run(eng._node_review(state)))  # type: ignore[arg-type]

    async def rewritten(_state: Any, _persona: str, markdown: str) -> tuple[str, list[str], bool]:
        return _paper(), [], True  # the whole-paper writer was asked again and its paper was kept

    eng._report_what_the_extension_added = rewritten  # type: ignore[assignment]
    out = asyncio.run(eng._node_write(state))  # type: ignore[arg-type]
    assert out["draft_edited"] is False


# --- {{calc:EXPR}} ------------------------------------------------------------------------------------------------------


def test_calc_computes_from_the_plans_numbers_and_refuses_a_number_of_its_own() -> None:
    plan = {"thresholds": {"alpha": 0.05}, "grid": {"m": [20, 200], "pi1": [0.05, 0.2]}}
    got = {e: paper_values.calc(e, plan, "a 95% interval").text for e in (
        "1 - 0.05 / 2", "ndtri(1 - 0.05 / 2)", "t_quantile(1 - 0.05 / 2, 3)", "(1 - 0.05) * 0.05",
        "floor(0.2 * 200 + 1/2)", "ndtri(1 - 0.05)")}
    assert got == {"1 - 0.05 / 2": "0.975", "ndtri(1 - 0.05 / 2)": "1.96", "t_quantile(1 - 0.05 / 2, 3)": "3.182",
                   "(1 - 0.05) * 0.05": "0.0475", "floor(0.2 * 200 + 1/2)": "40", "ndtri(1 - 0.05)": "1.645"}
    # A number the plan does not state, a derived level typed into the expression, a name, a value out of range.
    for expr, why in (("0.05 * 0.97", "0.97"), ("t_quantile(0.975, 3)", "0.975"), ("1.37 * 0.42", "1.37"),
                      ("floor(0.05 * 200 + 1/2) * 1.1", "1.1"), ("bias + 0.05", "bias"), ("ndtri(2 + 0.05)", "cannot"),
                      # spelled so a text search misses its number, or made of whole numbers alone, or too large
                      (".8473", "0.8473"), ("0.05 * .97", "0.97"), ("1_234 * 0.05", "1234"), ("0x1F * 0.05", "31"),
                      ("6 / 7", "none of the plan"), ("2", "none of the plan"),
                      ("1e99999999999999999999999 * 0.05", "inf"), ("1 - 0.05 /", "cannot read")):
        value = paper_values.calc(expr, plan)
        assert value.text == paper_values.NOT_RECORDED and value.hit and why in value.hit.reason, (expr, value)
    # A negative plan number written with its sign counts; a plan number that is a whole number 0-10 alone does not.
    assert paper_values.calc("-0.5 * 2", {"constants": {"shift": -0.5}}).text == "-1"
    assert paper_values.calc("3 * 7 / 9", {"constants": {"k": 3}}).hit is not None
    # The topic's numbers may be used, as they may be typed; without a plan nothing is computed.
    assert paper_values.calc("1 - 0.1 / 2", plan, "intervals at level 0.1").text == "0.95"
    assert paper_values.calc("1 - 0.05 / 2", None).text == paper_values.NOT_RECORDED


def test_calc_goes_through_the_paper_as_a_key_and_a_refused_one_is_a_hit_and_taken_out(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    rec = paper_values.records_of(root, PROTOCOL)
    paper = ("## Methods\nIntervals used the level {{calc:1 - (1 - 0.95) / 2}}.\n\n"
             "## Results\nThe reference was {{calc:0.95 * 0.97}}.\n")
    out = paper_facts.render(paper, root, records=rec)
    assert "fi:fact-calc" not in out.text and not out.unfilled  # a key, not a fact of the run
    assert paper_facts.without_marks(out.text).startswith("## Methods\nIntervals used the level 0.975.")
    assert paper_facts.to_placeholders(out.text) == paper  # the writer gets back what it typed
    assert paper_values.unfilled_in(out.text) == ["calc:0.95 * 0.97"]
    paper_values.write_record(root, list(out.values))
    hits = paper_facts.value_hits(out.text, root)
    assert len(hits) == 1 and "{{calc:0.95 * 0.97}}" in hits[0] and "0.97" in hits[0] and "{{v:calc" not in hits[0]
    assert "The reference was" in hits[0]  # the sentence, so the passage editor finds it
    taken, dropped = paper_facts.take_out_unfilled(out.text)
    assert [d["key"] for d in dropped] == ["calc:0.95 * 0.97"]
    assert "0.975" in paper_facts.without_marks(taken) and "not recorded" not in paper_facts.without_marks(taken)
    # Inside math the value is written bare, and the typed-number check never reads an expression's numbers.
    assert paper_facts.without_marks(paper_facts.render("In $t_{ {{calc:1 - (1 - 0.95) / 2}} }$ ok.", root,
                                                        records=rec).text) == "In $t_{ 0.975 }$ ok."
    assert not paper_facts.typed_hits(paper, PROTOCOL, "")
