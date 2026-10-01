"""A number a person's refine asked for, which the extended script now computes, reaches the paper.

A real quest (2026-09-30): the refine asked for "the maximum absolute error over [0,1] for each method and step size".
The script was extended and RESULT_JSON gained ``max_abs_errors``, but the rewritten paper never reported it; the review
flagged ``user_feedback_unaddressed`` and the quest went back to the person for a whole review round. The write step now
checks, without a model call, that what the extension added is in the paper; if not, it asks the writer once more with
the missing numbers named, and if the paper still leaves them out it says so plainly at the review pause.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig
from core.engine import Engine
from core.number_provenance import unreported_results

NOTE = "Also report, for each method and each step size h, the maximum absolute error over [0,1], in a table."
ASKED = ["the maximum absolute error over [0,1] for each method and step size h"]
BEFORE = {"errors": {"euler": [0.01920, 0.009394], "rk4": [3.332e-7, 1.998e-8]}, "observed_order": 1.0179}
AFTER = {**BEFORE, "max_abs_errors": {"euler": [0.02714, 0.01321], "rk4": [4.816e-7, 2.911e-8]}}
# After the refine was answered and the extended script ran: the next write answers no refine.
STATE: dict[str, Any] = {
    "topic": "t", "title": "t", "iteration": 2,
    "human_feedback": {"action": "refine", "feedback": NOTE},
    "feedback_history": [{"iteration": 1, "text": NOTE}], "refine_written_for": 1,
}
T1_PAPER = "# P\n\nAt t=1 the errors were 0.01920 and 0.009394 (Euler) and 3.332e-7 and 1.998e-8 (RK4).\n"


def _engine(tmp_path: Path) -> Engine:
    eng = Engine(Config(
        topic="t", title="t", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(clarify_mode="off", review_loop=True, max_iterations=3),  # type: ignore[arg-type]
        execution=ExecutionConfig(sandbox="venv", timeout_s=60), knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "outputs"),
    ))
    (eng.quest_root / "paper").mkdir(parents=True, exist_ok=True)
    return eng


def _write(eng: Engine, replies: list[str], state: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    prompts: list[str] = []

    async def chat(prompt: str, *, node: str = "") -> str:
        prompts.append(prompt)
        return replies[min(len(prompts) - 1, len(replies) - 1)]

    eng._chat = chat  # type: ignore[method-assign]
    return asyncio.run(eng._node_write(state)), prompts  # type: ignore[arg-type]


def _after_extension(result_json: dict[str, Any]) -> dict[str, Any]:
    return {**STATE, "result_json": result_json,
            "extend_check": {"asked": ASKED, "before": ["errors.euler", "errors.rk4", "observed_order"]}}


def test_implement_records_what_the_results_held_before_the_extension(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    (eng.quest_root / "code").mkdir(parents=True)
    (eng.quest_root / "code" / "experiment.py").write_text("print('RESULT_JSON: {}')\n", encoding="utf-8")

    async def chat(prompt: str, *, node: str = "") -> str:
        return "```python\nprint('RESULT_JSON: {}')\n```"

    eng._chat = chat  # type: ignore[method-assign]
    out = asyncio.run(eng._node_implement(  # type: ignore[arg-type]
        {**STATE, "design": {}, "refine_extend": ASKED, "result_json": BEFORE}))
    assert out["extend_check"]["asked"] == ASKED
    assert set(out["extend_check"]["before"]) == {"errors.euler", "errors.rk4", "observed_order"}
    # A script written from the design (no extension) leaves nothing to check.
    plain = asyncio.run(eng._node_implement({**STATE, "design": {}, "result_json": BEFORE}))  # type: ignore[arg-type]
    assert plain["extend_check"] == {} and plain["extend_unreported"] == []


def test_a_number_the_first_write_leaves_out_is_asked_for_once_and_then_is_in_the_paper(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    second = T1_PAPER + "\n| h | max abs error (Euler) |\n|---|---|\n| 0.1 | 0.02714 |\n| 0.05 | 0.01321 |\n"
    out, prompts = _write(eng, [T1_PAPER, second], _after_extension(AFTER))
    assert len(prompts) == 2, "the writer was asked once more"
    assert "max_abs_errors" in prompts[1] and "maximum absolute error" in prompts[1]
    assert "0.02714" in prompts[1], "the directive names the numbers the run computed"
    assert "0.02714" in Path(out["paper_md"]).read_text(encoding="utf-8")
    assert out["extend_unreported"] == []


def test_no_extra_call_when_the_number_is_already_in_the_paper(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    present = T1_PAPER + "\nThe largest errors over [0,1] were 0.02714 and 0.01321 (Euler).\n"
    out, prompts = _write(eng, [present], _after_extension(AFTER))
    assert len(prompts) == 1 and out["extend_unreported"] == []
    # Nothing to check: no extension happened.
    out, prompts = _write(_engine(tmp_path / "b"), [T1_PAPER], {**STATE, "result_json": AFTER})
    assert len(prompts) == 1 and out["extend_unreported"] == []


def test_a_number_both_writes_leave_out_is_said_plainly_at_the_review_pause(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    out, prompts = _write(eng, [T1_PAPER, T1_PAPER], _after_extension(AFTER))
    assert len(prompts) == 2
    assert out["extend_unreported"] and "max_abs_errors" in out["extend_unreported"][0]
    run_log = (eng.fi_dir / "run.log").read_text(encoding="utf-8")
    assert "but is not in the paper" in run_log
    seen: dict[str, Any] = {}

    def pause(**kw: Any) -> dict[str, str]:
        seen.update(kw)
        return {"action": "accept", "answer": "yes"}

    eng._pause_for_human = pause  # type: ignore[method-assign]
    asyncio.run(eng._node_human_feedback(  # type: ignore[arg-type]
        {**STATE, **out, "review": {"verdict": "revise", "score": 4, "status": "ok"}}))
    said = " ".join(seen["steps"])
    assert "but is not in the paper" in said and "maximum absolute error" in said


def test_the_same_values_under_a_new_name_still_need_the_name_in_the_paper(tmp_path: Path) -> None:
    """The observed quest: every maximum equals the t=1 error, so the values are in the paper already (as the t=1
    table) and only the name says whether the paper reports the maximum at all."""
    same = {**BEFORE, "max_abs_errors": BEFORE["errors"]}
    out, prompts = _write(_engine(tmp_path), [T1_PAPER, T1_PAPER], _after_extension(same))
    assert len(prompts) == 2 and out["extend_unreported"]
    named = T1_PAPER + "\nThe maximum absolute error over [0,1] equals the error at t=1 for every method and h.\n"
    out, prompts = _write(_engine(tmp_path / "b"), [named], _after_extension(same))
    assert len(prompts) == 1 and out["extend_unreported"] == []


def test_the_check_survives_the_redraw_bounce(tmp_path: Path) -> None:
    """A write after the layout redraw writes the whole paper again: the check runs there too."""
    eng = _engine(tmp_path)
    state = _after_extension(AFTER)
    out, prompts = _write(eng, [T1_PAPER + "\nMaximum absolute error: 0.02714.\n"], state)
    assert len(prompts) == 1
    # The write after the redraw leaves the number out: it is asked for again there.
    again, prompts = _write(eng, [T1_PAPER, T1_PAPER + "\nMaximum absolute error: 0.02714.\n"], {**state, **out})
    assert len(prompts) == 2 and again["extend_unreported"] == []


def test_unreported_results_matches_at_the_paper_precision() -> None:
    before = ["errors.euler", "errors.rk4", "observed_order"]
    assert unreported_results("0.0271 and 0.0132", AFTER, before, ASKED) == []
    # 3.3e-7 rounds to 0.0 at three decimals; a lax match would call any tiny number a match.
    only_tiny = {**BEFORE, "max_abs_errors": {"rk4": [4.816e-7]}}
    assert unreported_results("an error of 9.999e-8 was seen", only_tiny, before, ASKED) == ["max_abs_errors"]
    assert unreported_results("an error of 4.82e-7 was seen", only_tiny, before, ASKED) == []


def test_the_observed_quests_result_names() -> None:
    """The names the real quest's results had: ``max`` is in an old name too (``max_invariant_rel_err_euler``), so the
    words that set the new result apart come from its nearest old name, ``euler_errors``."""
    errs = [0.0192, 0.009394]
    res = {"euler_errors": errs, "max_invariant_rel_err_euler": [7.6e-16], "euler_max_abs_errors": errs}
    before = ["euler_errors", "max_invariant_rel_err_euler"]
    t1_table = "| h | Euler error at t=1 |\n|---|---|\n| 0.1 | 0.0192 |\n| 0.05 | 0.009394 |\n\nAbsolute error is small."
    assert unreported_results(t1_table, res, before, ASKED) == ["euler_max_abs_errors"]
    said = t1_table + "\n\nFor every h the maximum absolute error over [0,1] is the error at t=1."
    assert unreported_results(said, res, before, ASKED) == []


def test_naming_it_without_reporting_it_does_not_count() -> None:
    same = {**BEFORE, "max_abs_errors": BEFORE["errors"]}
    before = ["errors.euler", "errors.rk4", "observed_order"]
    for text in (
        "The maximum absolute error was not computed; see the limitations.",  # says it was not done
        "The reader asked for the maximum absolute error over the interval for each method.",  # no number
        "We chose the maximal step above the absolute stability limit of 2.785.",  # "maximal", "above"
        "The reader asked for the maximum absolute error over [0,1] for each method.",  # restates the request
        "## 4.2 Maximum absolute error",  # a heading
        "The maximum absolute error is shown in Table 2.",  # a table number
        "Figure 3 plots the maximum absolute error for each method.",  # a figure number
        "The maximum absolute error is a common metric [12].",  # a citation
        "For every h the maximum absolute error is discussed qualitatively.",
    ):
        assert unreported_results(T1_PAPER + "\n" + text, same, before, ASKED) == ["max_abs_errors"], text


def test_a_number_appended_to_a_list_must_be_printed() -> None:
    """The plainest refine, "add n=64": the list the results had holds one more number."""
    before = {"runtime": 3}
    res = {"runtime": [1.25, 2.51, 5.03, 10.07]}
    assert unreported_results("Runtimes were 1.25, 2.51 and 5.03 s.", res, before, ["the runtime for n=64"]) == ["runtime"]
    assert unreported_results("At n=64 it took 10.07 s.", res, before, ["the runtime for n=64"]) == []
    # The settings list grows too: saying "we added n = 64" is not reporting the error at n=64.
    res = {"ns": [8, 16, 32, 64], "errors": [0.1, 0.05, 0.025, 0.0125]}
    before = {"ns": 3, "errors": 3}
    assert unreported_results("We added n = 64 to the grid.", res, before, ["add n=64"]) == ["errors"]
    assert unreported_results("At n = 64 the error is 0.0125.", res, before, ["add n=64"]) == []


def test_a_whole_number_is_not_found_in_a_section_or_citation() -> None:
    res = {"old": [1.5], "new_count": [3.0]}
    assert unreported_results("See Section 3 and [3].", res, {"old": 1}, ["the new count"]) == ["new_count"]


def test_the_writer_is_asked_again_once_per_extension(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    out, prompts = _write(eng, [T1_PAPER, T1_PAPER], _after_extension(AFTER))
    assert len(prompts) == 2 and out["extend_check"]["reasked"] is True
    # A later write (a review rewrite, the redraw bounce) says it again without another call.
    again, prompts = _write(eng, [T1_PAPER], {**_after_extension(AFTER), **out})
    assert len(prompts) == 1 and again["extend_unreported"]


def test_a_rerun_the_review_asked_for_keeps_the_check(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    (eng.quest_root / "code").mkdir(parents=True)
    (eng.quest_root / "code" / "experiment.py").write_text("print('RESULT_JSON: {}')\n", encoding="utf-8")

    prompts: list[str] = []

    async def chat(prompt: str, *, node: str = "") -> str:
        prompts.append(prompt)
        return "```python\nprint('RESULT_JSON: {}')\n```"

    eng._chat = chat  # type: ignore[method-assign]
    import core.engine as engine_mod

    orig = engine_mod._review_sends_the_experiment_back
    engine_mod._review_sends_the_experiment_back = lambda review, state: "max_abs_errors"  # type: ignore[assignment]
    try:
        out = asyncio.run(eng._node_implement({**_after_extension(AFTER), "design": {}}))  # type: ignore[arg-type]
    finally:
        engine_mod._review_sends_the_experiment_back = orig  # type: ignore[assignment]
    assert "extend_check" not in out, "the check stays as it was"
    assert "Keep what was added for it" in prompts[0] and ASKED[0] in prompts[0]


def test_a_paper_that_leaves_out_the_request_is_not_accepted_automatically() -> None:
    from core.engine import _auto_accepts

    snap = {"verdict": "accept", "must_flag_hits": [], "review_status": "ok"}
    assert _auto_accepts(snap)
    assert not _auto_accepts({**snap, "not_in_paper": ['Your request "x" was computed (k) but is not in the paper']})
