"""A legend drawn over its data, or a figure title over a panel title, is sent back
for one redraw before the paper is written.

The plot-style recorder (``core/plot_style.py``) measures both on the saved
canvas and writes them to the figure's record as ``layout``. These tests cover
what the engine does with that: which findings count, the single repair round
that reaches ``execute_reflect`` with them, and everything that must NOT send a
run back -- clean figures, a figure the engine itself drew as the mean over the
seeds, a run that is broken some other way, and a second try.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig,
)
from core.engine import Engine, _figure_overlap_findings

TITLES = {
    "check": "title_overlap", "text": "Final Size Distribution for R0 = 1.5",
    "over": "N = 1000", "share": 0.52,
}
LEGEND = {
    "check": "legend_over_data", "axes_title": "Convergence to the limit",
    "legend": ["R0=0.9", "R0=1.5", "R0=3.0"], "line_px": 95.3, "points": 1,
}


def _state(**records: Any) -> dict[str, Any]:
    """A state after ``execute``: each keyword is a figure file (``a_png`` for
    ``a.png``) and its record."""
    files = {f"{k[:-4]}.png": v for k, v in records.items()}
    return {"figures": list(files), "figure_records": files}


# --- what counts as a finding ---------------------------------------------------


def test_each_overlap_is_one_sentence_naming_the_figure_and_what_overlaps() -> None:
    found = _figure_overlap_findings(_state(
        sizes_png={"axes": [], "layout": [TITLES]},
        limit_png={"axes": [], "layout": [LEGEND]},
    ))
    assert found == [
        'figures/sizes.png: the figure title "Final Size Distribution for R0 = 1.5" is drawn '
        'over the panel title "N = 1000"',
        'figures/limit.png: in the panel "Convergence to the limit", the legend (R0=0.9, R0=1.5, '
        "R0=3.0) is drawn over the lines or points it labels",
    ]


@pytest.mark.parametrize("record", [
    pytest.param({"axes": []}, id="an older record with no layout"),
    pytest.param({"axes": [], "layout": []}, id="measured and clean"),
    pytest.param({"axes": [], "layout": None}, id="not measured"),
    pytest.param({"axes": [], "layout": ["not a finding", {"check": "something_new"}]}, id="unknown findings"),
    pytest.param({"axes": [], "layout": [TITLES, LEGEND], "replicate_mean": {"n": 3}},
                 id="the engine drew it as the mean over the seeds"),
])
def test_nothing_is_reported_for_a_figure_that_needs_no_redraw(record: dict[str, Any]) -> None:
    assert _figure_overlap_findings(_state(a_png=record)) == []


def test_a_figure_that_is_not_in_the_run_is_not_reported() -> None:
    """The records of an earlier version of the script go with its figures."""
    state = _state(a_png={"axes": [], "layout": [LEGEND]})
    state["figures"] = ["b.png"]
    assert _figure_overlap_findings(state) == []


def test_once_the_redraw_was_asked_for_nothing_is_reported() -> None:
    state = _state(a_png={"axes": [], "layout": [TITLES]})
    assert len(_figure_overlap_findings(state)) == 1
    assert _figure_overlap_findings({**state, "figure_overlap_repaired": True}) == []


def test_a_figure_seed_0_alone_shows_is_still_reported() -> None:
    """The mean-over-the-seeds redraw owns a figure only when it drew it."""
    record = {"axes": [], "layout": [LEGEND], "single_seed": 0}
    assert len(_figure_overlap_findings(_state(a_png=record))) == 1


# --- the reflect node and its router ---------------------------------------------


def _engine(tmp_path: Path, *, max_iter: int = 3, replicates: int = 1) -> Engine:
    eng = Engine(Config(
        topic="figure overlap", title="figure overlap", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(
            max_iterations=1, review_loop=False, clarify_mode="off", ideate_reflect=False,
            exec_reflect_max_iterations=max_iter, execute_replicates=replicates,
        ),
        execution=ExecutionConfig(sandbox="venv", timeout_s=120),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "out"),
    ))
    eng.executor.install = AsyncMock(return_value=SimpleNamespace(returncode=0, stderr=""))  # type: ignore[method-assign]
    (eng.quest_root / "code").mkdir(parents=True, exist_ok=True)
    return eng


REDRAW = "import json\nprint('RESULT_JSON: ' + json.dumps({'slope': 1.0}))  # legend moved outside\n"


def _overlapping(**extra: Any) -> dict[str, Any]:
    """A run that succeeded, whose one figure has a legend over its data."""
    return {
        "exec_result": {"returncode": 0, "stdout_tail": "", "stderr_tail": "", "duration_s": 1.0},
        "result_json": {"slope": 1.0}, "design": {"hypothesis": "h"}, "code": "print('RESULT_JSON: {}')\n",
        "exec_reflect_iter": 0,
        **_state(limit_png={"axes": [], "layout": [LEGEND]}),
        **extra,
    }


def _ask(eng: Engine, answer: dict[str, Any] | str | Exception) -> AsyncMock:
    reply = answer if isinstance(answer, (str, Exception)) else json.dumps(answer)
    chat = AsyncMock(side_effect=reply if isinstance(reply, Exception) else None,
                     return_value=reply if not isinstance(reply, Exception) else None)
    eng._chat = chat  # type: ignore[method-assign]
    return chat


@pytest.mark.asyncio
async def test_a_run_whose_legend_covers_its_data_is_sent_back_once(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    chat = _ask(eng, {"code": REDRAW, "patch_summary": "legend outside the axes"})
    state = _overlapping()
    assert eng._route_after_execute_reflect(state) == "retry"

    patch = await eng._node_execute_reflect(state)  # type: ignore[arg-type]

    prompt = chat.await_args.args[0]
    # Which figure, what overlaps, what to change, and what not to touch.
    assert "FIGURE LAYOUT" in prompt and "the account of a crash above does not apply" in prompt
    assert 'figures/limit.png: in the panel "Convergence to the limit", the legend (R0=0.9, R0=1.5, R0=3.0)' in prompt
    assert "bbox_to_anchor" in prompt and "loc=\"best\" alone is not a fix" in prompt
    assert "Do NOT change any computation, data, parameter" in prompt and "RESULT_JSON" in prompt
    assert "IMPLAUSIBLE RESULT" not in prompt
    assert patch["figure_overlap_repaired"] is True
    assert patch["exec_patch_pending"] is True and patch["exec_reflect_iter"] == 1
    assert patch["code"] == REDRAW
    assert (eng.quest_root / "code" / "experiment.py").read_text(encoding="utf-8") == REDRAW
    # The redraw touches the script only: the results it leaves alone are not in the patch.
    assert not {"result_json", "result_json_replicates", "exec_result"} & set(patch)
    assert patch["exec_reflect_history"][-1]["patch_summary"] == "legend outside the axes"


@pytest.mark.asyncio
async def test_the_redraw_is_shown_the_whole_script(tmp_path: Path) -> None:
    """It answers with the whole script, so a script cut at the crash-repair
    limit would come back without its tail."""
    eng = _engine(tmp_path)
    chat = _ask(eng, {"code": REDRAW})
    script = "x = 1\n" * 3000 + "print('the last line')\n"
    assert len(script) > 8000

    await eng._node_execute_reflect(_overlapping(code=script))  # type: ignore[arg-type]

    assert "the last line" in chat.await_args.args[0]


@pytest.mark.asyncio
async def test_the_redraw_is_asked_for_once_however_the_figures_come_out(tmp_path: Path) -> None:
    """Drive the node and the router together: the first pass patches, the figures
    still overlap on the second, and the run goes on."""
    eng = _engine(tmp_path)
    chat = _ask(eng, {"code": REDRAW, "patch_summary": "moved the legend"})
    state = _overlapping()

    state.update(await eng._node_execute_reflect(state))  # type: ignore[arg-type]
    assert eng._route_after_execute_reflect(state) == "retry"  # the patch is run
    # What ``execute`` leaves behind: the same overlap, and the patch has run.
    state.update({"exec_patch_pending": False})
    assert await eng._node_execute_reflect(state) == {}  # type: ignore[arg-type]
    assert eng._route_after_execute_reflect(state) == "proceed"
    assert chat.await_count == 1


@pytest.mark.asyncio
async def test_a_run_with_clean_figures_is_left_alone(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    chat = _ask(eng, {"code": REDRAW})
    state = _overlapping(**_state(a_png={"axes": [], "layout": []}, b_png={"axes": []}))

    assert await eng._node_execute_reflect(state) == {}  # type: ignore[arg-type]
    assert eng._route_after_execute_reflect(state) == "proceed"
    assert chat.await_count == 0


@pytest.mark.asyncio
async def test_a_figure_drawn_as_the_mean_over_the_seeds_is_not_sent_back(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    chat = _ask(eng, {"code": REDRAW})
    state = _overlapping(**_state(limit_png={"axes": [], "layout": [LEGEND], "replicate_mean": {"n": 3}}))

    assert await eng._node_execute_reflect(state) == {}  # type: ignore[arg-type]
    assert eng._route_after_execute_reflect(state) == "proceed"
    assert chat.await_count == 0


@pytest.mark.asyncio
async def test_a_redraw_is_not_offered_when_no_attempt_would_be_left_to_repair_it(tmp_path: Path) -> None:
    """A rewrite can break a script that ran. On the last attempt that would leave
    no result, where the overlap left one."""
    eng = _engine(tmp_path, max_iter=3)
    chat = _ask(eng, {"code": REDRAW})
    for used, expected in ((0, "retry"), (1, "retry"), (2, "proceed")):
        state = _overlapping(exec_reflect_iter=used)
        assert eng._route_after_execute_reflect(state) == expected, used
    state = _overlapping(exec_reflect_iter=2)
    assert await eng._node_execute_reflect(state) == {}  # type: ignore[arg-type]
    assert chat.await_count == 0
    # And with a single attempt there is nothing to spare at all.
    assert _engine(tmp_path / "solo", max_iter=1)._route_after_execute_reflect(_overlapping()) == "proceed"


@pytest.mark.asyncio
async def test_a_run_that_is_broken_another_way_is_repaired_for_that_first(tmp_path: Path) -> None:
    """The figures are drawn again by any repair, and measured again after it, so
    the round is kept for a run whose numbers are settled."""
    eng = _engine(tmp_path)
    chat = _ask(eng, {"code": REDRAW})
    crashed = _overlapping(exec_result={"returncode": 1, "stdout_tail": "", "stderr_tail": "boom", "duration_s": 1.0},
                           result_json={})
    patch = await eng._node_execute_reflect(crashed)  # type: ignore[arg-type]
    prompt = chat.await_args.args[0]
    assert "FIGURE LAYOUT" not in prompt and "boom" in prompt
    assert "figure_overlap_repaired" not in patch

    out_of_range = _overlapping(design={"result_assertions": [{"path": "slope", "min": 0.0, "max": 0.5}]})
    patch = await eng._node_execute_reflect(out_of_range)  # type: ignore[arg-type]
    prompt = chat.await_args.args[0]
    assert "IMPLAUSIBLE RESULT" in prompt and "FIGURE LAYOUT" not in prompt
    assert "figure_overlap_repaired" not in patch


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", [
    pytest.param({"code": "", "give_up_reason": "cannot lay this out"}, id="the model gives up"),
    pytest.param({"code": ""}, id="an empty answer"),
    pytest.param("no json here at all", id="not json"),
    pytest.param({"code": "def broken(:\n    pass\n"}, id="a script that is not Python"),
    pytest.param(RuntimeError("provider down"), id="the call fails"),
])
async def test_a_redraw_that_gives_nothing_usable_leaves_the_run_as_it_was(
    tmp_path: Path, answer: Any,
) -> None:
    eng = _engine(tmp_path)
    original = "print('RESULT_JSON: {}')  # as written\n"
    (eng.quest_root / "code" / "experiment.py").write_text(original, encoding="utf-8")
    _ask(eng, answer)
    state = _overlapping()

    patch = await eng._node_execute_reflect(state)  # type: ignore[arg-type]

    # The round is spent, nothing is patched or spent from the repair budget, and
    # no give-up sentinel is set (it would switch off every later repair).
    assert patch == {"figure_overlap_repaired": True}
    state.update(patch)
    assert eng._route_after_execute_reflect(state) == "proceed"
    assert (eng.quest_root / "code" / "experiment.py").read_text(encoding="utf-8") == original


# --- the execute node, end to end ---------------------------------------------------

# One figure with a legend over its lines, and a result the redraw must not move.
COVERED = """\
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)
fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1], label="rising")
ax.plot([0, 1], [0.5, 0.5], label="level")
ax.legend(loc="center")
fig.savefig("figures/limit.png")
print("RESULT_JSON: " + json.dumps({"slope": 1.0}))
"""
OUTSIDE = COVERED.replace('ax.legend(loc="center")', 'ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))')
CLEAN = COVERED.replace('ax.legend(loc="center")', 'ax.legend(loc="upper left")').replace("[0.5, 0.5]", "[0.05, 0.05]")


async def _drive(eng: Engine, script: str, redraw: str | None, tmp_path: Path) -> tuple[dict[str, Any], AsyncMock, list[str]]:
    """Run the execute node and the reflect node as the graph does, running the
    real script under the real recorder, until the router says to go on."""
    ran: list[str] = []

    async def run(cmd: list[str], *, cwd: Path, timeout_s: float, env: dict[str, str] | None = None) -> Any:
        ran.append(Path(cmd[-1]).name)
        start = time.monotonic()
        done = subprocess.run([sys.executable, cmd[-1]], cwd=cwd, env=env,
                              capture_output=True, text=True, timeout=timeout_s)
        return SimpleNamespace(returncode=done.returncode, stdout=done.stdout, stderr=done.stderr,
                               duration_s=time.monotonic() - start, timed_out=False)

    eng.executor.execute = run  # type: ignore[method-assign]
    chat = _ask(eng, {"code": redraw or script, "patch_summary": "legend outside"})
    (eng.quest_root / "code" / "experiment.py").write_text(script, encoding="utf-8")
    state: dict[str, Any] = {"deps": [], "design": {"hypothesis": "h"}, "code": script}
    for _ in range(8):
        state.update(await eng._node_execute(state))  # type: ignore[arg-type]
        state.update(await eng._node_execute_reflect(state))  # type: ignore[arg-type]
        if eng._route_after_execute_reflect(state) == "proceed":
            return state, chat, ran
    pytest.fail("the repair loop never ended")


@pytest.mark.asyncio
async def test_a_legend_over_the_data_is_redrawn_once_and_the_result_is_untouched(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    eng = _engine(tmp_path)
    state, chat, ran = await _drive(eng, COVERED, OUTSIDE, tmp_path)

    assert chat.await_count == 1
    assert ran == ["experiment.py", "experiment.py"]  # the script, and the redrawn script
    assert state["figure_overlap_repaired"] is True
    assert state["result_json"] == {"slope": 1.0}
    assert state["figure_records"]["limit.png"]["layout"] == []
    assert (eng.quest_root / "code" / "experiment.py").read_text(encoding="utf-8") == OUTSIDE


@pytest.mark.asyncio
async def test_a_redraw_that_still_overlaps_is_not_asked_for_again(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    eng = _engine(tmp_path)
    state, chat, ran = await _drive(eng, COVERED, COVERED, tmp_path)

    assert chat.await_count == 1 and ran == ["experiment.py", "experiment.py"]
    assert [f["check"] for f in state["figure_records"]["limit.png"]["layout"]] == ["legend_over_data"]


@pytest.mark.asyncio
async def test_a_run_with_clean_figures_costs_nothing_extra(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    eng = _engine(tmp_path)
    state, chat, ran = await _drive(eng, CLEAN, None, tmp_path)

    assert chat.await_count == 0 and ran == ["experiment.py"]
    assert not state.get("figure_overlap_repaired")
    assert state["figure_records"]["limit.png"]["layout"] == []


# A line figure, which the seeds redraw as their mean, and a scatter, which they
# do not: both have a legend over their data.
SEEDED = """\
import json, os, random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

random.seed(int(os.environ.get("FI_REPLICATE_SEED", 0)))
os.makedirs("figures", exist_ok=True)
noise = [random.random() * 0.02 for _ in range(3)]
fig, ax = plt.subplots()
ax.plot([0, 0.5, 1], [0.0 + noise[0], 0.5 + noise[1], 1.0 + noise[2]], label="rising")
ax.plot([0, 0.5, 1], [0.5, 0.5, 0.5], label="level")
ax.legend(loc="center")
fig.savefig("figures/lines.png")
fig2, ax2 = plt.subplots()
ax2.scatter([0.1, 0.5, 0.9], [0.1, 0.5 + noise[0], 0.9], label="runs")
ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
ax2.legend(loc="center")
fig2.savefig("figures/points.png")
print("RESULT_JSON: " + json.dumps({"slope": 1.0 + noise[0]}))
"""


@pytest.mark.asyncio
async def test_only_a_figure_the_seeds_do_not_redraw_is_sent_back(tmp_path: Path) -> None:
    """With replicate seeds the line figure is drawn again by the engine as the
    mean, so a repair of the script would not change it; the scatter is the
    primary run's own, and is the one reported."""
    pytest.importorskip("matplotlib")
    eng = _engine(tmp_path, replicates=3)
    state, chat, ran = await _drive(eng, SEEDED, SEEDED.replace('ax2.legend(loc="center")', 'ax2.legend(loc="upper left")'), tmp_path)

    assert chat.await_count == 1
    prompt = chat.await_args.args[0]
    assert "figures/points.png: the legend (runs) is drawn over the lines or points it labels" in prompt
    assert "figures/lines.png: " not in prompt  # (the script itself names the file, without a colon)
    assert ran.count("replot_figures.py") == 2  # once after each pass: the lines are the engine's
    assert state["figure_overlap_repaired"] is True
