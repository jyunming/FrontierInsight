"""The studied system's schematic (``core/setup_schematic.py``, ``core/setup_schematic_host.py``): drawn only when
``output.system_schematic`` is on; a model lays out shapes by the plan's part ids and types no word; FI writes every text from
the plan, runs the script in the quest's interpreter, judges what it drew against its own texts, allows one redraw, and
otherwise leaves the schematic out with one QUALITY.md line; one recorded, non-gating score of the image; numbered Figure 1
before FI's flow figure. The fake model here is a function, never a real one. Neutral made-up system: a block on a slab
under a lamp."""

from __future__ import annotations

import asyncio
import json
import math
import string
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from core import frozen_protocol, quality_report, setup_flow
from core import setup_schematic as S
from core import setup_schematic_host as H
from core.engine import _figure_list_for_prompt, _figure_numbers, _place_figures

ROOT = Path(__file__).parent.parent
PROMPTS = {n: string.Template((ROOT / "agents" / f"{n}.md").read_text(encoding="utf-8"))
           for n in ("setup_schematic", "setup_schematic_redraw", "setup_schematic_score")}

SYSTEM: dict[str, Any] = {
    "parts": [
        {"id": "slab", "name": "Slab", "kind": "part", "carries": ["slab_mm"]},
        {"id": "block", "name": "Block A", "kind": "part", "carries": ["mass_g", "shade"]},
        {"id": "lamp", "name": "Lamp", "kind": "source", "carries": ["power_W"]},
        {"id": "rim", "name": "Side rims", "kind": "boundary", "boundary": "periodic", "carries": []},
    ],
    "links": [{"from": "lamp", "to": "block", "what": "light"}, {"from": "block", "to": "slab", "what": "contact"}],
    "direction": {"what": "Gravity", "points": "down"},
}
PLAN: dict[str, Any] = {
    "grid": {"mass_g": [10, 20, 40], "power_W": [5, 10]},
    "constants": {"slab_mm": 12, "shade": 0.3},
    "runs_per_setting": 3,
    "labels": {"mass_g": "Block mass (g)", "power_W": "Lamp power (W)", "slab_mm": "Slab thickness (mm)",
               "warm_C": "Block warming (C)"},
    "metrics": [{"id": "warm_C", "label": "Block warming", "role": "result"}],
    "system": SYSTEM,
}
RESULT = "warming.png"
STATE = {"figures": [RESULT], "figure_records": {RESULT: {"axes": [{"ylabel": "Block warming", "series": [
    {"label": "5 W", "min": 1.0, "max": 3.0, "shows": "yes"}]}]}}}

# A clean drawing script of the kind the model writes: shapes by part id, labels by id, arrows by link number.
GOOD = """sch.canvas(100, 80)
sch.rect('slab', 10, 10, 80, 8, style='hatched')
sch.rect('block', 40, 18, 20, 14, style='solid')
sch.rect('lamp', 44, 62, 12, 6, style='solid')
sch.line('rim', [(10, 10), (10, 34)], dashed=True)
sch.line('rim', [(90, 10), (90, 34)], dashed=True)
sch.arrow(0, (50, 61), (50, 33), label_at=(58, 47))
sch.arrow(1, (36, 22), (36, 18.5), label_at=(26, 24))
sch.direction((88, 70), (88, 52), label_at=(80, 74))
sch.label('slab', 50, 5)
sch.label('block', 4, 50, ha='left')
sch.leader('block', (30, 46), (42, 32))
sch.label('lamp', 30, 70)
sch.label('rim', 92, 40, ha='right')
"""


def _fill() -> dict[str, Any]:
    f, why = S.fill(PLAN)
    assert f is not None, why
    return f


def _in_process(body: str, plant: Any = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run a script through FI's host in this process (a test only; a study runs it in the quest's interpreter), with a
    fault planted on the finished drawing by ``plant(sch)``."""
    sch = H.Schematic(_fill())
    exec(compile(body, "body.py", "exec"), {"__builtins__": H._safe_builtins(), "sch": sch, "math": math})
    if plant is not None:
        plant(sch)
    return H.facts(sch), list(sch._calls)


def _quest(tmp_path: Path, protocol: dict[str, Any] | None = PLAN) -> Path:
    root = tmp_path / "quest"
    (root / "figures").mkdir(parents=True)
    (root / ".fi").mkdir()
    (root / "figures" / RESULT).write_bytes(b"\x89PNG run figure")
    if protocol is not None:
        frozen_protocol.freeze(root, protocol, approved_by="test", source="test")
    return root


class Model:
    """A fake model: the scripts it answers in turn, and the score it gives an image. Counts every call."""

    def __init__(self, *bodies: str, score: str = '{"clarity": 6, "physical_plausibility": 5, "readability": 7, '
                                                    '"problems": ["a little crowded"]}', image_error: Exception | None = None):
        self.bodies = list(bodies)
        self.prompts: list[str] = []
        self.images = 0
        self.score = score
        self.image_error = image_error

    async def chat(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "```python\n" + self.bodies.pop(0) + "\n```"

    async def chat_image(self, prompt: str, png: bytes) -> str:
        self.images += 1
        assert png.startswith(b"\x89PNG")
        if self.image_error is not None:
            raise self.image_error
        return self.score

    @property
    def calls(self) -> int:
        return len(self.prompts) + self.images


async def _run_here(root: Path, argv: list[str]) -> tuple[int, str, str]:
    """The quest's interpreter, as a shared-interpreter study has it: this Python, from the quest folder."""
    done = subprocess.run([sys.executable, "-B", *argv], cwd=root, capture_output=True, text=True, timeout=180)
    return done.returncode, done.stdout, done.stderr


def _draw(root: Path, model: Model, *, on: bool = True, run_figures: list[str] | None = None) -> dict[str, Any] | None:
    return asyncio.run(S.draw(root, on=on, run_figures=run_figures or [RESULT], chat=model.chat,
                              chat_image=model.chat_image, run=lambda argv: _run_here(root, argv), prompts=PROMPTS))


# ---- FI's texts, from the plan by id -------------------------------------------------------------------------------------

def test_every_text_is_fis_from_the_plan_by_id() -> None:
    f = _fill()
    assert f["parts"]["slab"]["text"] == "Slab\nSlab thickness (mm): 12"
    assert f["parts"]["block"]["text"] == "Block A\nBlock mass (g): 10, 20, 40"
    assert f["parts"]["block"]["left_out"] == 1, "a carried id with no reader label is never shown by its code name"
    assert "shade" not in json.dumps(f) and "0.3" not in json.dumps(f)
    assert f["parts"]["rim"]["text"] == "Side rims (periodic)", "a boundary says the condition set there"
    assert f["links"]["link0"]["text"] == "light" and f["direction"] == {"text": "Gravity", "points": "down"}


def test_a_plan_with_no_usable_system_gives_no_texts_and_says_why() -> None:
    assert S.fill({**PLAN, "system": None})[0] is None
    f, why = S.fill({**PLAN, "system": {**SYSTEM, "parts": [{**SYSTEM["parts"][0], "carries": ["nope"]}]}})
    assert f is None and "cannot be used" in why


# ---- the guard: the script types no word ---------------------------------------------------------------------------------

@pytest.mark.parametrize("line, said", [
    ("import os", "Import"),
    ("sch.ax.text(1, 1, 'hello')", "`.ax`"),
    ("sch.label('Block A = 12 g', 1, 1)", "is not a part id"),
    ("x = f'{1}'", "JoinedStr"),
    ("open('notes.txt')", None),
    ("__import__('os')", "the name `__import__`"),
    ("sch._fill['parts']['block']['text'] = 'Block A = 99 g'", "`._fill`"),
])
def test_the_guard_refuses_anything_but_the_drawing_calls(line: str, said: str | None) -> None:
    found = S.guard(GOOD + line + "\n", _fill())
    if said is None:  # a builtin that is not given to the script fails when it runs; the guard need not name it
        return
    assert found and any(said in x for x in found), found


def test_a_clean_script_passes_the_guard_and_every_check() -> None:
    assert S.guard(GOOD, _fill()) == []
    facts, calls = _in_process(GOOD)
    assert S.judge(facts, calls, _fill()) == []


# ---- the planted faults of the trial: each one caught --------------------------------------------------------------------

def test_a_line_drawn_across_a_label_is_caught() -> None:
    body = GOOD + "sch.line('slab', [(0, 0), (100, 80)])\n"
    assert S.guard(body, _fill()) == [], "a legal call: only the check of the drawing can catch it"
    found = S.judge(*_in_process(body), _fill())
    assert any("is crossed by a line or an outline of `slab`" in x for x in found), found


def test_a_white_label_is_caught() -> None:
    def white(sch: H.Schematic) -> None:
        next(t for t in sch._ax.texts if t.get_gid() == "fi:block").set_color("white")

    found = S.judge(*_in_process(GOOD, white), _fill())
    assert "the label of `block` is drawn in a colour too pale to read" in found


def test_an_untagged_text_is_caught() -> None:
    found = S.judge(*_in_process(GOOD, lambda sch: sch._ax.text(5, 75, "Block A = 12 g")), _fill())
    assert any(x.startswith("a text not placed by sch.label") for x in found), found


def test_a_wrong_value_under_a_valid_tag_is_caught() -> None:
    def wrong(sch: H.Schematic) -> None:  # the host's own copy of the texts changed: FI judges by its own
        sch._fill["parts"]["slab"]["text"] = "Slab\nSlab thickness (mm): 99"
        sch.label("slab", 50, 76)

    found = S.judge(*_in_process(GOOD, wrong), _fill())
    assert "the text drawn for `slab` is not FI's text for it" in found


def test_a_part_left_out_a_link_without_arrow_a_label_twice_and_a_wrong_direction_are_caught() -> None:
    body = (GOOD.replace("sch.rect('lamp', 44, 62, 12, 6, style='solid')\n", "")
            .replace("sch.arrow(1, (36, 22), (36, 18.5), label_at=(26, 24))\n", "")
            .replace("sch.direction((88, 70), (88, 52)", "sch.direction((70, 70), (95, 70)")
            + "sch.label('slab', 50, 40)\n")
    found = S.judge(*_in_process(body), _fill())
    assert "part `lamp` is not drawn (no shape of it)" in found
    assert any(x.startswith("link 1 (block -> slab) is not drawn as an arrow") for x in found)
    assert "part `slab` is labelled 2 times; label it once" in found
    assert "the direction arrow does not point down" in found


def test_overlap_off_canvas_and_small_type_are_caught() -> None:
    def small(sch: H.Schematic) -> None:
        next(t for t in sch._ax.texts if t.get_gid() == "fi:lamp").set_fontsize(4)

    body = GOOD.replace("sch.label('rim', 92, 40, ha='right')", "sch.label('rim', 50, 5)") + \
        "sch.rect('slab', 95, 10, 30, 4)\n"
    found = S.judge(*_in_process(body, small), _fill())
    assert "the labels of `slab` and `rim` overlap" in found
    assert "a shape of `slab` runs off the drawing" in found
    assert any(x.startswith("the label of `lamp` is 4.0 pt") for x in found)


# ---- the analysis stage's call -------------------------------------------------------------------------------------------

def test_off_by_default_nothing_is_drawn_and_no_model_is_called(tmp_path: Path) -> None:
    from core.config import OutputConfig

    assert OutputConfig().system_schematic is False
    root = _quest(tmp_path)
    model = Model(GOOD)
    assert _draw(root, model, on=False) is None
    assert model.calls == 0 and not (root / ".fi" / S.RECORD).exists() and not (root / ".fi" / S.WORK).exists()
    assert setup_flow.figures(root) == [] and S.quality_line(root) is None


def test_a_drawing_that_passes_is_kept_scored_once_and_not_gated(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    model = Model(GOOD)
    record = _draw(root, model)
    assert record["drawn"] and model.calls == 2, "the drawing and one reading of the image"
    assert (root / ".fi" / S.FILE).is_file() and not (root / "figures" / S.FILE).exists()
    calls = json.loads((root / ".fi" / "figure_records" / S.CALLS_RECORD).read_text(encoding="utf-8"))
    assert calls["kind"] == "setup_schematic" and calls["calls"][0]["call"] == "canvas"
    assert record["vision"]["gating"] is False and record["vision"]["score"] == 6.0 and "threshold" in record["vision"]["note"]
    assert record["checks"]["passed"] and record["calls_made"] == 2
    assert "Block A" in model.prompts[0] and "Block mass (g): 10, 20, 40" in model.prompts[0]
    assert "pointing down" in model.prompts[0]


def test_a_failed_check_is_drawn_once_more_with_the_findings_named(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    model = Model(GOOD + "sch.label('slab', 50, 40)\n", GOOD)
    record = _draw(root, model)
    assert record["drawn"] and model.calls == 3
    assert "part `slab` is labelled 2 times; label it once" in model.prompts[1]
    assert [a["passed"] for a in record["attempts"]] == [False, True]


def test_failing_twice_leaves_it_out_with_one_quality_line_and_never_raises(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    bad = GOOD.replace("sch.label('lamp', 30, 70)\n", "")
    model = Model(bad, bad)
    record = _draw(root, model)
    assert not record["drawn"] and model.calls == 2, "no score of a drawing FI left out"
    assert S.shown(root) is None and setup_flow.figures(root) == []
    line = S.quality_line(root)
    assert line and line.startswith("No schematic of the studied system in the paper: FI's check of the drawing failed 2")
    rows = quality_report._paper_problems(quality_report._Quest(root))
    assert [r["found_by"] for r in rows] == ["FI's schematic of the studied system"]


def test_an_analysis_that_runs_again_calls_no_model_for_the_same_system(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    first = Model(GOOD)
    _draw(root, first)
    again = Model()
    record = _draw(root, again)
    assert again.calls == 0 and record["drawn"] and S.shown(root) == S.FILE
    # a failure is kept too: a new run of the analysis does not spend calls on the same system
    root2 = _quest(tmp_path / "b")
    bad = GOOD.replace("sch.label('lamp', 30, 70)\n", "")
    _draw(root2, Model(bad, bad))
    again = Model()
    assert not _draw(root2, again)["drawn"] and again.calls == 0


class Stopped(Model):
    """A model whose call ``stop_at`` (1 is the first) never returns: the study is stopped while it is in flight."""

    def __init__(self, *bodies: str, stop_at: int, **kw: Any):
        super().__init__(*bodies, **kw)
        self.stop_at = stop_at

    async def chat(self, prompt: str) -> str:
        if self.calls + 1 == self.stop_at:
            self.prompts.append(prompt)
            raise asyncio.CancelledError
        return await super().chat(prompt)

    async def chat_image(self, prompt: str, png: bytes) -> str:
        if self.calls + 1 == self.stop_at:
            self.images += 1
            raise asyncio.CancelledError
        return await super().chat_image(prompt, png)


def test_a_study_stopped_mid_call_goes_on_with_the_calls_it_has_left_never_more_than_3(tmp_path: Path) -> None:
    # stopped during the first drawing: that call is counted, the drawing and its score are still made after the restart
    root = _quest(tmp_path)
    with pytest.raises(asyncio.CancelledError):
        _draw(root, Stopped(GOOD, stop_at=1))
    on_disk = json.loads((root / ".fi" / S.RECORD).read_text(encoding="utf-8"))
    assert on_disk["calls_made"] == 1 and on_disk["done"] is False and S.shown(root) is None
    again = Model(GOOD)
    record = _draw(root, again)
    assert record["drawn"] and record["done"] and again.calls == 2 and record["calls_made"] == 3
    assert _draw(root, Model())["calls_made"] == 3, "a finished outcome makes no call"
    # stopped during the redraw: only the score's call is left, so nothing more is drawn
    root2 = _quest(tmp_path / "b")
    with pytest.raises(asyncio.CancelledError):
        _draw(root2, Stopped(GOOD + "sch.label('slab', 50, 40)\n", stop_at=2))
    again = Model(GOOD)
    record = _draw(root2, again)
    assert not record["drawn"] and record["done"] and again.calls == 0 and record["calls_made"] == 2
    assert "no call was left" in record["why"]
    # stopped while the drawing was read: never shown while unfinished; after the restart the drawing that passed is kept,
    # unscored, with no further call
    root3 = _quest(tmp_path / "c")
    with pytest.raises(asyncio.CancelledError):
        _draw(root3, Stopped(GOOD, stop_at=2))
    assert S.shown(root3) is None
    again = Model(GOOD)
    record = _draw(root3, again)
    assert again.calls == 0 and record["calls_made"] == 2 and record["drawn"] and record["done"]
    assert record["vision"]["scored"] is False and record["vision"]["gating"] is False
    assert S.shown(root3) == S.FILE and (root3 / ".fi" / "figure_records" / S.CALLS_RECORD).is_file()
    assert _draw(root3, Model())["drawn"], "finished: reused with no call"


def test_a_newer_fi_check_or_prompt_draws_again(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    f = _fill()
    key = S.key_of(f, PROMPTS)
    changed = {**PROMPTS, "setup_schematic": string.Template(PROMPTS["setup_schematic"].template + "\nOne more rule.")}
    assert S.key_of(f, changed) != key
    newer = tmp_path / "setup_schematic.py"
    newer.write_text(Path(S.__file__).read_text(encoding="utf-8") + "\n# a changed check\n", encoding="utf-8")
    monkeypatch.setattr(S, "__file__", str(newer))
    assert S.key_of(f, PROMPTS) != key
    monkeypatch.undo()
    root = _quest(tmp_path / "q")
    _draw(root, Model(GOOD))
    again = Model(GOOD)
    record = asyncio.run(S.draw(root, on=True, run_figures=[RESULT], chat=again.chat, chat_image=again.chat_image,
                                run=lambda argv: _run_here(root, argv), prompts=changed))
    assert again.calls == 2 and record["drawn"], "an outcome kept by an older FI is not reused"


def test_a_model_that_cannot_read_images_leaves_it_unscored_not_out(tmp_path: Path) -> None:
    from core.provider import ImageInputUnsupported

    root = _quest(tmp_path)
    record = _draw(root, Model(GOOD, image_error=ImageInputUnsupported("this transport sends text only")))
    assert record["drawn"] and record["vision"]["scored"] is False and "text only" in record["vision"]["why"]


def test_a_script_that_raises_is_a_finding_and_a_fault_of_fis_never_raises(tmp_path: Path,
                                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    root = _quest(tmp_path)
    model = Model("sch.canvas(100, 1000)\n", GOOD)
    record = _draw(root, model)
    assert record["drawn"] and "the code did not run (line 1): ValueError" in model.prompts[1]
    root2 = _quest(tmp_path / "b")

    def broken(*_a: Any, **_k: Any) -> Any:
        raise OSError("disk full")

    monkeypatch.setattr(S, "_attempt", broken)
    asyncio.run(S.draw_quietly(root2, on=True, run_figures=[], chat=Model(GOOD).chat, chat_image=Model().chat_image,
                               run=lambda argv: _run_here(root2, argv), prompts=PROMPTS))
    assert "FI could not draw it (OSError)" in (S.quality_line(root2) or "")


# ---- Figure 1 before the flow, everywhere ---------------------------------------------------------------------------------

DRAFT = ("# T\n\n## Introduction\n\nWhy it matters.\n\n## Method\n\nFigure 1 shows the studied system and Figure 2 how "
         "the study is set up.\n\n## Results\n\nFigure 3 shows the block warming.\n\n## References\n\n1. A cited paper.\n")


def test_the_schematic_is_figure_1_the_flow_figure_2_and_the_runs_figures_follow(tmp_path: Path) -> None:
    from generation._paper_figures import paper_figure_set

    root = _quest(tmp_path)
    _draw(root, Model(GOOD))
    setup_flow.draw(root, [RESULT])
    st = {**STATE, "quest_root": str(root)}
    assert _figure_numbers(st) == {S.FILE: 1, setup_flow.FILE: 2, RESULT: 3}
    listing = _figure_list_for_prompt(STATE, root=root).splitlines()
    assert listing[0].startswith(f"- Figure 1: figures/{S.FILE} — a schematic of the studied system")
    assert listing[1].startswith(f"- Figure 2: figures/{setup_flow.FILE}") and listing[2].startswith(f"- Figure 3: figures/{RESULT}")
    S.place_quietly(root)
    setup_flow.place_quietly(root)
    paper, placed, _ = _place_figures(DRAFT, STATE, root=root)
    assert placed == [S.FILE, setup_flow.FILE, RESULT]
    assert f"![**Figure 1.** {S.CAPTION}](figures/{S.FILE})" in paper
    files = sorted(p.name for p in (root / "figures").iterdir())
    shown, numbers = paper_figure_set(paper, files, dict(STATE), root)
    assert numbers[S.FILE] == 1 and numbers[setup_flow.FILE] == 2 and S.FILE in shown
    assert setup_flow.without(root, [RESULT, S.FILE, setup_flow.FILE]) == [RESULT]


def test_set_up_figures_the_prose_never_names_go_before_the_runs_first_figure(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    _draw(root, Model(GOOD))
    setup_flow.draw(root, [RESULT])
    draft = DRAFT.replace("Figure 1 shows the studied system and Figure 2 how the study is set up.", "We heat a block.")
    lines = _place_figures(draft, STATE, root=root)[0].split("\n")
    at = [next(i for i, ln in enumerate(lines) if f"figures/{n}" in ln) for n in (S.FILE, setup_flow.FILE, RESULT)]
    assert lines.index("## Results") < at[0] < at[1] < at[2] < lines.index("## References")


@pytest.mark.parametrize("method", [
    "Figure 1 shows the studied system and Figure 2 how the study is set up.",
    "Figure 2 shows how the study is set up and Figure 1 the studied system.",
])
def test_set_up_figures_named_after_a_run_figure_still_stand_first_and_numbering_by_place_agrees(
        tmp_path: Path, method: str) -> None:
    # The prose names the run's figure before either set-up figure: numbering by place (each figure beside its first
    # reference, numbered in the order it stands) meets the rule that the set-up figures open the paper's figures. The
    # set-up figures win the place (they stand before the run's first figure, the schematic before the flow), so the
    # numbers by place are the planned ones and every reader gives each figure the same number.
    from core.engine_base import FIGURE_NUMBERS_RECORD, _keep_figure_numbers, _number_figures_by_place
    from generation._paper_figures import paper_figure_set

    root = _quest(tmp_path)
    _draw(root, Model(GOOD))
    setup_flow.draw(root, [RESULT])
    S.place_quietly(root)
    setup_flow.place_quietly(root)
    draft = ("# T\n\n## Introduction\n\nFigure 3 shows the block warming in a sentence.\n\n## Method\n\n" + method +
             "\n\n## Results\n\nThe block warms.\n\n## References\n\n1. A cited paper.\n")
    paper, placed, _ = _place_figures(draft, STATE, root=root)
    lines = paper.split("\n")
    at = [next(i for i, ln in enumerate(lines) if f"figures/{n}" in ln) for n in (S.FILE, setup_flow.FILE, RESULT)]
    assert at[0] < at[1] < at[2] < lines.index("## Method"), "the set-up figures stand before the run's first figure"
    numbered, by_place = _number_figures_by_place(paper, STATE, root=root)
    assert by_place == {S.FILE: 1, setup_flow.FILE: 2, RESULT: 3}
    assert numbered == paper, "already standing in number order: nothing renumbered"
    assert "Figure 3 shows the block warming" in numbered and method in numbered
    _keep_figure_numbers(root, by_place)
    kept = json.loads((root / ".fi" / FIGURE_NUMBERS_RECORD).read_text(encoding="utf-8"))
    assert kept == by_place == _figure_numbers({**STATE, "quest_root": str(root)})
    listing = _figure_list_for_prompt(STATE, root=root).splitlines()
    assert [ln.split(" — ")[0] for ln in listing[:3]] == [
        f"- Figure 1: figures/{S.FILE}", f"- Figure 2: figures/{setup_flow.FILE}", f"- Figure 3: figures/{RESULT}"]
    files = sorted(p.name for p in (root / "figures").iterdir())
    assert paper_figure_set(numbered, files, dict(STATE), root)[1] == {S.FILE: 1, setup_flow.FILE: 2}
    assert _place_figures(numbered, STATE, root=root)[0] == numbered, "placing again gives the same paper"


def test_with_the_key_off_the_numbers_are_todays(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    _draw(root, Model(), on=False)
    setup_flow.draw(root, [RESULT])
    assert _figure_numbers({**STATE, "quest_root": str(root)}) == {setup_flow.FILE: 1, RESULT: 2}


def test_turning_the_key_off_takes_an_earlier_schematic_away(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    _draw(root, Model(GOOD))
    S.place_quietly(root)
    assert (root / "figures" / S.FILE).is_file()
    _draw(root, Model(), on=False)
    assert not (root / "figures" / S.FILE).exists() and not (root / ".fi" / S.RECORD).exists()
    assert S.shown(root) is None


def test_a_copy_into_figures_that_fails_leaves_no_schematic_anywhere(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _quest(tmp_path)
    _draw(root, Model(GOOD))
    setup_flow.draw(root, [RESULT])
    assert S.shown(root) == S.FILE

    def refuse(*_a: Any, **_k: Any) -> None:
        raise PermissionError("figures/ is read only")

    monkeypatch.setattr(setup_flow.shutil, "copyfile", refuse)
    S.place_quietly(root)
    monkeypatch.undo()
    setup_flow.place_quietly(root)
    assert S.shown(root) is None and "could not put it in figures/" in (S.quality_line(root) or "")
    assert _figure_numbers({**STATE, "quest_root": str(root)}) == {setup_flow.FILE: 1, RESULT: 2}


def test_a_run_figure_saved_under_the_schematics_name_is_the_runs(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    model = Model(GOOD)
    record = _draw(root, model, run_figures=[RESULT, S.FILE])
    assert not record["drawn"] and "a figure of its own" in record["why"] and model.calls == 0
    assert setup_flow.without(root, [S.FILE]) == [S.FILE]


# ---- the engine: the step's provider, the quest's interpreter ----------------------------------------------------------------

def test_the_analysis_node_draws_it_through_the_steps_own_node_names(tmp_path: Path) -> None:
    from tests.test_fi_places_figures import _engine

    root = _quest(tmp_path)
    eng = _engine(root)
    eng.config.output.system_schematic = True
    nodes: list[str] = []

    async def fake_chat(prompt, *, node=None, temperature=None):  # noqa: ANN001
        nodes.append(str(node))
        if node == S.NODE:
            return "```python\n" + GOOD + "```"
        return json.dumps({"summary": "s", "key_findings": [], "next_step": "publish"})

    async def fake_messages(messages, *, temperature=0.2, node=None):  # noqa: ANN001
        nodes.append(str(node))
        assert messages[0]["content"][1]["type"] in ("image", "image_url", "input_image")
        return '{"clarity": 5, "physical_plausibility": 5, "readability": 5, "problems": []}'

    eng._chat = fake_chat  # type: ignore[assignment]
    eng._chat_messages = fake_messages  # type: ignore[assignment]
    asyncio.run(eng._node_analyze({"result_json": {"warm": 0.4}, "exec_result": {"returncode": 0},  # type: ignore[arg-type]
                                   "design": {}, **STATE}))
    assert nodes[:2] == [S.NODE, S.SCORE_NODE], nodes
    record = json.loads((root / ".fi" / S.RECORD).read_text(encoding="utf-8"))
    assert record["drawn"] and record["vision"]["score"] == 5.0
    eng.config.output.system_schematic = False
    nodes.clear()
    asyncio.run(eng._node_analyze({"result_json": {"warm": 0.4}, "exec_result": {"returncode": 0},  # type: ignore[arg-type]
                                   "design": {}, **STATE}))
    assert S.NODE not in nodes and S.SCORE_NODE not in nodes and not (root / ".fi" / S.RECORD).exists()
