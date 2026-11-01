"""What the prompts ask of a figure: at most three panels in one row, each on a slide of its own.

The effect on a model is not measured here (nothing in these tests calls one); only the
tick-label floor of the slide check enforces it. These pin what each prompt says.
"""

from __future__ import annotations

import string
from pathlib import Path

import pytest

AGENTS = Path(__file__).resolve().parent.parent / "agents"


def _prompt(name: str) -> str:
    return (AGENTS / name).read_text(encoding="utf-8")


@pytest.mark.parametrize("name", ["implement.md", "implement_body.md"])
def test_the_implement_prompts_cap_a_figure_at_three_panels_side_by_side(name: str) -> None:
    text = _prompt(name)
    assert "at most 3 panels, side by side in one row" in text
    assert "`plt.subplots(1, n)` with n <= 3" in text
    # A wider need is several figures, one file each, not one bigger grid.
    assert "SEVERAL figures of at most 3 panels each, one file per figure" in text


def test_the_outline_and_the_design_plan_figures_of_at_most_three_panels() -> None:
    outline = _prompt("implement_outline.md")
    assert "at most 3 panels, side by side in one row" in outline
    assert "SEVERAL figures of at most 3 panels each" in outline
    design = _prompt("design.md")
    assert "at most 3 panels side by side in one row" in design and "figures_planned" in design


def test_the_slides_prompt_gives_every_figure_a_slide_of_its_own_and_the_discussion_the_next() -> None:
    text = _prompt("slides.md")
    assert "One figure per slide" in text and "ONE bold lead sentence" in text and "then the figure alone" in text
    assert "The discussion goes on the NEXT slide" in text
    # The old ways of sharing a slide with text are gone: a figure in a side pane, and the
    # sizes the theme ignores.
    assert "![bg right" not in text.replace("never use `![bg ...]`", "") and "right:40%" not in text
    assert "![w:800]" not in text and "![h:380]" not in text
    assert "never use `![bg ...]`" in text
    # A figure slide does not count toward the deck's length.
    assert "plus one slide of its own for each figure" in text


def test_the_slides_prompt_still_fills_in_as_a_template() -> None:
    """It is a string.Template: a stray ``$`` in the new text would break every deck."""
    filled = string.Template(_prompt("slides.md")).substitute(paper_md="PAPER", figure_list="FIGURES", claims_rule="", claims="")
    assert "PAPER" in filled and "FIGURES" in filled


# ---------------------------------------------------------------------------
# Slides take the paper's own figure list and the paper's rule for sentences about a figure


def _quest(tmp_path: Path, with_calls: bool):
    import json

    from core.engine import QuestArtifacts

    root = tmp_path / "quest"
    (root / "paper").mkdir(parents=True)
    (root / "figures").mkdir()
    (root / "figures" / "result.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    paper = root / "paper" / "paper.md"
    paper.write_text("# Toy Paper\n\nMethods.\n\n![**Figure 1.** The toy curve against its fit.](figures/result.png)\n",
                     encoding="utf-8")
    art = QuestArtifacts(quest_id="q", quest_root=root, paper_md=paper, figures_dir=root / "figures")
    if with_calls:
        records = root / ".fi" / "figure_records"
        records.mkdir(parents=True)
        (records / "result.calls.json").write_text(json.dumps({
            "version": 1, "fallback": False,
            "panels": [{"panel": 1, "raw": False, "raw_reason": None,
                        "x": {"id": "t", "kind": "setting", "label": "time", "scale": "linear"},
                        "y": {"what": "energy", "label": "energy", "scale": "linear"},
                        "series": [{"id": "energy", "label": "energy", "kind": "line", "step": False,
                                    "x": [0, 1], "y": [1.0, 2.0], "yerr": None, "merged_with": []}],
                        "bands": []}]}), encoding="utf-8")
        art.raw_state = {"figures": ["result.png"],
                         "figure_records": {"result.png": {"axes": [{"series": [{"label": "e", "shows": "line"}]}]}}}
    return art


_DECK = ("---\nmarp: true\n---\n\n# Toy\n\n---\n\n## Result\n\nEnergy drifts upward as Figure 1 shows. "
         "The method is simple.\n\n![](figures/result.png)\n")


async def _deck(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, with_calls: bool, prompts: list[str]) -> str:
    from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig
    from generation.slides import SlideGenerator

    art = _quest(tmp_path, with_calls)
    cfg = Config(
        topic="slides figure list", title="slides-figure-list", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60), knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(kinds=["slides"], output_dir=tmp_path / "outputs"),
    )

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompts.append(messages[0]["content"])
        return _DECK

    monkeypatch.setattr("core.provider.LLMClient.chat", fake_chat)
    monkeypatch.setattr("generation.slides.shutil.which", lambda _n: None)
    await SlideGenerator(cfg).generate(art, art.quest_root)
    return (art.quest_root / "slides.md").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_slide_text_pointing_at_figure_is_filtered_by_pointer_rule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    prompts: list[str] = []
    with caplog.at_level("WARNING", logger="frontier_insight.slides"):
        body = await _deck(tmp_path, monkeypatch, True, prompts)
    assert "Energy drifts upward" not in body
    assert "The method is simple." in body  # the known positive: a sentence naming no figure stays
    assert "![](figures/result.png)" in body
    assert "Energy drifts upward as Figure 1 shows" in caplog.text
    # the model is told the paper's number and caption of the figure
    assert "- figures/result.png: Figure 1. The toy curve against its fit." in prompts[0]


@pytest.mark.asyncio
async def test_legacy_quest_slides_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prompts: list[str] = []
    body = await _deck(tmp_path, monkeypatch, False, prompts)
    assert "Energy drifts upward as Figure 1 shows. The method is simple." in body
    assert "- figures/result.png" in prompts[0] and "Figure 1. The toy" not in prompts[0]


_FI_SENTENCE = "Panel 1 draws a rising line"
_FI_MARK = "<!-- fi:fact-figure-lines-result -->" + _FI_SENTENCE + ".<!-- /fi:fact-figure-lines-result -->"


async def _deck_from(tmp_path, monkeypatch, deck: str, with_calls: bool = True) -> str:
    """The deck the slides keep when the model answers ``deck`` for a paper that holds FI's own statement."""
    import generation.slides as slides_mod
    from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig

    art = _quest(tmp_path, with_calls)
    art.paper_md.write_text(
        "# Toy Paper" + chr(10) * 2 + _FI_MARK + " (Figure 1)" + chr(10) * 2
        + "![**Figure 1.** The toy curve against its fit.](figures/result.png)" + chr(10), encoding="utf-8")
    cfg = Config(
        topic="slides figure list", title="slides-figure-list", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60), knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(kinds=["slides"], output_dir=tmp_path / "outputs"),
    )

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        return deck

    monkeypatch.setattr("core.provider.LLMClient.chat", fake_chat)
    monkeypatch.setattr("generation.slides.shutil.which", lambda _n: None)
    await slides_mod.SlideGenerator(cfg).generate(art, art.quest_root)
    return (art.quest_root / "slides.md").read_text(encoding="utf-8")


def _deck_with(line: str) -> str:
    return chr(10).join(["---", "marp: true", "---", "", "# Toy", "", "---", "", "## Result", "", line, "",
                         "Beside it the method is simple.", "", "![](figures/result.png)", ""])


@pytest.mark.asyncio
async def test_a_verbatim_fi_sentence_stays_on_a_slide_and_a_paraphrase_goes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    kept = await _deck_from(tmp_path / "a", monkeypatch, _deck_with("**" + _FI_SENTENCE + " (Figure 1).**"))
    assert _FI_SENTENCE + " (Figure 1)." in kept
    assert "fi:fact" not in kept
    plain = await _deck_from(tmp_path / "b", monkeypatch, _deck_with(_FI_SENTENCE + " (Figure 1)."))
    assert _FI_SENTENCE in plain
    gone = await _deck_from(tmp_path / "c", monkeypatch, _deck_with("Panel 1 shows a line that rises (Figure 1)."))
    assert "line that rises" not in gone and "Beside it the method is simple." in gone
    extended = await _deck_from(tmp_path / "d", monkeypatch, _deck_with(_FI_SENTENCE + " and so energy is conserved (Figure 1)."))
    assert "conserved" not in extended


@pytest.mark.asyncio
async def test_a_legacy_quest_keeps_a_paraphrase_of_a_sentence_about_a_figure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = await _deck_from(tmp_path, monkeypatch, _deck_with("Panel 1 shows a line that rises (Figure 1)."), with_calls=False)
    assert "line that rises" in body
