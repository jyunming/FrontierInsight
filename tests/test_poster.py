"""Poster generator (``generation/poster.py``).

Covers:
- the model's reply, as blocks and in the old two-column LaTeX shape;
- plain text to LaTeX;
- citations and the reference band;
- the layout planner and the measured fit loop;
- engine choice (pdflatex, tectonic, XeLaTeX for CJK text) and the skip
  diagnostics.

Most tests fake the model, the compile and the measurements. The real
compiles run when pdflatex (and, for CJK text, XeLaTeX plus a CJK font)
is installed.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.config import (
    Config,
    EngineConfig,
    ExecutionConfig,
    KnowledgeConfig,
    OutputConfig,
    ProviderConfig,
)
from core.engine import QuestArtifacts
from core.provider import ResolvedEndpoint
from generation import _cjk
from generation import poster as poster_mod
from generation.poster import PosterGenerator, _Block, _Layout, _SHEETS

A1 = _SHEETS["a1_portrait"]
PT_PER_CM = 72 / 2.54

REFS = [
    {
        "n": 1, "title": "Geometric Numerical Integration", "authors": ["E. Hairer", "C. Lubich", "G. Wanner"],
        "year": 2006, "venue": "Springer", "doi": "10.1007/3-540-30666-8",
        "url": "https://doi.org/10.1007/3-540-30666-8",
    },
    {
        "n": 2, "title": "Computer experiments on classical fluids.", "authors": ["L. Verlet"],
        "year": 1967, "venue": "Physical Review", "doi": "", "url": "https://example.org/verlet",
    },
    {"n": 3, "title": "Uncited work", "authors": [], "year": 2020, "venue": "", "doi": "10.1/x"},
]
FURTHER = [
    {"label": "W1", "title": "Energy of a damped oscillator", "url": "https://www.physics.stackexchange.com/q/1", "site": ""},
]


def _make_config(tmp_path: Path, kinds: list[str], **output) -> Config:
    return Config(
        topic="poster generator unit test",
        title="poster-test",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False),
        execution=ExecutionConfig(sandbox="venv", timeout_s=10),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "out", kinds=kinds, **output),
    )


def _make_artifacts(tmp_path: Path, *, real_figure: bool = False) -> QuestArtifacts:
    quest_root = tmp_path / "quest"
    paper_dir = quest_root / "paper"
    paper_dir.mkdir(parents=True)
    paper_md = paper_dir / "paper.md"
    paper_md.write_text(
        "# Toy Paper\n\nMethods. Results show $y = x^2$ scaling.\n\n"
        "![**Figure 1.** The toy curve against its fit.](figures/result.png)\n",
        encoding="utf-8",
    )
    figures = quest_root / "figures"
    figures.mkdir()
    if real_figure:
        from PIL import Image, ImageDraw

        image = Image.new("RGB", (1200, 750), (250, 250, 250))
        ImageDraw.Draw(image).line([(80, 680), (1120, 80)], fill=(14, 110, 107), width=12)
        image.save(figures / "result.png")
    else:
        (figures / "result.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    return QuestArtifacts(
        quest_id="qtest",
        quest_root=quest_root,
        paper_md=paper_md,
        figures_dir=figures,
    )


def _patch_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_resolve(provider, supervisor):  # noqa: ANN001
        return ResolvedEndpoint(
            base_url="http://127.0.0.1:1/v1", model="x", api_key="not-needed"
        )

    monkeypatch.setattr("generation.poster.resolve_endpoint_async", fake_resolve)


def _reply(**overrides) -> dict:
    reply = {
        "headline": "Toy scaling holds across three decades",
        "blocks": [
            {"type": "heading", "text": "Background"},
            {"type": "text", "text": "We study scaling [1]."},
            {"type": "heading", "text": "Result"},
            {"type": "figure", "file": "figures/result.png", "caption": "The curve rises as $x^2$."},
            {"type": "bullets", "items": ["Slope two [2]", "No outliers"]},
            {"type": "heading", "text": "What it means"},
            {"type": "text", "text": "Scaling holds [W1]."},
        ],
    }
    reply.update(overrides)
    return reply


async def _generate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reply, *,
    paper: str | None = None, output: dict | None = None,
    refs=(), further=(), real_figure: bool = False,
):
    cfg = _make_config(tmp_path, ["poster"], **(output or {}))
    art = _make_artifacts(tmp_path, real_figure=real_figure)
    if paper is not None:
        art.paper_md.write_text(paper, encoding="utf-8")

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        return reply if isinstance(reply, str) else json.dumps(reply)

    _patch_endpoint(monkeypatch)
    monkeypatch.setattr("generation.poster.LLMClient.chat", fake_chat)
    monkeypatch.setattr(poster_mod, "build_references", lambda lit, audience=None: list(refs))
    monkeypatch.setattr(poster_mod, "build_further_reading", lambda lit, audience=None: list(further))
    result = await PosterGenerator(cfg).generate(art, art.quest_root)
    return result, art


def _no_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(poster_mod, "find_pdf_engine", lambda: None)


# ---------------------------------------------------------------------------
# The generator end to end, without a compile


@pytest.mark.asyncio
async def test_poster_skipped_when_kind_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When 'poster' is not in output.kinds, generator returns {} and no
    LLM call happens (fail-loud sentinel inside the patched chat)."""
    cfg = _make_config(tmp_path, kinds=["paper_md"])
    art = _make_artifacts(tmp_path)

    async def must_not_call(self, messages, **kw):  # noqa: ANN001
        raise AssertionError("LLM should not be called when poster kind is off")

    monkeypatch.setattr("generation.poster.LLMClient.chat", must_not_call)

    result = await PosterGenerator(cfg).generate(art, art.quest_root)
    assert result == {}


@pytest.mark.asyncio
async def test_poster_tex_carries_the_headline_title_author_line_and_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_engine(monkeypatch)
    result, _art = await _generate(
        tmp_path, monkeypatch, _reply(),
        output={"author": "Jane Chen", "affiliation": "R&D Lab", "url": "https://example.org/p?a=1&b=2#x_y"},
        refs=REFS, further=FURTHER,
    )
    tex = result["poster_tex"].read_text(encoding="utf-8")
    assert "Toy scaling holds across three decades" in tex
    assert "Toy Paper" in tex  # the paper's own title, under the headline
    assert r"\posterhead{Background}" in tex
    assert r"Jane Chen~\textperiodcentered~R\&D Lab" in tex
    assert r"\qrcode[height=6.5cm]{https://example.org/p?a=1\&b=2\#x\_y}" in tex
    assert r"\postercaption{1}{The curve rises as $x^2$.}" in tex
    assert r"\item{} Slope two [2]" in tex
    assert "${" not in tex
    assert "poster_pdf" not in result
    assert "no_latex_engine" in result["poster_pdf_skipped"].read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_poster_sheet_follows_output_poster_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_engine(monkeypatch)
    result, _art = await _generate(tmp_path, monkeypatch, _reply(), output={"poster_size": "landscape_48x36"})
    tex = result["poster_tex"].read_text(encoding="utf-8")
    assert "width=121.92,height=91.44" in tex
    assert tex.count(r"\begin{column}") == 3
    assert r"\fontsize{36pt}{45pt}" in tex


@pytest.mark.asyncio
async def test_poster_honors_node_models_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``provider.node_models["poster"]`` must reach the chat call."""
    cfg = _make_config(tmp_path, kinds=["poster"])
    cfg.provider.node_models = {"poster": "gpt-4o-mini"}
    art = _make_artifacts(tmp_path)
    seen: dict = {}

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        seen.update(kw)
        return json.dumps(_reply())

    _patch_endpoint(monkeypatch)
    monkeypatch.setattr("generation.poster.LLMClient.chat", fake_chat)
    _no_engine(monkeypatch)

    await PosterGenerator(cfg).generate(art, art.quest_root)

    assert seen["model"] == "gpt-4o-mini"
    assert seen["node"] == "poster"


@pytest.mark.asyncio
async def test_the_prompt_carries_the_sheet_budget_figures_and_citable_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _make_config(tmp_path, kinds=["poster"], poster_size="a0_portrait")
    art = _make_artifacts(tmp_path)
    prompts: list[str] = []

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompts.append(messages[0]["content"])
        return json.dumps(_reply())

    _patch_endpoint(monkeypatch)
    monkeypatch.setattr("generation.poster.LLMClient.chat", fake_chat)
    monkeypatch.setattr(poster_mod, "build_references", lambda lit, audience=None: list(REFS))
    monkeypatch.setattr(poster_mod, "build_further_reading", lambda lit, audience=None: list(FURTHER))
    _no_engine(monkeypatch)
    await PosterGenerator(cfg).generate(art, art.quest_root)

    (prompt,) = prompts
    assert "A0 portrait" in prompt and "2 columns" in prompt and "380 words" in prompt
    assert "- figures/result.png: The toy curve against its fit." in prompt
    assert "[1] E. Hairer et al. (2006). Geometric Numerical Integration." in prompt
    assert "[W1] Energy of a damped oscillator. physics.stackexchange.com." in prompt
    assert "$E = mc^2$" in prompt  # the file's doubled dollars reach the model single


@pytest.mark.asyncio
async def test_the_headline_is_the_finding_with_the_paper_title_under_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_engine(monkeypatch)
    result, _art = await _generate(tmp_path, monkeypatch, _reply())
    tex = result["poster_tex"].read_text(encoding="utf-8")
    headline = tex.index("Toy scaling holds across three decades")
    assert headline < tex.index("Toy Paper")


@pytest.mark.parametrize(
    "paper, reply, headline, subtitle",
    [
        ("# Toy Paper\n\nBody.\n", {"blocks": []}, "Toy Paper", None),
        ("# Toy Paper\n\nBody.\n", {"headline": " ".join(["word"] * 25), "blocks": []}, "Toy Paper", None),
        ("Methods only, no heading.\n", {"title": "Model Title", "blocks": []}, "Model Title", None),
        ("Methods only, no heading.\n", "not json", "Untitled", None),
        ("---\n# generated\nauthor: x\n---\n# Real Title\n\nBody.\n", {"headline": "A finding", "blocks": []}, "A finding", "Real Title"),
    ],
)
@pytest.mark.asyncio
async def test_headline_fallbacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, paper, reply, headline, subtitle,
) -> None:
    _no_engine(monkeypatch)
    result, _art = await _generate(tmp_path, monkeypatch, reply, paper=paper)
    tex = result["poster_tex"].read_text(encoding="utf-8")
    title = re.search(r"\\fontsize\{80pt\}\{86\.4pt\}\\selectfont\\color\{fiink\}(.*?)\\par", tex)
    assert title and title.group(1) == headline
    if subtitle:
        assert r"\color{fimuted}" + subtitle + r"\par" in tex
    else:
        assert r"\color{fimuted}" not in tex.split(r"\begin{document}")[0].split("headline}{%")[1].split("footline")[0]


@pytest.mark.asyncio
async def test_inline_latex_math_and_currency_survive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_engine(monkeypatch)
    reply = _reply()
    reply["blocks"][1]["text"] = r"Inline math $\alpha + \beta = \gamma$ and $f(x) = \int_0^1 g(t)\,dt$; cost is $5 and value $v$."
    result, _art = await _generate(tmp_path, monkeypatch, reply)
    tex = result["poster_tex"].read_text(encoding="utf-8")
    assert r"$\alpha + \beta = \gamma$" in tex
    assert r"$f(x) = \int_0^1 g(t)\,dt$" in tex
    assert r"cost is \$5 and value $v$" in tex


@pytest.mark.asyncio
async def test_poster_falls_back_when_llm_returns_garbage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reply that is not JSON still writes poster.tex, under the paper's
    own title, with the paper's figures."""
    _no_engine(monkeypatch)
    result, _art = await _generate(tmp_path, monkeypatch, "not json at all, no braces here")
    tex = result["poster_tex"].read_text(encoding="utf-8")
    assert "Toy Paper" in tex
    assert r"\postercaption{1}{The toy curve against its fit.}" in tex


# ---------------------------------------------------------------------------
# Citations and the reference band


@pytest.mark.asyncio
async def test_the_band_lists_only_cited_sources_in_order_without_urls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_engine(monkeypatch)
    reply = _reply()
    reply["blocks"][1]["text"] = "We study scaling [2, 9] and more [1]."
    result, _art = await _generate(tmp_path, monkeypatch, reply, refs=REFS, further=FURTHER)
    tex = result["poster_tex"].read_text(encoding="utf-8")
    band = tex.split(r"{\bfseries References}", 1)[1].split(r"\end{multicols}", 1)[0]
    assert "[1] E. Hairer et al. (2006). Geometric Numerical Integration. Springer. doi:10.1007/3-540-30666-8." in band
    assert "[2] L. Verlet (1967). Computer experiments on classical fluids. Physical Review." in band
    assert "[W1] Energy of a damped oscillator. physics.stackexchange.com." in band
    assert band.index("[1]") < band.index("[2]") < band.index("[W1]")
    assert "Uncited work" not in tex
    assert "https://" not in band and "www." not in band
    assert "We study scaling [2] and more [1]." in tex


@pytest.mark.asyncio
async def test_with_nothing_cited_the_band_lists_selected_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_engine(monkeypatch)
    reply = {"headline": "A finding", "blocks": [{"type": "text", "text": "No citations here."}]}
    result, _art = await _generate(tmp_path, monkeypatch, reply, refs=REFS, further=FURTHER)
    tex = result["poster_tex"].read_text(encoding="utf-8")
    assert r"{\bfseries Selected sources}" in tex
    assert "Uncited work" in tex


def test_citation_ranges_and_the_eight_source_cap() -> None:
    blocks = [_Block("text", text="a [1–3] b [4; W2] c [5,6,7,8,9,10]")]
    known = {str(n) for n in range(1, 11)} | {"W2"}
    assert poster_mod._cited_labels(blocks, known) == ["1", "2", "3", "4", "5", "6", "7", "W2"]
    assert poster_mod._keep_citations("x [2, 9]. y [11].", {"2"}) == "x [2]. y."


def test_band_entries_name_authors_and_never_print_a_url() -> None:
    entry = poster_mod._band_entry
    assert entry({"authors": ["A. One", "B. Two"], "year": 2001, "title": "T", "arxiv_id": "2101.00001"}, "4") == (
        "[4] A. One and B. Two (2001). T. arXiv:2101.00001."
    )
    assert entry({"title": "Page", "url": "https://www.example.org/a_b?x=1"}, "W3") == "[W3] Page. example.org."


def test_qr_code_only_with_a_url() -> None:
    assert poster_mod._qr_latex("", A1) == ""
    assert r"\qrcode[height=6.5cm]{https://e.org/a\%20b\~c}" in poster_mod._qr_latex("https://e.org/a%20b~c", A1)


# ---------------------------------------------------------------------------
# Text and blocks


def test_inline_latex_escapes_specials_and_keeps_math() -> None:
    from generation.poster import _inline_latex as tex

    assert tex("R&D 50% #1 a_b ~x^y {z}") == (
        r"R\&D 50\% \#1 a\_b \textasciitilde{}x\textasciicircum{}y \{z\}"
    )
    assert tex("energy $E = mc^2$ costs $5 and $10") == r"energy $E = mc^2$ costs \$5 and \$10"
    assert tex("**bold** and *italic*") == r"\textbf{bold} and \emph{italic}"
    assert tex(r"\textbf{Rates} rise") == r"\textbf{Rates} rise"
    assert tex(r"broken $\frac{1}{2$ math") == r"broken \$\textbackslash{}frac\{1\}\{2\$ math"


def test_blocks_from_reply_keeps_known_figures_and_skips_empty_blocks() -> None:
    parsed = {"blocks": [
        {"type": "heading", "text": "  Results  "},
        {"type": "figure", "file": "figures/missing.png", "caption": "gone"},
        {"type": "figure", "file": "result.png", "caption": "kept"},
        {"type": "bullets", "items": ["", "one", 3]},
        {"type": "text", "text": ""},
        {"type": "table", "text": "unknown"},
        "not a block",
    ]}
    blocks = poster_mod._blocks_from_reply(parsed, {"result.png"})
    assert [(b.kind, b.text or b.file, b.items) for b in blocks] == [
        ("heading", "Results", []), ("figure", "result.png", []), ("bullets", "", ["one"]),
    ]


def test_paper_captions_drop_the_figure_number() -> None:
    md = "![**Figure 2.** Energy drift over time.](figures/drift.png) ![Fig 3: Plain](./figures/p.png)"
    assert poster_mod._paper_captions(md) == {"drift.png": "Energy drift over time.", "p.png": "Plain"}


def test_figures_the_model_left_out_go_before_the_closing_heading() -> None:
    blocks = [_Block("heading", text="A"), _Block("text", text="a"), _Block("heading", text="End"), _Block("text", text="e")]
    out = poster_mod._with_every_figure(blocks, ["x.png"], {"x.png": "X"})
    assert [b.kind for b in out] == ["heading", "text", "figure", "heading", "text"]
    assert out[2].caption == "X"


def test_lenient_json_keeps_latex_backslashes_the_model_left_single() -> None:
    """gemma4 wrote ``\\textbf`` with one backslash inside its JSON: the header
    rendered as "extbf…". An ``\\item`` or ``\\%`` would have failed the whole
    reply and left both columns empty."""
    from generation.poster import _lenient_json

    reply = (
        r'{"title": "T", '
        r'"left": "\textbf{A}\n\begin{itemize}\item 50\% \frac{1}{2}\end{itemize}", '
        r'"right": "\\textbf{B}\nNext line \u00e9 \noindent \times"}'
    )
    parsed = _lenient_json(reply)
    assert parsed is not None
    assert parsed["left"] == (
        "\\textbf{A}\n\\begin{itemize}\\item 50\\% \\frac{1}{2}\\end{itemize}"
    )
    assert parsed["right"] == "\\textbf{B}\nNext line \u00e9 \\noindent \\times"
    assert _lenient_json(r'{"t": "$\neg x \nleq y$"}')["t"] == "$\\neg x \\nleq y$"


# ---------------------------------------------------------------------------
# The old two-column reply


@pytest.mark.asyncio
async def test_a_two_column_latex_reply_still_renders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_engine(monkeypatch)
    reply = {"title": "T", "left": r"\textbf{Abstract.} We study scaling.", "right": r"\textbf{Results.} 50% up & more."}
    result, _art = await _generate(tmp_path, monkeypatch, reply)
    tex = result["poster_tex"].read_text(encoding="utf-8")
    assert r"\textbf{Abstract.} We study scaling." in tex
    assert r"\textbf{Results.} 50\% up \& more." in tex


def test_escape_latex_text_specials() -> None:
    from generation.poster import _escape_latex_text_specials as esc
    assert esc("Microlensing & Timing") == r"Microlensing \& Timing"
    assert esc("50% done, item #1") == r"50\% done, item \#1"
    assert esc(r"keep \& and \% intact") == r"keep \& and \% intact"
    assert esc("") == ""


def test_strip_column_commands_keeps_column_lengths() -> None:
    """A bare ``\\column`` inside the template's column stops pdflatex with
    'Missing number'; ``\\columnwidth`` and ``\\columnsep`` are real lengths."""
    from generation.poster import _strip_column_commands as strip
    assert strip("\\column\n\\textbf{Results}") == "\n\\textbf{Results}"
    assert strip(r"\column{0.47\linewidth}\textbf{A}") == r"\textbf{A}"
    assert strip(r"\includegraphics[width=\columnwidth]{f.png}") == (
        r"\includegraphics[width=\columnwidth]{f.png}"
    )
    assert strip("") == ""


@pytest.mark.asyncio
async def test_poster_strips_column_commands_from_llm_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real gemma4 poster began both columns with ``\\column`` and failed to
    compile; the generator removes them."""
    _no_engine(monkeypatch)
    reply = {"title": "T", "left": "\\column\n\\textbf{Left header}", "right": "  \\column\n\\textbf{Right header}"}
    result, _art = await _generate(tmp_path, monkeypatch, reply)
    tex = result["poster_tex"].read_text(encoding="utf-8")
    assert r"\textbf{Left header}" in tex
    assert r"\textbf{Right header}" in tex
    assert re.search(r"\\column(?![A-Za-z])", tex) is None


def test_literal_newlines_from_doubled_json_become_line_breaks() -> None:
    """A gemma4 poster doubled its newline escapes along with its backslashes;
    the literal ``\\n`` stopped pdflatex as an undefined command."""
    from generation.poster import _literal_newlines_to_breaks as fix

    assert fix(r"\textbf{Rates}\nThe slope") == "\\textbf{Rates}\nThe slope"
    assert fix(r"\noindent x \nabla y") == r"\noindent x \nabla y"
    assert fix(r"$\neg x \nleq y$") == r"$\neg x \nleq y$"
    assert fix(r"line\\next") == r"line\\next"


# ---------------------------------------------------------------------------
# Layout


def test_partition_keeps_order_and_fills_the_first_columns_first() -> None:
    assert poster_mod._partition([5, 5, 5, 5], 2) == [(0, 2), (2, 4)]
    assert poster_mod._partition([9, 1, 1, 1], 2) == [(0, 1), (1, 4)]
    assert poster_mod._partition([3], 3) == [(0, 1), (1, 1), (1, 1)]


def _sentences(n: int) -> str:
    return " ".join(f"Sentence number {k} says something about the result." for k in range(n))


def test_plan_cuts_lists_and_sentences_but_never_the_opening_or_closing_block() -> None:
    blocks = [
        _Block("heading", text="Background"), _Block("text", text=_sentences(12)),
        _Block("heading", text="Findings"), _Block("bullets", items=[f"Item {k} with a few words" for k in range(30)]),
        _Block("text", text=_sentences(30)),
        _Block("heading", text="What it means"), _Block("text", text=_sentences(10)),
    ]
    layout = _Layout(A1, blocks, {}, {})
    assert layout.trimmed == layout.dropped == 0  # never cut to fit the first guess
    available = 40 * PT_PER_CM
    layout.plan(available)
    assert max(layout.column_heights()) <= available
    assert layout.trimmed > 0
    assert layout.blocks[0].text == "Background" and layout.blocks[-1].kind == "text"
    assert layout.blocks[-2].text == "What it means"
    bullets = next(b for b in layout.blocks if b.kind == "bullets")
    assert len(bullets.items) >= 2


def test_plan_narrows_figures_only_in_the_column_that_is_over() -> None:
    # About 31 cm and 46 cm at full width, whether split by section or by
    # heading-and-block; 44 cm of room narrows only the second column.
    blocks = [
        _Block("heading", text="One"), _Block("text", text=_sentences(10)), _Block("figure", file="a.png", caption="a"),
        _Block("heading", text="Two"), _Block("figure", file="b.png", caption="b"),
        _Block("figure", file="c.png", caption="c"),
    ]
    layout = _Layout(A1, blocks, {"a.png": 0.6, "b.png": 0.75, "c.png": 0.75}, {})
    layout.plan(44 * PT_PER_CM)
    assert layout.groups == [[0, 1, 2], [3, 4, 5]]
    assert layout.widths[2] == 1.0
    assert layout.widths[4] == layout.widths[5] < 1.0
    assert max(layout.column_heights()) <= 44 * PT_PER_CM


def test_plan_keeps_sections_whole_and_spaces_short_columns_with_a_cap() -> None:
    blocks = []
    for name in ("One", "Two", "Three", "Four"):
        blocks += [_Block("heading", text=name), _Block("text", text=_sentences(4))]
    layout = _Layout(A1, blocks, {}, {})
    layout.plan(55 * PT_PER_CM)
    assert [[layout.blocks[i].text for i in g if layout.blocks[i].kind == "heading"] for g in layout.groups] == [
        ["One", "Two"], ["Three", "Four"],
    ]
    cap = 4.0 * PT_PER_CM
    assert 0 < layout.space[2] <= cap + 1e-6  # before the second heading of column 1
    assert layout.space[1] == 0  # never between a heading and its text
    tex = layout.columns_latex()
    assert r"\vspace{" in tex and tex.count(r"\begin{column}") == 2


def test_rescale_learns_from_the_measured_column_without_its_added_space() -> None:
    blocks = [_Block("heading", text="One"), _Block("text", text=_sentences(4)), _Block("heading", text="Two"), _Block("text", text=_sentences(4))]
    layout = _Layout(A1, blocks, {}, {})
    layout.plan(200 * PT_PER_CM)
    column = layout.groups[0]
    estimated = layout.column_height(0)
    added = sum(layout.space[i] for i in column)
    layout.rescale([2 * estimated + added, 0.0])
    assert layout.column_height(0) == pytest.approx(2 * estimated)


def test_the_text_after_a_figure_stays_with_the_figure() -> None:
    """The validation quest's poster opened its second column on the two
    sentences under "Energy Decay Performance"'s figure, with no heading."""
    blocks = [
        _Block("heading", text="Background"), _Block("text", text=_sentences(2)),
        _Block("heading", text="Energy decay"), _Block("figure", file="b.png", caption="b"),
        _Block("text", text=_sentences(2)),
        _Block("figure", file="c.png", caption="c"), _Block("bullets", items=["one", "two"]),
        _Block("heading", text="What it means"), _Block("text", text=_sentences(3)),
    ]
    assert poster_mod._units(blocks) == [[0, 1], [2, 3, 4], [5, 6], [7, 8]]


def test_a_column_that_opens_inside_a_section_repeats_its_heading() -> None:
    blocks = [
        _Block("heading", text="Energy decay"), _Block("text", text=_sentences(6)),
        _Block("text", text=_sentences(3)), _Block("heading", text="Drift"), _Block("text", text=_sentences(3)),
    ]
    layout = _Layout(A1, blocks, {}, {})
    layout.groups = [[0, 1], [2, 3, 4]]
    layout._spread(200 * PT_PER_CM)
    assert layout.continued_heading(0) is None and layout.continued_heading(1) == 0
    tex = layout.columns_latex()
    second = tex[tex.index(r"\begin{column}", tex.index(r"\begin{column}") + 1):]
    assert second.index(r"\posterhead{Energy decay (continued)}") < second.index("Sentence number 0")
    heading = layout._heading_height("Energy decay (continued)")
    assert layout.column_height(1) == pytest.approx(sum(layout.estimate(i) for i in (2, 3, 4)) + heading)


# ---------------------------------------------------------------------------
# The measured fit loop


def _fake_compile(monkeypatch: pytest.MonkeyPatch, engine=("pdflatex", "/fake/pdflatex")) -> list[dict]:
    monkeypatch.setattr(poster_mod, "find_pdf_engine", lambda: engine)
    runs: list[dict] = []
    real_run = subprocess.run

    def fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        # Only the compile is faked; the patch is on the shared module.
        if not isinstance(cmd, (list, tuple)) or not any(str(part).endswith("poster.tex") for part in cmd):
            return real_run(cmd, **kwargs)
        cwd = Path(kwargs.get("cwd") or ".")
        runs.append({"cmd": list(cmd), "tex": (cwd / "poster.tex").read_text(encoding="utf-8")})
        (cwd / "poster.pdf").write_bytes(b"%PDF-fake\n")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(poster_mod.subprocess, "run", fake_run)
    return runs


def _report(*checks: str) -> dict:
    return {
        "metrics": {"header_bottom_cm": 20.0, "band_top_cm": 74.0, "body_pt": 26, "references_pt": 16},
        "findings": [{"check": c, "severity": "high", "problem": c} for c in checks],
    }


@pytest.mark.asyncio
async def test_fit_loop_replans_until_the_sheet_measures_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = _fake_compile(monkeypatch)
    reports = iter([_report("band_overlap"), _report()])
    monkeypatch.setattr(poster_mod, "measure_pdf", lambda path: SimpleNamespace(pages=[]))
    monkeypatch.setattr(poster_mod, "poster_report", lambda doc, **kw: next(reports))
    monkeypatch.setattr(poster_mod, "_measured_columns", lambda doc, report, sheet: ([90 * PT_PER_CM, 90 * PT_PER_CM], 30 * PT_PER_CM))
    reply = _reply()
    reply["blocks"][1]["text"] = _sentences(20)
    result, art = await _generate(tmp_path, monkeypatch, reply)
    assert len(runs) == 2 and runs[0]["tex"] != runs[1]["tex"]
    fit = json.loads((art.quest_root / ".fi" / "poster_fit.json").read_text(encoding="utf-8"))
    assert fit["compiles"] == 2 and fit["findings"] == []
    assert fit["trimmed"] + fit["dropped_blocks"] > 0
    assert result["poster_pdf"].exists()


@pytest.mark.asyncio
async def test_fit_loop_stops_after_six_compiles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = _fake_compile(monkeypatch)
    monkeypatch.setattr(poster_mod, "measure_pdf", lambda path: SimpleNamespace(pages=[]))
    monkeypatch.setattr(poster_mod, "poster_report", lambda doc, **kw: _report("band_overlap"))
    monkeypatch.setattr(
        poster_mod, "_measured_columns",
        lambda doc, report, sheet: ([80 * PT_PER_CM, 80 * PT_PER_CM], 30 * PT_PER_CM),
    )
    # Every re-plan looks new, so only the compile cap can end the loop.
    monkeypatch.setattr(poster_mod._Layout, "signature", lambda self: object())
    await _generate(tmp_path, monkeypatch, _reply())
    assert len(runs) == 6


@pytest.mark.asyncio
async def test_uneven_columns_alone_recompile_only_when_replanning_helps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = _fake_compile(monkeypatch)
    monkeypatch.setattr(poster_mod, "measure_pdf", lambda path: SimpleNamespace(pages=[]))
    monkeypatch.setattr(poster_mod, "poster_report", lambda doc, **kw: _report("column_balance"))
    # Columns measured 1 cm apart: no plan can gain the 2 cm a compile needs.
    monkeypatch.setattr(
        poster_mod, "_measured_columns",
        lambda doc, report, sheet: ([40 * PT_PER_CM, 39 * PT_PER_CM], 50 * PT_PER_CM),
    )
    await _generate(tmp_path, monkeypatch, _reply())
    assert len(runs) == 1


@pytest.mark.asyncio
async def test_a_poster_that_measures_clean_compiles_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = _fake_compile(monkeypatch)
    monkeypatch.setattr(poster_mod, "measure_pdf", lambda path: SimpleNamespace(pages=[]))
    monkeypatch.setattr(poster_mod, "poster_report", lambda doc, **kw: _report("empty_space"))
    # The sheet measures about the room the first plan assumed (52 cm).
    monkeypatch.setattr(
        poster_mod, "_measured_columns",
        lambda doc, report, sheet: ([30 * PT_PER_CM, 30 * PT_PER_CM], 53 * PT_PER_CM),
    )
    await _generate(tmp_path, monkeypatch, _reply())
    assert len(runs) == 1


@pytest.mark.asyncio
async def test_more_room_than_planned_replans_without_cutting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = _fake_compile(monkeypatch)
    reports = iter([_report("empty_space"), _report()])
    monkeypatch.setattr(poster_mod, "measure_pdf", lambda path: SimpleNamespace(pages=[]))
    monkeypatch.setattr(poster_mod, "poster_report", lambda doc, **kw: next(reports))
    monkeypatch.setattr(
        poster_mod, "_measured_columns",
        lambda doc, report, sheet: ([30 * PT_PER_CM, 30 * PT_PER_CM], 60 * PT_PER_CM),
    )
    planned: list[tuple[float, bool]] = []
    plan = poster_mod._Layout.plan

    def recording_plan(self, available, **kw):  # type: ignore[no-untyped-def]
        planned.append((round(available / PT_PER_CM, 1), kw.get("cut", True)))
        return plan(self, available, **kw)

    monkeypatch.setattr(poster_mod._Layout, "plan", recording_plan)
    # A plan for other room reads as a new layout, whatever the estimates.
    monkeypatch.setattr(poster_mod._Layout, "signature", lambda self: round(self.available))
    reply = _reply()
    result, art = await _generate(tmp_path, monkeypatch, reply)
    first_guess = round(poster_mod._FIRST_GUESS_ROOM * A1.height_cm, 1)
    assert planned == [(first_guess, False), (60.0, True)]
    assert len(runs) == 2
    fit = json.loads((art.quest_root / ".fi" / "poster_fit.json").read_text(encoding="utf-8"))
    assert fit["compiles"] == 2 and fit["trimmed"] == 0 and fit["dropped_blocks"] == 0
    assert (art.quest_root / ".fi" / "poster_reply.txt").read_text(encoding="utf-8") == json.dumps(reply)


@pytest.mark.asyncio
async def test_a_first_plan_over_its_own_limit_is_planned_again_for_the_measured_room(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The validation quest's first plan was 54 cm against a 50.6 cm limit and
    fit only because the sheet had 57.5 cm. The sheet measured clean, so it
    was never planned again, and a section stayed split."""
    runs = _fake_compile(monkeypatch)
    monkeypatch.setattr(poster_mod, "measure_pdf", lambda path: SimpleNamespace(pages=[]))
    monkeypatch.setattr(poster_mod, "poster_report", lambda doc, **kw: _report())
    monkeypatch.setattr(
        poster_mod, "_measured_columns",
        lambda doc, report, sheet: ([50 * PT_PER_CM, 45 * PT_PER_CM], 57 * PT_PER_CM),
    )
    planned: list[float] = []
    plan = poster_mod._Layout.plan

    def recording_plan(self, available, **kw):  # type: ignore[no-untyped-def]
        plan(self, available, **kw)
        planned.append(round(available / PT_PER_CM, 1))
        if len(planned) == 1:
            self.over_limit = True

    monkeypatch.setattr(poster_mod._Layout, "plan", recording_plan)
    monkeypatch.setattr(poster_mod._Layout, "signature", lambda self: round(self.available))
    await _generate(tmp_path, monkeypatch, _reply())
    assert planned[1] == 57.0 and len(runs) == 2


def test_measured_columns_count_spilled_text_but_not_the_reference_list() -> None:
    from generation._pdf_measure import Line, Page

    page_h = 84.1 * PT_PER_CM
    top = page_h - 20 * PT_PER_CM
    floor = page_h - 74 * PT_PER_CM
    left_x, right_x = 3 * PT_PER_CM, 32 * PT_PER_CM
    lines = [
        Line("left column text", 26, False, (left_x, top - 300, left_x + 500, top - 280)),
        Line("right column spilled", 26, False, (right_x, floor - 40, right_x + 500, floor - 20)),
        Line("[1] A reference in the band", 16, False, (left_x, floor - 60, left_x + 500, floor - 45)),
        Line("Figure 3: a caption that slid into the band", 20, False, (right_x, floor - 70, right_x + 500, floor - 50)),
    ]
    doc = SimpleNamespace(pages=[Page(1, 59.4 * PT_PER_CM, page_h, lines, [], [])])
    heights, room = poster_mod._measured_columns(doc, _report(), A1)
    gap = 1.2 * PT_PER_CM
    assert heights[0] == pytest.approx(top - gap - (top - 300))
    assert heights[1] == pytest.approx(top - gap - (floor - 70))
    assert room == pytest.approx(top - gap - floor)


# ---------------------------------------------------------------------------
# Engines and diagnostics


@pytest.mark.asyncio
async def test_chinese_text_compiles_with_xelatex_and_its_font(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = _fake_compile(monkeypatch)
    monkeypatch.setattr(_cjk, "find_xelatex", lambda engine: ("xelatex", "/fake/xelatex"))
    monkeypatch.setattr(_cjk, "find_cjk_font", lambda text: "Noto Sans CJK TC")
    monkeypatch.setattr(poster_mod, "measure_pdf", lambda path: None)
    result, _art = await _generate(tmp_path, monkeypatch, _reply(), output={"author": "陳建明"})
    (run,) = runs
    assert run["cmd"][0] == "/fake/xelatex"
    assert r"\setCJKmainfont{Noto Sans CJK TC}" in run["tex"] and "陳建明" in run["tex"]
    assert "poster_pdf" in result


@pytest.mark.asyncio
async def test_chinese_text_without_a_font_skips_with_a_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = _fake_compile(monkeypatch)
    monkeypatch.setattr(_cjk, "find_xelatex", lambda engine: ("xelatex", "/fake/xelatex"))
    monkeypatch.setattr(_cjk, "find_cjk_font", lambda text: None)
    result, _art = await _generate(tmp_path, monkeypatch, _reply(headline="能量守恆的積分器"))
    assert runs == []
    assert "cjk_no_font" in result["poster_pdf_skipped"].read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_a_latin_poster_keeps_pdflatex(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = _fake_compile(monkeypatch)
    monkeypatch.setattr(poster_mod, "measure_pdf", lambda path: None)
    await _generate(tmp_path, monkeypatch, _reply(), output={"author": "Jane Chen"})
    (run,) = runs
    assert run["cmd"][0] == "/fake/pdflatex"
    assert "xeCJK" not in run["tex"]


@pytest.mark.asyncio
async def test_poster_falls_back_to_tectonic_when_pdflatex_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A user who followed the documented no-admin install
    (``python launch.py --install-tectonic``) gets a poster too: poster.py
    consults the same engine discovery as the paper."""
    def fake_which(name):  # type: ignore[no-untyped-def]
        return "/fake/tectonic.exe" if name == "tectonic" else None
    monkeypatch.setattr(poster_mod.shutil, "which", fake_which)
    captured_cmd: list[str] = []
    real_run = subprocess.run

    def fake_run(cmd, **_kwargs):  # type: ignore[no-untyped-def]
        # The patch is on the shared subprocess module, so other callers
        # (platform's ``ver`` lookup on Windows) reach it too.
        if not isinstance(cmd, (list, tuple)) or not any(str(part).endswith("poster.tex") for part in cmd):
            return real_run(cmd, **_kwargs)
        captured_cmd[:] = list(cmd)
        cwd = Path(_kwargs.get("cwd") or ".")
        (cwd / "poster.pdf").write_bytes(b"%PDF-fake\n")
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(poster_mod.subprocess, "run", fake_run)

    result, art = await _generate(tmp_path, monkeypatch, _reply())

    assert captured_cmd and captured_cmd[0] == "/fake/tectonic.exe"
    assert "-interaction=nonstopmode" in captured_cmd
    assert "-halt-on-error" in captured_cmd
    assert "poster_pdf" in result
    assert "poster_pdf_skipped" not in result
    assert not (art.quest_root / "poster_pdf_skipped.md").exists()


@pytest.mark.asyncio
async def test_poster_writes_skip_diagnostic_when_no_engine_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no pdflatex and no tectonic, ``poster_pdf_skipped.md`` lands next
    to ``poster.tex``, the same contract as ``paper_pdf_skipped.md``."""
    monkeypatch.setattr(poster_mod.shutil, "which", lambda _name: None)
    from generation import _pdf_engine
    monkeypatch.setattr(_pdf_engine, "_DEFAULT_REPO_ROOT", tmp_path)

    result, art = await _generate(tmp_path, monkeypatch, _reply())

    assert "poster_tex" in result and "poster_pdf" not in result
    diag = art.quest_root / "poster_pdf_skipped.md"
    body = diag.read_text(encoding="utf-8")
    assert "poster.pdf was requested but not produced" in body
    assert "no_latex_engine" in body
    assert "--install-tectonic" in body
    assert result.get("poster_pdf_skipped") == diag


@pytest.mark.asyncio
async def test_poster_writes_skip_diagnostic_when_pdf_missing_after_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The engine returned rc=0 but ``poster.pdf`` isn't on disk."""
    monkeypatch.setattr(poster_mod, "find_pdf_engine", lambda: ("pdflatex", "/fake/pdflatex.exe"))
    monkeypatch.setattr(
        poster_mod.subprocess, "run",
        lambda cmd, **_kw: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    result, art = await _generate(tmp_path, monkeypatch, _reply())
    assert "poster_pdf" not in result
    diag = art.quest_root / "poster_pdf_skipped.md"
    assert "output_missing_after_success" in diag.read_text(encoding="utf-8")
    assert result.get("poster_pdf_skipped") == diag


@pytest.mark.asyncio
async def test_poster_uses_same_engine_discovery_as_paper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whatever ``find_pdf_engine`` returns is the binary the compile runs."""
    sentinel = ("tectonic", "/sentinel/tectonic.exe")
    monkeypatch.setattr(poster_mod, "find_pdf_engine", lambda: sentinel)
    captured: list[list[str]] = []
    real_run = subprocess.run

    def fake_run(cmd, **_kwargs):  # type: ignore[no-untyped-def]
        if not isinstance(cmd, (list, tuple)) or not any(str(part).endswith("poster.tex") for part in cmd):
            return real_run(cmd, **_kwargs)
        captured.append(list(cmd))
        (Path(_kwargs.get("cwd") or ".") / "poster.pdf").write_bytes(b"%PDF-fake\n")
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(poster_mod.subprocess, "run", fake_run)

    await _generate(tmp_path, monkeypatch, _reply())
    assert captured[0][0] == sentinel[1]


def test_cleanup_poster_artifacts_success_keeps_pdf_and_tex(tmp_path: Path) -> None:
    from generation.poster import _cleanup_poster_artifacts
    for name in ("poster.pdf", "poster.tex", "poster.aux", "poster.log",
                 "poster.out", "poster.nav", "poster.toc"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    _cleanup_poster_artifacts(tmp_path, keep_log=False)
    assert sorted(p.name for p in tmp_path.glob("poster.*")) == [
        "poster.pdf", "poster.tex",
    ]


def test_cleanup_poster_artifacts_failure_keeps_log(tmp_path: Path) -> None:
    from generation.poster import _cleanup_poster_artifacts
    for name in ("poster.tex", "poster.log", "poster.aux", "poster.out"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    (tmp_path / "poster_pdf_skipped.md").write_text("x", encoding="utf-8")
    _cleanup_poster_artifacts(tmp_path, keep_log=True)
    remaining = sorted(p.name for p in tmp_path.glob("poster*"))
    assert "poster.tex" in remaining
    assert "poster.log" in remaining
    assert "poster_pdf_skipped.md" in remaining
    assert "poster.aux" not in remaining
    assert "poster.out" not in remaining


# ---------------------------------------------------------------------------
# Real compiles


def _missing_tex_packages(*names: str) -> list[str]:
    """The TeX files kpsewhich cannot find (all of them without kpsewhich)."""
    kpsewhich = shutil.which("kpsewhich")
    if kpsewhich is None:
        return list(names)
    return [
        name for name in names
        if not subprocess.run([kpsewhich, name], capture_output=True, text=True).stdout.strip()
    ]


@pytest.mark.slow
@pytest.mark.asyncio
async def test_a_real_poster_meets_the_poster_standards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if shutil.which("pdflatex") is None or _missing_tex_packages("beamerposter.sty", "qrcode.sty", "newpxtext.sty"):
        pytest.skip("needs pdflatex with beamerposter, qrcode and newpxtext")
    from generation._pdf_measure import measure_pdf, poster_report, render_pages

    url = "https://example.org/p?a=1&b=2#sec_3"
    result, _art = await _generate(
        tmp_path, monkeypatch, _reply(),
        output={"author": "Jane Chen", "affiliation": "R&D Lab", "contact_email": "jane_c@example.org", "url": url},
        refs=REFS, further=FURTHER, real_figure=True,
    )
    assert "poster_pdf" in result, result.get("poster_pdf_skipped") and result["poster_pdf_skipped"].read_text(encoding="utf-8")
    report = poster_report(measure_pdf(result["poster_pdf"]), header_terms=("Jane Chen", "R&D Lab", "jane_c@example.org"))
    checks = {f["check"] for f in report["findings"]}
    assert not checks & {
        "overflow", "band_overlap", "title_font", "body_font", "heading_font", "caption_font",
        "references_font", "captions", "line_length", "references_count", "references_urls", "header_info",
    }, report["findings"]
    m = report["metrics"]
    assert m["title_pt"] >= 72 and m["body_pt"] >= 24 and m["heading_pt"] >= 36 and m["references"] == 3
    try:
        import cv2
    except ImportError:
        return
    shot = render_pages(result["poster_pdf"], tmp_path / "shots", dpi=60)[0]
    decoded, _points, _raw = cv2.QRCodeDetector().detectAndDecode(cv2.imread(str(shot)))
    assert decoded == url


@pytest.mark.slow
@pytest.mark.asyncio
async def test_a_real_poster_prints_a_chinese_author_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from generation._pdf_engine import find_pdf_engine
    from generation._pdf_measure import measure_pdf

    engine = find_pdf_engine()
    if (
        engine is None or _cjk.find_xelatex(engine) is None or _cjk.find_cjk_font("陳建明") is None
        or _missing_tex_packages("beamerposter.sty", "qrcode.sty", "fontspec.sty", "xeCJK.sty")
    ):
        pytest.skip("needs XeLaTeX with beamerposter, qrcode, xeCJK and a CJK font")
    result, _art = await _generate(
        tmp_path, monkeypatch, _reply(), output={"author": "陳建明", "affiliation": "國立台灣大學"},
        real_figure=True,
    )
    assert "poster_pdf" in result
    text = " ".join(line.text for page in measure_pdf(result["poster_pdf"]).pages for line in page.lines)
    assert "陳建明" in text and "國立台灣大學" in text
