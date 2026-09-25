"""Figures with their captions out of a PDF (core/pdf_figures.py), and where the literature step puts them."""

from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path

import pytest

from core import pdf_figures


def _paper_pdf(path: Path, *, table_above: bool = False) -> Path:
    """A one-page paper: body text, a plot with its caption below, more body text (matplotlib writes text as text)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.08, 0.95, "Body text of the paper runs across the page above the figure, as it does in print.",
             fontsize=10)
    if table_above:
        fig.text(0.08, 0.90, "Table 1. Parameters used in every run, with the values the model was fitted to.",
                 fontsize=9)
        fig.text(0.08, 0.885, "The second line of the table caption.", fontsize=9)
    ax = fig.add_axes((0.15, 0.55, 0.7, 0.28))
    ax.plot([0, 1, 2, 3], [1, 3, 2, 5])
    ax.set_xlabel("time (days)")
    ax.set_ylabel("infected")
    fig.text(0.08, 0.49, "Figure 1. Infected count over time for the baseline run.", fontsize=9)
    fig.text(0.08, 0.475, "Shaded bands are 95% intervals over ten seeds.", fontsize=9)
    fig.text(0.08, 0.40, "Figure 2 shows nothing here: a sentence in the body that starts with the word.",
             fontsize=10)
    fig.text(0.08, 0.20, "Fig. 3. A caption with no drawing above it is not a figure.", fontsize=9)
    fig.savefig(path, format="pdf")
    plt.close(fig)
    return path


def test_a_figure_is_cut_with_its_whole_caption(tmp_path: Path) -> None:
    figures = pdf_figures.find(_paper_pdf(tmp_path / "p.pdf"))
    assert [f.number for f in figures] == ["1"], "a body sentence and a caption with no drawing are not figures"
    fig = figures[0]
    assert fig.page == 1 and not fig.scanned
    assert fig.caption.startswith("Figure 1. Infected count over time")
    assert "95% intervals over ten seeds" in fig.caption, "the caption's second line belongs to it"
    from PIL import Image

    image = Image.open(io.BytesIO(fig.png))
    # The plot is 0.7 x 0.28 of an 8.5 x 11 inch page, at 144 dpi: about 857 x 443 pixels, plus its axis labels.
    assert 800 < image.width < 1150 and 400 < image.height < 650


def test_a_table_caption_above_stops_the_figure(tmp_path: Path) -> None:
    fig = pdf_figures.find(_paper_pdf(tmp_path / "p.pdf", table_above=True))[0]
    from PIL import Image

    assert Image.open(io.BytesIO(fig.png)).height < 650, "the table's caption above is not part of the figure"


def test_an_unreadable_pdf_has_no_figures(tmp_path: Path) -> None:
    assert pdf_figures.find(b"not a pdf") == []


def test_a_scanned_page_figure_is_found_from_its_ocr_lines(tmp_path: Path) -> None:
    """A page that is only an image: the caption and the body text above come from OCR (given here)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig = plt.figure(figsize=(8.5, 11))
    fig.figimage(np.full((1100, 850), 0.9), cmap="gray", vmin=0, vmax=1)  # the scan: one image, no text
    path = tmp_path / "scan.pdf"
    fig.savefig(path, format="pdf", dpi=100)
    plt.close(fig)
    lines = [  # PDF points, y up, top to bottom
        ((50.0, 700.0, 560.0, 712.0), "Body text of the classic paper, a full line wide across the page."),
        ((200.0, 600.0, 260.0, 608.0), "infected"),
        ((50.0, 400.0, 400.0, 410.0), "FIG. 2. Infected count over time,"),
        ((50.0, 388.0, 300.0, 398.0), "from the 1927 data."),
        ((50.0, 360.0, 560.0, 372.0), "More body text below the caption, a full line wide across the page."),
    ]
    figures = pdf_figures.find(path, ocr_lines={1: lines})
    assert [(f.number, f.scanned) for f in figures] == [("2", True)]
    assert figures[0].caption == "FIG. 2. Infected count over time, from the 1927 data."
    from PIL import Image

    image = Image.open(io.BytesIO(figures[0].png))
    assert 500 < image.height < 600, "from the caption up to the body line above (about 288 pt at 2x)"


def test_figures_are_cut_once_and_cached_by_the_pdf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from core import knowledge

    monkeypatch.setattr(knowledge, "_figure_cache_dir", lambda: tmp_path / "cache")
    body = _paper_pdf(tmp_path / "p.pdf").read_bytes()
    first = knowledge._pdf_figures(body)
    assert len(first) == 1 and Path(first[0]["image"]).is_file() and first[0]["caption"].startswith("Figure 1.")

    def boom(*_a, **_k):  # noqa: ANN002, ANN003
        raise AssertionError("cut again")

    monkeypatch.setattr(pdf_figures, "find", boom)
    assert knowledge._pdf_figures(body) == first


def test_a_fetch_lists_the_figures_of_the_pdf_it_kept(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from core import knowledge
    from core.knowledge import RetrievedDoc

    monkeypatch.setattr(knowledge, "_figure_cache_dir", lambda: tmp_path / "cache")
    body = _paper_pdf(tmp_path / "p.pdf").read_bytes()

    def fetch(doc, *, timeout_s, max_kb):  # noqa: ANN001, ARG001 -- a route that downloads and reads a PDF
        return knowledge._pdf_bytes_to_text(body, cap=max_kb * 1024)

    docs = [RetrievedDoc(content="abstract", metadata={"title": "SIR", "pdf_url": "https://x/p.pdf"})]
    out = asyncio.run(knowledge._enrich_with_full_text(docs, timeout_s=5, total_budget_s=30, max_kb=64,
                                                      fetch_fn=fetch))
    figures = out[0].metadata.get("figures")
    assert out[0].metadata.get("fetched_full_text") and figures and figures[0]["number"] == "1"


def test_the_literature_files_carry_the_figures_and_the_data_walk_skips_them(tmp_path: Path) -> None:
    from core import engine as eng
    from core.summarizer import _walk_folder

    image = tmp_path / "cached.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    item = {"content": "the paper", "metadata": {"title": "SIR model", "url": "https://x", "figures": [
        {"number": "1", "page": 3, "caption": "Figure 1. Infected over time.", "image": str(image), "scanned": False},
        {"number": "2", "page": 4, "caption": "Figure 2. Gone.", "image": str(tmp_path / "missing.png")},
    ]}}
    out_dir = tmp_path / "data" / "literature"
    engine = eng.Engine.__new__(eng.Engine)
    import logging

    engine._log = logging.getLogger("t")
    assert engine._write_literature_files([item], out_dir) == 1
    text = next(out_dir.glob("lit_001_*.md")).read_text(encoding="utf-8")
    assert "## Figures" in text and "`figures/lit_001_fig1_p3.png`" in text and "Infected over time." in text
    assert "Gone." not in text, "a figure whose image is gone is not listed"
    assert "figures:" not in text.split("---", 2)[1], "the front matter stays flat"
    assert (out_dir / "figures" / "lit_001_fig1_p3.png").is_file()
    walked = [e.rel_path for e in _walk_folder(tmp_path / "data")]
    assert walked == [f"literature/{next(out_dir.glob('lit_001_*.md')).name}"], walked


# ---- reading the values off the figures (engine._read_literature_figures) ---------------------------------------


def _reading_engine(tmp_path: Path, **knowledge):  # noqa: ANN003, ANN202
    from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig
    from core.engine import Engine

    return Engine(Config(
        topic="SIR epidemic peak timing", title="figs", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(), execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False, **knowledge), output=OutputConfig(output_dir=tmp_path / "outputs"),
    ))


def _literature_with_figures(tmp_path: Path) -> list[dict]:
    images = []
    for n in (1, 2, 3):
        p = tmp_path / f"f{n}.png"
        p.write_bytes(b"\x89PNG\r\n\x1a\n" + bytes([n]))
        images.append(str(p))
    return [
        {"content": "SIR paper text", "metadata": {"title": "SIR peaks", "url": "https://a", "figures": [
            {"number": "1", "page": 2, "caption": "Figure 1. Infected over time.", "image": images[0]},
            {"number": "2", "page": 3, "caption": "Figure 2. Model diagram.", "image": images[1]},
        ]}},
        {"content": "Other paper", "metadata": {"title": "Peaks again", "url": "https://b", "figures": [
            {"number": "1", "page": 5, "caption": "Figure 1. Peak day vs R0.", "image": images[2]},
        ]}},
    ]


def test_the_model_picks_from_captions_then_reads_the_picked_images_into_the_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _reading_engine(tmp_path)
    lit = _literature_with_figures(tmp_path)
    sent: list[list] = []

    async def chat(prompt, *, node=None, **_k):  # noqa: ANN001, ANN003
        assert node == "figures" and "[1:1]" in prompt and "[2:1]" in prompt and "Model diagram" in prompt
        return '{"pick": ["1:1", "2:1", "9:9"]}'

    async def chat_messages(messages, *, node=None, **_k):  # noqa: ANN001, ANN003
        assert node == "figures"
        sent.append(messages[0]["content"])
        return json.dumps({"figures": [{"id": "1:1", "reading": "peak about 4,200 on day 38"},
                                       {"id": "2:1", "reading": "peak day falls from 60 to 20 as R0 goes 1.5 to 4"}]})

    monkeypatch.setattr(engine, "_chat", chat)
    monkeypatch.setattr(engine, "_chat_messages", chat_messages)
    read = asyncio.run(engine._read_literature_figures({"topic": "SIR epidemic peak timing"}, lit))
    assert read == 2 and len(sent) == 1, "two picked figures go in one call"
    assert sum(1 for part in sent[0] if part.get("type") == "image_url" or "image" in str(part.get("type"))) == 2
    first = lit[0]["metadata"]["figures"]
    assert first[0]["relevant"] and first[0]["reading"].startswith("peak about") and first[1]["relevant"] is False
    assert "VALUES READ FROM THE FIGURES" in lit[0]["content"] and "day 38" in lit[0]["content"]
    assert "R0 goes 1.5 to 4" in lit[1]["content"]

    async def never(*_a, **_k):  # noqa: ANN002, ANN003
        raise AssertionError("a judged figure is never sent again")

    monkeypatch.setattr(engine, "_chat", never)
    monkeypatch.setattr(engine, "_chat_messages", never)
    assert asyncio.run(engine._read_literature_figures({}, lit)) == 0
    assert lit[0]["content"].count("VALUES READ FROM THE FIGURES") == 1


def test_a_model_that_cannot_read_images_stops_the_quest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from core.provider import ImageInputUnsupported

    engine = _reading_engine(tmp_path)
    lit = _literature_with_figures(tmp_path)

    async def chat(*_a, **_k):  # noqa: ANN002, ANN003
        return '{"pick": ["1:1"]}'

    async def chat_messages(*_a, **_k):  # noqa: ANN002, ANN003
        raise ImageInputUnsupported("codex cannot send images to its model")

    class Paused(Exception):
        pass

    stops: list[dict] = []

    def pause(**kwargs):  # noqa: ANN003
        stops.append(kwargs)
        raise Paused

    monkeypatch.setattr(engine, "_chat", chat)
    monkeypatch.setattr(engine, "_chat_messages", chat_messages)
    monkeypatch.setattr(engine, "_pause_for_human", pause)
    with pytest.raises(Paused):
        asyncio.run(engine._read_literature_figures({}, lit))
    assert stops[0]["kind"] == "figures" and stops[0]["interaction"] == "supply"
    text = " ".join(stops[0]["steps"])
    assert "node_models" in text and "read_figures: false" in text
    assert "relevant" not in lit[0]["metadata"]["figures"][0], "the waiting figure is judged again after the stop"


def test_reading_figures_can_be_turned_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _reading_engine(tmp_path, read_figures=False)

    async def never(*_a, **_k):  # noqa: ANN002, ANN003
        raise AssertionError("no call when reading is off")

    monkeypatch.setattr(engine, "_chat", never)
    assert asyncio.run(engine._read_literature_figures({}, _literature_with_figures(tmp_path))) == 0


def test_the_step_after_the_literature_reads_the_figures_and_rewrites_the_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _reading_engine(tmp_path)
    lit = _literature_with_figures(tmp_path)

    async def chat(*_a, **_k):  # noqa: ANN002, ANN003
        return '{"pick": ["2:1"]}'

    async def chat_messages(*_a, **_k):  # noqa: ANN002, ANN003
        return '{"figures": [{"id": "2:1", "reading": "peak day 60 at R0 1.5, 20 at R0 4"}]}'

    monkeypatch.setattr(engine, "_chat", chat)
    monkeypatch.setattr(engine, "_chat_messages", chat_messages)
    out = asyncio.run(engine._node_pause_after_literature({"literature": lit, "topic": "SIR"}))
    assert "peak day 60" in out["literature"][1]["content"]
    written = (engine.quest_root / "data" / "literature").glob("lit_002_*.md")
    text = next(written).read_text(encoding="utf-8")
    assert "peak day 60 at R0 1.5" in text and "`figures/lit_002_fig1_p5.png`" in text
