"""A redo touches only the broken part: flagged slides and poster panels, unreadable JSON keys."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig,
)
from core.engine import Engine, QuestArtifacts, _readable_json_keys
from generation import _visual_check as vc
from generation.poster import PosterGenerator
from generation.slides import SlideGenerator, join_deck, restore_slides, split_deck


def _config(tmp_path: Path, kinds: list[str]) -> Config:
    return Config(
        topic="part redo", title="part-redo", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(kinds=kinds, output_dir=tmp_path / "outputs"),
    )


def _art(tmp_path: Path) -> QuestArtifacts:
    root = tmp_path / "quest"
    (root / "paper").mkdir(parents=True)
    paper = root / "paper" / "paper.md"
    paper.write_text("# Toy Title\n\nMethods. Results.\n", encoding="utf-8")
    return QuestArtifacts(quest_id="q", quest_root=root, paper_md=paper, figures_dir=None)


DECK = (
    "---\nmarp: true\ntheme: fi\n---\n\n# One\n\nfirst slide\n\n---\n\n## Two\n\n- broken \\textbf{x}\n\n---\n\n"
    "## Three\n\nthird slide\n\n---\n\n## Four\n\nlast slide\n"
)


def _feedback(*pages: int, quote: str = "broken textbf", block: str = "") -> vc.Feedback:
    findings = [{"page": p, "region": "body", "check": "raw_markup", "problem": "raw LaTeX", "severity": "high",
                 "quote": quote, **({"block": block} if block else {})} for p in pages]
    return vc.feedback_text(findings)


# ---------- slides ----------


def test_a_deck_splits_and_joins_back_exactly() -> None:
    head, segments, breaks = split_deck(DECK)
    assert len(segments) == 4 and head.startswith("---\nmarp: true")
    assert join_deck(head, segments, breaks) == DECK


@pytest.mark.asyncio
async def test_a_deck_with_one_flagged_slide_redoes_only_that_slide(tmp_path: Path, monkeypatch) -> None:
    art = _art(tmp_path)
    (art.quest_root / "slides.md").write_text(DECK, encoding="utf-8")
    prompts: list[str] = []

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompts.append(messages[-1]["content"])
        return "## Two\n\n- fixed bullet"

    monkeypatch.setattr("core.provider.LLMClient.chat", fake_chat)
    monkeypatch.setattr("generation.slides.shutil.which", lambda _n: None)
    await SlideGenerator(_config(tmp_path, ["slides"])).generate(art, art.quest_root, feedback=_feedback(2))
    assert len(prompts) == 1 and "slide(s) 2" in prompts[0] and "third slide" in prompts[0]
    new = (art.quest_root / "slides.md").read_text(encoding="utf-8")
    head, old_segments, breaks = split_deck(DECK)
    new_head, new_segments, new_breaks = split_deck(new)
    assert (new_head, new_breaks) == (head, breaks)
    assert [new_segments[i] for i in (0, 2, 3)] == [old_segments[i] for i in (0, 2, 3)], "the other slides are the same bytes"
    assert "fixed bullet" in new_segments[1] and "broken" not in new_segments[1]


@pytest.mark.asyncio
async def test_an_answer_with_the_wrong_number_of_slides_changes_nothing(tmp_path: Path, monkeypatch) -> None:
    art = _art(tmp_path)
    (art.quest_root / "slides.md").write_text(DECK, encoding="utf-8")

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        return "## A\n\none\n\n---\n\n## B\n\ntwo"

    monkeypatch.setattr("core.provider.LLMClient.chat", fake_chat)
    monkeypatch.setattr("generation.slides.shutil.which", lambda _n: None)
    await SlideGenerator(_config(tmp_path, ["slides"])).generate(art, art.quest_root, feedback=_feedback(2))
    assert (art.quest_root / "slides.md").read_text(encoding="utf-8") == DECK


def test_a_redone_slide_can_be_put_back_alone(tmp_path: Path) -> None:
    saved, current = tmp_path / "saved.md", tmp_path / "slides.md"
    saved.write_text(DECK, encoding="utf-8")
    current.write_text(DECK.replace("first slide", "new first").replace("third slide", "new third"), encoding="utf-8")
    restore_slides(current, saved, [3])
    text = current.read_text(encoding="utf-8")
    assert "new first" in text and "third slide" in text and "new third" not in text


@pytest.mark.asyncio
async def test_a_redone_page_that_checks_worse_is_put_back_and_the_better_one_stays(tmp_path: Path, monkeypatch) -> None:
    def problem(page: int, severity: str = "high") -> dict:
        return {"check": "raw_markup", "severity": severity, "page": page, "region": "body",
                "problem": "x", "quote": "some visible words"}

    reports = iter([
        {"transport": "images", "measured": {"findings": []}, "findings": [problem(2), problem(3, "medium")]},
        # after the redo: page 2 fixed, page 3 became worse
        {"transport": "images", "measured": {"findings": []}, "findings": [problem(3, "high")]},
        # after page 3 was put back
        {"transport": "images", "measured": {"findings": []}, "findings": [problem(3, "medium")]},
    ])

    async def fake_check(config, kind, pdf, quest_root, *, supervisor=None):  # noqa: ANN001
        return next(reports)

    monkeypatch.setattr(vc, "check_pdf", fake_check)
    monkeypatch.setattr(vc, "_screenshots", lambda pdf, quest_root, kind: [])
    (tmp_path / "slides.pdf").write_text("pdf", encoding="utf-8")
    (tmp_path / "slides.md").write_text("md", encoding="utf-8")
    restored: list[list[int]] = []

    async def regenerate(feedback: str) -> None:
        assert [f["page"] for f in feedback.findings] == [2, 3]  # the redo is told which pages
        (tmp_path / "slides.md").write_text("md redone", encoding="utf-8")

    async def restore(saved: Path, pages: list[int]) -> None:
        restored.append(pages)

    config = SimpleNamespace(output=SimpleNamespace(visual_check_max_redos=1))
    report = await vc.check_and_redo(config, "slides", tmp_path / "slides.pdf", tmp_path, regenerate, restore_pages=restore)
    assert restored == [[3]]
    assert report["attempts"][1]["restored_pages"] == [3] and report["attempts"][1]["kept"] is True


# ---------- poster ----------


POSTER_REPLY = {
    "headline": "A short headline",
    "blocks": [
        {"type": "heading", "text": "Background"},
        {"type": "text", "text": "First paragraph with \\textbf{raw} markup showing."},
        {"type": "text", "text": "Closing paragraph that is fine."},
    ],
}


@pytest.mark.asyncio
async def test_a_poster_redo_rewrites_only_the_flagged_panel(tmp_path: Path, monkeypatch) -> None:
    art = _art(tmp_path)
    (art.quest_root / ".fi").mkdir()
    (art.quest_root / ".fi" / "poster_reply.txt").write_text(json.dumps(POSTER_REPLY), encoding="utf-8")
    prompts: list[str] = []

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompts.append(messages[-1]["content"])
        return json.dumps({"blocks": {"1": {"type": "text", "text": "First paragraph, fixed."}, "2": {"type": "text", "text": "ignored"}}})

    monkeypatch.setattr("core.provider.LLMClient.chat", fake_chat)
    monkeypatch.setattr("generation.poster.find_pdf_engine", lambda: None)
    feedback = _feedback(1, block="b1")
    await PosterGenerator(_config(tmp_path, ["poster"])).generate(art, art.quest_root, feedback=feedback)
    assert len(prompts) == 1 and "block 1" in prompts[0] and "block 2" not in prompts[0]
    kept = json.loads((art.quest_root / ".fi" / "poster_reply.txt").read_text(encoding="utf-8"))
    assert kept["blocks"][0] == POSTER_REPLY["blocks"][0] and kept["blocks"][2] == POSTER_REPLY["blocks"][2]
    assert kept["blocks"][1]["text"] == "First paragraph, fixed." and kept["headline"] == "A short headline"


@pytest.mark.asyncio
async def test_a_poster_finding_that_matches_no_panel_asks_for_nothing(tmp_path: Path, monkeypatch) -> None:
    art = _art(tmp_path)
    (art.quest_root / ".fi").mkdir()
    (art.quest_root / ".fi" / "poster_reply.txt").write_text(json.dumps(POSTER_REPLY), encoding="utf-8")
    asked: list[str] = []

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        asked.append("called")
        return "{}"

    monkeypatch.setattr("core.provider.LLMClient.chat", fake_chat)
    monkeypatch.setattr("generation.poster.find_pdf_engine", lambda: None)
    result = await PosterGenerator(_config(tmp_path, ["poster"])).generate(
        art, art.quest_root, feedback=_feedback(1, quote="words found nowhere on the poster"))
    assert asked == [] and result == {}


# ---------- JSON re-answer ----------


def _engine(tmp_path: Path, replies: list[str]) -> tuple[Engine, AsyncMock]:
    cfg = Config(
        topic="integrators", title="integrators", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, clarify_mode="off"),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "out"),
    )
    eng = Engine(cfg)
    chat = AsyncMock(side_effect=replies)
    eng._client = type("Stub", (), {"chat": chat})()
    return eng, chat


def test_the_readable_part_of_a_broken_object_is_found_key_by_key() -> None:
    text = '{"verdict": "sufficient", "rationale": "he said "hi" and left", "notes": ["a", "b"]}'
    readable, bad, reason = _readable_json_keys(text)
    assert readable == {"verdict": "sufficient", "notes": ["a", "b"]} and bad == [("rationale",)] and reason
    assert _readable_json_keys("no json at all") is None
    assert _readable_json_keys('{"a": }') is None


@pytest.mark.asyncio
async def test_one_broken_key_is_asked_again_alone_and_merged(tmp_path: Path) -> None:
    broken = '{"verdict": "sufficient", "rationale": "he said "hi" and left", "extra": 3}'
    eng, chat = _engine(tmp_path, [broken, json.dumps({"rationale": "plain words"})])
    text, parsed = await eng._chat_json("judge it", node="evidence_gate", want=("verdict", "rationale"))
    assert parsed == {"verdict": "sufficient", "rationale": "plain words", "extra": 3} and json.loads(text) == parsed
    assert chat.await_count == 2
    retry = chat.await_args_list[1].args[0][0]["content"]
    assert "ONLY for these keys: `rationale`" in retry and "`verdict`" not in retry.split("---", 1)[1]


@pytest.mark.asyncio
async def test_a_wholly_unreadable_reply_is_asked_again_whole(tmp_path: Path) -> None:
    good = json.dumps({"verdict": "sufficient", "rationale": "r"})
    eng, chat = _engine(tmp_path, ["no verdict here", good])
    _, parsed = await eng._chat_json("judge it", node="evidence_gate", want=("verdict", "rationale"))
    assert parsed["verdict"] == "sufficient" and chat.await_count == 2
    assert "Your previous reply to the request above could not be read" in chat.await_args_list[1].args[0][0]["content"]


@pytest.mark.asyncio
async def test_a_cut_off_plan_answer_is_asked_again_only_for_the_cut_key(tmp_path: Path) -> None:
    cut = '{"study_type": "measure", "protocol": {"grid": {"a": [1, 2'
    eng, chat = _engine(tmp_path, [cut, json.dumps({"protocol": {"grid": {"a": [1, 2]}}})])
    parts = await eng._ask_for_plan_parts("ask")
    assert parts["protocol"]["grid"] == {"a": [1, 2]} and chat.await_count == 2
    assert "ONLY for these keys: `protocol`" in chat.await_args_list[1].args[0][0]["content"]


# ---------- findings of the review of the first version ----------


def test_a_break_line_inside_a_code_block_is_code_not_a_slide_break() -> None:
    deck = "---\nmarp: true\n---\n\n# A\n\n```yaml\n---\nk: v\n```\n\n---\n\n# B\n"
    head, segments, breaks = split_deck(deck)
    assert len(segments) == 2 and "k: v" in segments[0] and join_deck(head, segments, breaks) == deck


@pytest.mark.asyncio
async def test_a_slide_that_got_worse_is_put_back_while_the_fixed_ones_stay(tmp_path: Path, monkeypatch) -> None:
    def problem(page: int, severity: str = "high") -> dict:
        return {"check": "raw_markup", "severity": severity, "page": page, "region": "body",
                "problem": "x", "quote": "some visible words"}

    def checked(*findings: dict) -> dict:
        return {"transport": "images", "measured": {"findings": []}, "findings": list(findings)}

    reports = iter([checked(problem(1, "medium"), problem(2), problem(3)), checked(problem(1, "high")),
                    checked(problem(1, "medium"))])

    async def fake_check(config, kind, pdf, quest_root, *, supervisor=None):  # noqa: ANN001
        return next(reports)

    monkeypatch.setattr(vc, "check_pdf", fake_check)
    monkeypatch.setattr(vc, "_screenshots", lambda pdf, quest_root, kind: [])
    deck = "---\nmarp: true\n---\n\n# one old\n\n---\n\n# two old\n\n---\n\n# three old\n"
    (tmp_path / "slides.pdf").write_text("pdf", encoding="utf-8")
    (tmp_path / "slides.md").write_text(deck, encoding="utf-8")

    async def regenerate(feedback: str) -> None:
        (tmp_path / "slides.md").write_text(
            deck.replace("one old", "one new").replace("two old", "two new").replace("three old", "three new"),
            encoding="utf-8")

    async def restore(saved: Path, pages: list[int]) -> None:
        restore_slides(tmp_path / "slides.md", saved / "slides.md", pages)

    config = SimpleNamespace(output=SimpleNamespace(visual_check_max_redos=1))
    report = await vc.check_and_redo(config, "slides", tmp_path / "slides.pdf", tmp_path, regenerate,
                                     restore_pages=restore)
    text = (tmp_path / "slides.md").read_text(encoding="utf-8")
    assert "one old" in text and "two new" in text and "three new" in text
    assert report["attempts"][1]["restored_pages"] == [1] and report["attempts"][1]["kept"] is True


@pytest.mark.asyncio
async def test_the_launch_gives_the_slides_check_the_restore(tmp_path: Path, monkeypatch) -> None:
    import launch as fi_launch

    seen: dict[str, object] = {}

    async def fake_check_and_redo(cfg, kind, pdf, quest_root, regenerate, *, supervisor, **kw):  # noqa: ANN001
        seen[kind] = kw.get("restore_pages")
        return {"kind": kind, "transport": "images", "findings": [], "measured": {"findings": []}}

    class _G:
        def __init__(self, cfg):  # noqa: ANN001
            pass

        async def generate(self, art, out_dir, **kw):  # noqa: ANN001
            return {"slides_pdf": out_dir / "slides.pdf", "paper_pdf": out_dir / "paper.pdf",
                    "poster_pdf": out_dir / "poster.pdf"}

    monkeypatch.setattr(fi_launch, "_apply_paper_venue_override", lambda c, a: None)
    monkeypatch.setattr(fi_launch, "check_and_redo", fake_check_and_redo)
    class _Paper:
        def __init__(self, cfg):  # noqa: ANN001
            pass

        def generate(self, art, out_dir, **kw):  # noqa: ANN001
            return {"paper_pdf": out_dir / "paper.pdf"}

    monkeypatch.setattr(fi_launch, "SlideGenerator", _G)
    monkeypatch.setattr(fi_launch, "PosterGenerator", _G)
    monkeypatch.setattr(fi_launch, "PaperGenerator", _Paper)
    cfg = SimpleNamespace(output=SimpleNamespace(kinds=["paper_pdf", "slides", "poster"], require_pdf=False,
                                                  visual_check=True))
    art = QuestArtifacts(quest_id="q", quest_root=tmp_path, paper_md=tmp_path / "paper.md")
    await fi_launch._run_generators(cfg, art, supervisor=AsyncMock())
    assert callable(seen["slides"]) and seen["poster"] is None and seen["paper"] is None


def test_the_poster_check_is_given_the_parts_and_a_finding_keeps_the_part_it_names(tmp_path: Path) -> None:
    (tmp_path / ".fi").mkdir()
    (tmp_path / ".fi" / "poster_reply.txt").write_text(json.dumps(POSTER_REPLY), encoding="utf-8")
    parts = vc.poster_parts(tmp_path)
    assert set(parts) == {"headline", "b0", "b1", "b2"} and parts["b1"].startswith("text: First paragraph")
    assert "`b1`" in vc._parts_text(parts) and vc._parts_text({}) == ""
    doc = SimpleNamespace(pages=[SimpleNamespace(number=1, lines=[
        SimpleNamespace(text="First paragraph with raw markup showing")])])
    base = {"page": 1, "region": "body", "check": "raw_markup", "problem": "p", "severity": "high",
            "quote": "First paragraph with raw markup"}
    kept, _ = vc.grounded_findings([{**base, "block": "b1"}, {**base, "block": "b9"}], doc, {"raw_markup"}, set(parts))
    assert [f.get("block") for f in kept] == ["b1", None]


@pytest.mark.asyncio
async def test_poster_findings_naming_no_part_are_listed_as_unresolved_not_dropped(tmp_path: Path, monkeypatch) -> None:
    finding = {"check": "raw_markup", "severity": "high", "page": 1, "region": "body", "problem": "unplaced problem",
               "quote": "some visible words"}
    reports = iter([{"transport": "images", "measured": {"findings": []}, "findings": [finding]}])

    async def fake_check(config, kind, pdf, quest_root, *, supervisor=None):  # noqa: ANN001
        return next(reports)

    monkeypatch.setattr(vc, "check_pdf", fake_check)
    (tmp_path / "poster.pdf").write_text("pdf", encoding="utf-8")
    asked: list[str] = []

    async def regenerate(feedback: str) -> None:
        asked.append("called")

    config = SimpleNamespace(output=SimpleNamespace(visual_check_max_redos=2))
    report = await vc.check_and_redo(config, "poster", tmp_path / "poster.pdf", tmp_path, regenerate)
    assert asked == [] and report["attempts"][0]["unresolved"] == ["unplaced problem"]
    assert report["findings"] == [finding], "the finding stays in the report"


def test_a_value_cut_off_at_the_end_of_the_reply_is_not_trusted() -> None:
    readable, bad, _ = _readable_json_keys('{"unit": "m", "threshold": 0.12')
    assert readable == {"unit": "m"} and bad == [("threshold",)]


def test_a_key_written_twice_in_a_broken_reply_is_not_read_at_all() -> None:
    assert _readable_json_keys('{"a": 1, "a": 2, "b": "x "y" z"}') is None


@pytest.mark.asyncio
async def test_one_broken_nested_value_is_asked_by_its_path_and_merged_at_the_leaf(tmp_path: Path) -> None:
    broken = ('{"protocol": {"grid": {"a": [1, 2]}, "tolerance": "within "five" percent", "unit": "m"}, '
              '"study_type": "measure"}')
    eng, chat = _engine(tmp_path, [broken, json.dumps({"protocol": {"tolerance": "5 percent"}})])
    _, parsed = await eng._chat_json("ask", node="evidence_gate", want=("protocol", "study_type"))
    assert parsed == {"protocol": {"grid": {"a": [1, 2]}, "tolerance": "5 percent", "unit": "m"},
                      "study_type": "measure"}
    assert "ONLY for these keys: `protocol.tolerance`" in chat.await_args_list[1].args[0][0]["content"]


def test_an_unplaced_poster_problem_is_listed_in_the_quality_report(tmp_path: Path) -> None:
    from core import quality_report

    (tmp_path / ".fi").mkdir()
    entry = {"attempts": [{"attempt": 0, "score": 3, "kept": True, "unresolved": ["the third column is cut off"]}]}
    (tmp_path / ".fi" / "visual_check.json").write_text(json.dumps({"poster": entry}), encoding="utf-8")
    report = quality_report.build(tmp_path, running=False)
    texts = [i["text"] for stage in report["stages"] for i in stage["problems"]["items"]]
    assert any("the third column is cut off" in t and "poster" in t for t in texts)
    assert "the third column is cut off" in quality_report.render(report)
