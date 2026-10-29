"""Slides, poster and talk are given the whole final paper and its checked claims, never the first 8,000 characters."""
from __future__ import annotations

import json
import string
from pathlib import Path
from types import SimpleNamespace

import pytest

from core import receipts as _receipts
from generation._paper_context import paper_and_claims
from generation.poster import PosterGenerator
from generation.slides import SlideGenerator
from generation.speech import SpeechGenerator
from tests.test_poster import _make_artifacts as _poster_art
from tests.test_poster import _make_config as _poster_cfg
from tests.test_poster import _no_engine, _patch_endpoint, _reply
from tests.test_slides_speech import _FENCED_MARP, _PLAIN_TALK, _make_artifacts, _make_config

# A neutral made-up paper: the tail (past 8,000 characters) holds the sentence that must reach the model.
_TAIL = "Dilution slows the settling of the pale mineral beyond the range tested."
_BODY = "Filler sentence about a toy pond study. " * 400 + _TAIL + "\n"
_CLAIMS = [
    {"claim": "Settling time grows with pond depth", "basis": "experiment"},
    {"claim": "Settling is faster in winter", "basis": "citation"},
    {"claim": "Algae bloom in every pond", "basis": "unsupported"},
]


def _write_paper(art, ledger: str) -> None:
    """ledger: none | ok | stale (the receipt judged another draft) | failed | unsupported_only."""
    art.paper_md.write_text("# Pond Study\n\n" + _BODY, encoding="utf-8")
    if ledger == "none":
        return
    claims = _CLAIMS[2:] if ledger == "unsupported_only" else _CLAIMS
    (art.quest_root / "paper").mkdir(parents=True, exist_ok=True)
    (art.quest_root / "paper" / "claims.json").write_text(json.dumps({"claims": claims, "status": "ok"}), encoding="utf-8")
    judged = b"an earlier draft" if ledger == "stale" else art.paper_md.read_bytes()
    _receipts.write(
        art.quest_root, "claim_check", status="unknown" if ledger == "failed" else "pass", producer="claim_check",
        started_at=_receipts.now(), inputs={"paper": judged}, output={"claims": claims},
        error="the call failed" if ledger == "failed" else "",
    )


def _assert_prompt(prompt: str, ledger: str, art) -> dict:
    assert len(_BODY) > 8000 and _TAIL in prompt  # the whole paper, past the old cut
    assert "$claims" not in prompt and "$paper_md" not in prompt
    if ledger == "ok":
        assert "- Settling time grows with pond depth\n" in prompt and "- Settling is faster in winter" in prompt
        assert "Algae bloom in every pond" not in prompt  # unsupported
        assert "holds for" not in prompt
        # the rule is the first thing, the claim list stands before the paper
        assert prompt.index("state as a finding only") < prompt.index("# Your task")
        assert prompt.index("The claims you may state") < prompt.index(_TAIL)
    elif ledger == "unsupported_only":
        assert "no claim of this paper was confirmed" in prompt
        assert "The claims you may state" not in prompt and "Algae bloom" not in prompt
    else:  # no ledger, a ledger of another draft, or of a failed check: the whole paper, no list, no rule
        assert "The claims you may state" not in prompt and "Settling time grows" not in prompt
        assert "state as a finding only" not in prompt and "no claim of this paper was confirmed" not in prompt
    return json.loads((art.quest_root / ".fi" / "prompt_fit.json").read_text(encoding="utf-8"))


_LEDGERS = ["ok", "none", "stale", "failed", "unsupported_only"]


def test_the_helper_needs_the_paper_the_receipt_judged(tmp_path: Path) -> None:
    art = _make_artifacts(tmp_path, with_figure=False)
    _write_paper(art, "ok")
    assert paper_and_claims("w", art.quest_root, art.paper_md).ledger == "used"
    art.paper_md.write_text("# Pond Study\n\nrewritten after the check\n", encoding="utf-8")
    ctx = paper_and_claims("w", art.quest_root, art.paper_md)
    assert (ctx.ledger, ctx.claims_rule, ctx.claims) == ("stale", "", "")
    (art.quest_root / "paper" / "claims.json").write_text("{broken", encoding="utf-8")
    assert paper_and_claims("w", art.quest_root, art.paper_md).ledger == "none"


@pytest.mark.asyncio
@pytest.mark.parametrize("ledger", _LEDGERS)
async def test_slides_prompt_holds_the_whole_paper(tmp_path, monkeypatch, ledger) -> None:
    art = _make_artifacts(tmp_path, with_figure=False)
    _write_paper(art, ledger)
    seen: list[str] = []

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        seen.append(messages[-1]["content"])
        return _FENCED_MARP

    monkeypatch.setattr("core.provider.LLMClient.chat", fake_chat)
    monkeypatch.setattr("generation.slides.shutil.which", lambda _n: None)
    await SlideGenerator(_make_config(tmp_path, ["slides"])).generate(art, art.quest_root)
    fit = _assert_prompt(seen[0], ledger, art)
    # the record matches the prompt the call really received; no cap applies, so nothing is said about a cut
    assert fit["slides"]["prompt_chars"] == len(seen[0]) and "paper_trimmed_by_provider" not in fit["slides"]


@pytest.mark.asyncio
@pytest.mark.parametrize("ledger", _LEDGERS)
async def test_speech_prompt_holds_the_whole_paper(tmp_path, monkeypatch, ledger) -> None:
    art = _make_artifacts(tmp_path, with_figure=False)
    _write_paper(art, ledger)
    seen: list[str] = []

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        seen.append(messages[-1]["content"])
        return _PLAIN_TALK

    monkeypatch.setattr("core.provider.LLMClient.chat", fake_chat)
    await SpeechGenerator(_make_config(tmp_path, ["speech"])).generate(art, art.quest_root)
    fit = _assert_prompt(seen[0], ledger, art)
    assert fit["speech"]["prompt_chars"] == len(seen[0])


@pytest.mark.asyncio
@pytest.mark.parametrize("ledger", _LEDGERS)
async def test_poster_prompt_holds_the_whole_paper(tmp_path, monkeypatch, ledger) -> None:
    art = _poster_art(tmp_path)
    _write_paper(art, ledger)
    seen: list[str] = []

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        seen.append(messages[0]["content"])
        return json.dumps(_reply())

    _patch_endpoint(monkeypatch)
    monkeypatch.setattr("generation.poster.LLMClient.chat", fake_chat)
    _no_engine(monkeypatch)
    await PosterGenerator(_poster_cfg(tmp_path, kinds=["poster"])).generate(art, art.quest_root)
    fit = _assert_prompt(seen[0], ledger, art)
    assert fit["poster"]["prompt_chars"] == len(seen[0])


@pytest.mark.asyncio
async def test_a_stale_ledger_is_said_in_run_log_and_the_record(tmp_path, monkeypatch) -> None:
    art = _make_artifacts(tmp_path, with_figure=False)
    (art.quest_root / ".fi").mkdir(parents=True, exist_ok=True)
    _write_paper(art, "stale")

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        return _PLAIN_TALK

    monkeypatch.setattr("core.provider.LLMClient.chat", fake_chat)
    await SpeechGenerator(_make_config(tmp_path, ["speech"])).generate(art, art.quest_root)
    fit = json.loads((art.quest_root / ".fi" / "prompt_fit.json").read_text(encoding="utf-8"))
    assert fit["speech"]["ledger"] == "stale" and fit["speech"]["claims_listed"] == 0
    log = (art.quest_root / ".fi" / "run.log").read_text(encoding="utf-8")
    assert "not the claim check's result for this paper.md" in log


@pytest.mark.asyncio
async def test_the_redo_prompts_carry_the_rule_and_the_claims(tmp_path, monkeypatch) -> None:
    art = _make_artifacts(tmp_path, with_figure=False)
    _write_paper(art, "ok")
    (art.quest_root / "slides.md").write_text("---\nmarp: true\n---\n\n# One\n\n---\n\n# Two\n", encoding="utf-8")
    seen: list[str] = []

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        seen.append(messages[-1]["content"])
        return "# One again"

    monkeypatch.setattr("core.provider.LLMClient.chat", fake_chat)

    class _Feedback(SimpleNamespace):
        def __str__(self) -> str:
            return "page 1 overflows"

    feedback = _Feedback(findings=[{"page": 1}])
    await SlideGenerator(_make_config(tmp_path, ["slides"]))._redo_flagged(art, art.quest_root, feedback, supervisor=None)
    assert seen and "state as a finding only" in seen[0] and "- Settling is faster in winter" in seen[0]
    # the poster's redo template takes the same slots
    poster_redo = (Path(__file__).resolve().parent.parent / "agents" / "poster_redo.md").read_text(encoding="utf-8")
    assert "$claims_rule" in poster_redo and "$claims" in poster_redo
    string.Template(poster_redo).substitute(
        poster="P", figure_list="F", source_list="S", feedback="X", targets="T", claims_rule="R", claims="C")
