"""Slides, poster and talk are given the whole final paper and its checked claims, never the first 8,000 characters."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import generation.poster as poster_mod
from generation._paper_context import checked_claims, note_prompt_fit, paper_and_claims
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


def _write_paper(art, with_claims: bool) -> None:
    art.paper_md.write_text("# Pond Study\n\n" + _BODY, encoding="utf-8")
    if with_claims:
        (art.quest_root / "paper").mkdir(parents=True, exist_ok=True)
        (art.quest_root / "paper" / "claims.json").write_text(json.dumps({"claims": [
            {"claim": "Settling time grows with pond depth", "basis": "experiment", "scope": "depths 1 to 4 m"},
            {"claim": "Settling is faster in winter", "basis": "citation"},
            {"claim": "Algae bloom in every pond", "basis": "unsupported"},
        ]}), encoding="utf-8")


def _assert_whole(prompt: str, with_claims: bool) -> None:
    assert len(_BODY) > 8000
    assert _TAIL in prompt
    if with_claims:
        assert "The claims you may state" in prompt
        assert "- Settling time grows with pond depth (holds for: depths 1 to 4 m)" in prompt
        assert "- Settling is faster in winter" in prompt
        assert "Algae bloom in every pond" not in prompt  # the ledger marks it unsupported
        assert prompt.index("state as a finding only") < prompt.index("# Your task")  # the rule comes first
    else:
        assert "The claims you may state" not in prompt
        assert "state as a finding only" not in prompt
    assert "$claims" not in prompt and "$paper_md" not in prompt


def test_the_helper_reads_the_ledger_and_survives_its_absence(tmp_path: Path) -> None:
    art = _make_artifacts(tmp_path, with_figure=False)
    _write_paper(art, with_claims=False)
    assert checked_claims(art.quest_root) == []
    ctx = paper_and_claims("whole", art.quest_root)
    assert (ctx.paper, ctx.claims_rule, ctx.claims, ctx.n_claims) == ("whole", "", "", 0)
    (art.quest_root / "paper" / "claims.json").write_text("{broken", encoding="utf-8")
    assert checked_claims(art.quest_root) == []


def test_a_prompt_over_the_cap_is_disclosed(tmp_path: Path) -> None:
    (tmp_path / ".fi").mkdir()
    assert note_prompt_fit("x" * 50, 0, "poster", tmp_path) is None
    assert note_prompt_fit("x" * 50, 100, "poster", tmp_path) is None
    rec = note_prompt_fit("x" * 150, 100, "poster", tmp_path)
    assert rec and rec["prompt_chars"] == 150
    assert "poster" in json.loads((tmp_path / ".fi" / "prompt_fit.json").read_text())
    assert "cap is 100" in (tmp_path / ".fi" / "run.log").read_text()


@pytest.mark.asyncio
@pytest.mark.parametrize("with_claims", [True, False])
async def test_slides_prompt_holds_the_whole_paper(tmp_path, monkeypatch, with_claims) -> None:
    art = _make_artifacts(tmp_path, with_figure=False)
    _write_paper(art, with_claims)
    seen: list[str] = []

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        seen.append(messages[-1]["content"])
        return _FENCED_MARP

    monkeypatch.setattr("core.provider.LLMClient.chat", fake_chat)
    monkeypatch.setattr("generation.slides.shutil.which", lambda _n: None)
    await SlideGenerator(_make_config(tmp_path, ["slides"])).generate(art, art.quest_root)
    _assert_whole(seen[0], with_claims)


@pytest.mark.asyncio
@pytest.mark.parametrize("with_claims", [True, False])
async def test_speech_prompt_holds_the_whole_paper(tmp_path, monkeypatch, with_claims) -> None:
    art = _make_artifacts(tmp_path, with_figure=False)
    _write_paper(art, with_claims)
    seen: list[str] = []

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        seen.append(messages[-1]["content"])
        return _PLAIN_TALK

    monkeypatch.setattr("core.provider.LLMClient.chat", fake_chat)
    await SpeechGenerator(_make_config(tmp_path, ["speech"])).generate(art, art.quest_root)
    _assert_whole(seen[0], with_claims)


@pytest.mark.asyncio
@pytest.mark.parametrize("with_claims", [True, False])
async def test_poster_prompt_holds_the_whole_paper(tmp_path, monkeypatch, with_claims) -> None:
    art = _poster_art(tmp_path)
    _write_paper(art, with_claims)
    seen: list[str] = []

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        seen.append(messages[0]["content"])
        return json.dumps(_reply())

    _patch_endpoint(monkeypatch)
    monkeypatch.setattr("generation.poster.LLMClient.chat", fake_chat)
    _no_engine(monkeypatch)
    await PosterGenerator(_poster_cfg(tmp_path, kinds=["poster"])).generate(art, art.quest_root)
    _assert_whole(seen[0], with_claims)
