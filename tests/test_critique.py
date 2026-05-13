"""Direct tests for `core.critique`.

The critique module is a single-shot adversarial review wrapper over
an already-completed quest. Tests cover the quest-resolution helper
(quest_id validation + path-traversal refusal), artifact loading
across both quest-dir layouts (post- and pre-Phase-O), the artifact
truncation cap, and the end-to-end with a mocked LLM.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.config import ProviderConfig
from core.critique import (
    QuestArtifactsForCritique,
    _build_critique_prompt,
    _load_quest_artifacts,
    _read_capped,
    _resolve_quest_dir,
    generate_critique,
)


def _make_quest_with_paper(
    outputs: Path,
    quest_id: str,
    *,
    paper_body: str = "# A toy paper\n\nIt computes things.",
    code: str | None = None,
    prior_review: str | None = None,
    provider: str | None = None,
    layout: str = "post_phase_o",
) -> Path:
    """Create a fake completed quest dir with the chosen artifact set.
    ``layout`` is one of ``"post_phase_o"`` (paper under
    ``paper/paper.md``) or ``"pre_phase_o"`` (paper at quest_root/paper.md)
    to exercise both probing paths."""
    qdir = outputs / quest_id
    qdir.mkdir(parents=True, exist_ok=True)
    if layout == "post_phase_o":
        paper_dir = qdir / "paper"
        paper_dir.mkdir(exist_ok=True)
        (paper_dir / "paper.md").write_text(paper_body, encoding="utf-8")
        if prior_review is not None:
            (paper_dir / "review.md").write_text(prior_review, encoding="utf-8")
    else:
        (qdir / "paper.md").write_text(paper_body, encoding="utf-8")
        if prior_review is not None:
            (qdir / "review.md").write_text(prior_review, encoding="utf-8")
    if code is not None:
        code_dir = qdir / "code"
        code_dir.mkdir(exist_ok=True)
        (code_dir / "experiment.py").write_text(code, encoding="utf-8")
    if provider is not None:
        (qdir / "frontier_insight_summary.json").write_text(
            json.dumps({"provider": provider, "quest_id": quest_id}),
            encoding="utf-8",
        )
    return qdir


# ---------- quest resolution ------------------------------------------------


def test_resolve_quest_dir_happy_path(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    qid = "1700000700-toy-aabbcc"
    _make_quest_with_paper(outputs, qid)
    resolved = _resolve_quest_dir(outputs, qid)
    assert resolved == (outputs / qid).resolve()


def test_resolve_quest_dir_refuses_malformed_quest_id(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    for bad in ["not-a-quest-id", "../../etc/passwd", "_drafts", ""]:
        with pytest.raises(ValueError):
            _resolve_quest_dir(outputs, bad)


def test_resolve_quest_dir_refuses_unknown_quest(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    with pytest.raises(FileNotFoundError):
        _resolve_quest_dir(outputs, "1700000700-missing-aabbcc")


# ---------- _read_capped ----------------------------------------------------


def test_read_capped_returns_full_when_under_limit(tmp_path: Path) -> None:
    p = tmp_path / "small.txt"
    p.write_text("hello", encoding="utf-8")
    assert _read_capped(p, limit=100) == "hello"


def test_read_capped_truncates_with_marker(tmp_path: Path) -> None:
    p = tmp_path / "big.txt"
    p.write_text("a" * 1000, encoding="utf-8")
    out = _read_capped(p, limit=100)
    assert out.startswith("a" * 100)
    assert "truncated" in out
    assert "original 1000 chars" in out


def test_read_capped_missing_file_returns_empty(tmp_path: Path) -> None:
    assert _read_capped(tmp_path / "missing.txt", limit=100) == ""


# ---------- _load_quest_artifacts ------------------------------------------


def test_load_artifacts_post_phase_o_layout(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    qid = "1700000700-x-aabbcc"
    _make_quest_with_paper(
        outputs, qid,
        paper_body="# Title\n\nAbstract paragraph here.",
        code="import numpy\nprint('hi')\n",
        prior_review="# Review\n\nrating: accept",
        provider="openai",
    )
    art = _load_quest_artifacts(outputs / qid)
    assert art.quest_id == qid
    assert "Abstract paragraph" in art.paper_md
    assert "import numpy" in art.code
    assert "rating: accept" in art.prior_review
    assert art.quest_provider == "openai"
    assert art.paper_md_path is not None
    assert art.paper_md_path.name == "paper.md"
    assert "paper" in art.paper_md_path.parts


def test_load_artifacts_pre_phase_o_layout(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    qid = "1700000700-x-aabbcc"
    _make_quest_with_paper(
        outputs, qid, layout="pre_phase_o",
        paper_body="# Older title\n\nOlder abstract.",
    )
    art = _load_quest_artifacts(outputs / qid)
    assert "Older abstract" in art.paper_md
    # Resolved to quest_root/paper.md, not paper/paper.md.
    assert art.paper_md_path is not None
    assert art.paper_md_path.name == "paper.md"
    assert "paper" not in art.paper_md_path.parent.name


def test_load_artifacts_handles_missing_pieces(tmp_path: Path) -> None:
    """A quest can lack code and/or prior_review (interrupted runs,
    legacy quests); critique should still load with empty strings for
    the missing artifacts."""
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    qid = "1700000700-x-aabbcc"
    _make_quest_with_paper(outputs, qid)   # no code, no review, no summary json
    art = _load_quest_artifacts(outputs / qid)
    assert art.paper_md != ""
    assert art.code == ""
    assert art.prior_review == ""
    assert art.code_path is None
    assert art.prior_review_path is None
    assert art.quest_provider is None


def test_load_artifacts_resolves_moderator_review(tmp_path: Path) -> None:
    """When the quest used a reviewer panel (review_moderate.md exists
    instead of review.md), critique should still find the synthesis."""
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    qid = "1700000700-x-aabbcc"
    qdir = _make_quest_with_paper(outputs, qid)
    # No review.md, only review_moderate.md.
    (qdir / "paper" / "review_moderate.md").write_text(
        "## Moderator synthesis\n\nThe panel says: revise.\n",
        encoding="utf-8",
    )
    art = _load_quest_artifacts(qdir)
    assert "panel says: revise" in art.prior_review
    assert art.prior_review_path is not None
    assert art.prior_review_path.name == "review_moderate.md"


# ---------- prompt build ----------------------------------------------------


def test_build_critique_prompt_includes_quest_id_and_artifacts(tmp_path: Path) -> None:
    art = QuestArtifactsForCritique(
        quest_id="1700000700-x-aabbcc",
        quest_root=tmp_path,
        paper_md="# Toy paper\n\nclaim 1.\n",
        paper_md_path=tmp_path / "paper.md",
        code="x = 1\n",
        code_path=tmp_path / "code.py",
        prior_review="all good",
        prior_review_path=tmp_path / "review.md",
        quest_provider="openai",
    )
    prompt = _build_critique_prompt(
        art, critique_provider="claude_cli",
        generated_at=datetime(2026, 5, 13, tzinfo=timezone.utc),
    )
    assert "1700000700-x-aabbcc" in prompt
    assert "claude_cli" in prompt
    assert "openai" in prompt
    assert "Toy paper" in prompt
    assert "x = 1" in prompt
    assert "all good" in prompt


def test_build_critique_prompt_marks_missing_artifacts(tmp_path: Path) -> None:
    """When a code file or prior_review wasn't found, the prompt
    section says "(not found in quest directory)" so the LLM doesn't
    hallucinate content."""
    art = QuestArtifactsForCritique(
        quest_id="1700000700-x-aabbcc",
        quest_root=tmp_path,
        paper_md="paper",
        paper_md_path=tmp_path / "paper.md",
        code="", code_path=None,
        prior_review="", prior_review_path=None,
        quest_provider=None,
    )
    prompt = _build_critique_prompt(
        art, critique_provider="openai",
        generated_at=datetime(2026, 5, 13, tzinfo=timezone.utc),
    )
    assert "(not found in quest directory)" in prompt


# ---------- end-to-end ------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_critique_writes_critique_md_and_returns_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    qid = "1700000700-x-aabbcc"
    _make_quest_with_paper(
        outputs, qid,
        paper_body="# Paper\n\nThe abstract.",
        code="x = 1\n",
        prior_review="rating: accept",
        provider="openai",
    )

    captured: dict[str, str] = {}

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        captured["prompt"] = messages[-1]["content"]
        captured["node"] = kw.get("node", "")
        return "# Critique\n\n## Verdict\nreject — toy result.\n"

    monkeypatch.setattr("core.provider.LLMClient.chat", fake_chat)

    art = await generate_critique(
        qid, outputs,
        provider=ProviderConfig(name="claude_cli"),
        knowledge=None,
    )

    assert art.critique_path == (outputs / qid / "critique.md").resolve()
    assert art.critique_path.is_file()
    body = art.critique_path.read_text(encoding="utf-8")
    assert body.startswith("# Critique")
    assert art.critique_provider == "claude_cli"
    assert art.ingested_to_axon is False
    assert captured["node"] == "critique"
    # Prompt contains both providers (the cross-provider comparison framing).
    assert "openai" in captured["prompt"]
    assert "claude_cli" in captured["prompt"]


@pytest.mark.asyncio
async def test_generate_critique_raises_on_missing_paper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    qid = "1700000700-x-aabbcc"
    # Quest dir exists but contains no paper.md (e.g. a quest that
    # crashed during ideate). Critique should fail loudly.
    (outputs / qid).mkdir()
    # Patch chat so we'd know if it was incorrectly invoked.
    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        raise AssertionError("LLM should not be called when paper.md is missing")
    monkeypatch.setattr("core.provider.LLMClient.chat", fake_chat)

    with pytest.raises(FileNotFoundError):
        await generate_critique(
            qid, outputs,
            provider=ProviderConfig(name="openai"),
            knowledge=None,
        )


@pytest.mark.asyncio
async def test_generate_critique_rejects_malformed_quest_id(
    tmp_path: Path,
) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    with pytest.raises(ValueError):
        await generate_critique(
            "not-a-quest-id", outputs,
            provider=ProviderConfig(name="openai"),
            knowledge=None,
        )


@pytest.mark.asyncio
async def test_generate_critique_ingests_to_axon_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    qid = "1700000700-x-aabbcc"
    _make_quest_with_paper(outputs, qid, provider="openai")

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        return "# Critique body\n"

    monkeypatch.setattr("core.provider.LLMClient.chat", fake_chat)

    knowledge = MagicMock()
    knowledge.enabled = True
    knowledge.add_text = MagicMock(return_value=True)

    art = await generate_critique(
        qid, outputs,
        provider=ProviderConfig(name="claude_cli"),
        knowledge=knowledge,
    )
    assert art.ingested_to_axon is True
    knowledge.add_text.assert_called_once()
    kwargs = knowledge.add_text.call_args.kwargs
    assert kwargs["kind"] == "fi_critique"
    assert kwargs["metadata"]["quest_id"] == qid
    assert kwargs["metadata"]["critique_provider"] == "claude_cli"
    assert kwargs["metadata"]["quest_provider"] == "openai"
