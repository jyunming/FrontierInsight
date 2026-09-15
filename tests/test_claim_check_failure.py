"""A claim check that could not run, and papers longer than the old cut.

Two things a real quest (2026-09-15) exposed:

* The claim check's model call failed on a server with no capacity, and the
  node returned nothing. The rewrite kept the previous draft's grounding, the
  review never learned that no citation had been checked, and 27 unsupported
  citation numbers went through. A failed check is now recorded and the review
  forces ``citations_unchecked``, which sends the paper back to be written and
  checked again.
* The review and the claim check read the first 16,000 characters of a
  34,910-character paper. Both now read the whole paper.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig,
)
from core.engine import (
    _PAPER_PROMPT_CHARS,
    Engine,
    _citations_unchecked,
    _format_claim_grounding,
    _hits_need_only_a_rewrite,
    _paper_for_prompt,
)

FAILED = "_CliCapacityError: antigravity: API error: UNAVAILABLE (code 503): No capacity available"


def _engine(tmp_path: Path) -> Engine:
    eng = Engine(Config(
        topic="t", title="t", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=2, review_loop=False),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "out"),
    ))
    eng.quest_root = tmp_path  # type: ignore[attr-defined]
    eng.fi_dir = tmp_path / ".fi"  # type: ignore[attr-defined]
    return eng


def _long_paper(tmp_path: Path) -> Path:
    body = "# Title\n\n## Results\n\n" + ("The outbreak probability rises with R0. " * 900)
    body += "\n\n## Discussion\n\nTHE LAST SENTENCE OF THE DISCUSSION.\n"
    assert len(body) > 30_000
    paper = tmp_path / "paper.md"
    paper.write_text(body, encoding="utf-8")
    return paper


def test_a_failed_claim_check_is_a_forced_rewrite() -> None:
    hits = _citations_unchecked({"claim_check_failed": FAILED})  # type: ignore[typeddict-item]
    assert len(hits) == 1 and hits[0].startswith("citations_unchecked:")
    assert _hits_need_only_a_rewrite(hits)
    assert _citations_unchecked({"claim_check_failed": ""}) == []  # type: ignore[typeddict-item]
    assert _citations_unchecked({}) == []  # type: ignore[typeddict-item]


def test_the_review_prompt_says_the_check_failed() -> None:
    out = _format_claim_grounding({"claim_check_failed": FAILED, "claim_grounding": {"grounded": 3, "total": 3}})  # type: ignore[typeddict-item]
    assert "FAILED" in out and "No capacity" in out
    assert "3/3" not in out, "a failed check must not show an earlier draft's grounding"


def test_claim_check_failure_is_recorded_and_success_clears_it(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    paper = tmp_path / "paper.md"
    paper.write_text("# T\n\nA claim [1].\n", encoding="utf-8")

    async def boom(prompt, *, node=""):  # noqa: ANN001, ARG001
        raise RuntimeError("UNAVAILABLE (code 503): No capacity available")

    eng._chat = boom  # type: ignore[assignment,method-assign]
    stale = {"grounded": 5, "total": 5, "unsupported": []}
    out = asyncio.run(eng._node_claim_check({  # type: ignore[arg-type]
        "topic": "t", "paper_md": str(paper), "literature": [], "claim_grounding": stale,
    }))
    assert out["claim_grounding"] == {}, "the previous draft's grounding is cleared"
    assert "No capacity" in out["claim_check_failed"]

    async def ok(prompt, *, node=""):  # noqa: ANN001, ARG001
        return json.dumps({"claims": []})

    eng._chat = ok  # type: ignore[assignment,method-assign]
    out = asyncio.run(eng._node_claim_check({  # type: ignore[arg-type]
        "topic": "t", "paper_md": str(paper), "literature": [], "claim_check_failed": FAILED,
    }))
    assert out["claim_check_failed"] == ""


@pytest.mark.parametrize("panel", [False, True])
def test_review_forces_citations_unchecked_even_when_the_reviewer_accepts(tmp_path: Path, panel: bool) -> None:
    eng = _engine(tmp_path)
    if panel:
        eng.config.engine.review_panel = ["methodologist", "statistician"]

    async def accepting(prompt, *, node=None):  # noqa: ANN001, ARG001
        if node == "review_moderator":
            return json.dumps({"rationale": "both accept"})
        return json.dumps({"verdict": "accept", "score": 5, "suggestions": [], "must_flag_hits": []})

    eng._chat = accepting  # type: ignore[assignment]
    paper = tmp_path / "paper.md"
    paper.write_text("# T\n\nA claim [1].\n", encoding="utf-8")
    patch = asyncio.run(eng._node_review({  # type: ignore[arg-type]
        "topic": "t", "iteration": 0, "review": {}, "paper_md": str(paper),
        "claim_check_failed": FAILED,
    }))
    hits = patch["review"]["must_flag_hits"]
    assert any(str(h).startswith("citations_unchecked") for h in hits), hits
    assert patch["iteration"] == 1
    assert eng._route_after_review({"review": patch["review"], "iteration": 1}) == "rewrite"  # type: ignore[arg-type]


def test_review_without_a_failure_is_not_forced(tmp_path: Path) -> None:
    eng = _engine(tmp_path)

    async def accepting(prompt, *, node=None):  # noqa: ANN001, ARG001
        return json.dumps({"verdict": "accept", "score": 5, "suggestions": [], "must_flag_hits": []})

    eng._chat = accepting  # type: ignore[assignment]
    paper = tmp_path / "paper.md"
    paper.write_text("# T\n\nA claim [1].\n", encoding="utf-8")
    patch = asyncio.run(eng._node_review({  # type: ignore[arg-type]
        "topic": "t", "iteration": 0, "review": {}, "paper_md": str(paper), "claim_check_failed": "",
    }))
    assert not any(str(h).startswith("citations_unchecked") for h in patch["review"]["must_flag_hits"])


def test_review_and_claim_check_read_the_whole_paper(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    paper = _long_paper(tmp_path)
    prompts: dict[str, str] = {}

    async def capture(prompt, *, node=None):  # noqa: ANN001
        prompts[node or ""] = prompt
        if node == "claim_check":
            return json.dumps({"claims": []})
        return json.dumps({"verdict": "accept", "score": 4, "suggestions": [], "must_flag_hits": []})

    eng._chat = capture  # type: ignore[assignment]
    asyncio.run(eng._node_claim_check({"topic": "t", "paper_md": str(paper), "literature": []}))  # type: ignore[arg-type]
    asyncio.run(eng._node_review({"topic": "t", "iteration": 0, "review": {}, "paper_md": str(paper)}))  # type: ignore[arg-type]
    assert "THE LAST SENTENCE OF THE DISCUSSION." in prompts["claim_check"]
    assert "THE LAST SENTENCE OF THE DISCUSSION." in prompts["review"]


def test_a_runaway_paper_is_cut_with_a_note() -> None:
    long = "x" * (_PAPER_PROMPT_CHARS + 50)
    out = _paper_for_prompt(long, "review")
    assert out.startswith("x" * 100)
    assert "only its first" in out and len(out) < len(long) + 200
    assert _paper_for_prompt("short", "review") == "short"
