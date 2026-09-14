"""Every model call a quest makes lands in ``.fi/cost.jsonl``.

The engine's nodes always logged their calls. The output generators (slides,
poster, talk script, visual check) build their own client and did not, so a
quest's token total left them out. The summary was also written before the
output pass ran. No real model calls here.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import launch
from core.config import (
    Config,
    EngineConfig,
    ExecutionConfig,
    KnowledgeConfig,
    OutputConfig,
    ProviderConfig,
)
from core.engine import Engine, QuestArtifacts
from core.provider import MODEL_PRICING, ResolvedEndpoint, append_cost_row
from generation import _visual_check as vc
from generation.poster import PosterGenerator
from generation.slides import SlideGenerator
from generation.speech import SpeechGenerator

USAGE = {"prompt_tokens": 1200, "completion_tokens": 300, "total_tokens": 1500}


def _config(tmp_path: Path, kinds: list[str], **output) -> Config:
    return Config(
        topic="cost log test",
        title="cost-log-test",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False),
        execution=ExecutionConfig(sandbox="venv", timeout_s=30),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(kinds=kinds, output_dir=tmp_path / "outputs", **output),
    )


def _artifacts(tmp_path: Path) -> QuestArtifacts:
    quest_root = tmp_path / "quest"
    (quest_root / "paper").mkdir(parents=True)
    paper_md = quest_root / "paper" / "paper.md"
    paper_md.write_text("# Toy Title\n\n## Results\n\nThe curve rises.\n", encoding="utf-8")
    (quest_root / "figures").mkdir()
    return QuestArtifacts(
        quest_id="qid-cost", quest_root=quest_root, paper_md=paper_md, figures_dir=quest_root / "figures",
    )


def _model_replying(text: str):
    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        self.last_model = "gpt-4o"
        self.last_usage = dict(USAGE)
        return text

    return fake_chat


def _rows(quest_root: Path) -> list[dict]:
    path = quest_root / ".fi" / "cost.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _fake_endpoint(module: str, monkeypatch: pytest.MonkeyPatch) -> None:
    async def resolve(provider, supervisor):  # noqa: ANN001
        return ResolvedEndpoint(base_url="http://127.0.0.1:1/v1", model="gpt-4o", api_key="not-needed")

    monkeypatch.setattr(f"{module}.resolve_endpoint_async", resolve)


# ---------------------------------------------------------------------------
# The row writer
# ---------------------------------------------------------------------------


def test_a_call_is_logged_with_its_usage_and_its_price(tmp_path: Path) -> None:
    append_cost_row(tmp_path / ".fi", node="slides", model="gpt-4o", usage=USAGE)
    append_cost_row(tmp_path / ".fi", node="poster", model="gpt-4o", usage=USAGE)
    rows = _rows(tmp_path)
    assert [r["node"] for r in rows] == ["slides", "poster"]
    assert set(rows[0]) == {"ts", "node", "model", "usage", "cost_usd"}
    assert rows[0]["usage"] == USAGE and rows[0]["model"] == "gpt-4o"
    price = MODEL_PRICING["gpt-4o"]
    assert rows[0]["cost_usd"] == pytest.approx(1.2 * price["prompt_per_1k"] + 0.3 * price["completion_per_1k"])


def test_a_call_without_usage_still_counts(tmp_path: Path) -> None:
    append_cost_row(tmp_path / ".fi", node="speech", model=None, usage=None)
    assert _rows(tmp_path) == [
        {"ts": _rows(tmp_path)[0]["ts"], "node": "speech", "model": "", "usage": None, "cost_usd": None},
    ]


def test_the_engine_logs_its_nodes_through_the_same_writer(tmp_path: Path) -> None:
    engine = SimpleNamespace(_client=SimpleNamespace(last_usage=dict(USAGE), last_model="gpt-4o"), fi_dir=tmp_path / ".fi")
    Engine._log_chat_cost(engine, node="write")
    assert [(r["node"], r["model"], r["usage"]) for r in _rows(tmp_path)] == [("write", "gpt-4o", USAGE)]


def test_quest_finalization_writes_the_summary(tmp_path: Path) -> None:
    append_cost_row(tmp_path / ".fi", node="write", model="gpt-4o", usage=USAGE)
    Engine._write_cost_summary(SimpleNamespace(fi_dir=tmp_path / ".fi"))
    summary = json.loads((tmp_path / ".fi" / "cost.summary.json").read_text(encoding="utf-8"))
    assert summary["total_tokens"] == 1500 and set(summary["by_node"]) == {"write"}


# ---------------------------------------------------------------------------
# The output generators
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_slide_deck_logs_its_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    art = _artifacts(tmp_path)
    monkeypatch.setattr("core.provider.LLMClient.chat", _model_replying("---\nmarp: true\n---\n\n# Toy\n"))
    monkeypatch.setattr("generation.slides.shutil.which", lambda _n: None)
    await SlideGenerator(_config(tmp_path, ["slides"])).generate(art, art.quest_root)
    rows = _rows(art.quest_root)
    assert [(r["node"], r["model"], r["usage"]) for r in rows] == [("slides", "gpt-4o", USAGE)]


@pytest.mark.asyncio
async def test_the_poster_logs_its_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    art = _artifacts(tmp_path)
    reply = {"headline": "The curve rises", "blocks": [{"type": "heading", "text": "Result"}, {"type": "text", "text": "It rises."}]}
    _fake_endpoint("generation.poster", monkeypatch)
    monkeypatch.setattr("generation.poster.LLMClient.chat", _model_replying(json.dumps(reply)))
    monkeypatch.setattr("generation.poster.find_pdf_engine", lambda: None)
    await PosterGenerator(_config(tmp_path, ["poster"])).generate(art, art.quest_root)
    rows = _rows(art.quest_root)
    assert [(r["node"], r["usage"]) for r in rows] == [("poster", USAGE)]


@pytest.mark.asyncio
async def test_the_talk_script_logs_its_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    art = _artifacts(tmp_path)
    monkeypatch.setattr("core.provider.LLMClient.chat", _model_replying("# Talk\n\n[slide: 1] Hello.\n"))
    await SpeechGenerator(_config(tmp_path, ["speech"])).generate(art, art.quest_root)
    rows = _rows(art.quest_root)
    assert [(r["node"], r["usage"]) for r in rows] == [("speech", USAGE)]


@pytest.mark.asyncio
async def test_the_visual_check_logs_its_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_endpoint("generation._visual_check", monkeypatch)
    monkeypatch.setattr("core.provider.LLMClient.chat", _model_replying('{"findings": []}'))
    message = {"role": "user", "content": [{"type": "text", "text": "Check these pages."}]}
    reply = await vc._ask(_config(tmp_path, ["paper_pdf"]), [message], None, tmp_path)
    assert reply == '{"findings": []}'
    assert [(r["node"], r["usage"]) for r in _rows(tmp_path)] == [("visual_check", USAGE)]


# ---------------------------------------------------------------------------
# The summary
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_output_pass_rewrites_the_summary_the_quest_wrote(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    art = _artifacts(tmp_path)
    fi_dir = art.quest_root / ".fi"
    append_cost_row(fi_dir, node="write", model="gpt-4o", usage=USAGE)
    launch.write_cost_summary(fi_dir)
    before = json.loads((fi_dir / "cost.summary.json").read_text(encoding="utf-8"))
    assert set(before["by_node"]) == {"write"}

    def no_paper(self, art, out_dir):  # noqa: ANN001
        return {}

    async def slides_with_a_call(self, art, out_dir, *, supervisor):  # noqa: ANN001
        append_cost_row(art.quest_root / ".fi", node="slides", model="gpt-4o", usage=USAGE)
        return {}

    async def nothing(self, art, out_dir, *, supervisor):  # noqa: ANN001
        return {}

    monkeypatch.setattr("launch.PaperGenerator.generate", no_paper)
    monkeypatch.setattr("launch.SlideGenerator.generate", slides_with_a_call)
    monkeypatch.setattr("launch.PosterGenerator.generate", nothing)
    monkeypatch.setattr("launch.SpeechGenerator.generate", nothing)
    await launch._run_generators(_config(tmp_path, ["slides"], visual_check=False), art, supervisor=None)

    after = json.loads((fi_dir / "cost.summary.json").read_text(encoding="utf-8"))
    assert set(after["by_node"]) == {"write", "slides"}
    assert after["total_requests"] == 2 and after["total_tokens"] == 3000
