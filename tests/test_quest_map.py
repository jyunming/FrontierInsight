"""The quest map on the web quest page: every graph node in big blocks, the reached ones clickable, each with a plain
sentence, what a restart keeps and redoes, and the command line for it (core/rerun_from.py::node_map, the
``rerun-steps`` API, web/static/quest.html). No test calls a real model."""

from __future__ import annotations

from pathlib import Path

import pytest

from core import rerun_from
from core.engine import Engine

try:
    from fastapi.testclient import TestClient
except Exception:  # pragma: no cover
    pytest.skip("fastapi/httpx not installed", allow_module_level=True)

import yaml

from tests.test_rerun_from import _cfg
from web.server import make_app


def _steps(reached: set[str]) -> list[dict]:
    return [rerun_from.step_info(step, reached=step in reached) for step in rerun_from.STEPS]


def test_map_lists_exactly_the_graph_nodes(tmp_path: Path) -> None:
    graph_nodes = set(Engine(_cfg(tmp_path))._build_graph().nodes)
    assert set(rerun_from.NODES) == graph_nodes
    assert list(rerun_from.NODES.values()) and all(block in rerun_from.GROUPS for block, _ in rerun_from.NODES.values())
    assert all(sentence for _, sentence in rerun_from.NODES.values())


def test_only_reached_nodes_with_a_step_are_clickable() -> None:
    nodes = {n["node"]: n for n in rerun_from.node_map(_steps({"ideas", "literature", "plan", "design", "code", "run"}))}
    for name in ("ideate", "literature", "plan", "design", "implement_outline", "implement", "execute"):
        assert nodes[name]["clickable"] and nodes[name]["redoes"] and nodes[name]["outputs"]
    # Not reached: greyed.
    assert not nodes["analyze"]["clickable"] and not nodes["write"]["clickable"]
    # Not a step (a pause, the repair loop, a person's decision): greyed even when everything was reached.
    everything = {n["node"]: n for n in rerun_from.node_map(_steps(set(rerun_from.STEPS)))}
    for name in ("clarify", "pause_after_literature", "wait_for_data", "execute_reflect", "human_feedback"):
        assert not everything[name]["clickable"] and everything[name]["step"] == ""
    assert everything["web_figures"]["clickable"] and everything["web_figures"]["step"] == "figures"


def test_restart_at_or_before_design_needs_approval_and_later_does_not() -> None:
    nodes = {n["node"]: n for n in rerun_from.node_map(_steps(set(rerun_from.STEPS)))}
    for name in ("ideate", "literature", "plan", "design"):
        assert nodes[name]["needs_approval"], name
    for name in ("select_skills", "implement", "execute", "analyze", "write", "review"):
        assert not nodes[name]["needs_approval"], name


def test_a_node_without_a_backup_list_or_sentence_is_greyed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(rerun_from.REDOES, "analysis")
    monkeypatch.delitem(rerun_from.OUTPUTS, "review")
    steps = [{"name": s, "sentence": "x", "needs_approval": False, "reached": True, "outputs": ["a"], "group": ""}
             for s in rerun_from.STEPS]
    nodes = {n["node"]: n for n in rerun_from.node_map(steps)}
    assert not nodes["analyze"]["clickable"] and not nodes["review"]["clickable"]
    assert nodes["write"]["clickable"]


def test_rerun_steps_api_returns_the_map(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg(tmp_path)
    quest_root = cfg.output.output_dir / "qmap1"
    (quest_root / ".fi").mkdir(parents=True)
    (quest_root / ".fi" / "state.sqlite").write_bytes(b"")
    (quest_root / "config.yaml").write_text(yaml.safe_dump(cfg.model_dump(mode="json")), encoding="utf-8")

    async def fake_steps(self) -> list[dict]:
        return _steps({"ideas", "literature", "plan", "design", "code"})

    monkeypatch.setattr(Engine, "rerun_steps", fake_steps)
    body = TestClient(make_app(cfg.output.output_dir)).get("/api/quests/qmap1/rerun-steps").json()
    assert body["blocks"] == list(rerun_from.GROUPS) and body["config_path"].endswith("config.yaml")
    nodes = {n["node"]: n for n in body["nodes"]}
    assert len(nodes) == len(rerun_from.NODES)
    assert nodes["implement"]["clickable"] and not nodes["analyze"]["clickable"]
    assert nodes["design"]["needs_approval"] and not nodes["implement"]["needs_approval"]
    # The menu's own list is unchanged.
    assert [s["step"] for s in body["steps"]] == ["ideas", "literature", "plan", "design", "code"]


def test_the_button_uses_the_existing_resume_endpoint(tmp_path: Path) -> None:
    """A node name is accepted by ``POST /resume?from=`` as its step; one before the design is refused with its command."""
    cfg = _cfg(tmp_path)
    client = TestClient(make_app(cfg.output.output_dir))
    quest_root = cfg.output.output_dir / "qmap2"
    (quest_root / ".fi").mkdir(parents=True)
    (quest_root / ".fi" / "state.sqlite").write_bytes(b"")
    (quest_root / "config.yaml").write_text("topic: x", encoding="utf-8")
    res = client.post("/api/quests/qmap2/resume?from=design")
    assert res.status_code == 400 and "--approve-as" in res.text


def test_quest_page_has_the_map() -> None:
    html = (Path(__file__).resolve().parent.parent / "web" / "static" / "quest.html").read_text(encoding="utf-8")
    for needle in ("id=\"quest-map\"", "renderQuestMap", "mapRestart", "/resume?from=", "--approve-as",
                   "frozen protocol"):
        assert needle in html, needle
