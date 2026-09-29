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
    for name, meta in rerun_from.NODES.items():
        assert meta["block"] in rerun_from.MAP_BLOCKS, name
        for field in ("title", "sentence", "reads", "writes"):
            assert meta[field].strip(), (name, field)
    # A plain title is not the node's own name.
    assert all(meta["title"] != name for name, meta in rerun_from.NODES.items())
    for name, (block_name, desc, _off) in rerun_from.MAP_BLOCKS.items():
        assert block_name.strip() and desc.strip(), name


def test_every_block_holds_nodes_in_graph_order() -> None:
    blocks = [meta["block"] for meta in rerun_from.NODES.values()]
    order = list(rerun_from.MAP_BLOCKS)
    assert [b for i, b in enumerate(blocks) if i == 0 or b != blocks[i - 1]] == order


def test_the_loops_the_graph_has_are_noted() -> None:
    loops = {n: m["loop"] for n, m in rerun_from.NODES.items() if m.get("loop")}
    assert set(loops) == {"execute_reflect", "analyze", "review"}
    assert all("Run it" in loops["execute_reflect"] or "Design" in v for v in loops.values())


def test_map_blocks_mark_the_path_the_quest_does_not_take() -> None:
    sim = {b["id"]: b for b in rerun_from.map_blocks()}
    assert sim["data"]["off"] and not sim["run"]["off"]
    nosim = {b["id"]: b for b in rerun_from.map_blocks(no_simulation=True)}
    assert nosim["run"]["off"] and not nosim["data"]["off"]
    assert [b["id"] for b in rerun_from.map_blocks()] == list(rerun_from.MAP_BLOCKS)


def test_status_says_finished_stopped_here_and_not_reached() -> None:
    nodes = {n["node"]: n for n in rerun_from.node_map(
        _steps({"ideas", "literature", "plan", "design", "skills", "code"}))}
    for name in ("clarify", "ideate", "literature", "select_skills", "plan", "design"):
        assert nodes[name]["status"] == "done", name
    assert nodes["implement_outline"]["status"] == "now"
    for name in ("implement", "execute", "analyze", "write", "review"):
        assert nodes[name]["status"] == "todo", name
    # The other path is off whatever the checkpoint says.
    assert all(nodes[n]["status"] == "off" for n in ("auto_collect_data", "wait_for_data", "data_load", "web_figures"))


def test_a_finished_quest_has_nothing_that_is_still_open() -> None:
    nodes = rerun_from.node_map(_steps(set(rerun_from.STEPS) - {"figures"}), finished=True)
    assert {n["status"] for n in nodes} == {"done", "off"}


def test_a_quest_with_no_simulation_takes_the_data_path() -> None:
    nodes = {n["node"]: n for n in rerun_from.node_map(
        _steps({"ideas", "literature", "plan", "design", "skills", "run", "figures"}), no_simulation=True)}
    assert nodes["execute"]["status"] == "off" and not nodes["execute"]["clickable"]
    assert nodes["web_plots"]["status"] == "now" and nodes["web_plots"]["clickable"]
    assert nodes["data_load"]["status"] == "done" and nodes["data_load"]["clickable"]


def test_a_quest_that_reached_nothing_is_all_not_reached() -> None:
    nodes = rerun_from.node_map(_steps(set()))
    assert {n["status"] for n in nodes} == {"todo", "off"} and not any(n["clickable"] for n in nodes)


def test_a_node_on_the_other_path_is_never_clickable() -> None:
    nodes = {n["node"]: n for n in rerun_from.node_map(_steps(set(rerun_from.STEPS)))}
    assert not nodes["auto_collect_data"]["clickable"] and not nodes["data_load"]["clickable"]
    assert nodes["execute"]["clickable"]


def test_the_skills_node_says_where_to_tune_it() -> None:
    nodes = {n["node"]: n for n in rerun_from.node_map(_steps(set(rerun_from.STEPS)))}
    assert "engine.skills_exclude" in nodes["select_skills"]["hint"]
    assert not nodes["design"]["hint"]


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
    # web_figures is a real step, but on the other path a simulation quest does not take.
    assert everything["web_figures"]["step"] == "figures" and not everything["web_figures"]["clickable"]


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
    assert [b["id"] for b in body["blocks"]] == list(rerun_from.MAP_BLOCKS) and body["config_path"].endswith("config.yaml")
    assert all(b["name"] and b["desc"] for b in body["blocks"]) and body["finished"] is False
    nodes = {n["node"]: n for n in body["nodes"]}
    assert len(nodes) == len(rerun_from.NODES)
    assert nodes["implement"]["clickable"] and not nodes["analyze"]["clickable"]
    assert nodes["design"]["needs_approval"] and not nodes["implement"]["needs_approval"]
    assert nodes["implement"]["title"] == "Write the code" and nodes["implement_outline"]["status"] == "now"
    assert nodes["design"]["status"] == "done" and nodes["analyze"]["status"] == "todo"
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
    static = Path(__file__).resolve().parent.parent / "web" / "static"
    html = (static / "quest.html").read_text(encoding="utf-8")
    for needle in ("id=\"quest-map-root\"", "renderQuestMap", "FIQuestMap", "/static/quest_map.js",
                   "/static/quest_map.css", "/resume?from=", "--approve-as"):
        assert needle in html, needle
    script = (static / "quest_map.js").read_text(encoding="utf-8")
    # Text from the API is set as text, never as HTML.
    assert "innerHTML" not in script and "textContent" in script
    css = (static / "quest_map.css").read_text(encoding="utf-8")
    for token in ("--accent", "--redo", "prefers-color-scheme: dark", 'data-theme="dark"', "IBM Plex"):
        assert token in css, token


def test_map_payload_is_what_the_web_and_vs_code_draw() -> None:
    steps = _steps({"ideas", "literature"})
    payload = rerun_from.map_payload(steps, finished=False, no_simulation=True)
    assert set(payload) == {"blocks", "finished", "nodes"}
    assert payload["nodes"] == rerun_from.node_map(steps, no_simulation=True)
    assert payload["blocks"] == rerun_from.map_blocks(no_simulation=True)


def test_the_vs_code_panel_loads_the_shared_map_and_restarts_through_chat() -> None:
    ext = Path(__file__).resolve().parent.parent / "vscode-frontier-insight"
    ts = (ext / "src" / "quest-map.ts").read_text(encoding="utf-8")
    assert "quest_map.js" in ts and "--json" in ts and "workbench.action.chat.open" in ts
    pkg = (ext / "package.json").read_text(encoding="utf-8")
    assert "frontierInsight.questMap" in pkg and "FI: Quest map" in pkg and "copy-quest-map.js" in pkg
