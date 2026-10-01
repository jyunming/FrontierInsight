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
    assert set(loops) == {"execute_reflect", "improve", "analyze", "review"}
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

    async def no_values(self) -> dict:
        return {}

    monkeypatch.setattr(Engine, "rerun_steps", fake_steps)
    monkeypatch.setattr(Engine, "_checkpoint_values", no_values)
    body = TestClient(make_app(cfg.output.output_dir)).get("/api/quests/qmap1/rerun-steps").json()
    assert [b["id"] for b in body["blocks"]] == list(rerun_from.MAP_BLOCKS) and body["config_path"].endswith("config.yaml")
    assert all(b["name"] and b["desc"] for b in body["blocks"]) and body["finished"] is False
    nodes = {n["node"]: n for n in body["nodes"]}
    assert len(nodes) == len(rerun_from.NODES)
    assert nodes["implement_outline"]["clickable"] and not nodes["analyze"]["clickable"]
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


def test_a_node_the_quest_has_not_reached_is_not_clickable_even_when_its_step_was() -> None:
    """``implement_outline`` and ``implement`` share the ``code`` step: with the quest stopped at the outline, the second is
    not reached, so it is neither clickable nor offered a restart it would describe as 'nothing to redo'."""
    nodes = {n["node"]: n for n in rerun_from.node_map(_steps({"ideas", "literature", "plan", "design", "skills", "code"}))}
    assert nodes["implement_outline"]["status"] == "now" and nodes["implement_outline"]["clickable"]
    assert nodes["implement"]["status"] == "todo" and not nodes["implement"]["clickable"]


def test_the_map_redoes_the_whole_step_from_its_first_node() -> None:
    """The JS starts the 'redone' list at the first node of the clicked node's step."""
    script = (Path(__file__).resolve().parent.parent / "web" / "static" / "quest_map.js").read_text(encoding="utf-8")
    assert "x.step === n.step" in script and "Yes, restart now" in script


def test_the_node_a_quest_stopped_in_is_now_even_when_it_is_not_a_step() -> None:
    """A review pause waits in ``human_feedback`` and a clarify pause in ``clarify``: neither is a step, so the last
    reached step alone put "now" on the review (or on nothing)."""
    everything = _steps(set(rerun_from.STEPS) - {"figures"})
    nodes = {n["node"]: n for n in rerun_from.node_map(everything, at=["human_feedback"])}
    assert nodes["human_feedback"]["status"] == "now"
    assert all(nodes[n]["status"] == "done" for n in ("clarify", "design", "execute", "write", "claim_check", "review"))
    nodes = {n["node"]: n for n in rerun_from.node_map(_steps(set()), at=["clarify"])}
    assert nodes["clarify"]["status"] == "now" and nodes["ideate"]["status"] == "todo"
    # A node on the other path is never where a quest stopped: the last reached step is used instead.
    nodes = {n["node"]: n for n in rerun_from.node_map(_steps({"ideas", "literature"}), at=["wait_for_data"])}
    assert nodes["literature"]["status"] == "now" and nodes["wait_for_data"]["status"] == "off"
    # On the data path the same pause is where the quest stopped.
    nodes = {n["node"]: n for n in rerun_from.node_map(_steps({"ideas", "literature", "plan", "design", "skills", "run"}),
                                                       no_simulation=True, at=["wait_for_data"])}
    assert nodes["wait_for_data"]["status"] == "now" and nodes["auto_collect_data"]["status"] == "done"
    assert nodes["data_load"]["status"] == "todo"


def _quest_on_disk(tmp_path: Path, name: str) -> tuple[Engine, Path]:
    cfg = _cfg(tmp_path)
    quest_root = cfg.output.output_dir / name
    (quest_root / ".fi").mkdir(parents=True)
    return Engine(cfg, resume_quest_id=name), quest_root


async def test_a_failed_quest_is_not_finished_and_shows_where_it_failed(tmp_path: Path,
                                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    engine, quest_root = _quest_on_disk(tmp_path, "qfail")

    async def at_execute(self):
        return ("execute",)

    monkeypatch.setattr(Engine, "_checkpoint_next", at_execute)
    (quest_root / "quest_failed.md").write_text("the run crashed", encoding="utf-8")
    (quest_root / "frontier_insight_summary.json").write_text("{}", encoding="utf-8")
    payload = await engine.quest_map(_steps({"ideas", "literature", "plan", "design", "skills", "code", "run"}))
    nodes = {n["node"]: n for n in payload["nodes"]}
    assert payload["finished"] is False and nodes["execute"]["status"] == "now"
    assert nodes["implement"]["status"] == "done" and nodes["analyze"]["status"] == "todo"


@pytest.mark.parametrize("marker", [".fi/pause.json", "NEXT_STEP.md", "quest_failed.md", None])
async def test_finished_means_the_graph_ended_and_nothing_is_waiting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                    marker: str | None) -> None:
    engine, quest_root = _quest_on_disk(tmp_path, "qend")

    async def at_end(self):
        return ()

    monkeypatch.setattr(Engine, "_checkpoint_next", at_end)
    if marker:
        (quest_root / marker).write_text("{}", encoding="utf-8")
    assert await engine.stopped_at() == (marker is None, [])


async def test_a_quest_with_no_checkpoint_is_not_finished(tmp_path: Path) -> None:
    """An empty or unreadable state.sqlite has nothing next either: that is "never ran", not "finished"."""
    engine, quest_root = _quest_on_disk(tmp_path, "qnone")
    (quest_root / "frontier_insight_summary.json").write_text("{}", encoding="utf-8")
    assert await engine.stopped_at() == (False, [])
    (quest_root / ".fi" / "state.sqlite").write_bytes(b"")
    assert await engine.stopped_at() == (False, [])


@pytest.mark.slow
async def test_a_quest_paused_at_the_review_is_not_finished_until_it_is_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """The bug: a quest paused for the person's review decision showed every node finished, because a pause writes
    ``frontier_insight_summary.json`` and the map took that file as "finished". A real run (fake model, real venv and
    checkpoint) is paused at the review, then accepted; the CLI (what the VS Code panel reads) and the web map agree."""
    import asyncio
    import json

    import launch
    from core.provider import ProxySupervisor
    from tests.test_engine_smoke import _fake_response_for

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        return _fake_response_for(messages[-1]["content"])

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)
    cfg = _cfg(tmp_path)
    cfg.pauses.review = "ask"
    cfg.pauses.auto_accept_on_pass = False
    first = Engine(cfg)
    await asyncio.wait_for(first.run(), timeout=300)
    quest_id, quest_root = first.quest_id, first.quest_root
    assert (quest_root / ".fi" / "pause.json").is_file()
    # launch.py writes the summary after the run returns, a pause included ("[FI] summary -> ...").
    (quest_root / "frontier_insight_summary.json").write_text("{}", encoding="utf-8")
    (quest_root / "config.yaml").write_text(yaml.safe_dump(cfg.model_dump(mode="json")), encoding="utf-8")

    async def cli_map() -> dict:
        capsys.readouterr()
        assert await launch._list_rerun_steps(cfg, quest_id, supervisor=ProxySupervisor(), as_json=True) == 0
        return json.loads(capsys.readouterr().out.strip().splitlines()[-1])

    paused = await cli_map()
    nodes = {n["node"]: n["status"] for n in paused["nodes"]}
    assert paused["finished"] is False
    assert nodes["human_feedback"] == "now"
    assert all(nodes[n] == "done" for n in ("clarify", "ideate", "design", "execute", "write", "claim_check", "review"))
    web = TestClient(make_app(cfg.output.output_dir)).get(f"/api/quests/{quest_id}/rerun-steps").json()
    assert web["finished"] is False and web["nodes"] == paused["nodes"]

    (quest_root / ".fi" / "human_review_answer.json").write_text(json.dumps({"action": "accept", "answer": "yes"}), encoding="utf-8")
    await asyncio.wait_for(Engine(cfg, resume_quest_id=quest_id).run(), timeout=300)
    done = await cli_map()
    assert done["finished"] is True and {n["status"] for n in done["nodes"]} == {"done", "off"}

    # Run again from the writing, and the writing fails: the outputs were moved aside, so the quest is not finished,
    # and it stopped in the writing, not at the end of the line it left.
    async def broken_write(self, state):  # noqa: ANN001
        raise RuntimeError("the writer broke")

    monkeypatch.setattr(Engine, "_node_write", broken_write)
    again = Engine(cfg, resume_quest_id=quest_id)
    with pytest.raises(Exception):
        await asyncio.wait_for(again.run(from_step="writing"), timeout=300)
    assert list((quest_root / ".fi" / "previous").iterdir())
    assert await again.stopped_at() == (False, ["write"])
    # A run stopped (killed) in that node leaves no failure note: still not finished, still in the writing.
    (quest_root / "quest_failed.md").unlink(missing_ok=True)
    assert await again.stopped_at() == (False, ["write"])
    failed = await cli_map()
    assert failed["finished"] is False and {n["node"]: n["status"] for n in failed["nodes"]}["write"] == "now"


@pytest.mark.parametrize("flag", ["no_simulation", "survey_mode", "analyze_local_first"])
async def test_the_config_can_put_a_quest_on_the_data_path(tmp_path: Path, flag: str) -> None:
    cfg = _cfg(tmp_path)
    setattr(cfg.engine, flag, True)
    assert await Engine(cfg).rerun_no_simulation() is True


@pytest.mark.parametrize("key,expected", [("no_simulation_resolved", True), ("survey_mode_resolved", True), ("other", False)])
async def test_clarify_can_put_a_quest_on_the_data_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: str, expected: bool) -> None:
    async def values(self) -> dict:
        return {key: True}

    monkeypatch.setattr(Engine, "_checkpoint_values", values)
    assert await Engine(_cfg(tmp_path)).rerun_no_simulation() is expected


def test_the_rerun_steps_api_uses_the_data_path_clarify_chose(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg(tmp_path)
    quest_root = cfg.output.output_dir / "qmap3"
    (quest_root / ".fi").mkdir(parents=True)
    (quest_root / ".fi" / "state.sqlite").write_bytes(b"")
    (quest_root / "config.yaml").write_text(yaml.safe_dump(cfg.model_dump(mode="json")), encoding="utf-8")

    async def fake_steps(self) -> list[dict]:
        return _steps({"ideas", "literature", "plan", "design", "skills", "run"})

    async def values(self) -> dict:
        return {"no_simulation_resolved": True}

    monkeypatch.setattr(Engine, "rerun_steps", fake_steps)
    monkeypatch.setattr(Engine, "_checkpoint_values", values)
    body = TestClient(make_app(cfg.output.output_dir)).get("/api/quests/qmap3/rerun-steps").json()
    blocks = {b["id"]: b for b in body["blocks"]}
    assert blocks["run"]["off"] and not blocks["data"]["off"]
