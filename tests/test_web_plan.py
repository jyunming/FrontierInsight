"""The quest page's plan: read plan.md, save an edit of it, ask for a change."""

from __future__ import annotations

from pathlib import Path

import pytest

try:
    from fastapi.testclient import TestClient
except Exception:  # pragma: no cover
    pytest.skip("fastapi/httpx not installed", allow_module_level=True)

from core import plan
from web.server import make_app

DESIGN = {
    "hypothesis": "model-based OPC reduces EPE more than rule-based",
    "variables": {"independent": ["strategy"], "dependent": ["epe"], "controls": ["seed"]},
    "method": "compare strategies on synthetic clips",
    "expected_outcome": "model-based wins",
    "figures_planned": ["c.png"],
    "dependencies": ["numpy"],
}
EXTRA = {"in_short": "A comparison.", "gap": "Nothing compares them.", "literature": [], "risks": ["easy clips"]}


class _Proc:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.code: int | None = None

    def poll(self) -> int | None:
        return self.code


@pytest.fixture
def procs(monkeypatch: pytest.MonkeyPatch):
    launched: list[tuple[list[str], _Proc]] = []

    def fake_popen(argv, **_kwargs):
        proc = _Proc(9300 + len(launched))
        launched.append((argv, proc))
        return proc

    monkeypatch.setattr("web.quest_launcher.subprocess.Popen", fake_popen)
    return launched


def _client(tmp_path: Path) -> TestClient:
    root = tmp_path / "outputs"
    root.mkdir()
    return TestClient(make_app(root))


def _quest(client: TestClient, quest_id: str, *, with_plan: bool = True) -> Path:
    quest = client.app.state.output_root / quest_id  # type: ignore[attr-defined]
    (quest / ".fi").mkdir(parents=True)
    (quest / "config.yaml").write_text("topic: x", encoding="utf-8")
    if with_plan:
        text = plan.render("OPC", EXTRA, DESIGN, ["an independent evaluator"])
        plan.plan_path(quest).write_text(text, encoding="utf-8")
        plan.record_version(quest, text, by="model")
    return quest


def test_a_quest_without_a_plan_says_so(tmp_path: Path) -> None:
    client = _client(tmp_path)
    _quest(client, "p0", with_plan=False)
    body = client.get("/api/quests/p0/plan").json()
    assert body["exists"] is False and body["text"] == ""
    assert client.put("/api/quests/p0/plan", json={"text": "x"}).status_code == 404
    assert client.post("/api/quests/p0/plan/revise", json={"request": "x"}).status_code == 404


def test_the_plan_is_served_with_its_versions(tmp_path: Path) -> None:
    client = _client(tmp_path)
    _quest(client, "p1")
    body = client.get("/api/quests/p1/plan").json()
    assert body["exists"] is True
    assert "## The design (used as written)" in body["text"]
    assert body["design_error"] == ""
    assert [v["by"] for v in body["versions"]] == ["model"]


def test_an_edit_is_saved_and_kept_as_a_version_by_you(tmp_path: Path) -> None:
    client = _client(tmp_path)
    quest = _quest(client, "p2")
    text = client.get("/api/quests/p2/plan").json()["text"].replace("compare strategies", "compare four strategies")

    res = client.put("/api/quests/p2/plan", json={"text": text})

    assert res.status_code == 200, res.text
    assert plan.load_design(quest)[0]["method"] == "compare four strategies on synthetic clips"
    assert [v["by"] for v in res.json()["versions"]] == ["model", "user"]


def test_an_edit_whose_design_cannot_be_read_is_refused_and_changes_nothing(tmp_path: Path) -> None:
    client = _client(tmp_path)
    quest = _quest(client, "p3")
    before = plan.plan_path(quest).read_text(encoding="utf-8")
    broken = before.replace("hypothesis: model-based", "hypothesis: [model-based")

    res = client.put("/api/quests/p3/plan", json={"text": broken})

    assert res.status_code == 400 and "not saved" in res.text and "YAML" in res.text
    assert plan.plan_path(quest).read_text(encoding="utf-8") == before
    assert client.put("/api/quests/p3/plan", json={"text": "  "}).status_code == 400
    assert client.put("/api/quests/p3/plan", json={"nope": 1}).status_code == 400


def test_a_plan_broken_on_disk_shows_why(tmp_path: Path) -> None:
    client = _client(tmp_path)
    quest = _quest(client, "p4")
    plan.plan_path(quest).write_text("## The design (used as written)\n\n```yaml\nmethod: x\n```\n", encoding="utf-8")
    assert "hypothesis" in client.get("/api/quests/p4/plan").json()["design_error"]


def test_a_request_spawns_revise_plan_with_the_words_as_one_argument(tmp_path: Path, procs) -> None:
    client = _client(tmp_path)
    _quest(client, "p5")
    ask = 'use CD error, not EPE; and "quote" this'

    res = client.post("/api/quests/p5/plan/revise", json={"request": ask})

    assert res.status_code == 200 and res.json()["revising"] is True
    argv = procs[0][0]
    assert "--revise-plan" in argv and argv[argv.index("--revise-plan") + 1] == ask
    assert argv[argv.index("--resume") + 1] == "p5"
    assert "--config" in argv


def test_a_request_needs_words_and_only_one_runs_at_a_time(tmp_path: Path, procs) -> None:
    client = _client(tmp_path)
    _quest(client, "p6")
    assert client.post("/api/quests/p6/plan/revise", json={"request": "  "}).status_code == 400
    assert client.post("/api/quests/p6/plan/revise", json={"request": "x" * 5000}).status_code == 413
    assert client.post("/api/quests/p6/plan/revise", json={"request": "one"}).status_code == 200
    assert client.post("/api/quests/p6/plan/revise", json={"request": "two"}).status_code == 409
    procs[0][1].code = 0
    assert client.post("/api/quests/p6/plan/revise", json={"request": "three"}).status_code == 200


def test_the_rewrite_status_says_running_then_how_it_ended(tmp_path: Path, procs) -> None:
    client = _client(tmp_path)
    _quest(client, "p7")
    assert client.get("/api/quests/p7/plan/revise").json()["state"] == "not_started"
    client.post("/api/quests/p7/plan/revise", json={"request": "change it"})
    log = client.app.state.output_root / "_jobs" / "p7-plan" / "launch.log"  # type: ignore[attr-defined]
    log.write_text("[FI] cannot revise the plan: the revised plan could not be used (no fence)\n", encoding="utf-8")
    assert client.get("/api/quests/p7/plan/revise").json()["state"] == "running"
    procs[0][1].code = 1
    body = client.get("/api/quests/p7/plan/revise").json()
    assert body["state"] == "exited" and body["returncode"] == 1
    assert "could not be used" in body["last_line"]


def test_the_plan_routes_refuse_a_hostile_quest_id(tmp_path: Path) -> None:
    client = _client(tmp_path)
    for hostile in ("..%2F..%2Fetc", "a b", "x;rm"):
        assert client.get(f"/api/quests/{hostile}/plan").status_code in (400, 404, 422)
        assert client.put(f"/api/quests/{hostile}/plan", json={"text": "x"}).status_code in (400, 404, 422)


def test_the_quest_page_carries_the_plan_panel() -> None:
    html = (Path(__file__).resolve().parent.parent / "web" / "static" / "quest.html").read_text(encoding="utf-8")
    for needle in ('id="plan-section"', 'id="plan-editor"', 'id="plan-request"', "planRevise()", "planSave()",
                   "/plan/revise", 'id="next-step-plan"'):
        assert needle in html, needle
