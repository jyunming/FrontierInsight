"""End-to-end smoke: full DAG through the FastAPI server.

This is the validator that proves the GUI works:
  1. Start a quest via POST /api/quests/start with a fake-LLM config that
     names a commission (every study starts from one; it asks no setup
     questions, so the run goes through without a clarify pause).
  2. Wait for the quest task to finish.
  3. Assert paper.md + figures landed on disk AND surface via the
     /api/quests/{id} detail endpoint.

The LLMClient.chat is monkeypatched module-wide so no real API
calls happen. The server's TestClient runs the FastAPI app on the
same event loop as our test, so `asyncio.create_task` on the server
side and `await asyncio.sleep` on the client side cooperate.
"""

from __future__ import annotations

import asyncio
import json
import textwrap
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from web.server import make_app


# Re-use the canned LLM responses from the engine smoke test.
from tests import commissioned
from tests.test_engine_smoke import _fake_response_for

# A whole quest with a real venv runs behind each test; give it longer than pytest.ini's per-test default.
pytestmark = pytest.mark.timeout(1800)


_QUEST_YAML = textwrap.dedent("""\
    title: gui-smoke

    provider:
      name: openai

    engine:
      framework: langgraph
      max_iterations: 1
      review_loop: false
      # Default gate is "after_review"; auto-accept finalises the
      # clean-verdict path without wiring a callback in tests.
      auto_accept_on_pass: true

    execution:
      sandbox: venv
      timeout_s: 120

    knowledge:
      enabled: false

    output:
      kinds: [paper_md]
      output_dir: ./will-be-overridden
""")


@pytest.mark.asyncio
async def test_full_dag_via_server_from_a_commission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full GUI flow: start from a commission → run completes, with no setup questions."""
    # Monkeypatch the LLM so the engine consumes canned responses.
    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        return _fake_response_for(messages[-1]["content"])
    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)

    app = make_app(tmp_path)
    transport = ASGITransport(app=app)
    base_url = "http://test"

    async with AsyncClient(transport=transport, base_url=base_url) as client:
        # 1. Start a quest.
        yaml_text = commissioned.line(tmp_path / "order") + _QUEST_YAML
        r = await client.post("/api/quests/start", json={"yaml": yaml_text})
        assert r.status_code == 200, r.text
        quest_id = r.json()["quest_id"]
        quest_root = Path(r.json()["quest_root"])
        assert quest_root.is_relative_to(tmp_path)

        # 2. Wait for the engine run to finish. paper/paper.md marks its
        #    end; the server-side driver then runs the CLI's output pass
        #    and writes the summary file.
        paper_md = quest_root / "paper" / "paper.md"
        # Budget: ~300 s. First-run venv create + matplotlib install +
        # warmup is the bulk of this; ~110 s in practice on Windows.
        for _ in range(3000):
            await asyncio.sleep(0.1)
            if paper_md.exists():
                # paper.md is the very last file the engine writes, so
                # it's a safe completion signal.
                break
        else:
            log = (quest_root / ".fi" / "run.log").read_text(
                encoding="utf-8", errors="replace",
            )
            pytest.fail(
                f"quest {quest_id} did not finish in 300 s.\n--- run.log tail ---\n"
                + "\n".join(log.splitlines()[-30:])
            )

        # 3. Assert the GUI surfaces the finished quest correctly.
        # Wait for the driver task to actually finish: paper.md exists
        # before `engine.run` leaves its `finally:` block and before the
        # output pass runs.
        await asyncio.wait_for(app.state.registry._tasks[quest_id], timeout=120)
        r = await client.get(f"/api/quests/{quest_id}")
        body = r.json()

        assert r.status_code == 200
        assert body["alive"] is False
        assert body["paper_preview"] is not None
        assert "result.png" in (body.get("figures") or [])
        # The output pass ran, as it does after a CLI run of the same
        # config: paper.md is copied to the quest root and the summary
        # lists it.
        assert (quest_root / "paper.md").is_file()
        assert body["summary"]["outputs"]["paper_md"] == str(quest_root / "paper.md")
        assert body["errors"] is None
        assert body.get("pending_clarify") is False

        # 4. Rename the finished quest from the quest page: both copies of the paper, the config, the summary and
        # the saved state a later resume reads all carry the new title.
        r = await client.post(f"/api/quests/{quest_id}/title", json={"title": "A Better Title"})
        assert r.status_code == 200, r.text
        for paper in (quest_root / "paper" / "paper.md", quest_root / "paper.md"):
            assert "# A Better Title\n" in paper.read_text(encoding="utf-8")
        assert json.loads((quest_root / "frontier_insight_summary.json").read_text(encoding="utf-8"))["title"] == "A Better Title"
        assert r.json()["saved_state"] is True
        assert (await client.get(f"/api/quests/{quest_id}")).json()["title"] == "A Better Title"
