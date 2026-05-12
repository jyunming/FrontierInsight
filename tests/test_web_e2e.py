"""Phase J end-to-end smoke: full DAG through the FastAPI server.

This is the validator that proves the GUI works:
  1. Start a quest via POST /api/quests/start with a fake-LLM config.
  2. Poll until the clarify panel surfaces (engine paused at clarify node).
  3. Submit answers via POST /api/quests/{id}/clarify.
  4. Wait for the quest task to finish.
  5. Assert paper.md + figures landed on disk AND surface via the
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
from tests.test_engine_smoke import _fake_response_for


_QUEST_YAML = textwrap.dedent("""\
    topic: |
      End-to-end GUI smoke test: confirm a fake-LLM quest runs from
      `/api/quests/start` through `/api/quests/{id}/clarify` to a
      finished paper.md on disk.

    title: gui-smoke

    provider:
      name: openai

    engine:
      framework: langgraph
      max_iterations: 1
      review_loop: false
      clarify_mode: interactive

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
async def test_full_dag_via_server_with_clarify_pause_and_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full GUI flow: start → clarify pause → submit → run completes."""
    # Monkeypatch the LLM so the engine consumes canned responses.
    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        return _fake_response_for(messages[-1]["content"])
    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)

    app = make_app(tmp_path)
    transport = ASGITransport(app=app)
    base_url = "http://test"

    async with AsyncClient(transport=transport, base_url=base_url) as client:
        # 1. Start a quest.
        r = await client.post("/api/quests/start", json={"yaml": _QUEST_YAML})
        assert r.status_code == 200, r.text
        quest_id = r.json()["quest_id"]
        quest_root = Path(r.json()["quest_root"])
        assert quest_root.is_relative_to(tmp_path)

        # 2. Poll until clarify panel surfaces (interrupt() fired).
        # `Engine.run` does an upfront `executor.setup(quest_root)` which
        # on a cold checkout creates a real venv — on Windows that's
        # ~30 s — before the clarify node even runs. Budget generously.
        for _ in range(900):
            await asyncio.sleep(0.2)
            r = await client.get(f"/api/quests/{quest_id}/clarify")
            if r.json().get("pending"):
                break
        else:
            log = (quest_root / ".fi" / "run.log").read_text(
                encoding="utf-8", errors="replace",
            )
            pytest.fail(
                "clarify never surfaced within 180 s.\n--- run.log tail ---\n"
                + "\n".join(log.splitlines()[-30:])
            )
        questions = r.json()["questions"]
        assert "comparative_baseline" in questions
        assert "output_kinds" in questions

        # 3. Submit answers via the clarify endpoint.
        answers = {
            "comparative_baseline": "user-supplied baseline X",
            "empirical_vs_theoretical": "empirical",
            "success_metric": "user-supplied metric M",
            "budget": "1 minute",
            "output_kinds": ["paper_md"],
        }
        r = await client.post(
            f"/api/quests/{quest_id}/clarify",
            json={"answers": answers},
        )
        assert r.status_code == 200
        assert r.json() == {"ok": True}

        # 4. Wait for the background task to finish. We watch the
        #    paper.md to detect end-of-quest (the summary file is written
        #    by `launch.py` outside this code path — the server-side
        #    driver only runs `engine.run`, not the generators stage).
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

        # 5. Assert the GUI surfaces the finished quest correctly.
        # Wait for the driver task to actually finish (paper.md exists
        # but `engine.run` might still be in its `finally:` block).
        for _ in range(50):
            await asyncio.sleep(0.1)
            r = await client.get(f"/api/quests/{quest_id}")
            body = r.json()
            if not body.get("alive"):
                break

        assert r.status_code == 200
        assert body["paper_preview"] is not None
        assert "result.png" in (body.get("figures") or [])
        # The summary file is written by launch.py's `run_one`, not by
        # the server's quest driver — so it's None here. The paper
        # preview being non-empty is the sufficient "done" signal.
        assert body.get("pending_clarify") is False
