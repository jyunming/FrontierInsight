"""The web pages a first-time person meets: Settings (provider states, the knowledge-base card) and the Launch of a
new quest (a no-cost check of the chosen model first). No model is called and nothing leaves this computer: the
status commands, the HTTP getter and Axon are replaced."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from core import provider_readiness as pr  # noqa: E402
from web.server import make_app  # noqa: E402

STATIC = Path(__file__).resolve().parent.parent / "web" / "static"


def _client(tmp_path: Path) -> TestClient:
    output_root = tmp_path / "outputs"
    output_root.mkdir()
    return TestClient(make_app(output_root))


@pytest.fixture
def quiet_readiness(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pr, "_run_command", lambda argv, timeout_s: (None, ""))

    async def nothing(url, headers, timeout_s):  # noqa: ANN001, ANN202
        return None, None

    monkeypatch.setattr(pr, "_get_json", nothing)


# ---------------------------------------------------------------- Settings: providers


def test_availability_gives_a_state_and_a_sentence_not_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, quiet_readiness: None,
) -> None:
    monkeypatch.setattr(pr.shutil, "which", lambda name: "/bin/claude" if name == "claude" else None)
    monkeypatch.setattr(pr, "_run_command", lambda argv, timeout_s: (0, '{"loggedIn": true}'))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    body = _client(tmp_path).get("/api/providers/availability").json()
    by = {p["name"]: p for p in body["providers"]}
    assert by["claude_cli"]["state"] == "signed_in" and by["claude_cli"]["available"] is True
    assert by["openai"]["state"] == "key_present"  # a key alone is never more than "key present"
    assert by["codex_cli"]["state"] == "not_installed" and by["codex_cli"]["available"] is False
    for p in body["providers"]:
        assert p["sentence"] and p["state_label"] and "READY" not in p["state_label"].upper()


def test_availability_for_one_provider(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, quiet_readiness: None) -> None:
    monkeypatch.setattr(pr.shutil, "which", lambda name: None)
    body = _client(tmp_path).get("/api/providers/availability?provider=ollama&model=llama3.2").json()
    assert [p["name"] for p in body["providers"]] == ["ollama"]
    assert body["providers"][0]["state"] == "not_installed"


def test_settings_page_shows_states_not_ready() -> None:
    html = (STATIC / "settings.html").read_text(encoding="utf-8")
    assert "'READY'" not in html and "SETUP REQUIRED" not in html
    assert "p.sentence" in html and "p.fix" in html


# ---------------------------------------------------------------- Settings: the knowledge base


class _FakeConfig:
    axon_store_base = "/store"
    bm25_path = "/store/bm25"
    api_key = "secret-value"


@pytest.fixture
def fake_axon(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[str]]:
    from core import knowledge as kmod

    events: dict[str, list[str]] = {"brain": [], "loop": []}

    class FakeBrain:
        def __init__(self, cfg) -> None:  # noqa: ANN001
            events["brain"].append("opened")
            try:
                asyncio.get_running_loop()
                events["loop"].append("on the event loop")
            except RuntimeError:
                events["loop"].append("in a worker thread")

        def switch_project(self, name: str) -> None:
            pass

        def list_documents(self):  # noqa: ANN202
            return [{"source": "fi_quest_paper", "chunks": 3}, {"source": "web-reingest", "chunks": 2}]

    monkeypatch.setattr(kmod, "_AXON_AVAILABLE", True)
    monkeypatch.setattr(kmod, "AxonBrain", FakeBrain, raising=False)
    monkeypatch.setattr(kmod, "AxonConfig", _FakeConfig, raising=False)
    return events


def test_settings_first_view_does_not_open_the_knowledge_base(
    tmp_path: Path, fake_axon: dict[str, list[str]],
) -> None:
    client = _client(tmp_path)
    started = time.monotonic()
    body = client.get("/api/knowledge/info").json()
    assert body["available"] is True and body["store_path"] == "/store"
    assert body["inventory"] == "on request"
    assert "total_chunks" not in body
    assert fake_axon["brain"] == []  # the store was not opened
    assert body["axon_config"]["api_key"].startswith("<redacted")
    assert time.monotonic() - started < 2.0


def test_inventory_runs_on_request_off_the_event_loop(tmp_path: Path, fake_axon: dict[str, list[str]]) -> None:
    body = _client(tmp_path).post("/api/knowledge/inventory").json()
    assert body["total_chunks"] == 5 and body["total_documents"] == 2
    assert fake_axon["brain"] == ["opened"]
    assert fake_axon["loop"] == ["in a worker thread"]


def test_inventory_has_a_time_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_axon: dict[str, list[str]],
) -> None:
    from core import knowledge as kmod
    import web.server as server

    class SlowBrain(kmod.AxonBrain):  # type: ignore[misc, name-defined]
        def list_documents(self):  # noqa: ANN202
            time.sleep(1.5)
            return []

    monkeypatch.setattr(kmod, "AxonBrain", SlowBrain)
    monkeypatch.setattr(server, "_KNOWLEDGE_INVENTORY_TIMEOUT_S", 0.2)
    body = _client(tmp_path).post("/api/knowledge/inventory").json()
    assert "did not answer within" in body["counts_error"]


def test_settings_page_counts_only_when_asked() -> None:
    html = (STATIC / "settings.html").read_text(encoding="utf-8")
    assert "Inspect knowledge base" in html
    first_view = html[html.index("async function loadAxonInfo"):html.index("async function inspectKnowledge")]
    assert "/api/knowledge/inventory" not in first_view
    assert "/api/knowledge/inventory" in html


# ---------------------------------------------------------------- Launch: the chosen model is checked first


def _answers() -> dict:
    return {
        "topic": "Web test topic", "title": "web-test", "output_kinds": ["paper_md"], "paper_format": "generic",
        "no_simulation": False, "study_depth": "journal-length", "comparative_baseline": "b", "success_metric": "m",
        "budget": "5 minutes", "clarify_mode": "auto", "review_panel": [], "knowledge_enabled": False,
        "provider": "openai", "provider_model": "gpt-5",
    }


def test_launch_stops_before_spending_when_the_model_is_not_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    launched: list[str] = []
    asked: list[tuple[str, str | None]] = []

    async def not_ready(provider, *, model=None, base_url="", api_key_env="", timeout_s=5.0):  # noqa: ANN001, ANN202
        asked.append((provider, model))
        return pr.Readiness(provider, "key_rejected", "The service refused the key in OPENAI_API_KEY.",
                            "Check OPENAI_API_KEY.")

    monkeypatch.setattr(pr, "preflight", not_ready)
    client = _client(tmp_path)
    monkeypatch.setattr(client.app.state.launcher, "launch", lambda **kw: launched.append("launched"))
    res = client.post("/api/interview/submit?launch=true", json=_answers())
    assert res.status_code == 409
    body = res.json()
    assert "refused the key" in body["detail"] and "To fix: Check OPENAI_API_KEY." in body["detail"]
    assert launched == []
    assert asked == [("openai", "gpt-5")]
    # The answers are kept: the draft YAML is on disk to run once fixed.
    assert Path(body["yaml_path"]).is_file()


def test_interview_page_shows_the_reason_and_keeps_the_answers() -> None:
    html = (STATIC / "interview.html").read_text(encoding="utf-8")
    assert "res.status === 409" in html
    assert "Your answers are kept" in html
    assert "/api/providers/availability?provider=" in html  # the picked provider's state under the picker


# ---------------------------------------------------------------- text drift


def test_dashboard_does_not_hard_code_a_question_count() -> None:
    for page in ("index.html", "interview.html"):
        text = (STATIC / page).read_text(encoding="utf-8")
        assert "14-question" not in text and "14 question" not in text


def test_dashboard_counts_the_questions_from_the_schema(tmp_path: Path) -> None:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert 'id="question-count"' in html and "/api/interview/schema" in html
    schema = _client(tmp_path).get("/api/interview/schema").json()
    n = sum(1 for q in schema["questions"] if q["tier"] == 1 and "serve" in (q.get("frontends") or ["serve"]))
    assert n > 0
