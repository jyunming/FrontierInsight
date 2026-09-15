"""Pin the structure of the tier-1 → review tier-2/3 flow in
``web/static/interview.html`` so a future edit can't quietly drop
the review section, the advanced toggle, or the per-row [edit]
buttons.

These are content-level assertions on the static HTML + JS. The
flow is exercised end-to-end in the browser; here we just guarantee
the surface area exists.
"""
from __future__ import annotations

from pathlib import Path

import pytest


STATIC = Path(__file__).resolve().parent.parent / "web" / "static"
INTERVIEW_HTML = STATIC / "interview.html"


@pytest.fixture(scope="module")
def html() -> str:
    return INTERVIEW_HTML.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "needle",
    [
        # JS hooks
        "renderReview",
        "deriveTier2",
        "deriveTier3",
        "editReviewField",
        "saveReviewField",
        "cancelReviewEdit",
        "toggleAdvanced",
        "backToForm",
        "launchFromReview",
        # DOM hooks the JS keys off
        'id="review-section"',
        'id="interview-form"',
    ],
)
def test_tier_flow_function_or_hook_present(html: str, needle: str) -> None:
    """Every JS function name + DOM id the tier flow uses must
    appear in the static file. Catches a half-revert that removes
    the review section while leaving the rest intact."""
    assert needle in html, f"interview.html lost the {needle!r} hook"


def test_tier1_filter_excludes_non_tier1_questions(html: str) -> None:
    """The new flow only renders tier-1 questions in the initial
    form. The filter must reference q.tier === 1; if a future edit
    drops the predicate, the form regresses to showing all 14
    questions on one page."""
    assert "q.tier === 1" in html, (
        "interview.html must filter questions by tier===1 to keep the "
        "review-step flow"
    )


def test_continue_to_review_button_replaces_launch_button(html: str) -> None:
    """The form submit button is "Continue to Review", not "Launch
    Quest" — the launch happens at the bottom of the review page
    after the user confirms the derived defaults."""
    assert "Continue to Review" in html
    # Launch button still exists, but inside renderReview() (NOT as
    # the form-submit button). Easiest way to assert: the button
    # type for the in-review launcher is type="button", not
    # type="submit".
    assert "onclick=\"launchFromReview()\"" in html


def test_review_renders_three_cards_tier1_tier2_advanced(html: str) -> None:
    """The review section shows three cards: tier-1 summary,
    tier-2 derived defaults, and a collapsible Advanced block."""
    assert "Your answers" in html, "tier-1 summary card title missing"
    assert "Defaults (auto-derived)" in html, "tier-2 card title missing"
    # Advanced card has a button toggling visibility; its label is
    # the literal string "Advanced".
    assert ">Advanced<" in html or "'Advanced'" in html or '"Advanced"' in html, (
        "advanced toggle label missing"
    )


def test_per_row_edit_buttons_present(html: str) -> None:
    """Each derived-value row carries an inline Edit button that
    calls editReviewField(id). The button's label is "Edit" (plus
    an icon)."""
    assert "editReviewField" in html
    # Two Edit buttons: one for the tier-1 summary card ("Edit"
    # back to form), one per derived row.
    # We just need the JS hook + the visible label.
    assert ">Edit<" in html or 'aria-label="Edit"' in html


def test_axon_status_probed_for_knowledge_enabled_default(html: str) -> None:
    """The knowledge_enabled tier-2 default is auto-probed from
    /api/axon/status — the JS must hit that endpoint, otherwise
    every quest defaults to knowledge_enabled=false regardless of
    the sidecar state."""
    assert "/api/axon/status" in html


def test_prose_formats_set_drives_no_simulation_default(html: str) -> None:
    """no_simulation defaults to True for prose paper formats
    (essay / report / policy_brief / whitepaper). The JS keeps a
    small PROSE_FORMATS set mirroring core/interview.py."""
    assert "PROSE_FORMATS" in html
    for fmt in ("essay", "report", "policy_brief", "whitepaper"):
        assert f"'{fmt}'" in html, f"prose format {fmt!r} missing from PROSE_FORMATS set"


# ---------------------------------------------------------------------------
# End-to-end POST: the new tier-2/3 fields must round-trip into the YAML.
# ---------------------------------------------------------------------------


def test_submit_round_trips_audience_and_top_k(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The /api/interview/submit endpoint must accept audience +
    knowledge_top_k in the payload and persist them into the
    generated YAML. The prior code paths silently dropped both
    fields because _parse_answers wasn't extracting them."""
    from fastapi.testclient import TestClient
    from web.server import make_app

    app = make_app(tmp_path)
    client = TestClient(app)
    payload = {
        "topic": "Round-trip test",
        "title": "rt",
        "output_kinds": ["paper_md", "paper_pdf"],
        "paper_format": "report",
        "study_depth": "journal-length",
        "provider": "ollama", "provider_model": "qwen2.5:32b",
        "no_simulation": True, "clarify_mode": "auto",
        "review_panel": [], "knowledge_enabled": False,
        "comparative_baseline": "", "success_metric": "", "budget": "",
        # The two new tier-2/3 fields the audit caught.
        "audience": "internal",
        "knowledge_top_k": 12,
    }
    r = client.post("/api/interview/submit", json=payload)
    assert r.status_code == 200, r.text
    from pathlib import Path
    yaml_path = Path(r.json()["yaml_path"])
    text = yaml_path.read_text(encoding="utf-8")
    assert 'audience: "internal"' in text, (
        "audience didn't land in the generated YAML — _parse_answers "
        "may have stopped extracting it"
    )
    assert "top_k: 12" in text, (
        "knowledge.top_k didn't land in the generated YAML"
    )


def test_submit_omits_audience_at_default_for_compact_yaml(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """When audience == 'external' (the default), the emitter omits
    the line entirely. Keeps the generated YAML small for the common
    case. Same for ``top_k == 8`` (the new default after the split into
    Axon vs external caps)."""
    from fastapi.testclient import TestClient
    from web.server import make_app

    app = make_app(tmp_path)
    client = TestClient(app)
    payload = {
        "topic": "Default-audience test", "title": "def",
        "output_kinds": ["paper_md"],
        "paper_format": "generic",
        "study_depth": "journal-length",
        "provider": "ollama", "provider_model": "qwen2.5:32b",
        "no_simulation": False, "clarify_mode": "auto",
        "review_panel": [], "knowledge_enabled": False,
        "comparative_baseline": "", "success_metric": "", "budget": "",
        "audience": "external", "knowledge_top_k": 8,
    }
    r = client.post("/api/interview/submit", json=payload)
    assert r.status_code == 200, r.text
    from pathlib import Path
    text = Path(r.json()["yaml_path"]).read_text(encoding="utf-8")
    assert "audience:" not in text, (
        "audience=external is the default and shouldn't be emitted"
    )
    assert "top_k:" not in text, (
        "top_k=8 is the default and shouldn't be emitted"
    )


def _author_payload(**extra):  # type: ignore[no-untyped-def]
    payload = {
        "topic": "Author line test", "title": "authors",
        "output_kinds": ["paper_md", "poster"],
        "paper_format": "generic",
        "study_depth": "journal-length",
        "provider": "ollama", "provider_model": "qwen2.5:32b",
        "no_simulation": False, "clarify_mode": "auto",
        "review_panel": [], "knowledge_enabled": False,
        "comparative_baseline": "", "success_metric": "", "budget": "",
    }
    payload.update(extra)
    return payload


def test_submit_writes_the_author_line_and_poster_size(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The web form's author fields and poster size land on the config the
    generators read; a multi-line answer from the textarea becomes one line."""
    from pathlib import Path

    from fastapi.testclient import TestClient

    from core.config import Config
    from web.server import make_app

    client = TestClient(make_app(tmp_path))
    r = client.post("/api/interview/submit", json=_author_payload(
        author="  Jane \n Chen ", affiliation="R&D Lab",
        contact_email="jane@example.org", url="https://example.org/p",
        poster_size="a0_portrait",
    ))
    assert r.status_code == 200, r.text
    cfg = Config.from_yaml(Path(r.json()["yaml_path"]))
    assert (cfg.output.author, cfg.output.affiliation) == ("Jane Chen", "R&D Lab")
    assert (cfg.output.contact_email, cfg.output.url) == ("jane@example.org", "https://example.org/p")
    assert cfg.output.poster_size == "a0_portrait"


@pytest.mark.parametrize("bad", [{"poster_size": "a2_portrait"}, {"author": 5}, {"url": "x" * 301}])
def test_submit_rejects_a_bad_author_line_or_poster_size(tmp_path, bad) -> None:  # type: ignore[no-untyped-def]
    from fastapi.testclient import TestClient

    from web.server import make_app

    client = TestClient(make_app(tmp_path))
    r = client.post("/api/interview/submit", json=_author_payload(**bad))
    assert r.status_code == 400, r.text


def test_derive_tier3_seeds_reasoning_effort_default(html: str) -> None:
    """The review screen's advanced block starts Reasoning effort at the
    question default, so an untouched form writes nothing."""
    assert "reasoning_effort: 'default'" in html


def test_schema_offers_reasoning_effort_on_the_web_review_screen(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from fastapi.testclient import TestClient

    from core.config import REASONING_EFFORT_LEVELS
    from web.server import make_app

    client = TestClient(make_app(tmp_path))
    questions = client.get("/api/interview/schema").json()["questions"]
    q = next(q for q in questions if q["id"] == "reasoning_effort")
    assert "serve" in q["frontends"] and "vscode" in q["frontends"] and "cli" in q["frontends"]
    assert (q["tier"], q["kind"], q["default"]) == (3, "single", "default")
    assert [c["value"] for c in q["choices"]] == ["default", *REASONING_EFFORT_LEVELS]


def test_submit_writes_reasoning_effort_and_rejects_an_unknown_level(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from pathlib import Path

    from fastapi.testclient import TestClient

    from core.config import Config
    from web.server import make_app

    client = TestClient(make_app(tmp_path))
    r = client.post("/api/interview/submit", json=_author_payload(reasoning_effort="high"))
    assert r.status_code == 200, r.text
    assert Config.from_yaml(Path(r.json()["yaml_path"])).provider.reasoning_effort == "high"

    r = client.post("/api/interview/submit", json=_author_payload(
        title="authors-default", reasoning_effort="default",
    ))
    assert r.status_code == 200, r.text
    text = Path(r.json()["yaml_path"]).read_text(encoding="utf-8")
    assert "reasoning_effort:" not in text

    r = client.post("/api/interview/submit", json=_author_payload(reasoning_effort="extreme"))
    assert r.status_code == 400, r.text


def test_update_form_answers_carry_the_saved_reasoning_effort(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """/update/<id> seeds its fields from GET /api/quests/<id>/answers; if the
    saved level were missing there, the form would show "default" and a submit
    would drop the key from the quest's YAML."""
    from fastapi.testclient import TestClient

    from core.interview import InterviewAnswers, answers_to_yaml
    from web.server import make_app

    quest = tmp_path / "effort-quest"
    quest.mkdir()
    answers = InterviewAnswers(
        topic="t", title="effort-quest", output_kinds=["paper_md"],
        paper_format="generic", no_simulation=False, study_depth="journal-length",
        comparative_baseline="", success_metric="", budget="",
        clarify_mode="auto", review_panel=[], knowledge_enabled=False,
        provider="ollama", reasoning_effort="high",
    )
    (quest / "config.yaml").write_text(answers_to_yaml(answers, frontend="cli"), encoding="utf-8")
    r = TestClient(make_app(tmp_path)).get("/api/quests/effort-quest/answers")
    assert r.status_code == 200, r.text
    assert r.json()["reasoning_effort"] == "high"


def test_the_web_form_seeds_a_blank_page_limit_and_checks_it(html: str) -> None:
    """The review screen's advanced block starts the page limit blank, so an
    untouched form writes nothing; saving a review field and submitting an
    update both refuse anything but blank or a whole number of pages."""
    assert "page_limit: ''," in html
    assert "function validPageLimit(value)" in html
    assert "if (id === 'page_limit' && !validPageLimit(value))" in html
    assert "if (!validPageLimit(out.page_limit))" in html


def test_schema_offers_the_page_limit_on_the_web_review_screen(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from fastapi.testclient import TestClient

    from web.server import make_app

    schema = TestClient(make_app(tmp_path)).get("/api/interview/schema").json()
    q = next(q for q in schema["questions"] if q["id"] == "page_limit")
    assert {"cli", "vscode", "serve"} <= set(q["frontends"])
    assert (q["tier"], q["kind"], q["default"], q["mid_quest_editable"]) == (3, "text", "", True)


@pytest.mark.parametrize("sent, written", [(4, 4), ("4", 4), (" 6 ", 6), ("", None), (None, None)])
def test_submit_writes_a_page_limit_or_none(tmp_path, sent, written) -> None:  # type: ignore[no-untyped-def]
    from pathlib import Path

    from fastapi.testclient import TestClient

    from core.config import Config
    from web.server import make_app

    r = TestClient(make_app(tmp_path)).post("/api/interview/submit", json=_author_payload(page_limit=sent))
    assert r.status_code == 200, r.text
    assert Config.from_yaml(Path(r.json()["yaml_path"])).output.page_limit == written


@pytest.mark.parametrize("bad", [0, -2, "0", "four", "4.5", 4.5, True, [4]])
def test_submit_rejects_a_page_limit_that_is_not_a_whole_number_of_pages(tmp_path, bad) -> None:  # type: ignore[no-untyped-def]
    from fastapi.testclient import TestClient

    from web.server import make_app

    r = TestClient(make_app(tmp_path)).post("/api/interview/submit", json=_author_payload(page_limit=bad))
    assert r.status_code == 400, r.text


def _page_limit_quest(tmp_path, page_limit):  # type: ignore[no-untyped-def]
    from core.interview import InterviewAnswers, answers_to_yaml

    quest = tmp_path / "limit-quest"
    quest.mkdir()
    answers = InterviewAnswers(
        topic="t", title="limit-quest", output_kinds=["paper_md"],
        paper_format="generic", no_simulation=False, study_depth="journal-length",
        comparative_baseline="", success_metric="", budget="",
        clarify_mode="auto", review_panel=[], knowledge_enabled=False,
        provider="ollama", page_limit=page_limit,
    )
    (quest / "config.yaml").write_text(answers_to_yaml(answers, frontend="cli"), encoding="utf-8")
    return quest


def test_update_form_answers_carry_the_saved_page_limit(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """/update/<id> seeds its fields from GET /api/quests/<id>/answers; without
    the saved limit there, the form would show it blank and a submit would
    drop it from the quest's YAML."""
    from fastapi.testclient import TestClient

    from web.server import make_app

    _page_limit_quest(tmp_path, 4)
    r = TestClient(make_app(tmp_path)).get("/api/quests/limit-quest/answers")
    assert r.status_code == 200, r.text
    assert r.json()["page_limit"] == 4


def test_update_submit_changes_and_clears_the_page_limit(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The update route builds its answers field by field; the page limit must
    be one of them, or a changed limit is lost and every update clears it."""
    from fastapi.testclient import TestClient

    from core.config import Config
    from web.server import make_app

    quest = _page_limit_quest(tmp_path, 4)
    client = TestClient(make_app(tmp_path))
    saved = client.get("/api/quests/limit-quest/answers").json()

    r = client.post("/api/interview/update/limit-quest", json={**saved, "page_limit": "3"})
    assert r.status_code == 200, r.text
    assert r.json()["changes"] == {"page_limit": [4, 3]}
    assert r.json()["invalidated_stages"] == ["write", "review"]
    assert Config.from_yaml(quest / "config.yaml").output.page_limit == 3

    r = client.post("/api/interview/update/limit-quest", json={**saved, "page_limit": ""})
    assert r.status_code == 200, r.text
    assert Config.from_yaml(quest / "config.yaml").output.page_limit is None

    r = client.post("/api/interview/update/limit-quest", json={**saved, "page_limit": "four"})
    assert r.status_code == 400, r.text


def test_submit_rejects_vscode_extension_without_bridge_port(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """User-reported question: 'is it possible that i launch --serve
    but call vscode_extension?'. Yes — but only when a live bridge is
    wired. If the dashboard wasn't started with --vscode-bridge-port
    (and FI_VSCODE_BRIDGE_PORT isn't in env), submitting a quest with
    provider=vscode_extension must 400 at submit time, not crash mid-
    quest with the engine's cryptic 'requires extra[bridge_port]'."""
    from fastapi.testclient import TestClient
    from web.server import make_app

    app = make_app(tmp_path, vscode_bridge_port=0)
    client = TestClient(app)
    payload = {
        "topic": "t", "title": "t",
        "output_kinds": ["paper_md"], "paper_format": "generic",
        "study_depth": "journal-length",
        "provider": "vscode_extension", "provider_model": "",
        "no_simulation": False, "clarify_mode": "auto",
        "review_panel": [], "knowledge_enabled": False,
        "comparative_baseline": "", "success_metric": "", "budget": "",
        "audience": "external", "knowledge_top_k": 5,
    }
    r = client.post("/api/interview/submit", json=payload)
    assert r.status_code == 400
    assert "vscode_extension" in r.json()["detail"]
    assert "bridge" in r.json()["detail"].lower()


def test_submit_accepts_vscode_extension_when_bridge_port_set(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Same flow but with the dashboard launched with a live bridge —
    submit must succeed (the engine will route LLM calls through the
    bridge port the launcher inherits)."""
    from fastapi.testclient import TestClient
    from web.server import make_app

    app = make_app(tmp_path, vscode_bridge_port=12345)
    client = TestClient(app)
    payload = {
        "topic": "t", "title": "t",
        "output_kinds": ["paper_md"], "paper_format": "generic",
        "study_depth": "journal-length",
        "provider": "vscode_extension", "provider_model": "",
        "no_simulation": False, "clarify_mode": "auto",
        "review_panel": [], "knowledge_enabled": False,
        "comparative_baseline": "", "success_metric": "", "budget": "",
        "audience": "external", "knowledge_top_k": 5,
    }
    r = client.post("/api/interview/submit", json=payload)
    assert r.status_code == 200, r.text


def test_submit_rejects_bad_audience(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Defensive — invalid audience strings get a 400."""
    from fastapi.testclient import TestClient
    from web.server import make_app

    app = make_app(tmp_path)
    client = TestClient(app)
    payload = {
        "topic": "x", "title": "x",
        "output_kinds": ["paper_md"], "paper_format": "generic",
        "study_depth": "journal-length",
        "provider": "ollama", "provider_model": "x",
        "no_simulation": False, "clarify_mode": "auto",
        "review_panel": [], "knowledge_enabled": False,
        "comparative_baseline": "", "success_metric": "", "budget": "",
        "audience": "third-party",   # bogus
        "knowledge_top_k": 8,
    }
    r = client.post("/api/interview/submit", json=payload)
    assert r.status_code == 400
