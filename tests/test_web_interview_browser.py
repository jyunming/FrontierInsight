"""The web interview in a real browser, with faults injected: it never strands the person.

What a first-time user hit: a server that answered the question list with an error, dropped the connection or never
answered left the page on "Loading questions…" for good; a Launch whose answer was lost left "Launching quest…" and a
dead button; pressing Launch again could start a second quest; and a closed tab or a restarted server lost every
answer. Here each of those is injected (tests/web_browser_harness.py) and the page must say what happened, offer a way
on, keep the answers, and never start a second quest.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from tests.web_browser_harness import BASE, Site, browser_page, fill_first_screen

pytestmark = pytest.mark.slow

SCHEMA = "/api/interview/schema"
SUBMIT = "/api/interview/submit"


@pytest.fixture(autouse=True)
def _ready(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.test_web_interview import _model_ready

    _model_ready(monkeypatch)  # Launch checks the chosen model first; here it passes with no call made


def _site(tmp_path: Path) -> Site:
    return Site(tmp_path / "outputs")


def _to_review(page) -> None:  # noqa: ANN001
    fill_first_screen(page)
    page.click("#form-submit-btn")
    page.wait_for_selector("#launch-btn")


def _wait_saved_on_server(page, site: Site, topic: str, timeout: float = 10.0) -> None:  # noqa: ANN001
    # page.wait_for_timeout, not time.sleep: the page's requests are answered only while Playwright runs.
    deadline = time.monotonic() + timeout
    folder = site.output_root / "_drafts" / ".interview"
    while time.monotonic() < deadline:
        if any(topic in p.read_text(encoding="utf-8") for p in folder.glob("*.json")):
            return
        page.wait_for_timeout(100)
    raise AssertionError("the answers never reached the server")


@pytest.mark.parametrize("fault", ["fail_once", "drop_once"])
def test_a_question_list_that_does_not_load_says_so_and_retry_works(tmp_path: Path, fault: str) -> None:
    site = _site(tmp_path)
    getattr(site, fault)(SCHEMA)
    with browser_page(site) as page:
        page.goto(f"{BASE}/interview")
        error = page.wait_for_selector("#load-error")
        text = error.inner_text()
        assert "Could not load the questions" in text
        assert ("500" in text) if fault == "fail_once" else ("could not be reached" in text)
        assert "Loading questions" not in page.inner_text("#page-subtitle").replace(text, "")
        page.click("#load-retry")
        page.wait_for_selector("#q-topic")
        assert not page.query_selector("#load-error")


def test_a_server_that_never_answers_is_noticed_and_retry_works(tmp_path: Path) -> None:
    site = _site(tmp_path)
    site.hold_once(SCHEMA)
    with browser_page(site, init_script="window.__fi_fetch_timeout_ms = 1500;") as page:
        page.goto(f"{BASE}/interview")
        error = page.wait_for_selector("#load-error", timeout=10000)
        assert "did not answer within" in error.inner_text()
        page.click("#load-retry")
        page.wait_for_selector("#q-topic")


def test_a_launch_whose_answer_was_lost_can_be_retried_and_starts_one_quest(tmp_path: Path) -> None:
    site = _site(tmp_path)
    with browser_page(site) as page:
        page.goto(f"{BASE}/interview")
        _to_review(page)
        site.lose_answer_once(SUBMIT)
        page.click("#launch-btn")
        page.wait_for_selector("#result.fi-banner-error")
        assert "was not launched" in page.inner_text("#result")
        assert len(site.launches) == 1, "the server did start it; only its answer was lost"
        assert page.is_enabled("#launch-btn"), "the Launch button stayed disabled after the failure"
        page.click("#launch-btn")
        page.wait_for_url(f"{BASE}/quest/**")
        assert page.url.endswith(site.launches[0])
        assert len(site.launches) == 1, "the retry started a second quest"


def test_launch_pressed_twice_starts_one_quest(tmp_path: Path) -> None:
    site = _site(tmp_path)
    with browser_page(site) as page:
        page.goto(f"{BASE}/interview")
        _to_review(page)
        site.hold_once(SUBMIT)  # the first request is slow to answer
        page.evaluate("() => { launchFromReview(); launchFromReview(); document.getElementById('launch-btn').click(); }")
        page.wait_for_timeout(500)
        submits = [p for _m, p in site.requests if p == SUBMIT]
        assert len(submits) == 1, "a second Launch went out while the first was still waiting"
        assert page.is_disabled("#launch-btn")
        # The slow request finally gets its answer from the server.
        route = site.held.pop()
        site.fulfill(route, site.forward(route, route.request))
        page.wait_for_url(f"{BASE}/quest/**")
        assert len(site.launches) == 1


def test_a_failed_launch_keeps_the_answers_and_the_button(tmp_path: Path) -> None:
    site = _site(tmp_path)
    with browser_page(site) as page:
        page.goto(f"{BASE}/interview")
        _to_review(page)
        site.fail_once(SUBMIT, status=500)
        page.click("#launch-btn")
        page.wait_for_selector("#result.fi-banner-error")
        assert "answers are kept" in page.inner_text("#result")
        assert page.is_enabled("#launch-btn")
        assert site.launches == []
        # A reload (or the person coming back later) is back on the review screen with the same answers.
        page.reload()
        page.wait_for_selector("#launch-btn")
        assert "Heat flow through a cooling fin" in page.evaluate("() => JSON.stringify(pendingAnswers)")
        page.click("#launch-btn")
        page.wait_for_url(f"{BASE}/quest/**")
        assert len(site.launches) == 1


def test_answers_survive_a_reload_and_a_server_restart(tmp_path: Path) -> None:
    site = _site(tmp_path)
    topic = "Drag on a sphere at low Reynolds number"
    with browser_page(site) as page:
        page.goto(f"{BASE}/interview")
        page.wait_for_selector("#q-topic")
        page.fill("#q-topic", topic)
        page.select_option("#q-result_use", "explore")
        _wait_saved_on_server(page, site, topic)
        assert "d=" in page.url, "the saved answers' id is not in the address"
        # A reload in the same browser.
        page.reload()
        page.wait_for_selector("#q-topic")
        page.wait_for_function("() => document.getElementById('q-topic').value.length > 0")
        assert page.input_value("#q-topic") == topic
        assert page.input_value("#q-result_use") == "explore"
        # The server restarts and this browser's own copy is gone: the address still finds the answers.
        address = page.url
        site.restart()
        page.evaluate("() => localStorage.clear()")
        page.goto(address)
        page.wait_for_selector("#q-topic")
        page.wait_for_function("() => document.getElementById('q-topic').value.length > 0")
        assert page.input_value("#q-topic") == topic


def test_after_a_launch_a_new_interview_starts_empty(tmp_path: Path) -> None:
    site = _site(tmp_path)
    with browser_page(site) as page:
        page.goto(f"{BASE}/interview")
        _to_review(page)
        page.click("#launch-btn")
        page.wait_for_url(f"{BASE}/quest/**")
        page.goto(f"{BASE}/interview")
        page.wait_for_selector("#q-topic")
        page.wait_for_timeout(300)
        assert page.input_value("#q-topic") == ""
        assert not list((site.output_root / "_drafts" / ".interview").glob("*.json"))


def test_a_first_time_user_sees_three_steps_then_four_plain_cards(tmp_path: Path) -> None:
    """The first screen asks only the research question, what the result is for and the model (numbered as three
    steps, the byline not among them); the review screen is the four cards of core/interview.py REVIEW_CARDS, in plain
    words, with the rarer settings folded under Advanced."""
    from core.interview import REVIEW_CARDS

    site = _site(tmp_path)
    with browser_page(site) as page:
        page.goto(f"{BASE}/interview")
        page.wait_for_selector("#q-topic")
        legends = page.eval_on_selector_all(
            "#interview-form fieldset:not(.hidden) legend", "els => els.map((e) => e.innerText.trim())")
        import re

        assert {int(re.match(r"\s*(\d+)", text).group(1)) for text in legends} == {1, 2, 3}, legends
        assert "your research question" in legends[0].lower()
        assert not page.query_selector("#q-author"), "the byline is asked on the first screen"
        assert page.inner_text("#progress-pill").strip().lower().endswith("3 steps")
        _to_review(page)
        review = page.inner_text("#review-section")
        for card in REVIEW_CARDS:
            assert card["title"].lower() in review.lower()
        assert "paper byline (optional)" in review.lower()
        assert "Cost: " in review
        page.wait_for_function("() => !document.getElementById('review-ready').innerText.includes('Checking')")
        assert page.inner_text("#review-ready").startswith("Ready? ")
        shown = review.split("Advanced", 1)[0]
        for raw in ("paper_md", "paper_pdf", "methodologist", "devil_advocate", "Axon", "(unset)", "Title (short slug)"):
            assert raw not in shown, f"{raw!r} on the review cards"
        assert page.is_hidden("#review-advanced .space-y-4")


def test_answers_changed_after_a_lost_launch_answer_are_offered_as_a_new_quest(tmp_path: Path) -> None:
    """The first Launch started a quest but its answer was lost; the person changed an answer and pressed Launch again.
    The page names the quest that started, and launches the changed answers only when asked, as a new quest."""
    site = _site(tmp_path)
    with browser_page(site) as page:
        page.goto(f"{BASE}/interview")
        _to_review(page)
        site.lose_answer_once(SUBMIT)
        page.click("#launch-btn")
        page.wait_for_selector("#result.fi-banner-error")
        page.evaluate("() => { pendingAnswers.title = 'a-changed-name'; }")
        page.click("#launch-btn")
        page.wait_for_selector("#launch-changed-btn")
        assert site.launches[0] in page.inner_text("#result")
        assert len(site.launches) == 1
        page.click("#launch-changed-btn")
        page.wait_for_url(f"{BASE}/quest/**")
        assert len(site.launches) == 2 and page.url.endswith(site.launches[1])
