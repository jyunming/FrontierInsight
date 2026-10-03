"""What a first-time person sees, the same on the CLI, the web page and VS Code.

Bare ``fi`` offers three next steps (a terminal) or the short help (anything else). The first interview asks three
things (core/interview.py FIRST_STEPS); the review screen is four plain cards (REVIEW_CARDS) with every other setting
on exactly one card or under its Advanced part; the paper byline is folded there and asked once before the first paper
(launch._ask_byline_once), never blocking a quest. The web page and VS Code read the same steps and cards: the web page
from the schema, VS Code from a generated copy in interview-core.ts that must stay equal to it.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.interview import (
    BYLINE_FIELDS, FIRST_STEPS, QUESTIONS, REVIEW_CARDS, export_schema_json, plain_value, questions_for_tier,
)

REPO = Path(__file__).resolve().parent.parent
CORE_TS = REPO / "vscode-frontier-insight" / "src" / "interview-core.ts"
INTERVIEW_TS = REPO / "vscode-frontier-insight" / "src" / "interview.ts"
INTERVIEW_HTML = REPO / "web" / "static" / "interview.html"


# --- the three first steps and the four cards -------------------------------------------------------------------------


@pytest.mark.parametrize("frontend", ["cli", "serve", "vscode"])
def test_the_first_screen_is_three_steps_and_nothing_else(frontend: str) -> None:
    assert len(FIRST_STEPS) == 3
    in_steps = [qid for _sid, _title, qids in FIRST_STEPS for qid in qids]
    asked = [q.id for q in questions_for_tier(1, frontend)]
    assert asked == [qid for qid in in_steps if qid in asked], "a first-screen question is in no step, or out of order"
    assert not set(asked) & set(BYLINE_FIELDS), "the byline is asked up front"


def test_every_setting_is_on_exactly_one_card() -> None:
    placed = [qid for card in REVIEW_CARDS for qid in (*card["shown"], *card["advanced"])]
    assert len(placed) == len(set(placed)), "a setting is on two cards"
    assert set(placed) == {q.id for q in QUESTIONS} - {"topic"}
    assert [c["id"] for c in REVIEW_CARDS] == ["checks", "models", "data", "outputs"]


def test_no_internal_name_is_shown_on_a_card() -> None:
    """A card row's label and every value it can show by default read as plain words."""
    shown = {qid for card in REVIEW_CARDS for qid in card["shown"]}
    for q in QUESTIONS:
        if q.id not in shown:
            continue
        assert "Axon" not in q.label and "slug" not in q.label and "_" not in q.label, q.label
    assert plain_value("output_kinds", ["paper_md", "slides"]) == "paper (Markdown), slides"
    assert plain_value("review_panel", ["methodologist", "devil_advocate"]) == "method, devil's advocate"
    assert plain_value("review_panel", []) == "single reviewer"
    assert plain_value("page_limit", "") == "(not set)"


def test_vscode_carries_the_same_steps_cards_and_words() -> None:
    m = re.search(r"export const REVIEW_SCREEN = (\{.*?\}) as const;", CORE_TS.read_text(encoding="utf-8"), re.S)
    assert m, "REVIEW_SCREEN not found in interview-core.ts"
    schema = export_schema_json()
    keys = ("first_steps", "review_cards", "plain_values", "cost_notes", "checks_sentences", "byline_empty",
            "byline_fields")
    assert json.loads(m.group(1)) == {k: schema[k] for k in keys}, (
        "interview-core.ts REVIEW_SCREEN differs from core/interview.py: regenerate it from the schema")


def test_every_surface_builds_its_review_from_the_cards() -> None:
    html = INTERVIEW_HTML.read_text(encoding="utf-8")
    assert "schema.review_cards" in html and "schema.first_steps" in html
    assert "cardRows(card.shown)" in html and "cardRows(card.advanced)" in html
    ts = INTERVIEW_TS.read_text(encoding="utf-8")
    assert "REVIEW_SCREEN.review_cards" in ts and "REVIEW_SCREEN.first_steps" in ts
    run = ts[ts.index("export async function runInterview"):ts.index("function reviewBlockMarkdown")]
    assert "askAuthorLine(" not in run, "VS Code asks the byline before the review"


def test_vscode_has_a_row_for_every_setting_a_card_shows_there() -> None:
    """VS Code builds each card's rows from REVIEW_SCREEN.review_cards through VSCODE_CARD_ROWS; a setting added to a
    card in core/interview.py must have a row there, or it would silently not show in VS Code."""
    ts = INTERVIEW_TS.read_text(encoding="utf-8")
    block = ts[ts.index("export const VSCODE_CARD_ROWS"):ts.index("function reviewBlockMarkdown")]
    keys = set(re.findall(r"^\s{4}(\w+): \[", block, re.M))
    vscode_ids = {q.id for q in QUESTIONS if "vscode" in q.frontends}
    needed = {qid for card in REVIEW_CARDS for qid in card["shown"] if qid in vscode_ids} - set(BYLINE_FIELDS[1:])
    assert needed - keys == set(), f"no VS Code row for {sorted(needed - keys)}"
    assert "for (const id of card.shown" in ts


def test_only_a_single_terminal_quest_asks_the_byline() -> None:
    """A fleet, the web server's in-process quests, --emit and --watch never ask: only --config and --new pass
    ask_byline=True."""
    src = (REPO / "launch.py").read_text(encoding="utf-8")
    assert src.count("ask_byline=True") == 2
    assert "ask_byline" not in (REPO / "web" / "server.py").read_text(encoding="utf-8")
    fleet = src[src.index("async def run_fleet"):]
    fleet = fleet[:fleet.index("\nasync def ") if "\nasync def " in fleet else len(fleet)]
    assert "ask_byline" not in fleet


# --- the byline, asked once before the first paper --------------------------------------------------------------------


def _cfg(**output) -> SimpleNamespace:
    base = {"author": "", "affiliation": "", "contact_email": "", "url": "", "kinds": ["paper_md", "paper_pdf"]}
    return SimpleNamespace(output=SimpleNamespace(**{**base, **output}))


def _terminal(monkeypatch: pytest.MonkeyPatch, *, yes: bool = True) -> None:
    import launch

    monkeypatch.setattr(launch, "_stdin_is_terminal", lambda: yes)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: yes, raising=False)


def _typed(monkeypatch: pytest.MonkeyPatch, answer, asked: list | None = None) -> None:  # noqa: ANN001
    """What the person types at the byline question (launch._read_line_until reads the terminal itself)."""
    import launch

    def read(prompt: str, timeout_s: float, stop=None):  # noqa: ANN001, ANN202
        if asked is not None:
            asked.append(prompt)
        return answer(timeout_s) if callable(answer) else answer

    monkeypatch.setattr(launch, "_read_line_until", read)


def test_the_byline_is_asked_once_used_and_kept(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import launch
    from core import profile

    _terminal(monkeypatch)
    asked: list[str] = []
    _typed(monkeypatch, "  Jane   Chen ", asked)
    cfg = _cfg()
    asyncio.run(launch._ask_byline_once(cfg, tmp_path))
    assert cfg.output.author == "Jane Chen" and len(asked) == 1
    assert profile.load()["author"] == "Jane Chen"
    # Asked once: the next quest with no byline of its own is not asked again.
    again = _cfg()
    asyncio.run(launch._ask_byline_once(again, tmp_path / "other"))
    assert len(asked) == 1
    # This quest keeps it: a later --resume / --emit of it prints the name though its config has none.
    later = _cfg()
    launch._apply_quest_byline(later, tmp_path)
    assert later.output.author == "Jane Chen"
    other = _cfg()
    launch._apply_quest_byline(other, tmp_path / "other")
    assert other.output.author == ""


def test_enter_keeps_the_frontier_insight_byline_and_is_not_asked_again(monkeypatch: pytest.MonkeyPatch) -> None:
    import launch
    from core import profile

    _terminal(monkeypatch)
    _typed(monkeypatch, "")
    cfg = _cfg()
    asyncio.run(launch._ask_byline_once(cfg))
    assert cfg.output.author == ""
    assert profile.load() == {"author": "", "affiliation": "", "contact_email": "", "url": ""}


@pytest.mark.parametrize("case", ["no terminal", "quest has a byline", "no paper, slides or poster"])
def test_the_byline_is_not_asked_when_it_should_not_be(monkeypatch: pytest.MonkeyPatch, case: str) -> None:
    import launch
    from core import profile

    _terminal(monkeypatch, yes=case != "no terminal")
    _typed(monkeypatch, lambda _t: pytest.fail("asked"))
    cfg = _cfg(author="Ann Lee") if case == "quest has a byline" else (
        _cfg(kinds=["speech"]) if case == "no paper, slides or poster" else _cfg())
    asyncio.run(launch._ask_byline_once(cfg))
    assert profile.load() is None


def test_nobody_answering_never_holds_the_quest(monkeypatch: pytest.MonkeyPatch) -> None:
    """No answer by the deadline: the reader returns None (it polls the terminal, so nothing is left reading stdin to
    swallow a later prompt's line), the paper goes on without a byline, and it is asked again next time."""
    import launch
    from core import profile

    _terminal(monkeypatch)
    _typed(monkeypatch, None)
    cfg = _cfg()
    asyncio.run(asyncio.wait_for(launch._ask_byline_once(cfg), timeout=5))
    assert cfg.output.author == "" and profile.load() is None, "an unanswered question counts as answered"


def test_the_terminal_reader_gives_up_by_its_deadline_and_leaves_nothing_reading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threading
    import time

    import launch

    if os.name == "nt":
        import msvcrt

        monkeypatch.setattr(msvcrt, "kbhit", lambda: False)
    else:
        import select

        monkeypatch.setattr(select, "select", lambda r, w, x, t: (time.sleep(t), ([], [], []))[1])
    before = threading.active_count()
    started = time.monotonic()
    assert launch._read_line_until("? ", 0.4) is None
    assert time.monotonic() - started < 3
    assert threading.active_count() <= before
    # Interrupted (Ctrl-C cancels the waiting task): it returns at once, not at the end of its wait.
    stop = threading.Event()
    stop.set()
    started = time.monotonic()
    assert launch._read_line_until("? ", 60, stop) is None
    assert time.monotonic() - started < 3


# --- bare `fi` ----------------------------------------------------------------------------------------------------------


def test_bare_fi_with_no_terminal_prints_the_short_help_and_exits_2(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    import launch

    _terminal(monkeypatch, yes=False)
    with pytest.raises(SystemExit) as stop:
        launch._start_menu(ask=lambda _p: pytest.fail("asked"))
    assert stop.value.code == 2
    err = capsys.readouterr().err
    assert err.startswith("usage: fi") and "--help-all" in err
    assert "--digest-provider" not in err, "the full flag wall"


@pytest.mark.parametrize("typed, argv", [(["2"], ["--serve"]), (["x", "9", "1"], ["--new"]), (["3"], ["--doctor"])])
def test_bare_fi_in_a_terminal_offers_three_next_steps(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], typed: list[str], argv: list[str],
) -> None:
    import launch

    _terminal(monkeypatch)
    answers = iter(typed)
    assert launch._start_menu(ask=lambda _p: next(answers)) == argv
    out = capsys.readouterr().out
    for label in ("Start a new quest", "Open the web app", "Check my setup"):
        assert label in out
    assert "usage:" not in out


@pytest.mark.parametrize("typed", ["", "q"])
def test_bare_fi_enter_leaves(monkeypatch: pytest.MonkeyPatch, typed: str) -> None:
    import launch

    _terminal(monkeypatch)
    with pytest.raises(SystemExit) as stop:
        launch._start_menu(ask=lambda _p: typed)
    assert stop.value.code == 0


def test_bare_fi_from_the_command_line_with_no_terminal(tmp_path: Path) -> None:
    """The real entry point: `python launch.py` with stdin and stdout not a terminal."""
    import os
    import subprocess

    env = {**os.environ, "FI_SKIP_BOOTSTRAP": "1"}
    done = subprocess.run([sys.executable, str(REPO / "launch.py")], capture_output=True, text=True,
                          stdin=subprocess.DEVNULL, timeout=120, cwd=tmp_path, env=env)
    assert done.returncode == 2, done.stderr[-2000:]
    assert done.stderr.startswith("usage: fi") and "is required" not in done.stderr
