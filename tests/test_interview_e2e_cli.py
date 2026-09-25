"""E2E test for ``python launch.py --new`` — drives the CLI
interview frontend with monkeypatched stdin and a mocked preflight
LLM, then asserts the produced YAML round-trips through
``Config.model_validate`` without errors.

Skips the actual quest launch by setting ``--draft-only`` so the
test never spins up an Engine.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest
import yaml

from core.config import Config


@pytest.mark.asyncio
async def test_run_new_draft_only_produces_valid_config_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Walk the user through the interview by answering with the
    default (blank input) for every question, plus typing values for
    the three text-only research questions. Then assert the YAML
    written under ``<output_root>/_drafts/`` is a valid Config."""
    from launch import _run_new
    from core.provider import ProxySupervisor

    # Stub the preflight LLM call so the test doesn't depend on a
    # live provider. Returns a known-good dict; the interview must
    # surface these as defaults for questions 7-9.
    async def fake_preflight(
        *, topic: str, paper_format: str, provider_name: str | None,
        provider_model: str | None = None, timeout_s: float = 30.0,
    ) -> dict[str, str]:
        return {
            "comparative_baseline": "MockBaseline",
            "success_metric": "AUC >= 0.99",
            "budget": "1 minute",
        }

    import core.interview as interview_mod
    monkeypatch.setattr(interview_mod, "preflight_clarify", fake_preflight)
    # The launch.py _run_new imports preflight_clarify by name from
    # core.interview at call time; the monkeypatch above covers
    # `from core.interview import preflight_clarify` references too
    # because the original module attribute is what gets called.
    import launch as launch_mod
    # _run_new uses `from core.interview import (... preflight_clarify ...)`
    # which captures the reference at function-def time. Patch the
    # module attribute directly so the reference still resolves.
    monkeypatch.setattr("core.interview.preflight_clarify", fake_preflight)

    # Drive stdin with the answer for each question. Most answers
    # are blank (accept default); the three required text fields need
    # explicit values. Order MUST match QUESTIONS declaration order
    # in core/interview.py.
    answers = [
        "An EUV stochastic printing simulator",  # topic
        "",                                       # title (default = slug of topic)
        "",                                       # output_kinds (default)
        "",                                       # paper_format (default = generic)
        "",                                       # no_simulation (default = False)
        "",                                       # study_depth (default = journal-length)
        "",                                       # comparative_baseline (default = preflight MockBaseline)
        "",                                       # success_metric (default = preflight AUC >= 0.99)
        "",                                       # budget (default = preflight 1 minute)
        "",                                       # clarify_mode (default = auto)
        "",                                       # review_panel (default = single)
        "",                                       # knowledge_enabled (default = disabled)
        "1",                                      # provider (no default — first choice)
        "1",                                      # provider_model (no default — first choice)
    ]
    feed = io.StringIO("\n".join(answers) + "\n")
    monkeypatch.setattr("sys.stdin", feed)
    # `builtins.input` reads from sys.stdin via readline, which
    # StringIO supports. Just to be defensive:
    monkeypatch.setattr("builtins.input", lambda prompt="": feed.readline().rstrip("\n"))

    output_root = tmp_path / "outputs"
    supervisor = ProxySupervisor()
    rc = await _run_new(
        output_root=output_root,
        draft_only=True,
        vscode_bridge_port=0,
        interactive=False,
        supervisor=supervisor,
    )
    assert rc == 0, f"_run_new returned rc={rc}"

    drafts = list((output_root / "_drafts").glob("*.yaml"))
    assert len(drafts) == 1, f"expected exactly one YAML, got {drafts}"
    yaml_text = drafts[0].read_text(encoding="utf-8")

    data = yaml.safe_load(yaml_text)
    cfg = Config.model_validate(data)
    # The interview filled the three preflight slots with the
    # mocked defaults the user accepted via blank input.
    overrides = cfg.engine.clarify_overrides
    assert overrides["comparative_baseline"] == "MockBaseline"
    assert overrides["success_metric"] == "AUC >= 0.99"
    assert overrides["budget"] == "1 minute"
    # Smart-default cascade: paper_format=generic → no_simulation=False.
    assert cfg.engine.no_simulation is False
    # Provider was the first choice in PROVIDER_CHOICES.
    from core.interview import PROVIDER_CHOICES, PROVIDER_MODEL_OPTIONS
    assert cfg.provider.name == PROVIDER_CHOICES[0].value
    # Model picker was the first of that provider's curated list.
    first_model = PROVIDER_MODEL_OPTIONS[cfg.provider.name][0].value
    assert cfg.provider.model == first_model


@pytest.mark.asyncio
async def test_run_new_writes_the_tier1_ensemble_pick_and_author_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ensemble profile and the author line are tier-1 answers. The CLI
    used to read the ensemble pick from the tier-3 dict, where it never is,
    so every CLI quest ran with the ensemble off whatever the user chose."""
    from launch import _run_new
    from core.provider import ProxySupervisor

    async def fake_preflight(**_kw: Any) -> dict[str, str]:
        return {}

    monkeypatch.setattr("core.interview.preflight_clarify", fake_preflight)
    # One answer per tier-1 question, in order, then Enter to launch
    # from the review screen.
    answers = iter([
        "Author line probe topic",   # topic
        "",                          # result_use (default research)
        "",                          # paper_format (default generic)
        "",                          # output_kinds (default)
        "",                          # study_depth (default)
        "1",                         # provider
        "1",                         # provider_model
        "4",                         # ensemble_profile: full
        "model-a, model-b, model-c",  # ensemble_models: the user's own pick
        "  Jane   Chen ",            # author
        "R&D Lab",                   # affiliation
        "",                          # contact_email (skipped)
        "https://example.org/p",     # url
        "",                          # review screen: launch
    ])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers, ""))

    output_root = tmp_path / "outputs"
    rc = await _run_new(
        output_root=output_root,
        draft_only=True,
        vscode_bridge_port=0,
        interactive=False,
        supervisor=ProxySupervisor(),
    )
    assert rc == 0
    (draft,) = list((output_root / "_drafts").glob("*.yaml"))
    cfg = Config.model_validate(yaml.safe_load(draft.read_text(encoding="utf-8")))
    assert set(cfg.provider.node_ensemble) == {"cross_check", "ideate", "analyze"}
    # Exactly the models the user typed, in their order; FI added none.
    assert list(cfg.provider.node_ensemble["cross_check"].models) == [
        "model-a", "model-b", "model-c",
    ]
    assert (cfg.output.author, cfg.output.affiliation) == ("Jane Chen", "R&D Lab")
    assert (cfg.output.contact_email, cfg.output.url) == ("", "https://example.org/p")


@pytest.mark.asyncio
async def test_run_new_configures_no_ensemble_when_the_user_names_no_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Picking a fan-out profile but naming no models must not fall back to
    models FI likes: the quest runs single-model and the draft says why."""
    from launch import _run_new
    from core.provider import ProxySupervisor

    async def fake_preflight(**_kw: Any) -> dict[str, str]:
        return {}

    monkeypatch.setattr("core.interview.preflight_clarify", fake_preflight)
    answers = iter([
        "Ensemble without models probe", "", "", "", "", "1", "1",  # topic, result_use, format, kinds, depth, provider, model
        "4",                         # ensemble_profile: full
        "",                          # ensemble_models: none named
    ])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers, ""))

    output_root = tmp_path / "outputs"
    rc = await _run_new(
        output_root=output_root, draft_only=True, vscode_bridge_port=0,
        interactive=False, supervisor=ProxySupervisor(),
    )
    assert rc == 0
    (draft,) = list((output_root / "_drafts").glob("*.yaml"))
    text = draft.read_text(encoding="utf-8")
    cfg = Config.model_validate(yaml.safe_load(text))
    assert not cfg.provider.node_ensemble
    assert "no ensemble is configured" in text
    assert "opus" not in text and "gemini" not in text


@pytest.mark.asyncio
async def test_run_new_writes_a_page_limit_typed_on_the_review_screen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--new asks the page limit as an advanced field on the review screen. A
    typo is refused and the field stays blank; a whole number lands on
    output.page_limit, which the engine then reads as the limit."""
    from core.config import resolve_page_limit
    from core.interview import questions_for_tier
    from core.provider import ProxySupervisor
    from launch import _run_new

    async def fake_preflight(**_kw: Any) -> dict[str, str]:
        return {}

    monkeypatch.setattr("core.interview.preflight_clarify", fake_preflight)
    rows = [q.id for q in questions_for_tier(2, "cli")] + [q.id for q in questions_for_tier(3, "cli")]
    row = str(rows.index("page_limit") + 1)
    answers = iter([
        "Page limit probe topic",    # topic
        "",                          # result_use (default research)
        "", "", "",                  # paper_format, output_kinds, study_depth
        "1", "1",                    # provider, provider_model
        "", "", "", "", "",          # ensemble_profile, author line
        "a", row, "four",            # review screen: show advanced, a typo (refused)
        row, "4",                    # the page limit
        "",                          # launch
    ])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers, ""))

    output_root = tmp_path / "outputs"
    rc = await _run_new(
        output_root=output_root,
        draft_only=True,
        vscode_bridge_port=0,
        interactive=False,
        supervisor=ProxySupervisor(),
    )
    assert rc == 0
    assert next(answers, None) is None, "every scripted answer was read"
    (draft,) = list((output_root / "_drafts").glob("*.yaml"))
    text = draft.read_text(encoding="utf-8")
    assert "  page_limit: 4" in text.splitlines()
    cfg = Config.model_validate(yaml.safe_load(text))
    assert cfg.output.page_limit == 4 and resolve_page_limit(cfg) == 4


@pytest.mark.asyncio
@pytest.mark.parametrize("pick, profile, draft", [("", "research", False), ("2", "research", False), ("3", "default", True)])
async def test_run_new_writes_what_the_result_is_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    pick: str, profile: str, draft: bool,
) -> None:
    """"What is the result for?" is the second question. Research (the default) and a decision write the research
    profile, and the review screen already lists the reviewers that profile runs; exploring writes the cheaper draft's
    three engine settings. The same answer writes the same settings on every interface."""
    from launch import _run_new
    from core.provider import ProxySupervisor

    async def fake_preflight(**_kw: Any) -> dict[str, str]:
        return {}

    monkeypatch.setattr("core.interview.preflight_clarify", fake_preflight)
    # topic, result_use, paper_format, output_kinds, study_depth, provider, provider_model; then Enter to launch.
    answers = iter(["Result use probe topic", pick, "", "", "", "1", "1", "", "", "", "", "", ""])
    reads = {"n": 0}

    def fake_input(prompt: str = "") -> str:
        reads["n"] += 1
        # A scripted answer list that falls out of step would otherwise re-ask forever.
        assert reads["n"] < 60, "the interview asked far more questions than scripted"
        return next(answers, "")

    monkeypatch.setattr("builtins.input", fake_input)
    output_root = tmp_path / "outputs"
    rc = await _run_new(
        output_root=output_root, draft_only=True, vscode_bridge_port=0,
        interactive=False, supervisor=ProxySupervisor(),
    )
    assert rc == 0
    shown = capsys.readouterr().out
    (path,) = list((output_root / "_drafts").glob("*.yaml"))
    text = path.read_text(encoding="utf-8")
    cfg = Config.model_validate(yaml.safe_load(text))
    assert cfg.rigor_profile == profile
    assert (cfg.engine.ideate_reflect is False) is draft
    assert (cfg.engine.cross_check_per_finding_k == 0) is draft
    if profile == "research":
        # The review screen shows the panel that runs, reproducibility included, before launch.
        assert "reproducibility" in shown.split("Review before launch", 1)[1]
        assert "reproducibility" in cfg.engine.review_panel
    assert "Result for" in shown
