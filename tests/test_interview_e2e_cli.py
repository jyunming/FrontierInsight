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
