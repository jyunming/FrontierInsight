"""Run the VSCode interview's compiled YAML emitter
(``vscode-frontier-insight/out/interview-core.js``) and load what it writes
through the real ``Config``, so the TypeScript half of the interview is
checked against the schema the engine reads rather than by string search.

Skips when Node or a current compiled extension is missing: the Python CI
job does not build the extension (its own job compiles it)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from core.config import Config

EXT = Path(__file__).resolve().parent.parent / "vscode-frontier-insight"
CORE_JS = EXT / "out" / "interview-core.js"
CORE_TS = EXT / "src" / "interview-core.ts"

BASE = {
    "topic": "vscode emitter probe",
    "title": "vscode-probe",
    "output_kinds": ["paper_md", "poster"],
    "paper_format": "generic",
    "clarify_mode": "auto",
    "review_panel": [],
    "knowledge_enabled": False,
    "no_simulation": False,
    "study_depth": "journal-length",
    "comparative_baseline": "",
    "success_metric": "",
    "budget": "",
    "provider_model": "",
    "max_iterations": 2,
    "audience": "external",
    "knowledge_top_k": 8,
}


def _emit(tmp_path: Path, **answers) -> tuple[str, Config]:  # noqa: ANN003
    node = shutil.which("node")
    if node is None or not CORE_JS.is_file():
        pytest.skip("needs node and a compiled vscode-frontier-insight/out/interview-core.js")
    if CORE_JS.stat().st_mtime < CORE_TS.stat().st_mtime:
        pytest.skip("out/interview-core.js is older than the source; run npm run compile")
    script = (
        "const c = require(process.argv[1]);"
        "process.stdout.write(c.answersToYaml(JSON.parse(process.argv[2])));"
    )
    done = subprocess.run(
        [node, "-e", script, str(CORE_JS), json.dumps({**BASE, **answers})],
        capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    assert done.returncode == 0, done.stderr
    path = tmp_path / "vscode.yaml"
    path.write_text(done.stdout, encoding="utf-8")
    return done.stdout, Config.from_yaml(path)


def test_vscode_emitter_writes_the_author_line_and_poster_size(tmp_path: Path) -> None:
    _yaml, cfg = _emit(
        tmp_path,
        author="  陳 Jane ", affiliation='R&D "Lab"', contact_email="jane_doe@example.org",
        url="https://example.org/p", poster_size="landscape_48x36",
    )
    assert (cfg.output.author, cfg.output.affiliation) == ("陳 Jane", 'R&D "Lab"')
    assert (cfg.output.contact_email, cfg.output.url) == ("jane_doe@example.org", "https://example.org/p")
    assert cfg.output.poster_size == "landscape_48x36"


def test_vscode_emitter_writes_reasoning_effort_only_when_set(tmp_path: Path) -> None:
    yaml_text, cfg = _emit(tmp_path, reasoning_effort="high")
    assert 'reasoning_effort: "high"' in yaml_text
    assert cfg.provider.reasoning_effort == "high"
    yaml_text, cfg = _emit(tmp_path, reasoning_effort="default")
    assert "reasoning_effort:" not in yaml_text
    assert cfg.provider.reasoning_effort is None


def test_vscode_interview_offers_reasoning_effort_as_an_advanced_field() -> None:
    """The advanced-field picker in interview.ts must list the field; the
    answer type alone would leave it unreachable from @fi /new."""
    ts = (EXT / "src" / "interview.ts").read_text(encoding="utf-8")
    assert '{ label: "Reasoning effort", value: "reasoning_effort" }' in ts
    assert 'which.value === "reasoning_effort"' in ts


def test_vscode_emitter_writes_the_page_limit_only_when_set(tmp_path: Path) -> None:
    from core.config import resolve_page_limit

    yaml_text, cfg = _emit(tmp_path, page_limit=4)
    assert "  page_limit: 4" in yaml_text.splitlines()
    assert cfg.output.page_limit == 4 and resolve_page_limit(cfg) == 4
    for unset in (None, 0):
        yaml_text, cfg = _emit(tmp_path, page_limit=unset)
        assert "page_limit:" not in yaml_text and cfg.output.page_limit is None
    yaml_text, cfg = _emit(tmp_path)
    assert "page_limit:" not in yaml_text and cfg.output.page_limit is None


def test_vscode_interview_offers_the_page_limit_as_an_advanced_field() -> None:
    """The advanced-field picker in interview.ts must list the field, with an
    input box that takes blank or a whole number of pages."""
    ts = (EXT / "src" / "interview.ts").read_text(encoding="utf-8")
    assert '{ label: "Page limit", value: "page_limit" }' in ts
    assert 'which.value === "page_limit"' in ts
    assert 'validateInput: (s) => (s.trim() === "" ? null : validatePositiveInt(s))' in ts


def test_vscode_emitter_leaves_an_unset_author_line_out(tmp_path: Path) -> None:
    yaml_text, cfg = _emit(tmp_path, author="", poster_size="a1_portrait")
    for key in ("author:", "affiliation:", "contact_email:", "url:", "poster_size:"):
        assert key not in yaml_text
    assert (cfg.output.author, cfg.output.poster_size) == ("", "a1_portrait")


# --- questions that used to be unreachable from @fi /new --------------------
#
# Each of these was declared in the schema with `vscode` in its
# `frontends`, emitted by interview-core.ts, and never asked by
# interview.ts — so the VSCode YAML always carried the default. These
# tests take the answer the interview now collects all the way through
# the emitter and into a real Config, which is the only thing that
# proves the question does something.


def test_vscode_emitter_expands_the_ensemble_profile(tmp_path: Path) -> None:
    """The one that made something impossible: with no way to answer it,
    a VSCode user could not start an ensemble quest at all."""
    yaml_text, cfg = _emit(
        tmp_path, ensemble_profile="full", ensemble_models="picked-a, picked-b, picked-c",
    )
    assert "node_ensemble:" in yaml_text
    ensemble = cfg.provider.node_ensemble
    assert set(ensemble) == {"ideate", "analyze", "cross_check"}
    # The models are the ones the user ticked, in their order: FI picks none.
    assert list(ensemble["cross_check"].models) == ["picked-a", "picked-b", "picked-c"]
    assert ensemble["ideate"].moderator == "picked-a"
    # analyze MUST be tournament — the ProviderConfig validator rejects
    # synthesize there, so a wrong merge would fail Config construction.
    assert ensemble["analyze"].merge == "tournament"
    assert ensemble["cross_check"].merge == "vote"


def test_vscode_emitter_configures_no_ensemble_when_the_user_named_too_few_models(
    tmp_path: Path,
) -> None:
    """A profile without two models configures nothing, and says so, rather than
    FI filling in models of its own choosing. Same as core/interview.py."""
    for named in (None, "", "only-one"):
        answers = {"ensemble_profile": "full"}
        if named is not None:
            answers["ensemble_models"] = named
        yaml_text, cfg = _emit(tmp_path, **answers)
        assert "node_ensemble:" not in yaml_text
        assert "no ensemble is configured" in yaml_text
        assert not cfg.provider.node_ensemble
        assert "opus" not in yaml_text and "gemini" not in yaml_text


def test_vscode_emitter_writes_no_ensemble_block_when_off(tmp_path: Path) -> None:
    yaml_text, cfg = _emit(tmp_path, ensemble_profile="off")
    assert "node_ensemble:" not in yaml_text
    assert not cfg.provider.node_ensemble


def test_vscode_emitter_writes_paper_style_only_when_non_default(
    tmp_path: Path,
) -> None:
    yaml_text, cfg = _emit(tmp_path, paper_style="briefing")
    assert 'paper_style: "briefing"' in yaml_text
    assert cfg.output.paper_style == "briefing"
    yaml_text, cfg = _emit(tmp_path, paper_style="latex")
    assert "paper_style:" not in yaml_text
    assert cfg.output.paper_style == "latex"


def test_vscode_emitter_carries_the_iteration_budget(tmp_path: Path) -> None:
    yaml_text, cfg = _emit(tmp_path, max_iterations=4)
    assert "  max_iterations: 4" in yaml_text.splitlines()
    assert cfg.engine.max_iterations == 4


def test_vscode_emitter_translates_the_supply_pause(tmp_path: Path) -> None:
    """The interview asks in terms of the stage the user sees ("after
    design"); the engine names the edge it stops on ("before_build").
    The TS emitter must use the same table as core/interview.py, or the
    same answer would mean different things on different surfaces."""
    for answer, emitted in (
        ("after_literature", "after_literature"),
        ("after_design", "before_build"),
        ("after_paper", "before_review"),
        ("both", "both"),
    ):
        yaml_text, cfg = _emit(tmp_path, pause_for_user_input=answer)
        assert f'  supply: "{emitted}"' in yaml_text.splitlines()
        assert cfg.pauses.supply == emitted
    yaml_text, cfg = _emit(tmp_path, pause_for_user_input="never")
    assert "supply:" not in yaml_text
    assert cfg.pauses.supply == "never"


def test_vscode_interview_asks_the_newly_reachable_questions() -> None:
    """The emitter half is useless if nothing collects the answer — these
    are the UI hooks each question hangs from."""
    ts = (EXT / "src" / "interview.ts").read_text(encoding="utf-8")
    # ensemble_profile is an advanced field (tier 3): off by default, picked from the advanced menu.
    assert "multi-model ensemble?" in ts.lower()
    assert '{ label: "Multi-model ensemble (and its models)", value: "ensemble" }' in ts
    assert "a.ensemble_profile = v.profile" in ts
    # The paper format, deliverables and depth are tier-2 defaults, changed from the defaults menu.
    for value in ("paper_format", "output_kinds", "study_depth"):
        assert f'value: "{value}" }}' in ts and f'which.value === "{value}"' in ts
    # paper_style + max_iterations are tier-3 advanced fields.
    assert '{ label: "Paper style", value: "paper_style" }' in ts
    assert 'which.value === "paper_style"' in ts
    assert '{ label: "Design-revise iteration budget", value: "max_iterations" }' in ts
    assert 'which.value === "max_iterations"' in ts
    # pause_for_user_input is a tier-2 default.
    assert '{ label: "Pause for my papers / datasets", value: "pause_for_user_input" }' in ts
    assert 'case "pause_for_user_input"' in ts


def test_vscode_emitter_writes_the_plan_pause_only_when_asked(tmp_path: Path) -> None:
    yaml_text, cfg = _emit(tmp_path, pause_for_plan=True)
    assert '  plan: "ask"' in yaml_text.splitlines()
    assert cfg.pauses.plan == "ask"
    yaml_text, cfg = _emit(tmp_path, pause_for_plan=False)
    assert "plan:" not in yaml_text.split("pauses:")[1].split("execution:")[0]
    assert cfg.pauses.plan == "off"
    ts = (EXT / "src" / "interview.ts").read_text(encoding="utf-8")
    assert '{ label: "Stop to read and edit the plan", value: "pause_for_plan" }' in ts
    assert 'case "pause_for_plan"' in ts
