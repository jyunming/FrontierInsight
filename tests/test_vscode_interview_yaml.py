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
