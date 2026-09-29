"""The imported ``numerical-verification`` skill (e-eight/scicomp-skills, MIT).

It goes through the same steps a person's import does: the importer copies it, the static scan reads it, a person
approves it, and only then can a quest be offered it. The fixture is the upstream file, unchanged, with its licence.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

import launch
from core.skills import approval, discover, loadable_skills, scan
from core.skills.base import Kind
from core.skills.layers import domains_of

FIXTURE = Path(__file__).parent / "fixtures" / "skills" / "numerical-verification"
SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "import_scientist_skills.py"
NAME = "numerical-verification"


@pytest.fixture()
def isk():
    spec = importlib.util.spec_from_file_location("isk_numverif", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(spec.name, None)


@pytest.fixture()
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    own = tmp_path / "fi_skills"
    own.mkdir()
    monkeypatch.setenv("FI_SKILLS_DIR", str(own))
    monkeypatch.setenv("FI_EXTERNAL_SKILLS_DIRS", "")
    monkeypatch.setenv("FI_SKILLS_APPROVALS", str(tmp_path / "approvals.json"))
    return own


def _import() -> None:
    assert launch._import_skill(str(FIXTURE / "SKILL.md"), NAME, domains="", pip_requires=[]) == 0


def test_the_import_script_knows_the_skill_and_its_source(isk) -> None:
    repo_key, rel, needs_despite = isk.SKILLS[NAME]
    assert isk.REPOS[repo_key] == "https://github.com/e-eight/scicomp-skills.git"
    assert rel == "skills/numerical-verification" and needs_despite is False
    assert repo_key in isk.KEEP_LICENSE


def test_the_licence_travels_with_the_skill_and_is_not_overwritten(isk, home: Path, tmp_path: Path) -> None:
    repo = tmp_path / "clone"
    repo.mkdir()
    (repo / "LICENSE").write_text("MIT License\n\nCopyright (c) 2026 Soham Pal\n", encoding="utf-8")
    (home / NAME).mkdir()
    isk._keep_license(repo, NAME)
    assert "Soham Pal" in (home / NAME / "LICENSE").read_text(encoding="utf-8")
    (home / NAME / "LICENSE").write_text("edited", encoding="utf-8")
    isk._keep_license(repo, NAME)
    assert (home / NAME / "LICENSE").read_text(encoding="utf-8") == "edited"
    isk._keep_license(tmp_path / "nowhere", NAME)


def test_the_fixture_carries_the_upstream_licence() -> None:
    text = (FIXTURE / "LICENSE").read_text(encoding="utf-8")
    assert text.startswith("MIT License") and "Soham Pal" in text


def test_it_imports_and_its_instructions_scan_clean(home: Path) -> None:
    _import()
    (skill,) = [s for s in discover() if s.name == NAME]
    prov = json.loads((skill.path / "provenance.json").read_text(encoding="utf-8"))
    assert prov["origin"] == "imported" and prov["domains"] == []
    findings = scan.scan(skill)
    assert not [f for f in findings if f.path.lower().endswith("skill.md")], "SKILL.md itself must scan clean"


def test_it_asks_for_no_network_credential_gpu_or_literature_search() -> None:
    """The import policy, checked on the text itself rather than trusted."""
    body = (FIXTURE / "SKILL.md").read_text(encoding="utf-8").lower()
    for word in ("api key", "api_key", "token", "password", "cuda", "conda", "arxiv", "semantic scholar", "pip install"):
        assert word not in body, word


def test_it_is_not_usable_until_a_person_approves_it(home: Path, tmp_path: Path) -> None:
    _import()
    usable, rejected = loadable_skills([NAME])
    assert not usable and [s.skill.name for s in rejected] == [NAME]


@pytest.mark.asyncio
async def test_an_approved_copy_is_offered_to_a_simulation_quest_and_can_be_chosen(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.config import (
        Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig,
    )
    from core.engine import Engine

    _import()
    (skill,) = [s for s in discover() if s.name == NAME]
    approval.approve(skill.ledger_name, skill.content_hash(), approved_by="test", note="fixture")

    asked: list[str] = []

    async def chat(self, prompt, **kw):  # noqa: ANN001
        asked.append(prompt)
        return json.dumps({
            "skills": [{"name": NAME, "reason": "checks the integrator's answer", "use": "experiment"}],
            "declined": [],
        })

    monkeypatch.setattr(Engine, "_chat", chat)
    eng = Engine(Config(
        topic="Stiff ODE integrators compared on a decay problem", title="numverif",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, skills_scan_known_dirs=False),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "outputs"),
    ))
    out = await eng._node_select_skills({"topic": "Stiff ODE integrators compared on a decay problem"})

    assert out["selected_skills"] == [NAME]
    assert out["skill_selection"]["candidates"] == 1
    assert f"{NAME}" in asked[0]

    state = {"selected_skills": [NAME], "skill_selection": out["skill_selection"]}
    full = eng._skills_block(state)
    assert "independent" in full.lower()
    # About 4 characters a token: one skill must stay a small share of a node's prompt.
    assert len(full) / 4 < 2500


def test_it_is_general_not_tied_to_a_field_and_not_dropped_for_a_simulation(home: Path) -> None:
    from core.skills.selection import build_catalogue

    _import()
    (skill,) = [s for s in discover() if s.name == NAME]
    approval.approve(skill.ledger_name, skill.content_hash(), approved_by="test", note="fixture")
    usable, _ = loadable_skills([NAME])
    assert [s.skill.name for s in usable] == [NAME]
    assert domains_of(usable[0].skill) == []
    cat = build_catalogue(usable, survey_mode=False)
    assert cat.names == {NAME}
    assert usable[0].skill.kind in (Kind.LIBRARY, Kind.TOOL)


def test_a_licence_beside_the_skill_neither_trips_the_scan_nor_blocks_approval(isk, home: Path, tmp_path: Path) -> None:
    _import()
    (skill,) = [s for s in discover() if s.name == NAME]
    isk._keep_license(FIXTURE.parent.parent / "skills" / "numerical-verification", NAME)
    assert (skill.path / "LICENSE").is_file()
    assert not [f for f in scan.scan(skill) if f.path == "LICENSE"]
    approval.approve(skill.ledger_name, skill.content_hash(), approved_by="test", note="fixture")
    usable, _ = loadable_skills([NAME])
    assert [s.skill.name for s in usable] == [NAME]


def test_a_missing_licence_is_said_out_loud(isk, home: Path, tmp_path: Path, capsys) -> None:
    (home / NAME).mkdir()
    isk._keep_license(tmp_path / "no_clone", NAME)
    assert "no LICENSE" in capsys.readouterr().out
