"""Skills other agents installed (~/.codex/skills, ~/.claude/skills, ...) are read
in place: found, searchable, selectable, and usable once a person approves their
exact content. They have no FI self-test, so approval is their only gate, and
nothing about them is copied or written."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import launch
from core.skills import approval, discover, evaluate, loadable_skills
from core.skills.base import Status
from core.skills.registry import (
    _from_filesystem, configure_external_dirs, external_skill_dirs,
)


def _skill(root: Path, name: str, *, selftest: bool = False, script: bool = False) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: an external demo skill\n---\nDo the thing.\n",
        encoding="utf-8",
    )
    if selftest:
        (d / "selftest.py").write_text("import sys; sys.exit(0)", encoding="utf-8")
    if script:
        (d / "scripts").mkdir()
        (d / "scripts" / "run.sh").write_text("echo hi\n", encoding="utf-8")
    return d


@pytest.fixture()
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    own = tmp_path / "fi_skills"
    ext = tmp_path / "other_agent" / "skills"
    own.mkdir()
    ext.mkdir(parents=True)
    monkeypatch.setenv("FI_SKILLS_DIR", str(own))
    monkeypatch.setenv("FI_EXTERNAL_SKILLS_DIRS", str(ext))
    monkeypatch.setenv("FI_SKILLS_APPROVALS", str(tmp_path / "approvals.json"))
    return {"own": own, "ext": ext, "ledger": tmp_path / "approvals.json"}


def test_external_skills_are_discovered_in_place(env) -> None:
    _skill(env["ext"], "deepscientist-experiment")
    (found,) = discover()
    assert found.name == "deepscientist-experiment"
    assert found.external and found.path == env["ext"] / "deepscientist-experiment"


def test_an_explicit_folder_stays_hermetic(env) -> None:
    """Callers that pass a skills_dir look at that folder alone."""
    _skill(env["ext"], "outside")
    assert discover(env["own"]) == []


def test_fi_skill_shadows_an_external_one_of_the_same_name(env) -> None:
    _skill(env["own"], "same", selftest=True)
    _skill(env["ext"], "same")
    (found,) = discover()
    assert not found.external and found.path == env["own"] / "same"


def test_a_folder_without_skill_md_is_not_a_skill(env) -> None:
    (env["ext"] / "hooks").mkdir()
    assert discover() == []


def test_an_untested_external_skill_waits_for_approval_then_is_trusted(env) -> None:
    skill_dir = _skill(env["ext"], "ext-one", script=True)
    (skill,) = discover()

    st = evaluate(skill)
    assert st.status is Status.PROPOSED and not st.loadable
    assert "external skill" in st.reason and "never self-tested" in st.reason

    approval.approve(skill.ledger_name, skill.content_hash(), approved_by="me", note="t")
    st = evaluate(skill)
    assert st.status is Status.TRUSTED and st.loadable
    assert "no self-test" in st.reason

    (skill_dir / "scripts" / "run.sh").write_text("rm -rf /\n", encoding="utf-8")
    st = evaluate(skill)
    assert st.status is Status.PROPOSED and "content changed since approval" in st.reason


def test_an_fi_skill_without_a_selftest_is_still_never_promoted(env) -> None:
    """The exemption is for external skills only."""
    _skill(env["own"], "fi-notest")
    (skill,) = discover()
    approval.approve(skill.ledger_name, skill.content_hash(), approved_by="me", note="t")
    assert evaluate(skill).status is Status.UNTESTED


def test_an_external_skill_that_does_carry_a_selftest_is_tested(env) -> None:
    _skill(env["ext"], "ext-tested", selftest=True)
    (skill,) = discover()
    approval.approve(skill.ledger_name, skill.content_hash(), approved_by="me", note="t")
    assert evaluate(skill).status is Status.TRUSTED
    (skill.path / "selftest.py").write_text("raise SystemExit(1)", encoding="utf-8")
    approval.approve(skill.ledger_name, skill.content_hash(), approved_by="me", note="t")
    assert evaluate(skill).status is Status.QUARANTINED


def test_approving_an_external_skill_does_not_touch_an_fi_skill_of_that_name(env) -> None:
    """The ledger is keyed by name, so an external skill's approval is
    namespaced; it must neither inherit nor overwrite one for an FI skill."""
    _skill(env["own"], "clash", selftest=True)
    fi_skill = discover()[0]
    approval.approve(fi_skill.ledger_name, fi_skill.content_hash(), approved_by="me", note="t")

    other = _skill(env["ext"].parent / "second", "clash")
    (ext_skill,) = _from_filesystem(other.parent, source="external")
    assert ext_skill.ledger_name == "external:clash"
    assert approval.approved_hash(ext_skill.ledger_name) is None
    assert approval.approved_hash(fi_skill.ledger_name) == fi_skill.content_hash()


def test_bulk_approval_leaves_external_skills_out(env, capsys) -> None:
    _skill(env["own"], "fi-one", selftest=True)
    _skill(env["ext"], "ext-one")

    rc = launch._approve_all_skills("me")

    assert rc == 0
    assert "Leaving 1 external skill(s) out" in capsys.readouterr().out
    ledger = json.loads(env["ledger"].read_text(encoding="utf-8"))
    assert set(ledger) == {"fi-one"}, "an external skill must not be bulk-approved"


def test_one_at_a_time_approval_of_an_external_skill_records_its_namespace(env, capsys) -> None:
    _skill(env["ext"], "ext-one", script=True)

    assert launch._approve_skill("ext-one", "me") == 0

    out = capsys.readouterr().out
    assert "external skill, read in place" in out and "run.sh" in out
    ledger = json.loads(env["ledger"].read_text(encoding="utf-8"))
    assert set(ledger) == {"external:ext-one"}
    assert launch._revoke_skill("ext-one") == 0
    assert json.loads(env["ledger"].read_text(encoding="utf-8")) == {}


def test_loadable_skills_returns_an_approved_external_skill(env) -> None:
    _skill(env["ext"], "ext-one")
    (skill,) = discover()
    approval.approve(skill.ledger_name, skill.content_hash(), approved_by="me", note="t")
    usable, rejected = loadable_skills(["ext-one"], use_cache=False)
    assert [s.skill.name for s in usable] == ["ext-one"] and rejected == []


def test_recording_a_quest_never_writes_into_an_external_skill(env) -> None:
    from core.skills.usage import record_quest

    skill_dir = _skill(env["ext"], "ext-one")
    before = sorted(p.name for p in skill_dir.iterdir())

    assert record_quest(["ext-one"], "q1", "accepted") == []

    assert sorted(p.name for p in skill_dir.iterdir()) == before
    assert not (skill_dir / "provenance.json").exists()


def _mk(root: Path, *parts: str, skill: str = "s") -> Path:
    """<root>/<parts...>/<skill>/SKILL.md, returning the skills folder."""
    folder = root.joinpath(*parts)
    (folder / skill).mkdir(parents=True)
    (folder / skill / "SKILL.md").write_text(f"---\nname: {skill}\n---\nx\n", encoding="utf-8")
    return folder


@pytest.fixture()
def machine(tmp_path: Path, monkeypatch) -> dict[str, Path]:
    """A fake home directory and project, with the per-OS variables pointed into
    it: the common locations are searched, not named."""
    from core.skills import registry

    home = tmp_path / "home"
    project = tmp_path / "work" / "proj" / "sub"
    home.mkdir()
    project.mkdir(parents=True)
    monkeypatch.delenv("FI_EXTERNAL_SKILLS_DIRS", raising=False)
    monkeypatch.setenv("FI_SKILLS_DIR", str(tmp_path / "own"))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    for var in ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "APPDATA", "LOCALAPPDATA",
                "CODEX_HOME", "CLAUDE_CONFIG_DIR"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(project)
    registry._common_cache = None
    yield {"home": home, "project": project}
    registry._common_cache = None
    configure_external_dirs([], scan_known=True)


def _names(dirs: list[Path], root: Path) -> set[str]:
    return {d.relative_to(root).as_posix() for d in dirs}


def test_common_locations_are_searched_with_nothing_named(machine) -> None:
    """Linux, macOS and Windows locations, the tools' plugin trees, unknown
    tools, and the project, all found without a path being given."""
    home, project = machine["home"], machine["project"]
    _mk(home, ".codex", "skills")                                  # a known tool
    _mk(home, ".claude", "skills")
    _mk(home, ".someothertool", "skills")                          # a tool nobody listed
    _mk(home, ".claude", "plugins", "cache", "mkt", "plug", "abc123", "skills")   # a plugin
    _mk(home, ".config", "linuxtool", "skills")                    # XDG default
    _mk(home, ".local", "share", "sharetool", "skills")
    _mk(home, "Library", "Application Support", "mactool", "skills")  # macOS
    _mk(home, "AppData", "Roaming", "wintool", "skills")           # Windows
    _mk(home, "AppData", "Local", "winlocal", "skills")
    _mk(project.parent, ".agents", "skills")                       # project-level, in a parent
    _mk(project, "skills")                                         # ./skills
    # noise that must not count
    (home / ".emptytool" / "skills" / "not-a-skill").mkdir(parents=True)
    _mk(home, ".claude", "plugins", "node_modules", "deep", "skills")
    import os
    os.environ["APPDATA"] = str(home / "AppData" / "Roaming")
    os.environ["LOCALAPPDATA"] = str(home / "AppData" / "Local")
    from core.skills import registry
    registry._common_cache = None

    try:
        found = _names(external_skill_dirs(), machine["home"].parent)
    finally:
        os.environ.pop("APPDATA"); os.environ.pop("LOCALAPPDATA")

    expected = {
        "home/.codex/skills", "home/.claude/skills", "home/.someothertool/skills",
        "home/.claude/plugins/cache/mkt/plug/abc123/skills", "home/.config/linuxtool/skills",
        "home/.local/share/sharetool/skills", "home/Library/Application Support/mactool/skills",
        "home/AppData/Roaming/wintool/skills", "home/AppData/Local/winlocal/skills",
        "work/proj/.agents/skills", "work/proj/sub/skills",
    }
    assert expected <= found, sorted(expected - found)
    assert not any("emptytool" in f or "node_modules" in f for f in found)


def test_the_tools_own_variables_are_followed(machine, monkeypatch) -> None:
    other = machine["home"].parent / "elsewhere"
    _mk(other, "codex_home", "skills")
    _mk(other, "claude_cfg", "skills")
    monkeypatch.setenv("CODEX_HOME", str(other / "codex_home"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(other / "claude_cfg"))
    found = _names(external_skill_dirs(), other)
    assert {"codex_home/skills", "claude_cfg/skills"} <= found


def test_the_newest_plugin_version_wins_a_name_clash(machine) -> None:
    import os
    home = machine["home"]
    # The newer version is created FIRST and sorts LAST by name, so neither
    # creation order nor alphabetical order can make this pass by accident.
    new = _mk(home, ".claude", "plugins", "cache", "m", "p", "zzz999", "skills", skill="same")
    old = _mk(home, ".claude", "plugins", "cache", "m", "p", "aaa111", "skills", skill="same")
    (new / "same" / "SKILL.md").write_text("newer", encoding="utf-8")
    os.utime(old, (1_000_000_000, 1_000_000_000))
    os.utime(new, (1_900_000_000, 1_900_000_000))
    from core.skills import registry
    registry._common_cache = None

    (skill,) = [s for s in discover() if s.name == "same"]

    assert skill.path == new / "same"


def test_configured_folders_come_first_and_the_search_can_be_turned_off(machine) -> None:
    home = machine["home"]
    _mk(home, ".codex", "skills")
    mine = _mk(machine["home"].parent, "mine")
    configure_external_dirs([str(mine)], scan_known=True)
    dirs = external_skill_dirs()
    assert dirs[0] == mine and home / ".codex" / "skills" in dirs
    configure_external_dirs([str(mine)], scan_known=False)
    assert external_skill_dirs() == [mine]


def test_the_environment_override_still_replaces_everything(machine, monkeypatch) -> None:
    _mk(machine["home"], ".codex", "skills")
    monkeypatch.setenv("FI_EXTERNAL_SKILLS_DIRS", "")
    assert external_skill_dirs() == []


def test_an_unapproved_external_skill_is_not_hashed_or_scanned_in_bulk(env, monkeypatch) -> None:
    """Other agents' folders hold hundreds of skills, some of hundreds of files;
    reading all of them for every quest cost 25 seconds."""
    from core.skills import registry

    _skill(env["ext"], "ext-unasked", script=True)
    _skill(env["ext"], "ext-approved", script=True)
    approved = next(s for s in discover() if s.name == "ext-approved")
    approval.approve(approved.ledger_name, approved.content_hash(), approved_by="me", note="t")
    scanned: list[str] = []
    monkeypatch.setattr(registry, "_scan_findings", lambda s: (scanned.append(s.name) or [], True))

    usable, rejected = loadable_skills(["ext-unasked", "ext-approved"], use_cache=False)

    assert [s.skill.name for s in usable] == ["ext-approved"]
    assert [s.skill.name for s in rejected] == ["ext-unasked"]
    assert scanned == ["ext-approved"], "only the skill somebody approved is read"
    assert "awaiting approval" in rejected[0].reason and "--scan-skill" in rejected[0].reason


@pytest.mark.asyncio
async def test_the_run_log_says_once_that_other_agents_skills_were_found(
    env, tmp_path, monkeypatch,
) -> None:
    from core.config import (
        Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig,
        ProviderConfig,
    )
    from core.engine import Engine
    from tests.test_engine_smoke import _fake_response_for

    for n in range(6):
        _skill(env["ext"], f"ext-{n}")

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        return _fake_response_for(messages[-1]["content"])

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)
    engine = Engine(Config(
        topic="skills log probe", title="skills-log",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, auto_accept_on_pass=True),
        execution=ExecutionConfig(sandbox="venv", timeout_s=120),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "outputs"),
    ))
    await engine.run()

    log = (engine.fi_dir / "run.log").read_text(encoding="utf-8")
    assert "found 6 skill(s) in 1 folder(s) of other agents" in log
    assert "note: 'ext-" not in log, "six unasked-for skills must not be six warnings"


# --- engine.skills_required -------------------------------------------------


def test_required_skills_that_cannot_be_used_are_named_with_the_reason(env) -> None:
    from core.engine import _raise_if_required_skills_unusable

    _skill(env["ext"], "ext-one")
    usable, rejected = loadable_skills(["ext-one", "nope"], use_cache=False)

    with pytest.raises(RuntimeError) as exc:
        _raise_if_required_skills_unusable(["ext-one", "nope"], usable, rejected)

    msg = str(exc.value)
    assert "ext-one (proposed: awaiting approval" in msg
    assert "nope (no skill by that name was found)" in msg
    assert "--approve-skill" in msg
    _raise_if_required_skills_unusable([], usable, rejected)


@pytest.fixture()
def quest_env(env, tmp_path, monkeypatch):
    from core.config import (
        Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig,
        ProviderConfig,
    )

    def make(required: list[str]) -> Config:
        return Config(
            topic="skills required probe", title="skills-required",
            provider=ProviderConfig(name="openai"),
            engine=EngineConfig(
                max_iterations=1, review_loop=False, auto_accept_on_pass=True,
                skills_required=required,
            ),
            execution=ExecutionConfig(sandbox="venv", timeout_s=120),
            knowledge=KnowledgeConfig(enabled=False),
            output=OutputConfig(output_dir=tmp_path / "outputs"),
        )

    return make


@pytest.mark.asyncio
async def test_an_unusable_required_skill_stops_the_quest_before_any_llm_call(
    env, quest_env, monkeypatch,
) -> None:
    from core.engine import Engine
    from tests.test_engine_smoke import _fake_response_for

    _skill(env["ext"], "ext-one")
    calls: list[str] = []

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        calls.append(messages[-1]["content"][:40])
        return _fake_response_for(messages[-1]["content"])

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)

    with pytest.raises(RuntimeError, match="skills_required.*ext-one.*awaiting approval"):
        await Engine(quest_env(["ext-one"])).run()
    assert calls == [], "the quest spent LLM calls before checking the skill"


@pytest.mark.asyncio
async def test_a_required_skill_is_used_whatever_selection_says(
    env, quest_env, monkeypatch,
) -> None:
    """The fake selector answers nothing; the required skill is in the pick
    anyway, and an approved external skill is usable."""
    from core.engine import Engine
    from tests.test_engine_smoke import _fake_response_for

    _skill(env["ext"], "ext-one", script=True)
    (skill,) = discover()
    approval.approve(skill.ledger_name, skill.content_hash(), approved_by="me", note="t")

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        return _fake_response_for(messages[-1]["content"])

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)

    artifacts = await Engine(quest_env(["ext-one"])).run()

    state = artifacts.raw_state
    assert state["selected_skills"] == ["ext-one"]
    sel = state["skill_selection"]
    assert sel["forced"] == ["ext-one"]
    assert sel["reasons"]["ext-one"] == "required by engine.skills_required"
    assert sel["uses"]["ext-one"] == "experiment"
    assert not (skill.path / "provenance.json").exists(), "an external skill was written to"
