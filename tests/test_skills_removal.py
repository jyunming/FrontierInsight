"""Removing a skill from FI completely (``--remove-skill``), as opposed to revoking approval."""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import launch
from core.skills import approval, discover
from core.skills.removal import hidden_external, remove_skill, restore_external
from web.server import make_app


def _skill(parent: Path, name: str) -> Path:
    d = parent / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: demo skill\n---\nDo the thing.\n", encoding="utf-8",
    )
    (d / "selftest.py").write_text("import sys; sys.exit(0)\n", encoding="utf-8")
    return d


@pytest.fixture
def env(tmp_path: Path, monkeypatch):
    own = tmp_path / "own"
    ext = tmp_path / "ext"
    own.mkdir()
    ext.mkdir()
    monkeypatch.setenv("FI_SKILLS_DIR", str(own))
    monkeypatch.setenv("FI_SKILLS_APPROVALS", str(tmp_path / "approvals.json"))
    monkeypatch.setenv("FI_SKILLS_SELFTEST_CACHE", str(tmp_path / "cache.json"))
    monkeypatch.setenv("FI_EXTERNAL_SKILLS_DIRS", str(ext))
    return {"own": own, "ext": ext, "tmp": tmp_path}


def _get(name: str):
    return next((s for s in discover() if s.name == name), None)


def test_an_own_skill_is_deleted_with_its_approval(env) -> None:
    folder = _skill(env["own"], "mine")
    skill = _get("mine")
    approval.approve(skill.ledger_name, skill.content_hash(), approved_by="tester")
    assert approval.ledger_path().exists() and "mine" in approval.ledger_path().read_text(encoding="utf-8")

    result = remove_skill(skill)

    assert result.ok and result.deleted == folder
    assert not folder.exists()
    assert _get("mine") is None
    assert "mine" not in approval.ledger_path().read_text(encoding="utf-8")
    assert "approval withdrawn" in result.message


def test_a_symlink_is_removed_without_touching_its_target(env) -> None:
    target = _skill(env["tmp"] / "elsewhere", "real")
    link = env["own"] / "linked"
    try:
        os.symlink(target, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    skill = _get("linked")
    if skill is None:
        pytest.skip("a linked skill is not discovered here")
    remove_skill(skill)
    assert (target / "SKILL.md").exists()


def test_a_name_that_climbs_out_of_the_folder_is_refused(env) -> None:
    from dataclasses import replace

    _skill(env["own"], "mine")
    skill = _get("mine")
    for bad in ("..", "a/b", "a\\b", ""):
        try:
            evil = replace(skill, name=bad)
        except TypeError:
            pytest.skip("Skill is not a dataclass")
        assert not remove_skill(evil).ok
    assert (env["own"] / "mine" / "SKILL.md").exists()


def test_an_external_skill_is_hidden_and_its_files_stay(env) -> None:
    folder = _skill(env["ext"], "theirs")
    skill = _get("theirs")
    assert skill.external
    approval.approve(skill.ledger_name, skill.content_hash(), approved_by="tester")

    result = remove_skill(skill)

    assert result.ok and result.hidden and result.deleted is None
    assert (folder / "SKILL.md").exists()
    assert _get("theirs") is None
    assert hidden_external() == {"theirs"}
    assert "external:theirs" not in approval.ledger_path().read_text(encoding="utf-8")
    assert "--restore-skill theirs" in result.message


def test_restore_brings_a_hidden_external_skill_back_as_proposed(env) -> None:
    _skill(env["ext"], "theirs")
    remove_skill(_get("theirs"))
    assert restore_external("theirs") is True
    back = _get("theirs")
    assert back is not None and back.external
    assert restore_external("theirs") is False


def test_cli_round_trip(env, capsys) -> None:
    _skill(env["own"], "mine")
    _skill(env["ext"], "theirs")

    assert launch._remove_skill("mine") == 0
    assert "Removed skill mine" in capsys.readouterr().out
    assert launch._remove_skill("theirs") == 0
    assert launch._remove_skill("nope") == 1
    assert "No skill named" in capsys.readouterr().out
    assert launch._restore_skill("theirs") == 0
    assert launch._restore_skill("theirs") == 1


def test_a_package_skill_cannot_be_deleted(env, monkeypatch) -> None:
    from dataclasses import replace

    _skill(env["own"], "mine")
    skill = replace(_get("mine"), source="entry_point")
    result = remove_skill(skill)
    assert not result.ok and "pip uninstall" in result.message
    assert (env["own"] / "mine").exists()


def test_web_remove_and_restore(env) -> None:
    _skill(env["own"], "mine")
    _skill(env["ext"], "theirs")
    client = TestClient(make_app(env["tmp"] / "outputs"))

    r = client.post("/api/skills/mine/remove")
    assert r.status_code == 200 and r.json()["removed"] is True
    assert not (env["own"] / "mine").exists()
    assert client.post("/api/skills/mine/remove").status_code == 404

    assert client.post("/api/skills/theirs/remove").json()["hidden"] is True
    names = {s["name"] for s in client.get("/api/skills").json()["skills"]}
    assert "theirs" not in names
    assert client.post("/api/skills/theirs/restore").json()["restored"] is True
    names = {s["name"] for s in client.get("/api/skills").json()["skills"]}
    assert "theirs" in names


def test_web_remove_refuses_a_cross_site_request(env) -> None:
    _skill(env["own"], "mine")
    client = TestClient(make_app(env["tmp"] / "outputs"))
    r = client.post("/api/skills/mine/remove", headers={"Origin": "http://evil.example"})
    assert r.status_code == 403
    assert (env["own"] / "mine" / "SKILL.md").exists()
    ok = client.post("/api/skills/mine/remove", headers={"Origin": "http://testserver"})
    assert ok.status_code == 200


def test_a_failed_delete_is_reported_and_leaves_the_skill_unapproved(env, monkeypatch) -> None:
    _skill(env["own"], "mine")
    skill = _get("mine")
    approval.approve(skill.ledger_name, skill.content_hash(), approved_by="tester")

    def boom(_p):
        raise PermissionError("in use")

    monkeypatch.setattr("core.skills.removal._delete_tree", boom)
    result = remove_skill(skill)
    assert not result.ok and "in use" in result.message
    assert "mine" not in approval.ledger_path().read_text(encoding="utf-8")


def test_a_skill_outside_the_own_folder_is_never_deleted(env) -> None:
    from dataclasses import replace

    outside = _skill(env["tmp"] / "elsewhere", "mine")
    _skill(env["own"], "mine")
    skill = replace(_get("mine"), path=outside)
    assert not remove_skill(skill).ok
    assert (outside / "SKILL.md").exists()
