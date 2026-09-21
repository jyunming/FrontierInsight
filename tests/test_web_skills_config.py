"""A quest's own skill folders in the web dashboard (``?config=<quest.yaml>``).

A quest can name folders of skills other agents installed (``engine.skills_dirs``). The
command line reaches them with ``--config quest.yaml`` (``tests/test_skills_cli_config.py``);
until the dashboard took the same option, a skill living only in such a folder could not be
listed, reviewed or approved there, and the quest then stopped on it as "not approved".

The gate itself is not re-tested here (``tests/test_web_skills.py`` does that): these show
that the option changes where skills are looked for and nothing else.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from core.skills import registry
from web.server import make_app

NAME = "deepscientist-experiment"


def _skill(folder: Path, name: str, *, body: str = "Do the thing.\n") -> Path:
    d = folder / name
    (d / "scripts").mkdir(parents=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: an external demo skill\n---\n{body}",
        encoding="utf-8",
    )
    (d / "scripts" / "run.sh").write_text("echo hi\n", encoding="utf-8")
    return d


def _quest_yaml(path: Path, dirs: list[Path]) -> Path:
    path.write_text(
        yaml.safe_dump({
            "topic": "skills folder probe",
            "provider": {"name": "openai"},
            "engine": {"skills_dirs": [str(d) for d in dirs], "skills_scan_known_dirs": False},
        }),
        encoding="utf-8",
    )
    return path


@pytest.fixture()
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """One skill in a folder only a quest's YAML names, on a machine whose usual places hold nothing."""
    own = tmp_path / "fi_skills"
    own.mkdir()
    monkeypatch.setenv("FI_SKILLS_DIR", str(own))
    monkeypatch.setenv("FI_SKILLS_APPROVALS", str(tmp_path / "approvals.json"))
    # The suite sets this to "" for every test, and set-even-empty it replaces whatever a config
    # names: it would hide the very folders under test.
    monkeypatch.delenv("FI_EXTERNAL_SKILLS_DIRS", raising=False)
    monkeypatch.setattr(registry, "_common_skill_dirs", lambda: [])
    folder = tmp_path / "my-agent-skills"
    _skill(folder, NAME)
    config = _quest_yaml(tmp_path / "quest.yaml", [folder])
    client = TestClient(make_app(tmp_path / "outputs"))
    return client, config, folder, tmp_path / "approvals.json"


def _names(response) -> set[str]:
    return {s["name"] for s in response.json()["skills"]}


def test_the_skill_is_not_listed_without_the_config_and_is_with_it(world) -> None:
    client, config, folder, _ledger = world
    assert NAME not in _names(client.get("/api/skills"))
    got = client.get("/api/skills", params={"config": str(config)})
    assert got.status_code == 200
    rows = {s["name"]: s for s in got.json()["skills"]}
    assert rows[NAME]["source"] == "external"
    assert Path(rows[NAME]["path"]).parent == folder


def test_a_config_that_cannot_be_read_is_a_400_with_the_reason(world, tmp_path: Path) -> None:
    client, _config, _folder, _ledger = world
    missing = client.get("/api/skills", params={"config": str(tmp_path / "nope.yaml")})
    assert missing.status_code == 400
    assert "cannot be read" in missing.json()["detail"]
    bad = tmp_path / "bad.yaml"
    bad.write_text("topic: [unclosed\n", encoding="utf-8")
    broken = client.get("/api/skills", params={"config": str(bad)})
    assert broken.status_code == 400
    assert "not valid YAML" in broken.json()["detail"]


def test_an_empty_config_is_the_same_as_none(world) -> None:
    client, _config, _folder, _ledger = world
    assert _names(client.get("/api/skills", params={"config": ""})) == _names(client.get("/api/skills"))
    assert _names(client.get("/api/skills", params={"config": "   "})) == _names(client.get("/api/skills"))


def test_the_review_finds_the_skill_only_with_the_config(world) -> None:
    client, config, _folder, _ledger = world
    assert client.get(f"/api/skills/{NAME}/scan").status_code == 404
    got = client.get(f"/api/skills/{NAME}/scan", params={"config": str(config)})
    assert got.status_code == 200 and got.json()["name"] == NAME


def test_approval_needs_the_config_a_named_person_and_binds_to_the_content(world) -> None:
    client, config, _folder, ledger = world
    url = f"/api/skills/{NAME}/approve"
    assert client.post(url, json={"approved_by": "Ada"}).status_code == 404
    # the gate is unchanged: no anonymous approver, config or not
    assert client.post(url, params={"config": str(config)}, json={}).status_code == 400
    done = client.post(url, params={"config": str(config)}, json={"approved_by": "Ada"})
    assert done.status_code == 200 and done.json()["approved"] is True
    recorded = json.loads(ledger.read_text(encoding="utf-8"))
    assert "Ada" in json.dumps(recorded) and "external skill" in json.dumps(recorded)
    listed = client.get("/api/skills", params={"config": str(config)}).json()["skills"]
    assert {s["name"]: s for s in listed}[NAME]["status"] == "trusted"


def test_revoking_reaches_the_skill_through_the_config(world) -> None:
    client, config, _folder, _ledger = world
    client.post(f"/api/skills/{NAME}/approve", params={"config": str(config)}, json={"approved_by": "Ada"})
    gone = client.post(f"/api/skills/{NAME}/revoke", params={"config": str(config)})
    assert gone.json() == {"revoked": True, "name": NAME}
    listed = client.get("/api/skills", params={"config": str(config)}).json()["skills"]
    assert {s["name"]: s for s in listed}[NAME]["status"] != "trusted"


def test_the_page_carries_the_config_field() -> None:
    page = (Path(__file__).resolve().parent.parent / "web" / "static" / "skills.html").read_text(encoding="utf-8")
    assert 'id="config"' in page and "withConfig(" in page
    for route in ("/scan`", "/approve`", "/revoke`"):
        assert f"withConfig(`/api/skills/${{encodeURIComponent(name)}}{route}" in page, route
