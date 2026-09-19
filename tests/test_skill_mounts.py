"""An approved external skill is mounted read-only into the Docker sandbox.

Nothing here needs a Docker daemon. ``DockerExecutor`` talks to docker-py, not a
``docker`` command line, so the mount is what reaches ``client.containers.create
(volumes=...)``: ``{host: {"bind": container, "mode": "ro"}}`` is docker-py's
spelling of ``-v host:container:ro``, and one test renders it into the exact bind
string docker-py sends the daemon. The client is the same MagicMock the rest of
``test_docker_executor.py`` uses.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig,
    ProviderConfig,
)
from core.engine import Engine
from core.execution import DockerExecutor
from core.skills import approval, discover, loadable_skills
from core.skills.base import Skill, SkillState, Status
from core.skills.mounts import (
    CONTAINER_ROOT, SkillMount, bind_host_path, check_folder, container_dir_name,
    plan_mounts,
)


def _skill(root: Path, name: str, *, script: bool = True, text: str = "Do the thing.\n") -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: an external demo skill\n---\n{text}",
        encoding="utf-8",
    )
    if script:
        (d / "scripts").mkdir()
        (d / "scripts" / "run.py").write_text("print('hi')\n", encoding="utf-8")
        (d / "references").mkdir()
        (d / "references" / "notes.md").write_text("notes\n", encoding="utf-8")
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
    return {"own": own, "ext": ext}


def _approve(name: str) -> None:
    skill = next(s for s in discover() if s.name == name)
    approval.approve(skill.ledger_name, skill.content_hash(), approved_by="me", note="t")


def _usable(*names: str) -> list[SkillState]:
    usable, _ = loadable_skills(list(names), use_cache=False)
    return usable


# --- names and bind strings --------------------------------------------------


def test_a_plain_name_is_kept_and_any_other_is_made_safe_and_distinct() -> None:
    assert container_dir_name("my-skill_v1.2") == "my-skill_v1.2"
    odd = container_dir_name("my skill/../x")
    assert odd.startswith("my_skill_") and odd.endswith("_x" + odd[-9:])
    assert not {"/", " ", "\\"} & set(odd) and ".." not in odd
    assert container_dir_name("my skill/../x") == odd, "stable from run to run"
    assert container_dir_name("a..b") != "a..b" and ".." not in container_dir_name("a..b")
    assert container_dir_name("my skill") != container_dir_name("my_skill")
    assert container_dir_name("..").startswith("skill-")
    assert container_dir_name("") .startswith("skill-")
    assert container_dir_name(".hidden") != ".hidden"


def test_a_posix_host_path_is_the_bind_source_as_it_is() -> None:
    assert bind_host_path(PurePosixPath("/home/me/.claude/skills/x")) == "/home/me/.claude/skills/x"


def test_a_windows_drive_path_keeps_its_drive_colon_and_only_that() -> None:
    win = PureWindowsPath("C:\\Users\\me\\.claude\\skills\\x")
    assert bind_host_path(win) == "C:\\Users\\me\\.claude\\skills\\x"
    # what resolve() can return for a long path
    assert bind_host_path("\\\\?\\C:\\Users\\me\\skills\\x") == "C:\\Users\\me\\skills\\x"


@pytest.mark.parametrize("bad", [
    "\\\\server\\share\\skills\\x",            # a network path
    "\\\\?\\UNC\\server\\share\\skills\\x",    # the same, extended form
    "/home/me/skills/x:/etc:rw",               # would add a second bind
    "C:\\skills\\x:/etc:rw",                   # a drive letter does not excuse the rest
    "/home/me/skills/x\nrw",                   # a control character
])
def test_a_path_a_bind_string_cannot_carry_is_refused(bad: str) -> None:
    with pytest.raises(ValueError):
        bind_host_path(bad)


# --- what is mounted, what is refused ---------------------------------------


def test_an_approved_external_skill_is_mounted_read_only_at_a_fixed_path(env) -> None:
    folder = _skill(env["ext"], "ext-one")
    _approve("ext-one")

    plan = plan_mounts(_usable("ext-one"))

    (mount,) = plan.mounts.values()
    assert plan.refused == {}
    assert mount.container == f"{CONTAINER_ROOT}/ext-one" == "/fi-skills/ext-one"
    assert mount.host == folder.resolve()
    assert mount.volume == {"bind": "/fi-skills/ext-one", "mode": "ro"}
    assert mount.bind_arg == f"{folder.resolve()}:/fi-skills/ext-one:ro"


def test_an_unapproved_skill_is_refused_even_when_handed_in(env) -> None:
    _skill(env["ext"], "ext-one")
    rejected = loadable_skills(["ext-one"], use_cache=False)[1]

    plan = plan_mounts(rejected)

    assert plan.mounts == {}
    assert "not approved" in plan.refused["ext-one"]


def test_a_skill_whose_content_changed_since_approval_is_not_mounted(env) -> None:
    folder = _skill(env["ext"], "ext-one")
    _approve("ext-one")
    (folder / "scripts" / "run.py").write_text("print('changed')\n", encoding="utf-8")

    usable, rejected = loadable_skills(["ext-one"], use_cache=False)

    assert usable == []
    assert plan_mounts([*usable, *rejected]).mounts == {}


def test_an_fi_skill_is_not_this_features_business(env) -> None:
    own = _skill(env["own"], "fi-own")
    st = SkillState(skill=Skill(name="fi-own", path=own), status=Status.TRUSTED)

    plan = plan_mounts([st])

    assert plan.mounts == {} and plan.refused == {}


def test_a_skill_folder_that_is_a_symlink_is_refused(env, monkeypatch) -> None:
    """The branch runs on every OS through is_symlink; a real link is used
    where the OS lets this user make one."""
    real = _skill(env["ext"].parent / "elsewhere", "target")
    link = env["ext"] / "linked"
    try:
        os.symlink(real, link, target_is_directory=True)
        made = True
    except OSError:
        made = False
    if not made:
        _skill(env["ext"], "linked")
        monkeypatch.setattr(Path, "is_symlink", lambda self: self.name == "linked")
    _approve("linked")

    plan = plan_mounts(_usable("linked"))

    assert plan.mounts == {}
    assert "symbolic link" in plan.refused["linked"]


@pytest.mark.skipif(sys.platform != "win32", reason="a junction is a Windows thing")
def test_a_windows_junction_is_refused_though_it_is_not_a_symlink(env) -> None:
    """is_symlink() is False for a junction; resolving it is what gives it away."""
    real = _skill(env["ext"].parent / "elsewhere", "target")
    junction = env["ext"] / "junc"
    made = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(real)],
        capture_output=True, text=True,
    )
    assert made.returncode == 0, made.stderr
    assert not junction.is_symlink()
    _approve("junc")

    plan = plan_mounts(_usable("junc"))

    assert plan.mounts == {}
    assert "resolves outside its skills folder" in plan.refused["junc"]


def test_a_folder_that_resolves_outside_its_skills_folder_is_refused(env, monkeypatch) -> None:
    """The escape branch on every OS, with resolution redirected by hand."""
    folder = _skill(env["ext"], "ext-one")
    _approve("ext-one")
    elsewhere = env["ext"].parent / "elsewhere" / "ext-one"
    elsewhere.mkdir(parents=True)
    (elsewhere / "SKILL.md").write_text("x", encoding="utf-8")
    real_resolve = Path.resolve

    def resolve(self, strict=False):  # noqa: ANN001
        if self == folder:
            return real_resolve(elsewhere, strict=strict)
        return real_resolve(self, strict=strict)

    monkeypatch.setattr(Path, "resolve", resolve)

    real, why = check_folder(folder, "ext-one")

    assert real is None and "resolves outside its skills folder" in why


def test_the_home_directory_its_parents_and_a_config_folder_are_never_mounted(tmp_path) -> None:
    home = tmp_path / "home"
    config = home / ".claude"
    ok = home / ".claude" / "skills" / "fine"
    for d in (tmp_path, home, config, ok):
        d.mkdir(parents=True, exist_ok=True)
        (d / "SKILL.md").write_text("x", encoding="utf-8")

    for folder in (home, tmp_path):
        real, why = check_folder(folder, folder.name, home=home)
        assert real is None and "home directory or one of its parents" in why
    real, why = check_folder(config, ".claude", home=home)
    assert real is None and "tool's configuration folder" in why
    real, why = check_folder(ok, "fine", home=home)
    assert real == ok.resolve() and why == "", "a skill below a config folder is the normal case"


def test_without_a_known_home_directory_nothing_is_mounted(env, monkeypatch) -> None:
    """The home check cannot be made, so it is not skipped: the folder is refused."""
    _skill(env["ext"], "ext-one")
    _approve("ext-one")
    usable = _usable("ext-one")

    def no_home(cls):  # noqa: ANN001
        raise RuntimeError("Could not determine home directory.")

    monkeypatch.setattr(Path, "home", classmethod(no_home))

    plan = plan_mounts(usable)

    assert plan.mounts == {} and "home directory cannot be determined" in plan.refused["ext-one"]


def test_a_folder_without_skill_md_or_with_a_wrong_name_is_refused(tmp_path) -> None:
    empty = tmp_path / "skills" / "empty"
    empty.mkdir(parents=True)
    assert "no regular SKILL.md" in check_folder(empty, "empty", home=tmp_path / "h")[1]
    assert "not the skill's name" in check_folder(empty, "other", home=tmp_path / "h")[1]
    assert "cannot be resolved" in check_folder(tmp_path / "skills" / "gone", "gone")[1]


def test_a_folder_that_was_swapped_for_a_link_is_no_longer_safe(env, monkeypatch) -> None:
    folder = _skill(env["ext"], "ext-one")
    _approve("ext-one")
    (mount,) = plan_mounts(_usable("ext-one")).mounts.values()
    assert mount.still_safe()

    monkeypatch.setattr(Path, "is_symlink", lambda self: self == folder)

    assert not mount.still_safe()


# --- the paths a skill's own text carries -----------------------------------


def test_the_hosts_folder_in_a_skills_text_becomes_the_containers(tmp_path) -> None:
    host = tmp_path / "skills" / "foo"
    mount = SkillMount(name="foo", host=host, container="/fi-skills/foo", declared=host)
    text = (
        f"Run `python {host}/scripts/run.py --in {host.as_posix()}/data/a.csv`.\n"
        f"Not this one: {host}-bar/x and {host}.py but this: {host}."
    )

    out = mount.translate(text)

    assert "python /fi-skills/foo/scripts/run.py --in /fi-skills/foo/data/a.csv" in out
    assert f"{host}-bar/x" in out and f"{host}.py" in out, "a look-alike is left alone"
    assert out.endswith("this: /fi-skills/foo.")


def test_a_windows_spelling_is_translated_with_its_separators() -> None:
    host = PureWindowsPath("C:\\Users\\me\\skills\\foo")
    mount = SkillMount(name="foo", host=host, container="/fi-skills/foo", declared=host)  # type: ignore[arg-type]

    out = mount.translate("run C:\\Users\\me\\skills\\foo\\scripts\\run.py now")

    assert out == "run /fi-skills/foo/scripts/run.py now"


# --- the docker executor ----------------------------------------------------


def _container(out: bytes = b'RESULT_JSON: {"ok": 1}') -> MagicMock:
    c = MagicMock()
    c.wait.return_value = {"StatusCode": 0}
    c.logs.side_effect = lambda stdout=False, stderr=False, **_: (
        out if stdout and not stderr else b""
    )
    return c


def _client() -> MagicMock:
    client = MagicMock()
    client.containers.create.side_effect = lambda *a, **k: _container()
    return client


def _volumes(client: MagicMock, call: int = -1) -> dict[str, Any]:
    return client.containers.create.call_args_list[call].kwargs["volumes"]


def test_the_container_gets_the_quest_read_write_and_each_skill_read_only(env, tmp_path) -> None:
    folder = _skill(env["ext"], "ext-one")
    _approve("ext-one")
    exe = DockerExecutor()
    exe.set_skill_mounts(plan_mounts(_usable("ext-one")).mounts.values())
    client = _client()

    exe._run_sync(client, ["python", "-V"], tmp_path, 30, {})

    volumes = _volumes(client)
    assert volumes == {
        str(tmp_path.resolve()): {"bind": "/work", "mode": "rw"},
        str(folder.resolve()): {"bind": "/fi-skills/ext-one", "mode": "ro"},
    }


def test_docker_py_renders_the_mount_as_host_container_ro(env, tmp_path) -> None:
    """The exact string the daemon is sent: ``-v host:container:ro``."""
    docker_utils = pytest.importorskip("docker.utils")
    folder = _skill(env["ext"], "ext-one")
    _approve("ext-one")
    exe = DockerExecutor()
    exe.set_skill_mounts(plan_mounts(_usable("ext-one")).mounts.values())
    client = _client()
    exe._run_sync(client, ["python", "-V"], tmp_path, 30, {})

    binds = docker_utils.convert_volume_binds(_volumes(client))

    assert f"{folder.resolve()}:/fi-skills/ext-one:ro" in binds
    assert f"{tmp_path.resolve()}:/work:rw" in binds
    assert len(binds) == 2


def test_with_no_skills_the_container_gets_the_quest_alone(tmp_path) -> None:
    exe = DockerExecutor()
    client = _client()

    exe._run_sync(client, ["python", "-V"], tmp_path, 30, {})
    exe.set_skill_mounts([])
    exe._run_sync(client, ["python", "-V"], tmp_path, 30, {})

    assert list(_volumes(client, 0)) == [str(tmp_path.resolve())]
    assert list(_volumes(client, 1)) == [str(tmp_path.resolve())]


def test_a_mount_that_stopped_being_safe_after_planning_is_left_out(env, tmp_path, monkeypatch) -> None:
    folder = _skill(env["ext"], "ext-one")
    _approve("ext-one")
    exe = DockerExecutor()
    exe.set_skill_mounts(plan_mounts(_usable("ext-one")).mounts.values())
    client = _client()
    monkeypatch.setattr(Path, "is_symlink", lambda self: self == folder)

    exe._run_sync(client, ["python", "-V"], tmp_path, 30, {})

    assert list(_volumes(client)) == [str(tmp_path.resolve())]


# --- the engine: which skills, and the paths the prompt carries -------------


def _engine(tmp_path: Path, sandbox: str, *, replicates: int = 1, pilot: bool = False) -> Engine:
    cfg = Config(
        topic="skill mounts", title="skill-mounts",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(
            max_iterations=1, review_loop=False, execute_replicates=replicates,
            pilot_run=pilot,
        ),
        execution=ExecutionConfig(sandbox=sandbox, timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "out"),
    )
    eng = Engine(cfg)
    for sub in ("code", "figures"):
        (eng.quest_root / sub).mkdir(parents=True, exist_ok=True)
    (eng.quest_root / "code" / "experiment.py").write_text("print('hi')", encoding="utf-8")
    return eng


def _state(*names: str) -> dict[str, Any]:
    return {
        "deps": [],
        "selected_skills": list(names),
        "skill_selection": {
            "uses": {n: "experiment" for n in names},
            "reasons": {n: "needed" for n in names},
        },
    }


def test_only_the_selected_approved_external_skills_are_mounted(env, tmp_path) -> None:
    for name in ("picked", "approved-not-picked", "picked-not-approved"):
        _skill(env["ext"], name)
    _approve("picked")
    _approve("approved-not-picked")
    _skill(env["own"], "fi-own", script=False)
    eng = _engine(tmp_path, "docker")

    eng._mount_selected_skills(_state("picked", "picked-not-approved", "fi-own"))

    assert [m.name for m in eng.executor.skill_mounts] == ["picked"]
    recorded = json.loads((eng.fi_dir / "skill_mounts.json").read_text(encoding="utf-8"))
    assert recorded == {"skills": ["picked"]}


def test_a_skill_selected_for_writing_is_not_mounted(env, tmp_path) -> None:
    _skill(env["ext"], "writer")
    _approve("writer")
    eng = _engine(tmp_path, "docker")
    state = _state("writer")
    state["skill_selection"]["uses"]["writer"] = "writing"

    eng._mount_selected_skills(state)

    assert eng.executor.skill_mounts == ()


def test_a_refused_skill_is_logged_with_its_reason_and_not_mounted(env, tmp_path, monkeypatch) -> None:
    _skill(env["ext"], "ext-one")
    _approve("ext-one")
    monkeypatch.setattr(Path, "is_symlink", lambda self: self.name == "ext-one")
    eng = _engine(tmp_path, "docker")

    eng._mount_selected_skills(_state("ext-one"))

    assert eng.executor.skill_mounts == ()
    log = (eng.fi_dir / "run.log").read_text(encoding="utf-8")
    assert "ext-one is NOT mounted into the Docker sandbox: the skill folder is a symbolic link" in log


def test_other_sandboxes_mount_nothing_and_see_host_paths(env, tmp_path) -> None:
    folder = _skill(env["ext"], "ext-one")
    _approve("ext-one")
    eng = _engine(tmp_path, "venv")

    eng._mount_selected_skills(_state("ext-one"))
    block = eng._skills_block(_state("ext-one"))

    assert not hasattr(eng.executor, "skill_mounts")
    assert f"at paths relative to `{folder}`" in block
    assert "/fi-skills" not in block
    assert not (eng.fi_dir / "skill_mounts.json").exists()


def test_under_docker_the_prompt_names_the_container_path(env, tmp_path) -> None:
    folder = _skill(
        env["ext"], "ext-one",
        text=f"Run `python {env['ext'] / 'ext-one'}/scripts/run.py`.\n",
    )
    _approve("ext-one")
    eng = _engine(tmp_path, "docker")

    block = eng._skills_block(_state("ext-one"))

    assert "at paths relative to `/fi-skills/ext-one`" in block
    assert "this skill's folder is `/fi-skills/ext-one`" in block and "read-only" in block
    assert "python /fi-skills/ext-one/scripts/run.py" in block, "the skill's own text is translated"
    assert "`scripts/run.py` (executable)" in block and "`references/notes.md` (reference)" in block
    assert str(folder) not in block and str(folder.resolve()) not in block


def test_under_docker_a_refused_skill_is_not_described_as_reachable(env, tmp_path, monkeypatch) -> None:
    folder = _skill(env["ext"], "ext-one")
    _approve("ext-one")
    monkeypatch.setattr(Path, "is_symlink", lambda self: self.name == "ext-one")
    eng = _engine(tmp_path, "docker")

    block = eng._skills_block(_state("ext-one"))

    assert "NOT available in the Docker sandbox" in block
    assert "symbolic link" in block
    assert "not reachable from the sandbox" in block
    assert "at paths relative to" not in block
    assert str(folder) not in block and "/fi-skills" not in block


@pytest.mark.asyncio
async def test_every_container_of_the_experiment_gets_the_skill_read_only(env, tmp_path) -> None:
    """_node_execute mounts the plan before its first container, so the pilot,
    the run and the replicates all have it."""
    folder = _skill(env["ext"], "ext-one")
    _approve("ext-one")
    _skill(env["ext"], "approved-not-picked")
    _approve("approved-not-picked")
    eng = _engine(tmp_path, "docker", replicates=2, pilot=True)
    client = _client()
    eng.executor._client = client

    await eng._node_execute(_state("ext-one"))

    creates = client.containers.create.call_args_list
    assert len(creates) >= 3, "pilot + run + replicate at least"
    for c in creates:
        volumes = c.kwargs["volumes"]
        assert volumes[str(folder.resolve())] == {"bind": "/fi-skills/ext-one", "mode": "ro"}
        assert sorted(v["bind"] for v in volumes.values()) == ["/fi-skills/ext-one", "/work"]
        assert all(v["mode"] == "ro" for v in volumes.values() if v["bind"] != "/work")


@pytest.mark.asyncio
async def test_a_watch_mounts_the_skills_the_quest_mounted(env, tmp_path) -> None:
    """poll_job has no quest state: it reads what _node_execute recorded."""
    folder = _skill(env["ext"], "ext-one")
    _approve("ext-one")
    first = _engine(tmp_path, "docker")
    first.executor._client = _client()
    await first._node_execute(_state("ext-one"))

    watcher = Engine(first.config, resume_quest_id=first.quest_id)
    client = _client()
    watcher.executor._client = client
    watcher.executor.setup = AsyncMock()  # type: ignore[method-assign]
    await watcher.poll_job()

    volumes = _volumes(client)
    assert volumes[str(folder.resolve())] == {"bind": "/fi-skills/ext-one", "mode": "ro"}
