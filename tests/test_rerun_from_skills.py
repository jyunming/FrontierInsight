"""``--resume <id> --from skills``: the skills are picked again from the current config, the literature and the frozen
plan are left alone, and a skill that is a Node.js tool is no longer asked of pip (core/rerun_from.py, Engine._repick_skills)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from core import rerun_from
from core.config import Config
from core.engine import Engine
from core.skills import approval, discover
from tests.test_engine_smoke import _FAKE_EXPERIMENT_CODE, _classify, _fake_response_for
from tests.test_rerun_from import _Graph, _cfg

SKILL = "zzlieflat-charts"


def test_the_skills_step_is_offered_before_the_code_and_has_its_words() -> None:
    assert rerun_from.resolve("skills") == "skills" and rerun_from.resolve("select_skills") == "skills"
    assert rerun_from.resolve("Skill") == "skills"
    assert rerun_from.choices() == "ideas, literature, plan, design, skills, code, run, figures, analysis, crosscheck, evidence, writing, claims, review"
    assert "again from the quest's current config" in rerun_from.REDOES["skills"]
    assert rerun_from.picks_skills("skills") and not rerun_from.picks_skills("code")


@pytest.mark.asyncio
async def test_the_skills_step_forks_where_the_code_step_does_and_is_reached_with_it() -> None:
    graph = _Graph([("select_skills",), ("plan",), ("design",), ("implement_outline",), ("implement",),
                    ("execute",), ("analyze",)])
    assert await rerun_from.reached(graph, {}) == ["plan", "design", "skills", "code", "run", "analysis"]
    assert (await rerun_from.checkpoint_before(graph, {}, "skills"))["configurable"]["checkpoint_id"] == "3"
    # No simulation: the data is collected instead of the code written; the skills were picked before either.
    graph = _Graph([("design",), ("auto_collect_data",), ("wait_for_data",), ("data_load",), ("analyze",)])
    assert await rerun_from.reached(graph, {}) == ["design", "skills", "run", "analysis"]
    assert (await rerun_from.checkpoint_before(graph, {}, "skills"))["configurable"]["checkpoint_id"] == "1"
    # A quest that stopped before the code has no step to pick the skills again from.
    assert "skills" not in await rerun_from.reached(_Graph([("select_skills",), ("plan",), ("design",)]), {})
    assert await rerun_from.checkpoint_before(_Graph([("design",)]), {}, "skills") is None


def test_the_skills_backup_takes_the_code_and_what_follows_and_leaves_the_plan(tmp_path: Path) -> None:
    for rel in ("code/experiment.py", "figures/a.png", "paper/paper.md", "results.json", "plan.md",
                "needs/FROZEN_PROTOCOL.json"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("x", encoding="utf-8")
    where, moved = rerun_from.back_up(tmp_path, "skills")
    assert where is not None and {"code", "figures", "paper", "results.json"} <= set(moved)
    assert (tmp_path / "plan.md").is_file() and (tmp_path / "needs" / "FROZEN_PROTOCOL.json").is_file()


def _tool_skill(root: Path, name: str, *, pip_requires: list[str] | None = None) -> Path:
    """A skill that is a Node.js tool: nothing of it is a Python package."""
    d = root / name
    (d / "scripts").mkdir(parents=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: draws charts with a Node.js tool\n---\nRun `node scripts/chart.js`.\n",
        encoding="utf-8",
    )
    (d / "scripts" / "chart.js").write_text("console.log('chart')\n", encoding="utf-8")
    prov: dict[str, Any] = {"kind": "tool"}
    if pip_requires is not None:
        prov["pip_requires"] = pip_requires
    (d / "provenance.json").write_text(json.dumps(prov), encoding="utf-8")
    return d


@pytest.fixture()
def node_skill(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    own = tmp_path / "fi_skills"
    ext = tmp_path / "other_agent" / "skills"
    own.mkdir()
    ext.mkdir(parents=True)
    monkeypatch.setenv("FI_SKILLS_DIR", str(own))
    monkeypatch.setenv("FI_EXTERNAL_SKILLS_DIRS", str(ext))
    monkeypatch.setenv("FI_SKILLS_APPROVALS", str(tmp_path / "approvals.json"))
    _tool_skill(ext, SKILL)
    for sk in discover():
        approval.approve(sk.ledger_name, sk.content_hash(), approved_by="test", note="t")
    return ext


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def _state(engine: Engine) -> dict[str, Any]:
    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    async with aiosqlite.connect(f"{(engine.fi_dir / 'state.sqlite').resolve().as_uri()}?mode=ro", uri=True) as conn:
        saver = AsyncSqliteSaver(conn)
        saver.is_setup = True
        graph = engine._build_graph().compile(checkpointer=saver)
        snap = await graph.aget_state({"configurable": {"thread_id": engine.quest_id}})
    return dict(snap.values or {})


def _fake(monkeypatch: pytest.MonkeyPatch, calls: list[str], prompts: list[str], deps: list[str]) -> None:
    """The fake model: it picks the tool skill whenever the catalogue offers it, and lists its name as a package."""
    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        if "Skill Selection" in prompt[:200]:
            calls.append("SelectSkills")
            names = [SKILL] if SKILL in prompt else []
            return json.dumps({"skills": [{"name": n, "reason": "charts", "use": "experiment"} for n in names]})
        kind = _classify(prompt)
        calls.append(kind)
        prompts.append(prompt)
        if kind == "Implementation":
            return json.dumps({"code": _FAKE_EXPERIMENT_CODE, "deps": ["matplotlib", *deps]})
        return _fake_response_for(prompt)

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)


def _cfg_with_skill(tmp_path: Path) -> Config:
    cfg = _cfg(tmp_path)
    cfg.engine.skills = [SKILL]
    return cfg


@pytest.mark.asyncio
async def test_a_skill_excluded_since_the_run_is_gone_after_rerunning_from_skills_and_nothing_else_is_redone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, node_skill: Path,
) -> None:
    calls: list[str] = []
    prompts: list[str] = []
    searches: list[str] = []
    _fake(monkeypatch, calls, prompts, deps=[SKILL])

    from core.knowledge import Knowledge

    real_search = Knowledge.asearch

    async def counted(self, *a, **kw):  # noqa: ANN001
        import inspect

        searches.append(next((f.function for f in inspect.stack() if f.function.startswith("_node_")), "?"))
        return await real_search(self, *a, **kw)

    monkeypatch.setattr(Knowledge, "asearch", counted)
    cfg = _cfg_with_skill(tmp_path)
    first = Engine(cfg)
    await first.run()
    before = await _state(first)
    assert before["selected_skills"] == [SKILL]
    log = (first.fi_dir / "run.log").read_text(encoding="utf-8")
    # The failure the person met: the skill's name is asked of pip on every write of the code.
    first_installs = [ln for ln in log.splitlines() if "pip install" in ln and "[execute]" in ln]
    assert first_installs and SKILL in first_installs[0]
    frozen = first.quest_root / "needs" / "FROZEN_PROTOCOL.json"
    plan = first.quest_root / "plan.md"
    frozen_hash = _digest(frozen) if frozen.is_file() else ""
    plan_hash = _digest(plan) if plan.is_file() else ""
    searches_before, implements = len(searches), calls.count("Implementation")
    n_calls, n_prompts = len(calls), len(prompts)

    # The person adds the skill to skills_exclude and asks for the skills to be picked again.
    cfg2 = cfg.model_copy(deep=True)
    cfg2.engine.skills_exclude = [SKILL]
    second = Engine(cfg2, resume_quest_id=first.quest_id)
    await second.run(from_step="skills")

    after = await _state(second)
    assert after["selected_skills"] == []
    assert [d["name"] for d in after["skill_selection"]["dropped"]] == [SKILL]
    assert "skills_exclude" in after["skill_selection"]["dropped"][0]["why"]
    again = searches[searches_before:]
    assert "_node_literature" not in again and "_node_ideate" not in again, again
    assert set(again) <= {"_node_cross_check"}, again
    later = calls[n_calls:]
    assert not {"Ideation", "IdeateReflect", "Clarify"} & set(later), later
    assert calls.count("Implementation") > implements, "the code was not written again"
    assert frozen_hash == (_digest(frozen) if frozen.is_file() else "")
    assert plan_hash == (_digest(plan) if plan.is_file() else "")

    new_log = (second.fi_dir / "run.log").read_text(encoding="utf-8")[len(log):]
    assert f"dropped {SKILL}" in new_log
    assert "[literature]" not in new_log
    installs = [ln for ln in new_log.splitlines() if "pip install" in ln and "[execute]" in ln]
    assert installs and all(SKILL not in ln for ln in installs), installs
    rewritten = [p for p in prompts[n_prompts:] if "Not available to this quest" in p]
    assert rewritten and all(f"## Skill: {SKILL}" not in p for p in rewritten)
    assert (second.fi_dir / "previous").is_dir()


@pytest.mark.asyncio
async def test_from_code_keeps_the_pick_but_from_skills_uses_the_yaml_it_is_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, node_skill: Path,
) -> None:
    calls: list[str] = []
    _fake(monkeypatch, calls, [], deps=[])
    cfg = _cfg_with_skill(tmp_path)
    first = Engine(cfg)
    await first.run()
    excluded = cfg.model_copy(deep=True)
    excluded.engine.skills_exclude = [SKILL]
    selects = calls.count("SelectSkills")
    await Engine(excluded, resume_quest_id=first.quest_id).run(from_step="code")
    assert calls.count("SelectSkills") == selects, "--from code picked the skills again"
    assert (await _state(first))["selected_skills"] == [SKILL]
    await Engine(excluded, resume_quest_id=first.quest_id).run(from_step="skills")
    assert (await _state(first))["selected_skills"] == []


@pytest.mark.asyncio
async def test_a_skill_the_plan_names_is_reported_and_the_plan_is_left_as_it_is(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, node_skill: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[str] = []
    _fake(monkeypatch, calls, [], deps=[])
    cfg = _cfg_with_skill(tmp_path)
    first = Engine(cfg)
    await first.run()
    plan = first.quest_root / "plan.md"
    plan.write_text(plan.read_text(encoding="utf-8") + f"\nFigures are drawn with the {SKILL} skill.\n", encoding="utf-8") \
        if plan.is_file() else plan.write_text(f"Figures are drawn with the {SKILL} skill.\n", encoding="utf-8")
    plan_hash = _digest(plan)
    excluded = cfg.model_copy(deep=True)
    excluded.engine.skills_exclude = [SKILL]
    capsys.readouterr()
    log_path = first.fi_dir / "run.log"
    seen = len(log_path.read_text(encoding="utf-8"))
    await Engine(excluded, resume_quest_id=first.quest_id).run(from_step="skills")
    out = capsys.readouterr().out + log_path.read_text(encoding="utf-8")[seen:]
    assert f"skill {SKILL} is no longer used by this quest" in out
    assert f"the plan names the skill {SKILL}" in out and "--revise-plan" in out
    assert _digest(plan) == plan_hash


@pytest.mark.asyncio
async def test_a_skill_whose_own_packages_cannot_be_installed_is_dropped_from_the_pick(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, node_skill: Path,
) -> None:
    prov = node_skill / SKILL / "provenance.json"
    prov.write_text(json.dumps({"kind": "tool", "pip_requires": ["zz-no-such-package-fi-test"]}), encoding="utf-8")
    for sk in discover():
        approval.approve(sk.ledger_name, sk.content_hash(), approved_by="test", note="t")
    calls: list[str] = []
    _fake(monkeypatch, calls, [], deps=[])
    cfg = _cfg_with_skill(tmp_path)
    cfg.execution.shared_interpreter = False
    engine = Engine(cfg)
    await engine.run()
    state = await _state(engine)
    assert state["selected_skills"] == []
    assert state["skill_selection"]["dropped"][0]["name"] == SKILL
    assert "could not be installed here" in state["skill_selection"]["dropped"][0]["why"]
    log = (engine.fi_dir / "run.log").read_text(encoding="utf-8")
    assert f"dropped {SKILL}: its package zz-no-such-package-fi-test could not be installed here" in log


@pytest.mark.asyncio
async def test_a_skill_package_that_fails_once_but_installs_on_the_second_try_keeps_its_skill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, node_skill: Path,
) -> None:
    prov = node_skill / SKILL / "provenance.json"
    prov.write_text(json.dumps({"kind": "tool", "pip_requires": ["zz-flaky-package"]}), encoding="utf-8")
    for sk in discover():
        approval.approve(sk.ledger_name, sk.content_hash(), approved_by="test", note="t")
    calls: list[str] = []
    _fake(monkeypatch, calls, [], deps=[])
    real_install = Engine._install_packages
    attempts: list[str] = []

    async def flaky(self, packages):  # noqa: ANN001
        rest = [p for p in packages if p != "zz-flaky-package"]
        failed = await real_install(self, rest) if rest else []
        if "zz-flaky-package" in packages:
            attempts.append("zz-flaky-package")
            if len(attempts) == 1:
                failed.append(("zz-flaky-package", "connection reset"))
        return failed

    monkeypatch.setattr(Engine, "_install_packages", flaky)
    cfg = _cfg_with_skill(tmp_path)
    cfg.execution.shared_interpreter = False
    engine = Engine(cfg)
    await engine.run()
    assert len(attempts) == 2, "a failed install of a skill's package was not tried a second time"
    state = await _state(engine)
    assert state["selected_skills"] == [SKILL]
    assert not (state.get("skill_selection") or {}).get("dropped")
    assert f"dropped {SKILL}" not in (engine.fi_dir / "run.log").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_a_skill_whose_name_only_appears_inside_a_longer_name_is_not_reported_as_named_by_the_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, node_skill: Path,
) -> None:
    calls: list[str] = []
    _fake(monkeypatch, calls, [], deps=[])
    cfg = _cfg_with_skill(tmp_path)
    first = Engine(cfg)
    await first.run()
    plan = first.quest_root / "plan.md"
    plan.write_text(f"Use the {SKILL}-extras and pre-{SKILL} helpers.\n", encoding="utf-8")
    excluded = cfg.model_copy(deep=True)
    excluded.engine.skills_exclude = [SKILL]
    log_path = first.fi_dir / "run.log"
    seen = len(log_path.read_text(encoding="utf-8"))
    await Engine(excluded, resume_quest_id=first.quest_id).run(from_step="skills")
    new_log = log_path.read_text(encoding="utf-8")[seen:]
    assert f"the plan names the skill {SKILL}" not in new_log
    assert "the plan or design names" not in new_log


def test_a_dropped_library_skill_named_like_a_package_is_still_asked_of_pip(tmp_path: Path) -> None:
    engine = Engine(_cfg(tmp_path))
    state = {"skill_selection": {"dropped": [{"name": "matplotlib", "kind": "library", "why": "x"},
                                             {"name": "Chart_Tool", "kind": "tool", "why": "y"},
                                             {"name": "", "kind": "tool"}]}}
    assert engine._dropped_tool_names(state) == {"chart-tool"}
    assert engine._dropped_tool_names({}) == set()


@pytest.mark.asyncio
async def test_a_failing_package_that_is_not_the_skills_stays_failed_when_the_skills_package_is_tried_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, node_skill: Path,
) -> None:
    prov = node_skill / SKILL / "provenance.json"
    prov.write_text(json.dumps({"kind": "tool", "pip_requires": ["zz-no-such-package-fi-test"]}), encoding="utf-8")
    for sk in discover():
        approval.approve(sk.ledger_name, sk.content_hash(), approved_by="test", note="t")
    calls: list[str] = []
    _fake(monkeypatch, calls, [], deps=["zz-no-such-other-package-fi-test"])
    cfg = _cfg_with_skill(tmp_path)
    cfg.execution.shared_interpreter = False
    engine = Engine(cfg)
    await engine.run()
    state = await _state(engine)
    assert state["selected_skills"] == [] and state["skill_selection"]["dropped"][0]["name"] == SKILL
    assert "zz-no-such-other-package-fi-test" in (engine._packages_note or "")


def test_two_backups_in_the_same_second_get_their_own_folders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rerun_from.time, "strftime", lambda fmt: "20260929-100000")
    folders = []
    for n in range(3):
        (tmp_path / "code").mkdir(exist_ok=True)
        (tmp_path / "code" / "experiment.py").write_text(str(n), encoding="utf-8")
        where, moved = rerun_from.back_up(tmp_path, "code")
        assert where is not None and "code" in moved
        folders.append(where.name)
    assert folders == ["20260929-100000", "20260929-100000-2", "20260929-100000-3"]
    assert [(tmp_path / ".fi" / "previous" / f / "code" / "experiment.py").read_text(encoding="utf-8")
            for f in folders] == ["0", "1", "2"]
