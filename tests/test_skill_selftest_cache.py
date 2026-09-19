"""The self-test cache: a skill whose content has not changed is not tested again.

Loading skills for a quest ran every candidate's self-test, one after another,
on the event loop — measured at 340 s for a 108-skill library, with the web
server unable to answer meanwhile. These pin the three halves of the fix:

* a pass is recorded against the skill's content, the Python that ran it and
  the installed packages, and only a change to one of those tests it again;
* only passes are recorded, so a failing skill never becomes loadable by
  having been seen once;
* the tests that do run, run in parallel and off the event loop, with results
  identical to running them one at a time.
"""
from __future__ import annotations

import asyncio
import json
import textwrap
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.skills import Skill, Status, approval, evaluate, loadable_skills
from core.skills import registry, selftest_cache
from core.skills.base import SKILL_MD
from core.skills.selftest_cache import SelftestCache


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ledger(tmp_path: Path) -> Path:
    return tmp_path / "approvals.json"


@pytest.fixture
def cache_file(tmp_path: Path, monkeypatch) -> Path:
    p = tmp_path / "state" / "skill_selftest_cache.json"
    monkeypatch.setenv("FI_SKILLS_SELFTEST_CACHE", str(p))
    return p


@pytest.fixture
def runs(monkeypatch) -> list[str]:
    """Names of the self-tests actually executed, observed from outside."""
    seen: list[str] = []
    lock = threading.Lock()
    real = registry.run_selftest

    def counting(skill, **kw):
        with lock:
            seen.append(skill.name)
        return real(skill, **kw)

    monkeypatch.setattr(registry, "run_selftest", counting)
    return seen


def make_skill(
    root: Path, name: str, *, selftest: str = "import sys; sys.exit(0)",
    approve_in: Path | None = None,
) -> Skill:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / SKILL_MD).write_text(f"# {name}\n\nWhen to use it.\n", encoding="utf-8")
    (d / "selftest.py").write_text(textwrap.dedent(selftest), encoding="utf-8")
    s = Skill(name=name, path=d)
    if approve_in is not None:
        approval.approve(name, s.content_hash(), approved_by="tester", path=approve_in)
    return s


def rows(ok, rejected) -> list[tuple]:
    return [
        (st.skill.name, st.status, st.reason, st.approved_hash, tuple(st.findings))
        for st in ok + rejected
    ]


# ---------------------------------------------------------------------------
# a pass is reused, and only for the same key
# ---------------------------------------------------------------------------


def test_a_second_load_takes_the_pass_from_the_cache(
    tmp_path: Path, ledger: Path, cache_file: Path, runs: list[str],
) -> None:
    root = tmp_path / "skills"
    for n in ("alpha", "beta", "gamma"):
        make_skill(root, n, approve_in=ledger)

    first = loadable_skills(["alpha", "beta", "gamma"], skills_dir=root, ledger=ledger)
    assert sorted(runs) == ["alpha", "beta", "gamma"]
    assert [st.status for st in first[0]] == [Status.TRUSTED] * 3

    runs.clear()
    second = loadable_skills(["alpha", "beta", "gamma"], skills_dir=root, ledger=ledger)
    assert runs == [], "an unchanged skill was tested again"
    assert rows(*second) == rows(*first)
    assert len(json.loads(cache_file.read_text("utf-8"))["entries"]) == 3


def test_a_content_change_retests_only_that_skill(
    tmp_path: Path, ledger: Path, cache_file: Path, runs: list[str],
) -> None:
    root = tmp_path / "skills"
    for n in ("alpha", "beta", "gamma"):
        make_skill(root, n, approve_in=ledger)
    loadable_skills(["alpha", "beta", "gamma"], skills_dir=root, ledger=ledger)

    (root / "beta" / "notes.txt").write_text("a harmless new file\n", encoding="utf-8")
    runs.clear()
    ok, rejected = loadable_skills(["alpha", "beta", "gamma"], skills_dir=root, ledger=ledger)
    assert runs == ["beta"]
    # The approval gate is unchanged: new content needs a new signature.
    assert [st.skill.name for st in ok] == ["alpha", "gamma"]
    assert rejected[0].skill.name == "beta" and "re-approval required" in rejected[0].reason


def test_a_changed_package_fingerprint_retests_every_skill(
    tmp_path: Path, ledger: Path, cache_file: Path, runs: list[str], monkeypatch,
) -> None:
    root = tmp_path / "skills"
    for n in ("alpha", "beta", "gamma"):
        make_skill(root, n, approve_in=ledger)
    loadable_skills(["alpha", "beta", "gamma"], skills_dir=root, ledger=ledger)

    real = selftest_cache.package_fingerprint
    monkeypatch.setattr(selftest_cache, "package_fingerprint", lambda: "upgraded:" + real())
    runs.clear()
    loadable_skills(["alpha", "beta", "gamma"], skills_dir=root, ledger=ledger)
    assert sorted(runs) == ["alpha", "beta", "gamma"]


@pytest.mark.parametrize("field", ["python", "executable"])
def test_a_different_interpreter_retests_every_skill(
    tmp_path: Path, ledger: Path, cache_file: Path, runs: list[str], monkeypatch,
    field: str,
) -> None:
    root = tmp_path / "skills"
    for n in ("alpha", "beta"):
        make_skill(root, n, approve_in=ledger)
    loadable_skills(["alpha", "beta"], skills_dir=root, ledger=ledger)

    real = selftest_cache.current_environment

    def other():
        env = real()
        return selftest_cache.Environment(**{**env.__dict__, field: "other-" + field})

    monkeypatch.setattr(selftest_cache, "current_environment", other)
    runs.clear()
    loadable_skills(["alpha", "beta"], skills_dir=root, ledger=ledger)
    assert sorted(runs) == ["alpha", "beta"]


def test_the_fingerprint_is_computed_once_per_load(
    tmp_path: Path, ledger: Path, cache_file: Path, monkeypatch,
) -> None:
    root = tmp_path / "skills"
    for n in ("alpha", "beta", "gamma", "delta"):
        make_skill(root, n)
    calls: list[int] = []
    real = selftest_cache.package_fingerprint
    monkeypatch.setattr(
        selftest_cache, "package_fingerprint", lambda: calls.append(1) or real(),
    )
    loadable_skills(["alpha", "beta", "gamma", "delta"], skills_dir=root, ledger=ledger)
    assert len(calls) == 1
    loadable_skills(["alpha", "beta", "gamma", "delta"], skills_dir=root, ledger=ledger)
    assert len(calls) == 2, "a long-running process must see a pip install"


def test_a_cached_pass_does_not_bypass_approval(
    tmp_path: Path, ledger: Path, cache_file: Path, runs: list[str],
) -> None:
    root = tmp_path / "skills"
    make_skill(root, "alpha", approve_in=ledger)
    assert loadable_skills(["alpha"], skills_dir=root, ledger=ledger)[0]
    approval.revoke("alpha", ledger)
    runs.clear()
    ok, rejected = loadable_skills(["alpha"], skills_dir=root, ledger=ledger)
    assert runs == [] and ok == []
    assert rejected[0].status is Status.PROPOSED
    assert rejected[0].reason.startswith("awaiting approval")


# ---------------------------------------------------------------------------
# only passes are recorded
# ---------------------------------------------------------------------------


def test_a_failing_selftest_is_never_cached(
    tmp_path: Path, ledger: Path, cache_file: Path, runs: list[str],
) -> None:
    root = tmp_path / "skills"
    make_skill(root, "good", approve_in=ledger)
    make_skill(root, "broken", selftest="import sys; sys.exit(1)", approve_in=ledger)

    for _ in range(2):
        runs.clear()
        ok, rejected = loadable_skills(["good", "broken"], skills_dir=root, ledger=ledger)
        assert [st.skill.name for st in ok] == ["good"]
        assert rejected[0].status is Status.QUARANTINED
    assert runs == ["broken"], "the failing self-test was answered from the cache"
    names = {e["name"] for e in json.loads(cache_file.read_text("utf-8"))["entries"].values()}
    assert names == {"good"}


def test_a_timed_out_selftest_is_never_cached(
    tmp_path: Path, ledger: Path, cache_file: Path, runs: list[str], monkeypatch,
) -> None:
    # run_selftest never allows less than 20 s by itself, so hand the real
    # function a short ceiling rather than waiting that out twice.
    counted = registry.run_selftest
    monkeypatch.setattr(
        registry, "run_selftest",
        lambda skill, **kw: counted(skill, **{**kw, "timeout_s": 2}),
    )
    root = tmp_path / "skills"
    make_skill(root, "slow", selftest="import time; time.sleep(30)", approve_in=ledger)
    for _ in range(2):
        _, rejected = loadable_skills(["slow"], skills_dir=root, ledger=ledger)
        assert rejected[0].status is Status.QUARANTINED
        assert rejected[0].selftest_output.startswith("selftest timed out")
    assert runs == ["slow", "slow"]
    assert not cache_file.exists() or not json.loads(cache_file.read_text("utf-8"))["entries"]


def test_a_failure_on_demand_removes_the_recorded_pass(
    tmp_path: Path, ledger: Path, cache_file: Path, runs: list[str], monkeypatch,
) -> None:
    """The key cannot see what lives outside Python — an external binary, a
    service. A failure someone just watched must outrank an older pass, or
    quests keep loading a skill that ``--skills`` shows broken."""
    root = tmp_path / "skills"
    s = make_skill(
        root, "tool",
        selftest="import os, sys; sys.exit(1 if os.environ.get('FI_TEST_TOOL_GONE') else 0)",
        approve_in=ledger,
    )
    assert loadable_skills(["tool"], skills_dir=root, ledger=ledger)[0]

    monkeypatch.setenv("FI_TEST_TOOL_GONE", "1")
    # On demand: fresh, whatever the cache says.
    runs.clear()
    assert evaluate(s, ledger=ledger).status is Status.QUARANTINED
    assert runs == ["tool"]

    runs.clear()
    ok, rejected = loadable_skills(["tool"], skills_dir=root, ledger=ledger)
    assert runs == ["tool"] and ok == []
    assert rejected[0].status is Status.QUARANTINED


# ---------------------------------------------------------------------------
# on-demand checks stay fresh; run_test=False executes nothing
# ---------------------------------------------------------------------------


def test_evaluate_without_a_cache_always_runs_the_selftest(
    tmp_path: Path, ledger: Path, cache_file: Path, runs: list[str],
) -> None:
    root = tmp_path / "skills"
    s = make_skill(root, "alpha", approve_in=ledger)
    loadable_skills(["alpha"], skills_dir=root, ledger=ledger)
    runs.clear()
    assert evaluate(s, ledger=ledger).status is Status.TRUSTED
    assert evaluate(s, ledger=ledger).status is Status.TRUSTED
    assert runs == ["alpha", "alpha"]


def test_run_test_false_executes_nothing_even_with_a_cache(
    tmp_path: Path, ledger: Path, cache_file: Path, runs: list[str],
) -> None:
    s = make_skill(tmp_path / "skills", "alpha", approve_in=ledger)
    cache = SelftestCache.open(ledger=ledger)
    assert evaluate(s, ledger=ledger, run_test=False, cache=cache).status is Status.TRUSTED
    assert runs == [] and not cache_file.exists()


def test_use_cache_false_runs_every_selftest(
    tmp_path: Path, ledger: Path, cache_file: Path, runs: list[str],
) -> None:
    root = tmp_path / "skills"
    make_skill(root, "alpha", approve_in=ledger)
    loadable_skills(["alpha"], skills_dir=root, ledger=ledger)
    runs.clear()
    loadable_skills(["alpha"], skills_dir=root, ledger=ledger, use_cache=False)
    assert runs == ["alpha"]


# ---------------------------------------------------------------------------
# findings travel with the pass
# ---------------------------------------------------------------------------


def test_the_scan_is_reused_with_the_pass_and_redone_for_a_new_scanner(
    tmp_path: Path, ledger: Path, cache_file: Path, monkeypatch,
) -> None:
    root = tmp_path / "skills"
    # eval() is a finding, so there is something to carry.
    make_skill(root, "alpha", selftest="import sys\neval('1')\nsys.exit(0)\n", approve_in=ledger)
    scans: list[str] = []
    real_scan = registry._scan_findings
    monkeypatch.setattr(
        registry, "_scan_findings", lambda s: scans.append(s.name) or real_scan(s),
    )

    first = loadable_skills(["alpha"], skills_dir=root, ledger=ledger)
    assert scans == ["alpha"] and first[0][0].findings
    scans.clear()
    second = loadable_skills(["alpha"], skills_dir=root, ledger=ledger)
    assert scans == [] and second[0][0].findings == first[0][0].findings

    monkeypatch.setattr(selftest_cache, "scanner_version", lambda: "a-newer-scanner")
    loadable_skills(["alpha"], skills_dir=root, ledger=ledger)
    assert scans == ["alpha"]
    scans.clear()
    loadable_skills(["alpha"], skills_dir=root, ledger=ledger)
    assert scans == [], "the refreshed scan was not recorded"


def test_a_scan_that_crashed_is_not_recorded(
    tmp_path: Path, ledger: Path, cache_file: Path, monkeypatch,
) -> None:
    from core.skills import scan

    root = tmp_path / "skills"
    make_skill(root, "alpha", approve_in=ledger)
    monkeypatch.setattr(scan, "scan", lambda s: (_ for _ in ()).throw(RuntimeError("boom")))
    ok, _ = loadable_skills(["alpha"], skills_dir=root, ledger=ledger)
    assert ok[0].findings[0].startswith("scan did not run")
    entry = next(iter(json.loads(cache_file.read_text("utf-8"))["entries"].values()))
    assert entry["findings"] is None


# ---------------------------------------------------------------------------
# where it lives, and that it never gets in the way
# ---------------------------------------------------------------------------


def test_the_cache_lives_beside_the_ledger_unless_overridden(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.delenv("FI_SKILLS_SELFTEST_CACHE", raising=False)
    led = tmp_path / "somewhere" / "approvals.json"
    assert selftest_cache.cache_path(led) == led.parent / "skill_selftest_cache.json"
    monkeypatch.setenv("FI_SKILLS_APPROVALS", str(led))
    assert selftest_cache.cache_path() == led.parent / "skill_selftest_cache.json"
    monkeypatch.delenv("FI_SKILLS_APPROVALS")
    assert selftest_cache.cache_path().parent == approval.ledger_path().parent
    monkeypatch.setenv("FI_SKILLS_SELFTEST_CACHE", str(tmp_path / "x.json"))
    assert selftest_cache.cache_path(led) == tmp_path / "x.json"


def test_loading_never_writes_into_a_skill_directory(
    tmp_path: Path, ledger: Path, cache_file: Path,
) -> None:
    root = tmp_path / "skills"
    skills = [make_skill(root, n, approve_in=ledger) for n in ("alpha", "beta")]
    before = {s.name: (s.content_hash(), sorted(p.name for p in s.path.rglob("*"))) for s in skills}
    for _ in range(2):
        ok, _ = loadable_skills(["alpha", "beta"], skills_dir=root, ledger=ledger)
        assert len(ok) == 2
    after = {s.name: (s.content_hash(), sorted(p.name for p in s.path.rglob("*"))) for s in skills}
    assert after == before
    assert cache_file.is_file() and root not in cache_file.parents


@pytest.mark.parametrize("garbage", [b"{not json", b"[1, 2]", b'{"format": 99, "entries": {}}', b"\xff\xfe\x00"])
def test_a_corrupt_cache_is_an_empty_cache(
    tmp_path: Path, ledger: Path, cache_file: Path, runs: list[str], garbage: bytes,
) -> None:
    root = tmp_path / "skills"
    make_skill(root, "alpha", approve_in=ledger)
    cache_file.parent.mkdir(parents=True)
    cache_file.write_bytes(garbage)
    ok, _ = loadable_skills(["alpha"], skills_dir=root, ledger=ledger)
    assert [st.skill.name for st in ok] == ["alpha"] and runs == ["alpha"]
    # ...and is replaced by a valid one.
    assert json.loads(cache_file.read_text("utf-8"))["format"] == 1
    runs.clear()
    loadable_skills(["alpha"], skills_dir=root, ledger=ledger)
    assert runs == []


def test_an_unwritable_cache_never_stops_a_load(
    tmp_path: Path, ledger: Path, monkeypatch, runs: list[str],
) -> None:
    blocker = tmp_path / "is-a-directory"
    blocker.mkdir()
    monkeypatch.setenv("FI_SKILLS_SELFTEST_CACHE", str(blocker))
    root = tmp_path / "skills"
    make_skill(root, "alpha", approve_in=ledger)
    make_skill(root, "broken", selftest="import sys; sys.exit(1)")
    for _ in range(2):
        ok, rejected = loadable_skills(["alpha", "broken"], skills_dir=root, ledger=ledger)
        assert [st.skill.name for st in ok] == ["alpha"]
        assert rejected[0].status is Status.QUARANTINED
    assert sorted(runs) == ["alpha", "alpha", "broken", "broken"]


def test_an_environment_that_cannot_be_fingerprinted_runs_without_the_cache(
    tmp_path: Path, ledger: Path, cache_file: Path, runs: list[str], monkeypatch,
) -> None:
    def broken():
        raise RuntimeError("a dist-info with no METADATA")

    monkeypatch.setattr(selftest_cache, "package_fingerprint", broken)
    root = tmp_path / "skills"
    make_skill(root, "alpha", approve_in=ledger)
    for _ in range(2):
        assert loadable_skills(["alpha"], skills_dir=root, ledger=ledger)[0]
    assert runs == ["alpha", "alpha"] and not cache_file.exists()


def test_a_hand_edited_entry_is_a_miss(
    tmp_path: Path, ledger: Path, cache_file: Path, runs: list[str],
) -> None:
    root = tmp_path / "skills"
    make_skill(root, "alpha", approve_in=ledger)
    loadable_skills(["alpha"], skills_dir=root, ledger=ledger)
    doc = json.loads(cache_file.read_text("utf-8"))
    for entry in doc["entries"].values():
        entry["key"][1] = "0000000000000000"
    cache_file.write_text(json.dumps(doc), encoding="utf-8")
    runs.clear()
    loadable_skills(["alpha"], skills_dir=root, ledger=ledger)
    assert runs == ["alpha"]


# ---------------------------------------------------------------------------
# parallel, and identical to sequential
# ---------------------------------------------------------------------------


SLEEP_S = 1.5


def test_selftests_run_in_parallel(
    tmp_path: Path, ledger: Path, cache_file: Path,
) -> None:
    root = tmp_path / "skills"
    names = [f"slow{i}" for i in range(4)]
    for n in names:
        make_skill(root, n, selftest=f"import time; time.sleep({SLEEP_S})", approve_in=ledger)
    t = time.monotonic()
    ok, _ = loadable_skills(names, skills_dir=root, ledger=ledger, max_workers=4)
    elapsed = time.monotonic() - t
    assert [st.skill.name for st in ok] == names
    # One at a time is at least 4 x 1.5 = 6 s.
    assert elapsed < 0.75 * len(names) * SLEEP_S, f"took {elapsed:.1f}s — not parallel"


def test_the_default_worker_count_is_bounded(monkeypatch) -> None:
    monkeypatch.setattr(registry.os, "cpu_count", lambda: 64)
    assert registry.default_selftest_workers() == 8
    monkeypatch.setattr(registry.os, "cpu_count", lambda: 2)
    assert registry.default_selftest_workers() == 2
    monkeypatch.setattr(registry.os, "cpu_count", lambda: None)
    assert registry.default_selftest_workers() == 1


def test_parallel_results_equal_sequential_results(
    tmp_path: Path, ledger: Path, cache_file: Path,
) -> None:
    root = tmp_path / "skills"
    make_skill(root, "trusted-a", approve_in=ledger)
    make_skill(root, "proposed", selftest="import time; time.sleep(0.3)")
    make_skill(root, "broken", selftest="import sys; print('nope'); sys.exit(3)", approve_in=ledger)
    make_skill(root, "trusted-b", selftest="import time; time.sleep(0.2)", approve_in=ledger)
    stale = make_skill(root, "stale", approve_in=ledger)
    (stale.path / SKILL_MD).write_text("# stale\n\nRewritten.\n", encoding="utf-8")
    untested = root / "untested"
    untested.mkdir()
    (untested / SKILL_MD).write_text("# untested\n", encoding="utf-8")
    names = ["broken", "trusted-b", "missing", "proposed", "untested", "stale",
             "trusted-a", "trusted-b"]

    seq = loadable_skills(names, skills_dir=root, ledger=ledger, use_cache=False, max_workers=1)
    par = loadable_skills(names, skills_dir=root, ledger=ledger, use_cache=False, max_workers=4)
    assert rows(*par) == rows(*seq)
    assert [st.selftest_output for st in par[1]] == [st.selftest_output for st in seq[1]]
    assert [st.skill.name for st in par[0]] == ["trusted-b", "trusted-a", "trusted-b"]
    # And a cold cached load, then a warm one, decide the same.
    cold = loadable_skills(names, skills_dir=root, ledger=ledger, max_workers=4)
    warm = loadable_skills(names, skills_dir=root, ledger=ledger, max_workers=4)
    assert rows(*cold) == rows(*seq) == rows(*warm)


# ---------------------------------------------------------------------------
# off the event loop
# ---------------------------------------------------------------------------


def test_select_skills_keeps_the_event_loop_responsive(
    tmp_path: Path, ledger: Path, cache_file: Path, monkeypatch,
) -> None:
    """While the self-tests run, a concurrent task on the same loop keeps
    ticking — which is what lets the web server answer during a quest's
    skill loading."""
    from core.engine import Engine

    root = tmp_path / "skills"
    for i in range(3):
        # Unapproved: the catalogue ends up empty, so the node returns before
        # it would call a model.
        make_skill(root, f"slow{i}", selftest=f"import time; time.sleep({SLEEP_S})")
    monkeypatch.setenv("FI_SKILLS_DIR", str(root))
    real = registry.loadable_skills
    monkeypatch.setattr(
        "core.skills.loadable_skills",
        lambda names, **kw: real(names, skills_dir=root, ledger=ledger),
    )
    eng = SimpleNamespace(
        config=SimpleNamespace(engine=SimpleNamespace(skills=[], skills_exclude=[])),
        _skill_dirs=None,
        _log=SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None),
    )

    async def scenario():
        ticks: list[float] = []
        stop = asyncio.Event()

        async def ticker():
            while not stop.is_set():
                ticks.append(time.monotonic())
                await asyncio.sleep(0.1)
            ticks.append(time.monotonic())

        task = asyncio.create_task(ticker())
        await asyncio.sleep(0.3)
        t = time.monotonic()
        out = await Engine._node_select_skills(eng, {"topic": "anything"})  # type: ignore[arg-type]
        elapsed = time.monotonic() - t
        stop.set()
        await task
        gaps = [b - a for a, b in zip(ticks, ticks[1:])]
        return out, elapsed, max(gaps)

    out, elapsed, worst_gap = asyncio.run(scenario())
    assert out["skill_selection"] == {"candidates": 0}
    assert elapsed >= SLEEP_S, "the self-tests did not actually run"
    assert worst_gap < 0.5, f"event loop stalled for {worst_gap:.2f}s while skills loaded"
