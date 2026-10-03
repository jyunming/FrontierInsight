"""The three places that still stopped only the program they started now stop everything it started.

``core/engine.py`` (``pip freeze`` for the environment record), ``generation/slides.py`` (Marp / pandoc) and
``web/quest_launcher.py`` (a quest or tool job the web server started, on Cancel) killed only their direct child: a
helper it had started kept running, holding files and the output pipe. Each test here runs the real call site on a
program that starts a subprocess helper and a ``multiprocessing`` worker (the programs of ``test_proc_tree_async.py``),
then checks after the timeout or the cancel that the helpers stopped beating, their processes are gone, and neither
wrote its "still running" marker.
"""

from __future__ import annotations

import json
import sys
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.test_proc_tree_async import _assert_nothing_left, _wait_until_beating, _write_programs


@pytest.mark.asyncio
async def test_a_timed_out_pip_freeze_stops_with_everything_it_started(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import core.engine as engine_mod

    helper, program, folder = _write_programs(tmp_path)
    timeout_s = 10.0
    late_at = time.time() + timeout_s + 4
    # ``python -m pip freeze`` finds this ``pip`` first (PYTHONPATH comes before site-packages): it starts the program,
    # which starts the helpers, and waits.
    fake = tmp_path / "fakepip" / "pip"
    fake.mkdir(parents=True)
    (fake / "__init__.py").write_text("", encoding="utf-8")
    (fake / "__main__.py").write_text(textwrap.dedent(f"""
        import subprocess, sys
        subprocess.run([sys.executable, {str(program)!r}, {str(helper)!r}, {str(folder)!r}, {str(late_at)!r}, "stay"])
    """), encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(fake.parent))
    monkeypatch.setattr(engine_mod, "_PIP_FREEZE_TIMEOUT_S", timeout_s)
    quest_root = tmp_path / "quest"
    quest_root.mkdir()
    checks: list[str] = []
    fake_engine = SimpleNamespace(
        quest_root=quest_root,
        config=SimpleNamespace(execution=SimpleNamespace(
            sandbox="venv", shared_interpreter=False, system_site_packages=False)),
        executor=SimpleNamespace(python_path=lambda _root: Path(sys.executable)),
        _audit_check=lambda kind, *a, **k: checks.append(kind),
    )
    started = time.monotonic()
    await engine_mod.Engine._record_environment(fake_engine, stage="after installing the packages")
    assert time.monotonic() - started < timeout_s + 30
    record = json.loads((quest_root / "needs" / "ENVIRONMENT.json").read_text(encoding="utf-8"))
    assert "TimeoutError" in record["packages_error"]
    assert checks == ["environment"]
    await _assert_nothing_left(folder, late_at)


@pytest.mark.asyncio
async def test_a_timed_out_slide_renderer_stops_with_everything_it_started(tmp_path: Path) -> None:
    from generation.slides import _run_cli

    helper, program, folder = _write_programs(tmp_path)
    timeout_s = 10.0
    late_at = time.time() + timeout_s + 4
    ok, reason = await _run_cli(
        [sys.executable, str(program), str(helper), str(folder), str(late_at), "stay"],
        cwd=tmp_path, label="marp pdf", timeout_s=timeout_s,
    )
    assert ok is False
    assert reason is not None and "time" in " ".join(reason).lower()
    await _assert_nothing_left(folder, late_at)


@pytest.mark.asyncio
async def test_a_cancelled_slide_renderer_stops_with_everything_it_started(tmp_path: Path) -> None:
    import asyncio

    from generation.slides import _run_cli

    helper, program, folder = _write_programs(tmp_path)
    late_at = time.time() + 60
    task = asyncio.create_task(_run_cli(
        [sys.executable, str(program), str(helper), str(folder), str(late_at), "stay"],
        cwd=tmp_path, label="pandoc pptx", timeout_s=300,
    ))
    await _wait_until_beating(folder)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await _assert_nothing_left(folder, time.time())


@pytest.mark.asyncio
async def test_cancelling_a_quest_the_web_server_started_stops_everything_it_started(tmp_path: Path) -> None:
    """The launcher starts ``python launch.py ...`` from its repo root; here that ``launch.py`` is the program that
    starts the helpers and then ignores the polite stop (as an engine busy in a long script can)."""
    from web.quest_launcher import QuestLauncher

    helper, program, folder = _write_programs(tmp_path)
    late_at = time.time() + 60
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "launch.py").write_text(textwrap.dedent(f"""
        import signal, subprocess, sys
        for name in ("SIGTERM", "SIGBREAK"):
            if hasattr(signal, name):
                signal.signal(getattr(signal, name), signal.SIG_IGN)
        subprocess.run([sys.executable, {str(program)!r}, {str(helper)!r}, {str(folder)!r}, {str(late_at)!r}, "stay"])
    """), encoding="utf-8")
    launcher = QuestLauncher(repo_root=repo, python_path=sys.executable, output_root=tmp_path / "out",
                             work_dir=tmp_path)
    yaml_path = tmp_path / "q.yaml"
    yaml_path.write_text("topic: x\n", encoding="utf-8")
    entry = launcher.launch(quest_id="q-1", yaml_path=yaml_path)
    try:
        await _wait_until_beating(folder)
        assert launcher.cancel("q-1", grace_s=1.0) is True
        assert entry.process.poll() is not None
        await _assert_nothing_left(folder, time.time())
    finally:
        if entry.tree is not None:
            entry.tree.kill()


def test_a_quest_the_web_server_started_is_left_running_when_the_launcher_lets_go_of_it(tmp_path: Path) -> None:
    """Detached: dropping the launcher's handle (the server exiting) must not stop the quest."""
    import gc

    from web.quest_launcher import QuestLauncher

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "launch.py").write_text("import time\ntime.sleep(60)\n", encoding="utf-8")
    launcher = QuestLauncher(repo_root=repo, python_path=sys.executable, output_root=tmp_path / "out",
                             work_dir=tmp_path)
    yaml_path = tmp_path / "q.yaml"
    yaml_path.write_text("topic: x\n", encoding="utf-8")
    entry = launcher.launch(quest_id="q-2", yaml_path=yaml_path)
    proc, tree = entry.process, entry.tree
    assert tree is not None
    try:
        tree.close()
        del launcher, entry
        gc.collect()
        time.sleep(0.5)
        assert proc.poll() is None, "the quest stopped when the launcher let go of it"
    finally:
        tree.kill()
