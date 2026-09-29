"""The suite's template-venv shortcut (tests/conftest.py) must yield venvs that really run."""

from __future__ import annotations

from pathlib import Path

from core.execution import VenvExecutor


async def test_copied_venv_runs_python_and_pip(tmp_path: Path) -> None:
    ex = VenvExecutor()
    root = tmp_path / "quest"
    await ex.setup(root)
    py = ex.python_path(root)
    assert py.is_file()
    ran = await ex.execute(
        [str(py), "-c", "import sys; print(sys.prefix)"], cwd=root, timeout_s=60,
    )
    assert ran.returncode == 0, ran.stderr
    # The copy runs as itself, not as the template it was copied from.
    assert Path(ran.stdout.strip()).resolve() == (root / ".venv").resolve()
    pip = await ex.execute([str(py), "-m", "pip", "--version"], cwd=root, timeout_s=60)
    assert pip.returncode == 0, pip.stderr


async def test_two_quests_get_independent_copies(tmp_path: Path) -> None:
    ex = VenvExecutor()
    a, b = tmp_path / "a", tmp_path / "b"
    await ex.setup(a)
    await ex.setup(b)
    (a / ".venv" / "marker").write_text("a")
    assert not (b / ".venv" / "marker").exists()
