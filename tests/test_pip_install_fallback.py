"""``launch._pip_install``: one package pip cannot install must not take the others with it.

pip installs all or nothing. A skill library names dozens of packages, and one of them (say ``simpeg``, which has
no wheel for some Pythons, on a laptop without a compiler) failing to build made the single ``pip install a b c ...``
fail and left every skill quarantined, while the message promised that only the skills needing the missing package
would stay so. A failed batch is now retried one package at a time.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import launch


class _Pip:
    """Stands in for ``subprocess.run`` on pip commands: ``bad`` packages fail, everything else installs."""

    def __init__(self, bad: set[str], *, message: str = "ERROR: Failed building wheel for {pkg}") -> None:
        self.bad, self.message, self.calls = bad, message, []

    def __call__(self, cmd, *a, **k):
        argv = list(cmd)
        assert argv[1:4] == ["-m", "pip", "install"], argv
        packages = [p for p in argv[4:] if not p.startswith("-")]
        self.calls.append(packages)
        failing = [p for p in packages if p in self.bad]
        if not failing:
            return subprocess.CompletedProcess(argv, 0, "", "")
        return subprocess.CompletedProcess(
            argv, 1, "", f"Collecting {failing[0]}\n{self.message.format(pkg=failing[0])}\n",
        )


def test_a_batch_that_installs_is_one_call(monkeypatch) -> None:
    pip = _Pip(set())
    monkeypatch.setattr(subprocess, "run", pip)
    assert launch._pip_install(["numpy", "scipy"]) == (["numpy", "scipy"], {})
    assert pip.calls == [["numpy", "scipy"]]


def test_nothing_to_install_is_no_call(monkeypatch) -> None:
    pip = _Pip(set())
    monkeypatch.setattr(subprocess, "run", pip)
    assert launch._pip_install([]) == ([], {})
    assert pip.calls == []


def test_one_package_that_fails_does_not_take_the_others_with_it(monkeypatch, capsys) -> None:
    pip = _Pip({"simpeg"})
    monkeypatch.setattr(subprocess, "run", pip)
    installed, failed = launch._pip_install(["pymc", "simpeg", "astropy"])
    assert installed == ["pymc", "astropy"]
    assert failed == {"simpeg": "ERROR: Failed building wheel for simpeg"}
    # the whole batch first, then one at a time
    assert pip.calls == [["pymc", "simpeg", "astropy"], ["pymc"], ["simpeg"], ["astropy"]]
    out = capsys.readouterr().out
    assert "trying 3 one at a time" in out and "could not install simpeg: ERROR: Failed building wheel" in out


def test_every_package_failing_is_reported_for_each(monkeypatch) -> None:
    monkeypatch.setattr(subprocess, "run", _Pip({"a", "b"}, message="ERROR: No matching distribution found for {pkg}"))
    installed, failed = launch._pip_install(["a", "b"])
    assert installed == [] and set(failed) == {"a", "b"}
    assert failed["b"] == "ERROR: No matching distribution found for b"


def test_pip_can_never_wait_for_input(monkeypatch) -> None:
    seen = []

    def spy(cmd, *a, **k):
        seen.append(k.get("stdin"))
        return subprocess.CompletedProcess(list(cmd), 0, "", "")

    monkeypatch.setattr(subprocess, "run", spy)
    launch._pip_install(["numpy"])
    assert seen == [subprocess.DEVNULL]


# --- the bulk approval that uses it --------------------------------------------


def _skill(root: Path, name: str, module: str) -> None:
    d = root / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: demo skill\n---\nBody\n", encoding="utf-8")
    (d / "selftest.py").write_text(f"raise ModuleNotFoundError(\"No module named '{module}'\")", encoding="utf-8")
    (d / "provenance.json").write_text(json.dumps({"origin": "test"}), encoding="utf-8")


def test_bulk_approval_names_what_could_not_be_installed_and_keeps_going(tmp_path: Path, monkeypatch, capsys) -> None:
    root = tmp_path / "skills"
    root.mkdir()
    monkeypatch.setenv("FI_SKILLS_DIR", str(root))
    monkeypatch.setenv("FI_SKILLS_APPROVALS", str(tmp_path / "approvals.json"))
    _skill(root, "needs-qutip", "qutip")
    _skill(root, "needs-simpeg", "simpeg")
    real = subprocess.run
    pip = _Pip({"simpeg"})

    def run(cmd, *a, **k):
        argv = list(cmd) if isinstance(cmd, (list, tuple)) else [cmd]
        if argv[1:3] == ["-m", "pip"]:
            return pip(cmd, *a, **k)
        return real(cmd, *a, **k)

    monkeypatch.setattr(subprocess, "run", run)
    launch._approve_all_skills("tester", pip_install=True)
    out = capsys.readouterr().out
    assert "Not installed: simpeg. The skills that need them stay quarantined; the rest are unaffected." in out
    assert ["qutip"] in pip.calls  # installed on its own once the batch had failed


def test_the_import_script_uses_the_same_installer() -> None:
    script = (Path(__file__).resolve().parent.parent / "scripts" / "import_scientist_skills.py").read_text(encoding="utf-8")
    assert "launch._pip_install(all_pip)" in script


def test_a_package_pip_cannot_install_is_reported_with_the_python_and_what_pip_said(monkeypatch, capsys) -> None:
    import sys

    monkeypatch.setattr(subprocess, "run", _Pip({"landlab"}, message="ERROR: Could not find a version that satisfies landlab"))
    launch._pip_install(["landlab", "numpy"])
    out = capsys.readouterr().out
    assert f"under Python {sys.version.split()[0]} at {sys.executable}" in out
    assert "| ERROR: Could not find a version that satisfies landlab" in out
