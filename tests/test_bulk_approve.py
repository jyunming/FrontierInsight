"""`--approve-all-skills`: bulk approval that keeps the gates.

Bulk removes the typing, not the decision. Attribution is still required, a
failing self-test is still refused, and high-severity findings still need an
explicit sweep -- otherwise this would approve things `--approve-skill`
rejects, which is exactly what the gate exists to prevent.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

import launch


def _ledger(tmp_path: Path) -> dict:
    """The ledger file is only created once something is approved, so a run
    that approves nothing legitimately leaves no file."""
    f = tmp_path / "approvals.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.is_file() else {}


def _make_skill(
    root: Path, name: str, *, selftest: str | None = "import sys; sys.exit(0)",
    provenance: dict | None = None,
) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: demo skill\n---\nBody\n", encoding="utf-8",
    )
    if selftest is not None:
        (d / "selftest.py").write_text(selftest, encoding="utf-8")
    (d / "provenance.json").write_text(
        json.dumps(provenance or {"origin": "test"}), encoding="utf-8",
    )
    return d


def _spy_on_pip(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Record pip invocations, passing everything else through.

    evaluate() runs each skill's self-test via subprocess.run, so a blanket
    stub would break the very gate these tests are checking.
    """
    import subprocess as _sp
    real = _sp.run
    calls: list[list[str]] = []

    def spy(cmd, *a, **k):
        argv = list(cmd) if isinstance(cmd, (list, tuple)) else [cmd]
        if len(argv) >= 3 and argv[1:3] == ["-m", "pip"]:
            calls.append(argv)
            return type("R", (), {"returncode": 0})()
        return real(cmd, *a, **k)

    monkeypatch.setattr(_sp, "run", spy)
    return calls


@pytest.fixture()
def skills_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "skills"
    root.mkdir()
    monkeypatch.setenv("FI_SKILLS_DIR", str(root))
    monkeypatch.setenv("FI_SKILLS_APPROVALS", str(tmp_path / "approvals.json"))
    return root


# ------------------------------------------------------------ attribution

def test_bulk_refuses_without_approver(skills_env: Path, capsys) -> None:
    """The no-anonymous-approver rule is not relaxed by doing it in bulk."""
    _make_skill(skills_env, "alpha")
    assert launch._approve_all_skills("") == 2
    assert "requires --approve-as" in capsys.readouterr().out


def test_bulk_records_the_named_approver(skills_env: Path, tmp_path: Path) -> None:
    _make_skill(skills_env, "alpha")
    assert launch._approve_all_skills("jyunming") == 0
    assert _ledger(tmp_path)["alpha"]["approved_by"] == "jyunming"


# ------------------------------------------------------------------ gates

def test_bulk_refuses_failing_selftest(skills_env: Path, tmp_path: Path) -> None:
    """Approving something that provably does not work stays impossible."""
    _make_skill(skills_env, "broken", selftest="import sys; sys.exit(1)")
    assert launch._approve_all_skills("jyunming") == 0
    assert "broken" not in _ledger(tmp_path)


def test_bulk_refuses_skill_without_selftest(
    skills_env: Path, tmp_path: Path,
) -> None:
    _make_skill(skills_env, "untested", selftest=None)
    assert launch._approve_all_skills("jyunming") == 0
    assert "untested" not in _ledger(tmp_path)


def test_bulk_approves_the_good_and_skips_the_bad(
    skills_env: Path, tmp_path: Path, capsys,
) -> None:
    _make_skill(skills_env, "good")
    _make_skill(skills_env, "broken", selftest="import sys; sys.exit(1)")
    _make_skill(skills_env, "untested", selftest=None)
    assert launch._approve_all_skills("jyunming") == 0
    led = _ledger(tmp_path)
    assert "good" in led and "broken" not in led and "untested" not in led
    out = capsys.readouterr().out
    assert "Approved 1" in out


def test_skip_list_is_honoured(skills_env: Path, tmp_path: Path) -> None:
    _make_skill(skills_env, "keep")
    _make_skill(skills_env, "leave-alone")
    assert launch._approve_all_skills("jyunming", skip="leave-alone") == 0
    led = _ledger(tmp_path)
    assert "keep" in led and "leave-alone" not in led


# ---------------------------------------------------- dependency discovery

def test_pip_names_from_declared_provenance(skills_env: Path) -> None:
    from core.skills import discover
    _make_skill(skills_env, "declared",
                provenance={"pip_requires": ["scikit-learn", "arviz"]})
    sk = next(s for s in discover() if s.name == "declared")
    assert launch._pip_names_for_skill(sk) == ["scikit-learn", "arviz"]


def test_pip_names_inferred_from_missing_module(skills_env: Path) -> None:
    """Skills imported before pip_requires existed carry no declaration, so
    the failing self-test's own traceback is the fallback source."""
    from core.skills import discover
    _make_skill(skills_env, "inferred")
    sk = next(s for s in discover() if s.name == "inferred")
    out = "ModuleNotFoundError: No module named 'qutip'"
    assert launch._pip_names_for_skill(sk, out) == ["qutip"]


def test_module_name_is_mapped_to_package_name(skills_env: Path) -> None:
    """A traceback reports the IMPORT name, which is not always installable:
    `pip install sklearn` installs a deprecation stub, not scikit-learn."""
    from core.skills import discover
    _make_skill(skills_env, "aliased")
    sk = next(s for s in discover() if s.name == "aliased")
    out = "ModuleNotFoundError: No module named 'sklearn'"
    assert launch._pip_names_for_skill(sk, out) == ["scikit-learn"]


def test_declared_and_inferred_are_merged(skills_env: Path) -> None:
    from core.skills import discover
    _make_skill(skills_env, "both", provenance={"pip_requires": ["arviz"]})
    sk = next(s for s in discover() if s.name == "both")
    got = launch._pip_names_for_skill(sk, "No module named 'cv2'")
    assert got == ["arviz", "opencv-python"]


def test_pip_install_is_opt_in(skills_env: Path, monkeypatch, capsys) -> None:
    """Installing into a possibly-shared interpreter is a real side effect,
    so without the flag FI prints the command instead of running it."""
    _make_skill(skills_env, "needs-dep",
                selftest="raise ModuleNotFoundError(\"No module named 'qutip'\")")
    called = _spy_on_pip(monkeypatch)
    launch._approve_all_skills("jyunming")          # no pip_install
    assert not called
    assert "pip install" in capsys.readouterr().out


def test_pip_install_runs_before_approval(skills_env: Path, monkeypatch) -> None:
    """Order is the whole point: approve-first would hit the QUARANTINED
    refusal before an install could fix it."""
    _make_skill(skills_env, "needs-dep",
                selftest="raise ModuleNotFoundError(\"No module named 'qutip'\")")
    called = _spy_on_pip(monkeypatch)
    launch._approve_all_skills("jyunming", pip_install=True)
    assert called, "pip was never invoked"
    assert "qutip" in called[0]
    assert called[0][1:3] == ["-m", "pip"]
