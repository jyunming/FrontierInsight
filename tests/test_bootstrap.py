"""``launch.py``'s self-bootstrap: a missing dependency offers to install into (or switch to) an interpreter
that has it, instead of a bare traceback. ``launch._bootstrap_or_reraise`` is the only function under test
here — the surrounding ``try: from core.config import Config ... except ImportError:`` at module top level
needs no test of its own (it runs, harmlessly, on every normal `import launch`, and always takes the success
branch since this test environment has every dependency installed).

The three routes the function can take, and the fixture/monkeypatch each test uses to land on one:

- **Silent relaunch** — bare Python (``sys.prefix == sys.base_prefix``, true by default for the interpreter
  running these tests — confirmed once in ``test_this_interpreter_is_not_itself_a_venv``), ``.venv/`` already
  exists and isn't stale. No banner, no prompt, no install. The ``repo`` fixture alone lands here once a
  ``.venv/`` stub is added.
- **Ask + install into ``.venv/``** — bare Python, no ``.venv/`` yet (the ``repo`` fixture's default state), or
  one that turns out to still be missing something (simulated by also setting ``FI_BOOTSTRAPPED`` first, which
  skips straight past the silent-relaunch branch the same way a second real attempt would).
- **Ask + install into the active environment** — ``sys.prefix != sys.base_prefix`` (the ``in_custom_venv``
  fixture), never touches ``.venv/`` at all.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import launch


def _missing(name: str = "yaml") -> ImportError:
    return ImportError(f"No module named '{name}'")


def test_this_interpreter_is_not_itself_a_venv() -> None:
    """A precondition the other tests below rely on implicitly: if the interpreter running the suite were
    itself a venv or conda env, every "bare Python" test in this file would silently exercise the
    "active environment" route instead, and nothing here would say so."""
    assert sys.prefix == sys.base_prefix


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fake checkout: a real ``pyproject.toml`` (so bootstrapping applies), no ``.venv/`` yet, bare Python."""
    (tmp_path / "pyproject.toml").write_text("[project]\nname = \"x\"\n", encoding="utf-8")
    monkeypatch.setattr(launch, "_REPO_ROOT", tmp_path)
    monkeypatch.delenv("FI_SKIP_BOOTSTRAP", raising=False)
    monkeypatch.delenv("FI_BOOTSTRAPPED", raising=False)
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.delenv("CONDA_DEFAULT_ENV", raising=False)
    return tmp_path


@pytest.fixture
def in_custom_venv(monkeypatch: pytest.MonkeyPatch) -> Path:
    """A venv or conda env *activated* by the person (``CONDA_DEFAULT_ENV`` set, the marker an activation
    script sets — not merely "this interpreter happens to be a venv", which is also true of FI's own
    ``.venv/`` once relaunched into directly without ever being activated) — distinct from ``.venv/``. A
    plausible fake path is enough; nothing under it is ever actually read or written in these mocked tests."""
    fake = Path("/home/x/.conda/envs/myenv/bin/python")
    monkeypatch.setattr(sys, "executable", str(fake))
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "myenv")
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    return fake


def test_fi_skip_bootstrap_reraises_without_touching_anything(
    repo: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FI_SKIP_BOOTSTRAP", "1")
    run = MagicMock()
    monkeypatch.setattr(subprocess, "run", run)
    exc = _missing()
    with pytest.raises(ImportError) as info:
        launch._bootstrap_or_reraise(exc)
    assert info.value is exc
    run.assert_not_called()


def test_no_pyproject_toml_reraises_as_an_installed_package_would(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No ``pyproject.toml`` next to ``launch.py`` means this is an installed package with no source tree to
    bootstrap from (e.g. a plain ``pip install frontier-insight`` elsewhere) — never applicable, not just off."""
    monkeypatch.setattr(launch, "_REPO_ROOT", tmp_path)  # empty dir, no pyproject.toml
    monkeypatch.delenv("FI_SKIP_BOOTSTRAP", raising=False)
    exc = _missing()
    with pytest.raises(ImportError) as info:
        launch._bootstrap_or_reraise(exc)
    assert info.value is exc


def _expect_relaunch(monkeypatch: pytest.MonkeyPatch, plat: str) -> tuple[MagicMock, MagicMock]:
    """Arrange for, then let the caller assert on, the platform-specific relaunch mechanism: POSIX's
    ``os.execv`` genuinely replaces the process; Windows' does not (see
    ``test_windows_relaunch_propagates_...`` below), so there the code spawns a real child and exits with its
    returncode instead. Returns ``(run, execv)`` — one of the two is how THIS platform relaunches; the other
    must stay untouched."""
    monkeypatch.setattr(sys, "platform", plat)
    execv = MagicMock()
    monkeypatch.setattr("os.execv", execv)
    run = MagicMock(return_value=MagicMock(returncode=0))
    monkeypatch.setattr(subprocess, "run", run)
    return run, execv


# ---- silent relaunch: an existing, complete .venv/ reached from bare Python -------------------------------


@pytest.mark.parametrize("plat", ["win32", "linux"])
def test_an_existing_venv_is_used_silently_no_banner_no_install(
    repo: Path, monkeypatch: pytest.MonkeyPatch, plat: str,
) -> None:
    """The actual bug the advisor caught in the first version of this feature: every later run through bare
    Python re-asked and re-ran ``pip install -e .`` (a fast no-op, but still on every single run, and still an
    unwanted prompt in a real terminal) instead of "only the first" time. Once ``.venv/`` already has a
    complete environment, reaching it again must cost nothing but the relaunch itself."""
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)  # would ask, if it got that far — it must not
    run, execv = _expect_relaunch(monkeypatch, plat)
    venv_python = launch._venv_python(repo / ".venv")  # reads sys.platform — after _expect_relaunch patches it
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("", encoding="utf-8")

    if plat == "win32":
        with pytest.raises(SystemExit) as info:
            launch._bootstrap_or_reraise(_missing())
        assert info.value.code == 0
        run.assert_called_once()  # the relaunch itself; no venv-creation call, no pip-install call
        assert run.call_args.args[0][0] == str(venv_python)
    else:
        launch._bootstrap_or_reraise(_missing())
        run.assert_not_called()
        execv.assert_called_once()
        assert execv.call_args.args[0] == str(venv_python)


@pytest.mark.parametrize("plat", ["win32", "linux"])
def test_a_broken_venv_python_falls_through_to_the_normal_flow_instead_of_crashing(
    repo: Path, monkeypatch: pytest.MonkeyPatch, plat: str,
) -> None:
    """``.venv/python`` exists as a path but won't spawn on the silent-relaunch attempt (corrupted, locked,
    deleted mid-write) — the resulting ``OSError`` must fall through to the normal ask+install flow rather
    than propagating as an unhandled crash. It does NOT recreate ``.venv/`` on its own: since the path still
    exists, the fallback flow (reasonably) trusts it and runs ``pip install`` through it rather than rebuilding
    the venv from scratch — self-healing a venv that is broken in a way ``pip install`` alone cannot fix is
    out of scope, and this test's own mock lets that second attempt succeed, standing in for a merely
    transient failure (a lock briefly held, a file mid-write) rather than a permanently corrupted install."""
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr(sys, "platform", plat)
    venv_python = launch._venv_python(repo / ".venv")
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("", encoding="utf-8")
    # Only the FIRST relaunch attempt (the silent one, against the broken .venv/python) must fail; the later,
    # legitimate relaunch after a fresh install succeeds normally, same as any other test in this file.
    broken_once = {"used": False}

    def _execv(*_a: object, **_kw: object) -> None:
        if not broken_once["used"]:
            broken_once["used"] = True
            raise OSError("not a valid Win32 application")

    monkeypatch.setattr("os.execv", MagicMock(side_effect=_execv))

    def _run(argv: list[str], **_kw: object) -> MagicMock:
        # The silent relaunch, on Windows, goes through subprocess.run too -- that's the call that must fail
        # here, exactly like a broken os.execv does on POSIX; the later, genuine pip-install and relaunch
        # calls must not.
        if len(argv) > 1 and str(argv[1]).endswith("launch.py") and not broken_once["used"]:
            broken_once["used"] = True
            raise OSError("not a valid Win32 application")
        return MagicMock(returncode=0)

    run = MagicMock(side_effect=_run)
    monkeypatch.setattr(subprocess, "run", run)
    if plat == "win32":
        with pytest.raises(SystemExit) as info:
            launch._bootstrap_or_reraise(_missing())
        assert info.value.code == 0
    else:
        launch._bootstrap_or_reraise(_missing())
    # Falls through to the normal flow: since the stub file already exists, only pip install runs (not
    # venv-creation) -- this test is about the OSError being caught, not about venv recreation specifically.
    calls = [c for c in run.call_args_list if c.args[0][1:5] == ["-m", "pip", "install", "-q"]]
    assert len(calls) == 1


def test_bootstrapped_guard_skips_the_silent_relaunch_too(
    repo: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FI_BOOTSTRAPPED set means a full install-and-relaunch already happened once in this exec chain; a
    repeat failure must not even try the silent relaunch again (which would just recreate the same loop) --
    straight back to the original error."""
    venv_python = launch._venv_python(repo / ".venv")
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("", encoding="utf-8")
    monkeypatch.setenv("FI_BOOTSTRAPPED", "1")
    run = MagicMock()
    monkeypatch.setattr(subprocess, "run", run)
    exc = _missing()
    with pytest.raises(ImportError) as info:
        launch._bootstrap_or_reraise(exc)
    assert info.value is exc
    run.assert_not_called()


@pytest.mark.parametrize("plat", ["win32", "linux"])
def test_a_stale_venv_reached_via_the_silent_relaunch_does_not_relaunch_into_itself_again(
    repo: Path, monkeypatch: pytest.MonkeyPatch, plat: str,
) -> None:
    """The actual infinite-loop this design has to rule out: bare Python silently relaunches into an
    existing ``.venv/`` (no ``FI_BOOTSTRAPPED`` set yet — that flag is only ever set by a full install, and
    the silent relaunch is not one); if ``.venv/`` turns out to be stale too (a ``git pull`` added a
    dependency this ``.venv/`` predates), THAT interpreter's own call into this same function must recognise
    it is already running from ``.venv/``'s own python and fall to the full ask+install flow -- not attempt
    another silent relaunch into the exact same, still-broken interpreter it is already running as, forever.
    ``sys.executable`` is set to ``.venv/``'s own path to simulate being that interpreter."""
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    run, execv = _expect_relaunch(monkeypatch, plat)
    # _venv_python reads sys.platform itself -- compute the stub's path, and set sys.executable to match,
    # only AFTER _expect_relaunch patches the platform above.
    venv_python = launch._venv_python(repo / ".venv")
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("", encoding="utf-8")
    monkeypatch.setattr(sys, "executable", str(venv_python))

    if plat == "win32":
        with pytest.raises(SystemExit) as info:
            launch._bootstrap_or_reraise(_missing())
        assert info.value.code == 0
        # Not the silent path (no venv-creation call either, since venv_python already exists): exactly one
        # pip-install call, then the relaunch itself -- two subprocess.run calls, not a third for `-m venv`.
        assert run.call_count == 2
        pip_call, _relaunch_call = run.call_args_list
    else:
        launch._bootstrap_or_reraise(_missing())
        assert run.call_count == 1
        (pip_call,) = run.call_args_list
        execv.assert_called_once()
        assert execv.call_args.args[0] == str(venv_python)
    assert pip_call.args[0][:5] == [str(venv_python), "-m", "pip", "install", "-q"]
    import os as os_mod
    assert os_mod.environ.get("FI_BOOTSTRAPPED") == "1"
    os_mod.environ.pop("FI_BOOTSTRAPPED", None)  # see the loop-guard test above for why this cleanup matters


# ---- ask + install: no .venv/ yet, or already inside one that's stale -------------------------------------


@pytest.mark.parametrize("plat", ["win32", "linux"])
def test_non_interactive_proceeds_without_asking(repo: Path, monkeypatch: pytest.MonkeyPatch, plat: str) -> None:
    """No TTY (CI, a piped command): there is no one to answer a prompt, so it installs without asking."""
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    run, execv = _expect_relaunch(monkeypatch, plat)
    if plat == "win32":
        with pytest.raises(SystemExit) as info:
            launch._bootstrap_or_reraise(_missing())
        assert info.value.code == 0
        assert run.call_count == 3, "creates the venv, installs into it, then relaunches as a real child"
        venv_call, pip_call, _relaunch_call = run.call_args_list
        execv.assert_not_called()
    else:
        launch._bootstrap_or_reraise(_missing())
        assert run.call_count == 2, "creates the venv, then installs into it"
        venv_call, pip_call = run.call_args_list
        execv.assert_called_once()
        assert execv.call_args.args[0] == str(launch._venv_python(repo / ".venv"))
    assert venv_call.args[0][1:3] == ["-m", "venv"]
    assert pip_call.args[0][1:5] == ["-m", "pip", "install", "-q"]


@pytest.mark.parametrize("plat", ["win32", "linux"])
def test_interactive_blank_answer_defaults_to_yes(repo: Path, monkeypatch: pytest.MonkeyPatch, plat: str) -> None:
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt: "")
    _run, execv = _expect_relaunch(monkeypatch, plat)
    if plat == "win32":
        with pytest.raises(SystemExit):
            launch._bootstrap_or_reraise(_missing())
        execv.assert_not_called()
    else:
        launch._bootstrap_or_reraise(_missing())
        execv.assert_called_once()


@pytest.mark.parametrize("answer", ["n", "N", "no", "anything else"])
def test_interactive_decline_prints_manual_steps_and_exits_without_installing(
    repo: Path, monkeypatch: pytest.MonkeyPatch, answer: str, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt: answer)
    run = MagicMock()
    monkeypatch.setattr(subprocess, "run", run)
    execv = MagicMock()
    monkeypatch.setattr("os.execv", execv)
    with pytest.raises(SystemExit) as info:
        launch._bootstrap_or_reraise(_missing())
    assert info.value.code == 1
    run.assert_not_called()
    execv.assert_not_called()
    out = capsys.readouterr().out
    assert "pip install -e ." in out and "python -m venv .venv" in out


def test_ctrl_c_at_the_prompt_declines(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(_prompt: str) -> str:
        raise KeyboardInterrupt

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", _raise)
    run = MagicMock()
    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(SystemExit):
        launch._bootstrap_or_reraise(_missing())
    run.assert_not_called()


@pytest.mark.parametrize("plat", ["win32", "linux"])
def test_eof_at_the_prompt_proceeds_like_non_interactive(
    repo: Path, monkeypatch: pytest.MonkeyPatch, plat: str,
) -> None:
    """``isatty()`` can say yes with nothing actually there to read (Windows reports stdin redirected to NUL
    as a TTY, unlike POSIX's /dev/null) -- an EOF on the prompt must not read as a human declining, or exactly
    the unattended case this feature exists for would be the one case that refuses to run."""
    def _raise(_prompt: str) -> str:
        raise EOFError

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", _raise)
    _run, execv = _expect_relaunch(monkeypatch, plat)
    if plat == "win32":
        with pytest.raises(SystemExit):
            launch._bootstrap_or_reraise(_missing())
        execv.assert_not_called()
    else:
        launch._bootstrap_or_reraise(_missing())
        execv.assert_called_once()


def test_install_failure_prints_manual_steps_and_exits(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    run = MagicMock(side_effect=subprocess.CalledProcessError(1, ["pip"]))
    monkeypatch.setattr(subprocess, "run", run)
    execv = MagicMock()
    monkeypatch.setattr("os.execv", execv)
    with pytest.raises(SystemExit) as info:
        launch._bootstrap_or_reraise(_missing())
    assert info.value.code == 1
    execv.assert_not_called()


@pytest.mark.parametrize("plat", ["win32", "linux"])
def test_a_successful_install_sets_the_loop_guard_before_relaunch(
    repo: Path, monkeypatch: pytest.MonkeyPatch, plat: str,
) -> None:
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    _run, _execv = _expect_relaunch(monkeypatch, plat)
    monkeypatch.delenv("FI_BOOTSTRAPPED", raising=False)
    import os as os_mod
    try:
        if plat == "win32":
            with pytest.raises(SystemExit):
                launch._bootstrap_or_reraise(_missing())
        else:
            launch._bootstrap_or_reraise(_missing())
        assert os_mod.environ.get("FI_BOOTSTRAPPED") == "1"
    finally:
        # The code under test sets this directly on the real, process-wide os.environ (the relaunch inherits
        # it, whichever mechanism the current platform uses) rather than through monkeypatch, so monkeypatch's
        # own teardown never sees it and it would otherwise leak into every later test in this process --
        # including, concretely, the real subprocess spawned by the end-to-end test below, which inherits
        # this process's environment.
        os_mod.environ.pop("FI_BOOTSTRAPPED", None)


def test_windows_relaunch_propagates_the_real_exit_code_not_always_zero(
    repo: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The actual bug this platform branch exists to fix: ``os.execv`` on Windows does not replace the
    process the way POSIX's does — the CRT's emulation returns control to whatever spawned this command
    (with exit code 0, unconditionally) while the "replacement" keeps running as a separate process, so a
    caller waiting on the real outcome (a CI step checking the exit code, a script testing %ERRORLEVEL%) sees
    success immediately no matter what the relaunched command actually does. Confirmed by hand against a
    genuine two-process os.execv chain on this machine before writing this fix. A relaunch that itself exits
    non-zero must make this process exit with that same non-zero code, not silently report success."""
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr("os.execv", MagicMock())

    def _run(argv: list[str], **_kw: object) -> MagicMock:
        # The venv-creation and pip-install calls succeed; only the relaunch (recognisable by argv[1] being
        # this same launch.py, not "-m") reports the quest's own, non-zero exit code.
        if len(argv) > 1 and str(argv[1]).endswith("launch.py"):
            return MagicMock(returncode=17)
        return MagicMock(returncode=0)

    monkeypatch.setattr(subprocess, "run", MagicMock(side_effect=_run))
    with pytest.raises(SystemExit) as info:
        launch._bootstrap_or_reraise(_missing())
    assert info.value.code == 17


def test_windows_relaunch_turns_a_ctrl_c_into_130_not_a_traceback(
    repo: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ctrl-C during the Windows relaunch's own wait reaches both processes; the child's own handler already
    turns that into the conventional 130 (see ``main()``'s KeyboardInterrupt handling) -- the parent doing
    the waiting must match it, not dump a traceback about the wait itself being interrupted."""
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(subprocess, "run", MagicMock(side_effect=KeyboardInterrupt))
    with pytest.raises(SystemExit) as info:
        launch._relaunch(Path("/x/python"), [])
    assert info.value.code == 130


# ---- installing into an already-activated venv/conda env, never .venv/ -------------------------------------


@pytest.mark.parametrize("plat", ["win32", "linux"])
def test_a_custom_activated_venv_is_installed_into_in_place_not_switched_away_from(
    repo: Path, in_custom_venv: Path, monkeypatch: pytest.MonkeyPatch, plat: str,
) -> None:
    """agy's finding: a person who activated their own venv or conda env (GPU-enabled, project-specific,
    whatever) before a dependency happened to be missing must not be silently switched to a ``.venv/`` they
    never asked for. Installed into the interpreter that's actually running, in place; no ``.venv/`` created,
    no relaunch to a different interpreter."""
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    run, execv = _expect_relaunch(monkeypatch, plat)
    if plat == "win32":
        with pytest.raises(SystemExit) as info:
            launch._bootstrap_or_reraise(_missing())
        assert info.value.code == 0
        pip_call, relaunch_call = run.call_args_list
        assert relaunch_call.args[0][0] == str(in_custom_venv)
        execv.assert_not_called()
    else:
        launch._bootstrap_or_reraise(_missing())
        (pip_call,) = run.call_args_list
        execv.assert_called_once()
        assert execv.call_args.args[0] == str(in_custom_venv)
    assert pip_call.args[0][:5] == [str(in_custom_venv), "-m", "pip", "install", "-q"]
    assert not (repo / ".venv").exists()


def test_a_custom_activated_venv_missing_something_still_asks_and_can_be_declined(
    repo: Path, in_custom_venv: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")
    run = MagicMock()
    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(SystemExit) as info:
        launch._bootstrap_or_reraise(_missing())
    assert info.value.code == 1
    run.assert_not_called()
    out = capsys.readouterr().out
    assert "python -m venv" not in out, "never suggests creating .venv/ for an environment you already activated"
    assert str(in_custom_venv) in out


def test_venv_python_path_matches_platform_convention() -> None:
    venv_dir = Path("/x/.venv")
    p = launch._venv_python(venv_dir)
    assert p.name == "python.exe" or p.name == "python"
    assert ("Scripts" in p.parts) or ("bin" in p.parts)


def test_same_interpreter_does_not_follow_symlinks_to_a_shared_base_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real bug this diff's own CI run caught (the mocked unit tests above, and the real end-to-end test on
    this developer's Windows machine, both missed it -- only a real run on Linux CI did): on POSIX, a venv's
    own `bin/python` is normally a symlink chain ending at the SAME base interpreter for every venv built
    from it (`bin/python -> python3 -> /usr/bin/python3.12`), confirmed on a real Ubuntu venv pair --
    ``os.path.samefile`` follows that chain and reports two entirely unrelated, freshly-created venvs as the
    same file. That made ``in_own_venv`` true for ANY venv sharing FI's base interpreter, not just FI's own
    ``.venv/`` -- silently skipping the loop guard's purpose and sending every later bare-Python invocation
    back through the full ask+install flow forever, exactly the "only the first time" regression this whole
    feature exists to prevent. The fix compares the given paths themselves, not what they resolve to."""
    monkeypatch.setattr(os.path, "samefile", lambda a, b: True)  # simulate two paths sharing a base interpreter
    assert launch._same_interpreter(Path("/venvs/bare/bin/python"), Path("/repo/.venv/bin/python")) is False
    assert launch._same_interpreter(Path("/repo/.venv/bin/python"), Path("/repo/.venv/bin/python")) is True


# ---- real end-to-end: a genuinely bare interpreter, a real checkout, a real `pip install -e .` -------------------


@pytest.mark.slow
def test_a_bare_python_self_bootstraps_and_runs_for_real(tmp_path: Path) -> None:
    """The thing this feature is actually for: ``git clone && python launch.py --help-all`` on an interpreter
    that has never seen this project before. No mocks — a checkout of the actual working tree (so ``.git``,
    any stray ``.venv``, and build artifacts are never in play, and a real "test before commit" run against
    uncommitted edits is exercised, not a stale last commit), a real empty venv standing in for a bare system
    Python, and a real ``pip install -e .`` triggered by the real ImportError."""
    import os

    # A clean environment for every real subprocess below: an earlier test in this same pytest process may
    # have set FI_BOOTSTRAPPED or FI_SKIP_BOOTSTRAP directly on the real os.environ (the relaunch needs it on
    # the real, process-wide environment, not a monkeypatched copy), which subprocess.run would otherwise
    # inherit and use to skip the very thing this test exists to exercise. VIRTUAL_ENV / CONDA_DEFAULT_ENV are
    # stripped too: if this test suite is itself run from inside an activated environment, that must not leak
    # into `bare_python`, which stands in for a genuinely un-activated system Python.
    _skip = ("FI_BOOTSTRAPPED", "FI_SKIP_BOOTSTRAP", "VIRTUAL_ENV", "CONDA_DEFAULT_ENV")
    clean_env = {k: v for k, v in os.environ.items() if k not in _skip}

    checkout = tmp_path / "checkout"
    checkout.mkdir()
    repo_root = Path(__file__).resolve().parent.parent
    tracked = subprocess.run(
        ["git", "ls-files", "-z"], cwd=repo_root, check=True, capture_output=True,
    ).stdout.split(b"\0")[:-1]
    for rel in tracked:
        src = repo_root / rel.decode("utf-8")
        if not src.is_file():
            continue  # a path git ls-files still lists after a local delete
        dst = checkout / rel.decode("utf-8")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
    assert (checkout / "pyproject.toml").is_file() and (checkout / "launch.py").is_file()
    assert not (checkout / ".venv").exists()

    bare = tmp_path / "bare"
    subprocess.run([sys.executable, "-m", "venv", str(bare)], check=True)
    bare_python = launch._venv_python(bare)
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run([str(bare_python), "-c", "import pydantic"], check=True, capture_output=True)

    result = subprocess.run(
        [str(bare_python), "launch.py", "--help-all"],
        cwd=checkout, capture_output=True, text=True, timeout=300, env=clean_env,
        # An empty piped stdin, not ``subprocess.DEVNULL``: on Windows, NUL itself reports as a TTY via
        # isatty() (unlike POSIX's /dev/null), which is exactly the platform gap this feature had to handle --
        # an actual pipe with no data is what a real non-interactive caller (CI, a piped command) looks like.
        input="",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "[FI] Missing a dependency" in result.stdout
    assert "[FI] Done. Continuing..." in result.stdout
    assert "--fleet" in result.stdout  # the real --help-all text, not just the bootstrap banner
    checkout_venv_python = launch._venv_python(checkout / ".venv")
    assert checkout_venv_python.is_file(), "the checkout's own .venv/ was created"

    # Run again through the SAME bare interpreter (not the checkout's own venv): the actual regression this
    # test exists for -- every later invocation through plain system Python must be silent (no banner, no
    # reinstall), not the full ask-and-install dance repeating on every single run.
    second = subprocess.run(
        [str(bare_python), "launch.py", "--help-all"],
        cwd=checkout, capture_output=True, text=True, timeout=60, env=clean_env, input="",
    )
    assert second.returncode == 0, second.stdout + second.stderr
    assert "[FI]" not in second.stdout, "a second run through bare Python must be silent, not re-ask or reinstall"
    assert "--fleet" in second.stdout
