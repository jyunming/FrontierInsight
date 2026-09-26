"""``scripts/import_scientist_skills.py``: it has to parse, and fetching a source repository must never hang.

The script sat unable to compile for days (four ``print`` strings carried real line breaks), and on a company
laptop it sat for ever in ``git pull`` at the first repository: ``subprocess.run(capture_output=True)`` with no
timeout waits for the pipe to close, and a stalled connection, a credential prompt nobody sees, or a helper
git left behind after a kill keeps it open. These pin what replaced that: git never asks for anything, its
output goes to a file, a command that runs too long is stopped with everything it started, and a repository
that cannot be fetched is named and skipped while the rest is imported.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "import_scientist_skills.py"
GIT = shutil.which("git")


@pytest.fixture()
def isk():
    spec = importlib.util.spec_from_file_location("import_scientist_skills_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(spec.name, None)


def test_the_script_parses() -> None:
    compile(SCRIPT.read_text(encoding="utf-8"), str(SCRIPT), "exec")


# --- the runner: git must not be able to hang ---------------------------------


def test_git_is_never_allowed_to_ask_and_a_stalled_transfer_is_dropped(isk) -> None:
    code = (
        "import os; print(os.environ['GIT_TERMINAL_PROMPT'], os.environ['GCM_INTERACTIVE'], "
        "os.environ['GIT_HTTP_LOW_SPEED_LIMIT'], os.environ['GIT_HTTP_LOW_SPEED_TIME'])"
    )
    done = isk._git([sys.executable, "-c", code], timeout=30)
    assert done.ok and done.output.split() == ["0", "never", "1000", "60"]


def test_a_command_that_waits_for_input_gets_none_and_fails_at_once(isk) -> None:
    started = time.time()
    done = isk._git([sys.executable, "-c", "input('Username: ')"], timeout=30)
    assert not done.ok and not done.timed_out and time.time() - started < 20
    assert "EOFError" in done.output


def test_a_command_that_runs_too_long_is_stopped_with_the_helper_it_started(isk, tmp_path: Path) -> None:
    """``git-remote-https`` is a child of ``git`` and holds its output handle: with a pipe, the wait for
    the output never ends after git is killed. Here the helper writes a marker if it is left alive."""
    marker = tmp_path / "helper-alive"
    # The timeout leaves the parent time to have started its helper before it is stopped: a stop that lands while
    # the helper is still being created cannot see it (a slow Windows runner took over 2 s to get there, twice).
    parent = textwrap.dedent(
        """
        import subprocess, sys, time
        subprocess.Popen([sys.executable, "-c",
            "import sys, time, pathlib; time.sleep(10); pathlib.Path(sys.argv[1]).write_text('alive')", sys.argv[1]])
        time.sleep(60)
        """
    )
    started = time.time()
    done = isk._git([sys.executable, "-c", parent, str(marker)], timeout=6)
    assert done.timed_out and not done.ok and done.reason() == "timed out"
    assert time.time() - started < 20
    time.sleep(12)  # long enough for a helper that was not stopped to have written its marker
    assert not marker.exists()


def test_a_missing_git_is_a_message_and_not_a_traceback(isk) -> None:
    done = isk._git(["git-that-is-not-installed-anywhere", "--version"], timeout=5)
    assert not done.ok and done.returncode == 127 and "was not found on PATH" in done.output


def test_the_reason_is_the_last_line_git_wrote(isk) -> None:
    done = isk._git(
        [sys.executable, "-c", "import sys; print('a'); print('fatal: unable to access the remote', file=sys.stderr); sys.exit(128)"],
        timeout=30,
    )
    assert done.returncode == 128 and done.reason() == "fatal: unable to access the remote"


# --- cloning and refreshing, against a real local repository -------------------


def _source_repo(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    (src / "skills" / "pymc").mkdir(parents=True)
    (src / "skills" / "pymc" / "SKILL.md").write_text("# pymc\n", encoding="utf-8")
    (src / "skills" / "other").mkdir()
    (src / "skills" / "other" / "SKILL.md").write_text("# other\n", encoding="utf-8")

    def git(*args: str) -> None:
        subprocess.run([GIT, "-C", str(src), *args], check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    git("add", "-A")
    git("commit", "-q", "-m", "one")
    return src


def _commit(src: Path, name: str) -> None:
    (src / "skills" / "pymc" / name).write_text("x\n", encoding="utf-8")
    subprocess.run([GIT, "-C", str(src), "add", "-A"], check=True, capture_output=True)
    subprocess.run([GIT, "-C", str(src), "commit", "-q", "-m", name], check=True, capture_output=True)


needs_git = pytest.mark.skipif(GIT is None, reason="needs git")


@needs_git
def test_a_first_clone_checks_out_only_the_folders_the_script_imports(isk, tmp_path: Path) -> None:
    src = _source_repo(tmp_path)
    dest = tmp_path / "cache" / "kdense"
    assert isk._clone_or_refresh(src.as_uri(), dest, ["skills/pymc"], timeout=120) == (True, "")
    assert (dest / "skills" / "pymc" / "SKILL.md").is_file()


@needs_git
def test_a_refresh_brings_in_what_upstream_added(isk, tmp_path: Path) -> None:
    src = _source_repo(tmp_path)
    dest = tmp_path / "cache" / "kdense"
    isk._clone_or_refresh(src.as_uri(), dest, ["skills/pymc"], timeout=120)
    _commit(src, "new.txt")
    assert isk._clone_or_refresh(src.as_uri(), dest, ["skills/pymc"], timeout=120) == (True, "")
    assert (dest / "skills" / "pymc" / "new.txt").is_file()


@needs_git
def test_a_clone_stopped_before_its_folders_were_checked_out_is_finished_on_the_next_run(isk, tmp_path: Path) -> None:
    """A first run interrupted between the clone and the sparse-checkout step left a clone that a later run
    only pulled: the folders to import from were never there."""
    src = _source_repo(tmp_path)
    dest = tmp_path / "cache" / "kdense"
    subprocess.run([GIT, "clone", "-q", "--depth", "1", "--sparse", src.as_uri(), str(dest)], check=True, capture_output=True)
    assert not (dest / "skills" / "pymc" / "SKILL.md").exists()
    assert isk._clone_or_refresh(src.as_uri(), dest, ["skills/pymc"], timeout=120) == (True, "")
    assert (dest / "skills" / "pymc" / "SKILL.md").is_file()


@needs_git
def test_a_first_clone_that_fails_leaves_nothing_behind_to_be_mistaken_for_a_clone(isk, tmp_path: Path) -> None:
    dest = tmp_path / "cache" / "kdense"
    usable, why = isk._clone_or_refresh((tmp_path / "no-such-repo").as_uri(), dest, ["skills/pymc"], timeout=60)
    assert usable is False and why
    assert not dest.exists()


def _hanging_git(tmp_path: Path) -> list[str]:
    fake = tmp_path / "hang.py"
    fake.write_text("import time\ntime.sleep(60)\n", encoding="utf-8")
    return [sys.executable, str(fake)]


def test_a_clone_that_stalls_is_given_up_on_and_removed(isk, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(isk, "GIT", _hanging_git(tmp_path))
    dest = tmp_path / "cache" / "kdense"
    started = time.time()
    usable, why = isk._clone_or_refresh("https://example.invalid/x.git", dest, ["skills/pymc"], timeout=2)
    assert (usable, why) == (False, "timed out") and time.time() - started < 15
    assert not dest.exists()


def test_a_refresh_that_stalls_keeps_the_clone_that_is_already_there(isk, tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(isk, "GIT", _hanging_git(tmp_path))
    dest = tmp_path / "cache" / "kdense"
    (dest / ".git").mkdir(parents=True)
    started = time.time()
    assert isk._clone_or_refresh("https://example.invalid/x.git", dest, None, timeout=2) == (True, "")
    assert time.time() - started < 15
    assert "pull failed, timed out; using the existing clone as-is" in capsys.readouterr().out


def test_a_refresh_that_stalls_before_its_folders_are_there_is_not_usable(isk, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(isk, "GIT", _hanging_git(tmp_path))
    dest = tmp_path / "cache" / "kdense"
    (dest / ".git").mkdir(parents=True)
    usable, why = isk._clone_or_refresh("https://example.invalid/x.git", dest, ["skills/pymc"], timeout=1)
    assert usable is False and "could not be checked out" in why
    # ... but folders that are already checked out are enough
    (dest / "skills" / "pymc").mkdir(parents=True)
    assert isk._clone_or_refresh("https://example.invalid/x.git", dest, ["skills/pymc"], timeout=1) == (True, "")


def test_offline_touches_no_network_and_says_what_is_missing(isk, tmp_path: Path, monkeypatch) -> None:
    def refuse(*a, **k):
        raise AssertionError("git was run with --offline")

    monkeypatch.setattr(isk, "_git", refuse)
    (tmp_path / "kdense" / ".git").mkdir(parents=True)
    assert isk._clone_or_refresh("u", tmp_path / "kdense", None, offline=True) == (True, "")
    usable, why = isk._clone_or_refresh("u", tmp_path / "clawbio", None, offline=True)
    assert usable is False and "--offline" in why


# --- the whole script ---------------------------------------------------------


def _run_main(isk, monkeypatch, cache: Path, *extra: str, imported: list[str] | None = None) -> int:
    import launch

    def fake_import(source, name, **kw):
        if imported is not None:
            imported.append(name)
        return 0 if Path(source).exists() else 1

    monkeypatch.setattr(launch, "_import_skill", fake_import)
    monkeypatch.setattr(launch, "_force_utf8_streams", lambda: None)
    monkeypatch.setattr(sys, "argv", ["import_scientist_skills.py", "--offline", "--cache-dir", str(cache), *extra])
    return isk.main()


def test_repositories_that_cannot_be_fetched_are_named_and_only_their_skills_are_skipped(
    isk, tmp_path: Path, monkeypatch, capsys,
) -> None:
    (tmp_path / "kdense" / ".git").mkdir(parents=True)
    (tmp_path / "kdense" / "skills" / "pymc").mkdir(parents=True)
    (tmp_path / "kdense" / "skills" / "pymc" / "SKILL.md").write_text("# pymc\n", encoding="utf-8")
    imported: list[str] = []
    code = _run_main(isk, monkeypatch, tmp_path, imported=imported)
    out = capsys.readouterr().out
    assert code == 1
    assert f"Could not fetch {len(isk.REPOS) - 1} of {len(isk.REPOS)} source repos" in out and "clawbio" in out
    # and it says exactly what to run by hand
    assert f"git -c core.longpaths=true clone --depth 1 https://github.com/ClawBio/ClawBio.git {tmp_path / 'clawbio'}" in out
    assert "analyze-fasta: skipped (its source repo clawbio was not fetched)" in out
    assert "pymc" in imported and "analyze-fasta" not in imported
    kdense = sum(1 for repo, *_ in isk.SKILLS.values() if repo == "kdense")
    assert f"Imported 1/{kdense}" in out and "Not attempted, because their source repo could not be fetched" in out
    # the closing lines print, as text and not as a syntax error
    assert "Replace <you> with the name of whoever actually reviewed the skills." in out
    assert "Or one at a time:" in out


def test_when_everything_is_there_the_exit_code_is_zero(isk, tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(isk, "REPOS", {"kdense": "u"})
    monkeypatch.setattr(isk, "SKILLS", {"pymc": ("kdense", "skills/pymc", False)})
    (tmp_path / "kdense" / ".git").mkdir(parents=True)
    (tmp_path / "kdense" / "skills" / "pymc").mkdir(parents=True)
    (tmp_path / "kdense" / "skills" / "pymc" / "SKILL.md").write_text("# pymc\n", encoding="utf-8")
    assert _run_main(isk, monkeypatch, tmp_path) == 0
    out = capsys.readouterr().out
    assert "Could not fetch" not in out and "Imported 1/1" in out
    assert "python launch.py --approve-skill pymc --approve-as <you>" in out


def test_the_cache_stays_out_of_a_onedrive_folder(isk, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(isk, "REPO_ROOT", tmp_path / "OneDrive - ACME" / "Documents" / "FrontierInsight")
    assert isk._default_cache_dir() == Path.home() / ".frontier-insight" / "skill-sources"
    monkeypatch.setattr(isk, "REPO_ROOT", tmp_path / "dev" / "FrontierInsight")
    assert isk._default_cache_dir() == tmp_path / "dev" / "FrontierInsight" / ".skill-sources"
