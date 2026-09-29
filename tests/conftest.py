"""Make the repo root importable so `from core...` works inside tests."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
_NETWORK_LEAKS: dict[str, list[str]] = {}


@pytest.fixture(autouse=True)
def _no_real_network(request, monkeypatch):
    """No test reaches a real server through httpx.

    FI's retrieval code is best-effort by design: a failed request is logged
    and returns nothing. So a test that forgets to mock a request does not
    fail — it quietly depends on whether the machine is online and whether
    Crossref or Europe PMC answered, and it is slower. Such requests are
    refused here (the code under test sees a connection error, exactly as it
    would offline) and listed at the end of the run so they can be mocked.
    Loopback stays open for tests that start a local server;
    ``httpx.MockTransport`` never reaches this layer.
    ``FI_TEST_ALLOW_NETWORK=1`` lifts the block for a deliberate live run.
    """
    import os

    if os.environ.get("FI_TEST_ALLOW_NETWORK") == "1":
        return
    import httpx

    real_sync = httpx.HTTPTransport.handle_request
    real_async = httpx.AsyncHTTPTransport.handle_async_request

    def _refuse(req):
        _NETWORK_LEAKS.setdefault(request.node.nodeid, []).append(
            f"{req.method} {req.url.scheme}://{req.url.host}{req.url.path}"
        )
        return httpx.ConnectError("real network is disabled in tests", request=req)

    def handle(self, req):
        if (req.url.host or "") in _LOOPBACK_HOSTS:
            return real_sync(self, req)
        raise _refuse(req)

    async def handle_async(self, req):
        if (req.url.host or "") in _LOOPBACK_HOSTS:
            return await real_async(self, req)
        raise _refuse(req)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", handle)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", handle_async)


def pytest_terminal_summary(terminalreporter):
    if not _NETWORK_LEAKS:
        return
    terminalreporter.section("real network requests refused")
    for nodeid, urls in sorted(_NETWORK_LEAKS.items()):
        shown = sorted(set(urls))
        terminalreporter.write_line(f"{nodeid}: {len(urls)} request(s)")
        for url in shown[:5]:
            terminalreporter.write_line(f"    {url}")


@pytest.fixture(autouse=True)
def _no_embed_model_download(monkeypatch):
    """Keep the test suite hermetic: never let passage ranking download the
    sentence-transformer model (CI has no cache, and a download is slow +
    network-dependent). Force `_embed_model()` to return None so ranking
    stays on the deterministic lexical path unless a test injects its own
    (mocked) embedder. Real quests are unaffected — they load the model on
    first use."""
    import core.passages as _passages
    monkeypatch.setattr(_passages, "_EMBED_TRIED", True, raising=False)
    monkeypatch.setattr(_passages, "_EMBED_MODEL", None, raising=False)


@pytest.fixture(autouse=True)
def _isolate_skill_library(tmp_path_factory, monkeypatch):
    """No test may read the developer's real skill library.

    Skills live outside the repo, in ``FI_SKILLS_DIR`` (default
    ``~/.frontier-insight/skills``), because they are per-machine state
    rather than repository content. Nothing pointed the tests away from it,
    so every engine test discovered whatever happened to be installed —
    invisible while that was one skill, and decisive once it was 34: engine
    smoke tests began exercising real discovery, running each skill's
    self-test, and making a live selection call. Three of them failed on
    behaviour that had nothing to do with what they were testing, and the
    full sweep went from ~20 minutes to 1h55.

    A test that needs skills sets ``FI_SKILLS_DIR`` itself; the default is
    an empty directory so results depend on the repository, not the machine.
    """
    empty = tmp_path_factory.mktemp("fi_skills_empty")
    monkeypatch.setenv("FI_SKILLS_DIR", str(empty))
    # Skills other agents installed (~/.codex/skills, ~/.claude/skills, ...)
    # are read in place by default. Set-but-empty means none, so no test sees
    # the developer's own; a test that wants some sets this itself.
    monkeypatch.setenv("FI_EXTERNAL_SKILLS_DIRS", "")
    # The approval ledger lives outside the repo too, and an approval
    # recorded by a test must never reach the real one.
    monkeypatch.setenv(
        "FI_SKILLS_APPROVALS", str(empty.parent / "approvals.json"),
    )
    # So does the record of which skill contents passed their self-tests.
    # Per test, not shared: two tests building the same fake skill produce
    # the same content hash, and one test's recorded pass would answer the
    # other's self-test.
    monkeypatch.setenv(
        "FI_SKILLS_SELFTEST_CACHE",
        str(tmp_path_factory.mktemp("fi_selftest_cache") / "skill_selftest_cache.json"),
    )


@pytest.fixture(autouse=True)
def _isolate_profile(tmp_path_factory, monkeypatch):
    """No test may read or write the person's real profile (~/.frontier-insight/profile.json): a saved name there would
    change which questions an interview test is asked. Each test starts with none saved."""
    monkeypatch.setenv("FI_PROFILE_PATH", str(tmp_path_factory.mktemp("fi_profile") / "profile.json"))


@pytest.fixture(autouse=True)
def _hermetic_arxiv_gate(tmp_path_factory, monkeypatch):
    """The arXiv queue keeps its pacing state and a 24-hour response cache
    under ``FI_CACHE_DIR`` (default ``~/.frontier-insight/cache``). No test may
    read or write the developer's real cache — a stale entry there would
    answer a later test's fake request — and no test should wait three
    seconds between fake arXiv requests. Tests of the queue itself restore
    the spacing and drive a fake clock."""
    monkeypatch.setenv("FI_CACHE_DIR", str(tmp_path_factory.mktemp("fi_cache")))
    from core import arxiv_gate

    monkeypatch.setattr(arxiv_gate, "MIN_INTERVAL_S", 0.0)
    arxiv_gate._PAUSED_QUESTS.clear()


@pytest.fixture(autouse=True)
def _hermetic_pandoc_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep ``find_pandoc`` answerable by ``shutil.which`` alone in tests.

    ``generation/_pandoc.find_pandoc`` deliberately looks past PATH — into
    ``tools/pandoc[.exe]`` and then pypandoc's bundled binary — so a
    locked-down host can render paper.pdf with nothing but ``pip``. That is
    right for production and wrong for the suite: the many tests that
    simulate "no pandoc" by patching ``shutil.which`` would otherwise pass or
    fail based on whether ``pypandoc_binary`` happens to be installed on the
    machine running them, which is the same machine-dependence the
    ``FI_SKILLS_DIR`` fixture above exists to remove.

    Neutralising the pypandoc tier by default keeps ``shutil.which`` the
    single lever for those tests. Tests that specifically cover the new tiers
    call ``generation._pandoc`` helpers directly, so they are unaffected.
    """
    monkeypatch.setattr(
        "generation._pandoc._pypandoc_pandoc", lambda: None, raising=False,
    )


# ---------------------------------------------------------------------------
# Template venv for fast test execution
# ---------------------------------------------------------------------------
# Building a fresh per-quest venv from scratch takes ~10 s on CI.  Copying a
# pre-built template takes ~2.4 s.  The fixtures below build one template venv
# per Python interpreter per machine (keyed by sys.executable) and patch
# core.execution._build_venv to copy it whenever a test asks for a new venv.
#
# pytest-xdist safety: the template lives outside pytest's tmp hierarchy in a
# stable path under tempfile.gettempdir().  A FileLock on a sibling file
# ensures only one worker builds it; the rest wait and reuse it.
#
# Opt-out: tests/test_execution.py asserts real venv-creation behavior (the
# pyvenv.cfg flags, partial-build detection, python_version fallback logging,
# etc.) and must never see the stub.  Any other test that needs real venv
# creation can skip the patch by checking ``request.fspath.basename``.


def _template_venv_root() -> Path:
    """Stable per-interpreter path shared across pytest-xdist workers."""
    key = hashlib.sha256(sys.executable.encode()).hexdigest()[:12]
    return Path(tempfile.gettempdir()) / f"fi_template_venv_{key}"


def _venv_python(venv_dir: Path) -> Path:
    if sys.platform.startswith("win"):
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


@pytest.fixture(scope="session")
def _fi_template_venv() -> Path:
    """Return the path to a ready template venv, building it if needed.

    Session-scoped so it is built at most once per worker.  The FileLock
    serialises across workers so the first arrival builds and the rest wait.
    """
    from filelock import FileLock

    # Import the REAL _build_venv before any monkeypatching can reach it.
    # (Session fixtures run before function-scoped ones so this is safe, but
    # importing here rather than at module level is an extra guard.)
    from core.execution import _build_venv as _real_build_venv

    root = _template_venv_root()
    template_dir = root / "venv"
    sentinel = root / ".ready"
    lock_path = root / ".lock"

    root.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(lock_path), timeout=180)
    with lock:
        # Rebuild if the sentinel or the actual venv directory is missing.
        if not sentinel.exists() or not (template_dir / "pyvenv.cfg").exists():
            if template_dir.exists():
                shutil.rmtree(template_dir, ignore_errors=True)
            _real_build_venv(template_dir, with_pip=True, clear=False)
            # Verify the copied venv's python actually works and pip is reachable.
            py = _venv_python(template_dir)
            probe = subprocess.run(
                [str(py), "-c", "import pip; print('ok')"],
                capture_output=True, text=True, timeout=30,
            )
            if probe.returncode != 0:
                raise RuntimeError(
                    f"template venv pip probe failed (rc={probe.returncode}):\n"
                    f"{probe.stderr}"
                )
            sentinel.write_text("ok", encoding="utf-8")

    return template_dir


@pytest.fixture(autouse=True)
def _fast_build_venv(_fi_template_venv: Path, request: pytest.FixtureRequest,
                     monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch _build_venv to copy the template instead of building fresh.

    Opted out for tests/test_execution.py, which tests real venv creation.
    The copy preserves symlinks (so the ``python`` symlink in ``bin/`` still
    points to the system interpreter) and takes ~2.4 s vs ~10.7 s to build.
    VenvExecutor always calls ``python -m pip``, never the pip script, so
    hard-coded shebang paths in the copied scripts are never executed.
    """
    if request.fspath.basename == "test_execution.py":
        return

    template = _fi_template_venv

    def _copy_venv(
        venv_dir: Path,
        *,
        with_pip: bool,
        clear: bool = False,
        python_version: str = "3.11",
        system_site_packages: bool = True,
    ) -> None:
        if clear and venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)
        shutil.copytree(template, venv_dir, symlinks=True)

    monkeypatch.setattr("core.execution._build_venv", _copy_venv)
