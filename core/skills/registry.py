"""Finding skills, running their self-tests, and deciding what may load.

**FI ships no skills.** Not one, including any FI's own authors wrote.
A skill that arrives with the install is a curated library again, and
curation does not compound — the whole point is that capability is
*acquired*, from work done with this user, on this machine. So the
discovery root is user state, not repository content, and a fresh clone
starts with nothing.

Discovery has two sources, in this order:

1. ``<skills root>/<name>/`` — hand-authored and distilled skills. The
   root is ``FI_SKILLS_DIR``, else ``~/.frontier-insight/skills``. It is
   deliberately outside the repo: skills accumulate per machine and per
   person, and a skills directory inside the checkout would turn them
   into things you commit and ship.
2. The ``fi.skills`` entry-point group — skills installed from a package,
   so a library can ship its own skill without anyone editing FI. This is
   the reason the surface is pluggable at all: acquisition that requires
   a commit to FI does not compound.

A filesystem skill shadows an entry-point skill of the same name, so a
local edit always wins over an installed one.

The gate runs the self-test in a subprocess. That is not a sandbox and is
not claimed to be one — the same trust level as the venv executor FI
already uses for generated experiments. What it buys is that a self-test
which hangs, crashes the interpreter, or calls ``sys.exit`` cannot take
the quest down with it.

Loading skills for a quest (``loadable_skills``) runs the self-tests it
needs in parallel, and skips those whose pass is already recorded for the
same content, Python and installed packages — see ``selftest_cache``.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import sys
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from core.skills import approval, selftest_cache
from core.skills.base import EXTERNAL_SOURCE, SELFTEST_PY, Skill, SkillState, Status
from core.skills.selftest_cache import SelftestCache

_log = logging.getLogger("fi.skills")

ENTRY_POINT_GROUP = "fi.skills"

#: A self-test is a fast sanity check on known-good values, not a
#: benchmark. Anything slower is a defect in the test.
SELFTEST_TIMEOUT_S = 120


_ENV_SKILLS_DIR = "FI_SKILLS_DIR"


def skills_root() -> Path:
    """Where this machine's skills live.

    Outside the repository on purpose. Skills are what FI has learned
    working with one person on one machine; putting them in the checkout
    would make them something you commit, review and ship — which is the
    curated-library model this design exists to replace.
    """
    override = os.environ.get(_ENV_SKILLS_DIR, "").strip()
    if override:
        return Path(override).expanduser()
    base = os.environ.get("XDG_STATE_HOME") or os.path.expanduser("~")
    return Path(base) / ".frontier-insight" / "skills"


#: Skills other agents installed are found where they are, by looking in the
#: places those tools keep them — nobody has to name a folder. Read in place:
#: nothing is copied or converted, so a skill written for another agent is
#: searchable and selectable here without importing it.
#:
#: What is searched, on Linux, macOS and Windows alike (a folder counts only if
#: a sub-folder of it carries a SKILL.md):
#:   * every tool folder in the home directory (``~/.claude``, ``~/.codex``,
#:     ``~/.agents``, ``~/.cursor``, ... any dot-folder) for ``skills`` inside it,
#:     and its ``plugins`` tree for ``skills`` folders several levels down;
#:   * the per-OS application folders, one level of tool folders each:
#:     ``$XDG_CONFIG_HOME``, ``$XDG_DATA_HOME``, ``~/.config``, ``~/.local/share``,
#:     ``~/Library/Application Support``, ``%APPDATA%``, ``%LOCALAPPDATA%``, and
#:     the folders the tools' own variables name (``CODEX_HOME``,
#:     ``CLAUDE_CONFIG_DIR``);
#:   * the project: the current folder and its parents, for ``.<tool>/skills``
#:     (project-level skills) and, in the current folder, ``./skills``.
_TOOL_ENV_VARS = (
    "XDG_CONFIG_HOME", "XDG_DATA_HOME", "APPDATA", "LOCALAPPDATA",
    "CODEX_HOME", "CLAUDE_CONFIG_DIR",
)
_SKILLS_FOLDER_NAMES = ("skills", "skill")
_SEARCH_SKIP = frozenset({
    "node_modules", ".git", "__pycache__", ".venv", "venv", ".cache",
    "site-packages", ".tox", ".mypy_cache", ".pytest_cache",
})
_PLUGIN_SEARCH_DEPTH = 7
_PROJECT_SEARCH_LEVELS = 8
_COMMON_CACHE_TTL_S = 60.0

#: ``os.pathsep``-separated folders. Set (even empty) it is authoritative: it
#: replaces the config and the common locations, and empty means none.
_ENV_EXTERNAL_DIRS = "FI_EXTERNAL_SKILLS_DIRS"


def _is_dir(entry: "os.DirEntry[str]") -> bool:
    try:
        return entry.is_dir()
    except OSError:
        return False


def _holds_skills(folder: Path) -> bool:
    """True when some sub-folder of ``folder`` carries a SKILL.md."""
    try:
        with os.scandir(folder) as it:
            return any(_is_dir(e) and (Path(e.path) / "SKILL.md").is_file() for e in it)
    except OSError:
        return False


def _skills_folders_under(base: Path, depth: int) -> list[Path]:
    """Folders named ``skills``, at most ``depth`` levels below ``base``, that
    hold skills. ``depth`` 1 is ``base/skills`` alone. Never descends into a
    skills folder: its children are the skills."""
    found: list[Path] = []

    def walk(folder: Path, level: int) -> None:
        try:
            with os.scandir(folder) as it:
                children = sorted((e for e in it if _is_dir(e)), key=lambda e: e.name)
        except OSError:
            return
        for entry in children:
            child = Path(entry.path)
            if entry.name in _SKILLS_FOLDER_NAMES:
                if _holds_skills(child):
                    found.append(child)
                continue
            if level + 1 < depth and entry.name not in _SEARCH_SKIP:
                walk(child, level + 1)

    walk(base, 0)
    return found


def _hidden_folders(parent: Path) -> list[Path]:
    try:
        with os.scandir(parent) as it:
            return sorted(
                (Path(e.path) for e in it
                 if e.name.startswith(".") and e.name not in _SEARCH_SKIP and _is_dir(e)),
                key=lambda p: p.name,
            )
    except OSError:
        return []


def _newest_first(folders: list[Path]) -> list[Path]:
    """A plugin cache keeps one folder per version of a plugin, and the first
    skill of a name wins, so the most recently written folder goes first."""
    def mtime(p: Path) -> float:
        try:
            return p.stat().st_mtime
        except OSError:
            return 0.0
    return sorted(folders, key=mtime, reverse=True)


_common_cache: tuple[tuple, float, list[Path]] | None = None


def _common_skill_dirs() -> list[Path]:
    """The skill folders other agents keep in the usual places (see above).
    Cached briefly: many callers ask, and the answer does not change mid-run.
    The answer depends only on where and who is searching (folder, home,
    environment — the key), never on a caller's own setting, so sharing it
    between quests hands none of them another's folders."""
    global _common_cache
    home = Path.home()
    key = (str(Path.cwd()), str(home), tuple(os.environ.get(v, "") for v in _TOOL_ENV_VARS))
    now = time.monotonic()
    if _common_cache and _common_cache[0] == key and now - _common_cache[1] < _COMMON_CACHE_TTL_S:
        return list(_common_cache[2])

    found: list[Path] = []
    # 1. Tool folders in the home directory, and their plugin trees.
    for tool in _hidden_folders(home):
        found += _skills_folders_under(tool, 1)
        plugins = tool / "plugins"
        if plugins.is_dir():
            found += _newest_first(_skills_folders_under(plugins, _PLUGIN_SEARCH_DEPTH))
    # 2. Per-OS application folders: the folder itself (a tool's own variable
    #    names it) and one level of tool folders inside it.
    bases = [Path(v) for v in (os.environ.get(n) for n in _TOOL_ENV_VARS) if v]
    bases += [home / ".config", home / ".local" / "share",
              home / "Library" / "Application Support"]
    for base in bases:
        found += _skills_folders_under(base, 2)
    # 3. The project: this folder and its parents, up to the home directory.
    here = Path.cwd()
    for level, folder in enumerate([here, *here.parents]):
        if level > _PROJECT_SEARCH_LEVELS or folder == home:
            break
        for hidden in _hidden_folders(folder):
            found += _skills_folders_under(hidden, 1)
        if level == 0 and _holds_skills(folder / "skills"):
            found.append(folder / "skills")

    out: list[Path] = []
    seen: set[Path] = set()
    for d in found:
        try:
            resolved = d.resolve()
        except OSError:
            continue
        if resolved not in seen:
            seen.add(resolved)
            out.append(d)
    _common_cache = (key, now, out)
    return list(out)


@dataclass(frozen=True)
class ExternalSkillDirs:
    """Where ONE caller looks for the skills other agents installed.

    It belongs to the caller. The Engine builds one from its own
    ``engine.skills_dirs`` / ``engine.skills_scan_known_dirs`` and hands it to
    every lookup it makes, so quests sharing a process (``--fleet``) never see
    one another's folders. It used to be one setting for the whole process,
    which the last engine to start decided for all of them: an earlier quest
    lost its own skills, or found and used a skill from a folder it never named.

    Commands that run without a config (``--skills``, ``--approve-skill``, the
    web skills page) pass none, and get the default: no folder named, the usual
    places searched.
    """

    dirs: tuple[Path, ...] = ()
    scan_known: bool = True

    @classmethod
    def of(
        cls, dirs: Iterable[str | Path] = (), *, scan_known: bool = True,
    ) -> "ExternalSkillDirs":
        return cls(
            tuple(Path(d).expanduser() for d in dirs if str(d).strip()), scan_known,
        )


def external_skill_dirs(external_dirs: ExternalSkillDirs | None = None) -> list[Path]:
    """The existing external skill folders, the caller's own named ones first.

    ``external_dirs`` is the caller's setting; None is the default (see
    ``ExternalSkillDirs``). ``FI_EXTERNAL_SKILLS_DIRS`` outranks either."""
    if external_dirs is None:
        external_dirs = ExternalSkillDirs()
    env = os.environ.get(_ENV_EXTERNAL_DIRS)
    if env is not None:
        candidates = [Path(p).expanduser() for p in env.split(os.pathsep) if p.strip()]
    else:
        candidates = list(external_dirs.dirs)
        if external_dirs.scan_known:
            candidates += _common_skill_dirs()
    own = skills_root().resolve()
    out: list[Path] = []
    seen: set[Path] = {own}
    for d in candidates:
        try:
            resolved = d.resolve()
        except OSError:
            continue
        if resolved in seen or not d.is_dir():
            continue
        seen.add(resolved)
        out.append(d)
    return out


def _from_filesystem(skills_dir: Path, source: str = "filesystem") -> list[Skill]:
    out: list[Skill] = []
    try:
        entries = sorted(p for p in skills_dir.iterdir() if p.is_dir())
    except OSError:
        return out
    for d in entries:
        s = Skill(name=d.name, path=d, source=source)
        if s.valid:
            out.append(s)
        else:
            _log.debug("skills: %s has no SKILL.md; not a skill", d)
    return out


def _from_entry_points() -> list[Skill]:
    """Skills advertised by installed packages.

    Best-effort by design: a broken third-party package must not stop FI
    from finding the skills that do load.
    """
    out: list[Skill] = []
    try:
        from importlib.metadata import entry_points

        eps = entry_points(group=ENTRY_POINT_GROUP)
    except Exception:  # noqa: BLE001 - importlib shape varies across versions
        return out

    for ep in eps:
        try:
            target = ep.load()
        except Exception as e:  # noqa: BLE001
            _log.warning("skills: entry point %r failed to load: %s", ep.name, e)
            continue
        path = Path(getattr(target, "__file__", "") or "").parent \
            if not isinstance(target, (str, Path)) else Path(target)
        s = Skill(name=ep.name, path=path, source=ENTRY_POINT_GROUP)
        if s.valid:
            out.append(s)
        else:
            _log.warning(
                "skills: entry point %r resolved to %s with no SKILL.md", ep.name, path,
            )
    return out


def discover(
    skills_dir: Path | None = None, *, external: bool | None = None,
    external_dirs: ExternalSkillDirs | None = None,
) -> list[Skill]:
    """Every skill FI can see. FI's own folder shadows entry points, and both
    shadow an external skill of the same name.

    External folders are included when discovering the machine's skills
    (``skills_dir`` is None). Passing an explicit ``skills_dir`` looks at that
    folder alone unless ``external=True`` asks for the rest too. Which external
    folders is the caller's own ``external_dirs`` (see ``ExternalSkillDirs``).
    """
    fs = _from_filesystem(skills_dir or skills_root())
    seen = {s.name for s in fs}
    eps = [s for s in _from_entry_points() if s.name not in seen]
    seen |= {s.name for s in eps}
    ext: list[Skill] = []
    if external if external is not None else skills_dir is None:
        for d in external_skill_dirs(external_dirs):
            for s in _from_filesystem(d, source=EXTERNAL_SOURCE):
                if s.name in seen:
                    _log.info(
                        "skills: external %s (%s) is shadowed by another skill of "
                        "the same name", s.name, d,
                    )
                    continue
                seen.add(s.name)
                ext.append(s)
    return fs + eps + ext


def run_selftest(skill: Skill, *, timeout_s: int | None = None) -> tuple[bool, str]:
    """Run the skill's own check. Returns (passed, combined output).

    A missing self-test is not a pass — see ``evaluate``. This function
    only reports what happened when one exists.

    ``timeout_s`` defaults to a ceiling that scales with how many scripts
    a GENERATED selftest bundles, not a flat constant. A generated test
    probes every bundled script with its own ``--help`` timeout (see
    ``scaffold.PER_SCRIPT_PROBE_TIMEOUT_S``); a flat outer ceiling that
    doesn't grow with script count kills a multi-script selftest before
    every script gets its full per-script allowance, which orphans
    whichever probe subprocess was still running at that moment (observed
    live: a 5-script skill where several scripts ignore `--help` and run
    their demo instead reliably blew a flat 120s outer ceiling, each time
    leaving that script's subprocess running with nothing left to kill
    it — a real, reproducible resource leak, not a one-off). A hand-written
    selftest has no such structural reason to be slow, so scriptless /
    few-script skills still get the original flat floor.
    """
    script = skill.file(SELFTEST_PY)
    if not script.is_file():
        return False, "no selftest.py"

    if timeout_s is None:
        scripts_dir = skill.path / "scripts"
        script_count = (
            sum(1 for p in scripts_dir.rglob("*.py") if p.is_file())
            if scripts_dir.is_dir() else 0
        )
        from core.skills.scaffold import PER_SCRIPT_PROBE_TIMEOUT_S
        timeout_s = max(
            SELFTEST_TIMEOUT_S,
            script_count * (PER_SCRIPT_PROBE_TIMEOUT_S + 5) + 20,
        )

    env = dict(os.environ)
    # The skill directory on the path so selftest.py can import skill.py
    # beside it without the skill needing to be an installed package.
    env["PYTHONPATH"] = os.pathsep.join(
        [str(skill.path), env.get("PYTHONPATH", "")]
    ).strip(os.pathsep)

    # Run against a COPY of the skill, never the skill itself.
    #
    # A self-test that plots writes its PNG beside the script, and the
    # skill's content hash covers everything in the folder -- so the test
    # that proves a skill works was silently revoking its own approval.
    # Observed across three quests in one hour: scikit-learn reported
    # "content changed since approval" with a different hash every time
    # (91c8cd1d, 8e3b8fbf, 3a885971), because matplotlib output is not
    # byte-identical between runs. The ledger was right; it was being
    # asked about a directory the gate itself kept editing.
    #
    # Merely running elsewhere is not enough: three skills read their own
    # data files relative to cwd, and a bare temp directory quarantined
    # all three. Copying keeps cwd meaning what the test expects while the
    # writes land somewhere we throw away.
    try:
        with tempfile.TemporaryDirectory(prefix="fi-selftest-") as tmp:
            work = Path(tmp) / skill.path.name
            shutil.copytree(
                skill.path, work,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
            )
            env["PYTHONPATH"] = os.pathsep.join(
                [str(work), os.environ.get("PYTHONPATH", "")]
            ).strip(os.pathsep)
            # An absolute path to the real skill, for a test that wants its
            # installed location rather than this scratch copy.
            env["SKILL_DIR"] = str(skill.path)
            proc = subprocess.run(
                [sys.executable, str(work / SELFTEST_PY)],
                cwd=str(work),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
    except subprocess.TimeoutExpired:
        return False, f"selftest timed out after {timeout_s}s"
    except OSError as e:
        return False, f"selftest could not run: {e}"

    output = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0, output.strip()[-4000:]


def _scan_findings(skill: Skill) -> tuple[list[str], bool]:
    """Static findings for this skill, as rendered lines, and whether the
    scan completed.

    Best-effort: the scanner is a review aid, and a bug in it must never
    stop a working skill from being evaluated. A skill with no findings and
    a skill whose scan crashed are different, so the second says so — and
    is never recorded as the skill's scan.
    """
    try:
        from core.skills import scan as _scan

        return [f.render() for f in _scan.scan(skill)], True
    except Exception as e:  # noqa: BLE001
        _log.warning("skills: scan of %s failed: %s", skill.name, e)
        return [f"scan did not run ({e}) — review this skill by hand"], False


def evaluate(
    skill: Skill,
    *,
    ledger: Path | None = None,
    run_test: bool = True,
    cache: SelftestCache | None = None,
    defer_scan: bool = False,
) -> SkillState:
    """Decide what state a skill is in.

    ``defer_scan`` is for bulk use (loading skills into a quest, listing them): an
    external skill nobody has approved is reported as awaiting approval without
    hashing or scanning it. Other agents' folders can hold hundreds of skills, some
    of hundreds of files, and reading all of them costs seconds on every call for
    skills nobody asked for; the review (``--scan-skill``, the approval itself) reads
    the one being approved.

    Both gates must pass before a skill is loadable, and they are checked
    in this order because a failing self-test should be reported as a
    broken skill rather than an unapproved one:

    1. The self-test it carries must exist and pass.
    2. A person must have approved *this exact content*.

    ``run_test=False`` skips execution and reports on approval alone —
    used by listing commands that should not execute anything.

    ``cache`` is for loading skills into a quest. With one, a pass recorded
    for this content under this Python and these installed packages counts
    as the self-test passing, and a new pass is recorded. Without one — every
    command that checks a skill on demand — the self-test always runs.
    Either way a failure is never recorded, and it removes the skill's
    recorded passes: the key cannot see an external binary or a service a
    test depends on, and a failure just observed outranks an older pass. The
    approval check is the same on every path.

    The static scan runs on every path, including the ones that return
    early. This is the chokepoint every skill passes through — import,
    hand-authored, and entry-point alike — so scanning only at import would
    leave the other two routes unreviewed. The one exception is a recorded
    pass, which carries the scan taken of the same content by the same
    scanner, and that scan is reused rather than parsed again. Findings never
    change the status: they are attached for whoever is about to approve,
    because the rules are heuristics and QUARANTINED means *FI observed a
    failure*, not *FI guessed at intent*.
    """
    approved = approval.approved_hash(skill.ledger_name, ledger)
    if defer_scan and skill.external and approved is None:
        return SkillState(
            skill=skill,
            status=Status.PROPOSED,
            reason=(
                f"awaiting approval — external skill from {skill.path.parent}, never "
                "self-tested (--scan-skill reviews it; --approve-skill approves it)"
            ),
            approved_hash=None,
        )
    content_hash = skill.content_hash()
    cached = (
        cache.lookup(skill.name, content_hash)
        if cache is not None and run_test and skill.has_selftest
        else None
    )
    if cached is not None and cached.findings is not None:
        findings, scanned = cached.findings, True
    else:
        findings, scanned = _scan_findings(skill)
        if cached is not None and scanned and cache.scanner:
            # Recorded under an older scanner: refresh it, or every later
            # load would parse the skill again.
            cache.record_pass(
                skill.name, content_hash, output=cached.output, findings=findings,
            )

    # An external skill was written for another agent and has no FI self-test.
    # It is not refused for that: the approval of its exact content is the gate
    # (the static scan above still runs, and its findings reach whoever
    # approves). One that does carry a selftest.py is tested like any other.
    untested_external = skill.external and not skill.has_selftest
    if not skill.has_selftest and not untested_external:
        return SkillState(
            skill=skill,
            status=Status.UNTESTED,
            reason=(
                "no selftest.py — a skill that cannot demonstrate it still "
                "works is never promoted"
            ),
            approved_hash=approved,
            findings=findings,
        )

    output = ""
    if run_test and cached is not None:
        output = cached.output
    elif run_test and not untested_external:
        passed, output = run_selftest(skill)
        if cache is not None:
            cache.note_ran()
        if not passed:
            if cache is not None:
                cache.forget(skill.name)
            else:
                selftest_cache.forget_skill(skill.name, ledger=ledger)
            return SkillState(
                skill=skill,
                status=Status.QUARANTINED,
                reason="selftest failed",
                selftest_output=output,
                approved_hash=approved,
                findings=findings,
            )
        # Recorded only if the skill is still the content that was hashed:
        # an edit landing while the test ran must not inherit its pass.
        if cache is not None and skill.content_hash() == content_hash:
            cache.record_pass(
                skill.name, content_hash,
                output=output, findings=findings if scanned else None,
            )

    if approved is None:
        reason = "awaiting approval — no person has approved this skill"
        if untested_external:
            reason += (
                f" (external skill from {skill.path.parent}, never self-tested: "
                "read its scripts before approving)"
            )
    elif approved != content_hash:
        reason = (
            f"content changed since approval (approved {approved}, "
            f"now {content_hash}) — re-approval required"
        )
    else:
        return SkillState(
            skill=skill,
            status=Status.TRUSTED,
            reason=(
                "external skill, no self-test; this content is approved"
                if untested_external
                else "selftest passes and this content is approved"
            ),
            selftest_output=output,
            approved_hash=approved,
            findings=findings,
        )

    return SkillState(
        skill=skill,
        status=Status.PROPOSED,
        reason=reason,
        selftest_output=output,
        approved_hash=approved,
        findings=findings,
    )


#: Self-tests run at once while loading. Each is a subprocess waiting on its
#: own interpreter, so threads are enough; the ceiling is about not handing
#: the machine eight scientific imports at once more than about the GIL.
MAX_SELFTEST_WORKERS = 8


def default_selftest_workers() -> int:
    return max(1, min(MAX_SELFTEST_WORKERS, os.cpu_count() or 1))


def _evaluate_all(
    skills: list[Skill],
    *,
    ledger: Path | None,
    cache: SelftestCache | None,
    workers: int,
) -> list[SkillState]:
    """``evaluate`` each skill, in parallel, returning states in input order.

    Every skill keeps its own timeout, since each runs in its own subprocess
    through ``run_selftest``.
    """
    if workers <= 1 or len(skills) <= 1:
        return [evaluate(s, ledger=ledger, cache=cache, defer_scan=True) for s in skills]
    pool = ThreadPoolExecutor(
        max_workers=min(workers, len(skills)), thread_name_prefix="fi-selftest",
    )
    try:
        futures = [
            pool.submit(evaluate, s, ledger=ledger, cache=cache, defer_scan=True)
            for s in skills
        ]
        return [f.result() for f in futures]
    finally:
        # On an exception, drop what has not started rather than running
        # the rest of the library for a result nobody will read.
        pool.shutdown(wait=True, cancel_futures=True)


def loadable_skills(
    names: list[str],
    *,
    skills_dir: Path | None = None,
    ledger: Path | None = None,
    use_cache: bool = True,
    max_workers: int | None = None,
    external_dirs: ExternalSkillDirs | None = None,
) -> tuple[list[SkillState], list[SkillState]]:
    """Resolve requested skill names into (loadable, rejected).

    Requesting a skill that is not trusted is never an error — the quest
    proceeds with guided generation instead. Silently *using* an
    unapproved skill would be the error, so the rejects are returned
    rather than dropped, for the caller to log.

    This is the quest-time path, so it uses the self-test cache: a skill
    whose pass is recorded for its current content, this Python and these
    installed packages is not tested again. The self-tests that do run, run
    in parallel (``max_workers``, default ``default_selftest_workers()``).
    The lists come back in the order the names were given, exactly as a
    sequential run would return them. ``use_cache=False`` runs every test.

    ``external_dirs`` is the caller's own setting of where other agents' skills
    are looked for; a quest passes its own, so that a name resolves in the
    folders that quest named and not in ones only another quest did.

    Blocking: an async caller runs this in a thread.
    """
    started = time.monotonic()
    found = {s.name: s for s in discover(skills_dir, external_dirs=external_dirs)}
    # Each skill is evaluated once, however many times it is named.
    wanted = [n for n in dict.fromkeys(names) if n in found]
    cache = SelftestCache.open(ledger=ledger) if use_cache and wanted else None
    states = dict(zip(wanted, _evaluate_all(
        [found[n] for n in wanted],
        ledger=ledger,
        cache=cache,
        workers=max_workers or default_selftest_workers(),
    )))

    ok: list[SkillState] = []
    rejected: list[SkillState] = []
    for name in names:
        state = states.get(name)
        if state is None:
            rejected.append(
                SkillState(
                    skill=Skill(name=name, path=Path(name), source="missing"),
                    status=Status.UNTESTED,
                    reason="no skill by that name was found",
                )
            )
            continue
        (ok if state.loadable else rejected).append(state)

    if cache is not None:
        _log.info(
            "skills: evaluated %d in %.1fs — %d self-test(s) run, %d passed "
            "earlier for the same content and environment (%s)",
            len(wanted), time.monotonic() - started, cache.ran, cache.hits,
            cache.path,
        )
    return ok, rejected
