"""Start the topics of a round as independent processes, at most N at a time, and follow them.

Each topic is ``launch.py --config <its config>`` run with the frozen FI worktree as its working directory, in a process
that survives this one (a detached process group), with its own Axon store so runs cannot see each other. The round
manifest ``<round>/round.manifest.json`` records the pids, the start and the end of each. The ``spawn`` and ``alive``
functions are parameters so tests launch nothing.

The API key never reaches the manifest, a log line or an error message: the env file is read into the child's
environment only.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .suite import Suite, SuiteError

MANIFEST = "round.manifest.json"
ENDED = ("done", "ended", "adopted", "failed_to_start")


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_env_file(path: Path | None) -> dict[str, str]:
    """``KEY=VALUE`` lines of an env file (comments and blanks skipped, one pair of quotes removed)."""
    out: dict[str, str] = {}
    if path is None or not Path(path).is_file():
        return out
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            out[key] = value
    return out


def git_state(fi: Path) -> dict[str, Any]:
    """The frozen FI worktree's commit and whether it has uncommitted changes (``unknown`` when git cannot say)."""
    def run(*args: str) -> str | None:
        try:
            r = subprocess.run(["git", "-C", str(fi), *args], capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            return None
        return r.stdout.strip() if r.returncode == 0 else None

    sha = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {"sha": (sha or "unknown")[:40], "short": (sha or "unknown")[:8],
            "dirty": None if status is None else bool(status)}


# ----------------------------------------------------------------------------------------------- processes

def default_alive(pid: int) -> bool:
    """Whether a process with this pid is running. ``os.kill(pid, 0)`` is not safe on Windows, so ask ``tasklist``."""
    if not pid:
        return False
    if os.name == "nt":
        try:
            r = subprocess.run(["tasklist", "/FI", f"PID eq {int(pid)}", "/FO", "CSV", "/NH"], capture_output=True,
                               text=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            return False
        return f'"{int(pid)}"' in r.stdout
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def default_spawn(cmd: list[str], *, cwd: Path, env: dict[str, str], stdout: Path, stderr: Path) -> Any:
    """Start ``cmd`` detached from this process (it keeps running when this process or its console goes away)."""
    kwargs: dict[str, Any] = {}
    if os.name == "nt":
        base = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        kwargs["creationflags"] = base | getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0)
    else:
        kwargs["start_new_session"] = True
    out, err = open(stdout, "ab"), open(stderr, "ab")  # noqa: SIM115 -- handed to the child
    try:
        try:
            return subprocess.Popen(cmd, cwd=str(cwd), env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=err, **kwargs)
        except OSError:
            if os.name == "nt":  # a job that does not allow breakaway
                kwargs["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
                return subprocess.Popen(cmd, cwd=str(cwd), env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=err, **kwargs)
            raise
    finally:
        out.close()
        err.close()


# ----------------------------------------------------------------------------------------------- manifest

def manifest_path(round_dir: Path) -> Path:
    return Path(round_dir) / MANIFEST


def load_manifest(round_dir: Path) -> dict[str, Any]:
    try:
        value = json.loads(manifest_path(round_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SuiteError(f"no round manifest in {round_dir}: {exc}") from exc
    return value


def save_manifest(round_dir: Path, manifest: dict[str, Any]) -> None:
    path = manifest_path(round_dir)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def round_dir_of(suite: Suite, round_name: str) -> Path:
    return suite.rounds_dir / round_name


def prepare_round(suite: Suite, group: str, round_name: str, fi: Path, *, max_parallel: int, acceptance: bool = False,
                  python: str | None = None, git: Callable[[Path], dict[str, Any]] = git_state) -> Path:
    """Write the configs and the manifest of a new round (nothing is started). Refuses a held-back group without
    ``acceptance``, a round name that already has a manifest, and an FI folder that has no ``launch.py``."""
    suite.require_allowed(group, acceptance)
    fi = Path(fi).resolve()
    if not (fi / "launch.py").is_file():
        raise SuiteError(f"--fi {fi} has no launch.py: give the folder of a frozen FI worktree")
    if max_parallel < 1:
        raise SuiteError("--max-parallel must be at least 1")
    rdir = round_dir_of(suite, round_name)
    if manifest_path(rdir).exists():
        raise SuiteError(f"round `{round_name}` already exists in {rdir}; give a new --round name")
    configs = suite.write_configs(group, rdir)
    manifest = {
        "round": round_name, "group": group, "acceptance": bool(acceptance), "suite": str(suite.path),
        "fi": str(fi), "fi_git": git(fi), "python": python or sys.executable, "max_parallel": int(max_parallel),
        "created_at": now(),
        "topics": {t: {"config": str(p), "dir": str(p.parent), "quest_root": None, "state": "queued", "pid": None,
                       "started_at": None, "ended_at": None, "exit_code": None} for t, p in configs.items()},
    }
    save_manifest(rdir, manifest)
    return rdir


def child_env(suite: Suite, round_name: str, topic: str, topic_dir: Path, base: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if base is None else base)
    env.update(parse_env_file(suite.env_file))
    env["AXON_STORE_BASE"] = str(topic_dir / "axon_store")
    env["FI_AXON_PROJECT"] = f"suite-{round_name}-{topic}"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _refresh(entry: dict[str, Any], handle: Any, alive: Callable[[int], bool]) -> None:
    """Move a running topic to ``done`` when its process is gone (the exit code is known only to the starter)."""
    if entry["state"] != "running":
        return
    if handle is not None:
        code = handle.poll()
        if code is None:
            return
        entry.update(state="done", ended_at=now(), exit_code=code)
    elif not alive(int(entry.get("pid") or 0)):
        entry.update(state="done", ended_at=now())


LOCK = "driver.lock"
#: A lock file younger than this with no readable pid is a driver between creating it and writing its pid, not a stale one.
_LOCK_GRACE_SECONDS = 10.0


def lock_path(round_dir: Path) -> Path:
    return Path(round_dir) / LOCK


def _lock_owner(path: Path) -> int | None:
    """The pid written in a lock file, or ``None`` when it holds none that can be read."""
    try:
        return int(path.read_text(encoding="utf-8").strip().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def acquire_lock(round_dir: Path, *, pid: int | None = None, alive: Callable[[int], bool] = default_alive) -> Path:
    """Take the round's exclusive driver lock: a file created with ``O_CREAT | O_EXCL`` (atomic on Windows and POSIX) that
    holds the driver's pid. A second driver of the same round refuses with a plain message; a lock whose process is gone
    is taken over. Two drivers reading the same queued state would otherwise start the same topic twice."""
    path, me = lock_path(round_dir), int(pid if pid is not None else os.getpid())
    for _ in range(5):
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            owner = _lock_owner(path)
            if owner is None:
                try:
                    young = time.time() - path.stat().st_mtime < _LOCK_GRACE_SECONDS
                except OSError:
                    continue  # it went away between the two looks: try again
                if young:
                    raise SuiteError(f"round {Path(round_dir).name} is being started by another driver right now; "
                                     "wait a moment and look at its status") from None
            elif alive(owner):
                raise SuiteError(f"round {Path(round_dir).name} is already being driven by process {owner}: "
                                 f"follow it with `status`, or stop that process first ({path})") from None
            # stale: take it over by moving it aside (one mover wins), and make sure it was the stale file that was moved
            moved = path.with_name(f"{LOCK}.stale.{me}")
            try:
                os.replace(path, moved)
            except OSError:
                continue
            if _lock_owner(moved) != owner:  # a fresh lock was moved by mistake: put it back if the place is free
                try:
                    os.link(moved, path)
                except OSError:
                    pass
                finally:
                    try:
                        os.unlink(moved)
                    except OSError:
                        pass
                raise SuiteError(f"round {Path(round_dir).name} is being started by another driver right now; "
                                 "wait a moment and look at its status")
            try:
                os.unlink(moved)
            except OSError:
                pass
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(f"{me}\n")
        return path
    raise SuiteError(f"could not take the driver lock of round {Path(round_dir).name} ({path})")


def release_lock(round_dir: Path, *, pid: int | None = None) -> None:
    """Remove the lock if it is this driver's (never another driver's)."""
    path = lock_path(round_dir)
    if _lock_owner(path) == int(pid if pid is not None else os.getpid()):
        try:
            os.unlink(path)
        except OSError:
            pass


def drive(suite: Suite, round_dir: Path, *, spawn: Callable[..., Any] = default_spawn,
          alive: Callable[[int], bool] = default_alive, sleep: Callable[[float], None] = time.sleep,
          interval: float = 20.0, max_loops: int | None = None, pid: int | None = None) -> dict[str, Any]:
    """Start the queued topics, never more than ``max_parallel`` live at once, until every topic has ended.

    Safe to call again after this process died: topics that were started are followed by pid, queued ones are started.
    Only one driver may drive a round: it holds ``driver.lock`` for as long as it runs (a second one refuses with a
    ``SuiteError``). Each queued topic is claimed (``starting``, written to the manifest through a temporary file and
    ``os.replace``) from the manifest as it is on disk now, before its process is launched, so a topic is never launched
    twice; a topic left ``starting`` by a driver that died is ended as ``failed_to_start`` rather than risk a second run."""
    rdir = Path(round_dir)
    me = int(pid if pid is not None else os.getpid())
    acquire_lock(rdir, pid=me, alive=alive)
    try:
        return _drive_locked(suite, rdir, spawn=spawn, alive=alive, sleep=sleep, interval=interval, max_loops=max_loops)
    finally:
        release_lock(rdir, pid=me)


def _patch(rdir: Path, name: str, **fields: Any) -> dict[str, Any]:
    """Read the manifest as it is on disk, change one topic's entry and write it back through a temporary file and
    ``os.replace``: nothing another writer changed in between (an adopted topic, say) is lost."""
    manifest = load_manifest(rdir)
    manifest["topics"][name].update(fields)
    save_manifest(rdir, manifest)
    return manifest


def _drive_locked(suite: Suite, rdir: Path, *, spawn: Callable[..., Any], alive: Callable[[int], bool],
                  sleep: Callable[[float], None], interval: float, max_loops: int | None) -> dict[str, Any]:
    handles: dict[str, Any] = {}
    loops = 0
    while True:
        manifest = load_manifest(rdir)  # as it is on disk now, not as an earlier look left it
        for name, entry in manifest["topics"].items():
            before = dict(entry)
            if entry["state"] == "starting" and name not in handles:
                entry.update(state="failed_to_start", ended_at=now(), error="the driver stopped while starting it")
            _refresh(entry, handles.get(name), alive)
            if entry != before:
                _patch(rdir, name, **entry)
        for name in list(manifest["topics"]):
            fresh = load_manifest(rdir)  # claim from the state on disk, one topic at a time
            entry = fresh["topics"][name]
            live = sum(1 for e in fresh["topics"].values() if e["state"] in ("running", "starting"))
            if entry["state"] != "queued" or live >= fresh["max_parallel"]:
                continue
            tdir = Path(entry["dir"])
            cmd = [fresh["python"], "launch.py", "--config", entry["config"]]
            _patch(rdir, name, state="starting")  # the claim is on disk before the process exists
            try:
                h = spawn(cmd, cwd=Path(fresh["fi"]), env=child_env(suite, fresh["round"], name, tdir),
                          stdout=tdir / "stdout.log", stderr=tdir / "stderr.log")
            except OSError as exc:
                _patch(rdir, name, state="failed_to_start", ended_at=now(), error=type(exc).__name__)
                continue
            handles[name] = h
            started = now()
            _patch(rdir, name, state="running", pid=int(h.pid), started_at=started, quest_root=str(tdir / "out"))
            (tdir / "start.txt").write_text(f"{name} pid {h.pid} started {started} fi {fresh['fi_git']['short']}\n",
                                            encoding="utf-8")
        manifest = load_manifest(rdir)
        if all(e["state"] in ENDED for e in manifest["topics"].values()):
            manifest["ended_at"] = now()
            save_manifest(rdir, manifest)
            return manifest
        loops += 1
        if max_loops is not None and loops >= max_loops:
            return manifest
        sleep(interval)


def start_detached_driver(suite_path: Path, round_dir: Path, *, python: str | None = None, interval: float = 20.0,
                          spawn: Callable[..., Any] = default_spawn) -> Any:
    """A driver process that outlives this command: it keeps starting topics as slots free up."""
    repo = Path(__file__).resolve().parents[3]
    cmd = [python or sys.executable, "-m", "dev.evaluation.suite", "drive", "--suite", str(suite_path),
           "--round-dir", str(round_dir),
           "--interval", str(interval)]
    return spawn(cmd, cwd=repo, env=dict(os.environ), stdout=Path(round_dir) / "driver.log", stderr=Path(round_dir) / "driver.err")


def status(round_dir: Path, *, alive: Callable[[int], bool] = default_alive) -> dict[str, Any]:
    """The manifest with each running topic checked against its pid (not written back)."""
    manifest = load_manifest(round_dir)
    for entry in manifest["topics"].values():
        _refresh(entry, None, alive)
    return manifest


def render_status(manifest: dict[str, Any]) -> str:
    lines = [f"Round {manifest['round']} ({manifest.get('group', '-')}), FI {manifest.get('fi_git', {}).get('short', '?')}, "
             f"up to {manifest.get('max_parallel', '-')} at a time", "",
             "| Topic | State | PID | Started | Ended |", "|---|---|---|---|---|"]
    for name, e in manifest["topics"].items():
        lines.append(f"| {name} | {e['state']} | {e.get('pid') or '-'} | {e.get('started_at') or '-'} | {e.get('ended_at') or '-'} |")
    return "\n".join(lines) + "\n"


def wait(round_dir: Path, *, alive: Callable[[int], bool] = default_alive, sleep: Callable[[float], None] = time.sleep,
         interval: float = 30.0, timeout: float | None = None, clock: Callable[[], float] = time.monotonic) -> tuple[bool, dict[str, Any]]:
    """Block until every topic has ended (True) or ``timeout`` seconds pass (False)."""
    start = clock()
    while True:
        m = status(round_dir, alive=alive)
        if all(e["state"] in ENDED for e in m["topics"].values()):
            return True, m
        if timeout is not None and clock() - start >= timeout:
            return False, m
        sleep(interval)
