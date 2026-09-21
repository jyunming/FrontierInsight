"""Run the simulation once, analyse it as often as needed.

``execution.split_analysis`` keeps a quest's simulation and its analysis in two scripts:
``code/simulate.py`` runs the simulation and writes what it produced as files, and
``code/experiment.py`` -- the name every other part of FI already reads -- is the analysis:
it reads those files, computes the statistics, draws the figures and prints ``RESULT_JSON``.

Why: one script that simulates *and* analyses is repaired by writing it again and running it
again, so a plotting mistake, a wrong confidence interval or a mislabelled table costs the
whole simulation a second time. For a simulation that takes seconds that is invisible; for one
that takes hours it is the difference between a repair and a lost day. Kept apart, an analysis
that fails or is sent back by the review is rewritten and run against the raw files that are
already on disk.

Which raw files are still good is decided by the files, not by anyone's judgement. Each seed's
folder holds a manifest (``manifest.json``) recording the SHA-256 of the ``simulate.py`` that
wrote it, the seed it was written for, and every file with its size (and, up to a size, its
SHA-256). The simulation runs again exactly when that record does not fit: ``simulate.py`` was
rewritten (a repair, or a review that named the simulation), the seed differs, a file is gone or
a different size, or nothing was written yet. Rewriting only the analysis leaves the record as it
was, so the raw files are used as they are.

Every seed is its own pair -- simulation, then analysis -- in ``raw/seed<K>/``, so FI's
replicates, their confidence intervals, the per-seed figure records and the mean figures it draws
from them work as they did for one script.

The folder is named to the scripts by ``FI_RAW_DIR``, a path relative to the quest folder (or
absolute when ``execution.raw_dir`` puts it elsewhere, which the Docker sandbox cannot see).
FI records where the files are and what they are; it does not copy them.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .execution import ExecutionResult

RAW_DIRNAME = "raw"
MANIFEST_NAME = "manifest.json"
SIMULATE_NAME = "simulate.py"
ANALYSIS_NAME = "experiment.py"
RAW_DIR_ENV = "FI_RAW_DIR"
SEED_ENV = "FI_REPLICATE_SEED"
INDEX_ENV = "FI_REPLICATE_INDEX"
MANIFEST_VERSION = 1

# A file larger than this is recorded by its size alone: hashing tens of gigabytes to
# write a record costs more than the record is worth, and the size already tells a
# truncated or replaced file from the one the simulation wrote.
HASH_LIMIT_BYTES = 256 * 1024 * 1024

# What the response of the code-writing step marks each script with.
_FILE_MARKER = re.compile(r"^\s*#\s*file\s*:\s*([\w.\-/\\]+)\s*$", re.IGNORECASE)


def raw_root_of(quest_root: Path, configured: str = "") -> Path:
    """Where the raw files of every seed live: ``raw/`` in the quest folder, or
    ``execution.raw_dir`` (relative to the quest folder, or absolute)."""
    text = (configured or "").strip()
    if not text:
        return quest_root / RAW_DIRNAME
    path = Path(text).expanduser()
    return path if path.is_absolute() else quest_root / path


def raw_dir_for(raw_root: Path, index: int) -> Path:
    return raw_root / f"seed{index}"


def env_value(raw_dir: Path, quest_root: Path) -> str:
    """``FI_RAW_DIR`` for the scripts: relative to the quest folder when the folder is
    inside it (the scripts run there, and so does a container that mounts it), else
    absolute."""
    try:
        return raw_dir.resolve().relative_to(quest_root.resolve()).as_posix()
    except ValueError:
        return str(raw_dir)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _files_of(raw_dir: Path) -> list[Path]:
    if not raw_dir.is_dir():
        return []
    return sorted(p for p in raw_dir.rglob("*") if p.is_file() and p.name != MANIFEST_NAME)


def build_manifest(raw_dir: Path, *, simulate_sha: str, seed: int, index: int) -> dict[str, Any]:
    """The record of one seed's raw files: what wrote them, for which seed, and what they are."""
    files = []
    for path in _files_of(raw_dir):
        size = path.stat().st_size
        files.append({
            "path": path.relative_to(raw_dir).as_posix(),
            "bytes": size,
            "sha256": sha256_of(path) if size <= HASH_LIMIT_BYTES else None,
        })
    return {
        "version": MANIFEST_VERSION,
        "simulate_sha256": simulate_sha,
        "seed": seed,
        "index": index,
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": files,
    }


def write_manifest(raw_dir: Path, manifest: dict[str, Any]) -> Path:
    path = raw_dir / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return path


def read_manifest(raw_dir: Path) -> dict[str, Any] | None:
    try:
        data = json.loads((raw_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def stale_reason(raw_dir: Path, *, simulate_sha: str, seed: int) -> str | None:
    """Why the raw files in ``raw_dir`` cannot be used for this simulation and seed, or
    ``None`` when they can."""
    manifest = read_manifest(raw_dir)
    if manifest is None:
        return "no raw files have been written yet"
    if manifest.get("version") != MANIFEST_VERSION:
        return "their record is of another version"
    if manifest.get("simulate_sha256") != simulate_sha:
        return f"{SIMULATE_NAME} has changed since they were written"
    if manifest.get("seed") != seed:
        return f"they were written for seed {manifest.get('seed')}, not {seed}"
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        return "their record lists no files"
    for entry in files:
        try:
            path, size = raw_dir / str(entry["path"]), int(entry["bytes"])
        except (KeyError, TypeError, ValueError):
            return "their record is malformed"
        if not path.is_file():
            return f"{entry['path']} is missing"
        if path.stat().st_size != size:
            return f"{entry['path']} is not the size it was written at"
    return None


def same_script(a: str, b: str) -> bool:
    """Whether two versions of a script are the same but for line endings and trailing
    whitespace: a model handed a script to give back unchanged rarely reproduces its bytes."""
    def lines(text: str) -> list[str]:
        return [line.rstrip() for line in text.replace("\r\n", "\n").strip().split("\n")]

    return lines(a) == lines(b)


def parse_split_response(text: str, fence: re.Pattern[str]) -> dict[str, str] | None:
    """The two scripts of a code-writing reply: ``{"simulate": ..., "analysis": ...}``, or
    ``None`` when the reply does not hold both.

    Each fenced block names its script on its first line (``# file: simulate.py`` and
    ``# file: experiment.py``); with no such line, two blocks are read in that order. ``fence``
    is the engine's own pattern for a fenced Python block."""
    blocks = [m.group(1).strip("\n") for m in fence.finditer(text or "")]
    named: dict[str, str] = {}
    for block in blocks:
        first = next((line for line in block.splitlines() if line.strip()), "")
        marker = _FILE_MARKER.match(first)
        if not marker:
            continue
        name = Path(marker.group(1).replace("\\", "/")).name.lower()
        body = "\n".join(block.splitlines()[block.splitlines().index(first) + 1:]).strip("\n")
        if name == SIMULATE_NAME:
            named["simulate"] = body
        elif name == ANALYSIS_NAME:
            named["analysis"] = body
    if len(named) == 2:
        return named
    if len(blocks) == 2 and not named:
        return {"simulate": blocks[0], "analysis": blocks[1]}
    return None


class SplitRunner:
    """Stands in for the executor around one experiment: a run of the analysis script is
    preceded by a run of the simulation script, unless the raw files it would write are
    still good.

    ``_node_execute`` calls ``execute([python, experiment.py], ...)`` for the first run and
    for each replicate; handed this in place of the executor, it needs no other change. Any
    other command (a warm-up, a check) passes straight through.

    ``failed_script`` names the script of the last call that failed (``"simulate.py"`` or
    ``"experiment.py"``), ``None`` when it succeeded: the repair has to know which of the two
    to rewrite.
    """

    def __init__(
        self,
        inner: Any,
        *,
        quest_root: Path,
        raw_root: Path,
        simulate: Path,
        analysis: Path,
        log: Any,
    ) -> None:
        self.inner = inner
        self.quest_root = quest_root
        self.raw_root = raw_root
        self.simulate = simulate
        self.analysis = analysis
        self._log = log
        self.failed_script: str | None = None
        self.simulated = 0
        self.reused = 0

    async def execute(
        self, cmd: list[str], *, cwd: Path, timeout_s: int, env: dict[str, str] | None = None,
    ) -> ExecutionResult:
        if len(cmd) < 2 or Path(cmd[1]) != self.analysis:
            return await self.inner.execute(cmd, cwd=cwd, timeout_s=timeout_s, env=env)

        env = dict(env if env is not None else os.environ)
        index, seed = _int_of(env.get(INDEX_ENV)), _int_of(env.get(SEED_ENV))
        raw_dir = raw_dir_for(self.raw_root, index)
        env[RAW_DIR_ENV] = env_value(raw_dir, self.quest_root)
        label = "the first run" if index == 0 else f"replicate {index} (seed {seed})"

        sim: ExecutionResult | None = None
        simulate_sha = sha256_of(self.simulate)
        why = stale_reason(raw_dir, simulate_sha=simulate_sha, seed=seed)
        if why is None:
            self.reused += 1
            self._log.info(
                "[execute] %s: the raw files in %s are the ones %s wrote; running %s alone",
                label, env[RAW_DIR_ENV], SIMULATE_NAME, ANALYSIS_NAME,
            )
        else:
            self._log.info(
                "[execute] %s: running %s (%s); its files go to %s",
                label, SIMULATE_NAME, why, env[RAW_DIR_ENV],
            )
            # What a script of another version left could be read as this one's, so none stays.
            shutil.rmtree(raw_dir, ignore_errors=True)
            raw_dir.mkdir(parents=True, exist_ok=True)
            sim = await self.inner.execute(
                [cmd[0], str(self.simulate)], cwd=cwd, timeout_s=timeout_s, env=env,
            )
            if sim.returncode != 0:
                self.failed_script = SIMULATE_NAME
                return _annotated(sim, f"{SIMULATE_NAME} (the simulation) failed; {ANALYSIS_NAME} did not run")
            manifest = build_manifest(raw_dir, simulate_sha=simulate_sha, seed=seed, index=index)
            if not manifest["files"]:
                self.failed_script = SIMULATE_NAME
                return ExecutionResult(
                    1, sim.stdout,
                    (sim.stderr or "") + (
                        f"\n[FI] {SIMULATE_NAME} exited 0 but wrote no file into the folder the "
                        f"environment variable {RAW_DIR_ENV} names ({env[RAW_DIR_ENV]} for this run). "
                        f"Read it with os.environ[\"{RAW_DIR_ENV}\"]: it is a variable, and a folder "
                        f"called {RAW_DIR_ENV} is not it. {ANALYSIS_NAME} reads only that folder, so "
                        f"the simulation has to save what it produced there."
                    ),
                    sim.duration_s, sim.timed_out,
                )
            write_manifest(raw_dir, manifest)
            self.simulated += 1

        analysis = await self.inner.execute(cmd, cwd=cwd, timeout_s=timeout_s, env=env)
        self.failed_script = ANALYSIS_NAME if analysis.returncode != 0 else None
        return ExecutionResult(
            analysis.returncode, analysis.stdout, analysis.stderr,
            analysis.duration_s + (sim.duration_s if sim else 0.0), analysis.timed_out,
        )


def _int_of(value: str | None) -> int:
    try:
        return int(value or 0)
    except ValueError:
        return 0


def _annotated(result: ExecutionResult, note: str) -> ExecutionResult:
    return ExecutionResult(
        result.returncode, result.stdout, f"[FI] {note}\n{result.stderr or ''}",
        result.duration_s, result.timed_out,
    )


def listing(raw_dir: Path, *, limit: int = 40) -> str:
    """The raw files of one seed, one per line with its size, for the repair of an analysis
    that could not read them. Empty when there are none."""
    manifest = read_manifest(raw_dir)
    rows = (manifest or {}).get("files") or []
    if not rows:
        rows = [
            {"path": p.relative_to(raw_dir).as_posix(), "bytes": p.stat().st_size} for p in _files_of(raw_dir)
        ]
    lines = [f"- {row['path']} ({row['bytes']:,} bytes)" for row in rows[:limit]]
    if len(rows) > limit:
        lines.append(f"- ... and {len(rows) - limit} more")
    return "\n".join(lines)
