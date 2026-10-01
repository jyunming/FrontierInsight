"""Explore, then confirm: a quest run in two stages (``engine.phased: true``; off by default, on by default under
``rigor_profile: research``).

In the EXPLORATION stage the model may try designs, look at results and change the design, as a quest always could.
When exploration ends (the quest is about to write its paper on an accepted result) the protocol is frozen and the
frozen design is run ONCE more in the CONFIRM stage, on data or seeds the exploration never saw. Only the confirm
run's numbers can back a publication-ready claim; while a quest has none, its result is preliminary
(``core/evidence.py`` reads :func:`evidence_settings`).

What the confirm run is given, decided before exploration runs anything (:func:`prepare`):

- ``held_back_data``: the person supplied one tabular data file (``inputs/data/``) that can be split by rows, with
  at least :data:`MIN_ROWS` rows (and at least :data:`MIN_HELD_BACK_ROWS` of them held back). About :data:`HOLD_BACK_FRACTION` of each file's rows, picked at random by a hash of
  each row's text (so a row keeps its side when rows are added or the quest starts again), is taken out before
  exploration starts, so exploration sees only the rest; the confirm run sees only that part. The confirm run is also
  given a new seed base.
- ``fresh_seeds``: otherwise (no data, too few rows, a file that cannot be split by rows, more than one data file:
  rows of different files may belong together, and split file by file a held-back record would show in the other). The confirm run is given a
  seed base exploration never used, and the report says it was confirmed on new random seeds and why no data was held
  back.

The whole files the person supplied, and the part held back, are kept OUTSIDE the quest folder, in
``<output_dir>/_held_back/<quest_id>/{original,held_back}/`` (:func:`store_dir`): the experiment runs in the quest folder
(a container sees only that folder, at ``/work``), so a script that walks it cannot find the held-back rows. The whole
files are put back in ``inputs/data/`` as soon as the confirm run's result is recorded, and whenever a run stops
(finished, paused or failed); the next start holds the same rows back again. Outside a running quest ``inputs/data/``
holds the person's whole files. Once exploration has run, only rows that were held back before can be held back again
(a renamed or re-exported file cannot move a row exploration saw into the confirm part).

The confirm run is made once, and it is the confirm run's own result that is recorded: any second run on the confirm
data or seeds, a result that is not the one the confirm run produced, a held-back part that could not be put in place,
or an experiment that does not take the seed it is given (new-seeds confirm) leaves nothing confirmed.

A quest that runs no experiment of its own (a literature survey, a no-simulation quest, ``--analyze``) has no design to
run once more: :func:`mark_not_applicable` puts any held-back rows back (the analysis must see all of the data), records
why, and the quest goes on as it would without the two stages; its paper carries no stage note and the evidence ladder
no stage gap.

A research quest that began before research turned this on by default (its config does not set ``engine.phased``) goes
on as it began, without the two stages: :func:`began_before_default` tells the engine so, and the engine says it in
run.log.

This record (``.fi/phased.json``) is apart from the attempt records and the shadow recommendations
(``core/attempt_records.py``, ``core/attempt_memory.py``): it reads neither, and neither decides anything here.
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = "fi.phased/v1"
RECORD = "phased.json"  # under .fi/
DIR = "phased"  # where the kept files were before they moved out of the quest folder: .fi/phased/{original,held_back}/
#: The folder beside the quests (in the output folder) that keeps each quest's whole files and held-back part.
STORE = "_held_back"

EXPLORE = "explore"
CONFIRM = "confirm"
CONFIRMED = "confirmed"
#: The status of a quest that runs no experiment of its own (:func:`mark_not_applicable`).
NOT_APPLICABLE = "not_applicable"

HELD_BACK = "held_back_data"
FRESH_SEEDS = "fresh_seeds"

#: A file needs at least this many data rows (header excluded) to have a part held back.
MIN_ROWS = 40
#: The share of each file's rows held back for the confirm run.
HOLD_BACK_FRACTION = 0.3
#: The confirm run needs at least this many held-back rows; fewer, and no rows are held back (new seeds instead).
MIN_HELD_BACK_ROWS = 10
#: Tabular files split by rows; any other data file means no data is held back.
SPLITTABLE = (".csv", ".tsv")
#: Files under ``inputs/data/`` that are not data (the same ones ``engine._pick_up_user_dropped_datasets`` skips).
_NOT_DATA = ("README.md",)

#: The paper block's markers: a later write replaces it rather than adding a second one.
_BEGIN = "<!-- fi:phased -->"
_END = "<!-- /fi:phased -->"


def enabled(config: Any) -> bool:
    """``engine.phased`` is on."""
    return bool(getattr(getattr(config, "engine", None), "phased", False))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def record_path(quest_root: Path) -> Path:
    return Path(quest_root) / ".fi" / RECORD


def load(quest_root: Path) -> dict[str, Any] | None:
    path = record_path(quest_root)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and data.get("schema") == SCHEMA else None


def _save(quest_root: Path, record: dict[str, Any]) -> None:
    path = record_path(quest_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def stage(quest_root: Path) -> str | None:
    record = load(quest_root)
    return str(record.get("stage")) if record else None


# ---- the data held back -------------------------------------------------------------------------------------------


def _data_files(quest_root: Path) -> list[Path]:
    data = Path(quest_root) / "inputs" / "data"
    if not data.is_dir():
        return []
    return sorted(p for p in data.rglob("*")
                  if p.is_file() and not p.name.startswith(".") and p.name not in _NOT_DATA)


def _rows(raw: bytes) -> tuple[str, list[str], str] | None:
    """``(header, rows, newline)`` of a file split by lines, or None when it cannot be split by rows safely (not text,
    or a quoted field that spans lines, which makes a line not a row)."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    newline = "\r\n" if "\r\n" in text else "\n"
    lines = [line for line in text.split(newline) if line.strip()]
    if not lines or any(line.count('"') % 2 for line in lines):
        return None
    return lines[0], lines[1:], newline


def _join(header: str, rows: list[str], newline: str) -> bytes:
    return (newline.join([header, *rows]) + newline).encode("utf-8")


def _why_not_splittable(rel: str, raw: bytes) -> str | None:
    if Path(rel).suffix.lower() not in SPLITTABLE:
        return f"{rel} cannot be split by rows (only CSV and TSV files can)"
    parsed = _rows(raw)
    if parsed is None:
        return f"{rel} cannot be split by rows safely (not plain text, or a quoted value spans lines)"
    if len(parsed[1]) < MIN_ROWS:
        return f"{rel} has {len(parsed[1])} rows, fewer than the {MIN_ROWS} needed to hold a part back"
    return None


def _held_back_row(quest_id: str, row: str) -> bool:
    """Whether one row is held back for the confirm run. Decided by the row's own text (with the quest), not by its
    place in the file or the file's name: a row keeps its side when the person adds rows, edits other rows, renames the
    file, or the quest starts again, so a row exploration has seen never moves into the held-back part later."""
    digest = hashlib.sha256(f"{quest_id}\0{row}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < HOLD_BACK_FRACTION


def _split(rel: str, raw: bytes, quest_id: str, allowed: set[str] | None = None) -> tuple[bytes, bytes, dict[str, Any]]:
    """Exploration's part, the held-back part and what the record keeps of them. ``allowed``: once exploration has run,
    the rows held back before it ran; only those can be held back (any other row may have been seen)."""
    header, rows, newline = _rows(raw)  # type: ignore[misc] -- checked by _why_not_splittable
    held = [(r in allowed) if allowed is not None else _held_back_row(quest_id, r) for r in rows]
    held_rows = [r for r, h in zip(rows, held) if h]
    explore_rows = [r for r, h in zip(rows, held) if not h]
    explore, back = _join(header, explore_rows, newline), _join(header, held_rows, newline)
    info = {"file": rel, "rows": len(rows), "explore_rows": len(explore_rows), "held_back_rows": len(held_rows),
            "original_sha256": _sha(raw), "explore_sha256": _sha(explore), "held_back_sha256": _sha(back)}
    return explore, back, info


def store_dir(quest_root: Path) -> Path:
    """Where this quest's whole files and held-back part are kept: beside the quest folder, never inside it
    (``<output_dir>/_held_back/<quest_id>/``). The experiment runs in the quest folder and a container is given only
    that folder, so exploration's code cannot read the held-back rows by walking it."""
    root = Path(quest_root)
    return root.parent / STORE / root.name


def _originals(quest_root: Path) -> Path:
    return store_dir(quest_root) / "original"


def _held_back(quest_root: Path) -> Path:
    return store_dir(quest_root) / "held_back"


_TMP = ".fi-tmp"


def _write(path: Path, data: bytes, quest_root: Path | None = None) -> None:
    """Write through a temporary file, so a stop in the middle never leaves a file cut short. A file in the quest
    folder (``quest_root`` given) is written first in the store beside it, never in the quest folder: a temporary copy
    of a whole file there would hold the held-back rows. A failed replace removes the temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    folder = store_dir(quest_root) / "tmp" if quest_root is not None else path.parent
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / f".{path.name}{_TMP}"
    try:
        tmp.write_bytes(data)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _clear_stray_tmp(quest_root: Path) -> None:
    """Temporary files an earlier FI left in ``inputs/data/`` (it wrote them there): they may hold held-back rows."""
    data = Path(quest_root) / "inputs" / "data"
    if data.is_dir():
        for stray in data.rglob(f".*{_TMP}"):
            stray.unlink(missing_ok=True)


def _move_out_of_quest(quest_root: Path) -> None:
    """A quest started before the kept files moved out of the quest folder: move them to :func:`store_dir`. Raises
    ``OSError`` when they cannot be moved (the held-back rows would stay where exploration's code can read them)."""
    old = Path(quest_root) / ".fi" / DIR
    if not old.is_dir():
        return
    for sub in ("original", "held_back"):
        for src in sorted(p for p in (old / sub).rglob("*") if p.is_file()) if (old / sub).is_dir() else []:
            dst = store_dir(quest_root) / sub / src.relative_to(old / sub)
            if not dst.is_file():
                _write(dst, src.read_bytes())
    shutil.rmtree(old)


def _put(quest_root: Path, files: list[dict[str, Any]], folder: Path, want: str) -> list[str]:
    """Make ``inputs/data/<file>`` hold ``folder/<file>`` for every file whose on-disk content is one of the three
    recorded parts (whole, exploration's, held back) but not ``want`` of them. Returns the files it could not make
    hold ``want``: the person changed the file (it is left alone) or the kept part is missing. A file the person
    deleted is not brought back by a restore, and is not counted."""
    missed: list[str] = []
    for info in files:
        src = folder / info["file"]
        dst = Path(quest_root) / "inputs" / "data" / info["file"]
        current = _sha(dst.read_bytes()) if dst.is_file() else None
        if current == info.get(want):
            continue
        if current is None and want == "original_sha256":
            continue  # the person deleted it: a restore does not bring it back
        if not src.is_file() or (current is not None and current not in (
                info.get("original_sha256"), info.get("explore_sha256"), info.get("held_back_sha256"))):
            missed.append(str(info["file"]))
            continue
        _write(dst, src.read_bytes(), quest_root)
    return missed


def _restore(quest_root: Path, files: list[dict[str, Any]]) -> list[str]:
    return _put(quest_root, files, _originals(quest_root), "original_sha256")


def _not_restored(quest_root: Path, missed: list[str]) -> list[str]:
    out = []
    for rel in missed:
        kept = _originals(quest_root) / rel
        out.append(f"inputs/data/{rel} was changed while the quest had a part of it in place, so it was left as it is; "
                   f"the whole file you supplied is kept in {kept}" if kept.is_file() else
                   f"inputs/data/{rel} could not be put back whole: the copy kept of it is missing from {kept.parent} "
                   f"(when a quest folder is moved, move {STORE}/<quest id>/ beside it with it)")
    return out


def restore_inputs(quest_root: Path) -> list[str]:
    """Put the whole files the person supplied back in ``inputs/data/`` (called when a run stops: finished, paused or
    failed). The next start holds the same rows back again (:func:`prepare`), so outside a running quest
    ``inputs/data/`` always holds the person's whole files. Returns plain lines for run.log about a file it could not
    put back."""
    _clear_stray_tmp(quest_root)
    try:
        _move_out_of_quest(quest_root)
    except OSError:
        pass  # the whole files are copied out before the old folder is removed: put them back first, move later
    record = load(quest_root)
    if record and record.get("files"):
        return _not_restored(quest_root, _restore(quest_root, record["files"]))
    return []


def turned_off(quest_root: Path) -> list[str]:
    """A start with ``engine.phased`` off on a quest that has a record: the whole files go back in ``inputs/data/``,
    and, since a run while it is off can see every row, exploration's data can no longer be kept apart. Still exploring:
    nothing is held back when it is turned on again (new seeds). In the confirm stage: nothing can be confirmed."""
    record = load(quest_root)
    if record is None:
        return []
    _clear_stray_tmp(quest_root)
    try:
        _move_out_of_quest(quest_root)
    except OSError:
        pass
    lines = _not_restored(quest_root, _restore(quest_root, record.get("files") or []))
    if record.get("not_applicable"):
        return lines  # the two stages never applied: nothing was confirmed or held back to say anything about
    if record.get("stage") == EXPLORE and not record.get("late_start") and not record.get("compromised"):
        why = "explore-then-confirm was turned off while exploring, so the experiment may have run on all of the data"
        record.update(late_start=True, strategy=FRESH_SEEDS, why_no_data=why, files=[])
        _save(quest_root, record)
        lines.append(f"explore-then-confirm is off: the whole data files are in inputs/data/; if it is turned on again, "
                     f"the confirm run will use new random seeds ({why})")
    elif record.get("stage") == CONFIRM and not record.get("compromised"):
        mark_compromised(quest_root, "explore-then-confirm was turned off during the confirm stage")
        lines.append("explore-then-confirm was turned off during the confirm stage: nothing in this quest can be "
                     "confirmed any more, and its numbers stay exploratory")
    return lines


def _as_supplied(quest_root: Path, rel: str, raw: bytes, info: dict[str, Any] | None, quest_id: str,
                 allowed: set[str] | None = None) -> bytes:
    """The whole file behind what ``inputs/data/<rel>`` holds now: the kept original when the file on disk is one of
    its recorded parts, or the part of an interrupted start (the original was kept but the record not yet written)."""
    kept = _originals(quest_root) / rel
    if not kept.is_file():
        return raw
    current = _sha(raw)
    if info and current in (info.get("explore_sha256"), info.get("held_back_sha256")):
        return kept.read_bytes()
    if info is None:
        original = kept.read_bytes()
        if _why_not_splittable(rel, original) is None:
            _explore, _back, parts = _split(rel, original, quest_id, allowed)
            if current in (parts["explore_sha256"], parts["held_back_sha256"]):
                return original
    return raw


def _held_back_before(quest_root: Path, record: dict[str, Any]) -> set[str] | None:
    """Once exploration has run on held-back data: the rows held back so far (only they can be held back again).
    ``None`` while exploration has not run, when any row may still be held back."""
    ran = bool(record.get("explore_seed_bases")) or int(record.get("explore_runs") or 0) > 0
    if not ran or record.get("strategy") != HELD_BACK:
        return None
    rows: set[str] = set()
    for info in record.get("files") or []:
        part = _held_back(quest_root) / info["file"]
        parsed = _rows(part.read_bytes()) if part.is_file() else None
        if parsed:
            rows.update(parsed[1])
    return rows


def prepare(quest_root: Path, quest_id: str, *, already_ran: bool = False) -> tuple[dict[str, Any], list[str]]:
    """Start the exploration stage, or bring it up to date with the data the person supplied since; in the confirm
    stage, give the run the held-back part again. Idempotent, and safe to call again after a start that was cut short.
    ``already_ran``: runs were made before this record existed (the setting was turned on mid-quest), so no data is
    unseen and nothing is held back. Returns the record and plain lines for run.log about anything it did."""
    quest_root = Path(quest_root)
    _clear_stray_tmp(quest_root)
    _move_out_of_quest(quest_root)
    record = load(quest_root)
    lines: list[str] = []
    if record is None:
        record = {"schema": SCHEMA, "stage": EXPLORE, "started_at": _now(), "strategy": FRESH_SEEDS,
                  "why_no_data": "", "files": [], "explore_seed_bases": [], "confirm_seed_base": None,
                  "explore_runs": 0, "confirm_runs": 0, "confirm_executions": 0, "results_seen_in_confirm": 0}
        lines.append("exploration stage: the model may try designs and look at results; when exploration ends the "
                     "design is frozen and run once more on data or seeds exploration never saw")
        if already_ran:
            why = "the experiment had already run on all of the supplied data before explore-then-confirm was turned on"
            record.update(late_start=True, why_no_data=why)
            lines.append(f"the confirm run will use new random seeds because no held-back data is available ({why})")
            _save(quest_root, record)
            return record, lines
    if record.get("stage") == CONFIRM:
        if record.get("compromised"):
            return record, lines
        missed = _put(quest_root, record.get("files") or [], _held_back(quest_root), "held_back_sha256")
        if missed:
            why = ("the held-back part could not be put in place of inputs/data/" + ", inputs/data/".join(missed)
                   + " (the file was changed since, or the kept part is missing)")
            mark_compromised(quest_root, why)
            lines.append(f"{why}: the confirm run would not see only the held-back rows, so nothing in this quest can "
                         "be confirmed and its numbers stay exploratory")
            record = load(quest_root) or record
        return record, lines
    if record.get("stage") != EXPLORE:
        lines += _not_restored(quest_root, _restore(quest_root, record.get("files") or []))
        return record, lines
    if record.get("late_start") or record.get("compromised") or record.get("not_applicable"):
        return record, lines
    if record.get("strategy") == FRESH_SEEDS and record.get("explore_seed_bases"):
        # Exploration has already run on the whole files: none of their rows is unseen, so none can be held back now
        # (a file that became splittable since, a blocking file removed, is not new data).
        return record, lines
    split = {info["file"]: info for info in record.get("files") or []}
    allowed = _held_back_before(quest_root, record)
    files = _data_files(quest_root)
    # What each file holds as the person supplied it: the whole file kept aside for one already split and unchanged.
    supplied: dict[str, bytes] = {}
    for path in files:
        rel = path.relative_to(quest_root / "inputs" / "data").as_posix()
        supplied[rel] = _as_supplied(quest_root, rel, path.read_bytes(), split.get(rel), quest_id, allowed)
    reasons = [why for rel, raw in supplied.items() if (why := _why_not_splittable(rel, raw))]
    if not reasons:
        reasons = [(f"{rel} now holds only {n} of the rows held back before exploration ran (the file was changed "
                    f"since), fewer than the {MIN_HELD_BACK_ROWS} a confirm run needs") if allowed is not None else
                   f"{rel} would have only {n} rows held back, fewer than the {MIN_HELD_BACK_ROWS} a confirm run needs"
                   for rel, raw in supplied.items()
                   if (n := _split(rel, raw, quest_id, allowed)[2]["held_back_rows"]) < MIN_HELD_BACK_ROWS]
    if len(supplied) > 1:
        # Rows of different files may belong together (the same patients, x and y by row order): held back file by file
        # they would land on different sides, and exploration would see a held-back record through the other file.
        reasons.append(f"there is more than one data file ({len(supplied)}), and rows of different files may belong "
                       "together, so no rows are held back")
    if not supplied:
        examples = Path(quest_root) / "inputs" / "examples"
        reasons = (["the files given in execution.inputs (inputs/examples/) are not held back: only data in "
                    "inputs/data/ can be"]
                   if examples.is_dir() and any(p.is_file() for p in examples.rglob("*"))
                   else ["no data was supplied in inputs/data/"])
    if reasons:
        if split:
            missed = _restore(quest_root, list(split.values()))
            lines.append("the supplied data is no longer held back (" + "; ".join(reasons) + ")"
                         + ("" if missed else ": the whole files are back in inputs/data/"))
            lines += _not_restored(quest_root, missed)
        changed = record.get("strategy") != FRESH_SEEDS or record.get("why_no_data") != "; ".join(reasons)
        record.update(strategy=FRESH_SEEDS, why_no_data="; ".join(reasons), files=[])
        if changed:
            lines.append("the confirm run will use new random seeds because no held-back data is available ("
                         + "; ".join(reasons) + ")")
        _save(quest_root, record)
        return record, lines
    out: list[dict[str, Any]] = []
    for rel, raw in supplied.items():
        known = split.get(rel)
        explore, back, info = _split(rel, raw, quest_id, allowed)
        # The whole file and the held-back part are kept before exploration's part replaces the file, so a start cut
        # short anywhere in between is finished by the next one (_as_supplied); what is already right is not rewritten.
        kept = _originals(quest_root) / rel
        if kept.is_file() and known and _sha(kept.read_bytes()) == known.get("original_sha256") \
                and known.get("original_sha256") != info["original_sha256"]:
            # The file changed since it was kept (edited while the quest was not running, or while a part of it was in
            # its place after a hard stop): the earlier whole file is kept too, never overwritten.
            kept.replace(kept.with_name(f"{kept.name}.replaced-{str(known['original_sha256'])[:12]}"))
        for dst, data in ((kept, raw), (_held_back(quest_root) / rel, back),
                          (quest_root / "inputs" / "data" / rel, explore)):
            if not dst.is_file() or dst.read_bytes() != data:
                _write(dst, data, quest_root)
        out.append(info)
        if not known or known.get("original_sha256") != info["original_sha256"]:
            lines.append(f"held back {info['held_back_rows']} of {info['rows']} rows of inputs/data/{rel} for the "
                         f"confirm run; exploration sees the other {info['explore_rows']} (the whole file and the "
                         f"held-back part are kept outside the quest folder)")
    record.update(strategy=HELD_BACK, why_no_data="", files=out)
    _save(quest_root, record)
    return record, lines


# ---- seeds --------------------------------------------------------------------------------------------------------


def _pick_confirm_base(explore_bases: list[int], stride: int, replicates: int) -> int:
    """A seed base on a multiple of ``stride`` that no exploration run's range ``[base, base + replicates*stride)``
    overlaps, kept under 2**31 so any generator takes it. Random (the system's own source), recorded once chosen."""
    used = {b // stride + i for b in explore_bases for i in range(max(1, replicates))}
    top = (2**31 - 1) // stride - max(1, replicates) - 1
    low = min(1000, max(0, top // 2))
    rng = random.SystemRandom()
    for _ in range(1000):
        if top <= low:
            break
        k = rng.randrange(low, top)
        if not any(k + i in used for i in range(max(1, replicates))):
            return k * stride
    k = max(used, default=-1) + max(1, replicates) + 1  # a stride so large that no random slot fits: the next free one
    return k * stride


def seed_env(quest_root: Path, env: dict[str, str]) -> dict[str, str]:
    """The environment a run is given: in the confirm stage (or after it) the replicate's seed is moved to the confirm
    seed base; in exploration the seed is kept and remembered, so the confirm base can be chosen apart from it."""
    record = load(quest_root)
    if record is None:
        return env
    try:
        seed = int(env.get("FI_REPLICATE_SEED") or 0)
        index = int(env.get("FI_REPLICATE_INDEX") or 0)
    except ValueError:
        return env
    if record.get("stage") == EXPLORE:
        bases = list(record.get("explore_seed_bases") or [])
        if seed not in bases or index == 0:
            record["explore_seed_bases"] = sorted({*bases, seed})
            if index == 0:
                record["explore_runs"] = int(record.get("explore_runs") or 0) + 1
            _save(quest_root, record)
        return env
    base = record.get("confirm_seed_base")
    if base is None:
        return env
    if index == 0:
        # Every run on the confirm data or seeds is counted: the evidence gate records a confirm result only after one
        # (``confirm_run_started``), so a gate run again on a resume cannot take exploration's result for it; and more
        # than one (a repair of the confirm run, a re-run after its result was seen) means the confirm data was looked
        # at more than once (``status``). A run that only checks on the confirm run's own background job (it was
        # pending when the quest stopped: :func:`note_job_pending`) is the same run, not a second one.
        if record.pop("confirm_job_pending", False):
            _save(quest_root, record)
        else:
            record["confirm_executions"] = int(record.get("confirm_executions") or 0) + 1
            _save(quest_root, record)
    stride = int(record.get("seed_stride") or 1)
    return {**env, "FI_REPLICATE_SEED": str(int(base) + index * stride)}


def note_job_pending(quest_root: Path) -> None:
    """The confirm run submitted a background job and the quest waits for it: the run that checks on it after a resume
    is this confirm run going on, not a second run on the confirm data (:func:`seed_env`)."""
    record = load(quest_root)
    if record is not None and record.get("stage") == CONFIRM and not record.get("confirm_job_pending"):
        record["confirm_job_pending"] = True
        _save(quest_root, record)


def note_confirm_result(quest_root: Path, result: Any) -> None:
    """What the confirm run produced (the experiment step's result, in the confirm stage): the result the evidence gate
    records must be this one (:func:`record_confirm`), not one a rerun from an earlier step brought back."""
    record = load(quest_root)
    if record is not None and record.get("stage") == CONFIRM:
        record["confirm_run_result_sha256"] = _result_digest(result) if result else None
        _save(quest_root, record)


# ---- the boundary and the confirm result --------------------------------------------------------------------------


def _result_digest(result: Any) -> str:
    return _sha(json.dumps(result, sort_keys=True, default=str).encode("utf-8"))


def _design_versions(quest_root: Path) -> int:
    """How many versions of the design ``needs/DESIGN_HISTORY.json`` holds (0 when there is none or it cannot be read)."""
    try:
        history = json.loads((Path(quest_root) / "needs" / "DESIGN_HISTORY.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0
    return len(history) if isinstance(history, list) else 0


def enter_confirm(quest_root: Path, *, explore_result: Any, frozen_sha256: str | None, stride: int,
                  replicates: int, explore_runs: int) -> tuple[dict[str, Any], list[str]]:
    """End exploration: record what it produced, choose the confirm seed base, and (``held_back_data``) put the held-back
    part in ``inputs/data/`` in place of the part exploration saw. Idempotent once in the confirm stage."""
    quest_root = Path(quest_root)
    record = load(quest_root)
    if record is None:
        record, _ = prepare(quest_root, quest_root.name)
    if record.get("stage") != EXPLORE:
        return record, []
    explore_bases = [int(b) for b in record.get("explore_seed_bases") or []] or [0]
    base = _pick_confirm_base(explore_bases, max(1, int(stride)), max(1, int(replicates)))
    explore_runs = max(int(explore_runs), int(record.get("explore_runs") or 0))
    lines = [f"exploration ended after {explore_runs} run(s); the protocol is frozen and the confirm stage runs the "
             "frozen design once"]
    if record.get("strategy") == HELD_BACK:
        lines.append("confirm stage: the run is given only the data held back before exploration began ("
                     + ", ".join(f"{i['held_back_rows']} rows of inputs/data/{i['file']}" for i in record["files"])
                     + f") and a new seed base ({base}) that exploration never used")
    else:
        lines.append(f"confirm stage: the run is given new random seeds (seed base {base}; exploration used "
                     f"{', '.join(map(str, explore_bases))}) because no held-back data was available "
                     f"({record.get('why_no_data') or 'no reason recorded'})")
    record.update(stage=CONFIRM, confirm_started_at=_now(), confirm_seed_base=base, seed_stride=max(1, int(stride)),
                  frozen_sha256=frozen_sha256, explore_runs=max(int(explore_runs), int(record.get("explore_runs") or 0)),
                  explore_result_sha256=_result_digest(explore_result) if explore_result else None,
                  explore_result=explore_result, confirm_executions=0, confirm_run_result_sha256=None,
                  # How many versions of the design there were when exploration ended: a later one was made after the
                  # confirm run began, which then no longer confirms it (core/disclosure.py).
                  design_revisions_at_confirm=_design_versions(quest_root))
    # The stage is saved before the held-back part replaces exploration's: a start after a stop in between gives the
    # run the held-back part (``prepare`` in the confirm stage), never exploration's part again as new data.
    _save(quest_root, record)
    missed = _put(quest_root, record.get("files") or [], _held_back(quest_root), "held_back_sha256")
    if missed:
        why = ("the held-back part could not be put in place of inputs/data/" + ", inputs/data/".join(missed)
               + " (the file was changed during exploration, or the kept part is missing)")
        mark_compromised(quest_root, why)
        lines.append(f"{why}: nothing in this quest can be confirmed, so its numbers stay exploratory")
        record = load(quest_root) or record
    return record, lines


def confirm_run_started(quest_root: Path) -> bool:
    """In the confirm stage, whether the confirm run has started (its first execution was given the confirm seeds)."""
    record = load(quest_root)
    return bool(record and int(record.get("confirm_executions") or 0) > 0)


def record_confirm(quest_root: Path, result: Any) -> tuple[dict[str, Any] | None, list[str]]:
    """The result that reached the paper in the confirm stage (``None`` or empty: the confirm run produced none). The
    first one is the confirmed result, if the confirm run was made once. Any further run on the confirm data or seeds (a
    repair of the confirm run, a redesign or a re-run the review asked for) means the confirm data was looked at more
    than once, whatever that run produced."""
    quest_root = Path(quest_root)
    record = load(quest_root)
    if record is None or record.get("stage") == EXPLORE:
        return record, []
    lines: list[str] = []
    digest = _result_digest(result) if result else None
    runs = int(record.get("confirm_executions") or 0)
    if record.get("stage") == CONFIRM:
        # The result in hand must be the one the confirm run produced (``note_confirm_result``). A rerun from an earlier
        # step (``--from evidence``) can bring back exploration's result after the confirm run started; with no note
        # (the confirm run stopped before its result, or a record from before the note), exploration's own result is
        # recognised by its digest.
        produced = record.get("confirm_run_result_sha256")
        replayed = digest is not None and (
            digest != produced if produced else digest == record.get("explore_result_sha256"))
        record.update(stage=CONFIRMED, confirmed_at=_now(), confirm_result_sha256=digest, confirm_runs=1,
                      results_seen_in_confirm=1 if digest else 0, confirm_executions_recorded=runs)
        if replayed:
            record["compromised"] = ("the result that reached the paper is not the one the confirm run produced (a run "
                                     "from an earlier step brought back exploration's result)")
            lines.append(f"confirm stage: {record['compromised']}, so nothing in this quest is confirmed")
        elif (digest is not None and record.get("strategy") == FRESH_SEEDS
              and digest == record.get("explore_result_sha256")):
            # New seeds gave exactly exploration's numbers: the run does not depend on its seed (or ignores it), so it
            # repeated exploration's run rather than confirming it.
            record["not_confirmable"] = ("the confirm run on new seeds gave exactly exploration's numbers, so the study "
                                         "does not depend on its seed and the run only repeated exploration's")
            lines.append(f"confirm stage: {record['not_confirmable']}; nothing in this quest is confirmed")
        elif digest is None:
            lines.append("confirm stage: the confirm run produced no result, so nothing in this quest is confirmed")
        elif runs > 1:
            lines.append(f"confirm stage: the confirm run was changed and run again on the confirm data or seeds ({runs} "
                         "runs in all) after its first result was seen, so its numbers are preliminary, not confirmed")
        else:
            lines.append("confirm stage: the confirm run's result is recorded; the paper reports these numbers as "
                         "confirmed and exploration's numbers as exploratory")
        _save(quest_root, record)
        missed = _restore(quest_root, record.get("files") or [])
        if record.get("files") and not missed:
            lines.append("the whole data files are back in inputs/data/")
        lines += _not_restored(quest_root, missed)
        return record, lines
    seen = int(record.get("confirm_executions_recorded") or 0)
    if runs <= seen and (digest is None or digest == record.get("confirm_result_sha256")):
        return record, []
    record["confirm_executions_recorded"] = max(runs, seen)
    if digest is not None and digest != record.get("confirm_result_sha256"):
        record["confirm_runs"] = int(record.get("confirm_runs") or 1) + 1
        record["results_seen_in_confirm"] = int(record.get("results_seen_in_confirm") or 0) + 1
        record["confirm_result_sha256"] = digest
    lines.append(f"the frozen design was run again after the confirm numbers were seen ({runs} runs on the confirm data "
                 "or seeds in all): the numbers are no longer from one untouched confirm run, so they are preliminary")
    _save(quest_root, record)
    return record, lines


def mark_compromised(quest_root: Path, why: str) -> None:
    """The confirm run could not be kept apart from exploration (a file could not be written or put in place, the
    result in hand is not the confirm run's, the experiment does not take the seed it is given): nothing in this quest
    can be confirmed any more. With no record yet (a first start that failed), a record saying so is written, so the
    next start does not hold rows back that exploration may have seen. Best effort: when the record cannot be written
    either, it is missing or unchanged, and a missing record keeps the numbers exploratory too."""
    record = load(quest_root)
    if record is not None and record.get("compromised"):
        return
    if record is None:
        record = {"schema": SCHEMA, "stage": EXPLORE, "started_at": _now(), "strategy": FRESH_SEEDS,
                  "why_no_data": "", "files": [], "explore_seed_bases": [], "confirm_seed_base": None,
                  "explore_runs": 0, "confirm_runs": 0, "confirm_executions": 0, "results_seen_in_confirm": 0}
    record["compromised"] = why
    try:
        _save(quest_root, record)
    except OSError:
        pass


def mark_unconfirmable(quest_root: Path, why: str) -> None:
    """No data was held back and a run on new seeds could only repeat exploration's run (the study has no randomness,
    or the experiment ignores the seed): this quest cannot be confirmed, and says why."""
    record = load(quest_root)
    if record is None or record.get("not_confirmable"):
        return
    record["not_confirmable"] = why
    try:
        _save(quest_root, record)
    except OSError:
        pass  # best effort, as mark_compromised: the record stays in exploration, so nothing is confirmed either


def unconfirmable(quest_root: Path) -> bool:
    """Nothing in this quest can be confirmed any more (``compromised`` or ``not_confirmable``), or there is nothing to
    confirm (``not_applicable``)."""
    record = load(quest_root)
    return bool(record and (record.get("compromised") or record.get("not_confirmable") or record.get("not_applicable")))


#: What run.log and the plan say for a quest that runs no experiment of its own (``why`` is the reason, in plain words).
_NOT_APPLICABLE = ("explore-then-confirm does not apply: {why}, so there is no design to run once more on data or seeds "
                   "the exploration never saw, and the result is not confirmed that way")


def not_applicable_sentence(why: str) -> str:
    """The one plain sentence said when the two stages do not apply to this quest."""
    return _NOT_APPLICABLE.format(why=why)


def mark_not_applicable(quest_root: Path, why: str) -> list[str]:
    """This quest runs no experiment of its own (``why``: a literature survey, no simulation, ``--analyze``): there is no
    design to run once more. Any rows held back are put back in ``inputs/data/`` (the analysis must see all of the data
    the person supplied) and the record says why, so a later start holds nothing back. Returns the plain lines for
    run.log, the first time only. A quest already past exploration is left as it is (it did run an experiment)."""
    quest_root = Path(quest_root)
    record = load(quest_root)
    if record is not None and (record.get("not_applicable") or record.get("stage") != EXPLORE):
        return []
    if record is None:
        record = {"schema": SCHEMA, "stage": EXPLORE, "started_at": _now(), "strategy": FRESH_SEEDS,
                  "why_no_data": "", "files": [], "explore_seed_bases": [], "confirm_seed_base": None,
                  "explore_runs": 0, "confirm_runs": 0, "confirm_executions": 0, "results_seen_in_confirm": 0}
    _clear_stray_tmp(quest_root)
    files = record.get("files") or []
    missed = _restore(quest_root, files)
    record.update(not_applicable=why, files=[])
    _save(quest_root, record)
    back = [" (the rows held back at the start are back in inputs/data/)"] if files and not missed else [""]
    return [not_applicable_sentence(why) + back[0], *_not_restored(quest_root, missed)]


def began_before_default(quest_root: Path, *, set_in_config: bool, has_run: bool) -> bool:
    """Whether this research quest began before research quests explored first and confirmed once by default, so it
    goes on as it began, without the two stages: its config does not set ``engine.phased`` (``set_in_config``), it has
    run before (``has_run``: a step of the graph ran, or its saved state exists), and it has no record of the two stages
    (every quest that ran with them has one from its first start)."""
    return not set_in_config and has_run and not record_path(quest_root).exists()


#: The trace event a start with the two stages on writes (``Engine.run``): a quest that has it ran with them.
STARTED_EVENT = "phased_started"


def ran_without(quest_root: Path) -> bool:
    """Whether a step of this quest's graph ran on an earlier start with the two stages never on: its hash-chained trace
    has a step and no :data:`STARTED_EVENT` (so a quest that ran with them is never taken for one that began before the
    default, even when its record is lost); without a trace, its saved state exists."""
    from . import audit_log as _audit_log

    fi = Path(quest_root) / ".fi"
    trace = fi / "audit.jsonl"
    if trace.is_file():
        events = _audit_log.read(trace)
        if any(e.get("kind") == STARTED_EVENT for e in events):
            return False
        return any(str(e.get("kind") or "").startswith("node_") for e in events)
    return (fi / "state.sqlite").is_file()


def kept_off(quest_root: Path, config: Any) -> bool:
    """A research quest that has the two stages only from the research default (its own ``config.yaml`` does not set
    ``engine.phased``) and began before that default: it goes on as it began, without them. The engine runs it so
    (:func:`without`), and ``--update`` approves it so, so the approved settings say what runs. A quest built in code
    (no ``config.yaml``) counts as having set it."""
    if not enabled(config) or getattr(config, "rigor_profile", "default") != "research":
        return False
    from . import plan_settings as _plan_settings

    sets = _plan_settings.config_sets(Path(quest_root), "engine.phased")
    return began_before_default(quest_root, set_in_config=sets is not False, has_run=ran_without(quest_root))


def without(config: Any) -> Any:
    """A copy of ``config`` with the two stages off (the caller's object is not changed)."""
    return config.model_copy(update={"engine": config.engine.model_copy(update={"phased": False})})


def status(record: dict[str, Any] | None) -> str:
    """``explore`` (no confirm run yet), ``confirming``, ``confirmed``, ``confirm_failed`` (it produced no result),
    ``confirm_reused`` (the confirm data or seeds were run on more than once), ``compromised`` (the confirm run could
    not be kept apart from exploration: its held-back data, its new seeds or its result), ``not_confirmable`` (no data
    was held back and new seeds could not change the result) or ``not_applicable`` (the quest runs no experiment of its
    own, so there is nothing to confirm)."""
    if not record:
        return EXPLORE
    if record.get("not_applicable"):
        return NOT_APPLICABLE
    if record.get("compromised"):
        return "compromised"
    if record.get("not_confirmable"):
        return "not_confirmable"
    if record.get("stage") == EXPLORE:
        return EXPLORE
    if record.get("stage") == CONFIRM:
        return "confirming"
    if not record.get("confirm_result_sha256"):
        return "confirm_failed"
    if int(record.get("results_seen_in_confirm") or 0) > 1 or int(record.get("confirm_executions") or 0) > 1:
        return "confirm_reused"
    return CONFIRMED


def evidence_settings(quest_root: Path) -> dict[str, str]:
    """What the evidence ladder is told (``core/evidence.py``: ``settings["phased"]``)."""
    record = load(quest_root)
    if status(record) == NOT_APPLICABLE:  # no seeds and no data were set apart: there was nothing to confirm
        return {"phased": NOT_APPLICABLE, "phased_strategy": "",
                "phased_why_no_data": str((record or {}).get("not_applicable") or "")}
    return {"phased": status(record), "phased_strategy": str((record or {}).get("strategy") or FRESH_SEEDS),
            "phased_why_no_data": str((record or {}).get("why_no_data") or "")}


def _how_confirmed(record: dict[str, Any]) -> str:
    if record.get("strategy") == HELD_BACK:
        return "on a part of the supplied data that was held back at random before exploration began, and on new seeds"
    # No file names or row counts here: every number in the paper is held to the results (the full reason is in run.log
    # and .fi/phased.json).
    why = str(record.get("why_no_data") or "")
    reason = ("the experiment had already run before explore-then-confirm was turned on" if record.get("late_start")
              else "no data was supplied" if why.startswith("no data")
              else "the data was given as example inputs, which are not held back" if "execution.inputs" in why
              else "the data came in more than one file, whose rows may belong together" if "more than one data file" in why
              else "the supplied data was too small, or could not be split by rows")
    return ("on new random seeds that exploration never used (confirmed on new random seeds because no held-back data "
            f"was available: {reason})")


def summary(record: dict[str, Any] | None) -> str:
    """One plain sentence on which numbers are exploratory and which are confirmed (no numbers of the result in it: the
    paper's number audits hold every number in the paper to the results)."""
    state = status(record)
    if state == NOT_APPLICABLE:
        return ("This quest runs no experiment of its own, so there was no design to run once more on data or seeds that "
                "exploration never saw; nothing here is confirmed that way.")
    if state == EXPLORE:
        return ("These numbers come from the exploration stage, in which the design could still be changed after its "
                "results were seen, and they have not been confirmed on data or seeds that exploration never saw. "
                "Treat them as exploratory.")
    how = _how_confirmed(record or {})
    if state == "confirming":
        return f"The design was frozen when exploration ended; its confirm run {how} has not finished."
    if state == "confirm_failed":
        return (f"The design was frozen when exploration ended and run once more {how}, but that confirm run produced "
                "no result, so nothing here is confirmed; any numbers are exploratory.")
    if state == "confirm_reused":
        return (f"The design was frozen when exploration ended and run {how}; it was then changed or run again after "
                "the confirm results had been seen, so the numbers are no longer from one untouched confirm run. Treat "
                "them as exploratory.")
    if state == "not_confirmable":
        return ("No data was held back from exploration, and the study gives the same numbers whatever its random "
                "seeds (it has no randomness, or does not take the seed it is given), so a confirm run could only repeat "
                "exploration's run. Nothing here is confirmed; treat the numbers as exploratory.")
    if state == "compromised":
        return ("The confirm run could not be kept apart from exploration (its held-back data, its new seeds or its "
                "result), so nothing here is confirmed. Treat the numbers as exploratory.")
    return (f"The design was tried, and could be changed, in an exploration stage, then frozen and run once more {how}. "
            "The numbers reported as results are that confirm run's; exploration's numbers are exploratory and are not "
            "reported as findings.")


#: plan.md's section on confirming the result.
PLAN_HEADING = "How the result will be confirmed"

#: Said (plan.md, run.log) for a research quest that turned the two stages off.
OFF_SENTENCE = ("the result will not be confirmed once more on data or seeds the exploration never saw: explore-then-"
                "confirm is off (`engine.phased: false`), so the design may be changed after its results are seen and "
                "the numbers reported come from those runs")
def off_sentence(runs_code: bool, why: str = "") -> str:
    """What a research quest with the two stages turned off is told: :data:`OFF_SENTENCE`, or, for a quest that runs no
    experiment of its own (``why``), that they would not apply anyway."""
    return OFF_SENTENCE if runs_code else not_applicable_sentence(why or "this quest runs no experiment of its own")


def kept_off_sentence(runs_code: bool) -> str:
    """What a research quest that began before the two stages were on by default for research is told (plan.md,
    run.log). How to turn them on is said only to a quest that runs an experiment: for one that does not, they would
    not apply."""
    text = ("this quest began before research quests explored first and confirmed once by default, so it goes on as it "
            "began: the result will not be confirmed once more on data or seeds the exploration never saw")
    if runs_code:
        text += (" (to confirm it, set `engine.phased: true` in its config.yaml and approve that change with "
                 "`--update`)")
    return text


def plan_lines(record: dict[str, Any] | None, *, on: bool, research: bool, runs_code: bool, why: str = "",
               kept_off: bool = False) -> list[str]:
    """plan.md's section saying whether and how the result will be confirmed, and what the confirm run costs. Silent
    under the default profile with the two stages off (nothing changes there). ``why``: why the quest runs no
    experiment of its own, when it does not. Nothing here is read back."""
    head = [f"## {PLAN_HEADING}", ""]

    def said(text: str) -> list[str]:
        return head + [text[0].upper() + text[1:] + ".", ""]

    if not on:
        if not research:
            return []
        return said(kept_off_sentence(runs_code) if kept_off else off_sentence(runs_code, why))
    if not runs_code or status(record) == NOT_APPLICABLE:
        return said(not_applicable_sentence(
            str((record or {}).get("not_applicable") or why or "this quest runs no experiment of its own")))
    if (record or {}).get("compromised") or (record or {}).get("not_confirmable"):
        reason = str(record.get("compromised") or record.get("not_confirmable"))
        return said(f"the result cannot be confirmed once more on data or seeds the exploration never saw ({reason}), "
                    "so its numbers stay exploratory")
    if (record or {}).get("strategy") == HELD_BACK and record.get("files"):
        held = ", ".join(f"{i['held_back_rows']} of the {i['rows']} rows of `inputs/data/{i['file']}`" for i in record["files"])
        on_what = f"on data exploration never saw ({held}, held back at random before exploration began) and on new seeds"
    else:
        reason = str((record or {}).get("why_no_data") or "no data was supplied in inputs/data/")
        on_what = f"on new random seeds exploration never used (no data is held back: {reason})"
    return head + [
        "The model may try designs and look at results first (exploration). When exploration ends, the design is "
        f"frozen and run once more {on_what}. Only that run's numbers can be publication-ready; exploration's numbers "
        "are reported as exploratory.",
        "",
        "This costs one more full run of the frozen design: every setting again, with as many runs per setting as an "
        "exploration run, about as long as one run during exploration, and the checks after it. The confirm run is made "
        "once: a second run on the confirm data or seeds (a repair, a redesign) leaves the result preliminary.",
        "",
    ]


def write_note(quest_root: Path) -> str:
    """What the writer is told about the two stages (appended to the evidence note). Empty when they do not apply (the
    quest runs no experiment of its own): there is nothing to say about stages it never had."""
    record = load(quest_root)  # None (the record is missing): nothing is confirmed, and the note says so
    if status(record) == NOT_APPLICABLE:
        return ""
    return ("This quest ran in two stages, exploration then confirmation. State this in the methods, in these terms: "
            + summary(record) + " Report only the numbers in the results you are given as the findings, and do not "
            "call exploratory numbers confirmed.")


def mark_paper(markdown: str, record: dict[str, Any] | None) -> str:
    """``markdown`` with the stage note under its title (one, replacing an earlier one). No record: nothing is
    confirmed, and the note says the numbers are exploratory. The two stages do not apply (the quest runs no experiment
    of its own): no note, and an earlier one is taken out."""
    if status(record) == NOT_APPLICABLE:
        if _BEGIN in markdown and _END in markdown:
            head, rest = markdown.split(_BEGIN, 1)
            return head.rstrip("\n") + "\n\n" + rest.split(_END, 1)[1].lstrip("\n")
        return markdown
    block = f"{_BEGIN}\n> **Exploratory and confirmed numbers.** {summary(record)}\n{_END}"
    if _BEGIN in markdown and _END in markdown:
        head, rest = markdown.split(_BEGIN, 1)
        return head + block + rest.split(_END, 1)[1]
    lines = markdown.split("\n")
    at, fenced = 0, False
    for i, line in enumerate(lines):
        if line.lstrip().startswith(("```", "~~~")):
            fenced = not fenced
        elif not fenced and line.startswith("# "):
            at = i + 1
            break
    return "\n".join([*lines[:at], "", block, "", *lines[at:]])
