"""Explore, then confirm: a quest run in two stages (``engine.phased: true``; off by default, on by default under
``rigor_profile: research``).

In the EXPLORATION stage the model may try designs, look at results and change the design, as a quest always could.
When exploration ends (the quest is about to write its paper on an accepted result) the protocol is frozen and the
frozen design is run ONCE more in the CONFIRM stage, on data or seeds the exploration never saw. Only the confirm
run's numbers can back a publication-ready claim; while a quest has none, its result is preliminary
(``core/evidence.py`` reads :func:`evidence_settings`).

What the confirm run is given:

- ``held_back_data``: the person supplied one tabular data file (``inputs/data/``) that can be split by rows, with
  at least :data:`MIN_ROWS` rows (and at least :data:`MIN_HELD_BACK_ROWS` of them held back). At the start
  (:func:`prepare`) about :data:`HOLD_BACK_FRACTION` of its rows are set aside, so nothing that runs before the split is
  decided can read them. The split itself is decided right before the data is first read (:func:`decide_split`: the
  experiment's first run, or a data quest's first reading), from the plan's ``protocol.split``, your answer in plan.md,
  or the table's columns (``core/phased_data.py``); rows are split one by one only when something says they are
  independent, otherwise whole subjects, sites or other units, or the latest period, are held back. When nothing says
  which rows belong together, a research quest asks once; any other quest holds nothing back. Every split records a
  manifest whose zero-overlap check (no unit in both parts) must pass. The confirm run is also given a new seed base.
- ``fresh_seeds``: otherwise (no data, too few rows, a file that cannot be split by rows, more than one data file:
  rows of different files may belong together, and split file by file a held-back record would show in the other). The confirm run is given a
  seed base exploration never used, and the report says it was confirmed on new random seeds and why no data was held
  back.

The whole files the person supplied, and the part held back, are kept OUTSIDE the quest folder, in
``<output_dir>/_held_back/<quest_id>/{original,held_back}/`` (:func:`store_dir`). That alone is not isolation (a script
run in the quest folder can open ``../_held_back``): without a container the kept files are written encrypted with a
key that only the running FI holds (``core/phased_isolation.py``), and the quest's code is read for paths out of the
quest folder before the confirm run; with a container nothing mounts them. The record says which (``isolation``). The
whole files are put back in place as soon as the confirm run's result is recorded, and whenever a run stops (finished,
paused or failed); the next start holds the same rows back again. Outside a running quest ``inputs/data/`` holds the
person's whole files. A run of FI that is killed cannot put back the rows it kept encrypted: the next start says so and
nothing in the quest can be confirmed. Once exploration has run, only rows that were held back before can be held back
again (a renamed or re-exported file cannot move a row exploration saw into the confirm part).

The confirm run is made once, and it is the confirm run's own result that is recorded: any second run on the confirm
data or seeds, a result that is not the one the confirm run produced, a held-back part that could not be put in place,
or an experiment that does not take the seed it is given (new-seeds confirm) leaves nothing confirmed.

A quest that runs no experiment of its own (a literature survey, ``--analyze``) has no design to run once more:
:func:`mark_not_applicable` puts any held-back rows back (the analysis must see all of the data), records why, and the
quest goes on as it would without the two stages; its paper carries no stage note and the evidence ladder no stage gap.

A no-simulation quest that analyses data (``data_quest`` in the record) is confirmed on held-back rows too: its table
is read from ``data/`` only (where the quest asks for it, and what its data-reading step reads; a table in
``inputs/data/`` holds nothing back), the rows are held back by the same rules, and the confirm run
is the quest's own data-reading step (``data_load``) run once more by the engine with the frozen design on the
held-back rows only (the model reads those rows; it does not change the design). When its data cannot be split (:func:`data_quest_gate`, the first time the data is read) it
holds nothing back and is ``not_applicable``, which keeps a design revised after the analysis below publication-ready
(``core/disclosure.py``).

A research quest that began before research turned this on by default (its config does not set ``engine.phased``) goes
on as it began, without the two stages: :func:`began_before_default` tells the engine so, and the engine says it in
run.log.

Each version that reaches a confirm run is a *candidate* (``candidate`` in the record, 1 for the first). When the
confirm stage begins its code, protocol and environment are hashed (``frozen_at_confirm``, :func:`fingerprint`), and
the confirm run's verdict is appended to ``.fi/confirmations.jsonl`` (``core/confirmations.py``), which is never
rewritten. The same candidate run again on the confirm data or seeds adds a second verdict beside the first (the result
is then preliminary, as before). A candidate whose code or protocol (a data quest: its design) changed after its
confirm run began is replaced by a NEW candidate (:func:`new_candidate`): back in exploration, with its own confirm run
to come on seeds no earlier run used; the earlier verdict stays. Held-back rows that a confirm run read are not unseen
any more, so a new candidate of a quest that held rows back cannot be confirmed (``not_confirmable``, said why). The
record here holds the current candidate only; the evidence reads it, checked against its verdict lines.

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

#: Where a held-back file lives in the quest, relative to the quest folder. A record's file without a ``folder`` is in
#: ``inputs/data/`` (every record written before data quests).
INPUTS = "inputs/data"
#: A data quest's own data folder (``data/``), where the quest asks for the data to analyse.
DATA = "data"
#: Subfolders of ``data/`` that hold what the quest gathered itself (literature, pages and tables it collected, a
#: simulation's results): reading material shown to both stages, never the person's data, never split.
COLLECTED = ("literature", "auto_collected", "results")

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


def _data_files(quest_root: Path, *, data_quest: bool = False) -> list[tuple[str, str, Path]]:
    """``(folder, file, path)`` of every data file: those in ``inputs/data/`` and, for a data quest, the person's files in
    ``data/`` (not FI's README there, nor what the quest gathered itself, :data:`COLLECTED`)."""
    out: list[tuple[str, str, Path]] = []
    for folder in ((INPUTS, DATA) if data_quest else (INPUTS,)):
        base = Path(quest_root) / folder
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*")):
            if not p.is_file() or p.name.startswith(".") or (p.name in _NOT_DATA and p.parent == base):
                continue
            rel = p.relative_to(base).as_posix()
            if folder == DATA and rel.split("/", 1)[0] in COLLECTED:
                continue
            if folder == INPUTS and p.name in _NOT_DATA:
                continue
            out.append((folder, rel, p))
    return out


def _folder(info: dict[str, Any]) -> str:
    return str(info.get("folder") or INPUTS)


def shown(info: dict[str, Any]) -> str:
    """The file as a person finds it in the quest folder (``inputs/data/x.csv``, ``data/x.csv``)."""
    return f"{_folder(info)}/{info['file']}"


def _on_disk(quest_root: Path, info: dict[str, Any]) -> Path:
    return Path(quest_root) / _folder(info) / str(info["file"])


def _kept(quest_root: Path, info: dict[str, Any], part: str) -> Path:
    """Where the kept ``part`` (``original`` or ``held_back``) of one file is: ``store_dir/<part>/`` for ``inputs/data/``
    (as every earlier record has it), ``store_dir/data/<part>/`` for ``data/``."""
    base = store_dir(quest_root) / DATA if _folder(info) == DATA else store_dir(quest_root)
    return base / part / str(info["file"])


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


def _split(rel: str, raw: bytes, quest_id: str, allowed: set[str] | None = None, *, folder: str = INPUTS,
           rule: dict[str, Any] | None = None) -> tuple[bytes, bytes, dict[str, Any]]:
    """Exploration's part, the held-back part and what the record keeps of them. ``allowed``: once exploration has run,
    the rows held back before it ran; only those can be held back (any other row may have been seen). ``rule``: the
    split decided before the data was first read (``core/phased_data.py``), applied again the same way; without one,
    the rows set aside at the start (each by a hash of its text) until it is decided. Rows a time split leaves out
    between the two parts (its embargo) are in neither. The record keeps the rule and its manifest, with the zero-overlap
    check run on the parts as written."""
    from . import phased_data as _data

    header, rows, newline = _rows(raw)  # type: ignore[misc] -- checked by _why_not_splittable
    dropped: set[str] = set()
    manifest: dict[str, Any] = {}
    if allowed is not None:
        held = [r in allowed for r in rows]
        if rule is not None:
            # The rows a time split leaves out between the two parts stay out on every later reading too, and the
            # manifest keeps saying so.
            decision = _data.decide(rel, header, rows, _delimiter(rel), quest_id, declared_split=rule)
            dropped, manifest = decision.dropped - allowed, decision.manifest
            # A time split: rows at or after where exploration's part ended (the left-out periods, and rows the person
            # added in later periods since) stay out of exploration, whatever the rule gives on the table as it is now.
            dropped |= _data.not_before(header, rows, rule, _delimiter(rel)) - allowed
    elif rule is not None:
        decision = _data.decide(rel, header, rows, _delimiter(rel), quest_id, declared_split=rule)
        picked, dropped, manifest = decision.held, decision.dropped, decision.manifest
        held = [r in picked for r in rows]
    else:
        held = [_held_back_row(quest_id, r) for r in rows]
    held_rows = [r for r, h in zip(rows, held) if h]
    explore_rows = [r for r, h in zip(rows, held) if not h and r not in dropped]
    explore, back = _join(header, explore_rows, newline), _join(header, held_rows, newline)
    info: dict[str, Any] = {
        "file": rel, "rows": len(rows), "explore_rows": len(explore_rows), "held_back_rows": len(held_rows),
        "original_sha256": _sha(raw), "explore_sha256": _sha(explore), "held_back_sha256": _sha(back)}
    if folder != INPUTS:
        info["folder"] = folder
    if rule is not None:
        info["rule"] = dict(rule)
        info["manifest"] = {**manifest, "rows_explore": len(explore_rows), "rows_held_back": len(held_rows),
                            "rows_left_out": len(rows) - len(explore_rows) - len(held_rows),
                            # Run on the parts as written, so it also covers a split applied again after exploration ran.
                            "overlap": _data.overlap(header, explore_rows, held_rows, rule, _delimiter(rel))}
    return explore, back, info


def _delimiter(rel: str) -> str:
    return "\t" if Path(rel).suffix.lower() == ".tsv" else ","


def store_dir(quest_root: Path) -> Path:
    """Where this quest's whole files and held-back part are kept: beside the quest folder, never inside it
    (``<output_dir>/_held_back/<quest_id>/``). The experiment runs in the quest folder and a container is given only
    that folder, so exploration's code cannot read the held-back rows by walking it."""
    root = Path(quest_root)
    return root.parent / STORE / root.name


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
    """Temporary files an earlier FI left in ``inputs/data/`` (it wrote them there), or in the store's ``tmp/`` (a run
    stopped between writing one and putting it in place): they may hold held-back rows unencrypted."""
    for folder in (Path(quest_root) / "inputs" / "data", store_dir(quest_root) / "tmp"):
        if folder.is_dir():
            for stray in folder.rglob(f".*{_TMP}"):
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


def _read_kept(path: Path, key: bytes | None) -> bytes | None:
    """A kept file's plain bytes; None when it is missing, or encrypted with a key this run of FI does not have."""
    from . import phased_isolation as _iso

    if not path.is_file():
        return None
    return _iso.unseal(path.read_bytes(), key)


def _write_kept(path: Path, data: bytes, quest_root: Path, key: bytes | None) -> None:
    """Write a kept file (a whole file, a held-back part): encrypted when this run has a key (no container)."""
    from . import phased_isolation as _iso

    _write(path, _iso.seal(data, key), quest_root)


def _put(quest_root: Path, files: list[dict[str, Any]], part: str, want: str,
         key: bytes | None = None) -> list[dict[str, Any]]:
    """Make each recorded file in the quest (``inputs/data/<file>``, or ``data/<file>`` for a data quest's) hold its kept
    ``part`` (``original`` or ``held_back``) when its on-disk content is one of the three recorded parts (whole,
    exploration's, held back) but not ``want`` of them. Returns the files it could not make hold ``want``: the person
    changed the file (it is left alone), or the kept part is missing or encrypted with a key this run does not have. A
    file the person deleted is not brought back by a restore, and is not counted."""
    missed: list[dict[str, Any]] = []
    for info in files:
        src = _kept(quest_root, info, part)
        dst = _on_disk(quest_root, info)
        current = _sha(dst.read_bytes()) if dst.is_file() else None
        if current == info.get(want):
            continue
        if current is None and want == "original_sha256":
            continue  # the person deleted it: a restore does not bring it back
        if current is not None and current not in (
                info.get("original_sha256"), info.get("explore_sha256"), info.get("held_back_sha256")):
            missed.append(info)
            continue
        data = _read_kept(src, key)
        if data is None:
            missed.append(info)
            continue
        _write(dst, data, quest_root)
    return missed


def _restore(quest_root: Path, files: list[dict[str, Any]], key: bytes | None = None) -> list[dict[str, Any]]:
    return _put(quest_root, files, "original", "original_sha256", key)


def _locked(path: Path) -> bool:
    """A kept file written encrypted (by this run of FI or an earlier one)."""
    from . import phased_isolation as _iso

    try:
        with path.open("rb") as f:
            return _iso.sealed(f.read(len(_iso.MAGIC)))
    except OSError:
        return False


def _not_restored(quest_root: Path, missed: list[dict[str, Any]], key: bytes | None = None) -> list[str]:
    out = []
    for info in missed:
        kept = _kept(quest_root, info, "original")
        on_disk = _on_disk(quest_root, info)
        changed = on_disk.is_file() and _sha(on_disk.read_bytes()) not in (
            info.get("original_sha256"), info.get("explore_sha256"), info.get("held_back_sha256"))
        if changed and kept.is_file():
            out.append(f"{shown(info)} was changed while the quest had a part of it in place, so it was left as it is; "
                       f"the whole file you supplied is kept in {kept}"
                       + (" (encrypted: only the run of FI that wrote it could read it)" if _locked(kept) else ""))
        elif kept.is_file() and _read_kept(kept, key) is None:
            out.append(f"{shown(info)} could not be put back whole: FI stopped without putting it back (it was "
                       "killed), and the rows it held back were kept encrypted with a key only "
                       f"that run of FI had, so they cannot be read again. {on_disk} now holds only exploration's part: "
                       "put your own copy of the whole file there so the quest has all of your data again (the result "
                       "can no longer be confirmed either way)")
        else:
            out.append(f"{shown(info)} could not be put back whole: the copy kept of it is missing from {kept.parent} "
                       f"(when a quest folder is moved, move {STORE}/<quest id>/ beside it with it)")
    return out


def restore_inputs(quest_root: Path, key: bytes | None = None) -> list[str]:
    """Put the whole files the person supplied back in ``inputs/data/`` (called when a run stops: finished, paused or
    failed). The next start holds the same rows back again (:func:`prepare`), so outside a running quest
    ``inputs/data/`` always holds the person's whole files. ``key``: this run's key, which the kept files were written
    with. Returns plain lines for run.log about a file it could not put back."""
    _clear_stray_tmp(quest_root)
    try:
        _move_out_of_quest(quest_root)
    except OSError:
        pass  # the whole files are copied out before the old folder is removed: put them back first, move later
    record = load(quest_root)
    if record and record.get("files"):
        return _not_restored(quest_root, _restore(quest_root, record["files"], key), key)
    return []


def turned_off(quest_root: Path, key: bytes | None = None) -> list[str]:
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
    lines = _not_restored(quest_root, _restore(quest_root, record.get("files") or [], key), key)
    if record.get("not_applicable"):
        return lines  # the two stages never applied: nothing was confirmed or held back to say anything about
    if record.get("stage") == EXPLORE and not record.get("late_start") and not record.get("compromised"):
        why = "explore-then-confirm was turned off while exploring, so the experiment may have run on all of the data"
        record.update(late_start=True, strategy=FRESH_SEEDS, why_no_data=why, files=[])
        _save(quest_root, record)
        lines.append("explore-then-confirm is off: the whole data files are back in place; if it is turned on again, "
                     + ("nothing can be held back for a confirm run" if record.get("data_quest") else
                        "the confirm run will use new random seeds") + f" ({why})")
    elif record.get("stage") == CONFIRM and not record.get("compromised"):
        mark_compromised(quest_root, "explore-then-confirm was turned off during the confirm stage")
        lines.append("explore-then-confirm was turned off during the confirm stage: nothing in this quest can be "
                     "confirmed any more, and its numbers stay exploratory")
    return lines


def _as_supplied(quest_root: Path, folder: str, rel: str, raw: bytes, info: dict[str, Any] | None, quest_id: str,
                 allowed: set[str] | None = None, rule: dict[str, Any] | None = None,
                 key: bytes | None = None) -> bytes | None:
    """The whole file behind what ``<folder>/<rel>`` holds now: the kept original when the file on disk is one of its
    recorded parts, or the part of an interrupted start (the original was kept but the record not yet written). None
    when the file on disk is a recorded part and the whole file cannot be read back (kept encrypted by a run of FI that
    ended without putting it back)."""
    kept = _kept(quest_root, {"file": rel, "folder": folder}, "original")
    if not kept.is_file():
        return raw
    current = _sha(raw)
    if info and current in (info.get("explore_sha256"), info.get("held_back_sha256")):
        return _read_kept(kept, key)
    if info is None:
        original = _read_kept(kept, key)
        if original is not None and _why_not_splittable(rel, original) is None:
            for split_rule in ((rule, None) if rule is not None else (None,)):
                _explore, _back, parts = _split(rel, original, quest_id, allowed, folder=folder, rule=split_rule)
                if current in (parts["explore_sha256"], parts["held_back_sha256"]):
                    return original
    return raw


def _held_back_before(quest_root: Path, record: dict[str, Any], key: bytes | None = None) -> set[str] | None:
    """Once exploration has run on held-back data: the rows held back so far (only they can be held back again).
    ``None`` while exploration has not run, when any row may still be held back. A held-back part that cannot be read
    (kept encrypted by a run of FI that ended, and not rebuilt by :func:`_rekey`) counts as holding no row."""
    ran = bool(record.get("explore_seed_bases")) or int(record.get("explore_runs") or 0) > 0
    if not ran or record.get("strategy") != HELD_BACK:
        return None
    rows: set[str] = set()
    for info in record.get("files") or []:
        part = _read_kept(_kept(quest_root, info, "held_back"), key)
        parsed = _rows(part) if part is not None else None
        if parsed:
            rows.update(parsed[1])
    return rows


def _rekey(quest_root: Path, quest_id: str, record: dict[str, Any], key: bytes | None) -> None:
    """A new run of FI has a new key: the kept files an earlier run wrote encrypted are written again with this run's
    key, from the whole file in place (every run puts it back when it stops) and the split recorded for it, checked
    against the recorded fingerprints of both parts. A file that is not whole on disk, or whose split no longer gives
    the recorded parts, is left as it is (what that means is said where it matters)."""
    # Without a key (a container now) an earlier run's encrypted files are written again plain: the container does not
    # mount them.
    for info in record.get("files") or []:
        kept_whole, kept_back = _kept(quest_root, info, "original"), _kept(quest_root, info, "held_back")
        if all(not p.is_file() or _read_kept(p, key) is not None for p in (kept_whole, kept_back)):
            continue
        disk = _on_disk(quest_root, info)
        raw = disk.read_bytes() if disk.is_file() else None
        if raw is None or _sha(raw) != info.get("original_sha256") or _why_not_splittable(str(info["file"]), raw):
            continue
        _explore, back, parts = _split(str(info["file"]), raw, quest_id, None, folder=_folder(info),
                                       rule=info.get("rule"))
        if parts["held_back_sha256"] != info.get("held_back_sha256"):
            continue
        _write_kept(kept_whole, raw, quest_root, key)
        _write_kept(kept_back, back, quest_root, key)


def _new_record() -> dict[str, Any]:
    return {"schema": SCHEMA, "stage": EXPLORE, "started_at": _now(), "strategy": FRESH_SEEDS,
            "why_no_data": "", "files": [], "explore_seed_bases": [], "confirm_seed_base": None,
            "explore_runs": 0, "confirm_runs": 0, "confirm_executions": 0, "results_seen_in_confirm": 0}


def prepare(quest_root: Path, quest_id: str, *, already_ran: bool = False, data_quest: bool = False,
            key: bytes | None = None, mode: str | None = None) -> tuple[dict[str, Any], list[str]]:
    """Start the exploration stage, or bring it up to date with the data the person supplied since; in the confirm
    stage, give the run the held-back part again. Idempotent, and safe to call again after a start that was cut short.
    ``already_ran``: runs were made before this record existed (the setting was turned on mid-quest), so no data is
    unseen and nothing is held back. ``data_quest``: the quest analyses data instead of simulating (kept in the record
    from then on): its files in ``data/`` are held back too. ``key``: this run's key (no container): the kept files are
    written encrypted with it. ``mode``: how this run keeps them out of reach (``docker`` or ``encrypted``), added to the
    record's ``starts`` (``core/phased_isolation.py`` reads them). Until the split is decided (:func:`decide_split`) the
    rows are set aside one by one; after, by the rule decided. Returns the record and plain lines for run.log about
    anything it did."""
    quest_root = Path(quest_root)
    _clear_stray_tmp(quest_root)
    _move_out_of_quest(quest_root)
    record = load(quest_root)
    lines: list[str] = []
    fresh = record is None
    if record is None:
        record = _new_record()
        if data_quest:
            record["data_quest"] = True
        lines.append("exploration stage: the model may try designs and look at results; when exploration ends the "
                     "design is frozen and run once more on data or seeds exploration never saw")
        if already_ran:
            why = (("the data had already been analysed before explore-then-confirm was turned on" if data_quest else
                    "the experiment had already run on all of the supplied data before explore-then-confirm was turned on"))
            record.update(late_start=True, why_no_data=why)
            lines.append(f"no part of the data can be held back for a confirm run ({why})" if data_quest else
                         f"the confirm run will use new random seeds because no held-back data is available ({why})")
            _save(quest_root, record)
            return record, lines
    if mode and record.get("stage") in (EXPLORE, CONFIRM):
        # A record that ran before FI noted how each start kept the rows out of reach: those starts are not shown to.
        earlier = record.get("starts") if "starts" in record else ([] if fresh else ["unrecorded"])
        record["starts"] = [*(earlier or []), mode]
        _save(quest_root, record)
    _rekey(quest_root, quest_id, record, key)
    if data_quest and record.get("stage") == EXPLORE and not record.get("data_quest"):
        record["data_quest"] = True
        _save(quest_root, record)
    data_quest = bool(record.get("data_quest"))
    if record.get("stage") == CONFIRM:
        if record.get("compromised"):
            return record, lines
        missed = _put(quest_root, record.get("files") or [], "held_back", "held_back_sha256", key)
        if missed:
            why = ("the held-back part could not be put in place of " + ", ".join(shown(i) for i in missed)
                   + " (the file was changed since, or the kept part is missing or was kept encrypted by a run of FI "
                     "that ended without putting it back)")
            mark_compromised(quest_root, why)
            lines.append(f"{why}: the confirm run would not see only the held-back rows, so nothing in this quest can "
                         "be confirmed and its numbers stay exploratory")
            record = load(quest_root) or record
        return record, lines
    if record.get("stage") != EXPLORE:
        lines += _not_restored(quest_root, _restore(quest_root, record.get("files") or [], key), key)
        return record, lines
    if record.get("late_start") or record.get("compromised") or record.get("not_applicable"):
        return record, lines
    if record.get("strategy") == FRESH_SEEDS and (record.get("split_decided") or record.get("explore_seed_bases")
                                                  or (data_quest and int(record.get("explore_runs") or 0))):
        # Exploration has already run on the whole files: none of their rows is unseen, so none can be held back now
        # (a file that became splittable since, a blocking file removed, is not new data).
        return record, lines
    split = {shown(info): info for info in record.get("files") or []}
    rule = record.get("split") if isinstance(record.get("split"), dict) else None
    allowed = _held_back_before(quest_root, record, key)
    # What each file holds as the person supplied it: the whole file kept aside for one already split and unchanged.
    supplied: dict[str, tuple[str, str, bytes]] = {}
    lost: list[dict[str, Any]] = []
    for folder, rel, path in _data_files(quest_root, data_quest=data_quest):
        name = f"{folder}/{rel}"
        whole = _as_supplied(quest_root, folder, rel, path.read_bytes(), split.get(name), quest_id, allowed, rule, key)
        if whole is None:
            lost.append(split[name])
            continue
        supplied[name] = (folder, rel, whole)
    if lost:
        why = ("the whole file of " + ", ".join(shown(i) for i in lost) + " could not be read back (FI stopped without "
               "putting it back, and the rows held back were kept encrypted with a key only that run had)")
        mark_compromised(quest_root, why)
        return load(quest_root) or record, [*lines, *_not_restored(quest_root, lost, key),
                                           f"{why}: nothing in this quest can be confirmed, so its numbers stay "
                                           "exploratory"]

    def label(name: str) -> str:  # the name in a reason: as every earlier record said it for inputs/data/
        return supplied[name][1] if supplied[name][0] == INPUTS and not data_quest else name

    reasons = [why for name, (_f, _r, raw) in supplied.items() if (why := _why_not_splittable(label(name), raw))]
    if data_quest and any(folder == INPUTS for folder, _r, _raw in supplied.values()):
        # A data quest's analysis is its data-reading step, which reads data/ only: a table elsewhere is not what its
        # numbers come from, so holding part of it back would confirm nothing.
        reasons.append("the table is in inputs/data/, which the data-reading step does not read; put it in data/ to "
                       "have part of it held back for a confirm run")
    if not reasons:
        reasons = [(f"{label(name)} now holds only {n} of the rows held back before exploration ran (the file was "
                    f"changed since), fewer than the {MIN_HELD_BACK_ROWS} a confirm run needs") if allowed is not None
                   else f"{label(name)} would have only {n} rows held back, fewer than the {MIN_HELD_BACK_ROWS} a "
                        "confirm run needs"
                   for name, (folder, rel, raw) in supplied.items()
                   if (n := _split(rel, raw, quest_id, allowed, folder=folder, rule=rule)[2]["held_back_rows"])
                   < MIN_HELD_BACK_ROWS]
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
                   else ["no data has been supplied yet in data/" if data_quest
                         else "no data was supplied in inputs/data/"])
    if reasons:
        if split:
            missed = _restore(quest_root, list(split.values()), key)
            lines.append("the supplied data is no longer held back (" + "; ".join(reasons) + ")"
                         + ("" if missed else ": the whole files are back in place"))
            lines += _not_restored(quest_root, missed, key)
        changed = record.get("strategy") != FRESH_SEEDS or record.get("why_no_data") != "; ".join(reasons)
        record.update(strategy=FRESH_SEEDS, why_no_data="; ".join(reasons), files=[])
        if changed:
            lines.append(("no rows are held back for a confirm run yet (" + "; ".join(reasons) + "); the data is "
                          "checked again before it is first analysed") if data_quest else
                         ("the confirm run will use new random seeds because no held-back data is available ("
                          + "; ".join(reasons) + ")"))
        _save(quest_root, record)
        return record, lines
    from . import phased_data as _data

    out: list[dict[str, Any]] = []
    for name, (folder, rel, raw) in supplied.items():
        known = split.get(name)
        explore, back, info = _split(rel, raw, quest_id, allowed, folder=folder, rule=rule)
        if info.get("manifest", {}).get("overlap"):
            # A unit in both parts: the confirm part would not be unseen. Never written as a split.
            why = (f"the split of {shown(info)} put {info['manifest']['overlap']} {info['manifest'].get('unit', 'unit')} "
                   "value(s) in both the exploration part and the held-back part")
            mark_compromised(quest_root, why)
            return load(quest_root) or record, [*lines, f"{why}: nothing in this quest can be confirmed"]
        # The whole file and the held-back part are kept before exploration's part replaces the file, so a start cut
        # short anywhere in between is finished by the next one (_as_supplied); what is already right is not rewritten.
        kept = _kept(quest_root, info, "original")
        kept_now = _read_kept(kept, key) if kept.is_file() else None
        if kept_now is not None and known and _sha(kept_now) == known.get("original_sha256") \
                and known.get("original_sha256") != info["original_sha256"]:
            # The file changed since it was kept (edited while the quest was not running, or while a part of it was in
            # its place after a hard stop): the earlier whole file is kept too, never overwritten (without a container
            # it stays encrypted with this run's key, so only this run can read it).
            kept.replace(kept.with_name(f"{kept.name}.replaced-{str(known['original_sha256'])[:12]}"))
        for dst, data, sealed in ((kept, raw, True), (_kept(quest_root, info, "held_back"), back, True),
                                  (_on_disk(quest_root, info), explore, False)):
            current = (_read_kept(dst, key) if sealed else dst.read_bytes()) if dst.is_file() else None
            # A kept file written plain while this run has a key is written again, encrypted.
            if current != data or (sealed and key is not None and not _locked(dst)):
                if sealed:
                    _write_kept(dst, data, quest_root, key)
                else:
                    _write(dst, data, quest_root)
        out.append(info)
        if not known or known.get("original_sha256") != info["original_sha256"] or known.get("rule") != info.get("rule"):
            how = (f"{_data.how(info.get('rule'))}" if info.get("rule") else
                   "set aside one by one until it is decided which rows belong together, before the data is first read")
            lines.append(f"held back {info['held_back_rows']} of {info['rows']} rows of {shown(info)} ({how}); "
                         f"exploration sees the other {info['explore_rows']} (the whole file and the held-back part "
                         "are kept outside the quest folder" + (
                             ", encrypted with a key only this run of FI holds: if FI is killed rather than stopped, "
                             "the held-back rows cannot be put back, so keep your own copy of the file"
                             if key is not None and not known else ", encrypted" if key is not None else "") + ")")
    record.update(strategy=HELD_BACK, why_no_data="", files=out)
    _save(quest_root, record)
    return record, lines


def decide_split(quest_root: Path, quest_id: str, *, declared: dict[str, Any] | None = None, answer: str | None = None,
                 grouping: list[str] | None = None, research: bool = False, key: bytes | None = None,
                 outlives_run: bool = False) -> tuple[dict[str, Any] | None, list[str], str]:
    """Right before the data is first read (the experiment's first run, a data quest's first reading): decide which
    rows are held back (``core/phased_data.decide``: ``declared`` is the plan's ``protocol.split``, ``answer`` the line
    in plan.md, ``grouping`` the plan's variables), apply it, and record it with its manifest. Done once, while
    exploration has not run; after that the rows held back stay as they are.

    Returns ``(record, lines for run.log, question)``. ``question`` is set when nothing says which rows belong together
    and FI cannot tell, in a research quest: the caller stops and asks it, and nothing is read until it is answered. Any
    other quest holds nothing back (a split that ignores which rows belong together could put one subject in both
    parts). ``outlives_run``: the run's work outlives this run of FI (a background job), so rows kept encrypted with this
    run's key could not be put back: nothing is held back."""
    from . import phased_data as _data

    quest_root = Path(quest_root)
    record = load(quest_root)
    if (record is None or record.get("stage") != EXPLORE or record.get("not_applicable") or record.get("compromised")
            or record.get("late_start") or record.get("strategy") != HELD_BACK or not record.get("files")):
        return record, [], ""
    if record.get("explore_seed_bases") or int(record.get("explore_runs") or 0) or record.get("split_decided"):
        return record, [], ""
    info = record["files"][0]
    whole = _read_kept(_kept(quest_root, info, "original"), key)
    parsed = _rows(whole) if whole is not None else None
    reason = ""
    decision = None
    if outlives_run:
        reason = ("the experiment runs as a background job that outlives this run of FI, and without a container the "
                  "held-back rows are kept encrypted with a key only this run of FI has")
    elif parsed is None:
        reason = f"the whole file of {shown(info)} could not be read back"
    else:
        header, rows, _nl = parsed
        decision = _data.decide(shown(info), header, rows, _delimiter(str(info["file"])), quest_id,
                                declared_split=declared, answer=answer, grouping=grouping)
        if decision.ask and research:
            question = (f"{decision.why}. Which rows of {shown(info)} belong together? Answer on plan.md's line "
                        f"'Rows that belong together:' with the column that names the subject, site, device or other "
                        "unit several rows can share, or with `independent` if every row is a separate case (or give "
                        "protocol.split in the plan's design block)")
            return record, [], question
        if decision.ask:
            reason = (f"{decision.why}, and a split that ignores which rows belong together could put one subject in "
                      "both parts (say it in plan.md's line 'Rows that belong together:' to hold rows back)")
        elif decision.why:
            reason = decision.why
    if reason:
        missed = _restore(quest_root, record["files"], key)
        record = load(quest_root) or record
        record.update(strategy=FRESH_SEEDS, why_no_data=reason, files=[], split_decided=True)
        _save(quest_root, record)
        what = ("no part of the data is held back for a confirm run" if record.get("data_quest")
                else "the confirm run will use new random seeds because no held-back data is available")
        return record, [f"{what} ({reason})", *_not_restored(quest_root, missed, key)], ""
    assert decision is not None
    record.update(split=decision.rule)
    _save(quest_root, record)
    record, lines = prepare(quest_root, quest_id, key=key)
    if not record.get("compromised"):
        # Decided once: held back by the rule, or (too few rows by it) nothing held back from now on.
        record["split_decided"] = True
        _save(quest_root, record)
    return record, lines, ""


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
        # The confirm run finished (with a result or without one): a later reading is a second look.
        record["confirm_reading_done"] = True
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


# ---- candidates: each frozen version is confirmed once, on its own ------------------------------------------------

#: Files in ``code/`` that only redraw or show the results (written by FI's redraw and web steps), and notes (``.md``):
#: changing them does not change the study, so it does not make a new version. Counting them would let a re-run of an
#: unchanged study pass for a new version with a fresh confirm run.
_NOT_THE_STUDY = frozenset({"replot_figures.py", "replot_figures.json", "replot_layout.py", "web_plots.py",
                            # what FI writes beside the code to describe or repeat a run (core/code_project.py)
                            "run.py", "fi_search.py", "requirements.txt", "test_oracles.py", "CHANGELOG.md",
                            # derived from the protocol, which is compared on its own
                            "study.json"})


def fingerprint(quest_root: Path, *, protocol_sha256: str | None = None,
                design_sha256: str | None = None) -> dict[str, str | None]:
    """The hashes that say which version of the study this is: its code (every file of ``code/`` but
    :data:`_NOT_THE_STUDY` and notes), its frozen protocol, and its environment (Python, packages: recorded, not
    compared). ``protocol_sha256``: the frozen protocol's hash when the caller has it (else read here). A data quest
    adds its design (``design_sha256``)."""
    from . import frozen_protocol as _frozen

    root = Path(quest_root)
    # Read here, not through the attempt records (this record reads none of them).
    code: dict[str, str] = {}
    folder = root / "code"
    if folder.is_dir():
        for path in sorted(p for p in folder.rglob("*") if p.is_file()):
            rel = path.relative_to(folder).as_posix()
            # A folder whose name starts with a dot (.git, an editor's settings) is not the study.
            if ("__pycache__" in path.parts or any(part.startswith(".") for part in path.relative_to(folder).parts)
                    or path.name in _NOT_THE_STUDY
                    or rel.lower().endswith((".md", ".pyc"))):
                continue
            try:
                code[rel] = _code_digest(path)
            except OSError:
                code[rel] = ""
    if protocol_sha256 is None:
        frozen = _frozen.load(root)
        protocol_sha256 = str(frozen["sha256"]) if frozen else None
    out: dict[str, str | None] = {
        "code_sha256": _sha(json.dumps(code, sort_keys=True).encode("utf-8")) if code else None,
        "protocol_sha256": protocol_sha256,
        "environment_sha256": _environment_sha(root),
    }
    if design_sha256:
        out["design_sha256"] = design_sha256
    return out


def _code_digest(path: Path) -> str:
    """One file's hash as a version of the study: line endings do not count (a checkout that turns them into CRLF is
    not a change), and for Python only its tokens do (a comment or a blank line is not a new version). Tokens, not a
    syntax tree, so the same code hashes the same under every Python version."""
    raw = path.read_bytes().replace(b"\r\n", b"\n")
    if path.suffix == ".py":
        try:
            return _sha(_python_tokens(raw.decode("utf-8")).encode("utf-8"))
        except Exception:  # noqa: BLE001 -- not readable as Python: compared by its bytes
            pass
    return _sha(raw)


def _python_tokens(text: str) -> str:
    import io
    import tokenize

    skip = (tokenize.COMMENT, tokenize.NL, tokenize.ENCODING)
    return json.dumps([(tok.type, tok.string) for tok in tokenize.generate_tokens(io.StringIO(text).readline)
                       if tok.type not in skip])


def _environment_sha(root: Path) -> str | None:
    """What the environment is (``needs/ENVIRONMENT.json``: Python, platform, sandbox, packages), not where it lives."""
    try:
        record = json.loads((root / "needs" / "ENVIRONMENT.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(record, dict):
        return None
    keep = ("python", "platform", "sandbox", "shared_interpreter", "system_site_packages", "isolated", "packages")
    return _sha(json.dumps({k: record.get(k) for k in keep}, sort_keys=True, default=str).encode("utf-8"))


def candidate(record: dict[str, Any] | None) -> int:
    """Which version of the study the record is about (1 for a record from before versions were counted)."""
    try:
        return max(1, int((record or {}).get("candidate") or 1))
    except (TypeError, ValueError):
        return 1


def _verdict_line(quest_root: Path, record: dict[str, Any], verdict: str, why: str) -> None:
    """Append the current candidate's verdict to ``.fi/confirmations.jsonl`` (``core/confirmations.py``): never
    rewritten. Best effort: a line that cannot be written leaves the candidate without a recorded verdict, which the
    evidence reads as not confirmed (:func:`record_gap`)."""
    from . import confirmations as _confirmations

    frozen = record.get("frozen_at_confirm") if isinstance(record.get("frozen_at_confirm"), dict) else {}
    entry = {
        "candidate": candidate(record), "verdict": verdict, "why": why,
        "strategy": record.get("strategy"), "data_quest": bool(record.get("data_quest")),
        "confirm_seed_base": record.get("confirm_seed_base"),
        "frozen": {**frozen, "protocol_sha256": frozen.get("protocol_sha256", record.get("frozen_sha256")),
                   **({"design_sha256": record["design_sha256"]} if record.get("design_sha256") else {})},
        "confirm_started_at": record.get("confirm_started_at"),
        "confirm_executions": int(record.get("confirm_executions") or 0),
        "confirm_result_sha256": record.get("confirm_result_sha256"),
    }
    try:
        _confirmations.append(quest_root, entry)
    except OSError:
        pass


#: Said in run.log when a version's confirmation did not hold: it is not tried again.
ONE_SHOT = ("this version cannot be confirmed again: its confirm run was the one try (.fi/confirmations.jsonl keeps "
            "it); a changed version is a new candidate and is confirmed on its own")

#: Why a new version of a quest that held rows back cannot be confirmed.
HELD_BACK_USED = ("the rows held back were read by the confirm run of an earlier version of this study, so no rows "
                  "remain that no version has seen")


def new_candidate(quest_root: Path, *, design_sha256: str | None = None, key: bytes | None = None,
                  quest_id: str | None = None) -> list[str]:
    """Called before the experiment runs (and before a data quest's data is read). When the confirm stage has begun
    (or ended) and the study's code or protocol (a data quest: its design) is no longer the version frozen for it, that
    version is done: its verdict stays (a confirm run that gave none is recorded as failed), and the changed study is a
    NEW candidate, back in exploration, which needs a confirm run of its own on seeds no earlier run used. When rows were
    held back, the confirm run has read them, so the new candidate cannot be confirmed on them: the whole files go back
    and it is ``not_confirmable``. Nothing changed: nothing happens (the run is the same candidate's, counted as a
    second run on the confirm data or seeds, as before). A change before the confirm run began (nothing was run on the
    confirm data or seeds yet) only ends the freeze: the same version goes back to exploring, nothing is recorded as
    failed, and exploration's part of the data is put back in place. Returns plain lines for run.log."""
    root = Path(quest_root)
    record = load(root)
    if record is None or record.get("stage") not in (CONFIRM, CONFIRMED) or record.get("not_applicable"):
        return []
    frozen = record.get("frozen_at_confirm")
    if not isinstance(frozen, dict):
        return []  # a record from before versions were counted: it goes on as it did
    now = fingerprint(root, design_sha256=design_sha256)
    # A data quest's version is its design and protocol: what it reads the data with (its code/ is not what ran).
    compared = (("protocol_sha256", "protocol"),) if record.get("data_quest") else (
        ("code_sha256", "code"), ("protocol_sha256", "protocol"))
    changed = [what for key_, what in compared if now.get(key_) != frozen.get(key_)]
    if record.get("data_quest") and design_sha256 and record.get("design_sha256") \
            and design_sha256 != record["design_sha256"]:
        changed.append("design")
    if not changed:
        return []
    old = candidate(record)
    lines: list[str] = []
    unread = not (record.get("strategy") == HELD_BACK and record.get("confirm_data_handed"))
    if record.get("stage") == CONFIRM and not int(record.get("confirm_executions") or 0) and unread:
        # Nothing ran on the confirm data or seeds yet: there is no verdict to keep, and the held-back rows are unread.
        for field in ("confirm_started_at", "explore_result", "explore_result_sha256", "frozen_at_confirm",
                      "frozen_sha256", "design_sha256", "design_revisions_at_confirm", "isolation", "isolation_why",
                      "confirm_replicates", "confirm_run_result_sha256", "confirm_reading_done", "confirm_job_pending"):
            record.pop(field, None)
        record.update(stage=EXPLORE, confirm_seed_base=None, confirm_executions=0)
        _save(root, record)
        lines.append(f"the {' and '.join(changed)} changed before the confirm run began: version {old} goes back to "
                     "exploring, and is frozen and confirmed once when exploration ends")
        if record.get("strategy") == HELD_BACK and record.get("files"):
            _record, more = prepare(root, quest_id or root.name, data_quest=bool(record.get("data_quest")), key=key)
            lines += more
        return lines
    if record.get("stage") == CONFIRM:
        # Its confirm run gave no verdict before the version was changed (a repair of a crashed confirm run, say).
        why = (f"the {' and '.join(changed)} of version {old} changed during its confirm run, before the confirm "
               "result was recorded")
        _verdict_line(root, record, "confirm_failed", why)
        verdict = "confirm_failed"
    else:
        verdict = status(record)
    earlier = [*(record.get("earlier_candidates") or []), {"candidate": old, "verdict": verdict}]
    # Every seed an earlier confirm run used (its base and each replicate's), apart from the next one's.
    stride = max(1, int(record.get("seed_stride") or 1))
    used = [*(record.get("earlier_confirm_seed_bases") or []),
            *([int(record["confirm_seed_base"]) + i * stride
               for i in range(max(1, int(record.get("confirm_replicates") or 1)))]
              if record.get("confirm_seed_base") is not None else [])]
    held_back = record.get("strategy") == HELD_BACK or bool(record.get("data_quest"))
    files = list(record.get("files") or [])
    # The current candidate's own fields go; what is about the quest's data and its exploration stays.
    for field in ("confirm_started_at", "confirmed_at", "confirm_result_sha256", "confirm_run_result_sha256",
                  "confirm_reading_done", "confirm_job_pending", "explore_result", "explore_result_sha256",
                  "frozen_at_confirm", "frozen_sha256", "design_sha256", "design_revisions_at_confirm",
                  "confirm_compared", "confirm_differs", "confirm_executions_recorded", "not_confirmable",
                  "isolation", "isolation_why", "confirm_replicates", "confirm_data_handed"):
        record.pop(field, None)
    record.update(stage=EXPLORE, candidate=old + 1, earlier_candidates=earlier, earlier_confirm_seed_bases=sorted(set(used)),
                  confirm_seed_base=None, confirm_executions=0, confirm_runs=0, results_seen_in_confirm=0,
                  new_candidate_at=_now())
    lines.append(f"version {old + 1} of the study: the {' and '.join(changed)} changed after version {old} was "
                 f"frozen and confirmed on unseen data ({verdict.replace('_', ' ')}); that confirmation is kept as it "
                 "was (.fi/confirmations.jsonl) and does not count for this version, which is confirmed on its own")
    if held_back and not record.get("compromised"):
        # The whole files go back first; a file that could not be put back stays in the record, so the next stop (or
        # start) tries again (restore_inputs).
        missed = _restore(root, files, key) if files else []
        record.update(not_confirmable=HELD_BACK_USED, held_back_used=True, strategy=FRESH_SEEDS,
                      why_no_data=HELD_BACK_USED, files=list(missed))
        _save(root, record)
        lines.append(f"version {old + 1} cannot be confirmed: {HELD_BACK_USED}; its numbers stay exploratory"
                     + (" (the whole data files are back in place)" if files and not missed else ""))
        lines += _not_restored(root, missed, key)
        return lines
    _save(root, record)
    return lines


def note_confirm_handed(quest_root: Path) -> None:
    """The experiment's step is about to run in the confirm stage: from here the quest's code can read the held-back
    part (a known-answer check runs before the confirm run itself is counted), so a change made after this is never
    taken for one made before the confirm data was read (:func:`new_candidate`)."""
    record = load(quest_root)
    if record is not None and record.get("stage") == CONFIRM and not record.get("confirm_data_handed"):
        record["confirm_data_handed"] = True
        _save(quest_root, record)


def record_gap(quest_root: Path, record: dict[str, Any] | None,
               events: list[dict[str, Any]] | None = None) -> str:
    """Why the current candidate's verdict cannot be read as the record says, or ``""``: its verdict lines
    (``.fi/confirmations.jsonl``) were changed or are missing, or one of them says the confirmation did not hold while
    the record says it did. ``events``: the decision trace's events, when there is one."""
    from . import confirmations as _confirmations

    gaps = _confirmations.problems(quest_root, events)
    if record and isinstance(record.get("frozen_at_confirm"), dict) and status(record) == CONFIRMED:
        mine = _confirmations.for_candidate(_confirmations.read(quest_root), candidate(record))
        if not mine:
            gaps.append("the confirmation of this version is not in its record (.fi/confirmations.jsonl)")
        elif _confirmations.worst(mine) != CONFIRMED:
            gaps.append(f"this version's confirmation did not hold on one of its confirm runs "
                        f"({_confirmations.worst(mine).replace('_', ' ')}), and a run that did not hold is never "
                        "replaced by a later one")
    return "; ".join(gaps)


def enter_confirm(quest_root: Path, *, explore_result: Any, frozen_sha256: str | None, stride: int,
                  replicates: int, explore_runs: int, design_sha256: str | None = None, key: bytes | None = None,
                  isolation: tuple[str, str] | None = None) -> tuple[dict[str, Any], list[str]]:
    """End exploration: record what it produced, choose the confirm seed base, and (``held_back_data``) put the held-back
    part in place of the part exploration saw. Idempotent once in the confirm stage. ``design_sha256``: a data quest's
    design as exploration left it (the confirm run must read the held-back rows with that same design). ``isolation``:
    ``(status, why)`` from ``core/phased_isolation.py``, whether the held-back rows were out of exploration's reach."""
    quest_root = Path(quest_root)
    record = load(quest_root)
    if record is None:
        record, _ = prepare(quest_root, quest_root.name)
    if record.get("stage") != EXPLORE:
        return record, []
    explore_bases = [int(b) for b in record.get("explore_seed_bases") or []] or [0]
    # A later version's confirm seeds are apart from every exploration run's and every earlier confirm run's.
    earlier_confirms = [int(b) for b in record.get("earlier_confirm_seed_bases") or []]
    base = _pick_confirm_base(explore_bases + earlier_confirms, max(1, int(stride)), max(1, int(replicates)))
    explore_runs = max(int(explore_runs), int(record.get("explore_runs") or 0))
    data_quest = bool(record.get("data_quest"))
    lines = [f"exploration ended after {explore_runs} "
             + ("reading(s) of the data; the design is frozen and the confirm stage reads the held-back rows once with it"
                if data_quest else "run(s); the protocol is frozen and the confirm stage runs the frozen design once")]
    if record.get("strategy") == HELD_BACK and data_quest:
        lines.append("confirm stage: the data-reading step and the analysis after it run once more, with the frozen "
                     "design, on only the rows held back before exploration began ("
                     + ", ".join(f"{i['held_back_rows']} rows of {shown(i)}" for i in record["files"]) + ")")
    elif record.get("strategy") == HELD_BACK:
        lines.append("confirm stage: the run is given only the data held back before exploration began ("
                     + ", ".join(f"{i['held_back_rows']} rows of {shown(i)}" for i in record["files"])
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
                  design_revisions_at_confirm=_design_versions(quest_root),
                  # This version as it is frozen: a change to its code or protocol after this makes a new version
                  # (new_candidate), never a second confirm run of this one.
                  candidate=candidate(record), confirm_replicates=max(1, int(replicates)),
                  frozen_at_confirm=fingerprint(quest_root, protocol_sha256=frozen_sha256))
    if candidate(record) > 1:
        lines.append(f"version {candidate(record)} of the study is frozen and confirmed on its own; the confirmation of "
                     "earlier versions stays as it was and does not count for it")
    if design_sha256:
        record["design_sha256"] = design_sha256
    if isolation and record.get("strategy") == HELD_BACK:
        record["isolation"], record["isolation_why"] = isolation
        if isolation[1]:
            lines.append(f"the held-back rows are not shown to be out of exploration's reach: {isolation[1]}")
    # The stage is saved before the held-back part replaces exploration's: a start after a stop in between gives the
    # run the held-back part (``prepare`` in the confirm stage), never exploration's part again as new data.
    _save(quest_root, record)
    missed = _put(quest_root, record.get("files") or [], "held_back", "held_back_sha256", key)
    if missed:
        why = ("the held-back part could not be put in place of " + ", ".join(shown(i) for i in missed)
               + " (the file was changed during exploration, or the kept part is missing)")
        mark_compromised(quest_root, why)
        lines.append(f"{why}: nothing in this quest can be confirmed, so its numbers stay exploratory")
        record = load(quest_root) or record
    return record, lines


def confirm_run_started(quest_root: Path) -> bool:
    """In the confirm stage, whether the confirm run has started (its first execution was given the confirm seeds)."""
    record = load(quest_root)
    return bool(record and int(record.get("confirm_executions") or 0) > 0)


def record_confirm(quest_root: Path, result: Any, *, design_sha256: str | None = None,
                   key: bytes | None = None) -> tuple[dict[str, Any] | None, list[str]]:
    """The result that reached the paper in the confirm stage (``None`` or empty: the confirm run produced none). The
    first one is the confirmed result, if the confirm run was made once. Any further run on the confirm data or seeds (a
    repair of the confirm run, a redesign or a re-run the review asked for) means the confirm data was looked at more
    than once, whatever that run produced. ``design_sha256``: a data quest's design now; one that is not the design
    exploration ended with confirms nothing. A data quest's confirm result is set beside exploration's
    (``phased_data.compare``): numbers that differ are recorded and said, never hidden."""
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
        redesigned = bool(record.get("design_sha256") and design_sha256 and design_sha256 != record["design_sha256"])
        frozen_at = record.get("frozen_at_confirm") if isinstance(record.get("frozen_at_confirm"), dict) else None
        drifted: list[str] = []
        if frozen_at is not None:
            now = fingerprint(quest_root)
            drifted = [what for key_, what in ((("protocol_sha256", "protocol"),) if record.get("data_quest") else
                                               (("code_sha256", "code"), ("protocol_sha256", "protocol")))
                       if now.get(key_) != frozen_at.get(key_)]
        if drifted and not replayed and not redesigned:
            # Changed during the confirm run itself (a known-answer check's repair, say): what ran is not the version
            # that was frozen, so it confirms nothing; the changed version is a new one at its next run.
            record["not_confirmable"] = (f"the {' and '.join(drifted)} changed during the confirm run, after the "
                                         "version was frozen")
            lines.append(f"confirm stage: {record['not_confirmable']}, so nothing in this quest is confirmed by it")
        elif replayed:
            record["compromised"] = ("the result that reached the paper is not the one the confirm run produced (a run "
                                     "from an earlier step brought back exploration's result)")
            lines.append(f"confirm stage: {record['compromised']}, so nothing in this quest is confirmed")
        elif redesigned:
            record["compromised"] = ("the design the confirm run read the held-back rows with is not the one "
                                     "exploration ended with")
            lines.append(f"confirm stage: {record['compromised']}, so nothing in this quest is confirmed")
        elif (digest is not None and record.get("strategy") == FRESH_SEEDS
              and digest == record.get("explore_result_sha256")):
            # New seeds gave exactly exploration's numbers. A run that ignores its seed gives the same, so this cannot be
            # told apart from a repeat of exploration's run, and is not counted as a confirmation (the conservative
            # reading: a random study can also match exactly, a rare event absent on both sets of seeds, say).
            record["not_confirmable"] = ("the confirm run on new seeds gave exactly exploration's numbers, which a run "
                                         "that ignores its seed would also give, so it cannot be told apart from a "
                                         "repeat of exploration's run")
            lines.append(f"confirm stage: {record['not_confirmable']}; nothing in this quest is confirmed")
        elif digest is None:
            lines.append("confirm stage: the confirm run produced no result, so nothing in this quest is confirmed")
        elif runs > 1:
            lines.append(f"confirm stage: the confirm run was changed and run again on the confirm data or seeds ({runs} "
                         "runs in all) after its first result was seen, so its numbers are preliminary, not confirmed")
        else:
            lines.append("confirm stage: the confirm run's result is recorded; the paper reports these numbers as "
                         "confirmed and exploration's numbers as exploratory")
            if record.get("data_quest"):
                from . import phased_data as _data

                compared = _data.compare(record.get("explore_result"), result)
                record["confirm_compared"] = compared["compared"]
                record["confirm_differs"] = compared["differs"]
                lines += _data.compare_lines(compared)
        if isinstance(record.get("frozen_at_confirm"), dict):
            # The verdict is kept for good (core/confirmations.py), whatever happens to this version afterwards.
            _verdict_line(quest_root, record, status(record), lines[0] if lines else "")
            if status(record) != CONFIRMED:
                lines.append(ONE_SHOT)
        _save(quest_root, record)
        missed = _restore(quest_root, record.get("files") or [], key)
        if record.get("files") and not missed:
            lines.append("the whole data files are back in place")
        lines += _not_restored(quest_root, missed, key)
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
    if isinstance(record.get("frozen_at_confirm"), dict):
        # The same version confirmed again: both verdicts are kept, and the second run does not replace the first.
        _verdict_line(quest_root, record, status(record), lines[-1])
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


#: What run.log and the plan say for a data quest whose data cannot be split (``why``, in plain words).
_NOT_SPLIT = ("no part of the data can be held back for a confirm run: {why}; so the result is not confirmed on rows the "
              "analysis never saw, and a design revised after the data was analysed keeps it preliminary")

#: Why a data quest whose data is only what it gathered itself holds nothing back.
GATHERED_ONLY = ("the data analysed is what the quest gathered itself (literature, web pages, collected tables), not "
                 "one table you supplied")


def not_applicable_sentence(why: str, *, data_quest: bool = False) -> str:
    """The one plain sentence said when the two stages do not apply to this quest (``data_quest``: its data cannot be
    split, so no confirm run is possible)."""
    return (_NOT_SPLIT if data_quest else _NOT_APPLICABLE).format(why=why)


def mark_not_applicable(quest_root: Path, why: str, key: bytes | None = None) -> list[str]:
    """This quest runs no experiment of its own (``why``: a literature survey, ``--analyze``), or it analyses data that
    cannot be split: there is nothing to run once more. Any rows held back are put back (the analysis must see all of
    the data the person supplied) and the record says why, so a later start holds nothing back. Returns the plain lines
    for run.log, the first time only. A quest already past exploration is left as it is (it did run an experiment)."""
    quest_root = Path(quest_root)
    record = load(quest_root)
    if record is not None and (record.get("not_applicable") or record.get("stage") != EXPLORE):
        return []
    if record is None:
        record = _new_record()
    _clear_stray_tmp(quest_root)
    files = record.get("files") or []
    missed = _restore(quest_root, files, key)
    record.update(not_applicable=why, files=[])
    _save(quest_root, record)
    where = "back in place" if any(_folder(i) != INPUTS for i in files) else "back in inputs/data/"
    back = [f" (the rows held back at the start are {where})"] if files and not missed else [""]
    return [not_applicable_sentence(why, data_quest=bool(record.get("data_quest"))) + back[0],
            *_not_restored(quest_root, missed, key)]


def data_quest_gate(quest_root: Path, quest_id: str, *, declared: dict[str, Any] | None = None,
                    answer: str | None = None, grouping: list[str] | None = None, research: bool = False,
                    key: bytes | None = None, design_sha256: str | None = None) -> tuple[list[str], str]:
    """Called each time a data quest's data-reading step (``data_load``) is about to read the data. In exploration: the
    data supplied since is taken in, the split is decided the first time (:func:`decide_split`, with the plan's
    ``declared`` split, the ``answer`` in plan.md and the plan's variables as ``grouping``), and the reading is
    counted; when the data cannot be split, nothing is held back from then on and the quest is ``not_applicable``, said
    in one sentence (what keeps a design revised after the analysis below publication-ready). In the confirm stage: the
    held-back part is put in place and the reading counted as a run on the confirm data; after it (the whole data read
    again, after a redesign the review asked for), counted the same way, so the result is preliminary again.

    Returns ``(lines for run.log, question)``: a question (research, nothing says which rows belong together) means
    nothing may be read yet; the caller stops and asks it.

    ``design_sha256``: the design the data is about to be read with. After the confirm stage began, a design that is
    not the frozen one makes a new version (:func:`new_candidate`), which the held-back rows can no longer confirm."""
    quest_root = Path(quest_root)
    record = load(quest_root)
    if record is None or not record.get("data_quest") or record.get("not_applicable"):
        return [], ""
    lines: list[str] = []
    if design_sha256 and record.get("stage") in (CONFIRM, CONFIRMED):
        changed = new_candidate(quest_root, design_sha256=design_sha256, key=key, quest_id=quest_id)
        if changed:
            more, question = data_quest_gate(quest_root, quest_id, declared=declared, answer=answer, grouping=grouping,
                                             research=research, key=key)
            return changed + more, question
    if record.get("stage") == EXPLORE:
        if record.get("compromised") or record.get("not_confirmable"):
            return [], ""
        record, lines = prepare(quest_root, quest_id, data_quest=True, key=key)
        if record.get("compromised"):
            return lines, ""
        decided, more, question = decide_split(quest_root, quest_id, declared=declared, answer=answer,
                                               grouping=grouping, research=research, key=key)
        lines += more
        if question:
            return lines, question
        record = decided or record
        if record.get("compromised"):
            return lines, ""
        if record.get("strategy") != HELD_BACK:
            why = str(record.get("why_no_data") or "")
            if why.startswith("no data has been supplied"):
                why = GATHERED_ONLY
            return lines + mark_not_applicable(quest_root, why or "the data could not be split by rows", key), ""
        record = load(quest_root) or record
        record["explore_runs"] = int(record.get("explore_runs") or 0) + 1
        _save(quest_root, record)
        return lines, ""
    if record.get("stage") == CONFIRM:
        record, lines = prepare(quest_root, quest_id, data_quest=True, key=key)
        if record.get("compromised"):
            return lines, ""
        if int(record.get("confirm_executions") or 0) and not record.get("confirm_reading_done"):
            # The confirm reading was cut short before it produced a result (the model's connection failed, FI was
            # stopped): this reading is that one again, not a second look at the held-back rows.
            return lines, ""
    record = load(quest_root) or record
    record["confirm_executions"] = int(record.get("confirm_executions") or 0) + 1
    _save(quest_root, record)
    return lines, ""


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
    out = {"phased": status(record), "phased_strategy": str((record or {}).get("strategy") or FRESH_SEEDS),
           "phased_why_no_data": str((record or {}).get("why_no_data") or "")}
    # Only the current version's confirmation counts; earlier versions' verdicts are said, never counted for it.
    if candidate(record) > 1:
        out["phased_candidate"] = str(candidate(record))
        out["phased_earlier"] = "; ".join(
            f"version {e.get('candidate')}: {str(e.get('verdict') or '').replace('_', ' ')}"
            for e in (record or {}).get("earlier_candidates") or [] if isinstance(e, dict))
        if (record or {}).get("held_back_used"):
            out["phased_unconfirmable_why"] = HELD_BACK_USED
    from . import audit_log as _audit_log
    from . import confirmations as _confirmations

    if _confirmations.path(quest_root).is_file() or isinstance((record or {}).get("frozen_at_confirm"), dict):
        trace = Path(quest_root) / ".fi" / "audit.jsonl"
        try:
            events = _audit_log.read(trace) if trace.is_file() else None
        except Exception:  # noqa: BLE001 -- a trace that cannot be read is its own gap elsewhere
            events = None
        gap = record_gap(quest_root, record, events)
        if gap:
            out["phased_record_gap"] = gap
    differs = [str(d.get("name")) for d in (record or {}).get("confirm_differs") or [] if isinstance(d, dict)]
    if differs:
        # A data quest's confirm numbers that differ from exploration's: said, never a gap.
        out["phased_differs"] = ", ".join(differs)
    if (record or {}).get("strategy") == HELD_BACK and (record or {}).get("stage") in (CONFIRM, CONFIRMED):
        state, why = isolation(record)
        out["phased_isolation"] = state
        if why:
            out["phased_isolation_why"] = why
        if not (record or {}).get("split_decided"):
            out["phased_split_gap"] = ("the rows were held back one by one and nothing said they are independent, so "
                                       "rows of one subject, site or device may be in both the exploration part and the "
                                       "held-back part")
    return out


def isolation(record: dict[str, Any] | None) -> tuple[str, str]:
    """``(status, why)``: whether the held-back rows were out of exploration's reach (``core/phased_isolation.py``), as
    recorded when the confirm stage began. A record from before FI recorded it is not shown to have been."""
    from . import phased_isolation as _iso

    record = record or {}
    state = str(record.get("isolation") or "")
    if state in (_iso.DOCKER, _iso.ENCRYPTED):
        return state, ""
    return _iso.UNVERIFIED, str(record.get("isolation_why") or (
        "the quest held its rows back before FI recorded whether they were out of exploration's reach"))


def _how_confirmed(record: dict[str, Any]) -> str:
    if record.get("strategy") == HELD_BACK:
        from . import phased_data as _data
        from . import phased_isolation as _iso

        reach = {_iso.DOCKER: ", which exploration's code, run in a container without it, could not read",
                 _iso.ENCRYPTED: (", kept encrypted while exploration ran, and exploration's code showed no read "
                                  "outside the study's folder")}.get(
            isolation(record)[0], ", kept apart from exploration though not shown to be out of its code's reach")
        rule = next((i.get("rule") for i in record.get("files") or [] if i.get("rule")), None)
        picked = (f"held back before exploration began ({_data.how_in_paper(rule)})" if rule
                  else "held back at random before exploration began")
        if record.get("data_quest"):
            return f"on a part of the data {picked}{reach}"
        return f"on a part of the supplied data that was {picked}{reach}, and on new seeds"
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
    paper's number audits hold every number in the paper to the results). A later version of the study (a candidate
    after the first) says so: an earlier version's confirmation is kept and does not count for it."""
    text = _summary(record)
    if candidate(record) > 1 and status(record) != NOT_APPLICABLE and not (record or {}).get("held_back_used"):
        text += (" The study was changed after an earlier version of it had been confirmed this way; that earlier "
                 "confirmation is kept as it was and does not count for these numbers.")
    return text


def _summary(record: dict[str, Any] | None) -> str:
    state = status(record)
    if state == "not_confirmable" and (record or {}).get("held_back_used"):
        return ("An earlier version of the study was confirmed on data held back from exploration, and the study was "
                "changed after that confirmation; no data remains that no version has seen, so this version is not "
                "confirmed. Treat the numbers as exploratory.")
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
    if (record or {}).get("data_quest"):
        differs = (" Some numbers exploration found differ on the held-back rows (in sign, or by more than a quarter); "
                   "the paper says which, and reports the confirm run's." if (record or {}).get("confirm_differs")
                   else "")
        return (f"The design was tried, and could be changed, in an exploration stage, then frozen, and the data was "
                f"read and analysed once more with it {how}. The numbers reported as results are that confirm run's; "
                f"exploration's numbers are exploratory and are not reported as findings.{differs}")
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


#: What plan.md says about keeping the held-back rows out of exploration's reach.
_REACH = ("The held-back rows are kept outside the quest folder. Run in a container (`execution.sandbox: docker`), "
          "exploration's code cannot reach them; without one they are kept encrypted, with a key only the running FI "
          "holds, and before the confirm run the quest's code is read for any path out of the quest folder (one found "
          "keeps the result below publication-ready).")


def plan_lines(record: dict[str, Any] | None, *, on: bool, research: bool, runs_code: bool, why: str = "",
               kept_off: bool = False, data_quest: bool = False, split_note: str = "") -> list[str]:
    """plan.md's section saying whether and how the result will be confirmed, and what the confirm run costs. Silent
    under the default profile with the two stages off (nothing changes there). ``why``: why the quest runs no
    experiment of its own, when it does not. ``data_quest``: the quest analyses data instead of simulating (its confirm
    run reads held-back rows). ``split_note``: how the held-back rows will be chosen (:func:`split_preview`), with the
    question line when nothing says which rows belong together. Only that line is read back (``phased_data.plan_answer``)."""
    head = [f"## {PLAN_HEADING}", ""]

    def said(text: str) -> list[str]:
        return head + [text[0].upper() + text[1:] + ".", ""]

    if not on:
        if not research:
            return []
        return said(kept_off_sentence(runs_code or data_quest) if kept_off else off_sentence(runs_code or data_quest, why))
    data_quest = data_quest or bool((record or {}).get("data_quest"))
    if (not runs_code and not data_quest) or status(record) == NOT_APPLICABLE:
        return said(not_applicable_sentence(
            str((record or {}).get("not_applicable") or why or "this quest runs no experiment of its own"),
            # A record from before data quests were confirmed keeps its own sentence.
            data_quest=bool((record or {}).get("data_quest")) and bool((record or {}).get("not_applicable"))))
    if (record or {}).get("compromised") or (record or {}).get("not_confirmable"):
        reason = str(record.get("compromised") or record.get("not_confirmable"))
        return said(f"the result cannot be confirmed once more on data or seeds the exploration never saw ({reason}), "
                    "so its numbers stay exploratory")
    if data_quest:
        return head + _data_plan_lines(record or {}, split_note)
    held_back = (record or {}).get("strategy") == HELD_BACK and record.get("files")
    if held_back:
        held = ", ".join(f"{i['held_back_rows']} of the {i['rows']} rows of `{shown(i)}`" for i in record["files"])
        on_what = f"on the part of your data held back from exploration ({held}) and on new seeds"
    else:
        reason = str((record or {}).get("why_no_data") or "no data was supplied in inputs/data/")
        on_what = f"on new random seeds exploration never used (no data is held back: {reason})"
    return head + [
        "The model may try designs and look at results first (exploration). When exploration ends, the design is "
        f"frozen and run once more {on_what}. Only that run's numbers can be publication-ready; exploration's numbers "
        "are reported as exploratory.",
        "",
        *([split_note, "", _REACH, ""] if held_back and split_note else [_REACH, ""] if held_back else []),
        "This costs one more full run of the frozen design: every setting again, with as many runs per setting as an "
        "exploration run, about as long as one run during exploration, and the checks after it. The confirm run is made "
        "once: a second run on the confirm data or seeds (a repair, a redesign) leaves the result preliminary.",
        "",
    ]


def _data_plan_lines(record: dict[str, Any], split_note: str = "") -> list[str]:
    """plan.md's lines for a data quest: what is held back (or will be, when the data arrives) and what it costs."""
    from . import phased_data as _data

    if record.get("strategy") == HELD_BACK and record.get("files"):
        held = ", ".join(f"{i['held_back_rows']} of the {i['rows']} rows of `{shown(i)}`" for i in record["files"])
        what = f"Part of the data is held back from exploration ({held})."
    else:
        what = (f"When the data is first analysed, part of it is held back first, if it is one table you supplied (a "
                f"CSV or TSV file in `data/`) with at least {MIN_ROWS} rows. Which rows depends on "
                "which rows belong together: whole subjects, sites or other units when several rows share one, the "
                "latest period when the rows are a time series, and rows one by one only when each row is a separate "
                f"case (at least {_data.MIN_UNITS} units, about {round(HOLD_BACK_FRACTION * 100)}% of them held back). "
                "If nothing says which, a research quest asks you; otherwise run.log says why in one sentence and the "
                "result cannot be confirmed this way.")
    return [
        f"{what} The model analyses the rest and may change the design after seeing the results (exploration). When "
        "exploration ends, the design is frozen and the data is read and analysed once more with it, on the held-back "
        "rows only. Only that run's numbers can be publication-ready; exploration's numbers are reported as "
        "exploratory, and where the two disagree the paper says so.",
        "",
        *([split_note, ""] if split_note else []),
        _REACH,
        "",
        "This costs one more reading and analysis of the data (the held-back rows only) and the checks after it. It is "
        "made once: reading the held-back rows again (a repair, a redesign) leaves the result preliminary.",
        "",
    ]


def split_preview(quest_root: Path, quest_id: str, *, declared: dict[str, Any] | None = None, answer: str | None = None,
                  grouping: list[str] | None = None, key: bytes | None = None) -> str:
    """For plan.md: how the held-back rows will be chosen, in a sentence, or the question line
    (``phased_data.QUESTION_LINE``) when nothing says which rows belong together. Empty when no table is held back."""
    from . import phased_data as _data

    record = load(quest_root)
    if not record or record.get("strategy") != HELD_BACK or not record.get("files"):
        return ""
    info = record["files"][0]
    if record.get("split_decided") and info.get("rule"):
        return f"The rows held back are {_data.how(info['rule'])}."
    whole = _read_kept(_kept(Path(quest_root), info, "original"), key)
    parsed = _rows(whole) if whole is not None else None
    if parsed is None:
        return ""
    decision = _data.decide(shown(info), parsed[0], parsed[1], _delimiter(str(info["file"])), quest_id,
                            declared_split=declared, answer=answer, grouping=grouping)
    if decision.ask:
        return (f"Which rows belong together decides which rows can be held back: {decision.why}. Before the data is "
                f"first read, FI holds back by your answer here.\n\n{_data.QUESTION_LINE}")
    if decision.why:
        return f"The rows cannot be held back by the rule that applies: {decision.why}."
    return f"The rows held back will be {_data.how(decision.rule)}."


def write_note(quest_root: Path) -> str:
    """What the writer is told about the two stages (appended to the evidence note). Empty when they do not apply (the
    quest runs no experiment of its own): there is nothing to say about stages it never had."""
    record = load(quest_root)  # None (the record is missing): nothing is confirmed, and the note says so
    if status(record) == NOT_APPLICABLE:
        return ""
    differs = [str(d.get("name")) for d in (record or {}).get("confirm_differs") or [] if isinstance(d, dict)]
    named = (" On the held-back rows these results differ from what exploration found (in sign, or by more than a "
             f"quarter): {', '.join(differs[:12])}. Say so plainly in the limitations, naming them, without quoting "
             "exploration's values." if differs and status(record) == CONFIRMED else "")
    return ("This quest ran in two stages, exploration then confirmation. State this in the methods, in these terms: "
            + summary(record) + " Report only the numbers in the results you are given as the findings, and do not "
            "call exploratory numbers confirmed." + named)


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
