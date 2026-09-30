"""Explore, then confirm: a quest run in two stages (``engine.phased: true``; off by default).

In the EXPLORATION stage the model may try designs, look at results and change the design, as a quest always could.
When exploration ends (the quest is about to write its paper on an accepted result) the protocol is frozen and the
frozen design is run ONCE more in the CONFIRM stage, on data or seeds the exploration never saw. Only the confirm
run's numbers can back a publication-ready claim; while a quest has none, its result is preliminary
(``core/evidence.py`` reads :func:`evidence_settings`).

What the confirm run is given, decided before exploration runs anything (:func:`prepare`):

- ``held_back_data``: the person supplied tabular data (``inputs/data/``) that can be split by rows, every file with
  at least :data:`MIN_ROWS` rows. A random part of each file (:data:`HOLD_BACK_FRACTION`) is taken out before
  exploration starts, so exploration sees only the rest; the confirm run sees only that part. The confirm run is also
  given a new seed base.
- ``fresh_seeds``: otherwise (no data, too few rows, a file that cannot be split by rows). The confirm run is given a
  seed base exploration never used, and the report says it was confirmed on new random seeds and why no data was held
  back.

The whole files the person supplied are kept in ``.fi/phased/original/`` and put back in ``inputs/data/`` as soon as
the confirm run's result is recorded.

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
DIR = "phased"  # .fi/phased/{original,held_back}/

EXPLORE = "explore"
CONFIRM = "confirm"
CONFIRMED = "confirmed"

HELD_BACK = "held_back_data"
FRESH_SEEDS = "fresh_seeds"

#: A file needs at least this many data rows (header excluded) to have a part held back.
MIN_ROWS = 40
#: The share of each file's rows held back for the confirm run.
HOLD_BACK_FRACTION = 0.3
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


def _split(rel: str, raw: bytes, quest_id: str) -> tuple[bytes, bytes, dict[str, Any]]:
    header, rows, newline = _rows(raw)  # type: ignore[misc] -- checked by _why_not_splittable
    held = max(1, int(round(len(rows) * HOLD_BACK_FRACTION)))
    order = list(range(len(rows)))
    random.Random(f"{quest_id}:{rel}:{_sha(raw)}").shuffle(order)
    held_idx = set(order[:held])
    explore_rows = [r for i, r in enumerate(rows) if i not in held_idx]
    held_rows = [r for i, r in enumerate(rows) if i in held_idx]
    explore, back = _join(header, explore_rows, newline), _join(header, held_rows, newline)
    info = {"file": rel, "rows": len(rows), "explore_rows": len(explore_rows), "held_back_rows": len(held_rows),
            "original_sha256": _sha(raw), "explore_sha256": _sha(explore), "held_back_sha256": _sha(back)}
    return explore, back, info


def _originals(quest_root: Path) -> Path:
    return Path(quest_root) / ".fi" / DIR / "original"


def _held_back(quest_root: Path) -> Path:
    return Path(quest_root) / ".fi" / DIR / "held_back"


def _restore(quest_root: Path, files: list[dict[str, Any]]) -> None:
    for info in files:
        src = _originals(quest_root) / info["file"]
        if src.is_file():
            dst = Path(quest_root) / "inputs" / "data" / info["file"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)


def prepare(quest_root: Path, quest_id: str) -> tuple[dict[str, Any], list[str]]:
    """Start the exploration stage, or bring it up to date with the data the person supplied since. Idempotent; a no-op
    once exploration has ended. Returns the record and plain lines for run.log about anything it did."""
    quest_root = Path(quest_root)
    record = load(quest_root)
    lines: list[str] = []
    if record is None:
        record = {"schema": SCHEMA, "stage": EXPLORE, "started_at": _now(), "strategy": FRESH_SEEDS,
                  "why_no_data": "", "files": [], "explore_seed_bases": [], "confirm_seed_base": None,
                  "explore_runs": 0, "confirm_runs": 0, "results_seen_in_confirm": 0}
        lines.append("exploration stage: the model may try designs and look at results; when exploration ends the "
                     "design is frozen and run once more on data or seeds exploration never saw")
    if record.get("stage") != EXPLORE:
        return record, lines
    split = {info["file"]: info for info in record.get("files") or []}
    files = _data_files(quest_root)
    # What each file holds as the person supplied it: the whole file kept aside for one already split and unchanged.
    supplied: dict[str, bytes] = {}
    for path in files:
        rel = path.relative_to(quest_root / "inputs" / "data").as_posix()
        raw = path.read_bytes()
        info = split.get(rel)
        if info and _sha(raw) == info.get("explore_sha256") and (_originals(quest_root) / rel).is_file():
            raw = (_originals(quest_root) / rel).read_bytes()
        supplied[rel] = raw
    reasons = [why for rel, raw in supplied.items() if (why := _why_not_splittable(rel, raw))]
    if not supplied:
        reasons = ["no data was supplied in inputs/data/"]
    if reasons:
        if split:
            _restore(quest_root, list(split.values()))
            lines.append("the supplied data is no longer held back (" + "; ".join(reasons) + "): the whole files are "
                         "back in inputs/data/")
        changed = record.get("strategy") != FRESH_SEEDS or record.get("why_no_data") != "; ".join(reasons)
        record.update(strategy=FRESH_SEEDS, why_no_data="; ".join(reasons), files=[])
        if changed:
            lines.append("the confirm run will use new random seeds because no held-back data is available ("
                         + "; ".join(reasons) + ")")
        _save(quest_root, record)
        return record, lines
    out: list[dict[str, Any]] = []
    for rel, raw in supplied.items():
        info = split.get(rel)
        if info and info.get("original_sha256") == _sha(raw):
            out.append(info)
            continue
        explore, back, info = _split(rel, raw, quest_id)
        for folder, data in ((_originals(quest_root), raw), (_held_back(quest_root), back)):
            (folder / rel).parent.mkdir(parents=True, exist_ok=True)
            (folder / rel).write_bytes(data)
        (quest_root / "inputs" / "data" / rel).write_bytes(explore)
        out.append(info)
        lines.append(f"held back {info['held_back_rows']} of {info['rows']} rows of inputs/data/{rel} for the confirm "
                     f"run; exploration sees the other {info['explore_rows']} (the whole file is kept in "
                     f".fi/{DIR}/original/{rel})")
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
        if seed not in bases:
            record["explore_seed_bases"] = sorted([*bases, seed])
            _save(quest_root, record)
        return env
    base = record.get("confirm_seed_base")
    if base is None:
        return env
    stride = int(record.get("seed_stride") or 1)
    return {**env, "FI_REPLICATE_SEED": str(int(base) + index * stride)}


# ---- the boundary and the confirm result --------------------------------------------------------------------------


def _result_digest(result: Any) -> str:
    return _sha(json.dumps(result, sort_keys=True, default=str).encode("utf-8"))


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
    lines = [f"exploration ended after {explore_runs} run(s); the protocol is frozen and the confirm stage runs the "
             "frozen design once"]
    for info in record.get("files") or []:
        src = _held_back(quest_root) / info["file"]
        (quest_root / "inputs" / "data" / info["file"]).write_bytes(src.read_bytes())
    if record.get("strategy") == HELD_BACK:
        lines.append("confirm stage: the run is given only the data held back before exploration began ("
                     + ", ".join(f"{i['held_back_rows']} rows of inputs/data/{i['file']}" for i in record["files"])
                     + f") and a new seed base ({base}) that exploration never used")
    else:
        lines.append(f"confirm stage: the run is given new random seeds (seed base {base}; exploration used "
                     f"{', '.join(map(str, explore_bases))}) because no held-back data was available "
                     f"({record.get('why_no_data') or 'no reason recorded'})")
    record.update(stage=CONFIRM, confirm_started_at=_now(), confirm_seed_base=base, seed_stride=max(1, int(stride)),
                  frozen_sha256=frozen_sha256, explore_runs=int(explore_runs),
                  explore_result_sha256=_result_digest(explore_result) if explore_result is not None else None,
                  explore_result=explore_result)
    _save(quest_root, record)
    return record, lines


def record_confirm(quest_root: Path, result: Any) -> tuple[dict[str, Any] | None, list[str]]:
    """The result that reached the paper in the confirm stage (``None``: the confirm run produced none). The first one
    is the confirmed result; another one later (the frozen design run again after its confirm numbers were seen: a
    redesign, a re-run the review asked for) means the confirm data was looked at more than once."""
    quest_root = Path(quest_root)
    record = load(quest_root)
    if record is None or record.get("stage") == EXPLORE:
        return record, []
    lines: list[str] = []
    digest = _result_digest(result) if result is not None else None
    if record.get("stage") == CONFIRM:
        record.update(stage=CONFIRMED, confirmed_at=_now(), confirm_result_sha256=digest, confirm_runs=1,
                      results_seen_in_confirm=1 if digest else 0)
        if digest is None:
            lines.append("confirm stage: the confirm run produced no result, so nothing in this quest is confirmed")
        else:
            lines.append("confirm stage: the confirm run's result is recorded; the paper reports these numbers as "
                         "confirmed and exploration's numbers as exploratory")
        if record.get("files"):
            _restore(quest_root, record["files"])
            lines.append("the whole data files are back in inputs/data/")
    elif digest is not None and digest != record.get("confirm_result_sha256"):
        record["confirm_runs"] = int(record.get("confirm_runs") or 1) + 1
        record["results_seen_in_confirm"] = int(record.get("results_seen_in_confirm") or 0) + 1
        record["confirm_result_sha256"] = digest
        lines.append(f"the frozen design was run again after the confirm numbers were seen ({record['confirm_runs']} "
                     "confirm results in all): the numbers are no longer from one untouched confirm run, so they are "
                     "preliminary")
    else:
        return record, []
    _save(quest_root, record)
    return record, lines


def status(record: dict[str, Any] | None) -> str:
    """``explore`` (no confirm run yet), ``confirming``, ``confirmed``, ``confirm_failed`` (it produced no result) or
    ``confirm_reused`` (the confirm data was looked at more than once)."""
    if not record:
        return EXPLORE
    if record.get("stage") == EXPLORE:
        return EXPLORE
    if record.get("stage") == CONFIRM:
        return "confirming"
    if not record.get("confirm_result_sha256"):
        return "confirm_failed"
    if int(record.get("results_seen_in_confirm") or 0) > 1:
        return "confirm_reused"
    return CONFIRMED


def evidence_settings(quest_root: Path) -> dict[str, str]:
    """What the evidence ladder is told (``core/evidence.py``: ``settings["phased"]``)."""
    record = load(quest_root)
    return {"phased": status(record), "phased_strategy": str((record or {}).get("strategy") or FRESH_SEEDS),
            "phased_why_no_data": str((record or {}).get("why_no_data") or "")}


def _how_confirmed(record: dict[str, Any]) -> str:
    if record.get("strategy") == HELD_BACK:
        return "on a part of the supplied data that was held back at random before exploration began, and on new seeds"
    # No file names or row counts here: every number in the paper is held to the results (the full reason is in run.log
    # and .fi/phased.json).
    why = str(record.get("why_no_data") or "")
    reason = ("no data was supplied" if why.startswith("no data") else
              "the supplied data was too small, or could not be split by rows")
    return ("on new random seeds that exploration never used (confirmed on new random seeds because no held-back data "
            f"was available: {reason})")


def summary(record: dict[str, Any] | None) -> str:
    """One plain sentence on which numbers are exploratory and which are confirmed (no numbers of the result in it: the
    paper's number audits hold every number in the paper to the results)."""
    state = status(record)
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
        return (f"The design was frozen when exploration ended and run {how}; it was then run again after the confirm "
                "numbers had been seen, so the numbers are no longer from one untouched confirm run. Treat them as "
                "exploratory.")
    return (f"The design was tried, and could be changed, in an exploration stage, then frozen and run once more {how}. "
            "The numbers reported as results are that confirm run's; exploration's numbers are exploratory and are not "
            "reported as findings.")


def write_note(quest_root: Path) -> str:
    """What the writer is told about the two stages (appended to the evidence note)."""
    record = load(quest_root)
    if record is None:
        return ""
    return ("This quest ran in two stages, exploration then confirmation. State this in the methods, in these terms: "
            + summary(record) + " Report only the numbers in the results you are given as the findings, and do not "
            "call exploratory numbers confirmed.")


def mark_paper(markdown: str, record: dict[str, Any] | None) -> str:
    """``markdown`` with the stage note under its title (one, replacing an earlier one)."""
    if record is None:
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
