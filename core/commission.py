"""A study FI is commissioned to run: one file that fixes the question, its scope and limits, the values already known
with the passage and the full text that state each, what counts as success and as failure, and the known doubts.

The file is written by whoever commissions the study and is not edited after it is handed over. ``Config.commission``
names it; FI then skips making ideas and choosing a study (those are given), reads only the full texts the file
supplies, and shows the rest to the plan's model. A study that cannot be done as commissioned is handed back
(:func:`return_record`, ``needs/STUCK.json`` with ``return_reason``) instead of being made smaller.

The rules a file must meet are here, in one place: :func:`problems` (or :func:`validate`, which raises) is what FI
runs before any model call, and what another program runs to check a file before it hands it over. FI refuses a file
that breaks one; it never fills a missing value in itself.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import time
from pathlib import Path
from typing import Any

#: Where a quest keeps its frozen copy of the commission, and the full texts it supplies.
STAGED_NAME = "commission.json"
#: The three reasons a commissioned study is handed back (``needs/STUCK.json`` ``return_reason``).
RETURN_REASONS = ("known_value_mismatch", "plan_contradicts_sources", "scope_exceeds_cap")
#: The ``--from`` steps the commission gives, so a rerun from them has nothing to redo.
GIVEN_STEPS = ("ideas", "literature", "studies")
#: What a step the commission gives tells a person watching (empty: nothing is said for it).
GIVEN_STAGES = {"ideate": "", "scope": "", "literature": "Reading the sources the commission gives."}
#: The same floor of text as a paper dropped in ``inputs/papers/`` (shorter files are skipped there).
MIN_TEXT_CHARS = 200
_READ_SUFFIXES = (".pdf", ".md", ".txt")


class CommissionRefused(ValueError):
    """A commission FI cannot start from; the message names the key, file or value, in plain words."""


def _number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _text(v: Any) -> bool:
    return isinstance(v, str) and bool(v.strip())


# A number as a passage may write it: a sign, thousands separators, a bare leading point, and an exponent written
# ``e-3`` or ``x 10^-3``.
_NUMBER = re.compile(r"(?<![\w.])([-+−]?)(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d*\.\d+|\d+)"
                     r"(?:\s*(?:[eE]|[x×·]\s*10\^?)\s*([-+−]?\d+))?")


def number_in_text(value: float, text: str) -> bool:
    """Whether ``text`` writes the number ``value`` (0.888 as 0.8880 or .888, 1200 as 1,200, 0.001 as 1e-3 or
    1 x 10^-3); the sign aside, since a passage may state a fall of 0.3 for a value of -0.3."""
    for _sign, digits, exp in _NUMBER.findall(str(text or "")):
        found = float(digits.replace(",", "")) * 10.0 ** int((exp or "0").replace("−", "-"))
        if math.isclose(found, abs(float(value)), rel_tol=1e-9, abs_tol=0.0):
            return True
    return False


def read(path: Path) -> dict[str, Any]:
    """The commission at ``path`` (JSON, or YAML, which reads JSON too) as a mapping."""
    import yaml

    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except OSError as e:
        raise CommissionRefused(f"the commission {Path(path)} cannot be read ({e.strerror or e})") from e
    except yaml.YAMLError as e:
        raise CommissionRefused(f"the commission {Path(path)} is not valid JSON") from e
    if not isinstance(data, dict):
        raise CommissionRefused(f"the commission {Path(path)} is not a set of named keys")
    return data


def form_problems(data: dict[str, Any]) -> list[str]:
    """What is missing or of the wrong type in the commission ``data``, one plain sentence each."""
    out: list[str] = []
    for key in ("question", "scope"):
        if not _text(data.get(key)):
            out.append(f"the commission has no `{key}` (a sentence)" if key not in data else f"`{key}` is not a sentence")
    limits = data.get("limits")
    if not isinstance(limits, dict):
        out.append("the commission has no `limits` (`hours` and `model_calls`)")
    else:
        if not (_number(limits.get("hours")) and limits["hours"] > 0):
            out.append("`limits.hours` is missing or is not a number above 0")
        if not (isinstance(limits.get("model_calls"), int) and not isinstance(limits.get("model_calls"), bool)
                and limits["model_calls"] > 0):
            out.append("`limits.model_calls` is missing or is not a whole number above 0")
    known = data.get("known_values")
    if not isinstance(known, list):
        out.append("the commission has no `known_values` (a list; it may be empty only with `no_known_values: true`)")
        known = []
    elif not known and data.get("no_known_values") is not True:
        out.append("`known_values` is empty: give the values already known, or say `no_known_values: true`")
    for i, kv in enumerate(known):
        if not isinstance(kv, dict):
            out.append(f"known value {i + 1} is not a set of named keys")
            continue
        who = f"the known value {kv['name']!r}" if _text(kv.get("name")) else f"known value {i + 1}"
        if not _text(kv.get("name")):
            out.append(f"{who} has no `name`")
        for key in ("value", "tolerance"):
            if key not in kv:
                out.append(f"the commission has no `{key}` for {who}")
            elif not _number(kv[key]) or (key == "tolerance" and kv[key] < 0):
                out.append(f"the `{key}` of {who} is not a number" + (" of 0 or more" if key == "tolerance" else ""))
        if not _text(kv.get("case")):
            out.append(f"{who} has no `case` (the situation it applies to: which form, which parameter values)")
        if not _text(kv.get("passage")):
            out.append(f"{who} has no `passage` (the sentence of its source that states it)")
        elif _number(kv.get("value")) and not number_in_text(kv["value"], kv["passage"]):
            out.append(f"the passage of {who} does not state its value {kv['value']}")
        src = kv.get("source")
        if not isinstance(src, dict):
            out.append(f"{who} has no `source` (`title`, `file`, and `doi` when it has one)")
            continue
        for key in ("title", "file"):
            if not _text(src.get(key)):
                out.append(f"the source of {who} has no `{key}`")
        if "doi" in src and not isinstance(src["doi"], str):
            out.append(f"the `doi` of the source of {who} is not text")
    for key in ("success", "failure"):
        items = data.get(key)
        if not isinstance(items, list) or not items:
            out.append(f"the commission has no `{key}` (a list of `condition` and `verify`)")
            continue
        for i, item in enumerate(items):
            if not (isinstance(item, dict) and _text(item.get("condition")) and _text(item.get("verify"))):
                out.append(f"`{key}` item {i + 1} needs both a `condition` and how to `verify` it")
    doubts = data.get("doubts")
    if not isinstance(doubts, list) or not all(isinstance(d, str) for d in doubts):
        out.append("the commission has no `doubts` (a list of sentences; `[]` when there are none)")
    return out


def source_text(path: Path) -> str:
    """The text FI reads from a supplied full text, as ``inputs/papers/`` reads it (a PDF's every page)."""
    if path.suffix.lower() == ".pdf":
        from core import pdf_text

        return (pdf_text.extract(path.read_bytes()).text or "").strip()
    return path.read_text(encoding="utf-8", errors="replace").strip()


def source_problems(data: dict[str, Any], base: Path) -> list[str]:
    """Each supplied full text that is missing, cannot be read, has under :data:`MIN_TEXT_CHARS` characters of text, or
    does not have the passage of a known value it is given for. ``base``: the folder the commission's paths are in."""
    from core.engine_base import _quote_in_source

    out: list[str] = []
    texts: dict[str, str | None] = {}
    for kv in data.get("known_values") or []:
        raw = str(kv["source"]["file"])
        if raw not in texts:
            path = _resolve(raw, base)
            texts[raw] = None
            if not path.is_file():
                out.append(f"the source file {raw!r} of the known value {kv['name']!r} does not exist ({path})")
            elif path.suffix.lower() not in _READ_SUFFIXES:
                out.append(f"the source file {raw!r} is not one FI reads (.pdf, .md or .txt)")
            else:
                try:
                    text = source_text(path)
                except (OSError, ValueError) as e:
                    out.append(f"the source file {raw!r} cannot be read ({e})")
                else:
                    if len(text) < MIN_TEXT_CHARS:
                        out.append(f"the source file {raw!r} has under {MIN_TEXT_CHARS} characters of text")
                    else:
                        texts[raw] = text
        text = texts[raw]
        if text is not None and not _quote_in_source(str(kv["passage"]), text):
            out.append(f"the passage of the known value {kv['name']!r} is not in its source file {raw!r}")
    return out


def problems(path: Path) -> list[str]:
    """Every rule the commission at ``path`` breaks (empty: FI can start from it). Reads the full texts it supplies."""
    try:
        data = read(path)
    except CommissionRefused as e:
        return [str(e)]
    found = form_problems(data)
    return found or source_problems(data, Path(path).resolve().parent)


def validate(path: Path, *, sources: bool = True) -> None:
    """Raise :class:`CommissionRefused` naming every problem of the commission at ``path``; return when there is none.
    ``sources=False`` checks only its keys and their types (no full text is read)."""
    found = problems(path) if sources else form_problems(read(path))
    if found:
        raise CommissionRefused(f"the commission {Path(path)} cannot be used, so no study was started: "
                                + "; ".join(found) + ". Correct the file and start again.")


def locate(raw: Any, folder: Path) -> Path:
    """The commission a config names: a relative path is read against the config's own folder. The quest's own copy of
    the config (``<quest>/config.yaml``) finds the quest's frozen copy when the original is not beside it."""
    path = _resolve(str(raw), folder)
    staged = staged_path(folder)
    return staged if not path.is_file() and staged.is_file() else path


def _resolve(raw: str, folder: Path) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else Path(folder) / path


def staged_path(quest_root: Path) -> Path:
    return Path(quest_root) / "inputs" / STAGED_NAME


def staged_names(data: dict[str, Any]) -> dict[str, str]:
    """Each source file the commission names -> its name in ``inputs/papers/`` (the file's own name, numbered when two
    different files share one). Worked out from the file's words alone, so a resumed quest gets the same names."""
    out: dict[str, str] = {}
    for kv in data.get("known_values") or []:
        raw = str((kv.get("source") or {}).get("file") or "")
        if not raw or raw in out:
            continue
        name = Path(raw).name
        taken = set(out.values())
        n = 1
        while name in taken:
            n += 1
            name = f"{Path(raw).stem}-{n}{Path(raw).suffix}"
        out[raw] = name
    return out


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stage(path: Path, quest_root: Path) -> str:
    """Copy the commission to ``inputs/commission.json`` and its full texts to ``inputs/papers/``; return its sha256."""
    path = Path(path)
    data = read(path)
    papers = Path(quest_root) / "inputs" / "papers"
    papers.mkdir(parents=True, exist_ok=True)
    for raw, name in staged_names(data).items():
        shutil.copy2(_resolve(raw, path.resolve().parent), papers / name)
    shutil.copy2(path, staged_path(quest_root))
    return sha256(staged_path(quest_root))


def recorded_sha(quest_root: Path) -> str | None:
    """The sha256 of the commission this quest was started from: the last one its audit trace recorded, else its copy's."""
    from core import audit_log

    trace = Path(quest_root) / ".fi" / "audit.jsonl"
    named = trace.is_file() and "commission_recorded" in trace.read_text(encoding="utf-8", errors="replace")
    events = [e.get("sha256") for e in (audit_log.read(trace) if named else []) if e.get("kind") == "commission_recorded"]
    if events and events[-1]:
        return str(events[-1])
    staged = staged_path(quest_root)
    return sha256(staged) if staged.is_file() else None


def resume_problem(path: Path | None, quest_root: Path) -> str | None:
    """Why a resumed quest cannot go on with the commission its config names now (``None``: it can). Checked before any
    step and before the quest's own copy of the config is replaced."""
    had = recorded_sha(quest_root)
    if had is None:
        if path is not None and (Path(quest_root) / "plan.md").is_file():
            return (f"quest {Path(quest_root).name} already has a plan.md that was not written from a commission, so "
                    f"the commission {path} cannot be given to it; start a new quest with it")
        return None
    if path is None:
        return (f"quest {Path(quest_root).name} was started from a commission and the config given no longer names one; "
                "resume it with the config it was started with")
    if sha256(path) != had:
        return (f"the commission {path} is not the one quest {Path(quest_root).name} was started from (its content "
                "changed), and a commission cannot change after it is handed over; nothing was run. Resume with the "
                "file as it was, or start a new quest")
    return None


def door(path: Path | None, quest_root: Path | None) -> None:
    """The commission's checks before any model call: a new quest's file is validated; a resumed quest's is compared
    with the one it was started from (and validated when it is given for the first time). Raises :class:`CommissionRefused`."""
    if quest_root is not None:
        why = resume_problem(path, quest_root)
        if why:
            raise CommissionRefused(why)
        if path is None or recorded_sha(quest_root) is not None:
            return
    if path is not None:
        validate(path)


def given_sources(literature: list[Any], data: dict[str, Any]) -> list[Any]:
    """The literature with each supplied full text named by its source's title and DOI from the commission, so the plan
    and the paper can cite it by a label like any other source (a file dropped in ``inputs/papers/`` has no title)."""
    names = staged_names(data)
    by_name = {names[str(kv["source"]["file"])]: kv["source"] for kv in data.get("known_values") or []
               if str(kv["source"]["file"]) in names}
    out = []
    for entry in literature:
        meta = entry.get("metadata") if isinstance(entry, dict) else None
        src = by_name.get(Path(str((meta or {}).get("path") or "")).name) if meta else None
        if src and meta.get("source") == "user_supplied":
            entry = {**entry, "metadata": {**meta, "title": str(src["title"]),
                                           **({"doi": str(src["doi"])} if str(src.get("doi") or "").strip() else {})}}
        out.append(entry)
    return out


def prompt_block(data: dict[str, Any], labels: dict[str, str]) -> str:
    """What the commission fixes, for the plan's model. ``labels``: a known value's source file -> its label in the
    literature list ([n])."""
    lim = data.get("limits") or {}
    lines = ["WHAT THE COMMISSION FIXES (the study was commissioned; do not change the question, the scope or these values)",
             f"Question: {data.get('question')}", f"Scope: {data.get('scope')}",
             f"Limits: at most {lim.get('hours')} hours of computing and {lim.get('model_calls')} model calls."]
    known = data.get("known_values") or []
    if known:
        lines.append("Known values. Give each one a check in `protocol.oracles` of kind `published_value`, with this exact "
                     "`name`, the value as `expected`, this `tolerance`, and its source's label as `reference`:")
        for kv in known:
            label = labels.get(str(kv["source"]["file"]))
            where = f"[{label}]" if label else f"the supplied full text {Path(str(kv['source']['file'])).name}"
            lines.append(f"- name {kv['name']!r}: {kv['value']} with tolerance {kv['tolerance']}, for {kv['case']}; "
                         f"stated in {where} ({kv['source']['title']}): \"{kv['passage']}\"")
    else:
        lines.append("Known values: none were given.")
    for key, head in (("success", "Counts as supported"), ("failure", "Counts as not supported")):
        lines.append(f"{head}:")
        lines += [f"- {i['condition']} (checked by: {i['verify']})" for i in data.get(key) or []]
    doubts = data.get("doubts") or []
    lines.append("Known doubts:" if doubts else "Known doubts: none.")
    lines += [f"- {d}" for d in doubts]
    return "\n".join(lines)


def _fold(name: Any) -> str:
    return re.sub(r"\s+", " ", str(name or "")).strip().lower()


def oracle_gaps(data: dict[str, Any], protocol: Any) -> list[str]:
    """For plan.md's audit list: each known value the plan has no check for (no entry of ``protocol.oracles`` with its
    name, case and spaces aside). FI does not add the check itself here."""
    from core import oracle_check

    have = {_fold(o.get("name")) for o in oracle_check.declared(protocol if isinstance(protocol, dict) else None)}
    return [f"The commission's known value {kv['name']!r} ({kv['value']} with tolerance {kv['tolerance']}) has no check "
            f"in this plan: no entry of `protocol.oracles` is named {kv['name']!r}."
            for kv in data.get("known_values") or [] if _fold(kv.get("name")) not in have]


def too_long_problem(fit: dict[str, Any] | None, said: str) -> str:
    """The sentence of a commissioned study handed back because FI's timing says it cannot be done in the time allowed,
    from what FI measured (``too_long_fit``: seconds per setting, settings, settings that fit, the limit)."""
    if isinstance(fit, dict) and fit.get("seconds_per_setting") is not None and fit.get("settings"):
        said = (f"FI measured {'at least ' if fit.get('at_least') else 'about '}{float(fit['seconds_per_setting']):g} s "
                f"for each of the study's {int(fit['settings']):,} settings, and only "
                f"{int(fit.get('settings_that_fit') or 0):,} of them fit the {float(fit.get('limit_s') or 0) / 3600:g} h allowed")
    return (f"the study the commission fixes cannot be done in the time allowed ({said or 'FI timed it as too long'}); "
            "FI does not make a commissioned study smaller, so it is handed back")


def return_record(reason: str, problem: str, evidence: Any, tried: list[str], **rest: Any) -> dict[str, Any]:
    """The record of a study handed back, ``return_reason`` first (a reader that quotes the head of the file shows it)."""
    if reason not in RETURN_REASONS:
        raise ValueError(f"a study is handed back for one of {', '.join(RETURN_REASONS)}, not {reason!r}")
    rest = {k: v for k, v in rest.items() if k not in ("kind", "problem", "tried", "say", "evidence", "return_reason")}
    return {"return_reason": reason, "kind": "returned", "evidence": evidence, "problem": problem, "tried": list(tried),
            "say": f"FI handed the study back: {problem}. No paper was written; the results are kept.",
            "at": rest.pop("at", time.time()), **rest}


def write_return(quest_root: Path, reason: str, problem: str, evidence: Any, tried: list[str],
                 **rest: Any) -> dict[str, Any]:
    """Write :func:`return_record` to ``needs/STUCK.json`` (the honest stop's file) and return it."""
    record = return_record(reason, problem, evidence, tried, **rest)
    needs = Path(quest_root) / "needs"
    needs.mkdir(parents=True, exist_ok=True)
    (needs / "STUCK.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    return record
