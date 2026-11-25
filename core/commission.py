"""A study FI is commissioned to run from one file (``Config.commission``), and handed back when it cannot be done.

The file's rules live here only: :func:`problems` / :func:`validate` are what FI runs before any model call, and what
another program runs to check a file before it hands it over. FI refuses a file that breaks one; it never fills one in.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

STAGED_NAME = "commission.json"  # the quest's frozen copy, in inputs/
# The fourth: a study run one step at a time (Config.step) whose step would go back to an earlier one (its ``step``).
RETURN_REASONS = ("known_value_mismatch", "plan_contradicts_sources", "scope_exceeds_cap", "needs_earlier_step")
GIVEN_STEPS = ("ideas", "literature", "studies")  # the --from steps the commission gives
GIVEN_STAGES = {"ideate": "", "scope": "", "literature": "Reading the sources the commission gives."}
MIN_TEXT_CHARS = 200  # the floor of a paper dropped in inputs/papers/
# A number as a passage may write it: a sign, thousands separators, a bare leading point, an exponent (e-3, x 10^-3)
# on the same line (a page header "2023" over a line "e20230014" is two numbers, not 2023e20230014).
_NUMBER = re.compile(r"(?<![\w.])([-+−]?)(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d*\.\d+|\d+)"
                     r"(?:[^\S\r\n]*(?:[eE]|[x×·][^\S\r\n]*10\^?)[^\S\r\n]*([-+−]?\d+))?")


class CommissionRefused(ValueError):
    """A commission FI cannot start from; the message names the key, file or value."""


def _number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _text(v: Any) -> bool:
    return isinstance(v, str) and bool(v.strip())


def number_value(m: re.Match[str]) -> float | None:
    """The number a match of :data:`_NUMBER` writes, with its sign; ``None`` for one whose exponent no float holds
    (it is no value of a commission)."""
    sign, digits, exp = m.groups()
    try:
        found = float(digits.replace(",", "")) * 10.0 ** int((exp or "0").replace("−", "-"))
    except OverflowError:
        return None
    return -found if sign in ("-", "−") else found


def number_in_text(value: float, text: str) -> bool:
    """Whether ``text`` writes ``value`` with its sign (0.375 as 0.3750 or .375, 1200 as 1,200, 0.001 as 1e-3); a
    number that is one side of a fraction or ratio of two numbers written with "/" (1/2, 1/((2))) is not that value, while a
    unit written with one (0.25/h) leaves it the value."""
    text = str(text or "")
    for m in _NUMBER.finditer(text):
        if (re.search(r"\d[\s)\]]*/[\s(\[]*$", text[:m.start()])
                or re.match(r"[\s)\]]*/[\s(\[]*[-+−]?\.?\d", text[m.end():])):
            continue
        found = number_value(m)
        if found is not None and math.isclose(found, float(value), rel_tol=1e-9, abs_tol=0.0):
            return True
    return False


def _too_short(passage: str) -> int:
    """The length ``engine_base._quote_in_source`` needs when ``passage`` is shorter (measured as it does), else 0."""
    from core.engine_base import _QUOTE_MIN_CHARS, _normalized_text

    have = sum(len(p.strip(" .,;:[]")) for p in re.split(r"\.\.\.", _normalized_text(passage)))
    return _QUOTE_MIN_CHARS if have < _QUOTE_MIN_CHARS else 0


def read(path: Path) -> dict[str, Any]:
    """The commission at ``path``: JSON (as another program writes it), else YAML, where 5e-05 is a number too."""
    import yaml

    class Loader(yaml.SafeLoader):  # YAML 1.1 reads an exponent without a dot (json.dumps's 5e-05) as text
        pass

    Loader.add_implicit_resolver("tag:yaml.org,2002:float", re.compile(r"^[-+]?\d+[eE][-+]?\d+$"), list("-+0123456789"))
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as e:
        raise CommissionRefused(f"the commission {Path(path)} cannot be read ({e.strerror or e})") from e
    try:
        data = json.loads(text)
    except ValueError:
        try:
            data = yaml.load(text, Loader=Loader)  # noqa: S506 - a SafeLoader with one more number form
        except yaml.YAMLError as e:
            raise CommissionRefused(f"the commission {Path(path)} is not valid JSON") from e
    if not isinstance(data, dict):
        raise CommissionRefused(f"the commission {Path(path)} is not a set of named keys")
    return data


def form_problems(data: dict[str, Any]) -> list[str]:
    """What is missing or of the wrong type in the commission ``data``, one plain sentence each."""
    out = [f"the commission has no `{k}` (a sentence)" for k in ("question", "scope") if not _text(data.get(k))]
    limits = data.get("limits") if isinstance(data.get("limits"), dict) else None
    if limits is None:
        out.append("the commission has no `limits` (`hours` and `model_calls`)")
    else:
        if not (_number(limits.get("hours")) and limits["hours"] > 0):
            out.append("`limits.hours` is missing or is not a number above 0")
        if not (_number(limits.get("model_calls")) and float(limits["model_calls"]).is_integer() and limits["model_calls"] > 0):
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
        elif (need := _too_short(kv["passage"])):
            out.append(f"the passage of {who} is too short to be found in its source: give the sentence of the source "
                       f"that states it, at least {need} characters")
        elif _number(kv.get("value")) and not number_in_text(kv["value"], kv["passage"]):
            out.append(f"the passage of {who} does not state its value {kv['value']}")
        src = kv.get("source") if isinstance(kv.get("source"), dict) else None
        if src is None:
            out.append(f"{who} has no `source` (`title`, `file`, and `doi` when it has one)")
            continue
        out += [f"the source of {who} has no `{k}`" for k in ("title", "file") if not _text(src.get(k))]
        if "doi" in src and not isinstance(src["doi"], str):
            out.append(f"the `doi` of the source of {who} is not text")
    for key in ("success", "failure"):
        items = data.get(key)
        if not isinstance(items, list) or not items:
            out.append(f"the commission has no `{key}` (a list of `condition` and `verify`)")
            continue
        out += [f"`{key}` item {i + 1} needs both a `condition` and how to `verify` it" for i, item in enumerate(items)
                if not (isinstance(item, dict) and _text(item.get("condition")) and _text(item.get("verify")))]
    if not isinstance(data.get("doubts"), list) or not all(isinstance(d, str) for d in data["doubts"]):
        out.append("the commission has no `doubts` (a list of sentences; `[]` when there are none)")
    return out


def source_problems(data: dict[str, Any], base: Path) -> list[str]:
    """Each full text that is missing, unreadable, under :data:`MIN_TEXT_CHARS` characters, or without its passage."""
    from core import pdf_text
    from core.engine_base import _quote_in_source

    out: list[str] = []
    texts: dict[str, str | None] = {}
    for kv in data.get("known_values") or []:
        raw = str(kv["source"]["file"])
        if raw not in texts:
            path, texts[raw] = _resolve(raw, base), None
            if not path.is_file():
                out.append(f"the source file {raw!r} of the known value {kv['name']!r} does not exist ({path})")
            elif path.suffix.lower() not in (".pdf", ".md", ".txt"):
                out.append(f"the source file {raw!r} is not one FI reads (.pdf, .md or .txt)")
            else:
                try:
                    text = ((pdf_text.extract(path.read_bytes()).text or "") if path.suffix.lower() == ".pdf"
                            else path.read_text(encoding="utf-8", errors="replace")).strip()
                except (OSError, ValueError) as e:
                    out.append(f"the source file {raw!r} cannot be read ({e})")
                else:
                    if len(text) < MIN_TEXT_CHARS:
                        out.append(f"the source file {raw!r} has under {MIN_TEXT_CHARS} characters of text")
                    else:
                        texts[raw] = text
        if texts[raw] is not None and not _quote_in_source(str(kv["passage"]), texts[raw]):
            out.append(f"the passage of the known value {kv['name']!r} is not in its source file {raw!r}")
    return out


def problems(path: Path) -> list[str]:
    """Every rule the commission at ``path`` breaks (empty: FI can start from it). Reads the full texts it supplies."""
    try:
        data = read(path)
    except CommissionRefused as e:
        return [str(e)]
    return form_problems(data) or source_problems(data, Path(path).resolve().parent)


def validate(path: Path, *, sources: bool = True) -> None:
    """Raise :class:`CommissionRefused` naming every problem; ``sources=False`` reads no full text."""
    found = problems(path) if sources else form_problems(read(path))
    if found:
        raise CommissionRefused(f"the commission {Path(path)} cannot be used, so no study was started: "
                                + "; ".join(found) + ". Correct the file and start again.")


def locate(raw: Any, folder: Path) -> Path:
    """A config's commission, read against its folder; a quest's own config.yaml falls back to the frozen copy."""
    path, staged = _resolve(str(raw), folder), staged_path(folder)
    return staged if not path.is_file() and staged.is_file() else path


def _resolve(raw: str, folder: Path) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else Path(folder) / path


def staged_path(quest_root: Path) -> Path:
    return Path(quest_root) / "inputs" / STAGED_NAME


def staged_names(data: dict[str, Any]) -> dict[str, str]:
    """Each source file -> its name in inputs/papers/, from the file's words alone: numbered when two share one (case
    aside), and never a name the papers reader skips (``engine_base._ingest_user_dropped_papers``: a leading "." or
    README.md)."""
    out: dict[str, str] = {}
    for raw in (str((kv.get("source") or {}).get("file") or "") for kv in data.get("known_values") or []):
        if raw and raw not in out:
            path, n = Path(raw), 1
            stem = path.stem.lstrip(".")
            if path.name.startswith(".") or path.name.lower() == "readme.md":
                stem = f"source_{stem}"
            name = f"{stem}{path.suffix}"
            while name.lower() in {v.lower() for v in out.values()}:
                n += 1
                name = f"{stem}-{n}{path.suffix}"
            out[raw] = name
    return out


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stage(path: Path, quest_root: Path) -> dict[str, str]:
    """Copy the commission to inputs/commission.json and its full texts to inputs/papers/; return each copy (its path
    in the quest) -> its sha256, the commission's first."""
    papers = Path(quest_root) / "inputs" / "papers"
    papers.mkdir(parents=True, exist_ok=True)
    texts = {}
    for raw, name in staged_names(read(path)).items():
        shutil.copy2(_resolve(raw, Path(path).resolve().parent), papers / name)
        texts[f"inputs/papers/{name}"] = sha256(papers / name)
    files = {f"inputs/{STAGED_NAME}": sha256(path), **texts}  # the copy is the file itself, byte for byte
    manifest = _manifest_path(quest_root)  # the hashes are on disk before the copy that says the quest was staged
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.with_name(manifest.name + ".tmp").write_text(json.dumps(files, indent=1), encoding="utf-8")
    os.replace(manifest.with_name(manifest.name + ".tmp"), manifest)
    copy = staged_path(quest_root).with_name(STAGED_NAME + ".tmp")  # whole or not there: never half a file
    shutil.copy2(path, copy)
    os.replace(copy, staged_path(quest_root))  # last: this copy is what says the quest was staged
    return files


def _manifest_path(quest_root: Path) -> Path:
    return Path(quest_root) / ".fi" / "commission_files.json"


def _manifest(quest_root: Path) -> dict[str, str] | None:
    """The manifest written with the staging; ``None`` when it is missing or unreadable."""
    try:
        kept = json.loads(_manifest_path(quest_root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return kept if isinstance(kept, dict) else None


def staged_files(quest_root: Path) -> dict[str, str]:
    """Each file the staging copied (its path in the quest) -> its sha256: the manifest written with the staging, with
    what the trace recorded laid over it (so a quest stopped before the trace event is checked all the same)."""
    return {**(_manifest(quest_root) or {}), **(_recorded(quest_root).get("files") or {})}


def _recorded(quest_root: Path) -> dict[str, Any]:
    """The last ``commission_recorded`` event of the quest's trace (empty when there is none)."""
    from core import audit_log

    trace = Path(quest_root) / ".fi" / "audit.jsonl"
    named = trace.is_file() and "commission_recorded" in trace.read_text(encoding="utf-8", errors="replace")
    events = [e for e in (audit_log.read(trace) if named else []) if e.get("kind") == "commission_recorded"]
    return events[-1] if events else {}


def recorded_sha(quest_root: Path) -> str | None:
    """The sha256 of the commission the quest was started from: the one its trace recorded, else its copy's."""
    if _recorded(quest_root).get("sha256"):
        return str(_recorded(quest_root)["sha256"])
    return sha256(staged_path(quest_root)) if staged_path(quest_root).is_file() else None


def door(path: Path | None, quest_root: Path | None) -> None:
    """Before any model call: a new quest's commission is validated; a resumed quest's must be the one it began with.
    A study without a commission is refused, new or resumed (``Engine`` itself does not refuse: every start path calls
    this first)."""
    had = recorded_sha(quest_root) if quest_root is not None else None
    name = Path(quest_root).name if quest_root is not None else ""
    if had is None and path is None:
        raise CommissionRefused(
            "no study was started: every study starts from a commission file, which this config does not name "
            "(`commission: <file>`); `fi new` writes one to fill in, with a config that names it, and `fi check <file>` "
            "checks it" if quest_root is None
            else f"quest {name} was started without a commission file, and FI no longer goes on with a study that has "
            "none; start a new study from a commission (`fi new`)")
    if had is None and path is not None and quest_root is not None and (Path(quest_root) / "plan.md").is_file():
        raise CommissionRefused(f"quest {name} already has a plan.md that was not written from a commission, so the "
                                f"commission {path} cannot be given to it; start a new quest with it")
    if had is not None and path is None:
        raise CommissionRefused(f"quest {name} was started from a commission and the config given no longer names one; "
                                "resume it with the config it was started with")
    if had is not None and sha256(path) != had:  # type: ignore[arg-type]
        raise CommissionRefused(f"the commission {path} is not the one quest {name} was started from (its content "
                                "changed), and a commission cannot change after it is handed over; nothing was run. "
                                "Resume with the file as it was, or start a new quest")
    root = Path(quest_root or ".")
    if had is not None and not _recorded(root) and _manifest(root) is None:
        raise CommissionRefused(f"quest {name} was staged from a commission but the record of its files is missing, so "
                                "its copies cannot be checked; nothing was run. Start a new quest")
    changed = [rel for rel, want in (staged_files(root) if had else {}).items()
               if not (root / rel).is_file() or sha256(root / rel) != want]
    if changed:
        raise CommissionRefused(f"quest {name}'s own copy of what its commission gave is no longer the one it was started "
                                f"with ({', '.join(changed)} changed or missing); nothing was run. Put the files back as "
                                "they were, or start a new quest")
    if path is not None and had is None:
        validate(path)


def given_sources(literature: list[Any], data: dict[str, Any]) -> list[Any]:
    """The literature with each supplied full text titled from the commission, so it can be cited by a label."""
    names = staged_names(data)
    by_name = {names[str(kv["source"]["file"])]: kv["source"] for kv in data.get("known_values") or []}
    out = []
    for entry in literature:
        meta = entry.get("metadata") if isinstance(entry, dict) else None
        src = by_name.get(Path(str(meta.get("path") or "")).name) if meta else None
        if src and meta.get("source") == "user_supplied":
            doi = {"doi": str(src["doi"])} if str(src.get("doi") or "").strip() else {}
            entry = {**entry, "metadata": {**meta, "title": str(src["title"]), **doi}}
        out.append(entry)
    return out


def prompt_block(data: dict[str, Any], labels: dict[str, str]) -> str:
    """What the commission fixes, for the plan's model; ``labels``: source file -> its [n] in the literature."""
    lim = data.get("limits") or {}
    lines = ["WHAT THE COMMISSION FIXES (the study was commissioned; do not change the question, the scope or these values)",
             f"Question: {data.get('question')}", f"Scope: {data.get('scope')}",
             f"Limits: at most {lim.get('hours')} hours of computing and {lim.get('model_calls')} model calls."]
    known = data.get("known_values") or []
    lines.append("Known values. Give each one a check in `protocol.oracles` with this exact `name`, the study's `case` and "
                 "its `measure`; FI writes its kind, expected value, tolerance and source from the commission:"
                 if known else "Known values: none were given.")
    for kv in known:
        label = labels.get(str(kv["source"]["file"]))
        where = f"[{label}]" if label else f"the supplied full text {Path(str(kv['source']['file'])).name}"
        lines.append(f"- name {kv['name']!r}: {kv['value']} with tolerance {kv['tolerance']}, for {kv['case']}; "
                     f"stated in {where} ({kv['source']['title']}): \"{kv['passage']}\"")
    for key, head in (("success", "Counts as supported"), ("failure", "Counts as not supported")):
        lines += [f"{head}:", *(f"- {i['condition']} (checked by: {i['verify']})" for i in data.get(key) or [])]
    doubts = data.get("doubts") or []
    lines += ["Known doubts:" if doubts else "Known doubts: none.", *(f"- {d}" for d in doubts)]
    return "\n".join(lines)


def oracle_gaps(data: dict[str, Any], protocol: Any) -> list[str]:
    """plan.md audit notes: each known value no entry of ``protocol.oracles`` is named for (case and spaces aside), for
    which FI adds a check (``known_values.fill_checks``) when the plan has a protocol."""
    from core import oracle_check

    def fold(name: Any) -> str:
        return re.sub(r"\s+", " ", str(name or "")).strip().lower()

    have = {fold(o.get("name")) for o in oracle_check.declared(protocol if isinstance(protocol, dict) else None)}
    return [f"The commission's known value {kv['name']!r} ({kv['value']} with tolerance {kv['tolerance']}) has no check "
            f"in this plan's draft: no entry of `protocol.oracles` is named {kv['name']!r}"
            + (", so FI added one from the commission (its number, tolerance and source; the plan gives it no case or "
               "measure)." if isinstance(protocol, dict) else ".")
            for kv in data.get("known_values") or [] if fold(kv.get("name")) not in have]


def too_long_problem(fit: dict[str, Any] | None, said: str) -> str:
    """The sentence of a study handed back as too long, from what FI measured (``too_long_fit``)."""
    if isinstance(fit, dict) and fit.get("seconds_per_setting") is not None and fit.get("settings"):
        said = (f"FI measured {'at least ' if fit.get('at_least') else 'about '}{float(fit['seconds_per_setting']):g} s "
                f"for each of the study's {int(fit['settings']):,} settings, and only "
                f"{int(fit.get('settings_that_fit') or 0):,} of them fit the {float(fit.get('limit_s') or 0) / 3600:g} h allowed")
    return (f"the study the commission fixes cannot be done in the time allowed ({said or 'FI timed it as too long'}); "
            "FI does not make a commissioned study smaller, so it is handed back")


def mismatch_reason(oracle: dict[str, Any], data: dict[str, Any]) -> str:
    """Why a check whose numbers disagree hands the study back, by where its expected value came from: the commission
    or a quoted source (a published value) gives ``known_value_mismatch``; the plan's own working ``plan_contradicts_sources``."""
    from core import oracle_check

    def fold(name: Any) -> str:
        return re.sub(r"\s+", " ", str(name or "")).strip().lower()

    given = {fold(kv.get("name")) for kv in data.get("known_values") or [] if isinstance(kv, dict)}
    return ("known_value_mismatch" if oracle_check.kind_of(oracle) == "published_value" or fold(oracle.get("name")) in given
            else "plan_contradicts_sources")


def known_answer_problem(name: str, found: dict[str, Any], reason: str) -> str:
    """The sentence of a study handed back because a known-answer check's three numbers disagree
    (core/oracle_challenge.py::three_disagree), in a scientist's words."""
    n = found["numbers"]
    expected = "the commission's known value" if reason == "known_value_mismatch" else "the plan's expected value"
    return (f"the known-answer check {name!r} has three numbers that disagree beyond their own errors: FI's simulation "
            f"measured {n['simulation']['value']:.6g} (± {n['simulation']['error']:.2g}), FI's independent recomputation "
            f"gives {n['reference']['value']:.6g} (± {n['reference']['error']:.2g}) and {expected} is {n['plan']:.6g}; "
            "FI did not run the study, so it is handed back")


def write_return(quest_root: Path, reason: str, problem: str, evidence: Any, tried: list[str], **rest: Any) -> dict[str, Any]:
    """Write needs/STUCK.json for a study handed back, ``return_reason`` first, and return the record."""
    if reason not in RETURN_REASONS:
        raise ValueError(f"a study is handed back for one of {', '.join(RETURN_REASONS)}, not {reason!r}")
    rest = {k: v for k, v in rest.items() if k not in ("kind", "problem", "tried", "say", "evidence", "return_reason")}
    record = {"return_reason": reason, "kind": "returned", "evidence": evidence, "problem": problem, "tried": list(tried),
              "say": f"FI handed the study back: {problem}. No paper was written; the results are kept.",
              "at": rest.pop("at", time.time()), **rest}
    (Path(quest_root) / "needs").mkdir(parents=True, exist_ok=True)
    (Path(quest_root) / "needs" / "STUCK.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    return record


# What `fi new` writes: every key a commission has, each with a line saying what goes there. Left as it is, `fi check`
# names every key still to fill in.
TEMPLATE = """\
# The study FI is asked to run, written by `fi new`. Fill in every key, then check the file:  fi check {name}
# FI runs the study as written here and never changes it; a study it cannot do is handed back with the reason.

# The research question, in one sentence.
question: ""

# What the study covers and what it leaves out: the cases, the ranges of the parameters, the methods compared.
scope: ""

# The most the study may use: hours of computing and model calls.
limits:
  hours: 1
  model_calls: 60

# Values already known for a case of the study, each stated in a source whose full text you give. FI checks its own
# results against them. One block per value (remove the leading "# "):
# - name: ""          # a short name for the value
#   value: 0.0        # the number
#   tolerance: 0.0    # how far a result may be from it and still agree
#   case: ""          # the situation it applies to: which form, which parameter values
#   passage: ""       # the sentence of the source that states the value, copied as it is written
#   source:
#     title: ""       # the source's title
#     file: ""        # its full text (.pdf, .md or .txt), relative to this file
#     doi: ""         # its DOI, when it has one
known_values: []

# true when no value is known yet (the results then have no outside value to agree with).
no_known_values: false

# What counts as the question answered yes, and how each is checked.
success:
  - condition: ""
    verify: ""

# What counts as the question answered no, and how each is checked.
failure:
  - condition: ""
    verify: ""

# What you already doubt about the study, one sentence each; [] when there is nothing.
doubts: []
"""

# The config `fi new` writes beside it: the commission gives the study, the config how FI runs it.
CONFIG_TEMPLATE = """\
# How FI runs the study in {commission}, written by `fi new`. Fill in the commission, then start:  fi --config {name}

commission: {commission}

# The model: provider.name (openai, gemini, claude_cli, codex_cli, gemini_cli, ollama or vscode_extension) and, if
# you like, provider.model.
provider:
  name: {provider}

output:
  output_dir: ./outputs
"""


def write_new(folder: Path, provider: str) -> tuple[Path, Path]:
    """``fi new``: a commission to fill in and the config that names it, in ``folder``; never over an existing file
    (``commission-2.yaml`` with ``study-2.yaml``, and so on). Returns (commission, config)."""
    n = 1
    while True:
        tail = "" if n == 1 else f"-{n}"
        commission, config = Path(folder) / f"commission{tail}.yaml", Path(folder) / f"study{tail}.yaml"
        if not (commission.exists() or config.exists()):
            break
        n += 1
    for path, text in ((commission, TEMPLATE.format(name=commission.name)),
                       (config, CONFIG_TEMPLATE.format(commission=commission.name, name=config.name, provider=provider))):
        with open(path, "x", encoding="utf-8", newline="\n") as fh:  # "x": never over a file that appeared meanwhile
            fh.write(text)
    return commission, config


def check(path: Path) -> tuple[bool, list[str]]:
    """``fi check <file>``: whether the commission can be used, and one plain sentence per rule it breaks (the rules
    FI runs before a study starts), or one saying it can be used. No model is asked."""
    path = Path(path)
    try:
        raw = read(path)
    except CommissionRefused:
        raw = {}
    if raw.get("commission") and "question" not in raw:  # a config that names its commission: that file is checked
        path = locate(raw["commission"], path.resolve().parent)
    found = problems(path)
    if found:
        return False, [f"{p[:1].upper()}{p[1:]}." for p in found]
    sources = " and each passage is in its source" if (read(path).get("known_values") or []) else ""
    return True, [f"The commission {path} can be used: every key is there{sources}."]
