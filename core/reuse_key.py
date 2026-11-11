"""One key per kind of work FI decides "this was already done, reuse it" for.

Every place that reuses a computed record (the trials, the search for the best design, the timing of a run) builds its key
here, with the same function for the writer and the reader, and the record keeps what went into the key
(:class:`Inputs`), so a later reader can say in plain words what changed (:meth:`Inputs.changed`).

What a key covers, and what it does not:

* **The simulation part** (:func:`files`, ``"simulation"``): every file :func:`core.code_layout.simulation_inputs` lists
  (the simulation, its helper modules and package, parameter files, the person's ``inputs/``) and every file of
  ``data/`` but what the study writes under ``data/results/``: all of them are in the copy a cell runs from
  (:mod:`core.outside_run`), so the run can read them. Each file is hashed as the bytes the run can read: a comment may
  hold a value, and FI cannot show that a ``.py`` file is only executed. The cost of that is a rerun after a comment-only
  change, never a wrong reuse.
* **Left out, as a stated limit**: the analysis FI writes (it reads what the simulation wrote), ``tests/``, FI's own
  copies and notes, and what a run writes under ``cache``, ``results``, ``figures`` and ``raw``. Files reached by an
  absolute path, and packages installed in the environment, are not covered at all.
* **An unreadable file** makes the inputs incomplete (:attr:`Inputs.complete`): the record is then not reusable, and a
  reader whose inputs are incomplete reuses nothing, so two unreadable reads never match.
* **FI's own code** (its harness, and for a search its search code) is an input: a record made by a harness that no
  longer exists is not reused. A frozen study's search is the one exception, by what decides its result: the confirm stage
  may not search again, so it keeps a search whose files and plan are as recorded although FI's code changed since, and says
  so (core/optimise.py ``_same_study``).

A record written under this rule carries ``reuse: {"schema": 2, ...}`` (:func:`stamp`). A record without it was written
by an older FI and is read by the rule it was written with (:func:`is_current`).

Pure and synchronous: nothing here keeps state between calls."""
from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: The record format this module writes (``reuse.schema``).
SCHEMA = 2
#: What each kind's decision means: raised when FI changes the rule that decided it, so a record made under another
#: rule is decided again (a ``fits`` timing written before the speed-up rule existed was reused, and the study was never
#: sped up).
RULES = {"trials": 2, "search": 2, "search_study": 2, "search_study_bytes": 2, "sizing": 2, "rerun": 2}

#: The plain words :meth:`Inputs.changed` uses for what is not a file.
_SETTING_WORDS = {
    "block": "the plan's search block", "thresholds": "the plan's thresholds", "seed": "the seed",
    "entry": "the simulation's entry function", "rounds": "the extra rounds of the search", "stops": "the search's stop settings",
    "protocol": "the plan", "runs": "the number of runs", "deterministic": "how the simulation is run (once per setting or not)",
    "limit_s": "the time allowed", "at_once": "how many settings run at once",
}
_FI_WORDS = {"harness": "FI's own trial harness", "search": "FI's own search code"}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest(path: Path | str) -> str:
    """The hash of the raw bytes of one file: what a run can read. Raises ``OSError`` when it cannot be read."""
    return _sha(Path(path).read_bytes())


def version_digest(path: Path | str) -> str:
    """One file's hash as a version of the study (line endings and, for Python, comments and blank lines do not count):
    the rule :mod:`core.phased` decides "the same version" by, used only where a narrower key is the safe side (the
    frozen-search lookup)."""
    from . import phased as _phased

    return _phased._code_digest(Path(path))


@dataclass
class Files:
    """``{label: digest}`` of the files of one part, and whether every listed file could be read."""

    digests: dict[str, str] = field(default_factory=dict)
    complete: bool = True
    unreadable: list[str] = field(default_factory=list)


def _under_data(root: Path) -> list[Path]:
    """Every file of ``data/`` but what the study writes under ``data/results/`` (the files a cell's copy holds)."""
    from . import outside_run as _outside

    out: list[Path] = []
    base = root / "data"
    if not base.is_dir():
        return out
    skip = tuple(s.rstrip("/") + "/" for s in _outside.CELL_SKIP)
    for path in base.rglob("*"):
        try:
            if not path.is_file():
                continue
        except OSError:
            continue
        rel = path.relative_to(root).as_posix()
        if any((rel + "/").startswith(s) for s in skip) or rel in tuple(s.rstrip("/") for s in _outside.CELL_SKIP):
            continue
        out.append(path)
    return sorted(out, key=lambda p: p.relative_to(root).as_posix())


def _analysis_paths(root: Path) -> list[Path]:
    from . import code_layout as _layout

    code = root / "code"
    out: list[Path] = []
    for name in (_layout.ANALYSIS_SCRIPT, _layout.TABLE_SCRIPT, _layout.FIGURE_CALLS_SCRIPT, "derive.py", "fi_figures.py"):
        if (code / name).is_file():
            out.append(code / name)
    core = code / _layout.FI_CORE_FOLDER
    if core.is_dir():
        out.extend(p for p in core.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    return out


def files(quest_root: Path | str, part: str, *, by: str = "bytes", data: bool = True,
          code: Path | None = None) -> Files:
    """The digests of one part of the quest's files.

    ``"simulation"``: :func:`core.code_layout.simulation_inputs` and ``data/`` (see the module text). ``"analysis"``: the
    analysis FI writes and ``experiment.py``. ``"study"``: both, and ``tests/``. ``by="tokens"`` hashes a Python file as
    :func:`version_digest` does (the frozen-search lookup only); every computed record is hashed by its bytes. ``data=False``
    leaves ``data/`` out (the frozen-search lookup: the confirm stage may not search again, so a change to ``data/`` after
    the freeze, which main's lookup never saw either, is not a reason to stop it; the confirm run's own trials, keyed with
    ``data/``, run again)."""
    from . import code_layout as _layout

    root = Path(quest_root)
    paths: list[Path] = []
    if part in ("simulation", "study"):
        paths.extend(_layout.simulation_inputs(root, code))
        if data:
            paths.extend(_under_data(root))
    if part in ("analysis", "study"):
        paths.extend(_analysis_paths(root))
    if part == "study":
        tests = root / "code" / _layout.TESTS_DIR
        if tests.is_dir():
            paths.extend(p for p in tests.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    if part not in ("simulation", "analysis", "study"):
        raise ValueError(f"unknown part of the quest's files: {part!r}")
    out = Files()
    for path in sorted(set(paths), key=lambda p: _layout.input_label(p, root)):
        label = _layout.input_label(path, root)
        try:
            out.digests[label] = version_digest(path) if by == "tokens" else digest(path)
        except OSError:
            out.complete = False
            out.unreadable.append(label)
    return out


@dataclass
class Inputs:
    """Everything that decides one reusable record, as plain data (so the record can keep it)."""

    kind: str
    rule: int
    files: dict[str, str]
    settings: dict[str, Any]
    fi_code: dict[str, str]
    complete: bool = True
    unreadable: list[str] = field(default_factory=list)
    #: How to build the same inputs again with one thing changed (:meth:`prefix`, :meth:`stop_variants`); not stored.
    _recipe: Any = field(default=None, repr=False, compare=False)

    def as_json(self) -> dict[str, Any]:
        return {"kind": self.kind, "rule": self.rule, "files": self.files, "settings": self.settings,
                "fi_code": self.fi_code}

    @property
    def key(self) -> str:
        """The hash of the canonical JSON of the inputs. An incomplete read has a key no other read can match."""
        text = json.dumps(self.as_json(), sort_keys=True, default=str)
        if not self.complete and not self.kind.startswith("search_study"):
            # (A frozen study's lookup key skips the unreadable file as before: the confirm stage may not search again,
            # so refusing it would stop a quest that did nothing wrong.)
            text += "|unreadable:" + ",".join(self.unreadable) + f"|{uuid.uuid4().hex}"
        return _sha(text.encode("utf-8"))

    def changed(self, other: "Inputs | dict[str, Any] | None") -> list[str]:
        """Plain names of what differs from an earlier record's inputs (an ``Inputs`` or its stored JSON): file names,
        the plan's settings, FI's own code. Empty when nothing is known to differ."""
        if other is None:
            return []
        old = other.as_json() if isinstance(other, Inputs) else other
        if not isinstance(old, dict):
            return []
        out: list[str] = []
        old_files = old.get("files") if isinstance(old.get("files"), dict) else {}
        for name in sorted(set(old_files) | set(self.files)):
            if old_files.get(name) != self.files.get(name):
                out.append(name)
        old_settings = old.get("settings") if isinstance(old.get("settings"), dict) else {}
        for name in sorted(set(old_settings) | set(self.settings)):
            if old_settings.get(name) != self.settings.get(name):
                out.append(_SETTING_WORDS.get(name, name))
        old_code = old.get("fi_code") if isinstance(old.get("fi_code"), dict) else {}
        for name in sorted(set(old_code) | set(self.fi_code)):
            if old_code.get(name) != self.fi_code.get(name):
                out.append(_FI_WORDS.get(name, f"FI's {name}"))
        if old.get("rule") not in (None, self.rule):
            out.append("the rule FI decides it by")
        return out

    def prefix(self, k: int) -> "Inputs":
        """The inputs of the same search before round ``k`` (a continued search; ``k=0``: no rounds)."""
        if self._recipe is None or self.kind != "search":
            raise ValueError("only a search's inputs have an earlier round")
        recipe = dict(self._recipe)
        recipe["rounds"] = list(recipe["rounds"])[:k]
        return _build_search(**recipe)

    def stop_variants(self) -> dict[str, frozenset[str]]:
        """``{key: the stops the record must not have stopped on}`` of the searches a record of which is this search: the
        key with the stops as they are, and the key with a stop that is off now turned on (an armed stop that never fired
        changed nothing the search did, so a search that ran while a repair was still left is the same search once the
        repairs are spent)."""
        if self._recipe is None or self.kind != "search":
            raise ValueError("only a search's inputs have stops")
        from . import optimise as _optimise

        recipe = dict(self._recipe)
        stops = dict(recipe["stops"])
        off = [name for name, on in stops.items() if not on]
        out: dict[str, frozenset[str]] = {}
        for bits in range(1 << len(off)):
            armed = {name for i, name in enumerate(off) if bits >> i & 1}
            flags = {name: (on or name in armed) for name, on in stops.items()}
            key = _build_search(**{**recipe, "stops": flags}).key
            out.setdefault(key, frozenset(_optimise._STOP_REASONS[name] for name in armed))
        return out


def stamp(inputs: Inputs, *, outputs: dict[str, str] | None = None) -> dict[str, Any]:
    """The ``reuse`` block a record carries: what the key was made from, and the hashes of the files the reuse relies on."""
    return {"schema": SCHEMA, "kind": inputs.kind, "rule": inputs.rule, "key": inputs.key, "complete": inputs.complete,
            "inputs": inputs.as_json(), "outputs": dict(outputs or {})}


def is_current(record: Any, kind: str | None = None) -> bool:
    """Whether a record was written under this rule (it has a ``reuse`` block of this schema)."""
    block = record.get("reuse") if isinstance(record, dict) else None
    return isinstance(block, dict) and block.get("schema") == SCHEMA and (kind is None or block.get("kind") == kind)


def matches(record: Any, inputs: Inputs, kind: str | None = None) -> bool:
    """Whether ``record`` was made from exactly ``inputs``: written under this rule, complete on both sides, same key."""
    if not (is_current(record, kind or inputs.kind) and inputs.complete):
        return False
    block = record["reuse"]
    return bool(block.get("complete", True)) and block.get("key") == inputs.key and not block.get("opened_left_out")


def why_not(record: Any, inputs: Inputs) -> str:
    """One plain line for run.log: why an earlier record is not used (empty when there is none to talk about)."""
    if not isinstance(record, dict) or not record:
        return ""
    if not inputs.complete:
        return "FI could not read " + ", ".join(inputs.unreadable[:3]) + ", so the earlier result is not used"
    if not is_current(record, inputs.kind):
        return "the earlier result was made by an older version of FI's reuse check, so it is made again"
    left = (record.get("reuse") or {}).get("opened_left_out")
    if left and not inputs.changed((record.get("reuse") or {}).get("inputs")):
        return ("the earlier result is not used: the run read " + ", ".join(left[:3])
                + ", which the reuse check does not compare")
    names = inputs.changed((record.get("reuse") or {}).get("inputs"))
    return ("the earlier result is not used: " + ", ".join(names[:4]) + (" and more" if len(names) > 4 else "")
            + " changed since it was made") if names else ""


#: The line run.log ends a reuse decision with (what the key compared).
COMPARED = "(FI compared the code, inputs/ and data/ files this run reads)"


def left_out(opened: list[str], key_files: dict[str, str]) -> list[str]:
    """The files a run opened to read that the key does not compare: under ``code/``, ``inputs/`` or ``data/`` (paths
    relative to the folder the run ran in), not in the key's file list, and not in a folder the run itself writes into
    (``data/results/``, ``cache``, ``results``, ``figures``, ``raw`` and tool caches under ``code/``). A run that read one
    is not reused: the file may have changed without the key seeing it."""
    from . import code_layout as _layout

    known = {k.lower() for k in key_files}
    out: list[str] = []
    for rel in sorted(set(opened)):
        parts = rel.split("/")
        if len(parts) < 2 or parts[0].lower() not in ("code", "inputs", "data") or rel.lower() in known:
            continue
        if parts[0].lower() == "data" and parts[1].lower() == "results":
            continue
        if any(p in _layout.TOOL_CACHES for p in parts[:-1]) or rel.endswith(".pyc"):
            continue
        if parts[0].lower() == "code" and any(p in _layout.GENERATED_FOLDERS for p in parts[1:-1]) and not rel.endswith(".py"):
            continue
        out.append(rel)
    return out


def rerun(quest_root: Path | str, finding: str, *, test: str = "", repairs: list[str] | tuple[str, ...] = (), strict: bool = False,
          disputed: list[str] | tuple[str, ...] = (), extra: dict[str, Any] | None = None) -> Inputs:
    """What decides the answer of "run the study again with a repair of this finding": every file of the study, the finding's own
    test, the repairs kept for it (their text), how it was asked and FI's own code that judges the answer. A verdict is reused
    only when this key is the same (core/rerun_check.py)."""
    from . import rerun_check as _rerun

    root = Path(quest_root)
    found = files(root, "study")
    complete, unreadable = found.complete, list(found.unreadable)
    test_sha: str | None = None
    if test:
        try:
            test_sha = digest(root / test)
        except OSError:
            complete = False
            unreadable.append(test)
    settings = {"finding": str(finding), "test": test_sha, "repairs": [_sha(r.encode("utf-8", errors="surrogateescape")) for r in repairs],
                "strict": bool(strict), "disputed": sorted(str(x) for x in disputed), "extra": extra or {}}
    return Inputs("rerun", RULES["rerun"], found.digests, settings, {"rerun": _sha(Path(_rerun.__file__).read_bytes())}, complete,
                  unreadable)


def answer_key(prompt: str, model: str) -> str:
    """The key of one model request: the exact text sent and the model that answers it. A reply kept under it is the answer to
    that request and no other; how the reply is read is done again with today's rules."""
    return _sha(json.dumps([prompt, model], ensure_ascii=False).encode("utf-8", errors="replace"))


def digest_text(text: str) -> str:
    """The hash of a text (a record's own key built from :func:`files`)."""
    return _sha(text.encode("utf-8"))


def simulation_sha(quest_root: Path | str, extra: str = "", code: Path | None = None) -> str:
    """One hash of the simulation part (:func:`files`, ``"simulation"``) for the three small records keyed by it: the shared
    cache's code version, the timing estimate's design key and the one-trial record. ``extra`` is text hashed in with it (a
    candidate simulate.py not yet written). Unreadable files make a hash no other call can match."""
    found = files(quest_root, "simulation", code=code)
    text = json.dumps([extra, found.digests], sort_keys=True)
    if not found.complete:
        text += "|unreadable:" + ",".join(found.unreadable) + f"|{uuid.uuid4().hex}"
    return _sha(text.encode("utf-8"))


def _fi_code(kind: str) -> dict[str, str]:
    from . import optimise_search as _search
    from . import trial_runner as _trials

    out = {"harness": _sha(_trials.HARNESS_SOURCE.encode("utf-8"))}
    if kind in ("search", "search_study", "search_study_bytes"):
        out["search"] = _sha(Path(_search.__file__).read_bytes())
    return out


def _build_trials(quest_root: Path, grid: Any, runs: int, base: int, deterministic: bool, kind: str = "trials",
                  extra: dict[str, Any] | None = None) -> Inputs:
    found = files(quest_root, "simulation")
    settings = {"protocol": grid, "runs": runs, "seed": base, "deterministic": deterministic, **(extra or {})}
    return Inputs(kind, RULES[kind], found.digests, settings, _fi_code(kind), found.complete, found.unreadable)


def _build_search(*, quest_root: Path, block: dict[str, Any], base: int, entry: str, thresholds: Any,
                  rounds: list[dict[str, Any]], stops: dict[str, bool]) -> Inputs:
    from . import optimise as _optimise

    found = files(quest_root, "simulation")
    settings: dict[str, Any] = {"block": block, "seed": base, "entry": entry, "thresholds": thresholds}
    if rounds:
        settings["rounds"] = _optimise._key_rounds(rounds)
    off = {name: False for name, on in stops.items() if not on}
    if off:
        settings["stops"] = off
    out = Inputs("search", RULES["search"], found.digests, settings, _fi_code("search"), found.complete, found.unreadable)
    out._recipe = {"quest_root": quest_root, "block": block, "base": base, "entry": entry, "thresholds": thresholds,
                   "rounds": list(rounds), "stops": dict(stops)}
    return out


def computed(kind: str, quest_root: Path | str, protocol: Any = None, *, seed: int = 0, config: dict[str, Any] | None = None,
             finding: Any = None) -> Inputs:
    """The inputs of one kind of reusable work, read from the quest itself. Callers pass the quest, the protocol, the seed
    the run uses and the settings only that kind has (``config``); the rounds of a continued search are read here, and the
    stops are armed or not by ``config["stops"]``, so a reader cannot ask with other arguments than the writer did.

    ``trials`` and ``sizing``: ``config`` has ``runs`` and ``deterministic`` (and ``limit_s`` / ``at_once`` for a sizing);
    ``protocol`` is the plan's protocol (the whole of it is in the key). ``search``: ``protocol`` is the normalised
    optimisation block; ``config`` has ``entry``, ``thresholds`` and ``stops``. ``search_study`` and
    ``search_study_bytes``: a search without the seed, the rounds and the stops, by the study's version (tokens) or its
    bytes: what a frozen study's record is found by."""
    root = Path(quest_root)
    cfg = dict(config or {})
    if kind in ("trials", "sizing"):
        extra = {k: cfg[k] for k in ("limit_s", "at_once") if k in cfg}
        return _build_trials(root, protocol, int(cfg.get("runs") or 1), int(seed), bool(cfg.get("deterministic")), kind, extra)
    if kind == "search":
        from . import optimise as _optimise

        rounds = cfg["rounds"] if "rounds" in cfg else _optimise.read_rounds(root, protocol)
        stops = {"stop_flat": True, "stop_stepped": True, "stop_unchanged": True, **dict(cfg.get("stops") or {})}
        return _build_search(quest_root=root, block=protocol, base=int(seed), entry=str(cfg.get("entry") or ""),
                             thresholds=cfg.get("thresholds"), rounds=rounds, stops=stops)
    if kind in ("search_study", "search_study_bytes"):
        found = files(root, "simulation", by="tokens" if kind == "search_study" else "bytes", data=False)
        settings = {"block": protocol, "entry": str(cfg.get("entry") or ""), "thresholds": cfg.get("thresholds")}
        return Inputs(kind, RULES[kind], found.digests, settings, _fi_code(kind), found.complete, found.unreadable)
    raise ValueError(f"unknown kind of reusable work: {kind!r}")
