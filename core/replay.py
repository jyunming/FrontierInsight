"""Answer a quest's model calls from a recording, and record what was answered.

A recording is the list of a run's model calls in the order each step made them: ``(node, index)`` names a call (the
``index``-th call that step made in this run, from 1) and holds the answer, who answered it (provider, model, whether the
connection named the model) and its token counts. A prompt is not part of the key: it holds paths, times and retrieved
text that differ between two runs of the same quest, so a prompt hash would never match.

:class:`ReplayClient` stands where the engine's model client stands (``Engine._client``; it has the same ``chat``) and
serves each call one of four ways:

* ``planted``: a replacement answer given for that call (a benchmark's planted error), whatever the mode;
* ``replayed``: the recorded answer, with who answered it set as the recording says (``LAST_CALL``), so a replayed
  review panel still shows reviewers on different models;
* ``real``: passed on to the real client (partial replay: only the calls before the planted one are replayed);
* ``unrecorded``: a call the recording does not have, in a run that must not reach a model. It gets the fixed
  :data:`CANNOT_FIX` answer, and a ``divergence`` event names the step that asked: that a run asked for a call its
  recording never made is itself a finding (a repair after a check failed, a loop that went round once more).

Every call, whichever way it was served, is written to ``calls.jsonl`` in the folder the caller names (never the
quest's own ``.fi/``, whose records are sealed): that file is the recording a later full replay reads.

:class:`CrossrefReplay` does the same for the retraction lookups (``core/retractions.py::check_dois``): it records
Crossref's answer per DOI, and in replay a DOI with no recorded answer is "not checked" plus a divergence event, never
a fresh lookup. Nothing here calls a model or the network on its own.
"""

from __future__ import annotations

import contextvars
import json
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from . import attempt_records as _attempts
from . import provider as _provider

CALLS_FILE = "calls.jsonl"
EVENTS_FILE = "replay_events.jsonl"
CROSSREF_FILE = "crossref.json"

#: The answer a call the recording does not have gets: a step that parses JSON reads it as "no change", and a step
#: that writes prose stops, which is the point (the run cannot go on as it was recorded).
CANNOT_FIX = json.dumps({
    "cannot_fix": True,
    "reason": "This call was not recorded: the benchmark's replay has no answer for it, so nothing is changed.",
})

#: How a run is served: ``record`` (every call real), ``partial`` (replay until the first planted call, real after it;
#: with no planted call, every call is real), ``replay`` (every call from the recording; nothing reaches a model).
MODES = ("record", "partial", "replay")


def _key(node: str, index: int) -> str:
    return f"{node}#{int(index)}"


class Recording:
    """The calls of one recorded run, by ``(node, index)``."""

    def __init__(self, calls: list[dict[str, Any]] | None = None) -> None:
        self.calls: dict[str, dict[str, Any]] = {}
        self.errors: dict[str, dict[str, Any]] = {}
        for c in calls or []:
            if not isinstance(c, dict) or c.get("node") is None:
                continue
            key = _key(str(c["node"]), int(c.get("index") or 0))
            if isinstance(c.get("response"), str):
                self.calls.setdefault(key, c)
            elif c.get("source") == "error":
                self.errors.setdefault(key, c)

    def get(self, node: str, index: int) -> dict[str, Any] | None:
        return self.calls.get(_key(node, index))

    def failed(self, node: str, index: int) -> str | None:
        """Why the recorded call failed, when it did (and nothing answered it)."""
        row = self.failure(node, index)
        return str(row.get("error") or "an error") if row is not None else None

    def failure(self, node: str, index: int) -> dict[str, Any] | None:
        key = _key(node, index)
        return None if key in self.calls else self.errors.get(key)

    def after(self, ts: float) -> "Recording":
        """The calls started after ``ts`` (each row's start time), numbered again per step from 1, in the order they
        started: what a run resumed from a checkpoint taken at ``ts`` asks for."""
        # A step's first call can start in the same clock tick as the checkpoint (>=); calls started in one tick keep
        # their recorded order within the step (the index), never the order they finished.
        rows = sorted((c for c in [*self.calls.values(), *self.errors.values()] if float(c.get("ts") or 0) >= ts),
                      key=lambda c: (float(c.get("ts") or 0), int(c.get("index") or 0)))
        counts: dict[str, int] = {}
        out = []
        for c in rows:
            node = str(c["node"])
            counts[node] = counts.get(node, 0) + 1
            out.append({**c, "index": counts[node]})
        return Recording(out)

    def __len__(self) -> int:
        return len(self.calls)

    @classmethod
    def load(cls, path: Path) -> "Recording":
        """A recording written by :class:`ReplayClient` (``calls.jsonl``)."""
        rows = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
        return cls(rows)

    @classmethod
    def from_quest(cls, quest_root: Path, *, after: float | None = None) -> "Recording":
        """The calls a quest kept itself (``output.save_model_calls``: one file per call in ``.fi/io/``), numbered per
        step in the order their answers came back (calls one step makes at the same time can come back in another order
        than they were asked; a benchmark run's own ``calls.jsonl`` keeps the order they were asked, and is preferred),
        with who answered each from the quest's record of its calls
        (``.fi/model_calls.jsonl``, joined on the answer's hash). ``after``: only the calls made after this time (the
        calls of a run resumed from a step are the ones made after the checkpoint it resumed from)."""
        fi_dir = Path(quest_root) / ".fi"
        served: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in _attempts.read(fi_dir, _attempts.MODEL_CALLS):
            if row.get("outcome") == "ok" and row.get("response_sha256"):
                served.setdefault((str(row.get("node") or ""), str(row["response_sha256"])), []).append(row)
        files = sorted((fi_dir / _provider.IO_DIRNAME).glob("*.json")) if (fi_dir / _provider.IO_DIRNAME).is_dir() else []
        counts: dict[str, int] = {}
        calls: list[dict[str, Any]] = []
        for path in files:  # named <time_ns>-...: sorted by name is the order the answers came back
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            response = data.get("response")
            if not isinstance(response, str):
                continue  # a failed attempt kept for its cost: no answer
            if after is not None and float(data.get("ts") or 0) <= after:
                continue
            node = str(data.get("node") or "")
            counts[node] = counts.get(node, 0) + 1
            rows = served.get((node, _attempts._text_sha(response) or ""), [])
            who = rows.pop(0) if rows else {}
            calls.append({"node": node, "index": counts[node], "response": response,
                          "provider": who.get("provider"), "model": who.get("served_model") or data.get("model"),
                          "reported": bool(who.get("reported")), "usage": data.get("usage")})
        return cls(calls)


class ReplayClient:
    """A model client for ``Engine._client`` that answers from a :class:`Recording` (see the module docstring).

    ``real``: the client the engine would have used (``None`` in ``replay``, where nothing may reach a model).
    ``plants``: ``{(node, index): answer}`` served in place of that call. ``out_dir``: where ``calls.jsonl`` and the
    events are written."""

    def __init__(self, *, mode: str, recording: Recording | None = None, real: Any = None,
                 plants: dict[tuple[str, int], str] | None = None, out_dir: Path) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {', '.join(MODES)}, not {mode!r}")
        if mode != "replay" and real is None:
            raise ValueError(f"mode {mode!r} passes calls on to a model, so it needs the real client")
        self.mode = mode
        self.recording = recording if recording is not None else Recording()  # (an empty one is falsy: it has a len)
        self.real = real
        self.plants = {(str(n), int(i)): str(a) for (n, i), a in (plants or {}).items()}
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._counts: dict[str, int] = {}
        # Partial replay replays until the first planted call is served; with nothing planted it is a real run.
        self._replaying = mode == "replay" or (mode == "partial" and bool(self.plants))
        self.last_provider: str | None = None
        self.last_model: str | None = None
        self.last_usage: dict[str, Any] | None = None
        self.divergences = 0

    # The engine reads these on a client (core/engine.py): who answered, and the token counts.
    def _served(self, provider: Any, model: Any, reported: bool, usage: Any) -> None:
        self.last_provider = str(provider) if provider else None
        self.last_model = str(model) if model else None
        self.last_usage = usage if isinstance(usage, dict) else None
        _provider.LAST_CALL.set({"provider": self.last_provider, "model": self.last_model, "reported": bool(reported),
                                 "usage": self.last_usage})

    def _write(self, name: str, row: dict[str, Any]) -> None:
        try:
            with (self.out_dir / name).open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        except OSError:
            pass

    def event(self, kind: str, **fields: Any) -> None:
        self._write(EVENTS_FILE, {"ts": time.time(), "event": kind, **fields})

    def _write_failure(self, node: str, index: int, started: float, messages: Any, e: BaseException) -> None:
        self._write(CALLS_FILE, {"node": node, "index": index, "ts": started, "source": "error",
                                 "error": f"{type(e).__name__}: {e}"[:500],
                                 "error_type": f"{type(e).__module__}.{type(e).__qualname__}", "response": None,
                                 "prompt_sha256": _attempts.prompt_sha(messages)})

    async def _real(self, messages: Any, node: str, index: int, started: float, **kw: Any) -> str:
        try:
            response = await self.real.chat(messages, node=node, **kw)
        except Exception as e:
            # A failed call is part of the run: a replay of it fails the same way, never answers it.
            self._write_failure(node, index, started, messages, e)
            raise
        served = dict(_provider.LAST_CALL.get() or {})
        usage = served.get("usage") if isinstance(served.get("usage"), dict) else getattr(self.real, "last_usage", None)
        if served.get("provider") or served.get("model"):
            self.last_provider, self.last_model, self.last_usage = served.get("provider"), served.get("model"), usage
        else:  # a transport that names nobody: the client's own attributes, and not "reported"
            self._served(getattr(self.real, "last_provider", None), getattr(self.real, "last_model", None), False,
                         usage)
        return response

    async def chat(self, messages: list[dict[str, str]], *, temperature: float = 0.2, max_tokens: int | None = None,
                   extra: dict[str, Any] | None = None, model: str | None = None, node: str = "") -> str:
        node = node or ""
        # Numbered when the call starts, and the start time kept: a recording is replayed in the order calls were asked.
        index = self._counts[node] = self._counts.get(node, 0) + 1
        started = time.time()
        kw = {"temperature": temperature, "max_tokens": max_tokens, "extra": extra, "model": model}
        recorded = self.recording.get(node, index)
        failed = self.recording.failed(node, index)
        planted = self.plants.get((node, index))
        if planted is not None:
            source = "planted"
            response = planted
            who = recorded or {}
            # Who "answered" a planted answer is the recorded call's model; with no recorded call, nobody is named.
            self._served(who.get("provider") or "replay", who.get("model") or model or "replay",
                         bool(who.get("reported", False)), who.get("usage"))
            self._replaying = self.mode == "replay"  # partial replay: every call after the planted one is real
            self.event("planted", node=node, index=index)
        elif self._replaying and failed is not None:
            self.event("replayed_failure", node=node, index=index, error=failed)
            error = _rebuilt(self.recording.failure(node, index) or {}, failed)
            self._write_failure(node, index, started, messages, error)  # the replay's own recording has it too
            raise error
        elif self._replaying and recorded is not None:
            source = "replayed"
            response = str(recorded["response"])
            self._served(recorded.get("provider"), recorded.get("model"), bool(recorded.get("reported")),
                         recorded.get("usage"))
        elif self._replaying and self.mode == "partial":
            # Before the planted call, a call the recording does not have may still go to the model: it does, and says so.
            source = "real"
            self.divergences += 1
            self.event("divergence", node=node, index=index,
                       why="before the planted call, the run asked for a call its recording does not have; the model answered it")
            response = await self._real(messages, node, index, started, **kw)
        elif self._replaying:
            source = "unrecorded"
            response = CANNOT_FIX
            self.divergences += 1
            # Not "reported": nothing answered this call, and the record of who answered must not say otherwise.
            self._served("replay", "no recorded answer", False, None)
            self.event("divergence", node=node, index=index,
                       why="the run asked for a call its recording does not have; it got the fixed cannot-fix answer")
        else:
            source = "real"
            response = await self._real(messages, node, index, started, **kw)
        who = dict(_provider.LAST_CALL.get() or {})
        self._write(CALLS_FILE, {"node": node, "index": index, "ts": started, "source": source, "response": response,
                                 "provider": who.get("provider"), "model": who.get("model"),
                                 "reported": bool(who.get("reported")), "usage": who.get("usage"),
                                 "prompt_sha256": _attempts.prompt_sha(messages)})
        return response

    async def aclose(self) -> None:
        if self.real is not None and hasattr(self.real, "aclose"):
            await self.real.aclose()

    def __getattr__(self, name: str) -> Any:
        # Anything else the engine asks of its client (the fallback providers it releases at the end) is the real one's.
        real = self.__dict__.get("real")
        if real is None or name.startswith("__"):
            raise AttributeError(name)
        return getattr(real, name)


class CrossrefReplay:
    """Record Crossref's retraction answers, or answer from a recording (``core/retractions.py::check_dois``).

    ``record``: the real lookup runs and every DOI's answer is kept in ``crossref.json``. ``replay``: each DOI gets its
    recorded answer; one with none is "not checked" (never "not retracted") and a divergence event. Use
    :meth:`installed` around a run: it replaces the module's ``check_dois`` for that run only."""

    def __init__(self, *, mode: str, out_dir: Path, recorded: dict[str, Any] | None = None) -> None:
        if mode not in ("record", "replay"):
            raise ValueError("the Crossref lookups are either recorded or replayed")
        self.mode = mode
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.answers: dict[str, Any] = dict(recorded or {})

    @classmethod
    def load(cls, path: Path, *, out_dir: Path) -> "CrossrefReplay":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(mode="replay", out_dir=out_dir, recorded=dict(data.get("answers") or {}))

    def _save(self) -> None:
        (self.out_dir / CROSSREF_FILE).write_text(
            json.dumps({"answers": self.answers}, indent=1, ensure_ascii=False), encoding="utf-8")

    async def _lookup(self, real: Any, dois: list[str], **kw: Any) -> dict[str, dict[str, Any]]:
        from . import retractions as _retractions

        if self.mode == "record":
            out = await real(dois, **kw)
            self.answers.update(out)
            self._save()
            return out
        out = {}
        for doi in (d for d in (_retractions.normalize_doi(x) for x in dois) if d):
            if doi in self.answers:
                out[doi] = self.answers[doi]
            else:
                out[doi] = {"status": _retractions.NOT_CHECKED, "why": "no recorded Crossref answer", "notices": []}
                with (self.out_dir / EVENTS_FILE).open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"ts": time.time(), "event": "divergence", "node": "retractions",
                                         "doi": doi, "why": "a DOI the recording has no Crossref answer for"}) + "\n")
        return out

    @contextmanager
    def installed(self) -> Iterator["CrossrefReplay"]:
        """This replay answers the retraction lookups made in the current task (and the tasks it starts) while the
        block runs. Several runs in one process each get their own: one dispatcher stands in for ``check_dois`` while
        any is installed (counted), finds the current run's replay by context, and puts the real lookup back when the
        last one leaves; a lookup made outside every run goes to the real one."""
        from . import retractions as _retractions

        token = _CURRENT_CROSSREF.set(self)
        with _DISPATCH_LOCK:
            if _DISPATCH["count"] == 0:
                _DISPATCH["real"] = _retractions.check_dois
                _retractions.check_dois = _dispatch
            _DISPATCH["count"] += 1
        try:
            yield self
        finally:
            with _DISPATCH_LOCK:
                _DISPATCH["count"] -= 1
                if _DISPATCH["count"] == 0:
                    _retractions.check_dois = _DISPATCH["real"]
                    _DISPATCH["real"] = None
            _CURRENT_CROSSREF.reset(token)


# The one dispatcher that stands in for core.retractions.check_dois while any CrossrefReplay is installed; like the
# proxy supervisor, it is shared on purpose and counted (FI's rule: no process state that two runs could collide on).
_CURRENT_CROSSREF: contextvars.ContextVar[CrossrefReplay | None] = contextvars.ContextVar("fi_crossref_replay",
                                                                                         default=None)
_DISPATCH: dict[str, Any] = {"count": 0, "real": None}
_DISPATCH_LOCK = threading.Lock()


async def _dispatch(dois: list[str], **kw: Any) -> dict[str, dict[str, Any]]:
    real = _DISPATCH["real"]
    current = _CURRENT_CROSSREF.get()
    if current is None:
        return await real(dois, **kw)
    return await current._lookup(real, dois, **kw)


def _rebuilt(row: dict[str, Any], why: str) -> Exception:
    """The kind of error the recorded call raised (the engine treats some kinds in their own way), with its message;
    a ``RuntimeError`` when that kind cannot be made again from a message."""
    import importlib

    message = f"[replay] this call failed in the recorded run: {why}"
    module, _, name = str(row.get("error_type") or "").rpartition(".")
    try:
        cls = getattr(importlib.import_module(module), name) if module else None
        if isinstance(cls, type) and issubclass(cls, Exception):
            return cls(message)
    except Exception:  # noqa: BLE001 -- an error class that will not rebuild is reported as a RuntimeError
        pass
    return RuntimeError(message)


def read_events(out_dir: Path) -> list[dict[str, Any]]:
    """The planted / divergence events a replayed run wrote."""
    path = Path(out_dir) / EVENTS_FILE
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out
