"""Cheap checks made on the code before it runs, as methods the engine mixes in (no model reads anything here).

Each check is a small module that reads the code with ``ast`` and the plan's protocol, and finds a fault a person would
find in a minute: a setting the code reads under a name the study does not have (:mod:`core.grid_keys`), numbers typed into
the model's functions that the plan never gives (:mod:`core.typed_constants`), a "second implementation" that compares a
value with itself (:mod:`core.independent_check`). A fault is sent back to be repaired, in plain words and with no expected
value in the request, at most :data:`REPAIRS` times for the text of the code; what is still there afterwards is handled the
way its module says (an honest stop, or a note the evidence and the paper carry). Nobody is asked to edit code.

They live here and not in ``engine.py`` so the node only calls :meth:`StaticChecksMixin._static_gate`. Every method uses
the engine's own attributes (``quest_root``, ``fi_dir``, ``_log``, ``_chat`` ...).
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any

from . import grid_keys as _grid_keys
from . import independent_check as _independent

#: How many times, in all, one text of the code is sent back for the checks of this module.
REPAIRS = 2
_COUNTER = "static_checks.json"
RECORD_DIR = "needs"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


class StaticChecksMixin:
    """Methods of :class:`core.engine.Engine`; see the module docstring."""

    #: ``(name, module, what a fault that stands after the repairs does)``: ``stop`` ends the quest without a paper,
    #: ``note`` is recorded for the evidence and the paper, ``headline`` stops only when the module says a headline
    #: result rests on it (and is a note otherwise).
    def _static_checks(self) -> list[tuple[str, Any, str]]:
        return [("grid_keys", _grid_keys, "stop"), ("independent_check", _independent, "note")]

    def _static_files(self) -> dict[str, str]:
        return _grid_keys.code_files(self.quest_root / "code")  # type: ignore[attr-defined]

    def _static_fingerprint(self, files: dict[str, str]) -> str:
        digest = hashlib.sha256()
        for name in sorted(files):
            digest.update(name.encode("utf-8"))
            digest.update(files[name].encode("utf-8"))
        return digest.hexdigest()

    def _static_counter(self, name: str, sha: str) -> int:
        record = _read_json(self.fi_dir / _COUNTER)  # type: ignore[attr-defined]
        entry = record.get(name) if isinstance(record.get(name), dict) else {}
        return int(entry.get("used") or 0) if entry.get("sha") == sha else 0

    def _save_static_counter(self, name: str, sha: str, used: int) -> None:
        record = _read_json(self.fi_dir / _COUNTER)  # type: ignore[attr-defined]
        record[name] = {"sha": sha, "used": used}
        try:
            self.fi_dir.mkdir(parents=True, exist_ok=True)  # type: ignore[attr-defined]
            (self.fi_dir / _COUNTER).write_text(json.dumps(record), encoding="utf-8")  # type: ignore[attr-defined]
        except OSError:
            pass

    def _static_inputs(self, state: Any) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        protocol = self._protocol_block(state)  # type: ignore[attr-defined]
        design = state.get("design") if isinstance(state.get("design"), dict) else None
        return protocol, design

    async def _static_gate(self, state: Any) -> dict[str, Any] | None:
        """Read the code for the faults of :meth:`_static_checks`, send each back to be repaired (bounded), and handle what
        stands. ``None`` to go on; otherwise the quest's honest stop (no paper). The same text of the code is read once."""
        files = self._static_files()
        sha = self._static_fingerprint(files)
        if not files or getattr(self, "_static_checked_sha", None) == sha:
            return self._cannot_repair_stop() if files else None  # type: ignore[attr-defined]
        protocol, design = self._static_inputs(state)
        stop: dict[str, Any] | None = None
        for name, module, on_standing in self._static_checks():
            try:
                found = module.problems(files, protocol, design)
            except Exception as exc:  # noqa: BLE001 -- a check that cannot be made flags nothing
                self._log.warning("[execute] the check %s could not read the code (%r)", name, exc)  # type: ignore[attr-defined]
                continue
            used = self._static_counter(name, sha)
            while found and used < REPAIRS:
                used += 1
                self._log.warning(  # type: ignore[attr-defined]
                    "[execute] %s; asking for it to be repaired (%d of %d)", module.plain_note(found, protocol), used, REPAIRS)
                await self._repair_static(state, module, found, protocol, design)
                files = self._static_files()
                sha = self._static_fingerprint(files)
                self._save_static_counter(name, sha, used)
                found = module.problems(files, protocol, design)
            self._write_static_record(name, module, found, protocol)
            if found and stop is None:
                stop = self._static_standing(name, module, on_standing, found, files, protocol, design)
        self._static_checked_sha = sha
        return stop

    def _write_static_record(self, name: str, module: Any, found: list[Any], protocol: Any) -> None:
        """The record of what stood after the repairs (``needs/<name>_check.json``); an earlier pass's goes first."""
        path = self.quest_root / RECORD_DIR / f"{name}_check.json"  # type: ignore[attr-defined]
        path.unlink(missing_ok=True)
        if not found:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            say = module.writer_note(found, protocol) if hasattr(module, "writer_note") else ""
            path.write_text(json.dumps({**module.record(found, protocol), "note": module.plain_note(found, protocol),
                                        **({"say": say} if say else {})}, indent=1), encoding="utf-8")
        except OSError as exc:
            self._log.warning("[execute] could not write needs/%s_check.json (%r)", name, exc)  # type: ignore[attr-defined]

    def _static_standing(self, name: str, module: Any, on_standing: str, found: list[Any], files: dict[str, str],
                         protocol: Any, design: Any) -> dict[str, Any] | None:
        """What a fault that stands after the repairs does: the honest stop, or (a note is already written) going on."""
        stops = on_standing == "stop"
        if on_standing == "headline":
            hit = getattr(module, "headline_hit", None)
            stops = bool(hit and hit(found, files, protocol, design))
        text = module.plain_note(found, protocol)
        if not stops:
            self._log.warning("[execute] %s; it stands after the repairs, so the paper says so", text)  # type: ignore[attr-defined]
            return None
        self._record_cannot_repair(  # type: ignore[attr-defined]
            text, [f"read the code against the plan before running it ({name})", *[f.says([]) if hasattr(f, "says") else str(f)
                                                                                  for f in found[:2]],
                   f"sent the code back to be repaired {REPAIRS} times; it still had this"])
        return self._cannot_repair_stop()  # type: ignore[attr-defined]

    async def _repair_static(self, state: Any, module: Any, found: list[Any], protocol: Any, design: Any) -> bool:
        """ONE repair of the file the first fault is in: kept only if it parses, keeps what FI reads from it, and has fewer
        faults of this kind in that file than before."""
        from . import engine as _e
        from . import typed_results as _typed

        script = found[0].script
        path = self.quest_root / "code" / script  # type: ignore[attr-defined]
        try:
            code = path.read_text(encoding="utf-8")
        except OSError:
            return False
        here = [f for f in found if f.script == script]
        prompt = self._prompts["execute_reflect"].substitute(  # type: ignore[attr-defined]
            previous_code=code, returncode="(not run yet)", stdout_tail=module.directive(here, protocol), stderr_tail="",
            duration_s="0.00", figures_count="0", result_json_present="no (not run yet)",
            reflect_history_block=_e._format_reflect_history([]),
            design_block=self._code_design_block(state), clarify_block=_e._format_clarify(state),  # type: ignore[attr-defined]
        )
        try:
            text = await self._chat(prompt, node="execute_reflect")  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001 -- a repair is best-effort
            self._log.warning("[execute] the call to repair %s failed (%r); keeping it as written", script, exc)  # type: ignore[attr-defined]
            return False
        parsed: dict[str, Any] = {}
        if _e._strip_outer_fence(text).lstrip().startswith("{"):
            parsed = _e._parse_json_lenient(text, node="execute_reflect") or {}
        new_code = parsed.get("code")
        if not (isinstance(new_code, str) and new_code.strip()):
            new_code, _deps = _e._parse_implement_response(text)
        try:
            ast.parse(new_code)
            usable = bool(new_code.strip()) and all(mark in new_code for mark in _typed.KEEP_MARKS if mark in code)
        except (SyntaxError, ValueError):
            usable = False
        if not usable:
            self._log.warning(  # type: ignore[attr-defined]
                "[execute] the rewrite of %s is not the same code with the fault repaired; keeping it as written", script)
            return False
        files = self._static_files()
        before = len([f for f in module.problems(files, protocol, design) if f.script == script])
        path.write_text(new_code, encoding="utf-8")
        after = len([f for f in module.problems(self._static_files(), protocol, design) if f.script == script])
        if after >= before:
            path.write_text(code, encoding="utf-8")
            self._log.warning(  # type: ignore[attr-defined]
                "[execute] the rewritten %s still has %d fault(s) of this kind; keeping it as written", script, after)
            return False
        self._log.info(  # type: ignore[attr-defined]
            "[execute] rewrote %s to repair the fault (%d left in it): %s", script, after,
            str(parsed.get("patch_summary") or "no summary")[:120])
        return True

    # ---- what stood, for the paper and the evidence -------------------------------------------------------------------

    def _static_check_records(self) -> dict[str, dict[str, Any]]:
        """``{check name: its record}`` for every check of this module that has a fault standing this pass."""
        out: dict[str, dict[str, Any]] = {}
        for name, _module, _on in self._static_checks():
            record = _read_json(self.quest_root / RECORD_DIR / f"{name}_check.json")  # type: ignore[attr-defined]
            if record:
                out[name] = record
        return out

    def _static_check_gaps(self) -> list[str]:
        """Why the evidence ladder cannot count a check this pass found to prove nothing (one sentence each)."""
        return [str(r["gap"]) for r in self._static_check_records().values() if r.get("gap")]

    def _static_checks_note(self) -> str:
        """For the writer: what the checks of the code found that the repairs did not remove, so the limitations say so."""
        parts = []
        for record in self._static_check_records().values():
            note = record.get("note")
            if isinstance(note, str) and note.strip() and record.get("say"):
                parts.append(str(record["say"]))
        return "\n\n".join(parts)
