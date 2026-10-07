"""A check made on the simulation before it runs, as methods the engine mixes in.

* **A reported result is computed** (``core/typed_results.py``): the simulation is read, never run, and a returned value that
  is a number typed into the code is sent back to be computed (or removed), at most twice; one that is still typed in
  afterwards is left out of the results the paper may use, and the paper says so.

It lives here and not in ``engine.py`` so the node only calls it. Every method uses the engine's own attributes
(``quest_root``, ``fi_dir``, ``config``, ``_log``, ``_chat`` ...).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from . import oracle_forms as _forms
from . import split_run as _split_run
from . import typed_results as _typed

#: How many times, in all, a simulation that returns typed-in numbers is sent back.
TYPED_RESULT_REPAIRS = 2
TYPED_RECORD = "TYPED_RESULTS_CHECK.json"
_COUNTER = "typed_results.json"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


class RunChecksMixin:
    """Methods of :class:`core.engine.Engine`; see the module docstring."""

    # ---- a reported result is computed --------------------------------------------------------------------------------

    def _typed_result_exempt(self, state: Any) -> set[str]:
        """Names the plan fixes (its fixed settings, the grid's settings, the thresholds): echoed back, not results."""
        protocol = self._protocol_block(state) or {}  # type: ignore[attr-defined]
        design = state.get("design") if isinstance(state.get("design"), dict) else {}
        names = set(_forms.fixed_settings_of_design({**design, "protocol": protocol}))
        grid = protocol.get("grid") if isinstance(protocol.get("grid"), dict) else {}
        names |= {str(k) for k in grid}
        thresholds = protocol.get("thresholds") if isinstance(protocol.get("thresholds"), dict) else {}
        return names | {str(k) for k in thresholds}

    def _typed_results_in_simulation(self, state: Any) -> tuple[Path, list[_typed.TypedResult]]:
        path = self.quest_root / "code" / _split_run.SIMULATE_NAME  # type: ignore[attr-defined]
        if not path.is_file():
            return path, []
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return path, []
        return path, _typed.typed_results(text, path.name, exempt=self._typed_result_exempt(state))

    def _typed_counter(self, text: str) -> int:
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        record = _read_json(self.fi_dir / _COUNTER)  # type: ignore[attr-defined]
        return int(record.get("used") or 0) if record.get("sha") == sha else 0

    def _save_typed_counter(self, text: str, used: int) -> None:
        try:
            self.fi_dir.mkdir(parents=True, exist_ok=True)  # type: ignore[attr-defined]
            (self.fi_dir / _COUNTER).write_text(  # type: ignore[attr-defined]
                json.dumps({"sha": hashlib.sha256(text.encode("utf-8")).hexdigest(), "used": used}), encoding="utf-8")
        except OSError:
            pass

    async def _results_from_computation(self, state: Any) -> bool:
        """Send a simulation that returns numbers typed into its code back to compute them, then leave out what is still
        typed in. ``True`` when the simulation was rewritten. The left-out quantities are this pass's alone: the record of
        an earlier pass goes first."""
        record_path = self.quest_root / "needs" / TYPED_RECORD  # type: ignore[attr-defined]
        record_path.unlink(missing_ok=True)
        self._typed_result_keys: set[str] = set()
        path, found = self._typed_results_in_simulation(state)
        if not found:
            return False
        rewrote = False
        text = path.read_text(encoding="utf-8")
        used = self._typed_counter(text)
        while found and used < TYPED_RESULT_REPAIRS:
            used += 1
            self._log.warning(  # type: ignore[attr-defined]
                "[execute] %s returns numbers typed into the code (%s); asking for them to be computed (%d of %d)",
                path.name, "; ".join(f.says() for f in found[:4]), used, TYPED_RESULT_REPAIRS)
            rewrote = await self._repair_typed_results(state, path, found) or rewrote
            text = path.read_text(encoding="utf-8")
            self._save_typed_counter(text, used)
            path, found = self._typed_results_in_simulation(state)
        if not found:
            return rewrote
        keys = sorted({f.key for f in found})
        self._typed_result_keys = set(keys)
        note = _typed.plain_note(keys, found)
        try:
            record_path.parent.mkdir(parents=True, exist_ok=True)
            record_path.write_text(json.dumps(_typed.record(keys, found), indent=1), encoding="utf-8")
        except OSError as exc:
            self._log.warning("[execute] could not write needs/%s (%r)", TYPED_RECORD, exc)  # type: ignore[attr-defined]
        self._log.warning(  # type: ignore[attr-defined]
            "[execute] %s The paper does not use %s and says so in its limitations.", note,
            "it" if len(keys) == 1 else "them")
        return rewrote

    async def _repair_typed_results(self, state: Any, path: Path, found: list[_typed.TypedResult]) -> bool:
        """ONE repair of the simulation: kept only if it parses, keeps what FI reads from it, and has fewer typed-in
        results than before."""
        import ast

        from . import engine as _e

        code = path.read_text(encoding="utf-8")
        prompt = self._prompts["execute_reflect"].substitute(  # type: ignore[attr-defined]
            previous_code=code, returncode="(not run yet)", stdout_tail=_typed.directive(found), stderr_tail="",
            duration_s="0.00", figures_count="0", result_json_present="no (not run yet)",
            reflect_history_block=_e._format_reflect_history([]),
            design_block=json.dumps(state.get("design") or {}, indent=2), clarify_block=_e._format_clarify(state),
        )
        try:
            text = await self._chat(prompt, node="execute_reflect")  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001 -- a repair is best-effort
            self._log.warning(  # type: ignore[attr-defined]
                "[execute] the call to compute the typed-in results failed (%r); keeping %s as written", exc, path.name)
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
                "[execute] the rewrite of %s is not the same simulation with its results computed; keeping it as written",
                path.name)
            return False
        left = _typed.typed_results(new_code, path.name, exempt=self._typed_result_exempt(state))
        if len(left) >= len(found):
            self._log.warning(  # type: ignore[attr-defined]
                "[execute] the rewritten %s still returns %d number(s) typed into the code; keeping it as written",
                path.name, len(left))
            return False
        path.write_text(new_code, encoding="utf-8")
        self._log.info(  # type: ignore[attr-defined]
            "[execute] rewrote %s so its results are computed (%d typed-in left): %s", path.name, len(left),
            str(parsed.get("patch_summary") or "no summary")[:120])
        return True

    def _left_out_quantities(self) -> set[str]:
        """The quantities this pass's analysis and results must not use (still typed into the code after the repairs)."""
        return set(getattr(self, "_typed_result_keys", None) or ())

    def _without_typed_results(self, result_json: Any) -> Any:
        """``result_json`` without the left-out quantities."""
        keys = self._left_out_quantities()
        return _typed.strip_keys(result_json, keys) if keys and result_json else result_json

    # ---- the note for the writer ---------------------------------------------------------------------------------------

    def _run_checks_note(self) -> str:
        """For the writer: quantities left out because they were typed into the code,, so the limitations say so."""
        parts: list[str] = []
        record = _read_json(self.quest_root / "needs" / TYPED_RECORD)  # type: ignore[attr-defined]
        note = record.get("note")
        if isinstance(note, str) and note.strip():
            names = ", ".join(f"`{k}`" for k in record.get("removed_quantities") or [])
            parts.append(f"{note} Do not report, describe or quote any value for {names or 'them'}: say plainly, in the "
                         "limitations, that this quantity was not computed by the simulation and is left out.")
        return "\n\n".join(parts)
