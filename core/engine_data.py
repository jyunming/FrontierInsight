"""Methods of :class:`core.engine.Engine` for data that is collected instead of simulated.

``Engine`` inherits :class:`DataStageMixin`; every method here uses the engine's own attributes and its other methods
through ``self``. The shared names come from ``core/engine_base.py``.
"""

from __future__ import annotations

from .engine_base import (  # noqa: F401
    Any, Path, QuestState, RetrievedDoc, WORK_SCOPE_PAPERS, WORK_SCOPE_PAPERS_AND_BOOKS,
    _FI_INTERNAL_KINDS, _audit_open, _figures_section, 
    _is_preliminary_memory, _item_content, _list_user_data_files, _normalize_title,
    _parse_json_lenient, _phased, _render_auto_collected_md, _render_data_readme,
    _save_literature_figures, _slugify, _source_text, _split_figure_readings,
    _strip_plan_numbers, asyncio, copy, functools, json,
)


class DataStageMixin:
    """Methods of :class:`core.engine.Engine`; see the module docstring."""

    def _write_literature_files(self, lit: list, out_dir: Path) -> int:
        """Write each citable literature entry to ``out_dir`` as a
        ``lit_NNN_slug.md`` file (YAML front matter + content, via
        ``_render_auto_collected_md``). Called after the literature step to put the
        sources on disk under ``data/literature/`` (every quest, sim or no-sim): the
        single on-disk copy of the sources, which ``data_load`` reuses on the
        no-simulation path.

        Skips entries with no title/url and FI-internal cross-quest
        artifacts. ``out_dir`` is created lazily so a corpus with nothing
        citable leaves no empty directory behind. Returns the count
        written."""
        written = 0
        for idx, item in enumerate(lit, start=1):
            if isinstance(item, dict):
                meta = item.get("metadata") or {}
            else:
                meta = getattr(item, "metadata", {}) or {}
            content = _item_content(item)  # the whole text, not only the state's first part
            if not (meta.get("title") or meta.get("url")):
                continue
            if meta.get("kind") in _FI_INTERNAL_KINDS or _is_preliminary_memory(meta):
                continue
            slug = _slugify(
                str(meta.get("title") or meta.get("source") or f"lit{idx}")
            )[:40] or f"lit{idx}"
            target = out_dir / f"lit_{idx:03d}_{slug}.md"
            figures = _save_literature_figures(meta.get("figures"), out_dir / "figures", f"lit_{idx:03d}")
            body = _render_auto_collected_md(idx, {k: v for k, v in meta.items() if k != "figures"}, content)
            if figures:
                body = body.rstrip("\n") + "\n\n" + _figures_section(figures) + "\n"
            try:
                out_dir.mkdir(parents=True, exist_ok=True)
                target.write_text(body, encoding="utf-8")
                written += 1
            except OSError as e:
                self._log.warning(
                    "[literature] failed to write %s: %s — skipping",
                    target, e,
                )
        return written

    async def _node_wait_for_data(self, state: QuestState) -> QuestState:
        """no-simulation pause point. Creates ``<quest_root>/data/``,
        writes a README explaining what to drop into it, then either:

        * If the dir is empty (apart from the README we just wrote) —
          fire ``interrupt(...)`` so ``Engine.run`` can exit cleanly
          with rc=0 and tell the user to drop files and re-run.

        * If files are already present — return state with
          ``data_files`` populated, letting the graph proceed to
          ``data_load`` → ``analyze``.

        On resume after the user dropped files, LangGraph re-enters
        this node from the checkpoint. The dir now has files, so we
        proceed without pausing.
        """
        data_dir = self.quest_root / "data"
        data_dir.mkdir(exist_ok=True)
        readme = data_dir / "README.md"
        if not readme.exists():
            readme.write_text(
                _render_data_readme(state, self.quest_id), encoding="utf-8",
            )
        user_files = _list_user_data_files(data_dir)
        if not user_files:
            # Pause via LangGraph's interrupt mechanism. Engine.run
            # catches this and exits rc=0 cleanly. On resume the
            # interrupt re-fires; if the user has dropped files by
            # then, the resume payload carries them. (We don't trust
            # the payload — we re-walk the dir on every resume.)
            self._pause_for_human(
                kind="data",
                interaction="supply",
                headline="add a dataset to analyse",
                steps=[
                    "This observational quest has no data yet and runs no "
                    "simulation.",
                    "Drop your dataset (CSV / JSON / TSV / Parquet) into "
                    "`data/` — see `data/README.md` for the specifics it expects.",
                ],
                payload={
                    "data_required": True,
                    "quest_id": self.quest_id,
                    "data_dir": str(data_dir),
                },
                upload_targets=["root_data"],
            )
            # Unreachable in practice: interrupt() raises GraphInterrupt
            # which Engine.run catches above; on resume, the node is
            # re-invoked from the checkpoint and the early-return below
            # fires because user_files is now non-empty.
            return {}
        self._log.info(
            "[wait_for_data] found %d user file(s) under %s — proceeding "
            "to data_load", len(user_files), data_dir,
        )
        out: QuestState = {"data_files": [str(p) for p in user_files]}
        asked = [str(p).strip() for p in state.get("refine_extend") or [] if str(p).strip()]
        if asked:
            # A person's note asked for a number the study lacks. A data study analyses the person's own files and FI
            # looks for no other data, so the ask is not carried out: the writer is told, and says so in the paper.
            self._log.info("[wait_for_data] %d point(s) a note asked to add need data FI does not look for; they are "
                           "reported as not added", len(asked))
            out["refine_extend"] = []
            out["extend_missed"] = asked
        return out

    async def _node_data_load(self, state: QuestState) -> QuestState:
        """no-simulation mode: synthesize a ``result_json`` from the
        files the user dropped into ``<quest_root>/data/``. Reuses
        the ``core/summarizer.py`` patterns (file walk + classify +
        content-budget-aware prompt assembly) so we don't duplicate
        the mixed-format walker.

        The LLM call produces a JSON object with the same shape an
        analyze-node prompt expects: a top-level dict of findings
        + supporting evidence cited back to the user's source files.
        """
        from .summarizer import (
            _walk_folder, _render_file_manifest, _render_content_blocks,
        )

        if _phased.enabled(self.config):
            # Explore, then confirm: the rows held back are taken out (or, in the confirm stage, are the only ones in
            # place) before anything below reads the data (off the event loop: the files may be large).
            question = await asyncio.to_thread(self._phased_before_data_read, state)
            if question:
                self._phased_ask_split(question)
        data_dir = self.quest_root / "data"
        # Re-walk the dir on every invocation rather than trusting
        # ``state["data_files"]`` — the user may have edited / added
        # files between pause and resume.
        #
        # Filter out the FI-authored README.md at the top of the data
        # dir before rendering the prompt. ``_walk_folder`` doesn't
        # know it's there as instructions, not data; without this
        # filter the README's "drop your data here" text + the
        # hypothesis + the resume command would land in the prompt
        # and the LLM might cite it as a "primary source" in
        # key_findings. ``_list_user_data_files`` already excludes
        # the README (used by _node_wait_for_data's pause check);
        # data_load now matches that contract by re-numbering idents
        # after the filter so the manifest IDs stay sequential.
        all_entries = _walk_folder(data_dir)
        entries = [
            e for e in all_entries
            if not (e.rel_path == "README.md" and (data_dir / "README.md").is_file())
        ]
        # Re-number ident so the prompt sees [1], [2], … in order;
        # FileEntry is a dataclass so we mutate the ident field.
        for new_id, entry in enumerate(entries, start=1):
            entry.ident = new_id
        if not entries:
            self._log.warning(
                "[data_load] %s has no user data (only README.md) — "
                "analyze will run with an empty result_json", data_dir,
            )
            if _phased.enabled(self.config):
                _phased.note_confirm_result(self.quest_root, {})  # a confirm reading that found nothing still happened
            return {"result_json": {}, "data_files": []}

        manifest = _render_file_manifest(entries)
        content_blocks = _render_content_blocks(entries)
        prompt = self._prompts["data_load"].substitute(
            topic=state["topic"],
            design_block=json.dumps(state.get("design") or {}, indent=2),
            file_manifest=manifest,
            content_blocks=content_blocks,
        )
        text = await self._chat(prompt, node="data_load")
        result_json = _parse_json_lenient(text, node="data_load") or {}
        self._log.info(
            "[data_load] synthesized result_json with %d keys from %d files",
            len(result_json), len(entries),
        )
        if _phased.enabled(self.config):
            # In the confirm stage, what the evidence gate records must be this reading's result.
            _phased.note_confirm_result(self.quest_root, result_json)
        return {
            "result_json": result_json,
            "data_files": [str(e.path) for e in entries],
            # Mark the no-sim path explicitly so analyze can flavor its
            # interpretation ("the user collected this; treat citations
            # as primary sources, not simulation outputs").
            "exec_result": {
                "returncode": 0,
                "source": "no_simulation_user_data",
                "n_files": len(entries),
            },
            # No figures from a simulation — leave the list empty.
            "figures": [],
        }

    def _code_design_block(self, state: QuestState, *, names_only: bool = False) -> str:
        """The design as a request that writes or repairs code may see it: without the plan's own numbers for what the
        code must reproduce. Each equation's worked ``example`` is dropped (``names_only``: kept as the names of its
        inputs, which the outline needs to name a function's parameters), and each check's ``expected`` and
        ``expected_formula`` are dropped; the code is judged against them by FI, never written toward them."""
        design = copy.deepcopy(state.get("design") or {})
        if isinstance(design, dict) and isinstance(design.get("protocol"), dict):
            _strip_plan_numbers(design["protocol"], names_only=names_only)
        if isinstance(design, dict) and (open_items := _audit_open.for_code(self.quest_root)):
            design["proposed_by_the_methodology_audit_not_applied"] = open_items  # what the plan does not hold, said openly
        return json.dumps(design, indent=2)
