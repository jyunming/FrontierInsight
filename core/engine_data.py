"""Methods of :class:`core.engine.Engine` for data that is collected instead of simulated.

``Engine`` inherits :class:`DataStageMixin`; every method here uses the engine's own attributes and its other methods
through ``self``. The shared names come from ``core/engine_base.py``.
"""

from __future__ import annotations

from .engine_base import (  # noqa: F401
    Any, FOUNDATIONAL_MAX_SUGGESTED, Path, QuestState, RetrievedDoc, WORK_SCOPE_PAPERS, WORK_SCOPE_PAPERS_AND_BOOKS,
    _FI_INTERNAL_KINDS, _audit_open, _figures_section, _foundational_outcomes, _foundational_work_text,
    _is_preliminary_memory, _item_content, _list_user_data_files, _literature_query_prompt, _normalize_title,
    _parse_json_lenient, _phased, _record_query_set, _render_auto_collected_md, _render_data_readme,
    _save_literature_figures, _screen_grades, _slugify, _source_ref, _source_text, _split_figure_readings,
    _strip_plan_numbers, asyncio, copy, functools, json,
)


class DataStageMixin:
    """Methods of :class:`core.engine.Engine`; see the module docstring."""

    async def _node_auto_collect_data(self, state: QuestState) -> QuestState:
        """Agent-side data collection via Axon, run BEFORE
        the wait_for_data pause in no-simulation mode.

        Why this exists: the user said *"data can be collected by
        agent as well, not users only"* — for many no-simulation
        topics (literature reviews, cross-cultural comparisons,
        history surveys) Axon already holds enough to answer the
        question, and there's no reason to interrupt the user when
        the corpus already covers the topic. This node tries to
        pull ``top_k`` relevant docs and writes each one as a
        Markdown file under ``<quest_root>/data/auto_collected/`` so
        the downstream nodes (``wait_for_data`` → ``data_load``) see
        them as ordinary user data and proceed without pausing.

        Passthrough discipline (each case is distinguishable in run.log):
        * ``engine.auto_collect_data`` is False — user opted out. No
          Axon call. Logged INFO.
        * ``knowledge.enabled`` is False — no Axon to query. No Axon
          call. Logged INFO.
        * ``Knowledge.asearch`` IS called but raises — the exception
          is caught, logged WARNING, and the node returns
          ``auto_collected_count=0`` so ``wait_for_data`` takes over.
        * ``Knowledge.asearch`` returns zero docs — logged INFO. No
          files written. Empty ``auto_collected/`` directory is NOT
          left behind.

        On resume the node re-runs and re-walks Axon — files at the
        same rank slot get overwritten with the latest retrieval. The
        user's manually dropped files (anywhere in ``data/`` other
        than ``auto_collected/``) are untouched on each pass.
        """
        if not self.config.engine.auto_collect_data:
            self._log.info(
                "[auto_collect] engine.auto_collect_data=False — "
                "skipping; will pause for user data",
            )
            asked_off = [str(p).strip() for p in state.get("refine_extend") or [] if str(p).strip()]
            if asked_off:
                return {"auto_collected_count": 0, "refine_extend": [], "extend_missed": asked_off}
            return {"auto_collected_count": 0}

        # Build the query once and reuse for BOTH Axon and dataset
        # adapters. Topic alone is enough for first-pass retrieval;
        # hypothesis sharpens it on later iterations.
        topic = state.get("topic", "")
        design = state.get("design") or {}
        hypothesis = ""
        if isinstance(design, dict):
            hypothesis = str(design.get("hypothesis", "")).strip()
        # The keyword query the literature step derived from the topic (search engines and the dataset adapters
        # match terms, not the topic's sentences); the topic and hypothesis as written when there is none.
        derived = str(state.get("literature_query") or "").strip() if state.get("literature_query_derived") else ""
        # What a source must be about is judged against the question as written, not the few keywords.
        relevance_topic = f"{topic} {hypothesis}".strip() or topic
        query = derived or relevance_topic
        asked = [str(p).strip() for p in state.get("refine_extend") or [] if str(p).strip()]
        if asked:
            # A person asked for something to be added: look for exactly that, not the topic again.
            query = " ".join(asked)
            self._log.info("[auto_collect] collecting what the person asked to add: %s", "; ".join(p[:80] for p in asked))
        auto_dir = self.quest_root / "data" / "auto_collected"

        # ---- Reuse already-downloaded literature -------------------
        # The literature node ran earlier in the graph and already fetched
        # (and, with web_fetch_pages, read the full text of) a set of
        # web/Axon sources into state["literature"] AND wrote them to
        # data/literature/. data_load walks the whole data/ tree, so we
        # reuse those directly — counting them here (so the no-sim gate
        # doesn't pause) WITHOUT writing a duplicate copy under
        # auto_collected/. Avoids re-querying the web (wasteful, and the
        # keyless DuckDuckGo backend can rate-limit the Nth query) and
        # avoids double-feeding the same content into the analysis.
        # Reuse already-downloaded literature — but DROP off-topic sources
        # so an irrelevant hit (the classic "banana bread" result for a
        # "SpaceX revenue" topic) can't silently satisfy the no-sim gate
        # without the relevance check the Axon top-up path already applies.
        lit_docs = [
            d for d in (state.get("literature") or [])
            if isinstance(d, dict)
            and ((d.get("metadata") or {}).get("title")
                 or (d.get("metadata") or {}).get("url"))
            and (d.get("metadata") or {}).get("kind") not in _FI_INTERNAL_KINDS
            and not _is_preliminary_memory(d.get("metadata") or {})
        ]
        if asked:
            lit_written = self._literature_seed_step(state)
        elif lit_docs and self.config.knowledge.relevance_guard:
            relevant = await self._filter_relevant_docs(relevance_topic, lit_docs)
            lit_written = len(relevant)
            if lit_written < len(lit_docs):
                self._log.info(
                    "[auto_collect] relevance guard: %d/%d reused literature "
                    "doc(s) on-topic — off-topic ones don't satisfy the "
                    "no-sim gate", lit_written, len(lit_docs),
                )
            elif lit_written:
                self._log.info(
                    "[auto_collect] reusing %d already-downloaded literature "
                    "doc(s) from data/literature/", lit_written,
                )
        else:
            lit_written = self._literature_seed_step(state)

        # ---- Axon / web retrieval (top-up) -------------------------
        # Only fire a fresh knowledge-layer query when the literature
        # node gave us nothing to reuse — avoids the redundant,
        # rate-limit-prone re-search in the common case.
        axon_written = 0
        if lit_written == 0 or asked:
            axon_written = await self._axon_collect_step(
                query, auto_dir, work_scope=self._work_scope(state), relevance_topic=relevance_topic,
            )

        # ---- Dataset adapters --------------------------------------
        adapter_written = await self._run_dataset_adapters(query, auto_dir)

        # lit_written is a COUNT of sources reused from data/literature/ (no
        # files written there); only axon + adapters write into auto_dir.
        files_written = axon_written + adapter_written
        written = lit_written + files_written
        # If nothing landed in auto_collected/, clean up any empty top-level
        # dir that might have been created mid-flight.
        if files_written == 0 and auto_dir.is_dir() and not any(auto_dir.iterdir()):
            try:
                auto_dir.rmdir()
            except OSError as e:
                self._log.warning(
                    "[auto_collect] zero writes AND could not rmdir "
                    "leftover %s: %s", auto_dir, e,
                )
        self._log.info(
            "[auto_collect] %d source(s) available for analysis "
            "(literature_reuse=%d from data/literature/, axon=%d + "
            "adapters=%d written to auto_collected/)",
            written, lit_written, axon_written, adapter_written,
        )
        if asked:
            if files_written == 0:
                self._log.warning("[auto_collect] nothing new was found for: %s", "; ".join(p[:80] for p in asked))
            return {"auto_collected_count": written, "refine_extend": [],
                    "extend_missed": asked if files_written == 0 else []}
        return {"auto_collected_count": written}

    def _literature_seed_step(self, state: QuestState) -> int:
        """Count the web/Axon sources the literature node already fetched
        (``state['literature']``) so the no-simulation gate
        (``auto_collected_count``) treats the quest as having data and
        doesn't pause for user uploads.

        Does NOT write a second copy: those sources were already downloaded
        to ``data/literature/`` (the literature node does this for every
        quest), and ``data_load`` walks the whole ``data/`` tree — so a
        duplicate copy under ``auto_collected/`` only double-feeds the same
        content into the analysis. Counts the same citable entries
        ``_write_literature_files`` would keep (a title or URL, not an
        FI-internal artifact)."""
        n = 0
        for item in state.get("literature") or []:
            if not isinstance(item, dict):
                continue
            meta = item.get("metadata") or {}
            if not (meta.get("title") or meta.get("url")):
                continue
            if meta.get("kind") in _FI_INTERNAL_KINDS or _is_preliminary_memory(meta):
                continue
            n += 1
        if n:
            self._log.info(
                "[auto_collect] reusing %d already-downloaded literature "
                "doc(s) from data/literature/ (no duplicate copy written)", n,
            )
        return n

    def _write_literature_files(self, lit: list, out_dir: Path) -> int:
        """Write each citable literature entry to ``out_dir`` as a
        ``lit_NNN_slug.md`` file (YAML front matter + content, via
        ``_render_auto_collected_md``). Called by the ``literature`` node's
        always-on download into ``data/literature/`` (runs for EVERY quest,
        sim or no-sim) — the single on-disk copy of the retrieved sources,
        which ``data_load`` then reuses on the no-simulation path.

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

    async def _axon_collect_step(
        self, query: str, auto_dir: Path, *, work_scope: str = WORK_SCOPE_PAPERS, relevance_topic: str = "",
    ) -> int:
        """Axon-backed retrieval. Returns the
        count of files written under ``auto_dir`` (not in a
        sub-directory). Returns 0 on any of: knowledge disabled,
        asearch raised, zero hits, or every write failed. The
        caller is responsible for combining this with the dataset-
        adapter count and for the final empty-dir cleanup.
        """
        if not self.knowledge.enabled:
            self._log.info(
                "[auto_collect] knowledge.enabled=False — skipping "
                "Axon retrieval (dataset adapters may still run)",
            )
            return 0
        top_k = self.config.engine.auto_collect_top_k
        self._log.info(
            "[auto_collect] querying knowledge layer (Axon + web): "
            "top_k=%d query=%r",
            top_k, query[:120],
        )
        try:
            # Pass chat_fn so the source router (source_routing="auto") can
            # pick the right sources for the topic — crucially, for a
            # non-academic question ("SpaceX revenue") it selects general
            # web search and skips the scholarly adapters that would
            # otherwise return irrelevant nearest-neighbour papers. The
            # web layer runs in parallel regardless; routing just spares
            # the wasted academic calls.
            docs = await self.knowledge.asearch(
                query,
                top_k=top_k,
                chat_fn=functools.partial(self._chat_messages, node="source_router"),
                work_scope=work_scope,
            )
        except Exception as e:
            self._log.warning(
                "[auto_collect] knowledge search raised: %s — dataset "
                "adapters may still run", e,
            )
            return 0
        if not docs:
            self._log.info(
                "[auto_collect] knowledge layer returned 0 docs — dataset "
                "adapters may still run",
            )
            return 0

        # Relevance guard: a retrieval can come back full of confident-but-
        # irrelevant hits (the classic failure: academic adapters returning
        # physics preprints for a "SpaceX revenue" question). Drop the
        # off-topic ones so we don't write a paper on garbage; if NOTHING
        # is on-topic, return 0 so wait_for_data pauses for real user data
        # instead of proceeding.
        # An earlier FI result kept as preliminary is a reminder, never data (core/engine.py::_is_preliminary_memory).
        docs = [d for d in docs if not _is_preliminary_memory(getattr(d, "metadata", None) or {})]
        docs = await self._filter_relevant_docs(relevance_topic or query, docs) if docs else []
        if not docs:
            self._log.info(
                "[auto_collect] relevance guard dropped every auto-collected "
                "doc as off-topic — pausing for user-supplied sources",
            )
            return 0

        written_targets: list[Path] = []
        for idx, doc in enumerate(docs, start=1):
            meta = doc.metadata or {}
            source = str(meta.get("source") or meta.get("path") or "")
            slug_basis = Path(source).stem if source else f"doc{idx}"
            slug = _slugify(slug_basis)[:40] or f"doc{idx}"
            target = auto_dir / f"{idx:03d}_{slug}.md"
            body = _render_auto_collected_md(idx, meta, doc.content or "")
            try:
                auto_dir.mkdir(parents=True, exist_ok=True)
                target.write_text(body, encoding="utf-8")
                written_targets.append(target)
            except OSError as e:
                self._log.warning(
                    "[auto_collect] failed to write %s: %s — skipping doc",
                    target, e,
                )
        return len(written_targets)

    async def _filter_relevant_docs(self, topic: str, docs: list) -> list:
        """Relevance guard: ask the model which of the retrieved docs are
        actually on-topic and keep only those. A single cheap judging call
        (not one per doc). On any failure — guard disabled, LLM error,
        unparseable response — return the docs unchanged (fail-open: never
        silently drop a quest's whole corpus because the judge hiccuped).
        """
        if not docs or not self.config.knowledge.relevance_guard:
            return docs
        listing_lines: list[str] = []
        for i, d in enumerate(docs):
            # Dict-aware: state["literature"] entries are dicts, while the
            # Axon path passes RetrievedDoc objects. Without this the guard
            # judged blind on dicts (empty title/excerpt) and fail-opened.
            if isinstance(d, dict):
                meta = d.get("metadata") or {}
                content = d.get("content") or ""
            else:
                meta = getattr(d, "metadata", {}) or {}
                content = getattr(d, "content", "") or ""
            title = (
                (meta.get("title") or meta.get("source") or "")
                if isinstance(meta, dict) else ""
            )
            excerpt = str(content)[:200].replace("\n", " ")
            listing_lines.append(f"[{i}] {title} :: {excerpt}")
        prompt = (
            "You are a relevance filter for a research assistant. The user's "
            "topic is:\n\n"
            f"{topic[:600]}\n\n"
            "Below are candidate sources the retriever returned. Some may be "
            "confidently retrieved yet completely off-topic (e.g. physics "
            "papers returned for a corporate-finance question). Return ONLY "
            "the indices that are genuinely relevant to the topic.\n\n"
            "# Candidates\n"
            # Retrieved titles and excerpts are material to judge, never instructions (core/source_text.py).
            + _source_text.fence("\n".join(listing_lines))
            + "\n\nRespond with a single JSON object, no prose:\n"
            '{"relevant_indices": [<int>, ...]}\n'
            "If NONE are relevant, return an empty list."
        )
        try:
            raw = await self._chat(prompt, node="relevance_guard")
            parsed = _parse_json_lenient(raw, node="relevance_guard")
            if not parsed or "relevant_indices" not in parsed:
                return docs
            idxs = parsed.get("relevant_indices") or []
            keep_idx = {int(i) for i in idxs if isinstance(i, (int, float))}
        except Exception as e:
            self._log.info("[auto_collect] relevance guard failed: %s — keeping all", e)
            return docs
        kept = [d for i, d in enumerate(docs) if i in keep_idx]
        dropped = len(docs) - len(kept)
        if dropped:
            self._log.info(
                "[auto_collect] relevance guard kept %d/%d docs (%d off-topic dropped)",
                len(kept), len(docs), dropped,
            )
        return kept

    def _record_source_verdicts(
        self, stage: str, iteration: int | None, outcome: str, rows: list[dict[str, Any]], *,
        threshold: float | int | None = None, minimum: int | None = None, call_id: str | None = None,
    ) -> None:
        """Keep how the retrieved sources were judged in ``.fi/literature_queries.json`` beside the queries that found
        them: one entry per pass and stage (``floor``, ``screen``) with each source's score or grade, whether it was
        kept and why, and (``retractions``) what Crossref says of each source's retraction. Best-effort: a record never
        stops a quest."""
        if not self.config.knowledge.enabled:
            return
        try:
            _record_query_set(self.fi_dir, {
                "stage": stage, "iteration": iteration, "outcome": outcome, "threshold": threshold,
                "minimum": minimum, "call_id": call_id, "sources": rows,
            }, log=self._log)
        except Exception as e:  # noqa: BLE001
            self._log.debug("[literature] could not record the %s verdicts: %r", stage, e)

    def _record_floor_verdicts(self, docs: list, kept: list, stats: dict, iteration: int | None) -> None:
        """The on-topic score of every retrieved source and whether it cleared the floor."""
        if not docs:
            return
        scores = stats.get("scores")
        if not stats.get("scored") or not scores or len(scores) != len(docs):
            self._record_source_verdicts(
                "floor", iteration, "not_scored" if self.config.knowledge.relevance_min_score > 0 else "off", [])
            return
        threshold, minimum = stats["threshold"], stats["minimum"]
        kept_ids = {id(d) for d in kept}
        rows = []
        for d, sc in zip(docs, scores):
            cleared = sc >= threshold
            if id(d) in kept_ids:
                why = (f"score {sc:.2f}, at or above {threshold:.2f}" if cleared
                       else f"score {sc:.2f} is below {threshold:.2f}; kept to reach the minimum of {minimum} sources")
            else:
                why = f"score {sc:.2f} is below {threshold:.2f}"
            rows.append({**_source_ref(d), "score": round(sc, 4), "kept": id(d) in kept_ids, "why": why})
        self._record_source_verdicts("floor", iteration, "scored", rows, threshold=threshold, minimum=minimum)

    async def _screen_literature(
        self, topic: str, docs: list, *, work_scope: str = WORK_SCOPE_PAPERS, iteration: int | None = None,
    ) -> list:
        """Grade every retrieved source 0-3 in one batched call and keep the
        citable ones (rubric in ``agents/literature_screen.md``).

        The embedding floor that runs first scores word overlap, so a table of
        contents or a paper on another system that shares the search terms
        passes it; this asks whether the paper could cite the source for a
        claim. Scholarly records need a 2. Web pages are dropped only at 0:
        they are kept for the text the writer quotes, and a page of general
        background is exactly what a 1 describes. At least
        ``knowledge.relevance_min_keep`` sources survive, best grades first,
        so a thin retrieval is not emptied (the evidence gate can broaden).
        Fail-open: when the screen is off, the call fails or the reply cannot
        be read, every source is kept, and so is any source left ungraded.
        Graded sources carry ``screen_grade``. Every source's grade and the reason it was kept or dropped go to
        ``.fi/literature_queries.json`` (:meth:`_record_source_verdicts`).
        """
        kn = self.config.knowledge
        # Papers the user supplied are theirs to judge: never shown, never dropped.
        own = {
            i for i, d in enumerate(docs)
            if (d.metadata or {}).get("source") in ("local_paper", "user_supplied")
        }
        if not docs:
            return docs

        def _kept_all(outcome: str, why: str, call_id: str | None = None) -> list:
            self._record_source_verdicts(
                "screen", iteration, outcome,
                [{**_source_ref(d), "grade": None, "kept": True,
                  "why": "your own paper: never screened" if i in own else why} for i, d in enumerate(docs)],
                call_id=call_id)
            return docs

        if not kn.literature_screen:
            return _kept_all("off", "the screen is off (knowledge.literature_screen), so every source is kept")
        if len(own) == len(docs):
            return _kept_all("own_only", "your own paper: never screened")
        lines: list[str] = []
        for i, d in enumerate(docs):
            if i in own:
                continue
            md = d.metadata or {}
            kind = "web page" if md.get("source") == "web_search" else "paper"
            if md.get("foundational"):
                kind = "foundational " + ("book" if md.get("work_type") in ("book", "book-chapter") else "paper")
            title = " ".join(str(md.get("title") or md.get("url") or "(untitled)").split())
            facts = ", ".join(
                str(v) for v in (md.get("venue"), md.get("year"), md.get("work_type"), md.get("foundational")) if v
            )
            excerpt = " ".join(_split_figure_readings(str(d.content or ""))[0].split())[:300]
            lines.append(
                f"[{i}] ({kind}) {title}" + (f" — {facts}" if facts else "") + f" :: {excerpt}"
            )
        if work_scope == WORK_SCOPE_PAPERS_AND_BOOKS:
            guidance = (
                "This quest has no experiment, so books, book chapters and "
                "humanities or social-science scholarship count as fully as "
                "journal articles: judge them by subject, not by format."
            )
        else:
            guidance = (
                "This quest runs an experiment. Work on the same system, method "
                "or measured quantity is the most useful; a paper from another "
                "field that only shares vocabulary is a 0 or a 1."
            )
        prompt = self._prompts["literature_screen"].substitute(
            # Retrieved titles and excerpts are material to grade, never instructions (core/source_text.py).
            topic=topic[:1200], kind_guidance=guidance, candidates=_source_text.fence("\n".join(lines)),
        )
        self.__dict__.setdefault("_last_call_id", {}).pop("literature_screen", None)
        try:
            raw = await self._chat(prompt, node="literature_screen")
            parsed = _parse_json_lenient(raw, node="literature_screen")
        except Exception as e:  # noqa: BLE001 — the screen must never cost the corpus
            self._log.info("[literature] screen failed (%r); keeping all %d sources", e, len(docs))
            return _kept_all("failed", "the screen call failed, so every source is kept",
                             self.__dict__.get("_last_call_id", {}).get("literature_screen"))
        call_id = self.__dict__.get("_last_call_id", {}).get("literature_screen")
        grades = _screen_grades(parsed, len(docs))
        if not grades:
            self._log.info("[literature] screen reply unreadable; keeping all %d sources", len(docs))
            return _kept_all("unreadable", "the screen's answer could not be read, so every source is kept", call_id)
        # A grade the model gave a user-supplied paper anyway does not count.
        grades = {i: g for i, g in grades.items() if i not in own}
        keep = [
            i for i, d in enumerate(docs)
            if i not in grades
            or grades[i] >= (1 if (d.metadata or {}).get("source") == "web_search" else 2)
        ]
        minimum = min(kn.relevance_min_keep, len(docs))
        if len(keep) < minimum:
            # Stable sort: equal grades stay in retrieval order.
            rest = sorted((i for i in range(len(docs)) if i not in keep),
                          key=lambda i: -grades.get(i, 0))
            keep = sorted(keep + rest[:minimum - len(keep)])
        kept_set = set(keep)
        rows = []
        for i, d in enumerate(docs):
            need = 1 if (d.metadata or {}).get("source") == "web_search" else 2
            if i in own:
                why = "your own paper: never screened"
            elif i not in grades:
                why = "the screen gave it no grade, so it is kept"
            elif grades[i] >= need:
                why = f"graded {grades[i]}; {need} or higher is kept"
            elif i in kept_set:
                why = f"graded {grades[i]}, below {need}; kept to reach the minimum of {minimum} sources"
            else:
                why = f"graded {grades[i]}; needs {need} or higher"
            rows.append({**_source_ref(d), "grade": grades.get(i), "kept": i in kept_set, "why": why})
        self._record_source_verdicts("screen", iteration, "graded", rows, minimum=minimum, call_id=call_id)
        out = []
        for i in keep:
            md = dict(docs[i].metadata or {})
            if i in grades:
                md["screen_grade"] = grades[i]
            out.append(RetrievedDoc(content=docs[i].content, metadata=md))
        values = list(grades.values())
        self._log.info(
            "[literature] screen kept %d/%d sources (grade counts: %s)",
            len(out), len(docs), {g: values.count(g) for g in sorted(set(values))},
        )
        return out

    async def _foundational_works(
        self, topic: str, docs: list, *, work_scope: str = WORK_SCOPE_PAPERS,
    ) -> list:
        """Candidates a keyword search does not reach: the original papers and
        standard textbooks of what a topic rests on. One call asks the model
        for up to eight, each is looked up by title in OpenAlex, or by author and
        year when no title matches, and dropped when not found, and the works
        several retrieved papers cite are added. They
        are labelled foundational and go through the literature screen. Off
        with ``knowledge.foundational_works: false``; any failure adds nothing.

        The run log names every work the model suggested and what became of
        each, because a work the model never named cannot be told from one
        OpenAlex does not hold without it."""
        kn = self.config.knowledge
        if not (kn.enabled and kn.foundational_works):
            return []
        suggestions = await self._suggest_foundational_works(topic, work_scope=work_scope)
        self._log.info(
            "[literature] foundational works suggested (%d): %s",
            len(suggestions), "; ".join(_foundational_work_text(w) for w in suggestions) or "(none)",
        )
        try:
            found = await self.knowledge.find_foundational_works(suggestions, docs)
        except Exception as e:  # noqa: BLE001 — never costs the retrieval
            self._log.info("[literature] foundational works lookup failed: %r", e)
            return []
        have = {_normalize_title((d.metadata or {}).get("title") or "") for d in docs} - {""}
        new = [d for d in found if _normalize_title(d.metadata.get("title") or "") not in have]
        self._log.info(
            "[literature] foundational works: %d suggested, %d new candidate(s)%s",
            len(suggestions), len(new),
            (": " + "; ".join(f"{d.metadata.get('title')} ({d.metadata.get('year')}, "
                              f"{d.metadata.get('foundational')})" for d in new))[:600] if new else "",
        )
        if suggestions:
            added, kept_already, dropped = _foundational_outcomes(suggestions, new, found, docs)
            self._log.info(
                "[literature] foundational works: found in OpenAlex and added (%d): %s | "
                "already among the search results (%d): %s | dropped, not found in OpenAlex "
                "(%d): %s",
                len(added), "; ".join(added) or "-",
                len(kept_already), "; ".join(kept_already) or "-",
                len(dropped), "; ".join(dropped) or "-",
            )
        return new

    async def _suggest_foundational_works(
        self, topic: str, *, work_scope: str = WORK_SCOPE_PAPERS,
    ) -> list[dict]:
        """Up to eight foundational works the model names for ``topic``, each
        ``{title, authors, year}``; any beyond the eighth are ignored. Empty
        when the call fails."""
        if work_scope == WORK_SCOPE_PAPERS_AND_BOOKS:
            what = "the classic books and articles scholars of this subject cite as its foundations"
        else:
            what = ("the original papers that introduced the methods, models or effects it "
                    "relies on, and the standard textbooks on them")
        prompt = (
            f"List up to EIGHT foundational works for this research topic: {what}. "
            "Name only works you are sure exist, with their exact titles: each one is "
            "looked up by title, and a work that is not found is dropped.\n\n"
            f"RESEARCH TOPIC:\n{topic[:1200]}\n\n"
            'Reply as JSON only: {"works": [{"title": "<exact title>", '
            '"authors": "<first author\'s surname>", "year": <year>}]}'
        )
        try:
            raw = await self._chat(prompt, node="literature_foundational")
            parsed = _parse_json_lenient(raw, node="literature_foundational")
        except Exception as e:  # noqa: BLE001 — best-effort
            self._log.info("[literature] foundational works suggestion failed: %r", e)
            return []
        works = parsed.get("works") if isinstance(parsed, dict) else None
        return [
            w for w in (works if isinstance(works, list) else [])
            if isinstance(w, dict) and str(w.get("title") or "").strip()
        ][:FOUNDATIONAL_MAX_SUGGESTED]

    async def _derive_literature_queries(
        self, topic: str, idea_title: str = "", hypothesis: str = "",
        *, work_scope: str = WORK_SCOPE_PAPERS, node: str = "literature_query",
    ) -> list[str]:
        """Turn a topic statement into up to three keyword search queries,
        one per facet of the topic.

        A search engine matches keywords, and one query reaches only the work
        written in its own words. Three facets -- the core subject, a specific
        angle and the wider frame -- reach work a single query misses; the
        results are merged and screened afterwards. The vocabulary follows the
        kind of quest. One with an experiment searches under the names of
        methods, systems and measured quantities. One without searches under
        the names scholars of that subject write about -- works, people,
        periods, places, movements -- because wording decides what a database
        returns: on one humanities topic the same database gave 0 of 10
        on-topic hits for the STEM-style query and 8 of 10 for the other.

        Returns [] when retrieval is off (nothing would read the queries), when
        the model fails, or when no reply is a keyword query (a long sentence
        is the topic echoed back). The caller then keeps its own query, so
        this changes what gets sent but never stops the search.
        """
        if not self.config.knowledge.enabled:
            return []
        prompt = _literature_query_prompt(topic, idea_title, hypothesis, work_scope=work_scope)
        try:
            raw = await self._chat(prompt, node=node)
            parsed = _parse_json_lenient(raw, node=node)
        except Exception as e:  # noqa: BLE001 — best-effort; caller degrades
            self._log.info("[literature] query derivation failed: %r", e)
            return []
        if not isinstance(parsed, dict):
            return []
        replies = parsed.get("queries")
        if not isinstance(replies, list):
            # A reply in the one-query shape still yields one facet.
            replies = [parsed.get("query")]
        out: list[str] = []
        for reply in replies:
            q = " ".join(str(reply or "").split())[:300]
            if q and len(q.split()) <= 20 and q.lower() not in {o.lower() for o in out}:
                out.append(q)
        return out[:3]

    async def _propose_literature_queries(
        self, topic: str, tried: list[str], missed: list,
        *, work_scope: str = WORK_SCOPE_PAPERS,
    ) -> str:
        """Ask for ONE better search query after a retrieval missed entirely.

        Shows the model what was already tried and a sample of what came back,
        so it can tell "wrong vocabulary" (the usual cause — a field's papers
        use different terms than the topic statement) from "too narrow".
        Returns "" on any failure; the caller then keeps the original results
        rather than looping, so a flaky model degrades to today's behaviour.
        """
        titles = []
        for d in missed[:6]:
            meta = (d.get("metadata") if isinstance(d, dict) else getattr(d, "metadata", {})) or {}
            t = str(meta.get("title") or "").strip()
            if t:
                titles.append(f"- {t[:120]}")
        prompt = (
            "A literature search returned results that are all off-topic.\n\n"
            f"RESEARCH TOPIC:\n{topic[:600]}\n\n"
            "QUERIES ALREADY TRIED (do not repeat these):\n"
            + "\n".join(f"- {q[:160]}" for q in tried)
            + "\n\nWHAT CAME BACK (all judged off-topic):\n"
            + (_source_text.fence("\n".join(titles), "the titles of retrieved sources") if titles else "- (no titles)")
            + "\n\nThe likely cause is vocabulary: this field's papers may use "
            "different terminology than the topic statement does. Propose ONE "
            "alternative search query that uses the terms researchers in this "
            "field would actually publish under. "
            + ("Prefer the names scholars of this subject write under -- works, "
               "people, periods, places, movements, genres, concepts -- over "
               "method words. "
               if work_scope == WORK_SCOPE_PAPERS_AND_BOOKS else
               "Prefer domain-standard terms. ")
            + "Spell out acronyms. Keep it under 20 words.\n\n"
            'Reply as JSON only: {"query": "<your query>"}'
        )
        try:
            raw = await self._chat(prompt, node="literature_requery")
            parsed = _parse_json_lenient(raw, node="literature_requery")
            q = str((parsed or {}).get("query") or "").strip()
        except Exception as e:  # noqa: BLE001 — best-effort; caller degrades
            self._log.info("[literature] requery proposal failed: %r", e)
            return ""
        if not q or q.lower() in {t.lower() for t in tried}:
            return ""
        return q[:300]

    def _filter_docs_by_relevance(
        self, topic: str, docs: list, *, stats: dict | None = None,
    ) -> list:
        """Deterministic relevance floor for the literature node: score each
        retrieved doc by embedding cosine against the TOPIC and drop the
        off-topic tail. Unlike the LLM ``_filter_relevant_docs`` guard — which
        runs only on the auto_collect path — this runs on EVERY quest's fresh
        literature, so survey / simulation quests (which skip auto_collect)
        don't carry off-topic sources (e.g. change-point-math papers retrieved
        for a sculpture-history topic) into analyze/write.

        No LLM call — reuses the all-MiniLM cosine helper from ``passages``.
        Fail-open and never-starve: returns docs unchanged when embeddings are
        unavailable (FI_OFFLINE / no model / error), when ``relevance_min_score``
        is 0, or when docs is empty; and always retains at least
        ``relevance_min_keep`` top-scoring docs so a wholly-borderline corpus is
        not emptied (the evidence_gate can then broaden). Scores against the raw
        TOPIC (not the retrieval query, whose folded-in hypothesis keywords are
        exactly what drags off-topic hits in)."""
        min_score = self.config.knowledge.relevance_min_score
        min_keep = self.config.knowledge.relevance_min_keep
        if not docs or min_score <= 0.0:
            return docs
        from core.passages import _embed_scores  # shared MiniLM; FI_OFFLINE-safe

        blobs: list[str] = []
        for d in docs:
            # Dict- and RetrievedDoc-aware, mirroring _filter_relevant_docs.
            if isinstance(d, dict):
                meta = d.get("metadata") or {}
                content = d.get("content") or ""
            else:
                meta = getattr(d, "metadata", {}) or {}
                content = getattr(d, "content", "") or ""
            title = (
                (meta.get("title") or meta.get("source") or "")
                if isinstance(meta, dict) else ""
            )
            excerpt = str(content)[:300].replace("\n", " ")
            blobs.append(f"{title} {excerpt}".strip())
        scores = _embed_scores(blobs, str(topic or "")[:600])
        if scores is None:  # embeddings unavailable — never filter blind
            if stats is not None:
                stats["scored"] = False
            return docs
        order = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
        keep_idx = {i for i in range(len(docs)) if scores[i] >= min_score}
        if stats is not None:
            # ``above_floor`` counts docs that cleared the threshold on their
            # own merits, BEFORE never-starve retention pads the set back up.
            # The requery loop needs that distinction: "3 docs kept" can mean
            # "3 good hits" or "everything missed and min_keep held 3 back",
            # and only the second is worth re-searching for.
            stats["scored"] = True
            stats["above_floor"] = len(keep_idx)
            stats["best"] = max(scores) if scores else 0.0
            stats["scores"] = [float(s) for s in scores]
            stats["threshold"] = min_score
            stats["minimum"] = min_keep
        keep_idx.update(order[:max(0, min_keep)])  # never-starve retention
        kept = [d for i, d in enumerate(docs) if i in keep_idx]
        dropped = len(docs) - len(kept)
        if dropped:
            # ``lowest_kept`` can be below the threshold when the never-starve
            # retention (min_keep) held back a doc — label the values + surface
            # min_keep so that isn't misread as the floor being violated.
            lowest_kept = min((scores[i] for i in keep_idx), default=0.0)
            self._log.info(
                "[literature] relevance floor kept %d/%d docs (%d dropped; "
                "threshold=%.2f, min_keep=%d, lowest kept cosine=%.2f)",
                len(kept), len(docs), dropped, min_score, min_keep, lowest_kept,
            )
        return kept

    async def _run_dataset_adapters(
        self, query: str, auto_dir: Path,
    ) -> int:
        """Iterate the user's enabled
        ``engine.dataset_adapters``, ask each for up to
        ``engine.dataset_adapter_top_k`` rows, and render each row as
        a Markdown file under ``<auto_dir>/<adapter_name>/``.

        Returns the count of files actually written so the calling
        node can fold it into ``auto_collected_count``. Failures (an
        adapter raises, returns nothing, or writes can't land on
        disk) are logged and skipped — the engine never aborts the
        no-simulation flow because of a flaky dataset API.
        """
        adapter_names = self.config.engine.dataset_adapters or []
        if not adapter_names:
            return 0
        from .datasets import ADAPTER_REGISTRY

        top_k = self.config.engine.dataset_adapter_top_k
        total_written = 0
        for name in adapter_names:
            adapter_cls = ADAPTER_REGISTRY.get(name)
            if adapter_cls is None:
                self._log.warning(
                    "[auto_collect] unknown dataset adapter %r "
                    "(known: %s) — skipping",
                    name, sorted(ADAPTER_REGISTRY),
                )
                continue
            self._log.info(
                "[auto_collect] running dataset adapter %s (top_k=%d)",
                name, top_k,
            )
            try:
                rows = await adapter_cls().search(query, top_k=top_k)
            except Exception as e:
                self._log.warning(
                    "[auto_collect] dataset adapter %s raised: %s — "
                    "skipping its results",
                    name, e,
                )
                continue
            if not rows:
                self._log.info(
                    "[auto_collect] dataset adapter %s returned 0 rows",
                    name,
                )
                continue
            sub_dir = auto_dir / name
            for idx, row in enumerate(rows, start=1):
                slug_basis = (
                    row.metadata.get("indicator_id")
                    or row.metadata.get("title")
                    or f"row{idx}"
                )
                slug = _slugify(str(slug_basis))[:40] or f"row{idx}"
                fname = f"{idx:03d}_{slug}.md"
                target = sub_dir / fname
                body = _render_auto_collected_md(
                    idx, {**row.metadata, "adapter": name}, row.content,
                )
                try:
                    sub_dir.mkdir(parents=True, exist_ok=True)
                    target.write_text(body, encoding="utf-8")
                    total_written += 1
                except OSError as e:
                    self._log.warning(
                        "[auto_collect] adapter %s: failed to write %s: %s",
                        name, target, e,
                    )
            # If every write for this adapter failed, clean the empty
            # subdirectory like we do for Axon results.
            if sub_dir.is_dir() and not any(sub_dir.iterdir()):
                try:
                    sub_dir.rmdir()
                except OSError:
                    pass
        return total_written

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
        return {"data_files": [str(p) for p in user_files]}

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
