"""Typed configuration for Frontier Insight runs (post-DS redesign).

The schema is intentionally narrow — anything that varies per run goes in
YAML, anything that's a code-level concern stays in Python. Path fields
expand `~` on load via mode='before' validators.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

ProviderName = Literal[
    "codex",
    "openai",
    "gemini",
    "ollama",
    "vllm",
    "claude_code",
    "github_copilot_cli",
    "github_copilot_vscode",
    "codex_cli",      # local Codex CLI (uses ChatGPT Plus/Pro OAuth via `codex login`)
    "claude_cli",     # local Claude Code CLI (uses Claude Pro/Max OAuth via `claude login`)
    "copilot_cli",    # local GitHub Copilot CLI (uses `gh auth login` Copilot Pro/Business)
    "gemini_cli",     # local @google/gemini-cli (uses `gemini` OAuth / Google AI Studio key)
    # The FI VSCode extension spawns Python with --vscode-bridge-port N
    # and routes every LLM call through VSCode's vscode.lm.* Language Model
    # API. Sanctioned by GitHub; calls show up in your normal Copilot usage.
    # Not selectable from a stand-alone YAML — the extension launches FI
    # with this provider preset.
    "vscode_extension",
]
EngineFramework = Literal["langgraph"]
SandboxKind = Literal["venv", "docker"]
# Five scientific venues + four non-scientific prose formats. The
# non-scientific formats are IMRAD-free — prose-shaped output for
# cultural / historical / business / policy topics where Methods →
# Results doesn't fit.
# ``Engine._resolve_write_persona`` (NOT the prompt itself) reads
# the format and loads a per-format persona prefix from
# ``agents/write_persona_<name>.md``, which the prompt then prepends
# via ``$persona_block``. Scientific venues yield an empty persona
# block, so ``write.md`` falls through to its built-in IMRAD voice.
PaperFormat = Literal[
    "generic", "neurips", "iclr", "ieee_access", "nature_mi",
    "essay", "report", "policy_brief", "whitepaper",
]
# Subsets of PaperFormat used by Engine._resolve_write_persona and
# tests. Kept here (not in engine.py) so the next venue addition
# updates one location. The empty-intersection + union-equals-
# PaperFormat invariants are pinned by
# test_paper_format_subsets_partition_the_literal.
SCIENTIFIC_PAPER_FORMATS: frozenset[str] = frozenset({
    "generic", "neurips", "iclr", "ieee_access", "nature_mi",
})
NON_SCIENTIFIC_PAPER_FORMATS: frozenset[str] = frozenset({
    "essay", "report", "policy_brief", "whitepaper",
})
OutputKind = Literal["paper_md", "paper_pdf", "slides", "poster", "speech"]


def _expand(v: object) -> object:
    if isinstance(v, str):
        return Path(v).expanduser()
    if isinstance(v, Path):
        return v.expanduser()
    return v


class ProviderConfig(BaseModel):
    name: ProviderName = "codex"
    model: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
    # Per-node model routing. Maps an engine node name (or a
    # qualified subkey like `review_panel.methodologist`) to the model
    # string this provider should use when that node fires a chat call.
    # When None or the lookup misses, falls back to `model` above.
    #
    # Works for all three transport types:
    #   - CLI exec (claude_cli/codex_cli/copilot_cli/gemini_cli):
    #     overrides the value `_run_cli` injects after `_CliSpec.model_flag`.
    #   - HTTP proxy (claude_code/github_copilot_cli/github_copilot_vscode):
    #     overrides the `model` field in the OpenAI-compatible POST body.
    #   - HTTP direct (openai/codex/gemini/ollama/vllm):
    #     same — overrides the request body's `model`.
    #
    # Useful for picking a cheap model for low-value nodes (clarify,
    # cross_check) and a strong one for the demanding ones (write,
    # review). On `copilot_cli` and its proxy siblings, all three
    # models share one OAuth and one premium-request budget.
    node_models: dict[str, str] | None = None

    # Multi-model ensemble per node. Maps an engine node name (one of
    # ``ideate`` / ``analyze`` / ``cross_check``) to a configuration
    # that fires N parallel chat calls (one per model in ``models``)
    # and merges the responses via ``merge`` strategy.
    #
    # Cost: N + 1 calls per ensembled node (N fan-out + 1 moderator),
    # except ``merge=vote`` which is N (no moderator — pure tally).
    #
    # Example:
    #     node_ensemble:
    #       ideate:
    #         models: [gpt-5.5, claude-opus-4-7, gemini-2.5-pro]
    #         merge: tournament
    #         moderator: claude-opus-4-7
    #       analyze:
    #         models: [gpt-5.5, claude-opus-4-7]
    #         merge: synthesize
    #         moderator: claude-opus-4-7
    #       cross_check:
    #         models: [gpt-5.5, gemini-2.5-pro]
    #         merge: vote
    #
    # When None or the lookup misses, the node falls back to a single
    # call against ``model`` / ``node_models[node]`` (today's behavior
    # — no regression for quests without ensemble configured).
    node_ensemble: dict[str, "NodeEnsembleConfig"] | None = None

    @model_validator(mode="after")
    def _check_node_ensemble_node_specific_constraints(self) -> "ProviderConfig":
        """Reject merge strategies that violate the downstream parser
        contract for the node they're attached to.

        ``synthesize`` produces freeform markdown; ``analyze`` parses
        its node output as JSON. Pairing them silently corrupts the
        analysis result, so this is rejected at config load with a
        message pointing the user at ``tournament`` (which preserves
        one model's verbatim JSON output).
        """
        if not self.node_ensemble:
            return self
        for node, cfg in self.node_ensemble.items():
            if node == "analyze" and cfg.merge == "synthesize":
                raise ValueError(
                    "provider.node_ensemble.analyze.merge: 'synthesize' is not "
                    "supported — the analyze node parses its output as JSON, "
                    "but the synthesize merger emits markdown. Use "
                    "merge: tournament for analyze (keeps one model's verbatim "
                    "JSON), or move synthesize to a markdown-emitting node."
                )
        return self


MergeStrategy = Literal["tournament", "synthesize", "vote"]


class NodeEnsembleConfig(BaseModel):
    """Per-node ensemble setting — N parallel chat calls + merger.

    Used by the engine to fan out a node's LLM call across multiple
    models and merge into one final response. See ``core/ensemble.py``
    for the primitives.

    Constraints enforced at config load:

    * ``vote`` is only meaningful for nodes whose prompt emits a small
      structured field the merger can tally on (today that's
      ``cross_check`` — the engine extracts the ``verdict`` key per
      finding). Other nodes still accept ``vote`` but should consider
      ``tournament`` instead since the engine's vote path is keyed on
      ``"verdict"``.
    * ``synthesize`` produces freeform markdown by design — it MUST
      NOT be paired with nodes whose downstream parser expects JSON.
      ``analyze`` parses the merged text as JSON; the ProviderConfig
      validator rejects ``analyze.merge = synthesize`` with a clear
      error pointing the user at ``tournament`` instead.
    """
    models: list[str] = Field(..., min_length=1)
    merge: MergeStrategy = "tournament"
    # Which model performs the merge step. Required for ``tournament``
    # and ``synthesize`` (those merge strategies make an LLM call).
    # Ignored for ``vote`` (pure tally — no LLM call). Falls back to
    # the first model in ``models`` when None for tournament/synthesize.
    moderator: str | None = None


ClarifyMode = Literal["off", "auto", "interactive"]


class EngineConfig(BaseModel):
    framework: EngineFramework = "langgraph"
    max_iterations: int = 2
    review_loop: bool = True
    # Pre-flight clarification (`clarify` node before `ideate`).
    #
    #   "off"         — skip the node entirely. Default for tests and fleet.
    #   "auto"        — agent generates the questionnaire AND auto-answers
    #                   it from the topic alone (no human loop). Useful
    #                   when you want stronger framing than the bare topic
    #                   gives, without an interactive pause.
    #   "interactive" — agent generates the questionnaire; the engine
    #                   raises a LangGraph `interrupt()` so a CLI prompt
    #                   (`launch.py --interactive`) or the web UI's
    #                   clarify panel can collect answers from the user.
    clarify_mode: ClarifyMode = "off"
    # Execute-repair loop (`execute_reflect` node between
    # `execute` and `analyze`). When the experiment script returns
    # rc!=0 or no RESULT_JSON, the reflect node generates a patched
    # `code` from the traceback and routes back to `execute`, up to
    # this many times. Default 3 — covers typo / import-error / shape
    # mismatch / NaN bugs without runaway looping.
    exec_reflect_max_iterations: int = Field(default=3, ge=0)
    # Cross-paper check after analyze. When a finding lands,
    # we run a literature search keyed on the finding text (not just
    # the original topic) and classify hits as supporting / conflicting
    # / neutral. Per-finding top-K passed to the knowledge layer.
    cross_check_per_finding_k: int = Field(default=3, ge=0)
    # Also enables the `next_step` re-route emitted by analyze:
    # `publish` (default) / `re_experiment` / `broaden_lit`. Both
    # non-default values route back to `design` (via cross_check first)
    # and consume from `max_iterations` like the review-loop `revise`.
    enable_analyze_reroute: bool = True
    # Extra LLM call after `ideate` to self-critique the
    # chosen idea. May swap `chosen_idea` to a different entry from
    # the brainstormed list. Cheap (one extra LLM call per quest).
    ideate_reflect: bool = True
    # Pairwise tournament across the brainstormed ideas.
    # When True, REPLACES the single ``ideate_reflect`` critique call
    # with ``C(N, 2)`` parallel pairwise comparisons (3 calls for the
    # default 3 ideas) and picks the highest-win-count idea. Borrowed
    # from Google Co-Scientist's pairwise+Elo ranking pattern. Costs
    # 2 extra LLM calls vs. the single critique but produces a
    # measurably better-ranked ``chosen_idea``
    # on topics where the brainstormed alternatives are close in
    # quality. Off by default to keep the cost floor at 7-18 calls.
    # When True, ``ideate_reflect`` is ignored (tournament wins).
    ideate_tournament: bool = False
    # Multi-persona reviewer panel on the `review` node.
    # When empty (default), the review node behaves as before: one LLM
    # call producing one verdict. When non-empty, each entry runs in
    # parallel with a persona-specific system prefix prepended to
    # `agents/review.md`; a moderator LLM call then synthesizes a
    # single verdict (median score, union of weaknesses, intersection
    # of strengths, conservative verdict). Per-persona model selection
    # uses `provider.node_models["review_panel.<name>"]`.
    #
    # Built-in persona names: "methodologist", "statistician",
    # "devil_advocate", "reproducibility". Users can pass custom names
    # — the engine falls back to a generic persona prefix when the
    # specific persona prompt file is missing.
    review_panel: list[str] = Field(default_factory=list)
    # When True, the engine treats the topic as one that needs
    # real-world data the user will collect themselves — not one the
    # LLM should simulate via a Python experiment. The flow becomes:
    # clarify → ideate → literature → design → wait_for_data, at which
    # point the engine writes ``<quest_root>/data/README.md``
    # explaining what to drop into the dir and exits cleanly (rc=0).
    # The user collects data, drops it into ``<quest_root>/data/``,
    # then re-runs with ``fi --resume <quest_id>``. The engine picks
    # up at the ``data_load`` node which walks the dir (reusing
    # ``core/summarizer.py`` patterns), synthesizes a result_json,
    # and hands off to analyze → cross_check → write → review.
    #
    # When unset (default False), the clarify node may STILL set the
    # in-state ``no_simulation_resolved`` flag from its
    # ``empirical_vs_theoretical`` answer — auto-detection from
    # clarify is the second entry point. YAML flag wins when set.
    no_simulation: bool = False
    # When the engine enters no-simulation mode, an
    # ``auto_collect_data`` node runs BEFORE ``wait_for_data`` and
    # tries to pull relevant docs from the Knowledge layer (Axon).
    # Successful hits land under ``<quest_root>/data/auto_collected/``
    # as one ``.md`` file per doc (content + a YAML metadata header).
    # ``wait_for_data`` then sees those files via its existing rglob
    # walk and proceeds without pausing — so a no-simulation quest
    # that COULD be answered from Axon never blocks for user input.
    #
    # When False, the auto_collect_data node is a passthrough — the
    # engine pauses for user-supplied data exactly like before. Also
    # passes through if ``knowledge.enabled`` is False (no Axon to
    # query) — no need to set this False just to silence the node.
    #
    # The fallback to the manual pause is preserved: if Axon returns
    # zero hits, ``wait_for_data`` still writes the README and pauses
    # for the user just like the no-Axon case.
    auto_collect_data: bool = True
    # Top-K passed to ``Knowledge.asearch`` from the auto_collect_data
    # node. 5 is a deliberate ceiling — the data_load LLM call has a
    # finite content budget, and N=5 with one ~1k-token file each
    # comfortably fits a 16k-context prompt. Bump only when running on
    # a long-context model AND when the topic genuinely benefits from
    # more breadth (most don't — the top 5 from a good retriever
    # cover the space already). ``ge=1`` because passing top_k=0 to
    # Axon would request "zero hits" — a useless config that should
    # be a YAML error not silent passthrough.
    auto_collect_top_k: int = Field(default=5, ge=1)
    # Structured dataset adapters that run alongside the
    # Axon retrieval in the auto_collect_data node. Each name in the
    # list is looked up in ``core.datasets.ADAPTER_REGISTRY``;
    # unrecognized names are logged as a WARNING and skipped. Empty
    # list (default) means "Axon only" — Axon-only auto-collect.
    # Available names: ``"worldbank"``.
    #
    # Adapters write into ``<quest_root>/data/auto_collected/<source>/``
    # subdirectories so the manifest renders them grouped by origin
    # (and the user can spot which dataset contributed what to the
    # paper at a glance). Each adapter is best-effort — errors fall
    # through silently to the safety net (``wait_for_data`` pause).
    dataset_adapters: list[str] = Field(default_factory=list)
    # Per-adapter top_k. The Axon top_k stays at ``auto_collect_top_k``
    # above; this separate knob is for the dataset adapters because
    # they tend to hit external APIs and a smaller default is the
    # right cost trade-off. 3 indicators × a few countries each =
    # enough comparative baseline for most no-simulation quests.
    dataset_adapter_top_k: int = Field(default=3, ge=1)
    # User-pinned answers to the clarify slots, supplied by the
    # interview frontends (`launch.py --new`, the VSCode `@fi /new`
    # extension, the `--serve` web UI) or hand-written into YAML. When
    # present, the clarify node merges these into ``clarify_answers``
    # AFTER it computes its own auto-answers, so user pins always win.
    # When every clarify slot is pinned AND ``clarify_mode == "auto"``,
    # the clarify LLM call is skipped entirely (cost saving). When the
    # mode is ``interactive`` the LLM still fires so the human can
    # see the agent's questions, but pinned slots come pre-answered.
    # Keys are the clarify slot names — ``comparative_baseline`` /
    # ``empirical_vs_theoretical`` / ``simulatability`` /
    # ``success_metric`` / ``budget`` / ``output_kinds`` /
    # ``study_depth`` / ``paper_venue``. Values are whatever shape the
    # downstream prompts expect (strings, lists for output_kinds).
    clarify_overrides: dict[str, Any] = Field(default_factory=dict)


class ExecutionConfig(BaseModel):
    sandbox: SandboxKind = "venv"
    timeout_s: int = 60 * 30
    python_version: str = "3.11"
    docker_image: str = "python:3.11-slim"


class KnowledgeConfig(BaseModel):
    """Wraps the Axon knowledge layer.

    `axon_config` may be either an inline dict (passed straight to
    `AxonConfig.model_validate(...)`) or a path to an existing Axon YAML
    (`AxonConfig.from_yaml(path)`).
    """

    enabled: bool = True
    axon_config: Path | dict[str, Any] | None = None
    # Axon (RAG) retrieval cap. Small-k because dense embedding hits
    # are precise — 8 strong matches beat 20 medium ones for the writer
    # prompt. The literature node uses this when querying Axon.
    top_k: int = Field(default=8, ge=1)
    # External (arxiv / openalex / crossref / s2 / ...) retrieval cap.
    # Bigger than ``top_k`` because web search returns coarser matches
    # and the writer benefits from breadth — 20 retrieved abstracts
    # still fits comfortably in a prompt and a literature scan
    # genuinely needs that many hits. Set independently of ``top_k`` so
    # users can tune RAG precision without starving web breadth.
    external_top_k: int = Field(default=20, ge=1)
    # Pause-for-user-papers gate. When True, the literature node pauses
    # after retrieval IF any retrieved doc came back as abstract-only
    # (no full text available — typical for paywalled / Crossref / S2
    # metadata-only hits). Writes a ``needs/<slug>.json`` stub per
    # missing paper and creates ``inputs/papers/`` for the user to drop
    # downloaded PDFs into. On ``fi --resume``, the node walks
    # ``inputs/papers/``, indexes the files into the literature list,
    # and proceeds — giving the writer real full text instead of
    # abstracts.
    #
    # Default ``False`` so existing quests keep running unattended.
    # Pairs naturally with the ``knowledge.try_fetch_full_text`` knob:
    # turn that on first, then this gate only fires for the subset of
    # papers the host couldn't open-text-fetch.
    pause_for_user_papers: bool = False
    write_back_quests: bool = True
    # Ordered list of external literature sources used by
    # `Knowledge.search()` when Axon is disabled OR returns zero results.
    # Results from all listed sources run in parallel, are merged and
    # de-duplicated (by DOI / arXiv-id / PMID / normalized title), then
    # truncated to top_k. Source-list order is the dedup priority.
    # Override per-quest in YAML; set to `[]` or "none" to disable.
    #
    # Supported source names:
    #   openalex         — broadest single open index (~200M works, all fields, free).
    #   arxiv            — physics / CS / math / quant-ph / q-bio / stats (free).
    #   crossref         — DOI metadata across paywalled publishers (free).
    #   semantic_scholar — broad coverage + citation graph (free, rate-limited).
    #   pubmed           — biomedical via NCBI E-utilities (free).
    #   core             — 240M open-access papers (free; needs CORE_API_KEY env).
    #   google_scholar   — EXPERIMENTAL via `scholarly` package; no official API;
    #                      rate-limited / blocked by Google. Prefer openalex / s2.
    external_fallback: list[str] | str = Field(
        default_factory=lambda: ["openalex", "arxiv", "crossref"]
    )
    # How the literature node picks which external sources to query:
    #   "auto"   — agent asks the LLM to choose sources from the
    #              catalog (built-in + Axon-ingested supplements) given
    #              the current topic. Falls back to `external_fallback`
    #              if the LLM call fails or returns no valid sources.
    #   "manual" — use `external_fallback` verbatim, no LLM routing.
    source_routing: Literal["auto", "manual"] = "auto"
    # When True, seed the built-in source catalog into Axon at engine
    # construction so it's queryable as `kind="fi_source_catalog"`
    # knowledge later (lets users add their own venue/journal entries
    # for paywalled sources the agent should know about but FI can't
    # programmatically search, e.g. SPIE, IEEE Xplore).
    seed_source_catalog: bool = True
    # Files (PDF / Markdown / plain text) the user has manually placed
    # for this quest — e.g. paywalled PDFs they downloaded from SPIE /
    # IEEE / their institutional library. These are loaded at engine
    # construction and PINNED to the head of every literature-retrieval
    # result for this quest. If Axon is enabled, they are ALSO ingested
    # permanently as `kind="fi_local_paper"` so future quests find them.
    # Paths may be `~`-prefixed; globs are NOT expanded (pass each file).
    local_papers: list[Path] = Field(default_factory=list)
    # Phase 2 paywall-access support: when True, after external retrieval
    # returns Crossref/OpenAlex/etc. hits, FI opportunistically tries to
    # GET the publisher PDF using whatever network access the host has
    # (institutional VPN / Shibboleth / EZproxy already authenticated at
    # the OS level). Login-wall HTML pages are rejected by a
    # Content-Type + %PDF-magic check, so the quest never hangs on a
    # paywalled venue. Off by default — opt-in per quest.
    try_fetch_full_text: bool = False
    # Per-doc HTTP timeout (landing-page GET, PDF GET). Short is good.
    full_text_fetch_timeout_s: float = 15.0
    # Total budget across all docs in one literature batch — caps wall
    # time so a slow VPN can't stall the literature node indefinitely.
    full_text_fetch_total_s: float = 90.0
    # Hard cap on extracted text per doc so a 200-page review doesn't
    # blow up downstream prompts. Truncates the middle.
    full_text_max_kb: int = 64

    @field_validator("local_papers", mode="before")
    @classmethod
    def _expand_local_papers(cls, v: object) -> object:
        if v is None:
            return []
        if isinstance(v, (str, Path)):
            v = [v]
        return [_expand(p) for p in v]

    @field_validator("external_fallback", mode="before")
    @classmethod
    def _normalize_fallback(cls, v: object) -> object:
        # Accept "none" / "" / a single string / a list. Always
        # normalize to list[str]. Empty list = disabled.
        if v in (None, "", "none"):
            return []
        if isinstance(v, str):
            return [v]
        return v
    # When True (default), only quests whose final review verdict is
    # "accept" are written back into Axon's long-term store. When False
    # (and `write_back_quests` is True), every finished quest is
    # ingested regardless of verdict — useful while building up an
    # initial corpus or for debugging.
    write_back_only_on_accept: bool = True

    @field_validator("axon_config", mode="before")
    @classmethod
    def _expand_axon_path(cls, v: object) -> object:
        # Path-shaped input (str or Path) gets `~` expansion; dicts pass
        # through untouched so they reach AxonConfig.model_validate(...).
        if isinstance(v, (str, Path)):
            return _expand(v)
        return v


class OutputConfig(BaseModel):
    kinds: list[OutputKind] = Field(default_factory=lambda: ["paper_md", "paper_pdf"])
    paper_format: PaperFormat = "generic"
    output_dir: Path = Path("./outputs")
    # When True AND ``paper_pdf`` is in ``kinds``, treat a failed PDF
    # compile as a hard quest failure rather than a graceful skip.
    # Pairs with the engine's pre-flight check: at ``Engine.run``
    # startup the engine verifies pandoc + a LaTeX engine are
    # reachable BEFORE running any LLM calls. If a prerequisite is
    # missing and require_pdf is True, the run errors out
    # immediately (~saves ~15 min of LLM cost on a doomed quest); if
    # require_pdf is False the engine logs a warning and continues
    # so the user still gets paper.md + the
    # ``paper_pdf_skipped.md`` diagnostic.
    require_pdf: bool = False
    # Who will read this paper? Drives which Axon entries are eligible
    # to appear in the References section.
    #
    #   "external" (default) — the paper is written for an audience
    #       outside the user's organization (a journal, a venue, the
    #       open web). The writer must NOT cite FI-internal entries
    #       (kind=fi_critique / fi_digest / fi_portfolio / fi_proposal
    #       / fi_summary / fi_source_catalog), because those are
    #       cross-quest memory artifacts, not public sources an
    #       external reader could look up. Entries from the open
    #       literature (arxiv / openalex / crossref / semantic_scholar
    #       / pubmed / core) and ``fi_local_paper`` entries that
    #       carry a real DOI/URL are kept.
    #
    #   "internal" — the paper is an internal-facing write-up
    #       (a project report, an onboarding doc, a memo). Everything
    #       in Axon is fair game; FI-internal cross-quest summaries
    #       and proposals can be cited as prior work.
    #
    # Default is "external" because that's the safer default for a
    # one-shot paper produced by an automated pipeline.
    audience: Literal["external", "internal"] = "external"

    @field_validator("output_dir", mode="before")
    @classmethod
    def _expand_output(cls, v: object) -> object:
        return _expand(v)


class Config(BaseModel):
    topic: str
    title: str | None = None
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    engine: EngineConfig = Field(default_factory=EngineConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    knowledge: KnowledgeConfig = Field(default_factory=KnowledgeConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    extra_directives: str = ""

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(data)
