"""Typed configuration for Frontier Insight runs (post-DS redesign).

The schema is intentionally narrow — anything that varies per run goes in
YAML, anything that's a code-level concern stays in Python. Path fields
expand `~` on load via mode='before' validators.
"""

from __future__ import annotations

import logging
import os
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
# How paper.pdf is rendered. ``latex`` (default) = the venue LaTeX template
# (Computer Modern article). ``briefing`` = the Frontier Insight
# "Research Briefing" look (warm paper, teal accents, serif display)
# rendered via the HTML/Chromium PDF backend — same brand identity as the
# slides + poster. See OutputConfig.paper_style.
PaperStyle = Literal["latex", "briefing"]


def _expand(v: object) -> object:
    if isinstance(v, str):
        return Path(v).expanduser()
    if isinstance(v, Path):
        return v.expanduser()
    return v


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _offline_env_default() -> bool:
    """``knowledge.offline`` default — on when ``FI_OFFLINE`` is truthy."""
    return _truthy_env("FI_OFFLINE")


def _models_dir_env_default() -> Path | None:
    """``knowledge.models_dir`` default — from ``FI_MODELS_DIR`` if set."""
    v = os.environ.get("FI_MODELS_DIR")
    return Path(v).expanduser() if v else None


class ProviderConfig(BaseModel):
    name: ProviderName = "codex"
    model: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
    # Wall-clock budget (seconds) for a single chat call on transports
    # that lack a built-in per-request deadline. Applies to:
    #
    # - CLI providers (claude_cli/codex_cli/copilot_cli/gemini_cli):
    #   a stuck child process is killed and tenacity retries.
    # - vscode_extension: a stalled streaming response from
    #   ``model.sendRequest`` (Copilot HTTP/2 hang) surfaces as
    #   ``BridgeError("bridge stalled ...")`` which the transient-marker
    #   list catches → tenacity retries.
    #
    # Default 300 s — longer than the typical 10-90 s per call but
    # bounded so a silent stall can't wedge the engine indefinitely.
    # HTTP providers use their own httpx-level timeout (``timeout_s``
    # on :class:`LLMClient`) and ignore this field.
    cli_timeout_s: float = Field(default=300.0, gt=0)
    # Soft inactivity watchdog for the streaming reader. Resets on
    # every stdout line (or stream-json event); the child is killed if
    # this many seconds pass with no output even when the overall
    # ``cli_timeout_s`` budget hasn't expired. Distinguishes "model is
    # thinking — events flowing, no answer text yet" (events keep
    # ticking) from "process is hung" (no events at all). Set to None
    # to disable and rely only on the total ceiling. Default 180 s is
    # generous enough that complex Sonnet 4.6 ``thinking_delta``
    # streams don't false-positive, while still catching real hangs in
    # roughly 3 minutes.
    cli_inactivity_timeout_s: float | None = 180.0
    # Per-node ``cli_timeout_s`` override map. Reasoning-heavy nodes
    # (``implement``, ``execute_reflect``) routinely outrun the 300 s
    # default on complex topics under Sonnet 4.6 extended thinking;
    # bumping them here avoids tenacity retries that all hit the same
    # wall. Cheap nodes (``clarify``, ``ideate``) keep the default —
    # tightening their budget catches real hangs faster.
    #
    # Maps engine-node name → seconds. Lookup misses fall back to
    # ``cli_timeout_s``. The default carries explicit budgets for the
    # two nodes that have actually hit ``_CliTransientError`` in
    # production; everything else stays on the historical 300 s.
    node_cli_timeout_s: dict[str, float] = Field(
        default_factory=lambda: {
            # Outline is structurally bounded (signatures + constants),
            # so it doesn't need the full implement budget; 600 s is
            # generous for a ~30-80 line scaffold + JSON envelope.
            "implement_outline": 600.0,
            # ``implement`` covers BOTH the two-stage body call (which
            # consumes the outline so it typically finishes in 90-180s)
            # AND the legacy single-shot fallback path (pre-Phase-2
            # checkpoint resume, or when the outline call failed — this
            # is where the OPC quest's ~9 min extended-thinking span
            # lives). Sized for the WORST CASE so the fallback path
            # doesn't regress vs Phase 1's behaviour; the body call's
            # well-behaved short spans don't pay any cost since they
            # finish well before the ceiling.
            "implement": 1800.0,
            "execute_reflect": 900.0,
            "design_self_critique": 900.0,
            # web_plots writes a short matplotlib script — it should be
            # quick. A tight ceiling kills a stuck codex_cli call fast
            # instead of letting it dribble output past the default budget.
            "web_plots": 180.0,
        }
    )
    # Per-node MODEL ESCALATION on tenacity retry. Maps an engine
    # node name to a fallback model string. ``_chat_cli`` uses the
    # primary (per-node or endpoint) model on attempt 1; if that
    # raises ``_CliTransientError``, attempt 2+ uses the fallback.
    #
    # The primary motivation is empirical: Sonnet 4.6 on long
    # code-gen prompts (implement_body, write) goes into extended-
    # thinking and never produces text — 15-minute runaway with zero
    # output, reproduced 3× across prompt-size and output-shape
    # variants. Opus 4.7 lands the same prompt in ~5 minutes. By
    # escalating to Opus only on retry, we honour the user's
    # configured primary model (cheaper, often sufficient) on first
    # try, then degrade gracefully when it transient-fails.
    #
    # Default escalates the two demonstrably-paralysis-prone nodes
    # (implement, write) to ``claude-opus-4-7``. Cheap nodes stay
    # on the user's primary because retry-escalation has a cost
    # cliff and they rarely transient-fail. Set ``{}`` to disable
    # escalation entirely.
    node_model_fallbacks: dict[str, str] = Field(
        default_factory=lambda: {
            "implement": "claude-opus-4-7",
            "write": "claude-opus-4-7",
        }
    )
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
HumanFeedbackGate = Literal["off", "after_review"]
# Pause-drop-anytime points. The engine pauses with an interrupt at the
# selected stage(s); the user drops files into ``<quest_root>/inputs/papers/``
# (PDFs/MDs, ingested by the literature node) and/or ``<quest_root>/inputs/data/``
# (CSVs/datasets, picked up by analyze on resume), then runs
# ``fi --resume <quest_id>``. "never" (default) keeps today's flow.
PauseForUserInput = Literal[
    "never", "after_design", "after_paper", "both",
]

# ---------------------------------------------------------------------------
# Unified human-in-the-loop ("pauses") namespace.
#
# Every place the quest can stop and wait for a human is one of two
# interactions — ANSWER (the quest asks; you reply) or SUPPLY (the quest
# needs files; you drop them in) — and all of them are configured under one
# coherent ``pauses:`` section. The legacy flags (engine.clarify_mode,
# engine.human_feedback_gate, engine.pause_for_user_input,
# engine.auto_accept_on_pass, engine.human_feedback_timeout_s,
# knowledge.pause_for_user_papers) still parse and are merged into ``pauses``
# by Config's root validator, so existing YAML keeps working.
ClarifyPause = Literal["off", "auto", "ask"]       # was clarify_mode (interactive→ask)
ReviewPause = Literal["off", "ask"]                # was human_feedback_gate (after_review→ask)
SupplyPause = Literal[                              # was pause_for_user_input
    "never", "before_build", "before_review", "both",
]
# Legacy value → canonical value, applied field-by-field in PausesConfig.
_CLARIFY_ALIASES = {"interactive": "ask"}
_REVIEW_ALIASES = {"after_review": "ask"}
_SUPPLY_ALIASES = {"after_design": "before_build", "after_paper": "before_review"}

# Legacy flag location (section, old key) → canonical ``pauses`` field. Read by
# Config's root validator so old YAML keeps working without touching read sites.
_LEGACY_PAUSE_FLAGS = (
    ("engine", "clarify_mode", "clarify"),
    ("engine", "human_feedback_gate", "review"),
    ("engine", "pause_for_user_input", "supply"),
    ("engine", "auto_accept_on_pass", "auto_accept_on_pass"),
    ("engine", "human_feedback_timeout_s", "timeout_s"),
    ("knowledge", "pause_for_user_papers", "papers"),
)


class PausesConfig(BaseModel):
    """When (and how) the quest stops to involve a human. See the README
    section *When the quest needs you* for the full model."""

    # ANSWER — pre-flight clarifying questions before the research starts.
    #   "off"  skip · "auto" agent self-answers · "ask" pause for the human.
    clarify: ClarifyPause = "off"
    # SUPPLY — pause when a relevant paper came back abstract-only (paywalled)
    # so the user can download it and drop the PDF into inputs/papers/.
    papers: bool = False
    # SUPPLY — fixed drop-in checkpoint(s) for papers/data.
    #   "never" · "before_build" (after design) · "before_review" (after the
    #   first draft) · "both".
    supply: SupplyPause = "never"
    # ANSWER — pause after the review verdict for accept / reject / refine.
    #   "ask" (default) pauses · "off" lets the review-loop drive unattended.
    review: ReviewPause = "ask"
    # At the review pause, auto-accept *clean* papers (verdict==accept AND no
    # must-flag hits) instead of waiting — handy for --fleet.
    auto_accept_on_pass: bool = False
    # Max seconds to wait at an ANSWER pause for a wired callback (web POST /
    # VSCode pick) before falling back to the headless pause-exit path. 0 =
    # wait forever.
    timeout_s: float = Field(default=1800.0, ge=0)

    @field_validator("clarify", mode="before")
    @classmethod
    def _norm_clarify(cls, v: Any) -> Any:
        # YAML 1.1 "Norway problem": an unquoted `off` parses as the boolean
        # False before we see it — coerce it back so hand-edited `pauses:`
        # blocks don't need to remember the quotes.
        if v is False:
            return "off"
        return _CLARIFY_ALIASES.get(v, v)

    @field_validator("review", mode="before")
    @classmethod
    def _norm_review(cls, v: Any) -> Any:
        if v is False:
            return "off"
        return _REVIEW_ALIASES.get(v, v)

    @field_validator("supply", mode="before")
    @classmethod
    def _norm_supply(cls, v: Any) -> Any:
        return _SUPPLY_ALIASES.get(v, v)


class EngineConfig(BaseModel):
    framework: EngineFramework = "langgraph"
    max_iterations: int = Field(default=2, ge=0)
    review_loop: bool = True
    # Human-feedback gate. ``"after_review"`` (the default) pauses
    # the quest AFTER the review node fires and waits for the user
    # to accept / reject / refine the result before finalising. The
    # ``after_review`` callback receives the review verdict + scores +
    # paper md + any must-flag hits and returns one of:
    #
    #   {"action": "accept"}                     # finalise the quest as-is
    #   {"action": "reject"}                     # finalise with verdict=rejected
    #   {"action": "refine", "feedback": "..."}  # bump iteration; back to design
    #                                            #   with feedback injected
    #
    # ``"off"`` skips the gate and lets the engine's review-loop
    # verdict drive revise/done routing without a pause. Useful for
    # fully unattended runs where downstream review happens out-of-band.
    # Headless invocations (no callback wired) get an automatic
    # pause-exit + on-disk snapshot at ``<quest_root>/.fi/human_review.json``
    # instead of blocking forever — drop a response at
    # ``human_review_answer.json`` and ``--resume`` to continue.
    human_feedback_gate: HumanFeedbackGate = "after_review"
    # When True, the human-review gate auto-resumes with ``accept`` for
    # *clean* papers — verdict == "accept" AND ``must_flag_hits`` is
    # empty. Flagged or revise-verdict papers still pause for human
    # review. Use with ``--fleet`` so the production runner doesn't
    # block on every clean quest, and in tests so e2e fixtures finalise
    # without wiring a callback. The CLI flag ``--auto-accept-on-pass``
    # forces this to True at the launcher level.
    auto_accept_on_pass: bool = False
    # Max seconds to wait at the human-review gate for a wired callback
    # (web POST, VSCode QuickPick) to return. When it elapses, the engine
    # stops waiting and falls back to the headless path: it checks for an
    # on-disk ``human_review_answer.json`` and, failing that, pause-exits
    # cleanly (rc=0, checkpoint saved, resume hint logged) — so an orphaned
    # UI prompt can never park a quest forever. ``0`` disables the timeout
    # (wait indefinitely, legacy behaviour). Has no practical effect on the
    # CLI ``--interactive`` gate, whose blocking ``input()`` implies a human
    # is present.
    human_feedback_timeout_s: float = Field(default=1800.0, ge=0)
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
    # Degenerate-run guard. When True (default), a script that exits 0
    # but emits a RESULT_JSON whose numeric metrics are ALL ~0 is treated
    # as a soft failure: the execute-repair loop gets a chance to fix the
    # underlying bug (wrong grid/sampling, a threshold/edge finder
    # returning a zero sentinel, bad normalization) before a paper is
    # written, bounded by ``exec_reflect_max_iterations``. If the result
    # is still degenerate after repair, the quest proceeds with a
    # ``degenerate_result`` flag so analyze/review frame it as a failure
    # note. Set False to accept all-zero results as-is.
    degenerate_run_guard: bool = True
    # Claim grounding (claim_check node, between write and review). When
    # True (default), one LLM pass extracts the paper's substantive claims and
    # grounds each as `experiment` (traces to the run's result_json / a key
    # finding), `citation` (traces to a specific [N] reference), or
    # `unsupported`. A `paper/claims.json` + `paper/CLAIMS.md` ledger is
    # written, and any unsupported claims are surfaced to the reviewer, which
    # must-flags them and forces a bounded revise pass (capped by
    # `max_iterations`). Set False to make the claim_check node a no-op
    # passthrough (no LLM call, no ledger) — the graph edge stays.
    claim_grounding: bool = True
    # Evidence-sufficiency gate (evidence_gate node, between cross_check
    # and write). When True (default), one LLM pass weighs the assembled
    # evidence — source count, cross-check supporting/conflicting balance,
    # whether the run produced real measurements — against the research
    # question and returns a verdict: ``sufficient`` (write), ``broaden``
    # (re-enter the literature loop once more, budget permitting), or
    # ``insufficient`` (write anyway, but the assessment is handed to the
    # writer so the paper frames thin evidence honestly instead of
    # over-claiming). Fails OPEN: any parse error, or the flag off, routes
    # straight to write. The broaden loop is bounded by
    # ``evidence_gate_max_broaden`` so it can't spin.
    evidence_gate: bool = True
    evidence_gate_max_broaden: int = Field(default=1, ge=0)
    # Cross-paper check after analyze. When a finding lands,
    # we run a literature search keyed on the finding text (not just
    # the original topic) and classify hits as supporting / conflicting
    # / neutral. Per-finding top-K passed to the knowledge layer.
    cross_check_per_finding_k: int = Field(default=3, ge=0)
    # CoVe-style verification pass on each finding's first-pass
    # classification. A second LLM call (per finding) asks pointed
    # verification questions about each supporting/conflicting
    # assignment, then revises the classification — downgrading
    # over-claimed supports to neutral, flipping sign-errors, etc.
    # Cost: one extra LLM call per finding (typically 1–5 findings,
    # so +1–5 calls per quest). Off by default to keep the cost
    # baseline lean; enable for higher-stakes papers where the
    # citation-direction risk matters. When the first pass produces
    # no non-neutral assignments, the verification call is skipped.
    cross_check_verify: bool = False
    # Multi-seed replication. When > 1, the execute node runs the
    # generated experiment script N times with different seeds (the
    # env var ``FI_REPLICATE_SEED`` is set to a fresh integer per
    # run, which the implement prompt instructs the script to honour
    # for its own rng seeding). The N ``RESULT_JSON`` outputs land
    # in ``state['result_json_replicates']`` as a list; the analyze
    # node aggregates numeric fields with mean ± std and surfaces
    # the spread in its summary. Cost: N executions per design pass.
    # Default 1 (no replication, current behaviour).
    execute_replicates: int = Field(default=1, ge=1)
    # Generic mid-quest user-input pause point. When set, the engine
    # pauses (LangGraph ``interrupt()``) AFTER the named stage and
    # exits cleanly with rc=0; the user drops files into
    # ``<quest_root>/inputs/papers/`` (added to the literature on
    # the literature node's next pass) and/or
    # ``<quest_root>/inputs/data/`` (read by analyze as
    # ``user_supplied_datasets``), then runs ``fi --resume <quest_id>``.
    # A disk marker at ``<quest_root>/.fi/paused_at_<stage>.flag``
    # tracks which stages have already paused so a resume doesn't
    # loop the user through the same drop screen twice.
    pause_for_user_input: PauseForUserInput = "never"
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
    # Local-data-first mode for the ``--analyze`` workflow: the user has
    # already supplied the data, so the framing/retrieval nodes (ideate,
    # literature, design) are made PASSTHROUGHS — no ideation, no external
    # retrieval, no experiment design — and the quest goes straight from
    # clarify to loading + analysing the local data. Implies no_simulation.
    # Off for normal quests. (Set by ``analyze_cli.build_analyze_config``;
    # also disable retrieval at the knowledge layer — ``web_search: false``
    # — so nothing reaches out to the web/academic sources.)
    analyze_local_first: bool = False
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
    # In no-simulation mode, derive figures from the web/Axon content the
    # collector gathered: a small data-extraction + matplotlib step turns
    # quoted numbers (e.g. a revenue-by-year table scraped from sources)
    # into charts saved under ``figures/``, each annotated with the source
    # it came from. The no-simulation path runs NO experiment code, so this
    # is the only way a literature-driven quest gets plots. Off → the
    # paper/poster/slides stay text + tables only. Honours the same
    # execution sandbox as the simulation path.
    web_derived_plots: bool = True
    # Hard wall-clock cap (seconds) on the whole web_plots node — its LLM
    # call + matplotlib install + plot-script run. It's best-effort, so on
    # timeout the quest just proceeds figure-less rather than hanging (a
    # stuck codex_cli call once wedged a run for hours when the
    # provider-level timeout didn't fire).
    web_plots_timeout_s: float = Field(default=360.0, gt=0)
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
    timeout_s: int = Field(default=60 * 30, gt=0)
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
    # Air-gapped / offline model loading. ``models_dir`` is a local
    # Hugging Face cache root (HF-cache layout) that holds the embedding
    # + reranker models — set it on machines with no network access so
    # Axon loads weights from disk instead of huggingface.co. ``offline``
    # forces ``HF_HUB_OFFLINE`` / ``TRANSFORMERS_OFFLINE`` (and Axon's
    # ``local_assets_only``) so a missing/flaky network can never trigger
    # a download (and the ``huggingface_hub`` closed-client crash) at
    # quest start. Both honour env fallbacks (``FI_MODELS_DIR`` /
    # ``FI_OFFLINE``) so all three surfaces (CLI/Web/VSCode) and the Axon
    # sidecar pick them up without per-quest YAML. Produce a shippable
    # ``models_dir`` on a connected machine via ``launch.py
    # --export-models <dir>``.
    models_dir: Path | None = Field(default_factory=lambda: _models_dir_env_default())
    offline: bool = Field(default_factory=lambda: _offline_env_default())
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
    # ---- General web search (Brave / DuckDuckGo) ----------------------
    # The academic adapters above (arxiv/openalex/crossref/...) only cover
    # scholarly literature; a non-academic question ("SpaceX revenue by
    # year", "Belgium vs Taiwan work culture") has no arXiv match and the
    # academic APIs return irrelevant nearest-neighbours. ``web_search``
    # adds a general-web layer that runs IN PARALLEL with Axon + academic
    # for every quest and is merged/deduped with them — so non-academic
    # topics get real sources instead of nonsense. Web hits carry their
    # URL/title/site as citation metadata and flow into the paper /
    # poster / slides References.
    web_search: bool = True
    # Backend selection. ``auto`` uses Brave when ``brave_api_key`` (or the
    # ``BRAVE_API_KEY`` env var) is present, else falls back to the keyless
    # DuckDuckGo HTML endpoint so web search still works with no config.
    # Pin to ``brave`` to require the key (disabled with a clear log when
    # missing) or ``duckduckgo`` to always use the keyless backend.
    web_search_backend: Literal["auto", "brave", "duckduckgo"] = "auto"
    # Brave Search API key. Env fallback ``BRAVE_API_KEY`` (the same var
    # Axon uses) so a single key serves both. Free tier ~2k queries/mo.
    brave_api_key: str = Field(
        default_factory=lambda: os.environ.get("BRAVE_API_KEY", "").strip()
    )
    # Number of web results to request per query. Independent of
    # ``external_top_k`` (which caps the merged external set) so users can
    # widen web breadth without changing the academic cap.
    web_search_top_k: int = Field(default=10, ge=1)
    # When True, fetch + extract the readable text of each web result's
    # page (not just the snippet) so the writer can quote real content and
    # the plot step can pull numbers out of it. Budgeted by the same
    # ``full_text_fetch_*`` knobs below. Off → snippets only.
    web_fetch_pages: bool = True
    # When a direct page fetch is blocked (HTTP 403 — common for
    # Cloudflare-protected authoritative sites like IEA) or returns nothing,
    # retry by rendering the page in a headless Chromium via Playwright,
    # which executes JS and clears most anti-bot challenges so the full
    # article text is recovered (the difference between a 2-sentence snippet
    # and a full report). Requires the optional dependency:
    # ``pip install playwright && playwright install chromium``. Degrades to
    # a no-op (snippet only) when Playwright/Chromium isn't installed, so
    # it's safe to leave on. Only fires on a blocked/empty direct fetch.
    headless_fetch: bool = True
    # Relevance guard. After retrieval/auto-collect, score the gathered
    # docs against the topic; if they are clearly off-topic or empty
    # (e.g. an academic-only collector returning physics preprints for a
    # finance question), pause for user-supplied sources / escalate rather
    # than letting analyze+write produce an honest-but-useless "pipeline
    # failure" paper on garbage. Off → legacy behaviour (proceed anyway).
    relevance_guard: bool = True
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
    full_text_fetch_timeout_s: float = Field(default=15.0, gt=0)
    # Total budget across all docs in one literature batch — caps wall
    # time so a slow VPN can't stall the literature node indefinitely.
    full_text_fetch_total_s: float = 90.0
    # Hard cap on extracted text per doc so a 200-page review doesn't
    # blow up downstream prompts. Truncates the middle.
    full_text_max_kb: int = 64
    # How many characters of each source's fetched full text to put into a
    # node's prompt. Selected by RELEVANCE to the quest question (see
    # ``passage_ranking``) — the *relevant* N chars, not the first N — so
    # the writer/plotter see the passages that actually carry the numbers,
    # not just the abstract. The full text always lands on disk under
    # ``data/literature/`` regardless of this budget.
    literature_excerpt_chars: int = Field(default=4000, ge=500)
    # How prompt excerpts are ranked against the quest question:
    #   "auto"    — hybrid when a sentence-transformer model loads, else
    #               lexical (default; skips the model under FI_OFFLINE),
    #   "hybrid"  — blend of embedding cosine + lexical term overlap,
    #   "embed"   — pure sentence-transformer cosine (lexical fallback),
    #   "lexical" — fast, dependency-free term overlap.
    passage_ranking: Literal["auto", "hybrid", "embed", "lexical"] = "auto"
    # Topic-figure enrichment (no-simulation path): pull a few LICENSE-CLEAN
    # illustrative figures — a real figure from a cited CC-licensed arXiv
    # paper, plus Wikimedia Commons diagrams under CC / public-domain — and
    # embed them in the paper/poster/slides with attribution, so a study
    # carries a visual point of view beyond its own charts. Off by default
    # (it fetches + embeds external images); only ever embeds free-licensed
    # images, each attributed to its source + license.
    fetch_web_figures: bool = False
    # Max illustrative figures to embed when ``fetch_web_figures`` is on.
    web_figures_max: int = Field(default=2, ge=0, le=6)

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

    @field_validator("models_dir", mode="before")
    @classmethod
    def _expand_models_dir(cls, v: object) -> object:
        # Explicit value present in input: `~`-expand it. Empty string
        # means "no local models dir" (overrides the FI_MODELS_DIR env
        # default). When the field is absent the default_factory reads
        # FI_MODELS_DIR, so this validator only runs on explicit input.
        if v in (None, ""):
            return None
        return _expand(v)


class OutputConfig(BaseModel):
    kinds: list[OutputKind] = Field(default_factory=lambda: ["paper_md", "paper_pdf"])
    paper_format: PaperFormat = "generic"
    # Visual style of paper.pdf. ``latex`` (default) keeps the venue LaTeX
    # template (Computer Modern article). ``briefing`` renders the Frontier
    # Insight "Research Briefing" look (warm paper, deep-teal accents, serif
    # display, brand mark) via the HTML/Chromium PDF backend — the same
    # editorial identity as the slides + poster. ``briefing`` needs pandoc +
    # a Chromium-family browser (Edge/Chrome/Chromium); when those are
    # unavailable it falls back to the LaTeX path with a warning. It can't
    # reproduce a venue's two-column class, so it always renders single
    # column regardless of ``paper_format``.
    paper_style: PaperStyle = "latex"
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
    # When no LaTeX engine (pdflatex/tectonic) is reachable, fall back to
    # an HTML/Chromium renderer: pandoc turns paper.md into a Computer-
    # Modern-styled HTML page (matching the LaTeX `article` look), then a
    # headless system browser (Edge/Chrome/Chromium) prints it to
    # paper.pdf. No LaTeX, no admin install — only pandoc (already needed)
    # + a browser, both of which a locked-down machine almost always has.
    # The fallback can't reproduce a venue's two-column LaTeX class, so it
    # always renders the single-column house style; set False to force the
    # strict LaTeX-only path (skip-with-diagnostic when no engine).
    html_pdf_fallback: bool = True
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
    pauses: PausesConfig = Field(default_factory=PausesConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    extra_directives: str = ""

    @model_validator(mode="before")
    @classmethod
    def _merge_legacy_pauses(cls, data: Any) -> Any:
        """Fold legacy scattered flags into the unified ``pauses`` section.
        An explicit ``pauses.<field>`` always wins; a legacy flag only fills a
        field the user didn't set under ``pauses``. Value translation
        (interactive→ask, after_design→before_build, …) happens in
        PausesConfig's field validators, so we copy values through verbatim."""
        if not isinstance(data, dict):
            return data
        pauses = dict(data.get("pauses") or {})
        used: list[str] = []
        for section, old_key, new_key in _LEGACY_PAUSE_FLAGS:
            if new_key in pauses:
                continue  # explicit pauses.<field> always wins
            src = data.get(section)
            if isinstance(src, dict):
                # YAML / dict path: only map a legacy key the user actually set.
                if old_key in src:
                    pauses[new_key] = src[old_key]
                    used.append(f"{section}.{old_key}")
            elif src is not None:
                # Programmatic path: ``engine``/``knowledge`` passed as a model
                # instance. Every legacy default equals the matching ``pauses``
                # default, so copying the attribute through is always safe
                # (no deprecation noise for in-process construction).
                val = getattr(src, old_key, None)
                if val is not None:
                    pauses[new_key] = val
        if pauses:
            data["pauses"] = pauses
        if used:
            logging.getLogger("frontier_insight.config").info(
                "config: legacy pause flag(s) %s mapped into the `pauses:` "
                "section; prefer `pauses.*` going forward.",
                ", ".join(sorted(set(used))),
            )
        return data

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(data)
