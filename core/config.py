"""Typed configuration for Frontier Insight runs (post-DS redesign).

The schema is intentionally narrow — anything that varies per run goes in
YAML, anything that's a code-level concern stays in Python. Path fields
expand `~` on load via mode='before' validators.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Literal, get_args

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
    "antigravity_cli",  # local Google Antigravity CLI (`agy`), prompt via its
                        # stream-json stdin so long nodes are not argv-capped
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
# Levels ``provider.reasoning_effort`` accepts. Each transport maps the level
# onto its own setting (a request-body field, a CLI flag) and skips, with a
# one-time warning, the levels or providers it cannot express — see
# ``core.provider``.
ReasoningEffort = Literal["minimal", "low", "medium", "high", "xhigh", "max"]
REASONING_EFFORT_LEVELS: tuple[str, ...] = get_args(ReasoningEffort)
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
    # Ordered provider names to fall back to when the primary provider
    # terminally fails a chat call (after its own in-provider retries) —
    # e.g. ``["codex_cli", "gemini_cli"]``. Empty (default) keeps the
    # unchanged single-provider behaviour. Each fallback is resolved
    # lazily on first use (its proxy, if any, spins up only when the
    # primary is actually failing) and gets its own circuit breaker:
    # after repeated failures — or one auth/quota error — it is skipped
    # for the rest of the run. See ``core.provider.FallbackLLMClient``.
    fallback: list[ProviderName] = Field(default_factory=list)
    # How hard the model should reason before answering. Unset (default)
    # sends nothing, so every provider keeps its own default — for claude
    # and agy their own setting, for codex its built-in default (FI's codex
    # calls do not read ``~/.codex/config.toml``), for Ollama no thinking at
    # all. When set:
    #
    # - HTTP providers (openai/codex/gemini/ollama/vllm): ``reasoning_effort``
    #   in the chat-completions body. Ollama accepts only low/medium/high
    #   and answers anything else with a 400, so those levels are skipped
    #   there with a warning.
    # - claude_cli: ``--effort`` (low/medium/high/xhigh/max).
    # - codex_cli: ``-c model_reasoning_effort="<level>"``.
    # - antigravity_cli: ``--effort`` (low/medium/high).
    # - copilot_cli, gemini_cli, vscode_extension and the proxy providers
    #   have no setting FI can pass; the level is not sent and a warning
    #   says so once.
    #
    # A fallback provider inherits the level and applies the same rules.
    reasoning_effort: ReasoningEffort | None = None
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
    # HTTP read-timeout budget (seconds) for the OpenAI-compatible transport
    # (openai/codex/gemini/ollama/vllm + proxy providers). ``http_timeout_s``
    # is the base; ``node_http_timeout_s`` overrides it per engine node.
    # Reasoning-heavy nodes routinely outrun a flat 120 s on slow/local
    # models, so give them explicit headroom while cheap nodes keep the tight
    # base that catches a hung server fast. Lookup misses fall back to the
    # base. HTTP only — CLI transports use ``node_cli_timeout_s``.
    http_timeout_s: float = Field(default=120.0, gt=0)
    node_http_timeout_s: dict[str, float] = Field(
        default_factory=lambda: {
            "implement_outline": 300.0,
            "implement": 900.0,
            "write": 600.0,
            "execute_reflect": 600.0,
            "implement_seed": 600.0,
            "design_self_critique": 600.0,
            "analyze": 300.0,
        }
    )
    # Last-resort prompt-size guard (total characters across all messages).
    # 0 (default) disables it — no behaviour change. When >0, a prompt that
    # exceeds this is truncated in the MIDDLE of its largest message (head +
    # tail kept) with a loud warning before dispatch, so a runaway prompt
    # (e.g. a giant RESULT_JSON or literature dump) can't blow the context
    # window into a hard 400 / stall. See ``LLMClient._trim_messages``.
    max_prompt_chars: int = Field(default=0, ge=0)
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
            # Returns the whole script again, like execute_reflect does.
            "implement_seed": 900.0,
            "design_self_critique": 900.0,
            # web_plots writes a short matplotlib script — it should be
            # quick. A tight ceiling kills a stuck codex_cli call fast
            # instead of letting it dribble output past the default budget.
            "web_plots": 180.0,
        }
    )
    # Per-node MODEL ESCALATION on tenacity retry — OPT-IN, empty by
    # default. Maps an engine node name to a fallback model string.
    # ``_chat_cli`` uses the primary (per-node or endpoint) model on
    # attempt 1; if that raises ``_CliTransientError``, attempt 2+ uses
    # the fallback for that node. Unset (or ``{}``), every attempt uses
    # the primary model.
    #
    # Why the feature exists: a smaller Claude model on a long code-gen
    # prompt (implement_body, write) can go into extended thinking and
    # never produce text — a 15-minute runaway with zero output,
    # reproduced 3× across prompt-size and output-shape variants.
    # Retrying the same prompt on a stronger model escapes it. Doing
    # that only on retry keeps the user's primary model (cheaper, often
    # sufficient) for the first try.
    #
    # Why it is empty by default: FI never picks the models a quest
    # pays for; the user does. The fallback is sent to whichever CLI
    # provider is active (claude_cli, codex_cli, gemini_cli,
    # copilot_cli, antigravity_cli), and a model name belongs to one
    # vendor — a built-in default would name a model most of them do
    # not offer, and would spend on a model the user never chose. To
    # opt in, name a model the ACTIVE provider accepts:
    #
    #     provider:
    #       node_model_fallbacks:
    #         implement: <your stronger model>
    #         write: <your stronger model>
    node_model_fallbacks: dict[str, str] = Field(default_factory=dict)
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
    #
    # `poster` / `slides` / `speech` are also valid keys: those three
    # generators pour an already-written paper into a fixed template
    # (columns, slide outline, narration) rather than doing open-ended
    # reasoning, so a cheaper model than the quest's primary is usually
    # just as good there. Unset, they use the primary model like any
    # other node — no behavior change until you opt in.
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

    @field_validator("reasoning_effort", mode="before")
    @classmethod
    def _normalise_reasoning_effort(cls, v: object) -> object:
        """Accept a level in any case with surrounding spaces, treat a blank
        string as unset, and reject anything else with the list of levels —
        a typo must fail at load, not silently run at the provider default."""
        if v is None:
            return None
        if isinstance(v, str):
            level = v.strip().lower()
            if not level:
                return None
            if level in REASONING_EFFORT_LEVELS:
                return level
        raise ValueError(
            "provider.reasoning_effort must be one of "
            + ", ".join(REASONING_EFFORT_LEVELS)
            + f", or left unset; got {v!r}"
        )

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

    @model_validator(mode="after")
    def _cli_timeout_is_a_floor(self) -> "ProviderConfig":
        """An explicit ``cli_timeout_s`` is a floor, not just a fallback.

        ``node_cli_timeout_s`` ships defaults that are *lower* than a raised
        ``cli_timeout_s`` for some nodes (``implement_outline`` at 600 s), and
        the provider resolves the per-node entry first. A quest that asked for
        900 s therefore got 600 s on that node with no warning, and a codex
        ``implement_outline`` call was killed three times in a row, losing about
        40 minutes and the outline stage.

        So when the user sets ``cli_timeout_s`` and leaves the per-node map at
        its defaults, every built-in entry below it is raised to it. A
        user-supplied ``node_cli_timeout_s`` is left exactly as written: both
        numbers were deliberate, and a deliberately tighter node budget is a
        legitimate way to catch a hang on that node faster.
        """
        if "cli_timeout_s" not in self.model_fields_set:
            return self
        if "node_cli_timeout_s" in self.model_fields_set:
            return self
        raised = {
            node: seconds
            for node, seconds in self.node_cli_timeout_s.items()
            if seconds < self.cli_timeout_s
        }
        if not raised:
            return self
        for node in raised:
            self.node_cli_timeout_s[node] = self.cli_timeout_s
        logging.getLogger("frontier_insight.config").info(
            "provider.cli_timeout_s=%gs is a floor: raised the built-in per-node "
            "budget for %s to match.",
            self.cli_timeout_s,
            ", ".join(f"{n} (was {s:g}s)" for n, s in sorted(raised.items())),
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
PlanPause = Literal["off", "ask"]                  # stop once plan.md is written, to read and edit it
SupplyPause = Literal[                              # was pause_for_user_input
    "never", "after_literature", "before_build", "before_review", "both", "all",
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
    # On by default: a relevant paywalled paper the open-access cascade could
    # not fetch is listed in needs/WANTED_PAPERS.md and the quest waits for it.
    papers: bool = True
    # SUPPLY — fixed drop-in checkpoint(s) for papers/data.
    #   "never" · "after_literature" (the literature is done and saved; the
    #   rest of the quest — skills, design, experiment — starts on resume with
    #   it in hand) · "before_build" (after design) · "before_review" (after
    #   the first draft) · "both" (before_build + before_review, as it always
    #   was) · "all" (all three).
    supply: SupplyPause = "never"
    # Stop after the plan step has written ``plan.md`` (what the literature says, the gap, the hypothesis,
    # the experiment's design, what would count as support), so you can read it and edit it, or ask for a
    # change (``--revise-plan``), before compute is spent. ``--resume`` then runs the design that is in the
    # file. ``plan.md`` is written whatever this is set to; ``"off"`` (default) only means the quest does
    # not wait for you, so unattended and fleet runs are not held up.
    plan: PlanPause = "off"
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

    @field_validator("plan", mode="before")
    @classmethod
    def _norm_plan(cls, v: Any) -> Any:
        # The unquoted-`off` Norway problem again, as for ``review`` and ``clarify``.
        return "off" if v is False else v


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
    # generated experiment script N times, handing each run its own
    # ``FI_REPLICATE_SEED`` -- the first run included, strided by
    # ``replicate_seed_stride`` so no two runs can draw the same seed --
    # which the implement prompt instructs the script to honour
    # for its own rng seeding. The N ``RESULT_JSON`` outputs land
    # in ``state['result_json_replicates']`` as a list; the analyze
    # node aggregates numeric fields with mean ± std and surfaces
    # the spread in its summary. Cost: N executions per design pass.
    #
    # Default 3, because a single run is not a result. A number produced
    # once has no error bar, and a researcher reading "we ran it once" has
    # no way to tell a real effect from a lucky seed -- so reporting it as
    # a finding overstates what the experiment showed.
    #
    # This is the cheap kind of rigour: replication re-runs the generated
    # SCRIPT only. ``implement`` is not re-invoked, so there are no extra
    # LLM calls and no extra provider cost -- it spends local compute, and
    # buys a mean +/- std instead of a point estimate. (One exception: a
    # script that never reads FI_REPLICATE_SEED is sent back once, right
    # after implement, to read it -- the ``implement_seed`` call.) Wall-clock grows
    # ~N x the execute step (bounded by ``execution.timeout_s`` each), so
    # set 1 to opt out on a slow experiment, or higher when the measurement
    # is noisy.
    execute_replicates: int = Field(default=3, ge=1)
    # What the experiment does when its script contradicts the protocol the plan fixed (the grid, the runs per
    # setting, the thresholds; ``core/protocol_check.py``). ``block`` (default) sends the script back for up to
    # ``protocol_repair_attempts`` repairs and, if it still differs, stops the quest for you to read and edit the
    # plan or the script, then resume; ``warn`` only logs and records it; ``off`` does not look. Nothing is checked
    # when the plan has no ``protocol`` block, or for a study with no experiment.
    protocol_check: Literal["block", "warn", "off"] = "block"
    protocol_repair_attempts: int = Field(default=2, ge=0, le=5)
    # The oracles the plan's protocol declares (a closed form, a limiting case, an invariant, an exact small case:
    # ``core/oracle_check.py``) are checked before the pilot and the main run: the script is run with FI_ORACLE=1
    # and must run them and print ``ORACLE_JSON``. A check that fails, one that is missing, or a protocol that
    # declares none is sent back for up to ``oracle_repair_attempts`` repairs (a protocol with no oracle first asks the
    # plan to be rewritten with one); if it still does not pass, ``block`` (default) stops the quest before the main
    # sweep, ``warn`` only logs and records it, ``off`` does not look. Nothing is checked without a ``protocol`` in the
    # design, for a background job, or for a study with no experiment.
    oracle_check: Literal["block", "warn", "off"] = "block"
    # A run that exits 0 and prints its results can still have been told by its own numerics that something is wrong:
    # an overflow, an invalid value or a division by zero in NumPy, a solver or optimiser that did not converge, an
    # argument that had no effect (SciPy's ``solve_ivp`` given ``abs_tol`` instead of ``atol``), an imaginary part
    # discarded, a NaN or an infinity in a result (``core/numeric_warnings.py`` reads only these). ``block`` (default)
    # sends the script back through the same repairs as a failed run (``exec_reflect_max_iterations``) and, if the
    # warnings remain when they are used up, stops the quest for you to read them; a resume runs the script again if you
    # changed it, and accepts the run as it is if you did not. ``warn`` only logs and records them; ``off`` does not look.
    numeric_warnings: Literal["block", "warn", "off"] = "block"
    oracle_repair_attempts: int = Field(default=2, ge=0, le=5)
    # How far apart consecutive replicates' seeds sit. Replicate i is handed
    # ``FI_REPLICATE_SEED = i * replicate_seed_stride``, so the seeds it can
    # derive occupy ``[i*stride, (i+1)*stride)`` and no two replicates reach
    # the same one.
    #
    # The stride is the whole point. A script rarely uses the base seed
    # directly -- it derives one seed per trial from it, and the obvious way to
    # write that is ``base + counter``. With bases a few apart, replicate 1's
    # trial 0 is replicate 2's trial 1: a graded quest handed bases 42, 1 and 2
    # drew 259 of every 300 trials in common between one pair of replicates and
    # 299 of 300 between another, then reported the spread across those three
    # near-identical runs as a 95% confidence interval -- printed as
    # "0.581-0.581", and read back in its own Limitations section as evidence
    # of stability. Striding keeps the streams disjoint for any experiment
    # drawing fewer than a stride of seeds per run, which a default of a
    # million makes ample: the quest above drew 2,700.
    #
    # Raise it for an experiment drawing more than a million seeds in one run.
    # Lower it only for a script that derives seeds by multiplying rather than
    # adding. Either way the guarantee is the same: disjoint streams, so what
    # varies between replicates is variation the experiment produced.
    replicate_seed_stride: int = Field(default=1_000_000, ge=1)
    # Pilot pass: run the experiment small (``FI_PILOT=1``, which the
    # implement prompt tells the script to honour) before running it for real.
    #
    # ``execute_reflect`` already repairs a script that CRASHES. It cannot
    # catch one that runs fine and answers the wrong question -- a sweep over
    # the wrong parameter range, a resolution too coarse to show the effect --
    # which today costs the full timeout to discover. A researcher would run a
    # cheap version first, check the numbers are the right order of magnitude,
    # and only then commit the compute.
    #
    # The pilot's numbers are DISCARDED; it is a smoke test of the DESIGN, not
    # a measurement. It never fails a quest: a bad pilot logs a warning naming
    # the likely cause and the full run proceeds, where execute_reflect can
    # still repair a genuine code fault with the traceback it needs.
    #
    # OFF by default, unlike execute_replicates, because the two degrade
    # differently when the generated script ignores its env var. A script that
    # ignores FI_REPLICATE_SEED produces one run repeated -- correlated, not
    # independent -- which the execute node detects from the script's own
    # source and refuses to aggregate, rather than reporting the spread across
    # it as sampling error.
    # A script that ignores FI_PILOT runs the experiment at FULL scale under a
    # fifth of the timeout, so it times out, wastes that compute, and emits a
    # warning that misdescribes a compliance failure as a design problem --
    # on every quest. Honouring the var is a prompt instruction, not something
    # the engine can enforce, so this stays opt-in until a given setup has
    # shown its scripts comply. Turn it on per quest; the payoff is catching a
    # wrong parameter range before paying the full timeout for it.
    pilot_run: bool = False
    # Pilot timeout as a fraction of ``execution.timeout_s`` (floored at 30s).
    # Small on purpose -- a pilot that takes as long as the real run buys
    # nothing.
    pilot_timeout_frac: float = Field(default=0.2, gt=0.0, le=1.0)
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
    # Survey mode — a descriptive literature / history synthesis with NO
    # experiment AND NO dataset (e.g. "the evolution/history of X", an
    # overview or retrospective). A stronger form of ``no_simulation``: the
    # engine skips BOTH the experiment (implement/execute) AND the
    # data-collection path (auto_collect_data/wait_for_data/data_load/
    # web_plots), routing ``design → web_figures → analyze`` so the paper
    # synthesises the cited literature directly (illustrative license-clean
    # images are still embedded). Setting this implies ``no_simulation``.
    # When unset (default False), the clarify node may STILL turn it on when
    # the clarify agent classifies ``topic_shape == 'survey'`` — auto-detection
    # from clarify is the second entry point. YAML flag wins when set.
    # See ``Engine._resolve_survey_from_clarify``.
    survey_mode: bool = False
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
    # Simulation skills the quest may call instead of re-deriving the
    # physics. Each name is resolved against ``<FI_SKILLS_DIR>/<name>/`` and the
    # ``fi.skills`` entry-point group; a name that resolves to a skill
    # which is not TRUSTED is logged and skipped, and the quest falls back
    # to guided generation. Empty (default) means "generate everything".
    #
    # A skill is trusted only when the self-test it carries passes AND a
    # person has approved that exact content — so an unattended ``--fleet``
    # run can never adopt capability nobody signed off on. Approve with
    # ``core.skills.approval.approve``; editing a skill lapses its approval.
    #
    # FI ships no skills: a skill that arrives with the install is a
    # curated library again, and curation does not compound. Skills live
    # in ``FI_SKILLS_DIR`` (default ``~/.frontier-insight/skills``), so a
    # fresh clone starts with none.
    skills: list[str] = Field(default_factory=list)
    # Skills to keep out of this quest, by name. Separate from ``skills``
    # because excluding one thing should not require listing everything else
    # — which gets impractical as the library grows, and would silently
    # exclude anything added later. Also the A/B handle: run one topic with a
    # skill and once without.
    skills_exclude: list[str] = Field(default_factory=list)
    # Skills this quest MUST use, by folder name. Selection still picks what it
    # judges useful; these are added to the pick whatever it says, and a name
    # that cannot be loaded (not found, not approved) stops the quest with the
    # reason instead of being skipped: naming a skill is an instruction, not a
    # suggestion. Independent of ``skills`` (which only narrows the candidates).
    skills_required: list[str] = Field(default_factory=list)
    # Extra folders of skills installed by other agents, read IN PLACE (nothing
    # is copied or converted), searchable and selectable like FI's own. Each is
    # a folder whose sub-folders hold a SKILL.md. You only need this for a folder
    # FI would not look in itself. An external skill has no FI self-test, so it is
    # loadable only after a person approves its exact content (edit it and the
    # approval lapses). ``python launch.py --config <this YAML> --approve-skill
    # NAME --approve-as YOU`` finds a skill that lives only in one of these
    # folders (likewise --skills, --why-skills, --scan-skill, --revoke-skill).
    skills_dirs: list[str] = Field(default_factory=list)
    # FI looks for other agents' skills by itself, in the usual places on Linux,
    # macOS and Windows (tool folders in the home directory and their plugin
    # trees, the per-OS application folders, and the project folder and its
    # parents). False turns that search off. See core/skills/registry.py.
    skills_scan_known_dirs: bool = True
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
    # Which interpreter builds each quest's venv. Previously declared but
    # never read — venv.EnvBuilder().create() always used sys.executable
    # (whichever interpreter happened to be running FI that invocation),
    # so on a machine with more than one Python install, the same
    # declared version could silently mean a different interpreter run
    # to run. Now actually resolved (the Windows `py` launcher, or
    # `python<version>` on PATH elsewhere) — see
    # core.execution._resolve_python_for_version.
    python_version: str = "3.11"
    docker_image: str = "python:3.11-slim"
    # Example files or folders the experiment should start from: a simulation
    # setup, an input deck, a config, a script, a document — any file type. They
    # are copied into <quest>/inputs/examples/ (files dropped there by hand while
    # the quest is paused count too), the design and the code-writing steps are
    # shown their names and the text of the small ones, and the experiment finds
    # the folder in the FI_INPUT_DIR environment variable. A path that does not
    # exist stops the quest before its first LLM call.
    inputs: list[str] = Field(default_factory=list)
    # The real simulation runs as a background job (HPC, a cluster, anything
    # longer than timeout_s). experiment.py is then written as an idempotent
    # driver rather than something that waits: the first run submits the job and
    # prints a pending line, later runs check it and, once it is finished, collect
    # the results. The quest pauses and exits cleanly while the job runs;
    # `--resume` checks once, and `--watch <quest>` checks on a timer and resumes
    # by itself. The selected skill supplies the simulator-specific part (how to
    # submit, how to tell finished from failed, how to read the results). The
    # contract is in core/job_watch.py.
    background_jobs: bool = False
    # Keep the simulation and its analysis in two scripts. ``code/simulate.py`` runs the
    # simulation and saves what it produced under ``raw/seed<K>/``; ``code/experiment.py``
    # reads those files, computes the statistics, draws the figures and prints the results.
    # An analysis that fails, or that the review sends back, is rewritten and run again
    # against the raw files already on disk; the simulation runs again only when its own
    # script changed (or its files are gone), which is what a simulation that takes hours
    # needs, and the raw outcomes of every run stay on disk for any later analysis (an
    # interval computed from the runs themselves, not from a few batch summaries).
    # ``auto`` (default) turns it on for a study whose design is stochastic (its protocol
    # fixes two or more runs per setting, or it describes a Monte Carlo / stochastic /
    # Gillespie ... process) and leaves it off otherwise, and for a background job, a
    # no-simulation study or a survey; ``true`` and ``false`` decide for every quest. Not
    # yet combinable with ``background_jobs`` when ``true``. The contract is in core/split_run.py.
    split_analysis: bool | Literal["auto"] = "auto"
    # Where ``split_analysis`` keeps the raw files: a folder relative to the quest folder,
    # or an absolute path (a big disk, an HPC scratch area). Empty means ``raw/`` in the
    # quest folder. FI records the path and each file's size and hash; it does not copy
    # them. The Docker sandbox sees only the quest folder, so there it must be relative.
    raw_dir: str = ""
    # Run quest code with the interpreter that runs FI itself — no per-quest
    # venv. One Python for everything: a package installed once (pip install
    # -e ., or an earlier quest) is just there for the next quest, and nothing
    # is built under the quest's own output path, which on Windows is often
    # long enough that pip cannot install torch there (260-char limit). The
    # cost is no isolation: what a quest installs stays in FI's environment.
    # Set false to get a fresh venv per quest (see system_site_packages).
    shared_interpreter: bool = True
    # (venv mode only) Each quest's venv inherits whatever FI's own interpreter already has
    # installed (matplotlib, numpy, pandas, ... are near-universal across
    # quests) instead of every quest re-downloading and re-building them
    # from an empty venv. A quest's own `pip install` still installs INTO
    # the venv and takes precedence there — this only fills in what a
    # quest doesn't ask for itself. Set false to restore full per-quest
    # isolation (nothing visible but what that quest explicitly installs).
    system_site_packages: bool = True

    @model_validator(mode="after")
    def _check_split_analysis(self) -> "ExecutionConfig":
        if self.raw_dir.strip() and self.split_analysis is False:
            raise ValueError(
                "execution.raw_dir only means something with execution.split_analysis: true (or auto)"
            )
        if self.split_analysis is True and self.background_jobs:
            raise ValueError(
                "execution.split_analysis cannot be combined with execution.background_jobs yet: "
                "a background job's script already submits, waits for and collects the run, "
                "and the analysis of what it collects is not split from it"
            )
        if (
            self.split_analysis is not False and self.sandbox == "docker"
            and self.raw_dir.strip() and Path(self.raw_dir).expanduser().is_absolute()
        ):
            raise ValueError(
                "execution.raw_dir must be relative to the quest folder with sandbox: docker, "
                "which sees only that folder"
            )
        return self


class KnowledgeConfig(BaseModel):
    """Wraps the Axon knowledge layer.

    `axon_config` may be either an inline mapping or a path to an existing
    Axon YAML. Both end up at `AxonConfig.load(path)`, which is the only
    constructor Axon offers that understands its own file format: an inline
    mapping is written to a temp YAML first. Use Axon's **nested** shape
    (`embedding: {provider:, model:}`), not the flat dataclass field names —
    `load` performs that mapping, along with env overrides and dropping keys
    Axon has retired.
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
    # OpenAlex API key (free at openalex.org). Since February 2026 OpenAlex
    # gives its full free daily budget only to keyed requests; without a key
    # a machine gets about a tenth of it — roughly 100 searches a day, and a
    # quest uses dozens, since arXiv is searched through OpenAlex too. Env
    # fallback ``OPENALEX_API_KEY``; a set env var wins over YAML. Sent in the
    # ``Authorization`` header, never in the URL: httpx logs every request URL
    # at INFO, so a key in the query string would land in the console output.
    openalex_api_key: str = Field(
        default_factory=lambda: os.environ.get("OPENALEX_API_KEY", "").strip()
    )
    # Semantic Scholar API key (free on request). The keyless pool is shared
    # by everyone and answers 429 most of the time. Env fallback
    # ``SEMANTIC_SCHOLAR_API_KEY``; a set env var wins over YAML.
    semantic_scholar_api_key: str = Field(
        default_factory=lambda: os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "").strip()
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
    # Deterministic relevance floor applied to freshly-retrieved literature in
    # the literature node — a cheap complement to the LLM ``relevance_guard``,
    # which runs ONLY on the auto_collect path (so survey / simulation quests,
    # which skip auto_collect, otherwise carry off-topic sources into
    # analyze/write). Each retrieved doc is scored by embedding cosine (the
    # all-MiniLM model shared with passage ranking) against the TOPIC; docs
    # below ``relevance_min_score`` are dropped. Fail-open: no filtering when
    # embeddings are unavailable (FI_OFFLINE / no model). 0.0 disables. The
    # 0.20 default sits in the empirically-measured gap between on-topic
    # (~0.4+) and off-topic (<0.1) docs for a humanities/history topic.
    relevance_min_score: float = Field(default=0.20, ge=0.0, le=1.0)
    # Never-starve retention: always keep at least this many top-scoring docs
    # even when all fall below ``relevance_min_score``, so a wholly-borderline
    # corpus is not emptied (the evidence_gate can then broaden instead).
    relevance_min_keep: int = Field(default=3, ge=0)
    # Re-search with different keywords when a retrieval comes back
    # wholly off-topic. When NO doc clears ``relevance_min_score`` on its own
    # merits, the first query was probably worded badly -- so ask the model
    # for alternative phrasings and search again, rather than handing the
    # writer the ``relevance_min_keep`` least-bad hits and proceeding as if
    # they were evidence. Each retry costs one small LLM call plus a
    # retrieval, so it is bounded and only fires on the wholly-missed case
    # (not on "few but good" results).
    #
    # Skipped entirely when embeddings are unavailable: without scores there
    # is no signal that the query was bad, and retrying blind would just
    # multiply cost on exactly the air-gapped machines that can least
    # afford it.
    requery_on_low_relevance: bool = True
    requery_max: int = Field(default=2, ge=0, le=5)
    # LLM literature screen. After the relevance floor, one batched call
    # grades every retrieved source 0-3 on "could the paper cite this for a
    # claim?". Scholarly records need a 2 to stay; web pages are dropped only
    # at 0, because they supply quotable text rather than citations. Keeps at
    # least ``relevance_min_keep`` sources (highest grades first) and fails
    # open: a failed or unreadable call keeps everything, and a source the
    # model did not grade is kept. The embedding floor alone cannot tell a
    # table of contents or an unrelated paper that shares the search terms
    # from a real on-topic source; this can. Off → floor only.
    literature_screen: bool = True
    # Foundational works. A search engine matches every word of a query, so
    # the original method papers and textbooks a topic rests on rarely come
    # back: a damped-oscillator integrator quest found none of Verlet (1967),
    # Hairer, Lubich & Wanner, or Butcher. One call asks the model for up to
    # eight, each is looked up by title in OpenAlex and dropped when not found,
    # and the works at least two retrieved papers cite are added (books too,
    # whatever the quest's work scope). All of them go through the literature
    # screen, labelled foundational, and the writer is asked to cite the ones
    # that bear on the paper. Costs one LLM call and up to ten OpenAlex
    # requests per literature pass. Off → no extra candidates.
    foundational_works: bool = True
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
    # Default ``True``, matching ``pauses.papers``: this legacy flag is merged
    # into it, and the programmatic merge copies the attribute through, so
    # the two defaults must agree. Pairs with ``knowledge.try_fetch_full_text``
    # (also on): the gate fires only for the papers the open-access cascade
    # could not fetch. Set ``pauses.papers: false`` for an unattended run.
    pause_for_user_papers: bool = True
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
    #   core             — 240M open-access papers (free, keyless; CORE_API_KEY
    #                      raises its rate limit).
    #   openaire         — European open-access research graph (free, keyless).
    #   doaj             — Directory of Open Access Journals articles (free, keyless).
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
    # paywalled venue. On by default: a legal copy is the difference between
    # the writer quoting a paper and quoting its abstract. The whole batch
    # shares ``full_text_fetch_total_s``; set false for abstracts only.
    try_fetch_full_text: bool = True
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
        # Path-shaped input (str or Path) gets `~` expansion; mappings pass
        # through untouched for `core.knowledge._axon_config_from` to render
        # into the temp YAML that `AxonConfig.load` reads.
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
    # Who made the outputs: printed under the paper title, on the slides'
    # title slide and in the poster header. All optional; an empty author
    # keeps the "Frontier Insight" byline. They go only into the quest's own
    # files (and so into the page screenshots the visual check sends to the
    # configured LLM provider), never to a search service.
    author: str = ""
    affiliation: str = ""
    contact_email: str = ""
    # A link for the work (repository, lab page). The poster prints it as a
    # QR code; without it there is no QR code.
    url: str = ""
    # Poster sheet. The portrait sizes lay the poster out in two columns,
    # 48 x 36 in landscape in three; font sizes meet the published poster
    # minimums at every size.
    poster_size: Literal["a1_portrait", "a0_portrait", "landscape_48x36"] = "a1_portrait"
    # After the outputs render, measure each PDF (paper, slides, poster) and
    # screenshot its pages into the report folder. The report is
    # .fi/visual_check.json; nothing here stops the quest.
    visual_check: bool = True
    # Also send the screenshots to the configured LLM provider and ask it to
    # check the pages against a fixed checklist of what a script cannot see.
    # Off by default: it costs about 12,000 tokens a quest, and the problems
    # the measurements find are found without it. A provider that cannot take
    # images leaves a measurements-only check.
    visual_check_ai: bool = False
    # How many times an output may be redone after its check finds problems
    # the redo can fix, before the findings are only reported.
    visual_check_max_redos: int = Field(2, ge=0, le=2)
    # The most pages paper.pdf may take. Unset, a limit the topic states
    # ("≤ 4 pages", "at most 4 pages", "a 4-page paper") applies
    # (``resolve_page_limit``); with neither, the paper keeps its usual layout
    # and length. With a limit, the LaTeX templates use 2 cm margins (where
    # they had 1 in), figures at most a third of the text height and smaller
    # source lists, the writer gets a word budget, and the review renders each
    # draft and counts its pages: a draft over the limit is sent back to be
    # shortened, at most twice, without using ``engine.max_iterations``.
    page_limit: int | None = Field(None, ge=1)

    @field_validator("author", "affiliation", "contact_email", "url", mode="before")
    @classmethod
    def _one_line(cls, v: object) -> object:
        # YAML hands over None for an empty key, and a block scalar keeps
        # its newlines; each field prints as one line.
        if v is None:
            return ""
        return " ".join(v.split()) if isinstance(v, str) else v

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
    def _audit_unknown_keys(cls, data: dict[str, Any]) -> None:
        """Raise ``ValueError`` listing any YAML keys FI doesn't recognize.

        Pydantic's default ``extra='ignore'`` silently drops an unknown key, so
        a typo (``node_http_timeoutt_s``, ``reviewer_panel``) validates cleanly
        and the setting quietly reverts to its default — disabling a knob the
        user believes they enabled. This checks the top level and the six known
        sub-sections one level deep. It deliberately does NOT recurse into
        free-form dict fields (``node_models``, ``node_cli_timeout_s``,
        ``node_http_timeout_s``, ``node_ensemble``, ``clarify_overrides``,
        ``extra``, …) whose keys are arbitrary."""
        unknown: list[str] = []
        for k in data:
            if k not in cls.model_fields:
                unknown.append(k)
        # Legacy pause flags live under engine/knowledge but map into `pauses:`
        # via ``_merge_legacy_pauses`` — they are valid, not typos.
        legacy: dict[str, set[str]] = {}
        for section, old_key, _new in _LEGACY_PAUSE_FLAGS:
            legacy.setdefault(section, set()).add(old_key)
        section_models = {
            "provider": ProviderConfig, "engine": EngineConfig,
            "execution": ExecutionConfig, "knowledge": KnowledgeConfig,
            "pauses": PausesConfig, "output": OutputConfig,
        }
        for section, model in section_models.items():
            sub = data.get(section)
            if not isinstance(sub, dict):
                continue
            allowed = set(model.model_fields) | legacy.get(section, set())
            for k in sub:
                if k not in allowed:
                    unknown.append(f"{section}.{k}")
        if unknown:
            raise ValueError(
                "unrecognized config key(s): " + ", ".join(sorted(unknown))
                + " — check for typos. Unknown keys are rejected here because "
                "they would otherwise be dropped and the setting silently revert "
                "to its default instead of taking effect."
            )

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if isinstance(data, dict):
            cls._audit_unknown_keys(data)
        return cls.model_validate(data)


# A page limit written into the topic. The number must come right before
# "page(s)", with a bound word in front of it ("≤ 4 pages", "at most four
# pages", "up to 4 pages") or after it ("4 pages max"), or as "a 4-page
# paper". A range ("4–8 pages", "4 to 8 pages") is a length, not a limit.
_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20,
}
_PAGE_COUNT = (
    r"(?<![\w\-‐‑–—./])(?P<n>\d{1,3}|"
    + "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True))
    + r")(?![\w.])"
)
_PAGE_LIMIT_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"(?:≤|⩽|<=|=<|\bat\s+most|\bno\s+more\s+than|\bnot\s+more\s+than|\bno\s+longer\s+than"
        r"|\bup\s+to|\b(?:a\s+)?max(?:imum)?(?:\s+of)?|\bmax\.|\bpage\s+limit\s+(?:of|is)|\blimit(?:ed)?\s+(?:of|to))"
        r"\s*" + _PAGE_COUNT + r"\s*pages?\b",
        _PAGE_COUNT + r"\s*pages?\s*(?:max(?:imum)?\b|at\s+most\b|or\s+(?:fewer|less)\b)",
        r"\bpage\s+limit\s*[:=]?\s*(?:of\s+)?" + _PAGE_COUNT,
        _PAGE_COUNT + r"\s*[-‐‑–]\s*page\b",
    )
]
# What turns the number into the top of a range: "4–8", "4 to 8", "4 or 5".
_RANGE_BEFORE = re.compile(
    r"(?:\d|\b(?:" + "|".join(_NUMBER_WORDS) + r"))\s*(?:[-‐‑–—]|to|or)\s*$", re.IGNORECASE,
)


def page_limit_from_text(text: str) -> int | None:
    """The page limit ``text`` states, or ``None``. With several, the
    smallest. A bound said in words ("≤ 6 pages") wins over the "N-page"
    form, which can name a part ("a 2-page appendix") rather than the
    paper."""
    bounds: list[int] = []
    sized: list[int] = []
    for index, pattern in enumerate(_PAGE_LIMIT_PATTERNS):
        for m in pattern.finditer(text or ""):
            if _RANGE_BEFORE.search(text[: m.start("n")]):
                continue
            word = m.group("n").lower()
            n = _NUMBER_WORDS[word] if word in _NUMBER_WORDS else int(word)
            if n >= 1:
                (sized if index == len(_PAGE_LIMIT_PATTERNS) - 1 else bounds).append(n)
    found = bounds or sized
    return min(found) if found else None


def resolve_page_limit(config: Any) -> int | None:
    """The quest's page limit: ``output.page_limit`` when set, otherwise the
    one its topic states, otherwise ``None`` (no limit, today's layout)."""
    set_limit = getattr(getattr(config, "output", None), "page_limit", None)
    if isinstance(set_limit, int) and set_limit >= 1:
        return set_limit
    topic = getattr(config, "topic", "")
    return page_limit_from_text(topic) if isinstance(topic, str) else None
