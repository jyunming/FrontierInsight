"""Single source of truth for the Frontier Insight interview.

Every frontend that collects research-quest config from the user reads
its question list, defaults, choice catalog, and YAML emitter from
THIS module:

* ``launch.py --new``                — CLI frontend, prompts on stdin.
* ``launch.py --update <quest_id>``  — mid-quest re-entry, edits the
                                        editable subset.
* ``vscode-frontier-insight``         — VSCode `@fi /new` and
                                        `@fi /update`. Reads
                                        ``core/interview_schema.json``
                                        (the JSON snapshot of QUESTIONS
                                        below) at extension activation
                                        so the TS UI renders identical
                                        questions to the CLI.
* ``--serve`` web UI                 — FastAPI endpoint that mounts
                                        QUESTIONS as an HTMX form.

Keeping all three frontends in lockstep is enforced by
``tests/test_interview_schema_parity.py``, which loads the JSON
snapshot + parses the TS literal + parses the HTMX form and asserts
the question keys / venue list / provider list agree.

Cost discipline: the preflight LLM call (``preflight_clarify``) fires
ONCE per interview to generate topic-tuned defaults for the three
clarify slots (``comparative_baseline`` / ``success_metric`` /
``budget``). When no provider is reachable, the call degrades to
static placeholders so the interview never blocks on LLM availability.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal


# ---------------------------------------------------------------------------
# Question schema
# ---------------------------------------------------------------------------

QuestionKind = Literal["text", "single", "multi"]


@dataclass(frozen=True)
class Choice:
    """One option in a single- or multi-select question."""

    value: Any
    label: str
    description: str = ""


@dataclass(frozen=True)
class Question:
    """A single interview question.

    ``id`` is the stable identifier all frontends key off. Changing it
    is a breaking change for the schema-parity test.

    ``mid_quest_editable`` is consulted by ``interview_update`` — only
    fields where this is True are surfaced when the user runs
    ``launch.py --update <id>``.

    ``frontends`` lists which surfaces ask this question. The
    ``provider`` and ``model`` questions are CLI-only because the
    VSCode extension always pins ``provider.name = "vscode_extension"``
    and captures ``provider.model`` silently from
    ``vscode.lm.selectChatModels()``.
    """

    id: str
    label: str
    prompt: str
    kind: QuestionKind
    choices: tuple[Choice, ...] = ()
    placeholder: str = ""
    # Static default. May be overridden by ``smart_default`` (which
    # gets the partial answers and can derive a context-aware default).
    default: Any = None
    mid_quest_editable: bool = False
    frontends: tuple[str, ...] = ("cli", "vscode", "serve")
    # When the user picks a value not in ``choices`` is allowed via a
    # free-text fallback. Today used only for ``provider_model``.
    allow_other: bool = False
    # Interview presentation tier. Frontends consult this to decide
    # which questions to show by default:
    #
    #   tier=1 — always-ask. The handful of slots that genuinely need a
    #            user decision (topic, paper_format, outputs, depth,
    #            and on CLI/web also provider + model).
    #
    #   tier=2 — auto-derive. Smart defaults populate these from tier-1
    #            answers (title from slug, no_simulation from format,
    #            knowledge from Axon-sidecar status, etc.). Frontends
    #            still SHOW them on a review screen so the user can
    #            click-to-edit before launch.
    #
    #   tier=3 — advanced. Topic-tuned slots whose default is good
    #            enough for 95% of quests (the preflight LLM call
    #            suggests baselines / metrics / budgets). Hidden behind
    #            a "Show advanced" toggle on each frontend.
    tier: int = 1


# ---------------------------------------------------------------------------
# Choice catalogs
# ---------------------------------------------------------------------------

# Output bundles. Match the four bundles VSCode `/new` already offered
# so existing users see the same options. Order from "most usual" to
# "cheapest".
OUTPUT_BUNDLES: tuple[Choice, ...] = (
    Choice(
        value=["paper_md", "paper_pdf"],
        label="paper + PDF (recommended)",
        description="Markdown + rendered PDF; needs pandoc + LaTeX on PATH. Falls back to MD-only when missing.",
    ),
    Choice(
        value=["paper_md", "paper_pdf", "slides", "poster", "speech"],
        label="everything",
        description="paper, PDF, slides, poster, talk script. Maximum tooling needed (pandoc + LaTeX + marp + Beamer).",
    ),
    Choice(
        value=["paper_md", "paper_pdf", "slides"],
        label="paper + PDF + slides",
        description="PDF via pandoc + LaTeX; slides via Marp; .pptx via pandoc.",
    ),
    Choice(
        value=["paper_md"],
        label="paper only (Markdown)",
        description="Fastest path; no extra system tools needed.",
    ),
)


PAPER_FORMATS: tuple[Choice, ...] = (
    # Scientific venues — keep order in sync with PaperFormat in core/config.py.
    Choice("generic", "generic — scientific paper, IMRAD",
           "Default for scientific topics. Surveys, comparative reviews, theoretical derivations, brief preprints."),
    Choice("neurips", "NeurIPS — ML benchmark / algorithm",
           "Empirical ML, neural networks, learning algorithms, journal-length."),
    Choice("iclr", "ICLR — representation learning",
           "Representations, generative models, ML theory."),
    Choice("ieee_access", "IEEE Access — engineering / systems",
           "Hardware/software architectures, measurement studies, engineering experiments. Two-column layout."),
    Choice("nature_mi", "Nature MI — physical sciences",
           "Physics / chemistry / materials simulation, scientific-method experiments."),
    # Non-scientific prose formats.
    Choice("essay", "essay — long-form argumentative prose",
           "Cultural / historical / intellectual / qualitative cross-case analysis. Argue a thesis."),
    Choice("report", "report — consulting-style exec report",
           "Business / operational / market analysis with cover + TOC. Decision-maker audience."),
    Choice("policy_brief", "policy brief — 2-4 page recommendation",
           "Single decision for policymakers. Issue + context + recommendation."),
    Choice("whitepaper", "whitepaper — 8-20 page industry analysis",
           "Vendor-neutral tech trends / standards / architecture comparisons. Practitioner audience."),
)


PROSE_FORMATS: frozenset[str] = frozenset(
    {"essay", "report", "policy_brief", "whitepaper"}
)


STUDY_DEPTHS: tuple[Choice, ...] = (
    Choice("brief preprint", "brief preprint",
           "1–2 pages, terse opening, focus on novel findings only. Citations OK to be few."),
    Choice("journal-length", "journal-length (recommended)",
           "4–8 pages with full IMRAD (or prose equivalent). ~15 citations, 1500–2500 words."),
    Choice("comprehensive review", "comprehensive review",
           "10–15 pages with Background + Comparison + Synthesis sections. 4000+ words, 10+ discussed citations."),
)


CLARIFY_MODES: tuple[Choice, ...] = (
    Choice("auto", "Agent self-clarifies (recommended)",
           "Agent generates the questionnaire AND auto-answers it from the topic. Pinned interview answers still win."),
    Choice("interactive", "Pause for me to answer",
           "Engine pauses; you confirm/edit each clarify slot. Highest quality, most interruption."),
    Choice("off", "Just run it",
           "Skip the clarify node entirely. Engine relies on the interview answers + topic alone."),
)


REVIEW_PANELS: tuple[Choice, ...] = (
    Choice([], "Single reviewer (cheapest)",
           "1 LLM call per review pass."),
    Choice(["methodologist", "statistician", "devil_advocate"], "3-persona panel",
           "Methodologist + Statistician + Devil's-advocate + moderator. ~4× the review cost."),
    Choice(["methodologist", "statistician", "devil_advocate", "reproducibility"], "4-persona panel",
           "Adds Reproducibility reviewer. ~5× the review cost."),
)


KNOWLEDGE_CHOICES: tuple[Choice, ...] = (
    Choice(False, "Disabled (recommended for first runs)",
           "Literature retrieval falls back to free public sources (arXiv, OpenAlex, Crossref) when needed."),
    Choice(True, "Enabled (requires Axon installed)",
           "Use the Axon corpus for retrieval + write-back. Skip if you haven't set Axon up."),
)


NO_SIMULATION_CHOICES: tuple[Choice, ...] = (
    Choice(False, "Computational — a Python script can produce the data",
           "Run normal implement → execute pipeline. For physics / ML / algorithmic / benchmark topics."),
    Choice(True, "Observational — needs real-world data the engine can't simulate",
           "Skip implement / execute; route to wait_for_data + auto_collect_data. For cultural / historical / qualitative / policy topics."),
)


ENSEMBLE_PROFILES: tuple[Choice, ...] = (
    Choice("off", "Single model (default, cheapest)",
           "One LLM call per node. Cost baseline = 1×."),
    Choice("cross_check_only", "Fan out cross_check only (~1.3× cost)",
           "3 models vote per finding; catches issues the writer might miss. Minimal extra spend."),
    Choice("ideate_and_check", "Fan out ideate + cross_check (~2.0× cost)",
           "3 models brainstorm + moderator picks the strongest; 3 models vote on each finding."),
    Choice("full", "Fan out ideate + analyze + cross_check (~2.5× cost)",
           "Multi-model on the three highest-leverage nodes. Best signal; highest cost."),
)


# Cost multipliers vs a single-model baseline. Frontends display these
# next to the profile choice so the user sees the trade-off before
# launch. The numbers are conservative estimates: ideate goes from 1
# call to 4 (3 fan-out + 1 moderator); analyze same; cross_check goes
# from K calls to 3K (no moderator). Across a typical quest of 7-18
# LLM calls, that's the rough multiplier each profile lands at.
_ENSEMBLE_COST_MULTIPLIERS: dict[str, float] = {
    "off": 1.0,
    "cross_check_only": 1.3,
    "ideate_and_check": 2.0,
    "full": 2.5,
}


def estimate_ensemble_cost_multiplier(profile: str) -> float:
    """Return the rough LLM-cost multiplier vs the no-ensemble baseline.
    Frontends render ``~Nx more LLM calls`` next to the profile picker
    so the user sees the cost discipline implication at decision time."""
    return _ENSEMBLE_COST_MULTIPLIERS.get(profile, 1.0)


# Per-provider model trios used by every ensemble profile. These are
# the models the interview will populate node_ensemble[*].models with
# when the user picks a profile != "off". Edit the trio for a new
# provider here (NOT in the YAML emitter) so all three frontends agree.
_ENSEMBLE_MODEL_TRIOS: dict[str, tuple[str, str, str]] = {
    "openai":           ("gpt-4o", "gpt-4o-mini", "o1-mini"),
    "codex":            ("gpt-5", "gpt-4o", "gpt-4o-mini"),
    "claude_cli":       ("claude-opus-4-7", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"),
    "codex_cli":        ("gpt-5", "gpt-4o", "gpt-4o-mini"),
    "copilot_cli":      ("gpt-4o", "claude-3-5-sonnet", "o1-mini"),
    "gemini_cli":       ("gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-pro"),
    "ollama":           ("llama3.1:70b", "qwen2.5:32b", "llama3.1:8b"),
    "vscode_extension": ("gpt-4o", "claude-3-5-sonnet", "gemini-2.0-flash"),
}


def ensemble_model_trio(provider: str, provider_model: str | None = None) -> list[str]:
    """Return the three model ids the ensemble profile will fan out
    over for ``provider``. Falls back to repeating ``provider_model``
    (or "default") three times for unknown providers — the engine will
    still fire 3 calls, just against the same model, which is mostly
    useful as a smoke test for new providers."""
    trio = _ENSEMBLE_MODEL_TRIOS.get(provider)
    if trio is not None:
        return list(trio)
    fallback = provider_model or "default"
    return [fallback, fallback, fallback]


AUDIENCE_CHOICES: tuple[Choice, ...] = (
    Choice("external", "External — journal / open web (recommended)",
           "FI's own cross-quest memory (fi_critique / fi_digest / fi_portfolio / fi_proposal / fi_summary) is dropped from References; an outside reader can't look those up. Real external sources + your own ingested papers are kept."),
    Choice("internal", "Internal — team report / memo / onboarding",
           "Cite anything in Axon, including FI's own cross-quest summaries. Use for write-ups your team will read."),
)


# Providers. The ``model_options`` table below carries one curated
# model list per provider; ``provider_model`` reuses these at runtime.
# vscode_extension is intentionally absent from the user-facing list —
# the VSCode extension pins it silently and the CLI shouldn't ask.
PROVIDER_CHOICES: tuple[Choice, ...] = (
    Choice("openai", "openai — HTTP direct (OpenAI API)",
           "Needs OPENAI_API_KEY. Fastest path, billed against your OpenAI account."),
    Choice("codex", "codex — HTTP direct (ChatGPT Plus/Pro OAuth backend)",
           "Same transport as openai, points at ChatGPT's backend. Needs OPENAI_API_KEY."),
    Choice("claude_cli", "claude_cli — local Claude Code CLI",
           "Needs `claude` binary on PATH + `claude login`. Uses your Claude Pro/Max OAuth."),
    Choice("codex_cli", "codex_cli — local Codex CLI",
           "Needs `codex` binary on PATH + `codex login`. Uses your ChatGPT Plus/Pro OAuth."),
    Choice("copilot_cli", "copilot_cli — local GitHub Copilot CLI",
           "Needs `gh` + Copilot Pro/Business. Uses `gh auth login`."),
    Choice("gemini_cli", "gemini_cli — local Gemini CLI",
           "Needs `gemini` binary on PATH. Uses Google AI Studio OAuth."),
    Choice("ollama", "ollama — local server",
           "Needs Ollama running. Free, fully local; smaller models."),
)


# Per-provider curated model list. Keep these aligned with the actual
# models each provider supports today; users can pick "Other" to type
# a brand-new model name not yet on the list.
PROVIDER_MODEL_OPTIONS: dict[str, tuple[Choice, ...]] = {
    "openai": (
        Choice("gpt-4o", "gpt-4o", "Latest GA. Balanced quality + cost."),
        Choice("gpt-4o-mini", "gpt-4o-mini", "Cheap; good for high-volume nodes."),
        Choice("gpt-4-turbo", "gpt-4-turbo", "Older flagship; still strong."),
        Choice("o1-preview", "o1-preview", "Reasoning model; slower + pricier."),
        Choice("o1-mini", "o1-mini", "Cheaper reasoning model."),
    ),
    "codex": (
        Choice("gpt-5", "gpt-5", "ChatGPT backend; quality matches Plus/Pro web UI."),
        Choice("gpt-4o", "gpt-4o", "Older default."),
    ),
    "claude_cli": (
        Choice("claude-opus-4-7", "claude-opus-4-7", "Latest Opus; strongest model."),
        Choice("claude-sonnet-4-6", "claude-sonnet-4-6", "Latest Sonnet; balanced."),
        Choice("claude-haiku-4-5-20251001", "claude-haiku-4-5", "Latest Haiku; cheapest."),
    ),
    "codex_cli": (
        Choice("gpt-5", "gpt-5", "ChatGPT backend default."),
        Choice("gpt-4o", "gpt-4o", "Older default."),
    ),
    "copilot_cli": (
        Choice("gpt-4o", "gpt-4o", "Copilot Pro/Business default."),
        Choice("claude-3-5-sonnet", "claude-3-5-sonnet", "Available to some Copilot tiers."),
    ),
    "gemini_cli": (
        Choice("gemini-2.5-pro", "gemini-2.5-pro", "Latest Pro; long context."),
        Choice("gemini-2.5-flash", "gemini-2.5-flash", "Cheaper / faster."),
    ),
    "ollama": (
        Choice("llama3.1:70b", "llama3.1:70b", "Largest Llama 3.1 that fits 80GB VRAM."),
        Choice("llama3.1:8b", "llama3.1:8b", "Fits 12GB VRAM; modest quality."),
        Choice("qwen2.5:32b", "qwen2.5:32b", "Strong open-source mid-size."),
    ),
}


# ---------------------------------------------------------------------------
# Question list — THE canonical interview definition
# ---------------------------------------------------------------------------

QUESTIONS: tuple[Question, ...] = (
    # ─── Tier 1 — always-ask ──────────────────────────────────────────
    Question(
        id="topic",
        label="Topic",
        prompt="What do you want to study? Be specific about the question, what's known, and what success looks like.",
        kind="text",
        placeholder="e.g. Compare three numerical integrators on a damped harmonic oscillator and report energy drift.",
        mid_quest_editable=False,
        tier=1,
    ),
    Question(
        id="paper_format",
        label="Paper format / venue",
        prompt="Picks the LaTeX template + writing persona. 'generic' is safe for scientific topics; 'essay' for non-computational humanities/social-science.",
        kind="single",
        choices=PAPER_FORMATS,
        default="generic",
        mid_quest_editable=True,
        tier=1,
    ),
    Question(
        id="output_kinds",
        label="Outputs",
        prompt="Which deliverables matter? PDF gracefully degrades to MD if pandoc isn't installed.",
        kind="single",  # bundles are pre-defined combos, not per-kind multi-select.
        choices=OUTPUT_BUNDLES,
        default=["paper_md", "paper_pdf"],
        mid_quest_editable=True,
        tier=1,
    ),
    Question(
        id="study_depth",
        label="Study depth",
        prompt="How mature should the resulting paper be? Drives word count + citation depth.",
        kind="single",
        choices=STUDY_DEPTHS,
        default="journal-length",
        mid_quest_editable=True,
        tier=1,
    ),
    Question(
        id="provider",
        label="LLM provider",
        prompt="Which transport for LLM calls. VSCode pins this to vscode_extension automatically.",
        kind="single",
        choices=PROVIDER_CHOICES,
        # No default — CLI auto-picks the first available, but the
        # interview never preselects so the user makes an explicit
        # choice.
        default=None,
        mid_quest_editable=False,
        frontends=("cli", "serve"),
        tier=1,
    ),
    Question(
        id="provider_model",
        label="Provider model",
        prompt="Which model to call. Curated list per provider; pick 'Other' to type a model name not yet on the list.",
        kind="single",
        # choices is computed at render time based on the picked
        # provider (see ``model_choices_for``); kept empty here.
        choices=(),
        default=None,
        mid_quest_editable=False,
        frontends=("cli", "serve"),
        allow_other=True,
        tier=1,
    ),
    # ─── Tier 2 — auto-derive, shown in review screen ────────────────
    Question(
        id="title",
        label="Title (short slug)",
        prompt="Short identifier for this quest. Used in folder names. Auto-slugged from your topic; edit if you want something different.",
        kind="text",
        mid_quest_editable=False,
        tier=2,
    ),
    Question(
        id="no_simulation",
        label="Research approach",
        prompt="Decides whether the engine runs a Python experiment or waits for real-world data. Auto-derived from paper format.",
        kind="single",
        choices=NO_SIMULATION_CHOICES,
        default=False,
        # Mid-quest editing of this is dangerous: the engine routes
        # divergently in the two branches. interview_update.py
        # refuses the flip with a clear message.
        mid_quest_editable=False,
        tier=2,
    ),
    Question(
        id="clarify_mode",
        label="Pre-flight clarification mode",
        prompt="Whether the engine pauses to confirm clarify slots. Pinned interview answers win regardless.",
        kind="single",
        choices=CLARIFY_MODES,
        default="auto",
        mid_quest_editable=True,
        tier=2,
    ),
    Question(
        id="review_panel",
        label="Reviewer panel",
        prompt="Single reviewer is fine for most. Use the panel when correctness matters more than cost.",
        kind="single",
        choices=REVIEW_PANELS,
        default=[],
        mid_quest_editable=True,
        tier=2,
    ),
    Question(
        id="knowledge_enabled",
        label="Knowledge layer (Axon)",
        prompt="Auto-detected from the Axon sidecar status. Override only if you have a specific reason.",
        kind="single",
        choices=KNOWLEDGE_CHOICES,
        default=False,
        mid_quest_editable=True,
        tier=2,
    ),
    Question(
        id="audience",
        label="Paper audience",
        prompt="External-facing (journal, open web) drops FI-internal cross-quest entries from the References; internal-facing keeps everything (good for project reports, memos, onboarding docs).",
        kind="single",
        choices=AUDIENCE_CHOICES,
        default="external",
        mid_quest_editable=True,
        tier=2,
    ),
    # ─── Tier 3 — advanced, behind a [Show advanced] toggle ──────────
    Question(
        id="comparative_baseline",
        label="Comparative baseline",
        prompt="What existing method / dataset / regime should this study be compared against? Topic-tuned default suggested by the preflight clarify call.",
        kind="text",
        placeholder="e.g. RandomForest baseline on the same features",
        mid_quest_editable=True,
        tier=3,
    ),
    Question(
        id="success_metric",
        label="Success metric",
        prompt="What number changing in what direction = headline result?",
        kind="text",
        placeholder="e.g. AUC ≥ 0.9 on held-out test set",
        mid_quest_editable=True,
        tier=3,
    ),
    Question(
        id="budget",
        label="Time / compute budget",
        prompt="Soft cap on wall-clock for the experiment.",
        kind="text",
        placeholder="e.g. a few minutes on a laptop CPU",
        mid_quest_editable=True,
        tier=3,
    ),
    Question(
        id="knowledge_top_k",
        label="Axon hits per quest",
        prompt="Dense-embedding retrievals from Axon fed to the writer. Small-k is precision (8 default); bump to 12-15 for comprehensive reviews. External web hits are sized separately via 'External hits per quest'.",
        kind="text",
        default=8,
        placeholder="8",
        mid_quest_editable=True,
        tier=2,
    ),
    Question(
        id="knowledge_external_top_k",
        label="External hits per quest",
        prompt="arXiv / OpenAlex / Crossref / S2 results when Axon misses. Bigger than Axon's cap because web search is coarser and a literature scan needs breadth. 20 default; 30+ for survey-shaped quests.",
        kind="text",
        default=20,
        placeholder="20",
        mid_quest_editable=True,
        tier=3,
    ),
    Question(
        id="ensemble_profile",
        label="Multi-model ensemble",
        prompt="Run the same node across multiple LLMs and merge their answers. Cost multiplies; agreement signal increases. Pick 'off' for single-model (cheapest) or one of the fan-out profiles for higher-stakes work.",
        kind="single",
        choices=ENSEMBLE_PROFILES,
        default="off",
        mid_quest_editable=True,
        tier=3,
    ),
)


# ---------------------------------------------------------------------------
# Editable subset for mid-quest --update
# ---------------------------------------------------------------------------

EDITABLE_FIELDS: frozenset[str] = frozenset(
    q.id for q in QUESTIONS if q.mid_quest_editable
)


# Stage invalidation matrix — which LangGraph nodes need to re-run
# when a given field changes mid-quest. Consumed by
# ``interview_update.invalidate_stages``. Keep entries ordered by
# expected pipeline position so the engine resumes from the EARLIEST
# affected node when multiple fields change.
STAGE_INVALIDATION: dict[str, tuple[str, ...]] = {
    # paper_format only affects the write stage (template choice) and
    # downstream generators. analyze + everything before is reusable.
    "paper_format": ("write", "review"),
    # output_kinds doesn't invalidate ANY LLM node — generators just
    # re-fire for the newly-added kinds. Empty tuple is the sentinel.
    "output_kinds": (),
    # study_depth flows into write's word-count target + review's
    # depth-check. Re-run from write.
    "study_depth": ("write", "review"),
    # Topic-tuned slots: analyze reads them when interpreting
    # results; write cites them in the Methods/Discussion. Conservative
    # re-run from analyze covers both.
    "comparative_baseline": ("analyze", "cross_check", "write", "review"),
    "success_metric": ("analyze", "cross_check", "write", "review"),
    "budget": ("analyze", "write", "review"),
    # clarify_mode is meta. Takes effect on the next clarify call,
    # which usually never happens after the initial pause. No-op.
    "clarify_mode": (),
    # review_panel only affects review. Re-run review.
    "review_panel": ("review",),
    # knowledge changes are warned, not invalidated: Axon retrievals
    # already happened (or didn't). Future calls use the new setting.
    "knowledge_enabled": (),
}


# ---------------------------------------------------------------------------
# Smart defaults
# ---------------------------------------------------------------------------

def smart_default_no_simulation(partial: dict[str, Any]) -> bool:
    """When the user already picked a prose paper_format (essay /
    report / policy_brief / whitepaper), the typical answer is
    ``no_simulation=True``. Other formats default to False (Python
    can produce data)."""
    fmt = partial.get("paper_format", "generic")
    return fmt in PROSE_FORMATS


def smart_default_study_depth(partial: dict[str, Any]) -> str:
    """policy_brief is by definition 2-4 pages → 'brief preprint'.
    The other formats keep the journal-length default unless the
    topic string contains 'survey' / 'review' / 'comparison', which
    bumps to comprehensive review."""
    fmt = partial.get("paper_format", "generic")
    if fmt == "policy_brief":
        return "brief preprint"
    topic = (partial.get("topic") or "").lower()
    if any(token in topic for token in ("survey", "review of", "compar")):
        return "comprehensive review"
    return "journal-length"


def smart_default_title(partial: dict[str, Any]) -> str:
    """Slugify the topic to ~40 chars for the title default."""
    topic = (partial.get("topic") or "").strip()
    if not topic:
        return "quest"
    return slugify(topic)[:40] or "quest"


def smart_default_clarify_mode(_partial: dict[str, Any]) -> str:
    """Auto-clarify covers ~95% of cases — the engine self-generates
    AND auto-answers the clarify questionnaire from the topic, with
    interview-pinned slots winning."""
    return "auto"


def smart_default_review_panel(_partial: dict[str, Any]) -> list[str]:
    """Single reviewer (empty panel list) is the cost-conscious
    default. Multi-persona panels cost ~4× per review pass; users
    who want them flip the slot consciously."""
    return []


def smart_default_audience(_partial: dict[str, Any]) -> str:
    """External by default — safer for a one-shot paper, since FI's
    cross-quest memory artifacts (fi_critique / fi_digest / ...)
    aren't lookups an outside reader can resolve. Internal-facing
    users flip to "internal" for project reports / memos."""
    return "external"


def smart_default_knowledge_enabled(_partial: dict[str, Any]) -> bool:
    """Probe the Axon API sidecar; enable the knowledge layer when
    it's reachable, otherwise leave it off. Saves the user from
    flipping a switch that requires Axon anyway."""
    try:
        from core.axon_sidecar import axon_status
        return bool(axon_status(timeout=0.8)["running"])
    except Exception:
        return False


def smart_default_knowledge_top_k(partial: dict[str, Any]) -> int:
    """8 is the new default that balances prior-work breadth against
    prompt size. Bump to 12 for ``study_depth: comprehensive review``
    automatically — the writer needs more sources for a survey."""
    if partial.get("study_depth") == "comprehensive review":
        return 12
    return 8


def smart_default_knowledge_external_top_k(partial: dict[str, Any]) -> int:
    """20 is the default external (web) cap — bigger than the Axon cap
    because web hits are coarser and a literature scan benefits from
    breadth. Comprehensive reviews bump to 30."""
    if partial.get("study_depth") == "comprehensive review":
        return 30
    return 20


# Map of question id → smart-default callable. Frontends call
# ``build_smart_defaults(partial)`` to get a {id: default} dict that
# considers what's already answered.
SMART_DEFAULTS: dict[str, Callable[[dict[str, Any]], Any]] = {
    "title": smart_default_title,
    "no_simulation": smart_default_no_simulation,
    "study_depth": smart_default_study_depth,
    "clarify_mode": smart_default_clarify_mode,
    "review_panel": smart_default_review_panel,
    "audience": smart_default_audience,
    "knowledge_enabled": smart_default_knowledge_enabled,
    "knowledge_top_k": smart_default_knowledge_top_k,
    "knowledge_external_top_k": smart_default_knowledge_external_top_k,
}


def build_smart_defaults(partial: dict[str, Any]) -> dict[str, Any]:
    """Compute the {question_id: default} dict given the answers
    collected so far. Frontends call this after EACH answer to
    update the next question's pre-fill."""
    out: dict[str, Any] = {}
    for q in QUESTIONS:
        if q.id in SMART_DEFAULTS:
            out[q.id] = SMART_DEFAULTS[q.id](partial)
        elif q.default is not None:
            out[q.id] = q.default
    return out


def questions_for_tier(tier: int, frontend: str = "cli") -> tuple[Question, ...]:
    """Return the tier-N questions that apply to ``frontend``.

    Frontends call ``questions_for_tier(1, frontend)`` to get the
    always-ask list, ``questions_for_tier(2, frontend)`` for the
    auto-derived review-screen fields, and ``questions_for_tier(3,
    frontend)`` for what to show behind a 'Show advanced' toggle.

    The ``frontend`` filter respects each question's ``frontends``
    tuple — VSCode skips provider/provider_model because the
    extension pins both silently.
    """
    return tuple(
        q for q in QUESTIONS
        if q.tier == tier and frontend in q.frontends
    )


def derive_tier2(tier1_answers: dict[str, Any]) -> dict[str, Any]:
    """Given the user's tier-1 answers (topic / paper_format / outputs
    / depth / provider / model), compute every tier-2 default through
    the SMART_DEFAULTS pipeline.

    Frontends show the returned dict on a 'Review before launch'
    screen with click-to-edit per field, then submit the combined
    tier-1 + tier-2 dict as the final interview answers.
    """
    merged = dict(tier1_answers)
    out: dict[str, Any] = {}
    for q in QUESTIONS:
        if q.tier != 2:
            continue
        if q.id in SMART_DEFAULTS:
            out[q.id] = SMART_DEFAULTS[q.id](merged)
        elif q.default is not None:
            out[q.id] = q.default
        else:
            out[q.id] = None
    return out


def derive_tier3(tier1_answers: dict[str, Any]) -> dict[str, Any]:
    """Same shape as ``derive_tier2`` but for the tier-3 advanced
    slots. These are usually empty strings (topic-tuned slots that
    only the preflight clarify LLM call can suggest) — the frontend
    shows them under a 'Show advanced' toggle so the user can
    type a value before launch instead of waiting for the engine
    to suggest one mid-quest."""
    merged = dict(tier1_answers)
    out: dict[str, Any] = {}
    for q in QUESTIONS:
        if q.tier != 3:
            continue
        if q.id in SMART_DEFAULTS:
            out[q.id] = SMART_DEFAULTS[q.id](merged)
        elif q.default is not None:
            out[q.id] = q.default
        else:
            out[q.id] = ""
    return out


# ---------------------------------------------------------------------------
# Slugify (shared with vscode-frontier-insight; mirrored by the TS half)
# ---------------------------------------------------------------------------

_SLUG_STRIP = re.compile(r"[^a-z0-9\s-]")
_SLUG_SPACE = re.compile(r"\s+")
_SLUG_DUP_DASH = re.compile(r"-+")


def slugify(s: str) -> str:
    """Lowercase + remove non-alphanum + collapse spaces to dashes.
    Mirrors ``vscode-frontier-insight/src/interview-core.ts:slugify``."""
    lowered = s.lower()
    stripped = _SLUG_STRIP.sub("", lowered)
    dashed = _SLUG_SPACE.sub("-", stripped)
    collapsed = _SLUG_DUP_DASH.sub("-", dashed)
    return collapsed.strip("-")


# ---------------------------------------------------------------------------
# Provider availability
# ---------------------------------------------------------------------------

# Map provider id → environment probe. The interview uses this to
# decide which providers to highlight as "available" in the CLI
# picker. A provider that fails its probe is still selectable (the
# user may know they're about to set up auth), but it shows a hint.
_PROVIDER_PROBES: dict[str, Callable[[], bool]] = {
    "openai": lambda: bool(os.environ.get("OPENAI_API_KEY", "").strip()),
    "codex": lambda: bool(os.environ.get("OPENAI_API_KEY", "").strip()),
    "claude_cli": lambda: shutil.which("claude") is not None,
    "codex_cli": lambda: shutil.which("codex") is not None,
    "copilot_cli": lambda: shutil.which("gh") is not None,
    "gemini_cli": lambda: shutil.which("gemini") is not None,
    "ollama": lambda: shutil.which("ollama") is not None,
}


def available_providers() -> list[str]:
    """Return the subset of provider ids whose environment probe
    passes. CLI frontends mark these with a "✓ ready" hint; the
    others fall through with an "ⓘ needs setup" hint."""
    return [name for name, probe in _PROVIDER_PROBES.items() if probe()]


def model_choices_for(provider: str) -> tuple[Choice, ...]:
    """Look up the curated model list for a provider. Returns an
    empty tuple when the provider isn't in the catalog — the frontend
    should then fall through to a free-text input."""
    return PROVIDER_MODEL_OPTIONS.get(provider, ())


# ---------------------------------------------------------------------------
# Preflight clarify (one LLM call for topic-tuned defaults)
# ---------------------------------------------------------------------------

_STATIC_PREFLIGHT_DEFAULTS = {
    "comparative_baseline": "(none specified — agent will pick a sensible baseline)",
    "success_metric": "(none specified — agent will pick a metric)",
    "budget": "a few minutes on a laptop CPU",
}


async def preflight_clarify(
    topic: str,
    paper_format: str,
    *,
    provider_name: str | None,
    provider_model: str | None = None,
    timeout_s: float = 30.0,
) -> dict[str, str]:
    """One LLM call that produces topic-tuned defaults for the three
    interview slots that need topic context: comparative_baseline,
    success_metric, budget.

    Returns a dict with those three keys mapped to LLM-generated
    strings. On timeout, no-provider, or any error, returns the
    static fallback dict — the interview never blocks indefinitely.

    Frontends typically render a visible spinner while this runs;
    the spinner copy is part of the frontend layer, not this module.
    """
    fallback = dict(_STATIC_PREFLIGHT_DEFAULTS)
    if not provider_name or not topic.strip():
        return fallback

    prompt_path = Path(__file__).resolve().parent.parent / "agents" / "clarify_preflight.md"
    try:
        prompt_template = prompt_path.read_text(encoding="utf-8")
    except OSError:
        return fallback
    prompt = (
        prompt_template
        .replace("$topic", topic.strip())
        .replace("$paper_format", paper_format or "generic")
    )

    # Lazy-import the provider layer so this module stays importable
    # in tests that don't need a real LLM.
    try:
        from core.provider import build_provider  # type: ignore[import-not-found]
    except Exception:
        return fallback

    try:
        provider = build_provider(provider_name, model=provider_model)
    except Exception:
        return fallback

    import asyncio

    try:
        response = await asyncio.wait_for(
            provider.chat(prompt),
            timeout=timeout_s,
        )
    except (asyncio.TimeoutError, Exception):
        return fallback

    parsed = _parse_preflight_json(response)
    if parsed is None:
        return fallback

    # Fill any missing key from the static fallback so callers always
    # get the three expected keys.
    return {**fallback, **parsed}


def _parse_preflight_json(text: str) -> dict[str, str] | None:
    """Extract the JSON object from the LLM's response. Tolerant of
    fenced code blocks and surrounding prose — the preflight prompt
    asks for a bare object but LLMs add fences anyway."""
    # Strip fences if present.
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if m is None:
        return None
    try:
        parsed = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    out: dict[str, str] = {}
    for key in ("comparative_baseline", "success_metric", "budget"):
        val = parsed.get(key)
        if isinstance(val, str) and val.strip():
            out[key] = val.strip()
    return out or None


# ---------------------------------------------------------------------------
# YAML emitter
# ---------------------------------------------------------------------------

@dataclass
class InterviewAnswers:
    """Container for collected answers. Mirrors
    ``vscode-frontier-insight/src/interview-core.ts:InterviewAnswers``
    in field names so the YAML emitters produce identical shapes."""

    topic: str
    title: str
    output_kinds: list[str]
    paper_format: str
    no_simulation: bool
    study_depth: str
    comparative_baseline: str
    success_metric: str
    budget: str
    clarify_mode: str
    review_panel: list[str]
    knowledge_enabled: bool
    # CLI / serve only — VSCode pins ``vscode_extension`` silently.
    provider: str = "vscode_extension"
    provider_model: str | None = None
    max_iterations: int = 2
    # Audience for the published paper. "external" drops FI-internal
    # cross-quest memory from the References section; "internal"
    # keeps everything. Default external for safe one-shot quests.
    audience: str = "external"
    # How many Axon (RAG) prior-work entries the literature node
    # retrieves into the writer's context block. 8 is the default;
    # the smart-default helper bumps it to 12 for comprehensive reviews.
    knowledge_top_k: int = 8
    # How many EXTERNAL (web search — arXiv / OpenAlex / Crossref /
    # S2 / ...) hits to fetch when Axon misses. Bigger than the Axon
    # cap because web search is coarser; default 20 → 30 for surveys.
    knowledge_external_top_k: int = 20
    # Multi-model ensemble preset. Expanded by ``answers_to_yaml`` into
    # the ``provider.node_ensemble`` block when non-"off". See
    # ``ENSEMBLE_PROFILES`` for the four options and their cost
    # multipliers; ``ensemble_model_trio`` for the per-provider models.
    ensemble_profile: str = "off"


def expand_ensemble_profile(
    profile: str, *, provider: str, provider_model: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Expand a profile name to a ``node_ensemble`` dict shape.

    Returns a dict ready to plug into ``provider.node_ensemble`` in the
    YAML, mapping engine-node name → ``{models, merge[, moderator]}``.
    Returns an empty dict for ``off`` so the YAML emitter can omit the
    block entirely (no regression for users who don't pick a profile).

    The shape mirrors ``core.config.NodeEnsembleConfig`` but stays a
    plain dict so this module doesn't import from ``core.config`` (that
    would create a circular import via ``core.config``'s use of pydantic
    + ``core.interview`` schema-export plumbing).
    """
    if profile == "off" or not profile:
        return {}
    trio = ensemble_model_trio(provider, provider_model)
    moderator = trio[0]
    nodes: dict[str, dict[str, Any]] = {}
    # cross_check: every profile that fans out includes it. Vote is the
    # only safe merger here (engine's cross_check parses ``verdict``);
    # the moderator field is unused for vote and intentionally omitted.
    if profile in ("cross_check_only", "ideate_and_check", "full"):
        nodes["cross_check"] = {"models": trio, "merge": "vote"}
    if profile in ("ideate_and_check", "full"):
        nodes["ideate"] = {"models": trio, "merge": "tournament", "moderator": moderator}
    if profile == "full":
        # analyze MUST use tournament — synthesize is rejected by the
        # ProviderConfig validator because analyze's downstream parser
        # expects JSON, which only tournament's verbatim winner preserves.
        nodes["analyze"] = {"models": trio, "merge": "tournament", "moderator": moderator}
    return nodes


def answers_to_yaml(answers: InterviewAnswers, *, frontend: str = "cli") -> str:
    """Render the interview answers as a complete config.yaml string.

    ``frontend`` ("cli" / "vscode" / "serve") controls the leading
    comment header and a handful of cost-discipline defaults that
    differ between surfaces:

    * ``"vscode"`` writes the original cost-discipline trio
      (``ideate_reflect: false`` etc.) for parity with the existing
      VSCode interview output.
    * ``"cli"`` and ``"serve"`` keep the engine defaults — the CLI
      user gets the full research-quality loop unless they edit YAML.
    """
    indent = "  "
    lines: list[str] = []

    if frontend == "vscode":
        lines.append("# Auto-generated by the Frontier Insight VSCode extension.")
        lines.append("# You can edit this file and re-run with `@fi /start <path>`.")
    elif frontend == "serve":
        lines.append("# Auto-generated by the Frontier Insight web UI (--serve).")
    else:
        lines.append("# Auto-generated by Frontier Insight `python launch.py --new`.")
    lines.append("# Edit and re-run with `python launch.py --config <path>`.")
    lines.append("")

    lines.append("topic: |")
    for line in answers.topic.split("\n"):
        lines.append(f"{indent}{line}")
    lines.append("")

    lines.append(f"title: {json.dumps(answers.title)}")
    lines.append("")

    lines.append("provider:")
    lines.append(f"{indent}name: {json.dumps(answers.provider)}")
    if answers.provider_model:
        lines.append(f"{indent}model: {json.dumps(answers.provider_model)}")
    # Multi-model ensemble preset. Only emit when the user picked
    # something non-"off"; otherwise leave the engine on its
    # single-model path (no regression for default quests).
    node_ensemble = expand_ensemble_profile(
        answers.ensemble_profile,
        provider=answers.provider,
        provider_model=answers.provider_model,
    )
    if node_ensemble:
        lines.append(f"{indent}node_ensemble:")
        for node, cfg in node_ensemble.items():
            lines.append(f"{indent}{indent}{node}:")
            quoted_models = ", ".join(json.dumps(m) for m in cfg["models"])
            lines.append(f"{indent}{indent}{indent}models: [{quoted_models}]")
            lines.append(f"{indent}{indent}{indent}merge: {json.dumps(cfg['merge'])}")
            if "moderator" in cfg:
                lines.append(f"{indent}{indent}{indent}moderator: {json.dumps(cfg['moderator'])}")
    lines.append("")

    lines.append("engine:")
    lines.append(f"{indent}framework: \"langgraph\"")
    lines.append(f"{indent}max_iterations: {answers.max_iterations}")
    lines.append(f"{indent}review_loop: true")
    lines.append(f"{indent}clarify_mode: {json.dumps(answers.clarify_mode)}")
    lines.append(f"{indent}no_simulation: {'true' if answers.no_simulation else 'false'}")

    # Cost-discipline trio — preserved for VSCode parity. CLI and
    # serve users get the engine defaults.
    if frontend == "vscode":
        lines.append(f"{indent}ideate_reflect: false")
        lines.append(f"{indent}cross_check_per_finding_k: 0")
        lines.append(f"{indent}enable_analyze_reroute: false")

    if answers.review_panel:
        lines.append(f"{indent}review_panel:")
        for persona in answers.review_panel:
            lines.append(f"{indent}{indent}- {json.dumps(persona)}")
    else:
        lines.append(f"{indent}review_panel: []")

    # Clarify overrides — pin the three topic-tuned slots + study_depth +
    # paper_venue + output_kinds + simulatability so the clarify node
    # short-circuits in auto mode. Keys mirror agents/clarify.md.
    overrides: dict[str, Any] = {
        "study_depth": answers.study_depth,
        "paper_venue": answers.paper_format,
        "output_kinds": answers.output_kinds,
        "simulatability": "no" if answers.no_simulation else "yes",
        "comparative_baseline": answers.comparative_baseline,
        "success_metric": answers.success_metric,
        "budget": answers.budget,
    }
    lines.append(f"{indent}clarify_overrides:")
    for key, value in overrides.items():
        if isinstance(value, list):
            lines.append(f"{indent}{indent}{key}: {json.dumps(value)}")
        else:
            lines.append(f"{indent}{indent}{key}: {json.dumps(value)}")
    lines.append("")

    lines.append("execution:")
    lines.append(f"{indent}sandbox: \"venv\"")
    lines.append(f"{indent}timeout_s: 600")
    lines.append("")

    lines.append("knowledge:")
    lines.append(f"{indent}enabled: {'true' if answers.knowledge_enabled else 'false'}")
    if answers.knowledge_top_k != 8:
        lines.append(f"{indent}top_k: {answers.knowledge_top_k}")
    if answers.knowledge_external_top_k != 20:
        lines.append(f"{indent}external_top_k: {answers.knowledge_external_top_k}")
    if not answers.knowledge_enabled:
        lines.append(f"{indent}external_fallback: [\"openalex\", \"arxiv\", \"crossref\"]")
        lines.append(f"{indent}source_routing: \"manual\"")
    lines.append("")

    lines.append("output:")
    quoted = ", ".join(json.dumps(k) for k in answers.output_kinds)
    lines.append(f"{indent}kinds: [{quoted}]")
    lines.append(f"{indent}paper_format: {json.dumps(answers.paper_format)}")
    if answers.audience != "external":
        lines.append(f"{indent}audience: {json.dumps(answers.audience)}")
    lines.append(f"{indent}output_dir: \"./outputs\"")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Schema export — the JSON the TS + HTMX frontends consume
# ---------------------------------------------------------------------------

def export_schema_json() -> dict[str, Any]:
    """Render QUESTIONS + choice catalogs + invalidation matrix as a
    JSON-serializable dict. Snapshot lands at
    ``core/interview_schema.json`` (checked in). The TS interview
    reads this at extension activation; the HTMX form reads it at
    server startup."""
    def _choice(c: Choice) -> dict[str, Any]:
        return {"value": c.value, "label": c.label, "description": c.description}

    questions_json: list[dict[str, Any]] = []
    for q in QUESTIONS:
        q_dict = asdict(q)
        q_dict["choices"] = [_choice(c) for c in q.choices]
        q_dict["frontends"] = list(q.frontends)
        questions_json.append(q_dict)

    return {
        "version": 1,
        "questions": questions_json,
        "paper_formats": [_choice(c) for c in PAPER_FORMATS],
        "prose_formats": sorted(PROSE_FORMATS),
        "study_depths": [_choice(c) for c in STUDY_DEPTHS],
        "clarify_modes": [_choice(c) for c in CLARIFY_MODES],
        "review_panels": [_choice(c) for c in REVIEW_PANELS],
        "audience_choices": [_choice(c) for c in AUDIENCE_CHOICES],
        "ensemble_profiles": [_choice(c) for c in ENSEMBLE_PROFILES],
        "ensemble_cost_multipliers": dict(_ENSEMBLE_COST_MULTIPLIERS),
        "ensemble_model_trios": {
            name: list(trio) for name, trio in _ENSEMBLE_MODEL_TRIOS.items()
        },
        "providers": [_choice(c) for c in PROVIDER_CHOICES],
        "provider_models": {
            name: [_choice(c) for c in opts]
            for name, opts in PROVIDER_MODEL_OPTIONS.items()
        },
        "editable_fields": sorted(EDITABLE_FIELDS),
        "stage_invalidation": {
            field: list(stages) for field, stages in STAGE_INVALIDATION.items()
        },
    }


def write_schema_json(path: Path | None = None) -> Path:
    """Write the schema JSON to disk. Called by the
    ``export-interview-schema`` script + the CI parity guard."""
    if path is None:
        path = Path(__file__).resolve().parent / "interview_schema.json"
    payload = export_schema_json()
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
