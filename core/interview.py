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

from core.config import _RESEARCH_PROFILE, REQUIRED_REVIEW_ROLES

# The panel the research profile runs when none is written (core/config.py's profile, one source of truth).
RESEARCH_REVIEW_PANEL: tuple[str, ...] = tuple(_RESEARCH_PROFILE["engine"]["review_panel"])


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
    #   tier=1 — always-ask: the topic, what the result is for, and on
    #            CLI/web the provider + model. The author line is tier 1
    #            too but asked only while no profile is saved
    #            (core/profile.py): the first interview asks it, later
    #            ones fill it from the profile.
    #
    #   tier=2 — auto-derive. Smart defaults populate these from tier-1
    #            answers (title from slug, paper format, deliverables,
    #            study depth, no_simulation from format, knowledge from
    #            Axon-sidecar status, etc.). Frontends still SHOW them on
    #            a review screen so the user can click-to-edit before
    #            launch.
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

# How paper.pdf is styled (output.paper_style). Keep in sync with PaperStyle
# in core/config.py.
PAPER_STYLES: tuple[Choice, ...] = (
    Choice("latex", "LaTeX — Computer Modern article (default)",
           "The classic typeset look via the venue LaTeX template. Best typography; needs a LaTeX engine (or the HTML fallback)."),
    Choice("briefing", "Briefing — Frontier Insight brand look",
           "Warm paper, deep-teal accents, serif display + brand mark — the same identity as the slides and poster. Rendered via pandoc + a browser (no LaTeX); single-column regardless of venue."),
)

# Poster sheet sizes. Mirrors ``OutputConfig.poster_size`` in core/config.py.
POSTER_SIZES: tuple[Choice, ...] = (
    Choice("a1_portrait", "A1 portrait — 59.4 × 84.1 cm (default)",
           "Two columns. The usual conference poster size outside North America."),
    Choice("a0_portrait", "A0 portrait — 84.1 × 118.9 cm",
           "Two columns on a larger sheet, so it holds more words."),
    Choice("landscape_48x36", "48 × 36 in landscape — 121.9 × 91.4 cm",
           "Three columns. A common size at North American conferences."),
)

# Reasoning effort. Mirrors ``ProviderConfig.reasoning_effort`` in
# core/config.py; "default" is the interview's name for "unset" and writes
# nothing to the YAML.
REASONING_EFFORT_CHOICES: tuple[Choice, ...] = (
    Choice("default", "Provider default (not set)",
           "Sends nothing: each provider keeps its own default. A local Ollama model then does not think at all."),
    Choice("minimal", "minimal",
           "Not accepted by Ollama, claude_cli or antigravity_cli; skipped there with a warning."),
    Choice("low", "low", "Accepted by every provider FI can set it on."),
    Choice("medium", "medium", "Accepted by every provider FI can set it on."),
    Choice("high", "high", "Accepted by every provider FI can set it on."),
    Choice("xhigh", "xhigh",
           "Not accepted by Ollama or antigravity_cli; skipped there with a warning."),
    Choice("max", "max",
           "Not accepted by Ollama or antigravity_cli; skipped there with a warning."),
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
    Choice(["methodologist", "statistician", "devil_advocate"], "3-persona panel (default)",
           "Methodologist + Statistician + Devil's-advocate + moderator. ~4× the review cost. "
           "Methodologist is the must-flag-rule reviewer — drop the panel and you lose those checks."),
    Choice([], "Single reviewer (cheapest, no must-flag enforcement)",
           "1 LLM call per review pass. Loses the methodologist must-flag rules."),
    Choice(["methodologist", "statistician", "devil_advocate", "reproducibility"], "4-persona panel",
           "Adds Reproducibility reviewer (seeds, raw data, the protocol; worth it for a stochastic simulation study). ~5× the review cost."),
)


KNOWLEDGE_CHOICES: tuple[Choice, ...] = (
    Choice(False, "Disabled — turns OFF all retrieval",
           "Master switch: no Axon, no academic search (arXiv/OpenAlex/Crossref), and no web search. The quest runs on the model's own knowledge only."),
    Choice(True, "Enabled (recommended)",
           "Turns retrieval ON. Axon is used as the corpus only if installed — academic + web search work either way, so enable this even without Axon."),
)


NO_SIMULATION_CHOICES: tuple[Choice, ...] = (
    Choice(False, "Computational — a Python script can produce the data",
           "Run normal implement → execute pipeline. For physics / ML / algorithmic / benchmark topics."),
    Choice(True, "Observational — needs real-world data the engine can't simulate",
           "Skip implement / execute; route to wait_for_data + auto_collect_data. For cultural / historical / qualitative / policy topics."),
)


SURVEY_MODE_CHOICES: tuple[Choice, ...] = (
    Choice(False, "No — run an experiment or analyse data (default)",
           "Normal pipeline: the engine designs an experiment (or, for observational topics, analyses collected data)."),
    Choice(True, "Yes — a literature / history synthesis, no experiment or data",
           "Skip BOTH the experiment AND the dataset. The paper is a descriptive history / overview synthesised from the cited sources, with license-clean illustrative images. For 'history of X' / 'evolution of X' / overview topics. Implies observational."),
)


WEB_RESEARCH_CHOICES: tuple[Choice, ...] = (
    Choice(True, "On (recommended)",
           "Search the public web (Brave / DuckDuckGo) for current sources and download them into the quest's data/literature/ folder — for BOTH simulation and observational quests. Adds real-world reports / news on top of academic retrieval."),
    Choice(False, "Off",
           "Skip the general web layer; rely on academic sources (and Axon, if enabled) only. No web pages are searched or downloaded."),
)


SUPPLY_PAPERS_CHOICES: tuple[Choice, ...] = (
    Choice(False, "Off",
           "Use whatever the open-web / open-access fetch can get. Paywalled papers (SPIE, IEEE, …) are cited by their abstract only. Choose this for an unattended run."),
    Choice(True, "Pause for my PDFs (default)",
           "When the agent can only get the abstract of a relevant paper, it pauses and writes a ranked needs/WANTED_PAPERS.md (download links + why each matters). Drop the PDFs into inputs/papers/ and resume — they're ingested as full text. Tip: you can also pre-load a folder of papers via knowledge.local_papers."),
)


PAUSE_FOR_PLAN_CHOICES: tuple[Choice, ...] = (
    Choice(False, "Don't stop (default)",
           "The plan is written to plan.md in the quest folder, and the experiment is designed from it, but the quest does not wait. Choose this for an unattended run."),
    Choice(True, "Stop to read and edit the plan",
           "After the literature is in, the quest writes plan.md (what the literature says, the gap, the design) and stops. Edit the file, or ask for a change (--revise-plan, the quest page's Plan box, @fi /plan), then resume: the design block in the file is what runs."),
)


RESULT_USE_CHOICES: tuple[Choice, ...] = (
    Choice("research", "Research (default)",
           "Checked the way a study must be before its result can be trusted: the plan waits for you to read it, "
           "every check stops the quest instead of only reporting, the experiment runs in its own clean environment, "
           "and four reviewers (method, statistics, reproducibility, devil's advocate) read the paper. Slower, and it "
           "stops for you more often."),
    Choice("decision", "A decision",
           "The same checks as Research: a result someone will act on needs every one of them."),
    Choice("explore", "Exploring (cheaper draft)",
           "A quick look: fewer model calls (no self-critique of the ideas, no per-finding cross-check, no redesign "
           "after the analysis), and the result is a preliminary draft, never ready to publish as it stands."),
)

# What each "result use" answer turns into. ``explore`` keeps rigor_profile at its default and writes the three
# cost-saving engine settings below; ``research`` and ``decision`` are the research profile. The same on every
# interface: the setting follows the person's answer, never the interface they answered in.
DRAFT_ENGINE_SETTINGS: tuple[tuple[str, str], ...] = (
    ("ideate_reflect", "false"),
    ("cross_check_per_finding_k", "0"),
    ("enable_analyze_reroute", "false"),
)


def rigor_profile_for(result_use: str) -> str:
    """The ``rigor_profile`` a "result use" answer sets: ``default`` for exploring, ``research`` otherwise."""
    return "default" if result_use == "explore" else "research"


def resolve_review_panel(panel: list[str], result_use: str) -> list[str]:
    """The reviewer panel as it will run: under the research profile every role it requires is added (a panel
    without them is refused at start). Called before anything is shown, so the confirm screen lists the panel
    that runs, not the one picked before the requirement was applied."""
    panel = list(panel)
    if rigor_profile_for(result_use) != "research":
        return panel
    if not panel:
        # A single reviewer is not available under the research profile, which runs its own four-person panel.
        return list(RESEARCH_REVIEW_PANEL)
    return [*panel, *[r for r in REQUIRED_REVIEW_ROLES if r not in panel]]


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


# An ensemble fans one node out over several models and merges what they say, so it
# needs at least two. FI never picks them: which models are worth the money, and which
# the user has access to, is the user's decision. The interview asks; a profile without
# models configures nothing.
ENSEMBLE_MIN_MODELS = 2


def parse_ensemble_models(raw: Any) -> list[str]:
    """The model ids the user named for an ensemble: a comma / semicolon / newline
    separated string, or a list. Order kept, blanks and repeats dropped."""
    if raw is None:
        return []
    parts = re.split(r"[,;\n]", raw) if isinstance(raw, str) else list(raw)
    out: list[str] = []
    for part in parts:
        model = str(part).strip()
        if model and model not in out:
            out.append(model)
    return out


# The nodes measured as safe to run on a cheaper model. Each is short-output or template
# work: cross_check judges retrieved candidates, select_skills picks skill names,
# literature_screen filters candidate sources, slides and poster pour a finished paper into
# a template. Measured on one simulation topic, three runs per arm (codex terra everywhere vs
# these five on luna, same effort): 15% fewer tokens, and no drop in paper, slide or poster
# scores that three runs could show. Three runs cannot show a drop under about ten points, so
# this is "not seen", not "ruled out". Every other node is untested and is left to the user.
# FI never picks the cheaper model: it does not know which models a user has, or what they cost.
LIGHT_NODES: tuple[str, ...] = (
    "cross_check", "select_skills", "literature_screen", "slides", "poster",
)


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
        Choice("gpt-5", "gpt-5", "Latest flagship. Best overall quality."),
        Choice("gpt-5-mini", "gpt-5-mini", "Cheap; good for high-volume nodes."),
        Choice("gpt-4o", "gpt-4o", "Previous-generation default; still strong."),
        Choice("o3", "o3", "Reasoning model; slower + pricier."),
        Choice("o3-mini", "o3-mini", "Cheaper reasoning model."),
    ),
    "codex": (
        Choice("gpt-5", "gpt-5", "ChatGPT backend; matches Plus/Pro web UI."),
        Choice("gpt-5-mini", "gpt-5-mini", "Cheaper Plus/Pro backend variant."),
        Choice("gpt-4o", "gpt-4o", "Previous-generation default."),
    ),
    "claude_cli": (
        Choice("claude-opus-4-7", "claude-opus-4-7", "Latest Opus; strongest model."),
        Choice("claude-sonnet-4-6", "claude-sonnet-4-6", "Latest Sonnet; balanced."),
        Choice("claude-haiku-4-5-20251001", "claude-haiku-4-5", "Latest Haiku; cheapest."),
    ),
    "codex_cli": (
        Choice("gpt-5", "gpt-5", "ChatGPT backend default."),
        Choice("gpt-5-mini", "gpt-5-mini", "Cheaper ChatGPT backend variant."),
        Choice("gpt-4o", "gpt-4o", "Previous-generation default."),
    ),
    "copilot_cli": (
        Choice("gpt-5", "gpt-5", "Copilot Pro/Business default since 2025."),
        Choice("claude-opus-4-7", "claude-opus-4-7", "Available to Copilot Business+."),
        Choice("claude-sonnet-4-6", "claude-sonnet-4-6", "Cheaper Anthropic option."),
        Choice("gemini-2.5-pro", "gemini-2.5-pro", "Available when Copilot federates Gemini."),
    ),
    "gemini_cli": (
        Choice("gemini-2.5-pro", "gemini-2.5-pro", "Latest Pro; long context."),
        Choice("gemini-2.5-flash", "gemini-2.5-flash", "Cheaper / faster."),
        Choice("gemini-2.0-flash", "gemini-2.0-flash", "Previous-generation flash."),
    ),
    "ollama": (
        Choice("llama3.3:70b", "llama3.3:70b", "Latest large Llama; needs ~48GB VRAM."),
        Choice("llama3.2:3b", "llama3.2:3b", "Tiny; fits a laptop GPU."),
        Choice("qwen2.5:32b", "qwen2.5:32b", "Strong open-source mid-size."),
        Choice("qwen2.5:7b", "qwen2.5:7b", "Lighter Qwen for laptops."),
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
        id="result_use",
        label="What is the result for?",
        prompt="Research or a decision gets every check the result needs before it can be trusted; exploring is a cheaper, preliminary draft.",
        kind="single",
        choices=RESULT_USE_CHOICES,
        default="research",
        # Fixed for the quest: it sets how strictly the experiment is checked, which a quest cannot change midway.
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
        tier=2,
    ),
    Question(
        id="output_kinds",
        label="Outputs",
        prompt="Which deliverables matter? PDF gracefully degrades to MD if pandoc isn't installed.",
        kind="single",  # bundles are pre-defined combos, not per-kind multi-select.
        choices=OUTPUT_BUNDLES,
        default=["paper_md", "paper_pdf"],
        mid_quest_editable=True,
        tier=2,
    ),
    Question(
        id="paper_style",
        label="Paper style",
        prompt="How should paper.pdf look? 'latex' is the classic Computer Modern article; 'briefing' is the Frontier Insight brand look (warm paper + teal, matching the slides/poster) rendered without LaTeX.",
        kind="single",
        choices=PAPER_STYLES,
        default="latex",
        mid_quest_editable=True,
        tier=3,
    ),
    Question(
        id="poster_size",
        label="Poster size",
        prompt="Sheet size for poster.pdf. Text meets the poster font-size standards at every size; a bigger sheet holds more words.",
        kind="single",
        choices=POSTER_SIZES,
        default="a1_portrait",
        mid_quest_editable=True,
        tier=3,
    ),
    Question(
        id="study_depth",
        label="Study depth",
        prompt="How mature should the resulting paper be? Drives word count + citation depth.",
        kind="single",
        choices=STUDY_DEPTHS,
        default="journal-length",
        mid_quest_editable=True,
        tier=2,
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
        id="survey_mode",
        label="Literature synthesis (survey) mode",
        prompt="Is this a pure literature / history synthesis (a history, overview, or evolution of a subject) with NO experiment and NO dataset? If yes, the engine skips both and writes a descriptive synthesis. Auto-suggested for history/overview topics.",
        kind="single",
        choices=SURVEY_MODE_CHOICES,
        default=False,
        # Like no_simulation, flipping this mid-quest reroutes the graph.
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
        prompt=(
            "Recommended: 3-persona panel — the methodologist is the must-flag-rule reviewer; "
            "dropping the panel loses those checks. Single-reviewer is cheaper but ships "
            "without the non-bypassable design-quality enforcement."
        ),
        kind="single",
        choices=REVIEW_PANELS,
        default=["methodologist", "statistician", "devil_advocate"],
        mid_quest_editable=True,
        tier=2,
    ),
    Question(
        id="pause_for_user_input",
        label="Pause for user-supplied papers / datasets",
        prompt="Pause mid-quest so you can drop reference PDFs into inputs/papers/ and datasets into inputs/data/ before the engine continues. After-literature stops once the literature is saved, so the experiment is designed with it in hand; after-design lets you correct the methodology; after-paper lets you augment the first draft. Resume with `fi --resume <quest_id>`.",
        kind="single",
        choices=(
            Choice("never", "Never (default)",
                   "Engine runs to completion without pause-drop opportunities."),
            Choice("after_literature", "Pause after literature",
                   "Stop once the literature is saved; skills, design and the experiment start on resume with it in hand (the search is not run again)."),
            Choice("after_design", "Pause after design",
                   "Drop reference papers / data BEFORE the implement → execute → analyze stages spend compute."),
            Choice("after_paper", "Pause after paper draft",
                   "Drop reference papers / data AFTER the first paper.md is written; useful for revise iterations."),
            Choice("both", "Both",
                   "Pause after design AND after paper — most cost. Use for high-stakes manual review."),
        ),
        default="never",
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
        id="web_research",
        label="Web research",
        prompt="Search the public web for current sources and download them into the quest's data/literature/ folder. Runs for simulation AND observational quests, on top of academic retrieval. Does NOT require Axon, but it does require the knowledge layer above to be Enabled — that setting is the master switch for all retrieval.",
        kind="single",
        choices=WEB_RESEARCH_CHOICES,
        default=True,
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
        id="supply_papers",
        label="Supply paywalled papers",
        prompt="When a relevant paper is paywalled (SPIE / IEEE / Elsevier …) and only its abstract is reachable, pause and list the papers to download (ranked, with links) so you can drop the PDFs into inputs/papers/ and resume with real full text. On by default.",
        kind="single",
        choices=SUPPLY_PAPERS_CHOICES,
        default=True,
        mid_quest_editable=True,
        tier=3,
    ),
    Question(
        id="pause_for_plan",
        label="Stop to read and edit the plan",
        prompt="Every quest writes its plan (plan.md: what the literature says, the gap, the design the experiment will run) after the literature and before the experiment is designed. Stop there to read it and edit it, or ask for a change, before any compute is spent.",
        kind="single",
        choices=PAUSE_FOR_PLAN_CHOICES,
        default=False,
        mid_quest_editable=True,
        tier=3,
    ),
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
        # Advanced: it multiplies the cost, and the models are the
        # person's to name, so it is off unless they open Advanced.
        tier=3,
    ),
    Question(
        id="ensemble_models",
        label="Models for the ensemble",
        prompt="Only when you picked a fan-out profile above: the models to fan out over, one id per model, comma separated (at least 2). FI does not choose them for you — which models are worth the cost, and which you have access to, is your call. Leave empty and no ensemble is configured.",
        kind="text",
        default="",
        placeholder="e.g. model-a, model-b, model-c",
        mid_quest_editable=True,
        tier=3,
    ),
    Question(
        id="max_iterations",
        label="Design-revise iteration budget",
        prompt="Hard cap on the design → review → revise loop (and the cross_check redirects to design / literature). Lower = cheaper + faster; higher = more chances to fix issues review caught. 2 default — bump to 3-4 only when you specifically want extra revise passes.",
        kind="text",
        default=2,
        placeholder="2",
        mid_quest_editable=True,
        tier=3,
    ),
    Question(
        id="node_models",
        label="Per-node model overrides",
        prompt="Route specific nodes to a different model than the primary one, e.g. a cheaper model for the light nodes. Measured on one simulation topic (3 runs each), moving cross_check, select_skills, literature_screen, slides and poster to a cheaper model used about 15% fewer tokens and no drop in scores was seen; three runs cannot rule out a small one, and every other node is untested. Comma-separated node:model pairs; leave blank to use the primary model everywhere. FI does not choose the model for you, and the name must be one your provider's current catalogue actually has: it is passed straight through, not validated.",
        kind="text",
        default="",
        placeholder="cross_check:MODEL, select_skills:MODEL, literature_screen:MODEL, slides:MODEL, poster:MODEL",
        mid_quest_editable=True,
        tier=3,
    ),
    Question(
        id="reasoning_effort",
        label="Reasoning effort",
        prompt="How hard the model reasons before it answers. 'default' sends nothing, so each provider keeps its own default (a local Ollama model then does not think at all). A level is sent as reasoning_effort to HTTP providers (Ollama takes low/medium/high), as --effort to claude_cli and antigravity_cli, and as model_reasoning_effort to codex_cli. copilot_cli, gemini_cli, the VS Code bridge and the proxy providers have no such setting: the level is not sent and the quest log says so once.",
        kind="single",
        choices=REASONING_EFFORT_CHOICES,
        default="default",
        mid_quest_editable=True,
        tier=3,
    ),
    Question(
        id="provider_base_url",
        label="Custom endpoint (base_url)",
        prompt="Only for openai/codex/gemini/ollama/vllm: the base URL of an OpenAI-compatible endpoint other than that provider's own default (a self-hosted vLLM server, a corporate gateway, or a third-party model reached the OpenAI-compatible way, such as Moonshot's Kimi at https://api.moonshot.ai/v1). Leave blank to use the provider's own default endpoint.",
        kind="text",
        default="",
        placeholder="https://api.moonshot.ai/v1",
        mid_quest_editable=False,
        frontends=("cli", "serve"),
        tier=3,
    ),
    Question(
        id="provider_api_key_env",
        label="Key environment variable",
        prompt="Only when the key for the endpoint above is not in that provider's usual variable (OPENAI_API_KEY, GEMINI_API_KEY, ...): the name of the environment variable (or a line in your .env file) FI should read instead. Leave blank to use the provider's own conventional name.",
        kind="text",
        default="",
        placeholder="MOONSHOT_API_KEY",
        mid_quest_editable=False,
        frontends=("cli", "serve"),
        tier=3,
    ),
    Question(
        id="provider_fixed_temperature",
        label="Fixed temperature",
        prompt="Only for a model that answers just one temperature and rejects any other with HTTP 400 (Moonshot's Kimi K2.6 / K3: 0.6 with thinking off, 1 with it on). When set, sent on every call in place of FI's own per-node temperatures. Leave blank to use FI's defaults.",
        kind="text",
        default="",
        placeholder="0.6",
        mid_quest_editable=False,
        frontends=("cli", "serve"),
        tier=3,
    ),
    Question(
        id="page_limit",
        label="Page limit",
        prompt="The most pages paper.pdf may take, as a whole number. Leave blank (or type none) for no set limit; a limit the topic states, such as '≤ 4 pages', still applies. With a limit the paper gets a tighter layout and the writer a word budget, and each draft is rendered and its pages counted: a draft over the limit is written again, shorter, at most twice.",
        kind="text",
        default="",
        placeholder="e.g. 4",
        mid_quest_editable=True,
        tier=3,
    ),
    # ─── Author line (tier 1, every field optional) ──────────────────
    # Printed on the paper, slides and poster. Asked on the first
    # interview on any frontend and kept in the profile
    # (core/profile.py); later interviews fill it from there.
    Question(
        id="author",
        label="Author (optional)",
        prompt="Your name as it should appear on the paper, slides and poster. Leave blank to keep the 'Frontier Insight' byline.",
        kind="text",
        placeholder="e.g. Jane Chen",
        default="",
        mid_quest_editable=True,
        tier=1,
    ),
    Question(
        id="affiliation",
        label="Affiliation (optional)",
        prompt="Lab, company or school to print under the author. Leave blank to omit.",
        kind="text",
        placeholder="e.g. Materials Lab, Example University",
        default="",
        mid_quest_editable=True,
        tier=1,
    ),
    Question(
        id="contact_email",
        label="Contact email (optional)",
        prompt="Printed on the poster and under the paper title so readers can reach you. It goes only into your own output files. Leave blank to omit.",
        kind="text",
        placeholder="e.g. jane@example.org",
        default="",
        mid_quest_editable=True,
        tier=1,
    ),
    Question(
        id="url",
        label="Project link (optional)",
        prompt="A web page for this work, such as a repository or lab page. The poster prints it as a QR code. Leave blank for no QR code.",
        kind="text",
        placeholder="e.g. https://github.com/you/project",
        default="",
        mid_quest_editable=True,
        tier=1,
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
    # web_research flips the web-search layer + the data/literature/
    # download. Re-fetch literature and re-write so the new corpus
    # reaches the paper (mirrors knowledge_top_k below).
    "web_research": ("literature", "write"),
    "supply_papers": ("literature",),
    # The plan is written once, before design; the pause only decides whether the quest waits for it.
    "pause_for_plan": (),
    # max_iterations is the design/review-loop hard cap. Lowering it
    # mid-quest just means the next review-revise iteration won't fire;
    # raising it gives the loop more attempts. Either way no node
    # needs to re-run — the iteration counter in QuestState decides.
    "max_iterations": (),
    # audience flips the writer's References handling (drops FI-internal
    # cross-quest entries on external; keeps everything on internal).
    # Only the write stage emits References — re-run write.
    "audience": ("write",),
    # ensemble_profile changes which nodes fan out across multiple
    # models. Conservatively re-run every node that any profile would
    # fan out across (ideate / analyze / cross_check) so a switch from
    # off → full re-runs all three with the new ensemble shape.
    "ensemble_profile": ("ideate", "analyze", "cross_check"),
    # Different models are a different ensemble: same nodes re-run.
    "ensemble_models": ("ideate", "analyze", "cross_check"),
    # knowledge.top_k / external_top_k change how many Axon + web
    # results the literature node fetches. The writer reads the
    # retrieved list, so a re-fetch should be followed by a re-write.
    "knowledge_top_k": ("literature", "write"),
    "knowledge_external_top_k": ("literature", "write"),
    # The author line and the poster size change only the rendered
    # outputs, which no LLM node produces. An output that already exists
    # is regenerated with `launch.py --resume <id> --emit <kind>`.
    "author": (),
    "affiliation": (),
    "contact_email": (),
    "url": (),
    "poster_size": (),
    # Reasoning effort applies to the calls made after the resume; nothing
    # already produced is re-run.
    "reasoning_effort": (),
    # The page limit sets the writer's word budget and the review's page
    # check: the paper is written and reviewed again.
    "page_limit": ("write", "review"),
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


_SURVEY_TOPIC_MARKERS = (
    "history of", "evolution of", "overview of", "development of",
    "retrospective", "the story of", "a survey of", "over time",
)


def smart_default_survey_mode(partial: dict[str, Any]) -> bool:
    """Auto-suggest survey mode (a descriptive history / overview synthesis
    with no experiment and no dataset) when the topic string reads as a
    history / overview / "evolution of X" question. The clarify agent makes
    the same call at runtime; this pre-selects it in the interview so the
    user only has to confirm."""
    topic = (partial.get("topic") or "").lower()
    return any(m in topic for m in _SURVEY_TOPIC_MARKERS)


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


def smart_default_review_panel(partial: dict[str, Any]) -> list[str]:
    """Three-persona heterogeneous panel by default. Three personas
    cost ~3× per review pass relative to single-reviewer, but each
    persona enforces a distinct rubric — methodologist owns the
    non-bypassable must-flag rules (circular evaluation, single-point
    eval, weak baseline, pseudo-units), statistician owns numbers
    and uncertainties, devil_advocate owns load-bearing assumptions.
    Without the methodologist firing, the must-flag mechanism added
    in the always-on-review-gate change has no enforcement surface.
    Cost-conscious users flip back to single-reviewer with an empty
    list (and lose the must-flag enforcement). Research and decision
    add the reproducibility reviewer the research profile requires."""
    return resolve_review_panel(
        ["methodologist", "statistician", "devil_advocate"], str(partial.get("result_use") or "research"),
    )


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
    "survey_mode": smart_default_survey_mode,
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
        # A later default reads an earlier one (study depth, no_simulation and the retrieval sizes follow the paper
        # format, now derived here rather than asked), unless the caller already holds a value for it.
        merged.setdefault(q.id, out[q.id])
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

# Anything outside Unicode-letter + digit + whitespace + dash gets
# stripped. Using ``[^\w\s-]`` with the (default) re.UNICODE flag
# preserves CJK / Cyrillic / Greek so a Traditional Chinese topic
# yields a real slug instead of an empty string.
_SLUG_STRIP = re.compile(r"[^\w\s-]", re.UNICODE)
_SLUG_SPACE = re.compile(r"\s+")
_SLUG_DUP_DASH = re.compile(r"-+")
_SLUG_UNDERSCORE = re.compile(r"_+")


def slugify(s: str) -> str:
    """Lowercase + remove non-alphanum + collapse spaces to dashes.
    Preserves Unicode letters (CJK / Cyrillic / Greek / etc.) so topics
    in non-Latin scripts produce a real slug, not an empty string.
    Mirrors ``vscode-frontier-insight/src/interview-core.ts:slugify``."""
    lowered = s.lower()
    stripped = _SLUG_STRIP.sub("", lowered)
    # ``\w`` accepts underscores; fold to dash so quest_ids visually
    # match the dash-separated style.
    no_underscores = _SLUG_UNDERSCORE.sub("-", stripped)
    dashed = _SLUG_SPACE.sub("-", no_underscores)
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
    pause_for_user_input: str = "never"
    # CLI / serve only — VSCode pins ``vscode_extension`` silently.
    provider: str = "vscode_extension"
    provider_model: str | None = None
    # HTTP-direct providers (openai/codex/gemini/ollama/vllm) only, and only when the endpoint or
    # key differ from that provider's own default: an OpenAI-compatible proxy or gateway (a
    # self-hosted vLLM server, Moonshot's Kimi, a corporate gateway), an env var other than the
    # provider's conventional one, or a model that rejects the per-node temperatures FI would
    # otherwise send. Blank (the default) leaves all three unset, matching today's behavior for
    # every quest that doesn't need them. Emitted into ``provider.base_url`` / ``api_key_env`` /
    # ``fixed_temperature`` exactly as ``docs/USAGE.md``'s schema documents them.
    provider_base_url: str = ""
    provider_api_key_env: str = ""
    provider_fixed_temperature: str = ""
    max_iterations: int = 2
    # Audience for the published paper. "external" drops FI-internal
    # cross-quest memory from the References section; "internal"
    # keeps everything. Default external for safe one-shot quests.
    audience: str = "external"
    # Visual style for paper.pdf: "latex" (Computer Modern article, default)
    # or "briefing" (the FI Research-Briefing look via the HTML/Chromium
    # backend). Emitted as ``output.paper_style``.
    paper_style: str = "latex"
    # How many Axon (RAG) prior-work entries the literature node
    # retrieves into the writer's context block. 8 is the default;
    # the smart-default helper bumps it to 12 for comprehensive reviews.
    knowledge_top_k: int = 8
    # How many EXTERNAL (web search — arXiv / OpenAlex / Crossref /
    # S2 / ...) hits to fetch when Axon misses. Bigger than the Axon
    # cap because web search is coarser; default 20 → 30 for surveys.
    knowledge_external_top_k: int = 20
    # Web research layer. When True (default), the literature node searches
    # the public web (Brave/DuckDuckGo) and downloads sources into
    # data/literature/ for every quest — sim and observational alike.
    # Maps to ``knowledge.web_search`` + ``web_fetch_pages``. Independent
    # of ``knowledge_enabled`` (the Axon corpus toggle).
    # Survey mode — a descriptive literature / history synthesis with NO
    # experiment AND NO dataset (the third "research approach"). Emitted as
    # ``engine.survey_mode``; implies ``no_simulation`` at runtime. Mirrors
    # ``interview-core.ts:InterviewAnswers.survey_mode``.
    survey_mode: bool = False
    web_research: bool = True
    # When True (the default), pause on a paywalled/abstract-only relevant
    # paper and write needs/WANTED_PAPERS.md so the user can drop the PDF
    # into inputs/papers/ and resume → ``pauses.papers``.
    supply_papers: bool = True
    # Stop once plan.md is written, to read and edit it → ``pauses.plan``.
    pause_for_plan: bool = False
    # ``rigor_profile`` at the top of the config: "default" (writes nothing) or "research". Must stay in sync with
    # vscode-frontier-insight/src/interview-core.ts. When ``result_use`` is set it decides this (see
    # ``rigor_profile_for``); left blank (a caller from before the question existed) this field is used as given.
    rigor_profile: str = "default"
    # The "What is the result for?" answer: "research" | "decision" | "explore", or "" when not asked.
    result_use: str = ""
    # Multi-model ensemble preset. Expanded by ``answers_to_yaml`` into
    # the ``provider.node_ensemble`` block when non-"off". See
    # ``ENSEMBLE_PROFILES`` for the four options and their cost
    # multipliers. FI does not choose the models: ``ensemble_models`` is what
    # the user named (comma separated, at least ``ENSEMBLE_MIN_MODELS``); a
    # profile without them configures nothing.
    ensemble_profile: str = "off"
    ensemble_models: str = ""
    # Comma-separated "node:model" pairs, parsed by ``answers_to_yaml``
    # into the ``provider.node_models`` block. Empty (default) emits
    # nothing — no behavior change until the user opts in. See
    # ``parse_node_models_answer``.
    node_models: str = ""
    # ``provider.reasoning_effort``: "default" (writes nothing) or one of
    # ``REASONING_EFFORT_LEVELS``. Must stay in sync with
    # vscode-frontier-insight/src/interview-core.ts.
    reasoning_effort: str = "default"
    # Author line printed on the paper, slides and poster. All optional;
    # ``answers_to_yaml`` writes each under ``output:`` only when set.
    author: str = ""
    affiliation: str = ""
    contact_email: str = ""
    url: str = ""
    # Poster sheet: "a1_portrait" (default) | "a0_portrait" | "landscape_48x36".
    poster_size: str = "a1_portrait"
    # ``output.page_limit``: the most pages paper.pdf may take, or ``None``
    # for no set limit (a limit the topic states still applies). Must stay in
    # sync with vscode-frontier-insight/src/interview-core.ts.
    page_limit: int | None = None


def parse_page_limit_answer(
    value: Any, *, on_error: Callable[[str], None] | None = None,
) -> int | None:
    """The ``page_limit`` answer as a page count, or ``None`` for no set limit.

    Blank, ``None`` and "none" are no limit. Otherwise a whole number of at
    least 1: ``4`` or ``"4"``, not ``4.5``, ``"four"``, ``0`` or ``True``.
    Anything else raises ``ValueError``; given ``on_error``, the message goes
    to it instead and the answer is ``None``."""
    def bad(message: str) -> None:
        if on_error is None:
            raise ValueError(message)
        on_error(message)

    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() == "none":
            return None
        digits = text.removeprefix("+")
        if not (digits.isascii() and digits.isdigit()):
            bad(f"page_limit must be a whole number of pages, or blank for no limit; got {value!r}")
            return None
        number = int(digits)
    elif isinstance(value, int) and not isinstance(value, bool):
        number = value
    else:
        bad(f"page_limit must be a whole number of pages, or blank for no limit; got {value!r}")
        return None
    if number < 1:
        bad(f"page_limit must be at least 1; got {number}")
        return None
    return number


def parse_fixed_temperature_answer(
    value: Any, *, on_error: Callable[[str], None] | None = None,
) -> float | None:
    """The ``provider_fixed_temperature`` answer as a float, or ``None`` when blank.

    Unlike ``parse_node_models_answer``, a bad value here is not silently dropped: the model this
    field exists for (Kimi K2.6 / K3) answers HTTP 400 to any temperature but the one or two it
    accepts, so a typo that silently emitted nothing would surface as a cryptic provider error deep
    into a quest instead of here, at config-generation time, where it is a one-line fix."""
    def bad(message: str) -> None:
        if on_error is None:
            raise ValueError(message)
        on_error(message)

    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        bad(f"provider_fixed_temperature must be a number, or blank to leave it unset; got {value!r}")
        return None


def parse_node_models_answer(raw: str) -> dict[str, str]:
    """Parse the ``node_models`` interview answer into a ``{node: model}``
    dict, ready to emit as ``provider.node_models``.

    Format: comma-separated ``node:model`` pairs (``poster:gpt-4o-mini,
    slides:gpt-4o-mini``). Split on the FIRST colon per pair, since a model
    name can itself contain a colon (``ollama:gemma3:4b``). Blank input,
    a pair with no colon, or a pair with an empty node or model half is
    silently skipped rather than raising — an interview answer is free
    text a person typed, and a typo here should degrade to "no override
    for that one node", not crash config generation.
    """
    out: dict[str, str] = {}
    for pair in (raw or "").split(","):
        pair = pair.strip()
        if not pair or ":" not in pair:
            continue
        node, _, model = pair.partition(":")
        node, model = node.strip(), model.strip()
        if node and model:
            out[node] = model
    return out


def expand_ensemble_profile(
    profile: str, *, models: Any,
) -> dict[str, dict[str, Any]]:
    """Expand a profile name and the models the USER named to a ``node_ensemble``
    dict shape.

    Returns a dict ready to plug into ``provider.node_ensemble`` in the
    YAML, mapping engine-node name → ``{models, merge[, moderator]}``.
    Returns an empty dict for ``off``, and for fewer than
    ``ENSEMBLE_MIN_MODELS`` models: FI does not pick models to make up the
    number, so a profile with nothing to fan out over configures nothing. The
    moderator is the first model the user listed.

    The shape mirrors ``core.config.NodeEnsembleConfig`` but stays a
    plain dict so this module doesn't import from ``core.config`` (that
    would create a circular import via ``core.config``'s use of pydantic
    + ``core.interview`` schema-export plumbing).
    """
    if profile == "off" or not profile:
        return {}
    trio = parse_ensemble_models(models)
    if len(trio) < ENSEMBLE_MIN_MODELS:
        return {}
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


# Interview answer vocabulary → canonical `pauses.*` vocabulary.
_CLARIFY_TO_PAUSE = {"off": "off", "auto": "auto", "interactive": "ask"}
_SUPPLY_TO_PAUSE = {
    "never": "never", "after_literature": "after_literature",
    "after_design": "before_build",
    "after_paper": "before_review", "both": "both",
}


def answers_to_yaml(answers: InterviewAnswers, *, frontend: str = "cli") -> str:
    """Render the interview answers as a complete config.yaml string.

    ``frontend`` ("cli" / "vscode" / "serve") controls only the leading
    comment header. What is checked, and how strictly, follows the
    answers alone: the same answers write the same settings from every
    interface. (The VS Code interview once wrote three cost-saving
    engine settings on its own; they now follow ``result_use:
    explore``, on every interface.)
    """
    indent = "  "
    lines: list[str] = []
    rigor_profile = rigor_profile_for(answers.result_use) if answers.result_use else answers.rigor_profile

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
    if rigor_profile == "research":
        lines.append('rigor_profile: "research"')
        lines.append("")

    lines.append("provider:")
    lines.append(f"{indent}name: {json.dumps(answers.provider)}")
    if answers.provider_model:
        lines.append(f"{indent}model: {json.dumps(answers.provider_model)}")
    # HTTP-direct transports only, and only when set: an endpoint or key env var other than the
    # provider's own default, or a fixed temperature a picky model demands. Blank (the default for
    # every quest that doesn't need them) emits nothing.
    if answers.provider_base_url.strip():
        lines.append(f"{indent}base_url: {json.dumps(answers.provider_base_url.strip())}")
    if answers.provider_api_key_env.strip():
        lines.append(f"{indent}api_key_env: {json.dumps(answers.provider_api_key_env.strip())}")
    fixed_temperature = parse_fixed_temperature_answer(answers.provider_fixed_temperature)
    if fixed_temperature is not None:
        lines.append(f"{indent}fixed_temperature: {fixed_temperature}")
    # Multi-model ensemble preset. Only emit when the user picked
    # something non-"off"; otherwise leave the engine on its
    # single-model path (no regression for default quests).
    node_ensemble = expand_ensemble_profile(
        answers.ensemble_profile, models=answers.ensemble_models,
    )
    if answers.ensemble_profile not in ("off", "") and not node_ensemble:
        # A profile was chosen but the user named fewer than two models, and FI
        # does not choose them. Say so where the person editing this file will see it.
        lines.append(
            f"{indent}# ensemble_profile {json.dumps(answers.ensemble_profile)} was chosen but "
            f"fewer than {ENSEMBLE_MIN_MODELS} models were named, so no ensemble is configured. "
            f"List the models under node_ensemble.<node>.models to fan out over."
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
    # Per-node model overrides. Only emit when the user typed something;
    # blank (default) leaves every node on the primary model, matching
    # today's behavior for a quest that doesn't set this.
    node_models = parse_node_models_answer(answers.node_models)
    if node_models:
        lines.append(f"{indent}node_models:")
        for node, model in node_models.items():
            lines.append(f"{indent}{indent}{node}: {json.dumps(model)}")
    # Reasoning effort: only a level is written; "default" leaves the key out
    # so every provider keeps its own default.
    if answers.reasoning_effort and answers.reasoning_effort != "default":
        lines.append(f"{indent}reasoning_effort: {json.dumps(answers.reasoning_effort)}")
    lines.append("")

    lines.append("engine:")
    lines.append(f"{indent}framework: \"langgraph\"")
    lines.append(f"{indent}max_iterations: {answers.max_iterations}")
    lines.append(f"{indent}review_loop: true")
    # Survey mode implies no-simulation (no experiment AND no dataset), so
    # the emitted no_simulation flag is the OR of the two — keeps the YAML
    # internally consistent even though the engine also forces this at
    # resolve time.
    survey_mode = bool(getattr(answers, "survey_mode", False))
    no_simulation = bool(answers.no_simulation) or survey_mode
    lines.append(f"{indent}no_simulation: {'true' if no_simulation else 'false'}")
    if survey_mode:
        lines.append(f"{indent}survey_mode: true")

    # A cheaper draft: exploring skips three model-call-heavy loops. Follows the answer, never the interface.
    if answers.result_use == "explore":
        for key, value in DRAFT_ENGINE_SETTINGS:
            lines.append(f"{indent}{key}: {value}")

    panel = list(answers.review_panel)
    if answers.result_use:
        # Every interface completes the panel before its confirm screen (``resolve_review_panel``), so this changes
        # nothing that was shown; it is here so no caller can write a panel other than the one that runs.
        panel = resolve_review_panel(panel, answers.result_use)
    if panel:
        if not answers.result_use and rigor_profile == "research" and panel == list(REVIEW_PANELS[0].value):
            # A caller that does not set result_use (from before the question existed): the old narrow completion
            # of the three-person smart default, which predates the research profile's requirement. A genuinely
            # custom panel is written as given, and Config refuses it with the missing role named.
            panel = [*panel, *[r for r in REQUIRED_REVIEW_ROLES if r not in panel]]
        lines.append(f"{indent}review_panel:")
        for persona in panel:
            lines.append(f"{indent}{indent}- {json.dumps(persona)}")
    elif rigor_profile != "research":
        # An empty panel written beside the research profile would contradict it (the profile brings its own panel).
        lines.append(f"{indent}review_panel: []")

    # Clarify overrides — pin the three topic-tuned slots + study_depth +
    # paper_venue + output_kinds + simulatability so the clarify node
    # short-circuits in auto mode. Keys mirror agents/clarify.md.
    overrides: dict[str, Any] = {
        "study_depth": answers.study_depth,
        "paper_venue": answers.paper_format,
        "output_kinds": answers.output_kinds,
        "simulatability": "no" if no_simulation else "yes",
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

    # Unified human-in-the-loop section: every place the quest stops for you.
    # Values use the coherent `pauses.*` vocabulary; legacy interview answers
    # are translated here (interactive→ask, after_design→before_build, …).
    lines.append("pauses:")
    lines.append(
        f"{indent}clarify: {json.dumps(_CLARIFY_TO_PAUSE.get(answers.clarify_mode, answers.clarify_mode))}"
    )
    _supply = _SUPPLY_TO_PAUSE.get(
        answers.pause_for_user_input, answers.pause_for_user_input
    )
    if _supply and _supply != "never":
        lines.append(f"{indent}supply: {json.dumps(_supply)}")
    # Written either way: the engine default is on, so "off" must be explicit.
    lines.append(f"{indent}papers: {'true' if answers.supply_papers else 'false'}")
    if answers.pause_for_plan:
        lines.append(f"{indent}plan: \"ask\"")
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
    # Web research layer — search the public web (Brave/DuckDuckGo) and
    # download the sources into data/literature/, on top of academic
    # retrieval. Emitted explicitly (on AND off) so the generated config
    # self-documents the interview choice. Independent of `enabled` above.
    lines.append(f"{indent}web_search: {'true' if answers.web_research else 'false'}")
    lines.append(f"{indent}web_fetch_pages: {'true' if answers.web_research else 'false'}")
    if not answers.knowledge_enabled:
        lines.append(f"{indent}external_fallback: [\"openalex\", \"arxiv\", \"crossref\"]")
        lines.append(f"{indent}source_routing: \"manual\"")
    lines.append("")

    lines.append("output:")
    quoted = ", ".join(json.dumps(k) for k in answers.output_kinds)
    lines.append(f"{indent}kinds: [{quoted}]")
    lines.append(f"{indent}paper_format: {json.dumps(answers.paper_format)}")
    # Only emit a non-default paper_style so existing YAML stays clean.
    if getattr(answers, "paper_style", "latex") != "latex":
        lines.append(f"{indent}paper_style: {json.dumps(answers.paper_style)}")
    if answers.audience != "external":
        lines.append(f"{indent}audience: {json.dumps(answers.audience)}")
    # Author line: only the fields the user filled in. ensure_ascii=False
    # keeps a CJK name readable in the YAML (still a valid quoted scalar).
    for key in ("author", "affiliation", "contact_email", "url"):
        value = " ".join(str(getattr(answers, key, "") or "").split())
        if value:
            lines.append(f"{indent}{key}: {json.dumps(value, ensure_ascii=False)}")
    if getattr(answers, "poster_size", "a1_portrait") != "a1_portrait":
        lines.append(f"{indent}poster_size: {json.dumps(answers.poster_size)}")
    # Page limit: only a whole number of pages is written; unset leaves the
    # key out, so a limit the topic states still applies.
    page_limit = getattr(answers, "page_limit", None)
    if isinstance(page_limit, int) and not isinstance(page_limit, bool) and page_limit >= 1:
        lines.append(f"{indent}page_limit: {page_limit}")
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
        "ensemble_min_models": ENSEMBLE_MIN_MODELS,
        "providers": [_choice(c) for c in PROVIDER_CHOICES],
        "provider_models": {
            name: [_choice(c) for c in opts]
            for name, opts in PROVIDER_MODEL_OPTIONS.items()
        },
        # What the research profile needs on a review panel and runs when none is given: the web review screen
        # completes the panel from these (``resolve_review_panel``) so it shows the panel that runs.
        "required_review_roles": list(REQUIRED_REVIEW_ROLES),
        "research_review_panel": list(RESEARCH_REVIEW_PANEL),
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
