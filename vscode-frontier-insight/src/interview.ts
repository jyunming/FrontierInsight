/**
 * Interview — VSCode-UI half. The pure YAML-generation logic lives in
 * `interview-core.ts` so it can be unit-tested from plain Node without
 * a VSCode runtime present.
 *
 * When a user types `@fi` (no command) or `@fi /new`, the extension
 * runs this sequence of `showInputBox` / `showQuickPick` modals to
 * collect the essentials, generates a config.yaml via
 * `writeInterviewYaml`, and returns the path so the existing
 * quest-launch pipeline can pick it up.
 */
import * as vscode from "vscode";
import { execSync } from "child_process";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import {
    ENSEMBLE_MIN_MODELS,
    InterviewAnswers,
    LIGHT_NODES,
    OTHER_NODES,
    PaperFormat,
    answersToYaml,
    parseNodeModelsAnswer,
    resolveReviewPanel,
    rigorProfileFor,
    serializeNodeModels,
    withModelFor,
    writeInterviewYaml,
    truncate,
    slugify,
} from "./interview-core";
import { discoverAxon } from "./axon-endpoint";

// Re-export so callers don't have to know about the split.
export { InterviewAnswers, answersToYaml, writeInterviewYaml };

/**
 * Let the user tick the models an ensemble fans out over. The list is what THIS
 * VSCode offers through `vscode.lm` — Copilot's, an Ollama server's, a BYOK
 * endpoint's — and nothing is pre-ticked: FI does not choose models for you.
 * Returns the picked model ids (fewer than two is allowed: no ensemble then),
 * or undefined when the user pressed Esc.
 */
async function pickEnsembleModels(): Promise<string[] | undefined> {
    const models = await vscode.lm.selectChatModels();
    if (models.length === 0) {
        vscode.window.showWarningMessage(
            "Frontier Insight: this VSCode lists no language models, so no ensemble can be configured. " +
            "Sign in to Copilot Chat or add a model provider, then edit provider.node_ensemble in the YAML.",
        );
        return [];
    }
    const items = models
        .map((m) => ({ label: m.name || m.id, description: `${m.vendor} · ${m.family}`, detail: m.id, id: m.id }))
        .sort((a, b) => a.label.localeCompare(b.label));
    const picked = await vscode.window.showQuickPick(items, {
        canPickMany: true,
        title: "Frontier Insight — which models should the ensemble fan out over?",
        placeHolder: `Tick at least ${ENSEMBLE_MIN_MODELS}. Nothing is chosen for you; each extra model multiplies the cost.`,
        ignoreFocusOut: true,
    });
    return picked ? picked.map((p) => p.id) : undefined;
}

/**
 * Let the user pick ONE of the models this VSCode offers through `vscode.lm`. Nothing
 * is chosen for them: FI has no way to know which of these is cheap, so the list is
 * just what is there. `removeLabel` adds a first entry that clears the choice.
 * Returns the model id, "" for the remove entry, or undefined when the user pressed Esc.
 */
async function pickOneModel(title: string, removeLabel?: string): Promise<string | undefined> {
    const models = await vscode.lm.selectChatModels();
    if (models.length === 0) {
        vscode.window.showWarningMessage(
            "Frontier Insight: this VSCode lists no language models. Sign in to Copilot Chat or add a " +
            "model provider, or type node:model pairs instead.",
        );
        return undefined;
    }
    const items: { label: string; description?: string; detail?: string; id: string }[] = models
        .map((m) => ({ label: m.name || m.id, description: `${m.vendor} · ${m.family}`, detail: m.id, id: m.id }))
        .sort((a, b) => a.label.localeCompare(b.label));
    if (removeLabel) {
        items.unshift({ label: removeLabel, id: "" });
    }
    const picked = await vscode.window.showQuickPick(items, {
        title,
        placeHolder: "Nothing is chosen for you: the names show the tier (mini, haiku, flash, ...), FI does not know the price.",
        matchOnDescription: true,
        matchOnDetail: true,
        ignoreFocusOut: true,
    });
    return picked ? picked.id : undefined;
}

/**
 * The editor behind "Per-node model overrides": pick models from the list VSCode
 * offers instead of typing node:model pairs. The five nodes measured as safe to
 * move to a cheaper model are one entry ("Light nodes"); any single node can also be
 * set, with the untested ones marked. The result is the same "node:model, ..." answer
 * the typed field produced, so the YAML and every other surface are unchanged.
 */
export async function editNodeModels(a: InterviewAnswers): Promise<void> {
    for (;;) {
        const current = parseNodeModelsAnswer(a.node_models);
        const summary = Object.keys(current).length
            ? Object.entries(current).map(([n, m]) => `${n} → ${m}`).join(", ")
            : "every node uses your Chat-picker model";
        const action = await vscode.window.showQuickPick(
            [
                {
                    label: "Light nodes → one cheaper model",
                    description: `${LIGHT_NODES.join(", ")} (measured: ~15% fewer tokens, no score drop seen in 3 runs)`,
                    value: "light",
                },
                { label: "One node…", description: "pick a node, then a model", value: "one" },
                { label: "Clear all overrides", description: "every node uses your Chat-picker model", value: "clear" },
                { label: "Type node:model pairs", description: "advanced: edit the text directly", value: "type" },
                { label: "Done", value: "done" },
            ],
            { title: `Per-node models — ${summary}`, ignoreFocusOut: true },
        );
        if (!action || action.value === "done") {
            return;
        }
        try {
            if (action.value === "clear") {
                a.node_models = "";
            } else if (action.value === "light") {
                const id = await pickOneModel("Frontier Insight — which model should the light nodes use?", "Use my Chat-picker model (remove)");
                if (id !== undefined) {
                    a.node_models = serializeNodeModels(withModelFor(current, LIGHT_NODES, id));
                }
            } else if (action.value === "one") {
                const node = await vscode.window.showQuickPick(
                    [
                        ...LIGHT_NODES.map((n) => ({ label: n, description: "measured: safe on a cheaper model" })),
                        ...OTHER_NODES.map((n) => ({ label: n, description: "not tested on a cheaper model" })),
                    ],
                    { title: "Which node?", ignoreFocusOut: true },
                );
                if (node) {
                    const id = await pickOneModel(`Frontier Insight — which model for ${node.label}?`, "Use my Chat-picker model (remove)");
                    if (id !== undefined) {
                        a.node_models = serializeNodeModels(withModelFor(current, [node.label], id));
                    }
                }
            } else {
                const v = await vscode.window.showInputBox({
                    title: "Per-node model overrides",
                    value: a.node_models || "",
                    placeHolder: "cross_check:MODEL, poster:MODEL",
                    ignoreFocusOut: true,
                });
                if (v !== undefined) {
                    a.node_models = v.trim();
                }
            }
        } catch (e) {
            vscode.window.showWarningMessage(`Frontier Insight: ${(e as Error).message}`);
        }
    }
}

/** Return true iff `where`/`which` finds the given binary on PATH. */
function isOnPath(binary: string): boolean {
    const cmd = process.platform === "win32" ? `where ${binary}` : `which ${binary}`;
    try {
        execSync(cmd, { stdio: "ignore" });
        return true;
    } catch {
        return false;
    }
}

// Prose paper formats — their tier-2 no_simulation default is `true`.
const PROSE_FORMATS = new Set(["essay", "report", "policy_brief", "whitepaper"]);

// Topic markers that auto-suggest survey mode (a descriptive history /
// overview synthesis, no experiment, no dataset). Mirrors
// core/interview.py:_SURVEY_TOPIC_MARKERS.
const SURVEY_TOPIC_MARKERS = [
    "history of", "evolution of", "overview of", "development of",
    "retrospective", "the story of", "a survey of", "over time",
];
function looksLikeSurvey(topic: string): boolean {
    const t = (topic || "").toLowerCase();
    return SURVEY_TOPIC_MARKERS.some((m) => t.includes(m));
}

/**
 * Every interview question this frontend actually asks, by its id in
 * ``core/interview_schema.json``.
 *
 * This is the VS Code half of the frontend contract. Each Python
 * ``Question`` carries a ``frontends`` tuple saying which surfaces ask
 * it; ``tests/test_interview_schema_parity.py`` derives the expected set
 * from that tuple and requires it to equal this list exactly. So a
 * question that declares ``vscode`` but is never asked fails CI, and so
 * does an id listed here that the schema does not mark for VS Code.
 *
 * Before this existed the parity test hardcoded four prompt names, which
 * is why ``ensemble_profile``, ``paper_style`` and ``max_iterations``
 * could ship unreachable for months while CI stayed green.
 *
 * Asked inline as modals: topic, output_kinds, paper_format, study_depth,
 * ensemble_profile, and the four author-line fields. Reachable from the
 * review screen: everything else ("Edit a default" / "Edit an advanced
 * field"). `title`, `no_simulation` and `survey_mode` are derived first
 * and then editable there; `survey_mode` rides the research-approach
 * picker.
 */
export const VSCODE_ASKED_QUESTIONS: readonly string[] = [
    "topic",
    "title",
    "paper_format",
    "paper_style",
    "output_kinds",
    "study_depth",
    "no_simulation",
    "survey_mode",
    "clarify_mode",
    "review_panel",
    "pause_for_user_input",
    "knowledge_enabled",
    "web_research",
    "supply_papers",
    "pause_for_plan",
    "result_use",
    "audience",
    "knowledge_top_k",
    "knowledge_external_top_k",
    "comparative_baseline",
    "success_metric",
    "budget",
    "ensemble_profile",
    "ensemble_models",
    "max_iterations",
    "node_models",
    "reasoning_effort",
    "poster_size",
    "page_limit",
    "author",
    "affiliation",
    "contact_email",
    "url",
];

/**
 * Run the interview. Returns the answers, or `undefined` if the user
 * cancelled at any step (Esc on a modal, or empty topic).
 *
 * Flow:
 *   1. Tier-1 — five modals (topic, paper_format, output_kinds, study_depth,
 *      ensemble_profile), then the optional author line (author, affiliation,
 *      contact email, project link), where an empty box skips the field.
 *      Provider + model are pinned silently by the extension (provider =
 *      vscode_extension, model = whatever the chat picker showed).
 *   2. Tier-2 — derive title / no_simulation / clarify_mode / review_panel /
 *      knowledge_enabled / audience from the tier-1 answers + smart defaults.
 *      Show the derived values as markdown in the chat.
 *   3. Action picker — Launch / Edit defaults / Show advanced / Cancel.
 *      "Edit defaults" loops back to a per-field picker; "Show advanced"
 *      surfaces the tier-3 slots (baseline / metric / budget / top_k).
 */
export async function runInterview(
    stream: vscode.ChatResponseStream,
): Promise<InterviewAnswers | undefined> {
    stream.markdown(
        "🧪 **Let's set up a new research quest.** Two questions (and your author line the first time), then I'll show you the defaults worked out from your topic — edit anything before launch.\n\n",
    );

    // 1. Topic — the only mandatory input.
    const topic = await vscode.window.showInputBox({
        title: "Frontier Insight — quest topic",
        prompt: "What do you want to study? Be specific about the question, what's known, and what success looks like.",
        placeHolder:
            "e.g. Compare three numerical integrators on a damped harmonic oscillator and report energy drift.",
        ignoreFocusOut: true,
    });
    if (!topic || !topic.trim()) {
        stream.markdown(
            "\n— interview cancelled (no topic provided). Try `@fi /new` again, or `@fi /start <path-to-config.yaml>` if you have one ready.\n",
        );
        return undefined;
    }
    stream.markdown(`  **Topic:** ${truncate(topic, 200)}\n\n`);

    // 2. Title — auto-slugged from topic (no modal). The user can
    // override it from the review screen below if they want a
    // different folder name.
    const suggestedTitle = slugify(topic).slice(0, 40) || "quest";

    // The deliverables, the paper format and the study depth are worked out (tier 2 in core/interview.py) and shown on
    // the review screen to change; the multi-model ensemble is an advanced field (tier 3). Only the topic, what the
    // result is for and (on the first interview) the author line are asked.
    // 3b. What is the result for? Sets how strictly the quest is checked (core/interview.py:RESULT_USE_CHOICES, same
    // wording): research or a decision is the research profile; exploring is a cheaper draft. The same answer writes
    // the same settings on every interface.
    const resultUseChoice = await vscode.window.showQuickPick(
        [
            {
                label: "$(verified) Research (default)",
                description: "Every check stops the quest, the plan waits for you, a clean environment, four reviewers. Slower.",
                value: "research" as const,
            },
            {
                label: "$(law) A decision",
                description: "The same checks as Research: a result someone will act on needs every one of them.",
                value: "decision" as const,
            },
            {
                label: "$(telescope) Exploring (cheaper draft)",
                description: "Fewer model calls; the result is a preliminary draft, never ready to publish as it stands.",
                value: "explore" as const,
            },
        ],
        {
            title: "Frontier Insight — what is the result for?",
            placeHolder: "Research or a decision gets every check; exploring is a cheaper, preliminary draft.",
            ignoreFocusOut: true,
        },
    );
    if (!resultUseChoice) return undefined;
    const resultUse = resultUseChoice.value;
    stream.markdown(`  **Result for:** \`${resultUse}\`

`);

    const paperFormat: PaperFormat = paperFormatFor(topic);
    const outputKinds = outputKindsFor(topic);
    const studyDepth = studyDepthFor(paperFormat, topic);
    const missingNote = missingToolsNote(outputKinds);
    if (missingNote) stream.markdown(missingNote);

    // Author line — asked on the first interview only, then kept in the profile every interface reads
    // (core/profile.py); a later interview fills it from there and shows it below to change.
    const savedProfile = loadProfile();
    let authorLine: AuthorLine | undefined = savedProfile ?? undefined;
    if (!authorLine) {
        authorLine = await askAuthorLine();
        if (!authorLine) return undefined;
    }
    const byline = formatAuthorLine(authorLine);
    if (byline) {
        stream.markdown(`  **Author line:** ${truncate(byline, 200)}${savedProfile ? " _(from your saved details)_" : ""}\n\n`);
    }

    // ─── Tier-2 derivation (mirrors core/interview.py SMART_DEFAULTS) ───
    // title — slug from topic; no_simulation — prose vs scientific;
    // others are static or auto-probed.
    // Mirror core/interview.py's smart-default helpers: a survey-shaped
    // quest legitimately wants more prior-work entries (Axon RAG cap 12,
    // external web cap 30) than the cost-conscious defaults (8 / 20).
    const compReview = studyDepth === "comprehensive review";
    const answers: InterviewAnswers = {
        topic: topic.trim(),
        title: suggestedTitle,
        output_kinds: outputKinds,
        paper_format: paperFormat,
        clarify_mode: "auto",
        // Match the Python smart default: a 3-persona panel. An empty list
        // is single-reviewer and silently disables must-flag enforcement
        // (the methodologist owns the non-bypassable rules) — VSCode users
        // must not quietly get weaker rigor than CLI users.
        // Research and decision add the reviewers the research profile requires, before the review block below,
        // so it shows the panel that runs (core/interview.py:resolve_review_panel).
        review_panel: resolveReviewPanel(["methodologist", "statistician", "devil_advocate"], resultUse),
        knowledge_enabled: await probeAxonReachable(),
        // survey_mode implies no_simulation, so a history topic forces
        // observational even under a scientific paper_format (invariant parity
        // with the runtime + the other frontends).
        no_simulation: PROSE_FORMATS.has(paperFormat) || looksLikeSurvey(topic),
        // Auto-suggest survey mode (a descriptive history / overview synthesis
        // with no experiment and no dataset) for history/overview topics —
        // mirrors core/interview.py:smart_default_survey_mode. Editable below.
        survey_mode: looksLikeSurvey(topic),
        study_depth: studyDepth,
        comparative_baseline: "",
        success_metric: "",
        budget: "",
        node_models: "",
        reasoning_effort: "default",
        provider_model: "",
        // Advanced (tier 3): off unless changed there; anything else expands into provider.node_ensemble.
        ensemble_profile: "off",
        ensemble_models: "",
        // Schema defaults, editable from the review screen below.
        paper_style: "latex",
        pause_for_user_input: "never",
        max_iterations: 2,
        audience: "external",
        knowledge_top_k: compReview ? 12 : 8,
        knowledge_external_top_k: compReview ? 30 : 20,
        // Web research on by default — searches the public web and
        // downloads sources into data/literature/ for every quest.
        web_research: true,
        // Pause for paywalled papers on by default (the engine default).
        supply_papers: true,
        // The plan is always written; stopping for it is opt-in (unattended runs).
        pause_for_plan: false,
        result_use: resultUse,
        rigor_profile: rigorProfileFor(resultUse),
        ...authorLine,
        poster_size: "a1_portrait",
        // Blank: no set page limit (a limit the topic states still applies).
        page_limit: null,
    };

    // ─── Review block + action picker loop ──────────────────────────
    // Fields changed by hand here: a changed paper format works the others out again, never these.
    const edited = new Set<string>();
    while (true) {
        stream.markdown(reviewBlockMarkdown(answers));
        const action = await vscode.window.showQuickPick(
            [
                { label: "$(rocket) Launch quest", value: "launch" },
                { label: "$(edit) Edit a default", value: "edit_default" },
                { label: "$(settings-gear) Edit an advanced field", value: "edit_advanced" },
                { label: "$(x) Cancel", value: "cancel" },
            ],
            {
                title: "Frontier Insight — ready to launch?",
                placeHolder:
                    "Launch with the defaults above, edit one, expose the advanced fields, or cancel.",
                ignoreFocusOut: true,
            },
        );
        if (!action || action.value === "cancel") {
            stream.markdown("\n— interview cancelled.\n");
            return undefined;
        }
        if (action.value === "launch") {
            return answers;  // the caller keeps the author line once the config is written (keepAuthorLine)
        }
        if (action.value === "edit_default") {
            const before = JSON.stringify(answers);
            await editTier2Field(answers, edited);
            markChanged(before, answers, edited);
            continue;
        }
        if (action.value === "edit_advanced") {
            const before3 = JSON.stringify(answers);
            await editTier3Field(answers, edited);
            markChanged(before3, answers, edited);
            continue;
        }
    }
}

/**
 * Find the Axon API sidecar — if one is reachable, the knowledge layer
 * defaults to enabled. Otherwise off. Matches
 * core/interview.py:smart_default_knowledge_enabled.
 *
 * Goes through the shared discovery sweep rather than a fixed port:
 * this default used to land on "disabled" for anyone whose Axon wasn't
 * on the old 8000, which is now everyone running a current Axon.
 */
async function probeAxonReachable(): Promise<boolean> {
    const overrideUrl = vscode.workspace
        .getConfiguration("frontierInsight")
        .get<string>("axonUrl");
    // Shorter than the default per-probe budget: this runs inline in the
    // interview, and several dead candidates at 5 s each would stall the
    // question the user is waiting on.
    const found = await discoverAxon({
        overrideUrl: overrideUrl?.trim() || undefined,
        timeoutMs: 1500,
    });
    if (!found.live) {
        // Say WHY this fell back to disabled. A silent catch made
        // "axon is up but the interview defaulted to disabled" reports
        // impossible to debug.
        console.warn(`[fi] axon not found. Tried: ${found.attempts.join("; ")}`);
        return false;
    }
    return true;
}

function reviewBlockMarkdown(a: InterviewAnswers): string {
    const lines: string[] = [];
    lines.push("\n### Review before launch\n");
    lines.push("| Field | Value |");
    lines.push("|---|---|");
    lines.push(`| Paper format | \`${a.paper_format}\` |`);
    lines.push(`| Deliverables | ${a.output_kinds.join(", ")} |`);
    lines.push(`| Study depth | \`${a.study_depth}\` |`);
    lines.push(`| Title (auto-slug) | \`${a.title}\` |`);
    lines.push(`| Research approach | ${a.survey_mode === true ? "literature synthesis (survey — no experiment/data)" : a.no_simulation ? "observational" : "computational"} |`);
    lines.push(`| Clarify mode | \`${a.clarify_mode}\` |`);
    lines.push(`| Reviewer panel | ${a.review_panel.length === 0 ? "single reviewer" : a.review_panel.join(", ")} |`);
    lines.push(`| Knowledge layer (Axon) | ${a.knowledge_enabled ? "enabled (sidecar detected)" : "disabled"} |`);
    lines.push(`| Web research (download sources) | ${a.web_research === false ? "off" : "on"} |`);
    lines.push(`| Supply paywalled papers | ${a.supply_papers === false ? "off" : "pause for my PDFs"} |`);
    lines.push(`| Stop to read and edit the plan | ${a.pause_for_plan === true ? "yes (plan.md)" : "no"} |`);
    lines.push(`| Result for | ${a.result_use === "explore" ? "exploring: a cheaper preliminary draft (no idea self-critique, no per-finding cross-check, no redesign after the analysis)" : `${a.result_use ?? "research"}: every check stops the quest, the plan waits for you, a clean environment per quest`} |`);
    lines.push(`| Pause for my papers / datasets | \`${a.pause_for_user_input ?? "never"}\` |`);
    lines.push(`| Multi-model ensemble | \`${a.ensemble_profile ?? "off"}\` |`);
    if ((a.ensemble_profile ?? "off") !== "off") {
        lines.push(`| Ensemble models (your choice) | ${a.ensemble_models ? a.ensemble_models : "none named: no ensemble"} |`);
    }
    lines.push(`| Paper audience | \`${a.audience}\` |`);
    const authorCell = formatAuthorLine(a).replace(/\|/g, "\\|");
    lines.push(`| Author line | ${authorCell || "Frontier Insight (no author set)"} |`);
    // ``knowledge_top_k`` is Tier-2 in core/interview.py — show it in
    // the always-visible review block alongside the other defaults so
    // the VSCode flow matches the schema-backed web/CLI tiering.
    lines.push(`| Axon hits per quest (top_k) | \`${a.knowledge_top_k}\` |`);
    // Tier-3 fields only render when the user set them (anything other
    // than the static defaults). External cap default = 20.
    const ext = a.knowledge_external_top_k;
    const hasOverride = (
        a.comparative_baseline || a.success_metric || a.budget || a.node_models
        || (a.reasoning_effort !== undefined && a.reasoning_effort !== "default")
        || (ext !== undefined && ext !== 20)
        || (a.poster_size !== undefined && a.poster_size !== "a1_portrait")
        || (a.paper_style !== undefined && a.paper_style !== "latex")
        || a.max_iterations !== 2
        || typeof a.page_limit === "number"
    );
    if (hasOverride) {
        lines.push("\n_Advanced overrides:_");
        if (a.comparative_baseline) lines.push(`  • baseline: ${a.comparative_baseline}`);
        if (a.success_metric) lines.push(`  • metric: ${a.success_metric}`);
        if (a.budget) lines.push(`  • budget: ${a.budget}`);
        if (a.node_models) lines.push(`  • per-node models: ${a.node_models}`);
        if (a.reasoning_effort !== undefined && a.reasoning_effort !== "default") {
            lines.push(`  • reasoning effort: ${a.reasoning_effort}`);
        }
        if (ext !== undefined && ext !== 20) lines.push(`  • external_top_k (web): ${ext}`);
        if (a.poster_size !== undefined && a.poster_size !== "a1_portrait") {
            lines.push(`  • poster size: ${a.poster_size}`);
        }
        if (a.paper_style !== undefined && a.paper_style !== "latex") {
            lines.push(`  • paper style: ${a.paper_style}`);
        }
        if (a.max_iterations !== 2) lines.push(`  • iteration budget: ${a.max_iterations}`);
        if (typeof a.page_limit === "number") lines.push(`  • page limit: ${a.page_limit} pages`);
    }
    lines.push("");
    return lines.join("\n");
}


// Strict decimal-integer parse. The Python schema uses ``int(s)`` which
// rejects ``2.0``, ``1e2``, ``0x10``; ``Number(s)`` accepts all three.
// We match a trimmed ``+? digits`` regex and parse with base 10 so the
// VSCode validator behaves the same as Python's int().
const POSITIVE_INT_RE = /^\+?\d+$/;

function parsePositiveInt(s: string): number | null {
    const trimmed = s.trim();
    if (!POSITIVE_INT_RE.test(trimmed)) return null;
    const n = parseInt(trimmed, 10);
    if (!Number.isSafeInteger(n) || n < 1) return null;
    return n;
}

function validatePositiveInt(s: string): string | null {
    return parsePositiveInt(s) === null
        ? "must be a positive decimal integer (>= 1)"
        : null;
}


interface AuthorLine {
    author: string;
    affiliation: string;
    contact_email: string;
    url: string;
}

// One input box per field; the prompts mirror the tier-1 author questions
// in core/interview.py.
const AUTHOR_LINE_FIELDS: { key: keyof AuthorLine; title: string; prompt: string; placeHolder: string }[] = [
    {
        key: "author",
        title: "Frontier Insight — author (optional)",
        prompt: "Your name as it should appear on the paper, slides and poster. Leave blank to keep the 'Frontier Insight' byline.",
        placeHolder: "e.g. Jane Chen",
    },
    {
        key: "affiliation",
        title: "Frontier Insight — affiliation (optional)",
        prompt: "Lab, company or school to print under the author. Leave blank to omit.",
        placeHolder: "e.g. Materials Lab, Example University",
    },
    {
        key: "contact_email",
        title: "Frontier Insight — contact email (optional)",
        prompt: "Printed on the poster and under the paper title so readers can reach you. It goes only into your own output files. Leave blank to omit.",
        placeHolder: "e.g. jane@example.org",
    },
    {
        key: "url",
        title: "Frontier Insight — project link (optional)",
        prompt: "A web page for this work, such as a repository or lab page. The poster prints it as a QR code. Leave blank for no QR code.",
        placeHolder: "e.g. https://github.com/you/project",
    },
];

/**
 * Ask the four optional author-line fields. An empty box skips that field;
 * Esc returns undefined. ``current`` pre-fills the boxes when the user edits
 * the line from the review screen.
 */
async function askAuthorLine(current?: Partial<AuthorLine>): Promise<AuthorLine | undefined> {
    const out: AuthorLine = { author: "", affiliation: "", contact_email: "", url: "" };
    for (const field of AUTHOR_LINE_FIELDS) {
        const v = await vscode.window.showInputBox({
            title: field.title,
            prompt: field.prompt,
            placeHolder: field.placeHolder,
            value: current?.[field.key] ?? "",
            ignoreFocusOut: true,
        });
        if (v === undefined) return undefined;
        out[field.key] = v.trim().replace(/\s+/g, " ");
    }
    return out;
}

function formatAuthorLine(a: Partial<AuthorLine>): string {
    return [a.author, a.affiliation, a.contact_email, a.url]
        .map((v) => (v ?? "").trim())
        .filter(Boolean)
        .join(" · ");
}


/** The deliverables picker (review screen). */
async function pickOutputKinds(): Promise<string[] | undefined> {
    const outputChoice = await vscode.window.showQuickPick(
        [
            {
                label: "$(book) paper + PDF (recommended)",
                description: "MD + PDF; needs pandoc + LaTeX on PATH",
                value: ["paper_md", "paper_pdf"],
            },
            {
                label: "$(symbol-class) everything",
                description: "paper, PDF, slides, poster, talk script",
                value: ["paper_md", "paper_pdf", "slides", "poster", "speech"],
            },
            {
                label: "$(layout) paper + PDF + slides",
                description: "PDF via pandoc+LaTeX; slides via Marp + .pptx via pandoc",
                value: ["paper_md", "paper_pdf", "slides"],
            },
            {
                label: "$(file-text) paper only (Markdown)",
                description: "fastest; no extra system tools needed",
                value: ["paper_md"],
            },
        ],
        {
            title: "Frontier Insight — which deliverables?",
            placeHolder:
                "Default is paper + PDF — most users want the rendered file. PDF gracefully degrades to MD if pandoc isn't installed.",
            ignoreFocusOut: true,
        },
    );
    return outputChoice ? (outputChoice.value as string[]) : undefined;
}

/** What is missing on PATH for the chosen deliverables, as markdown ("" when nothing is). */
function missingToolsNote(kinds: string[]): string {
    const missing: string[] = [];
    if (kinds.includes("paper_pdf") && !(isOnPath("pandoc") && isOnPath("pdflatex"))) {
        missing.push(
            "`paper_pdf` requires **pandoc** + a LaTeX engine (`pdflatex`) on PATH. " +
            "Install: pandoc.org/installing.html + miktex.org (Windows) / tinytex.org (mac/Linux). " +
            "Without these, the quest produces `paper.md` only.",
        );
    }
    if (kinds.includes("slides") && !isOnPath("marp")) {
        missing.push(
            "`slides` requires **marp** CLI. Install: `npm i -g @marp-team/marp-cli`. " +
            "Without it, the quest produces `slides.md` only (no `.html`/`.pdf`).",
        );
    }
    if (kinds.includes("poster") && !isOnPath("pdflatex")) {
        missing.push(
            "`poster` requires `pdflatex`. See the PDF prerequisites above.",
        );
    }
    if (missing.length === 0) return "";
    return "⚠️ **Missing tools** (selected outputs will be partial):\n\n" +
        missing.map((m) => `  - ${m}`).join("\n") + "\n\n";

}

/** The paper format picker (review screen). */
async function pickPaperFormat(): Promise<PaperFormat | undefined> {
    const paperFormatChoice = await vscode.window.showQuickPick(
        [
            // Scientific venues — pick when the topic is computational
            // / experimental. The clarify agent maps these to
            // simulatability == "yes". Order: most generic first.
            {
                label: "$(file) generic — scientific paper, IMRAD",
                description: "Default. Surveys, comparative reviews, theoretical derivations, brief preprints.",
                value: "generic" as const,
            },
            {
                label: "$(beaker) NeurIPS — ML benchmark / algorithm",
                description: "Empirical ML, neural networks, learning algorithms, journal-length.",
                value: "neurips" as const,
            },
            {
                label: "$(beaker) ICLR — representation learning",
                description: "Representations, generative models, ML theory.",
                value: "iclr" as const,
            },
            {
                label: "$(circuit-board) IEEE Access — engineering / systems",
                description: "Hardware/software architectures, measurement studies, engineering experiments.",
                value: "ieee_access" as const,
            },
            {
                label: "$(symbol-namespace) Nature MI — physical sciences",
                description: "Physics / chemistry / materials simulation, scientific-method experiments.",
                value: "nature_mi" as const,
            },
            // Non-scientific prose — pick when the topic is
            // qualitative / historical / cultural / business / policy.
            // The clarify agent maps these to simulatability == "no".
            {
                label: "$(book) essay — long-form argumentative prose",
                description: "Cultural / historical / intellectual / qualitative cross-case analysis. Argue a thesis.",
                value: "essay" as const,
            },
            {
                label: "$(briefcase) report — consulting-style exec report",
                description: "Business / operational / market analysis with cover + TOC. Decision-maker audience.",
                value: "report" as const,
            },
            {
                label: "$(law) policy brief — 2-4 page recommendation",
                description: "Single decision for policymakers. Issue + context + recommendation.",
                value: "policy_brief" as const,
            },
            {
                label: "$(file-pdf) whitepaper — 8-20 page industry analysis",
                description: "Vendor-neutral tech trends / standards / architecture comparisons. Practitioner audience.",
                value: "whitepaper" as const,
            },
        ],
        {
            title: "Frontier Insight — paper format / venue?",
            placeHolder:
                "Picks the LaTeX template + writing persona. 'generic' is the safe default for scientific topics; 'essay' for non-computational humanities/social-science.",
            ignoreFocusOut: true,
        },
    );
    return paperFormatChoice ? paperFormatChoice.value : undefined;
}

/** Mirrors core/interview.py:smart_default_study_depth. */
function studyDepthFor(paperFormat: string, topic: string): "brief preprint" | "journal-length" | "comprehensive review" {
    if (paperFormat === "policy_brief") return "brief preprint";
    const t = topic.toLowerCase();
    if (["survey", "review of", "compar"].some((tok) => t.includes(tok))) return "comprehensive review";
    return "journal-length";
}

/** Mirrors core/interview.py:FORMAT_TOPIC_MARKERS / smart_default_paper_format. */
const FORMAT_TOPIC_MARKERS: [string, PaperFormat][] = [
    ["policy brief", "policy_brief"], ["white paper", "whitepaper"], ["whitepaper", "whitepaper"],
    ["an essay", "essay"], ["essay on", "essay"], ["business report", "report"], ["market report", "report"],
    ["history of", "essay"], ["evolution of", "essay"], ["the story of", "essay"], ["development of", "essay"],
    ["retrospective", "essay"],
];

function paperFormatFor(topic: string): PaperFormat {
    const t = topic.toLowerCase();
    const hit = FORMAT_TOPIC_MARKERS.find(([m]) => t.includes(m));
    return hit ? hit[1] : "generic";
}

/** The fields the person really changed in one edit (a picker opened and cancelled changes nothing). */
function markChanged(before: string, after: InterviewAnswers, edited: Set<string>): void {
    const old = JSON.parse(before) as Record<string, unknown>;
    const now = after as unknown as Record<string, unknown>;
    // What a changed paper format or depth re-derived is not the person's own change.
    const derived = new Set(["study_depth", "no_simulation", "knowledge_top_k", "knowledge_external_top_k"]);
    const changedFormat = old.paper_format !== now.paper_format || old.study_depth !== now.study_depth;
    for (const k of Object.keys(now)) {
        if (JSON.stringify(old[k]) === JSON.stringify(now[k])) continue;
        if (changedFormat && derived.has(k) && !(k === "study_depth" && old.paper_format === now.paper_format)) continue;
        edited.add(k);
    }
}

/** Mirrors core/interview.py:smart_default_output_kinds. */
function outputKindsFor(topic: string): string[] {
    const t = topic.toLowerCase();
    const kinds = ["paper_md", "paper_pdf"];
    if (/\bslides?\b|slide deck|presentation/.test(t)) kinds.push("slides");
    if (/\bposter\b/.test(t)) kinds.push("poster");
    if (/\btalk\b|\bspeech\b|talk script/.test(t)) kinds.push("speech");
    return kinds;
}

/** Keep the author line for the next quests (once the config is written): when it is new or was changed. */
export function keepAuthorLine(a: InterviewAnswers): void {
    const line: AuthorLine = {
        author: a.author ?? "", affiliation: a.affiliation ?? "",
        contact_email: a.contact_email ?? "", url: a.url ?? "",
    };
    const saved = loadProfile();
    if (!saved || JSON.stringify(saved) !== JSON.stringify(line)) saveProfile(line);
}

/** The study depth picker (review screen). */
async function pickStudyDepth(paperFormat: string, topic: string): Promise<"brief preprint" | "journal-length" | "comprehensive review" | undefined> {
    const isPolicyBrief = paperFormat === "policy_brief";
    const looksSurvey = /\b(survey|review of|compar)/i.test(topic);
    const studyDepthDefaultLabel = isPolicyBrief
        ? "brief preprint (recommended for policy_brief)"
        : looksSurvey
            ? "comprehensive review (recommended for survey topics)"
            : "journal-length (recommended)";
    const studyDepthChoice = await vscode.window.showQuickPick(
        [
            {
                label: `$(symbol-file) ${studyDepthDefaultLabel}`,
                description: isPolicyBrief
                    ? "1–2 pages; terse opening; novel findings only. Default for policy_brief."
                    : looksSurvey
                        ? "10–15 pages with Background + Comparison + Synthesis. ~4000+ words, 10+ discussed citations."
                        : "4–8 pages, full IMRAD or prose equivalent. ~1500–2500 words, ~15 citations.",
                value: (isPolicyBrief
                    ? "brief preprint"
                    : looksSurvey
                        ? "comprehensive review"
                        : "journal-length") as
                    | "brief preprint"
                    | "journal-length"
                    | "comprehensive review",
            },
            {
                label: "$(book) journal-length",
                description: "4–8 pages, full IMRAD. ~1500–2500 words, ~15 citations.",
                value: "journal-length" as const,
            },
            {
                label: "$(zap) brief preprint",
                description: "1–2 pages, terse, novel findings only.",
                value: "brief preprint" as const,
            },
            {
                label: "$(library) comprehensive review",
                description: "10–15 pages with extensive prior-work discussion.",
                value: "comprehensive review" as const,
            },
        ],
        {
            title: "Frontier Insight — study depth?",
            placeHolder:
                "Gates the paper's length and citation count. journal-length is the safe default.",
            ignoreFocusOut: true,
        },
    );
    return studyDepthChoice ? studyDepthChoice.value : undefined;
}

/** The multi-model ensemble picker and its models (advanced). FI does not choose the models. */
async function pickEnsemble(): Promise<{ profile: string; models: string } | undefined> {
    const ensembleChoice = await vscode.window.showQuickPick(
        [
            {
                label: "$(circle-outline) Single model (default, cheapest)",
                description: "One LLM call per node. Cost baseline = 1×.",
                value: "off" as const,
            },
            {
                label: "$(git-compare) Fan out cross_check only (~1.3× cost)",
                description: "3 models vote per finding; catches issues the writer might miss. Minimal extra spend.",
                value: "cross_check_only" as const,
            },
            {
                label: "$(lightbulb) Fan out ideate + cross_check (~2.0× cost)",
                description: "3 models brainstorm + moderator picks the strongest; 3 models vote on each finding.",
                value: "ideate_and_check" as const,
            },
            {
                label: "$(organization) Fan out ideate + analyze + cross_check (~2.5× cost)",
                description: "Multi-model on the three highest-leverage nodes. Best signal; highest cost.",
                value: "full" as const,
            },
        ],
        {
            title: "Frontier Insight — multi-model ensemble?",
            placeHolder:
                "Run the same node across multiple LLMs and merge their answers. Cost multiplies; agreement signal increases.",
            ignoreFocusOut: true,
        },
    );
    if (!ensembleChoice) return undefined;
    if (ensembleChoice.value === "off") return { profile: "off", models: "" };
    const picked = await pickEnsembleModels();
    if (picked === undefined) return undefined;
    if (picked.length < ENSEMBLE_MIN_MODELS) {
        void vscode.window.showWarningMessage(
            `Fewer than ${ENSEMBLE_MIN_MODELS} models picked — no ensemble will be configured. FI does not choose models for you.`,
        );
    }
    return { profile: ensembleChoice.value, models: picked.join(", ") };
}

/** The saved author line (core/profile.py; FI_PROFILE_PATH overrides the place), or undefined when not asked yet. */
function profilePath(): string {
    return process.env.FI_PROFILE_PATH || path.join(os.homedir(), ".frontier-insight", "profile.json");
}

function loadProfile(): AuthorLine | undefined {
    try {
        const raw = JSON.parse(fs.readFileSync(profilePath(), "utf-8"));
        if (!raw || typeof raw !== "object") return undefined;
        const clean = (v: unknown) => String(v ?? "").trim().replace(/\s+/g, " ").slice(0, 300);
        return { author: clean(raw.author), affiliation: clean(raw.affiliation),
            contact_email: clean(raw.contact_email), url: clean(raw.url) };
    } catch {
        return undefined;
    }
}

function saveProfile(line: AuthorLine): void {
    try {
        const target = profilePath();
        fs.mkdirSync(path.dirname(target), { recursive: true });
        const tmp = `${target}.${process.pid}.tmp`;
        fs.writeFileSync(tmp, JSON.stringify(line, null, 2) + "\n", "utf-8");
        fs.renameSync(tmp, target);
    } catch (e) {
        console.warn(`[fi] the author line could not be kept for the next quests: ${e}`);
    }
}

/** Inline editor for the tier-2 defaults. */
async function editTier2Field(a: InterviewAnswers, edited: Set<string> = new Set()): Promise<void> {
    const which = await vscode.window.showQuickPick(
        [
            { label: "Paper format / venue", value: "paper_format" },
            { label: "Deliverables", value: "output_kinds" },
            { label: "Study depth", value: "study_depth" },
            { label: "Title (folder slug)", value: "title" },
            { label: "Research approach (no_simulation)", value: "no_simulation" },
            { label: "Clarify mode", value: "clarify_mode" },
            { label: "Reviewer panel", value: "review_panel" },
            { label: "Knowledge layer (Axon)", value: "knowledge_enabled" },
            { label: "Web research (download sources)", value: "web_research" },
            { label: "Supply paywalled papers", value: "supply_papers" },
            { label: "Stop to read and edit the plan", value: "pause_for_plan" },
            { label: "What is the result for?", value: "result_use" },
            { label: "Pause for my papers / datasets", value: "pause_for_user_input" },
            { label: "Paper audience", value: "audience" },
            { label: "Axon (RAG) retrievals per quest (top_k)", value: "knowledge_top_k" },
            { label: "Author line (author, affiliation, email, link)", value: "author_line" },
        ],
        { title: "Edit which default?", ignoreFocusOut: true },
    );
    if (!which) return;
    if (which.value === "author_line") {
        const v = await askAuthorLine(a);
        if (v) Object.assign(a, v);
        return;
    }
    if (which.value === "paper_format") {
        const v = await pickPaperFormat();
        if (v) {
            a.paper_format = v;
            // What follows from it, as on the other interfaces (core/interview.py SMART_DEFAULTS), except what was
            // set by hand.
            if (!edited.has("study_depth")) a.study_depth = studyDepthFor(v, a.topic);
            if (!edited.has("no_simulation")) a.no_simulation = PROSE_FORMATS.has(v) || looksLikeSurvey(a.topic);
            const comp = a.study_depth === "comprehensive review";
            if (!edited.has("knowledge_top_k")) a.knowledge_top_k = comp ? 12 : 8;
            if (!edited.has("knowledge_external_top_k")) a.knowledge_external_top_k = comp ? 30 : 20;
        }
        return;
    }
    if (which.value === "output_kinds") {
        const v = await pickOutputKinds();
        if (v) {
            a.output_kinds = v;
            const note = missingToolsNote(v);
            if (note) void vscode.window.showWarningMessage(note.replace(/[*`#⚠️]/g, "").trim());
        }
        return;
    }
    if (which.value === "study_depth") {
        const v = await pickStudyDepth(a.paper_format, a.topic);
        if (v) {
            a.study_depth = v;
            const comp = v === "comprehensive review";
            if (!edited.has("knowledge_top_k")) a.knowledge_top_k = comp ? 12 : 8;
            if (!edited.has("knowledge_external_top_k")) a.knowledge_external_top_k = comp ? 30 : 20;
        }
        return;
    }
    if (which.value === "knowledge_top_k") {
        const v = await vscode.window.showInputBox({
            title: "Axon (RAG) retrievals per quest",
            value: String(a.knowledge_top_k),
            placeHolder: "8",
            ignoreFocusOut: true,
            validateInput: validatePositiveInt,
        });
        if (v !== undefined) {
            const n = parsePositiveInt(v);
            if (n !== null) a.knowledge_top_k = n;
        }
        return;
    }
    switch (which.value) {
        case "title": {
            const v = await vscode.window.showInputBox({ title: "Title", value: a.title, ignoreFocusOut: true });
            if (v !== undefined && v.trim()) a.title = v.trim();
            return;
        }
        case "no_simulation": {
            const v = await vscode.window.showQuickPick(
                [
                    { label: "$(zap) Computational (Python script produces data)", value: "computational" },
                    { label: "$(eye) Observational (needs real-world data)", value: "observational" },
                    { label: "$(book) Literature synthesis / survey (no experiment, no data)", value: "survey" },
                ],
                { title: "Research approach", ignoreFocusOut: true },
            );
            if (v) {
                a.survey_mode = v.value === "survey";
                // Survey and observational both skip the experiment; survey
                // additionally skips the dataset. Computational runs it.
                a.no_simulation = v.value !== "computational";
            }
            return;
        }
        case "clarify_mode": {
            const v = await vscode.window.showQuickPick(
                [
                    { label: "$(zap) auto — agent self-clarifies", value: "auto" as const },
                    { label: "$(question) interactive — pause for me", value: "interactive" as const },
                    { label: "$(rocket) off — just run it", value: "off" as const },
                ],
                { title: "Clarify mode", ignoreFocusOut: true },
            );
            if (v) a.clarify_mode = v.value;
            return;
        }
        case "review_panel": {
            const v = await vscode.window.showQuickPick(
                [
                    { label: "$(person) Single reviewer", value: [] as string[] },
                    {
                        label: "$(organization) 3-persona panel",
                        value: ["methodologist", "statistician", "devil_advocate"] as string[],
                    },
                    {
                        label: "$(organization) 4-persona panel",
                        value: ["methodologist", "statistician", "devil_advocate", "reproducibility"] as string[],
                    },
                ],
                { title: "Reviewer panel", ignoreFocusOut: true },
            );
            // Shown as it will run: research and a decision add the reviewers the research profile requires.
            if (v) a.review_panel = resolveReviewPanel(v.value, a.result_use ?? "research");
            return;
        }
        case "knowledge_enabled": {
            const v = await vscode.window.showQuickPick(
                [
                    { label: "$(circle-slash) Disabled", value: false },
                    { label: "$(database) Enabled (needs Axon)", value: true },
                ],
                { title: "Knowledge layer", ignoreFocusOut: true },
            );
            if (v) a.knowledge_enabled = v.value;
            return;
        }
        case "web_research": {
            const v = await vscode.window.showQuickPick(
                [
                    { label: "$(globe) On — search & download web sources", value: true },
                    { label: "$(circle-slash) Off — academic sources only", value: false },
                ],
                { title: "Web research (downloads into data/literature/)", ignoreFocusOut: true },
            );
            if (v) a.web_research = v.value;
            return;
        }
        case "supply_papers": {
            const v = await vscode.window.showQuickPick(
                [
                    { label: "$(book) Pause for my PDFs — list paywalled papers to download (default)", value: true },
                    { label: "$(circle-slash) Off — use what open access can get", value: false },
                ],
                { title: "Supply paywalled papers (pause → drop PDFs into inputs/papers/)", ignoreFocusOut: true },
            );
            if (v) a.supply_papers = v.value;
            return;
        }
        case "pause_for_plan": {
            // Maps to pauses.plan. Same wording as core/interview.py so the answer means the same
            // thing on every surface. The plan is written either way; this only decides whether the
            // quest waits for you to read and edit it (`@fi /plan <id>`), then `@fi /resume <id>`.
            const v = await vscode.window.showQuickPick(
                [
                    { label: "$(circle-slash) Don't stop (default) — plan.md is written, the quest goes on", value: false },
                    { label: "$(notebook) Stop to read and edit the plan — then @fi /resume", value: true },
                ],
                { title: "Stop to read and edit the plan (plan.md)", ignoreFocusOut: true },
            );
            if (v) a.pause_for_plan = v.value;
            return;
        }
        case "result_use": {
            // Same choices as the question asked first (core/interview.py:RESULT_USE_CHOICES). Changing it re-applies
            // the panel the new answer runs, so the review block keeps showing what will run.
            const v = await vscode.window.showQuickPick(
                [
                    { label: "$(verified) Research — every check stops the quest", value: "research" as const },
                    { label: "$(law) A decision — the same checks as Research", value: "decision" as const },
                    { label: "$(telescope) Exploring — a cheaper preliminary draft", value: "explore" as const },
                ],
                { title: "What is the result for?", ignoreFocusOut: true },
            );
            if (v) {
                a.result_use = v.value;
                a.rigor_profile = rigorProfileFor(v.value);
                a.review_panel = resolveReviewPanel(a.review_panel, v.value);
            }
            return;
        }
        case "pause_for_user_input": {
            // Maps to pauses.supply. Choices are verbatim from
            // core/interview.py so the same answer means the same thing
            // on every surface. The quest pause-exits and is picked up
            // again with `@fi /resume` (or `launch.py --resume`).
            const v = await vscode.window.showQuickPick(
                [
                    {
                        label: "$(circle-slash) Never (default)",
                        description: "Engine runs to completion without pause-drop opportunities.",
                        value: "never" as const,
                    },
                    {
                        label: "$(book) Pause after literature",
                        description: "Stop once the literature is saved; skills, design and the experiment start on resume with it in hand (the search is not run again).",
                        value: "after_literature" as const,
                    },
                    {
                        label: "$(debug-step-over) Pause after design",
                        description: "Drop reference papers / data BEFORE the implement → execute → analyze stages spend compute.",
                        value: "after_design" as const,
                    },
                    {
                        label: "$(debug-step-out) Pause after paper draft",
                        description: "Drop reference papers / data AFTER the first paper.md is written; useful for revise iterations.",
                        value: "after_paper" as const,
                    },
                    {
                        label: "$(debug-pause) Both",
                        description: "Pause after design AND after paper — most cost. Use for high-stakes manual review.",
                        value: "both" as const,
                    },
                ],
                {
                    title: "Pause for user-supplied papers / datasets",
                    placeHolder: "Stop mid-quest so you can drop PDFs into inputs/papers/ and data into inputs/data/, then resume.",
                    ignoreFocusOut: true,
                },
            );
            if (v) a.pause_for_user_input = v.value;
            return;
        }
        case "audience": {
            const v = await vscode.window.showQuickPick(
                [
                    {
                        label: "$(globe) external — journal / open web",
                        description: "Drop FI-internal entries from References",
                        value: "external" as const,
                    },
                    {
                        label: "$(briefcase) internal — team report / memo",
                        description: "Keep everything in Axon as citable prior work",
                        value: "internal" as const,
                    },
                ],
                { title: "Paper audience", ignoreFocusOut: true },
            );
            if (v) a.audience = v.value;
            return;
        }
    }
}

/** Inline editor for the tier-3 advanced fields. */
async function editTier3Field(a: InterviewAnswers, edited: Set<string> = new Set()): Promise<void> {
    const which = await vscode.window.showQuickPick(
        [
            { label: "Comparative baseline", value: "comparative_baseline" },
            { label: "Success metric", value: "success_metric" },
            { label: "Time / compute budget", value: "budget" },
            { label: "Per-node model overrides", value: "node_models" },
            { label: "External (web) retrievals per quest (external_top_k)", value: "knowledge_external_top_k" },
            { label: "Poster size", value: "poster_size" },
            { label: "Paper style", value: "paper_style" },
            { label: "Reasoning effort", value: "reasoning_effort" },
            { label: "Design-revise iteration budget", value: "max_iterations" },
            { label: "Page limit", value: "page_limit" },
            { label: "Multi-model ensemble (and its models)", value: "ensemble" },
        ],
        { title: "Edit which advanced field?", ignoreFocusOut: true },
    );
    if (!which) return;
    if (which.value === "ensemble") {
        const v = await pickEnsemble();
        if (v) {
            a.ensemble_profile = v.profile as InterviewAnswers["ensemble_profile"];
            a.ensemble_models = v.models;
        }
        return;
    }
    if (which.value === "page_limit") {
        // Written to output.page_limit. Blank is no set limit, and a limit the
        // topic states ("≤ 4 pages") still applies.
        const v = await vscode.window.showInputBox({
            title: "Page limit",
            prompt: "The most pages paper.pdf may take. Leave blank for no set limit; a limit the topic states still applies.",
            value: typeof a.page_limit === "number" ? String(a.page_limit) : "",
            placeHolder: "e.g. 4",
            ignoreFocusOut: true,
            validateInput: (s) => (s.trim() === "" ? null : validatePositiveInt(s)),
        });
        if (v !== undefined) a.page_limit = v.trim() === "" ? null : parsePositiveInt(v);
        return;
    }
    if (which.value === "paper_style") {
        // Written to output.paper_style. Labels verbatim from
        // PAPER_STYLES in core/interview.py.
        const v = await vscode.window.showQuickPick(
            [
                {
                    label: "LaTeX — Computer Modern article (default)",
                    description: "The classic typeset look via the venue LaTeX template. Best typography; needs a LaTeX engine (or the HTML fallback).",
                    value: "latex" as const,
                },
                {
                    label: "Briefing — Frontier Insight brand look",
                    description: "Warm paper, deep-teal accents, serif display + brand mark — the same identity as the slides and poster. Rendered via pandoc + a browser (no LaTeX); single-column regardless of venue.",
                    value: "briefing" as const,
                },
            ],
            { title: "Paper style", ignoreFocusOut: true },
        );
        if (v) a.paper_style = v.value;
        return;
    }
    if (which.value === "max_iterations") {
        // engine.max_iterations — the design → review → revise cap.
        const v = await vscode.window.showInputBox({
            title: "Design-revise iteration budget",
            prompt: "Hard cap on the design → review → revise loop. Lower = cheaper + faster; higher = more chances to fix what review caught. 2 is the default; bump to 3-4 only when you specifically want extra revise passes.",
            value: String(a.max_iterations),
            placeHolder: "2",
            ignoreFocusOut: true,
            validateInput: validatePositiveInt,
        });
        if (v !== undefined) {
            const n = parsePositiveInt(v);
            if (n !== null) a.max_iterations = n;
        }
        return;
    }
    if (which.value === "reasoning_effort") {
        // Written to provider.reasoning_effort. The VS Code bridge (vscode.lm)
        // has no such setting, so the level takes effect only once this YAML's
        // provider is one that has one — say so here, not only in the log.
        const levels = ["default", "minimal", "low", "medium", "high", "xhigh", "max"] as const;
        const v = await vscode.window.showQuickPick(
            levels.map((level) => ({
                label: level === "default" ? "Provider default (not set)" : level,
                value: level,
            })),
            {
                title: "Reasoning effort",
                placeHolder: "Applies on ollama, openai, codex_cli, claude_cli and antigravity_cli; the VS Code bridge has no such setting",
                ignoreFocusOut: true,
            },
        );
        if (v) a.reasoning_effort = v.value;
        return;
    }
    if (which.value === "poster_size") {
        const v = await vscode.window.showQuickPick(
            [
                { label: "A1 portrait — 59.4 × 84.1 cm (default)", description: "Two columns", value: "a1_portrait" as const },
                { label: "A0 portrait — 84.1 × 118.9 cm", description: "Two columns on a larger sheet", value: "a0_portrait" as const },
                { label: "48 × 36 in landscape — 121.9 × 91.4 cm", description: "Three columns", value: "landscape_48x36" as const },
            ],
            { title: "Poster size", ignoreFocusOut: true },
        );
        if (v) a.poster_size = v.value;
        return;
    }
    if (which.value === "knowledge_external_top_k") {
        const v = await vscode.window.showInputBox({
            title: "External (web) hits per quest",
            value: String(a.knowledge_external_top_k ?? 20),
            placeHolder: "20",
            ignoreFocusOut: true,
            // Use the strict decimal-integer parser so the validator
            // accepts the same set of strings Python's ``int(...)``
            // would — rejects ``2.0`` / ``1e2`` / ``0x10`` / ``0`` /
            // negatives. The save path uses the same predicate so a
            // value the validator marked invalid can never land
            // silently (no fallback to a stale hard-coded default).
            validateInput: validatePositiveInt,
        });
        if (v !== undefined) {
            const n = parsePositiveInt(v);
            if (n !== null) a.knowledge_external_top_k = n;
        }
        return;
    }
    if (which.value === "node_models") {
        // Not a text box: the models are the ones this VSCode offers, picked from a list.
        await editNodeModels(a);
        return;
    }
    const aBag = a as unknown as Record<string, unknown>;
    const v = await vscode.window.showInputBox({
        title: which.label,
        value: (aBag[which.value] as string) || "",
        // comparative_baseline, success_metric and budget are plain free text
        // with no format to hint.
        ignoreFocusOut: true,
    });
    if (v === undefined) return;
    aBag[which.value] = v.trim();
}
