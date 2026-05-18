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
import {
    InterviewAnswers,
    PaperFormat,
    answersToYaml,
    writeInterviewYaml,
    truncate,
    slugify,
} from "./interview-core";

// Re-export so callers don't have to know about the split.
export { InterviewAnswers, answersToYaml, writeInterviewYaml };

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

/**
 * Run the interview. Returns the answers, or `undefined` if the user
 * cancelled at any step (Esc on a modal, or empty topic).
 *
 * Flow:
 *   1. Tier-1 — four modals (topic, paper_format, output_kinds, study_depth).
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
        "🧪 **Let's set up a new research quest.** Four quick questions, then I'll show you the auto-derived defaults — edit anything before launch.\n\n",
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

    // 3. Output kinds — multi-select.
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
    if (!outputChoice) return undefined;
    stream.markdown(
        `  **Outputs:** ${(outputChoice.value as string[]).join(", ")}\n\n`,
    );

    // If the user asked for paper_pdf or slides or poster, check that
    // the required system tools are installed BEFORE the quest fires.
    // We don't refuse the choice — generators degrade gracefully and
    // skip the missing format — but the user should know upfront so
    // the "wait, where's my PDF" question doesn't happen later.
    const kinds = outputChoice.value as string[];
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
    if (missing.length > 0) {
        stream.markdown(
            "⚠️ **Missing tools** (selected outputs will be partial):\n\n" +
            missing.map((m) => `  - ${m}`).join("\n") +
            "\n\n",
        );
    }

    // 4. Paper format (article type) — the venue/style for the
    // written deliverable. Must stay in sync with PaperFormat in
    // core/config.py (scientific venues + non-scientific prose).
    // The clarify agent picks this slot when clarify_mode != "off";
    // exposing it here lets a user lock the format upfront without
    // burning a clarify call to "discover" what they already know.
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
    if (!paperFormatChoice) return undefined;
    const paperFormat: PaperFormat = paperFormatChoice.value;
    stream.markdown(`  **Paper format:** \`${paperFormat}\`\n\n`);

    // 5. Research approach — auto-derived in tier-2 (PROSE_FORMATS
    // → no_simulation=true). The user can override it from the
    // review screen.

    // 6. Study depth — drives paper length + citation depth.
    //    Smart-defaulted off the chosen paper_format: policy_brief
    //    is by definition 2-4 pages, so we default it to "brief
    //    preprint"; other formats default to "journal-length"
    //    unless the topic looks survey-shaped. Matches
    //    core/interview.py:smart_default_study_depth.
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
    if (!studyDepthChoice) return undefined;
    const studyDepth = studyDepthChoice.value;
    stream.markdown(`  **Study depth:** \`${studyDepth}\`\n\n`);

    // ─── Tier-2 derivation (mirrors core/interview.py SMART_DEFAULTS) ───
    // title — slug from topic; no_simulation — prose vs scientific;
    // others are static or auto-probed.
    const answers: InterviewAnswers = {
        topic: topic.trim(),
        title: suggestedTitle,
        output_kinds: outputChoice.value as string[],
        paper_format: paperFormat,
        clarify_mode: "auto",
        review_panel: [],
        knowledge_enabled: await probeAxonReachable(),
        no_simulation: PROSE_FORMATS.has(paperFormat),
        study_depth: studyDepth,
        comparative_baseline: "",
        success_metric: "",
        budget: "",
        provider_model: "",
        max_iterations: 2,
        audience: "external",
        knowledge_top_k: 5,
    };

    // ─── Review block + action picker loop ──────────────────────────
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
            return answers;
        }
        if (action.value === "edit_default") {
            await editTier2Field(answers);
            continue;
        }
        if (action.value === "edit_advanced") {
            await editTier3Field(answers);
            continue;
        }
    }
}

/**
 * Probe the Axon API sidecar on 127.0.0.1:8000 — if reachable, the
 * knowledge layer defaults to enabled. Otherwise off. Matches
 * core/interview.py:smart_default_knowledge_enabled.
 */
async function probeAxonReachable(): Promise<boolean> {
    const host = process.env.AXON_HOST || "127.0.0.1";
    const port = process.env.AXON_PORT || "8000";
    try {
        const ctrl = new AbortController();
        const timer = setTimeout(() => ctrl.abort(), 5000);
        const res = await fetch(`http://${host}:${port}/health/live`, { signal: ctrl.signal });
        clearTimeout(timer);
        return res.ok;
    } catch (e) {
        // Mirror the diagnostic probe in extension.ts so reviewers can
        // see WHY this fell back to disabled when the user expected
        // Axon to be detected. The original silent catch made
        // "axon is up but the interview defaulted to disabled" reports
        // impossible to debug.
        console.warn(
            `[fi] axon probe failed at http://${host}:${port}/health/live: `,
            e instanceof Error ? `${e.name}: ${e.message}` : String(e),
        );
        return false;
    }
}

function reviewBlockMarkdown(a: InterviewAnswers): string {
    const lines: string[] = [];
    lines.push("\n### Review before launch\n");
    lines.push("| Field | Value |");
    lines.push("|---|---|");
    lines.push(`| Title (auto-slug) | \`${a.title}\` |`);
    lines.push(`| Research approach | ${a.no_simulation ? "observational" : "computational"} |`);
    lines.push(`| Clarify mode | \`${a.clarify_mode}\` |`);
    lines.push(`| Reviewer panel | ${a.review_panel.length === 0 ? "single reviewer" : a.review_panel.join(", ")} |`);
    lines.push(`| Knowledge layer (Axon) | ${a.knowledge_enabled ? "enabled (sidecar detected)" : "disabled"} |`);
    lines.push(`| Paper audience | \`${a.audience}\` |`);
    // Tier-3 fields are only worth printing if the user has set them.
    if (a.comparative_baseline || a.success_metric || a.budget || a.knowledge_top_k !== 5) {
        lines.push("\n_Advanced overrides:_");
        if (a.comparative_baseline) lines.push(`  • baseline: ${a.comparative_baseline}`);
        if (a.success_metric) lines.push(`  • metric: ${a.success_metric}`);
        if (a.budget) lines.push(`  • budget: ${a.budget}`);
        if (a.knowledge_top_k !== 5) lines.push(`  • top_k: ${a.knowledge_top_k}`);
    }
    lines.push("");
    return lines.join("\n");
}

/** Inline editor for the six tier-2 defaults. */
async function editTier2Field(a: InterviewAnswers): Promise<void> {
    const which = await vscode.window.showQuickPick(
        [
            { label: "Title (folder slug)", value: "title" },
            { label: "Research approach (no_simulation)", value: "no_simulation" },
            { label: "Clarify mode", value: "clarify_mode" },
            { label: "Reviewer panel", value: "review_panel" },
            { label: "Knowledge layer (Axon)", value: "knowledge_enabled" },
            { label: "Paper audience", value: "audience" },
        ],
        { title: "Edit which default?", ignoreFocusOut: true },
    );
    if (!which) return;
    switch (which.value) {
        case "title": {
            const v = await vscode.window.showInputBox({ title: "Title", value: a.title, ignoreFocusOut: true });
            if (v !== undefined && v.trim()) a.title = v.trim();
            return;
        }
        case "no_simulation": {
            const v = await vscode.window.showQuickPick(
                [
                    { label: "$(zap) Computational (Python script produces data)", value: false },
                    { label: "$(eye) Observational (needs real-world data)", value: true },
                ],
                { title: "Research approach", ignoreFocusOut: true },
            );
            if (v) a.no_simulation = v.value;
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
            if (v) a.review_panel = v.value;
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

/** Inline editor for the four tier-3 advanced fields. */
async function editTier3Field(a: InterviewAnswers): Promise<void> {
    const which = await vscode.window.showQuickPick(
        [
            { label: "Comparative baseline", value: "comparative_baseline" },
            { label: "Success metric", value: "success_metric" },
            { label: "Time / compute budget", value: "budget" },
            { label: "Prior-work retrievals per quest (top_k)", value: "knowledge_top_k" },
        ],
        { title: "Edit which advanced field?", ignoreFocusOut: true },
    );
    if (!which) return;
    if (which.value === "knowledge_top_k") {
        const v = await vscode.window.showInputBox({
            title: "Prior-work retrievals per quest",
            value: String(a.knowledge_top_k),
            placeHolder: "5",
            ignoreFocusOut: true,
            validateInput: (s) => (s && /^\d+$/.test(s.trim()) ? null : "must be a positive integer"),
        });
        if (v !== undefined) a.knowledge_top_k = parseInt(v.trim(), 10) || 5;
        return;
    }
    const aBag = a as unknown as Record<string, unknown>;
    const v = await vscode.window.showInputBox({
        title: which.label,
        value: (aBag[which.value] as string) || "",
        ignoreFocusOut: true,
    });
    if (v === undefined) return;
    aBag[which.value] = v.trim();
}
