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

/**
 * Run the interview. Returns the answers, or `undefined` if the user
 * cancelled at any step (Esc on a modal, or empty topic).
 */
export async function runInterview(
    stream: vscode.ChatResponseStream,
): Promise<InterviewAnswers | undefined> {
    stream.markdown(
        "🧪 **Let's set up a new research quest.** I'll ask a few quick questions; press Esc on any modal to cancel.\n\n",
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

    // 2. Title — auto-suggest from topic.
    const suggestedTitle = slugify(topic).slice(0, 40) || "quest";
    const title = await vscode.window.showInputBox({
        title: "Frontier Insight — quest title (short slug)",
        prompt: "Short identifier for this quest. Used in folder names.",
        value: suggestedTitle,
        ignoreFocusOut: true,
    });
    if (title === undefined) return undefined;
    stream.markdown(`  **Title:** \`${title || suggestedTitle}\`\n\n`);

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

    // 5. Research approach — computational vs. observational.
    // Maps to ``engine.no_simulation`` in YAML. Pre-setting this
    // bypasses the clarify-auto-detect path: when True the engine
    // skips implement → execute and routes to wait_for_data /
    // auto_collect_data instead. The clarify agent calls this
    // ``simulatability``; we surface it under a name a non-technical
    // user can answer without reading the engine docs.
    //
    // Auto-hint the default off the paper_format choice the user
    // just made — non-scientific formats almost always want
    // no_simulation=true. Saves the user one extra click in the
    // typical case while leaving the gate fully overridable.
    const isProseFormat = ["essay", "report", "policy_brief", "whitepaper"].includes(paperFormat);
    const simulatableLabel = "$(zap) Computational — a Python script can produce the data";
    const observationalLabel = "$(eye) Observational — needs real-world data the engine can't simulate";
    const noSimChoice = await vscode.window.showQuickPick(
        [
            isProseFormat
                ? {
                    label: `${observationalLabel} (recommended for ${paperFormat})`,
                    description: "Skip implement/execute; route to wait_for_data + auto_collect_data. For cultural / historical / qualitative / policy topics.",
                    value: true,
                }
                : {
                    label: `${simulatableLabel} (recommended for ${paperFormat})`,
                    description: "Run normal implement → execute pipeline. For physics / ML / algorithmic / benchmark topics.",
                    value: false,
                },
            isProseFormat
                ? {
                    label: simulatableLabel,
                    description: "Override: run the implement/execute pipeline even though the paper format is prose.",
                    value: false,
                }
                : {
                    label: observationalLabel,
                    description: "Override: skip simulation even though the paper format is scientific.",
                    value: true,
                },
        ],
        {
            title: "Frontier Insight — research approach?",
            placeHolder:
                "Decides whether the engine runs a Python experiment or waits for real-world data. Matches the clarify agent's 'simulatability' slot.",
            ignoreFocusOut: true,
        },
    );
    if (!noSimChoice) return undefined;
    const noSimulation = noSimChoice.value as boolean;
    stream.markdown(
        `  **Research approach:** ${noSimulation ? "observational (no_simulation=true)" : "computational (no_simulation=false)"}\n\n`,
    );

    // 6. Study depth — drives paper length + citation depth.
    //    Smart-defaulted off the chosen paper_format: policy_brief
    //    is by definition 2-4 pages, so we default it to "brief
    //    preprint"; other formats default to "journal-length"
    //    unless the topic looks survey-shaped. Matches Phase R in
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

    // 7-9. Topic-tuned clarify slots. The Python frontends make ONE
    // LLM call here (via agents/clarify_preflight.md) to suggest
    // topic-tuned defaults; in VSCode, the user has already picked
    // a model and we *could* fire it, but staying simple for the
    // initial parity ship: prompt with static placeholders. The
    // engine's clarify-overrides path still honors whatever the
    // user types here.
    stream.markdown(
        "ℹ️ The next 3 questions sharpen the agent's framing. Press Enter to accept the bracketed placeholder.\n\n",
    );
    const comparativeBaseline = await vscode.window.showInputBox({
        title: "Frontier Insight — comparative baseline",
        prompt: "What existing method / dataset / regime should this study be compared against?",
        placeHolder: "e.g. RandomForest baseline on the same features",
        ignoreFocusOut: true,
    });
    if (comparativeBaseline === undefined) return undefined;

    const successMetric = await vscode.window.showInputBox({
        title: "Frontier Insight — success metric",
        prompt: "What number changing in what direction = headline result?",
        placeHolder: "e.g. AUC ≥ 0.9 on held-out test set",
        ignoreFocusOut: true,
    });
    if (successMetric === undefined) return undefined;

    const budget = await vscode.window.showInputBox({
        title: "Frontier Insight — time / compute budget",
        prompt: "Soft cap on wall-clock for the experiment.",
        placeHolder: "e.g. a few minutes on a laptop CPU",
        ignoreFocusOut: true,
    });
    if (budget === undefined) return undefined;

    // 10. Clarify mode — does the agent ask follow-up questions before starting?
    const clarifyChoice = await vscode.window.showQuickPick(
        [
            {
                label: "$(zap) Agent self-clarifies (recommended)",
                description: "Agent generates 7 questions AND answers them itself — study_depth/paper_venue flow through to write+review",
                value: "auto" as const,
            },
            {
                label: "$(question) Ask me 7 questions",
                description: "Pauses after generating questions; you fill in answers — highest quality, most interruption",
                value: "interactive" as const,
            },
            {
                label: "$(rocket) Just run it",
                description: "Agent picks everything from the topic alone (no clarify; paper may be shallower)",
                value: "off" as const,
            },
        ],
        {
            title: "Frontier Insight — pre-flight clarification?",
            placeHolder:
                "Helps the agent narrow scope before designing the experiment. 'Just run it' is fastest.",
            ignoreFocusOut: true,
        },
    );
    if (!clarifyChoice) return undefined;
    stream.markdown(`  **Clarify mode:** \`${clarifyChoice.value}\`\n\n`);

    // 7. Reviewer panel — debate or single reviewer?
    const panelChoice = await vscode.window.showQuickPick(
        [
            {
                label: "$(person) Single reviewer",
                description: "1 LLM call per review pass (default, cheapest)",
                value: [] as string[],
            },
            {
                label: "$(organization) 3-persona panel",
                description:
                    "Methodologist + Statistician + Devil's-advocate, then moderator. ~4× the review cost.",
                value: ["methodologist", "statistician", "devil_advocate"] as string[],
            },
            {
                label: "$(organization) 4-persona panel",
                description: "Adds Reproducibility reviewer. ~5× the review cost.",
                value: ["methodologist", "statistician", "devil_advocate", "reproducibility"] as string[],
            },
        ],
        {
            title: "Frontier Insight — reviewer panel?",
            placeHolder:
                "Single reviewer is fine for most. Use the panel when correctness matters more than cost.",
            ignoreFocusOut: true,
        },
    );
    if (!panelChoice) return undefined;
    const panel = panelChoice.value as string[];
    stream.markdown(
        `  **Reviewer panel:** ${panel.length === 0 ? "single reviewer" : panel.join(", ")}\n\n`,
    );

    // 8. Knowledge layer — opt in only if Axon is set up.
    const knowledgeChoice = await vscode.window.showQuickPick(
        [
            {
                label: "$(circle-slash) Disabled (recommended for first runs)",
                description:
                    "Literature retrieval falls back to free public sources (arXiv, OpenAlex, Crossref) when needed.",
                value: false,
            },
            {
                label: "$(database) Enabled (requires Axon installed)",
                description:
                    "Use your Axon corpus for retrieval + write-back. Skip if you haven't set Axon up.",
                value: true,
            },
        ],
        {
            title: "Frontier Insight — Axon knowledge layer?",
            placeHolder: "Most first-time users should leave this disabled.",
            ignoreFocusOut: true,
        },
    );
    if (!knowledgeChoice) return undefined;
    stream.markdown(
        `  **Knowledge layer:** ${knowledgeChoice.value ? "Axon (enabled)" : "disabled (public-source fallback)"}\n\n`,
    );

    return {
        topic: topic.trim(),
        title: (title || suggestedTitle).trim(),
        output_kinds: outputChoice.value as string[],
        paper_format: paperFormat,
        clarify_mode: clarifyChoice.value,
        review_panel: panel,
        knowledge_enabled: knowledgeChoice.value,
        no_simulation: noSimulation,
        // Phase R — research-shaping fields.
        study_depth: studyDepth,
        comparative_baseline: (comparativeBaseline || "").trim(),
        success_metric: (successMetric || "").trim(),
        budget: (budget || "").trim(),
        // ``runInterview`` is called from extension.ts which knows the
        // active Copilot model. It overrides this stub with
        // ``userPickedModel.family`` before invoking writeInterviewYaml.
        provider_model: "",
        max_iterations: 2,
    };
}
