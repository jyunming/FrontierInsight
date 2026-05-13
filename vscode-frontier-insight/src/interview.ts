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
                label: "$(layout) paper + slides",
                description: "slides via Marp; .pptx via pandoc",
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

    // 4. Clarify mode — does the agent ask follow-up questions before starting?
    const clarifyChoice = await vscode.window.showQuickPick(
        [
            {
                label: "$(zap) Agent self-clarifies (recommended)",
                description: "Agent generates 6 questions AND answers them itself — study_depth/paper_venue flow through to write+review",
                value: "auto" as const,
            },
            {
                label: "$(question) Ask me 6 questions",
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

    // 5. Reviewer panel — debate or single reviewer?
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

    // 6. Knowledge layer — opt in only if Axon is set up.
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
        clarify_mode: clarifyChoice.value,
        review_panel: panel,
        knowledge_enabled: knowledgeChoice.value,
        max_iterations: 2,
    };
}
