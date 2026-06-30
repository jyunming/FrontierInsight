/**
 * interview-core — pure-logic half of the interview module, free of
 * any `vscode` imports so it can be unit-tested from plain Node.
 *
 * The UI half (`interview.ts`) lives next to this file; it owns the
 * `showInputBox` / `showQuickPick` flow. Anything that doesn't touch
 * `vscode.*` goes here so the tests can exercise it without the
 * VSCode runtime.
 */
import * as fs from "fs";
import * as path from "path";

/** The nine paper-format venues the engine supports. Must stay in
 * sync with ``PaperFormat`` in core/config.py and the venue rules in
 * agents/clarify.md — both pin the same Literal. */
export type PaperFormat =
    | "generic"
    | "neurips"
    | "iclr"
    | "ieee_access"
    | "nature_mi"
    | "essay"
    | "report"
    | "policy_brief"
    | "whitepaper";

export interface InterviewAnswers {
    topic: string;
    title: string;
    output_kinds: string[];
    paper_format: PaperFormat;
    paper_style?: "latex" | "briefing";   // paper.pdf look; default "latex"
    clarify_mode: "off" | "auto" | "interactive";
    review_panel: string[];           // empty = single reviewer
    knowledge_enabled: boolean;
    no_simulation: boolean;
    // Study depth + three topic-tuned clarify slots. The interview
    // collects them so the engine's clarify node short-circuits in
    // auto mode (pinned values win over LLM-generated defaults).
    // Must stay in sync with core/interview.py:InterviewAnswers
    // (Python source of truth). The schema-parity test in
    // tests/test_interview_schema_parity.py fails CI if these drift.
    study_depth: "brief preprint" | "journal-length" | "comprehensive review";
    comparative_baseline: string;
    success_metric: string;
    budget: string;
    // The Copilot model the user has selected in the chat view at
    // interview time, snapshotted into ``provider.model`` so the
    // quest stays on a consistent model even if the user changes
    // Copilot model later. Captured via vscode.lm.selectChatModels()
    // — empty string means "no model captured yet" and the bridge
    // resolves per-call.
    provider_model: string;
    max_iterations: number;
    // Citation audience for the published paper. "external" drops
    // FI-internal cross-quest memory from References; "internal"
    // keeps everything (project reports, internal memos).
    audience: "external" | "internal";
    // Axon (RAG) retrievals per quest. 8 default; the Python
    // smart-default bumps to 12 for ``study_depth: comprehensive review``.
    knowledge_top_k: number;
    // External (arXiv / OpenAlex / Crossref / S2 / ...) hits per quest
    // when Axon misses. 20 default; bumped to 30 for comprehensive
    // reviews. Sized independently of ``knowledge_top_k`` because web
    // search is coarser than RAG and a literature scan needs breadth.
    knowledge_external_top_k?: number;
    // Web research layer. When true (default), the literature node
    // searches the public web (Brave/DuckDuckGo) and downloads sources
    // into data/literature/ for every quest — sim and observational
    // alike. Maps to knowledge.web_search + web_fetch_pages. Independent
    // of knowledge_enabled (the Axon corpus toggle). Must stay in sync
    // with core/interview.py:InterviewAnswers.
    web_research?: boolean;
    // When true, pause on a paywalled / abstract-only relevant paper and
    // write needs/WANTED_PAPERS.md so the user can drop the PDF into
    // inputs/papers/ and resume. Maps to pauses.papers.
    // Off by default. Must stay in sync with core/interview.py.
    supply_papers?: boolean;
    // Multi-model ensemble preset. "off" (default) keeps single-call
    // semantics; other values expand into provider.node_ensemble via
    // the Python `expand_ensemble_profile` helper at YAML emit time.
    ensemble_profile?: "off" | "cross_check_only" | "ideate_and_check" | "full";
}


/**
 * Per-provider model trios used by every ensemble profile. Mirrors
 * ``_ENSEMBLE_MODEL_TRIOS`` in core/interview.py.
 *
 * Drift between the two definitions is caught by
 * ``test_ts_ensemble_model_trios_match_python`` in
 * ``tests/test_interview_schema_parity.py`` — that test parses this
 * record out of the TS file and compares it to the JSON snapshot.
 */
export const ENSEMBLE_MODEL_TRIOS: Record<string, [string, string, string]> = {
    "openai": ["gpt-5", "gpt-5-mini", "o3-mini"],
    "codex": ["gpt-5", "gpt-5-mini", "gpt-4o"],
    "claude_cli": ["claude-opus-4-7", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
    "codex_cli": ["gpt-5", "gpt-5-mini", "gpt-4o"],
    "copilot_cli": ["gpt-5", "claude-opus-4-7", "gemini-2.5-pro"],
    "gemini_cli": ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash"],
    "ollama": ["llama3.3:70b", "qwen2.5:32b", "qwen2.5:7b"],
    "vscode_extension": ["gpt-5", "claude-opus-4-7", "gemini-2.5-pro"],
};


/**
 * Expand the ensemble profile into a node_ensemble dict shape.
 * Returns null for "off" — and also for any unrecognized profile —
 * so the YAML emitter can skip the block entirely instead of writing
 * a dangling ``node_ensemble:`` line with no children.
 */
type EnsembleProfile = "off" | "cross_check_only" | "ideate_and_check" | "full";

export function expandEnsembleProfile(
    profile: string,
    provider: string,
    providerModel: string | undefined,
): Record<string, {models: string[]; merge: string; moderator?: string}> | null {
    const known: ReadonlySet<EnsembleProfile> = new Set([
        "off", "cross_check_only", "ideate_and_check", "full",
    ]);
    if (!profile || profile === "off" || !known.has(profile as EnsembleProfile)) {
        return null;
    }
    const trio = ENSEMBLE_MODEL_TRIOS[provider]
        ?? [providerModel || "default", providerModel || "default", providerModel || "default"];
    const moderator = trio[0];
    const out: Record<string, {models: string[]; merge: string; moderator?: string}> = {};
    if (profile === "cross_check_only" || profile === "ideate_and_check" || profile === "full") {
        out["cross_check"] = { models: [...trio], merge: "vote" };
    }
    if (profile === "ideate_and_check" || profile === "full") {
        out["ideate"] = { models: [...trio], merge: "tournament", moderator };
    }
    if (profile === "full") {
        // analyze MUST be tournament — the ProviderConfig validator
        // rejects analyze.merge: synthesize because the analyze node
        // parses output as JSON, which only tournament preserves verbatim.
        out["analyze"] = { models: [...trio], merge: "tournament", moderator };
    }
    // Defensive: if no branches matched (future profile name added on
    // the Python side and not yet here), prefer returning null so the
    // YAML emitter skips the block instead of writing ``node_ensemble:``
    // with no children.
    return Object.keys(out).length > 0 ? out : null;
}

/**
 * Render the interview answers as a complete config.yaml string,
 * ready to write to disk. The provider block is always
 * `vscode_extension` — we're generating this from inside the FI
 * VSCode extension, so the bridge transport is the only sensible
 * default. A user who wants a different provider can edit the YAML
 * afterwards and re-run via `@fi /start <path>`.
 */
export function answersToYaml(answers: InterviewAnswers): string {
    const indent = "  ";
    const lines: string[] = [];

    lines.push("# Auto-generated by the Frontier Insight VSCode extension.");
    lines.push("# You can edit this file and re-run with `@fi /start <path>`.");
    lines.push("");

    lines.push("topic: |");
    for (const line of answers.topic.split("\n")) {
        lines.push(`${indent}${line}`);
    }
    lines.push("");

    // Always quote string-valued scalars. YAML 1.1's "Norway
    // problem" auto-coerces unquoted bareword tokens like
    // `off` / `on` / `no` / `yes` / `n` / `y` / `false` / `true`
    // into Python booleans before the schema validator sees them.
    // `clarify_mode: off` was the most common bite; quoting every
    // string field is the cheap, defensive fix.
    lines.push(`title: "${yamlEscape(answers.title)}"`);
    lines.push("");

    lines.push("provider:");
    lines.push(`${indent}name: "vscode_extension"`);
    // Pin the active Copilot model the user selected when
    // they ran @fi /new. Empty string means "let the bridge decide
    // per call" (older flow); non-empty pins it for the whole quest.
    if (answers.provider_model) {
        lines.push(`${indent}model: "${yamlEscape(answers.provider_model)}"`);
    }
    // Multi-model ensemble preset. Only emit when non-"off".
    const nodeEnsemble = expandEnsembleProfile(
        answers.ensemble_profile ?? "off",
        "vscode_extension",
        answers.provider_model,
    );
    if (nodeEnsemble) {
        lines.push(`${indent}node_ensemble:`);
        for (const [node, cfg] of Object.entries(nodeEnsemble)) {
            lines.push(`${indent}${indent}${node}:`);
            const quoted = cfg.models.map((m) => `"${yamlEscape(m)}"`).join(", ");
            lines.push(`${indent}${indent}${indent}models: [${quoted}]`);
            lines.push(`${indent}${indent}${indent}merge: "${cfg.merge}"`);
            if (cfg.moderator) {
                lines.push(`${indent}${indent}${indent}moderator: "${yamlEscape(cfg.moderator)}"`);
            }
        }
    }
    lines.push("");

    lines.push("engine:");
    lines.push(`${indent}framework: "langgraph"`);
    lines.push(`${indent}max_iterations: ${answers.max_iterations}`);
    lines.push(`${indent}review_loop: true`);
    lines.push(`${indent}no_simulation: ${answers.no_simulation ? "true" : "false"}`);
    // ---- COST DISCIPLINE — these defaults are LOW so a basic quest is
    // ~6 LLM calls instead of ~26. Each line below switches off an
    // expensive feature that the Python defaults turn ON. A user who
    // wants the full research-quality loop edits the YAML by hand.
    lines.push(`${indent}ideate_reflect: false`);
    lines.push(`${indent}cross_check_per_finding_k: 0`);
    lines.push(`${indent}enable_analyze_reroute: false`);
    if (answers.review_panel.length > 0) {
        lines.push(`${indent}review_panel:`);
        for (const persona of answers.review_panel) {
            lines.push(`${indent}${indent}- "${yamlEscape(persona)}"`);
        }
    } else {
        lines.push(`${indent}review_panel: []`);
    }
    // Pin the interview's research-shaping answers into
    // clarify_overrides so the engine's clarify node short-circuits
    // in auto mode (no LLM call needed). Mirrors the equivalent
    // block in core/interview.py:answers_to_yaml.
    lines.push(`${indent}clarify_overrides:`);
    lines.push(`${indent}${indent}study_depth: "${yamlEscape(answers.study_depth)}"`);
    lines.push(`${indent}${indent}paper_venue: "${yamlEscape(answers.paper_format)}"`);
    const kindsCsv = answers.output_kinds.map((k) => `"${yamlEscape(k)}"`).join(", ");
    lines.push(`${indent}${indent}output_kinds: [${kindsCsv}]`);
    lines.push(`${indent}${indent}simulatability: "${answers.no_simulation ? "no" : "yes"}"`);
    lines.push(`${indent}${indent}comparative_baseline: "${yamlEscape(answers.comparative_baseline)}"`);
    lines.push(`${indent}${indent}success_metric: "${yamlEscape(answers.success_metric)}"`);
    lines.push(`${indent}${indent}budget: "${yamlEscape(answers.budget)}"`);
    lines.push("");

    // Unified human-in-the-loop section: every place the quest stops for you.
    // Uses the coherent `pauses.*` vocabulary (clarify interactive → ask).
    // Mirrors the equivalent block in core/interview.py:answers_to_yaml.
    const clarifyPause =
        answers.clarify_mode === "interactive" ? "ask" : answers.clarify_mode;
    lines.push("pauses:");
    lines.push(`${indent}clarify: "${clarifyPause}"`);
    if (answers.supply_papers === true) {
        lines.push(`${indent}papers: true`);
    }
    lines.push("");

    lines.push("execution:");
    lines.push(`${indent}sandbox: "venv"`);
    lines.push(`${indent}timeout_s: 600`);
    lines.push("");

    lines.push("knowledge:");
    lines.push(`${indent}enabled: ${answers.knowledge_enabled ? "true" : "false"}`);
    // Emit top_k only when it differs from the engine default (8).
    if (answers.knowledge_top_k && answers.knowledge_top_k !== 8) {
        lines.push(`${indent}top_k: ${answers.knowledge_top_k}`);
    }
    // Emit external_top_k only when it differs from the engine default (20).
    const extK = answers.knowledge_external_top_k;
    if (extK !== undefined && extK !== null && extK !== 20) {
        lines.push(`${indent}external_top_k: ${extK}`);
    }
    // Web research layer — search the public web (Brave/DuckDuckGo) and
    // download the sources into data/literature/, on top of academic
    // retrieval. Emitted explicitly (on AND off) so the generated config
    // self-documents the interview choice. Independent of `enabled`.
    // Mirrors core/interview.py:answers_to_yaml.
    const webResearch = answers.web_research !== false;
    lines.push(`${indent}web_search: ${webResearch ? "true" : "false"}`);
    lines.push(`${indent}web_fetch_pages: ${webResearch ? "true" : "false"}`);
    if (!answers.knowledge_enabled) {
        lines.push(`${indent}external_fallback: ["openalex", "arxiv", "crossref"]`);
        // `source_routing: manual` skips the LLM-driven catalog source
        // picker — saves 1 call per asearch (so ~3–5 calls per quest).
        // With Axon disabled the router would otherwise fire for every
        // ideate-seed and literature retrieval.
        lines.push(`${indent}source_routing: "manual"`);
    }
    lines.push("");

    lines.push("output:");
    const quotedKinds = answers.output_kinds.map((k) => `"${yamlEscape(k)}"`).join(", ");
    lines.push(`${indent}kinds: [${quotedKinds}]`);
    lines.push(`${indent}paper_format: "${yamlEscape(answers.paper_format)}"`);
    // Only emit a non-default paper_style so configs stay clean.
    if (answers.paper_style && answers.paper_style !== "latex") {
        lines.push(`${indent}paper_style: "${yamlEscape(answers.paper_style)}"`);
    }
    // Emit audience only when it differs from the safer default ("external").
    if (answers.audience && answers.audience !== "external") {
        lines.push(`${indent}audience: "${yamlEscape(answers.audience)}"`);
    }
    lines.push(`${indent}output_dir: "./outputs"`);
    lines.push("");

    return lines.join("\n");
}

/**
 * Write the generated YAML to a draft folder inside the repo's
 * `outputs/_drafts/` so it lives adjacent to where the quest's
 * artifacts will land. Returns the absolute path to the file.
 */
export function writeInterviewYaml(
    answers: InterviewAnswers,
    repoPath: string,
): string {
    const yaml = answersToYaml(answers);
    const draftDir = path.join(repoPath, "outputs", "_drafts");
    fs.mkdirSync(draftDir, { recursive: true });
    const filename = `${stamp()}-${slugifyTitle(answers.title)}.yaml`;
    const filepath = path.join(draftDir, filename);
    fs.writeFileSync(filepath, yaml, "utf-8");
    return filepath;
}

/**
 * Escape a string for inclusion in a YAML double-quoted scalar.
 * Handles the two characters that matter inside `"..."`: backslash
 * and the closing double-quote. (Tabs/newlines aren't possible here
 * because our inputs come from `showInputBox`, which is one-line.)
 */
export function yamlEscape(s: string | null | undefined): string {
    // Tolerate undefined / null — callers may pass an optional field
    // that the answers object hasn't populated yet. Treat missing as
    // empty string instead of crashing with a .replace TypeError.
    if (s === null || s === undefined) return "";
    return s.replace(/\\/g, "\\\\").replace(/"/g, '\\"');
}


export function slugify(s: string): string {
    // Preserve Unicode letters (CJK / Cyrillic / Greek / ...) so a
    // Traditional Chinese topic yields a real slug instead of an empty
    // string. ``\p{L}`` is the Unicode-letter category; ``\p{N}`` digits.
    // Mirrors ``core/interview.py:slugify``.
    return s
        .toLowerCase()
        .replace(/[^\p{L}\p{N}\s_-]/gu, "")
        .replace(/_+/g, "-")
        .replace(/\s+/g, "-")
        .replace(/-+/g, "-")
        .replace(/^-|-$/g, "");
}

function slugifyTitle(s: string): string {
    // Same Unicode-friendly policy as ``slugify`` so titles in non-Latin
    // scripts survive the per-character normalization pass.
    return s
        .toLowerCase()
        .replace(/[^\p{L}\p{N}-]/gu, "-");
}

function stamp(): string {
    const d = new Date();
    const pad = (n: number) => String(n).padStart(2, "0");
    return (
        `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}` +
        `-${pad(d.getHours())}${pad(d.getMinutes())}`
    );
}

export function truncate(s: string, n: number): string {
    return s.length > n ? s.slice(0, n - 1) + "…" : s;
}
