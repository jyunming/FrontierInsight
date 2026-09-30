/**
 * Frontier Insight VSCode extension — entry point.
 *
 * Registers the `@fi` chat participant. When the user types
 * `@fi /start <path-to-config.yaml>` (or `@fi /fleet a.yaml b.yaml ...`)
 * we:
 *
 *   1. Bind a free localhost TCP port via the Bridge class.
 *   2. Spawn the FI Python engine with --vscode-bridge-port <N>.
 *   3. Forward every Python `lm_request` to `vscode.lm.*` and stream
 *      the response back. Render progress in the chat panel.
 *
 * The Python engine drives the actual research loop (clarify, ideate,
 * literature, design, implement, execute, execute_reflect, analyze,
 * cross_check, write, review). All LLM calls go through `vscode.lm`,
 * which means: sanctioned API, user-consented, normal Copilot quota.
 */
import * as vscode from "vscode";
import * as fs from "fs";
import * as fsPromises from "fs/promises";
import * as path from "path";
import { spawn } from "child_process";
import {
    runApproveAllSkills,
    runApproveAmendment,
    runAcceptChecks,
    runApproveSkill,
    runImportSkill,
    runListSkills,
    runRevokeSkill,
    runRemoveSkill,
    runScanSkill,
    runTeachSkill,
} from "./skills";
import { Bridge } from "./bridge";
import { splitRevisePlan, unknownResumeFlags } from "./resume-args";
import { forgetFoundFi, rootsForCommand, workFolderForCommand } from "./roots-config";
import type { Roots } from "./roots";
import { PersistentBridge } from "./persistent-bridge";
import { persistentBridgePath } from "./bridge-path";
import {
    ShellKind,
    detectShell,
    updateTerminalCommand,
} from "./terminal-command";
import { keepAuthorLine, runInterview, writeInterviewYaml } from "./interview";
import { AxonDiscovery, discoverAxon } from "./axon-endpoint";
import { runProbe } from "./probe";
import { openQuestMap, tabAreaColumn } from "./quest-map";
import { IndexEntry, describeAmbiguous, findInIndex, loadIndex, matchIds, shortId } from "./quest-index";
import { runFollow, runRename, runRerunSteps, runTrace, runWhy } from "./trace";


/**
 * The `frontierInsight.axonUrl` override, or undefined when unset.
 * An explicit setting is the one candidate that beats discovery — it is
 * how a user points the extension at a remote or oddly-placed sidecar.
 */
/**
 * How long the activation probe keeps watching for a sidecar before it
 * says anything. Axon loads an embedding model and opens vector indexes
 * before its server accepts requests, and a user may well start it
 * minutes after opening the editor — so the default is generous. The
 * wait is silent: a sidecar that shows up at any point during it
 * produces no notification at all.
 */
const DEFAULT_AXON_WAIT_SEC = 600;


function axonUrlSetting(): string | undefined {
    const raw = vscode.workspace
        .getConfiguration("frontierInsight")
        .get<string>("axonUrl");
    return raw && raw.trim() ? raw.trim() : undefined;
}


/** Async existence check — avoids the sync fs.existsSync call that
 *  blocks the extension host event loop on slow filesystems. */
async function fsExists(p: string): Promise<boolean> {
    try {
        await fsPromises.access(p);
        return true;
    } catch {
        return false;
    }
}

/**
 * Discriminated result for ``waitForChildExit``. The three outcomes
 * are intentionally distinct so callers can branch correctly:
 *
 *   - ``"exit"`` — child started, ran, exited. ``code`` is the
 *     non-null exit code on normal exit, or ``null`` when ``signal``
 *     is non-null (process killed by a signal — SIGTERM from a
 *     user-cancel, SIGKILL from OOM, etc.). Callers that want to
 *     show "non-zero exit" diagnostics check ``code`` here.
 *   - ``"spawn-error"`` — ``spawn()`` itself failed (no child exists).
 *     ``error.code`` carries the ENOENT / EACCES / EPERM hint.
 *     The helper has already streamed the user-facing diagnostic.
 */
type ChildExitResult =
    | { kind: "exit"; code: number | null; signal: NodeJS.Signals | null }
    | { kind: "spawn-error"; error: NodeJS.ErrnoException };

/**
 * Wait for a freshly-`spawn()`-ed Python child to exit, with the
 * `error` event wired up. Without an `error` listener, a `spawn()`
 * failure (typically ENOENT — `python` not on PATH, or
 * `frontierInsight.pythonPath` pointing at a non-existent / non-
 * executable file) means the child never starts, `close` never fires,
 * and the chat panel hangs forever on the streaming spinner. The user
 * has to restart VSCode to clear the stuck participant.
 *
 * Returns a discriminated ``ChildExitResult`` so callers can tell
 * apart a spawn failure (no process ever ran) from a signal exit
 * (process started but was killed) from a normal exit code. The
 * helper streams the user-facing diagnostic on ``spawn-error``;
 * callers handle ``exit`` themselves (code 0 = success, non-zero
 * = show stderr, signal-only = inform user of the kill).
 */
async function waitForChildExit(
    child: import("child_process").ChildProcess,
    bridge: Bridge,
    pythonPath: string,
    stream: vscode.ChatResponseStream,
): Promise<ChildExitResult> {
    const result: ChildExitResult = await new Promise((resolve) => {
        child.on("close", (code, signal) =>
            resolve({ kind: "exit", code, signal }));
        child.on("error", (e) =>
            resolve({ kind: "spawn-error", error: e as NodeJS.ErrnoException }));
    });
    await bridge.close();
    if (result.kind === "spawn-error") {
        const err = result.error;
        // Distinguish ENOENT (path doesn't exist / not on PATH) from
        // EACCES / EPERM (path exists but isn't executable — chmod +x
        // is the fix, not pointing somewhere else). Generic branch
        // covers everything else (EBADF, ENOMEM, etc.).
        let cause: string;
        if (err.code === "ENOENT") {
            cause =
                `\`${pythonPath}\` was not found. Likely causes: ` +
                "(a) `python` isn't on PATH (try setting `frontierInsight.pythonPath` to an " +
                "absolute path like `C:\\Python312\\python.exe` or `/usr/bin/python3`); " +
                "(b) the path in `frontierInsight.pythonPath` is wrong.";
        } else if (err.code === "EACCES" || err.code === "EPERM") {
            cause =
                `\`${pythonPath}\` exists but isn't executable (${err.code}). ` +
                "On macOS / Linux, run `chmod +x " + pythonPath + "` to set the " +
                "executable bit. On Windows, check that your user has execute permission " +
                "on the file (rare; usually a result of a hardened security policy).";
        } else {
            cause = `\`${pythonPath}\` failed to launch (${err.code || "(no code)"}): ${err.message}.`;
        }
        stream.markdown(
            `\n❌ **Failed to start Python.** ${cause}\n\n` +
            "Set `frontierInsight.pythonPath` in VSCode settings to a working interpreter, " +
            "then retry the command.\n",
        );
    }
    return result;
}

let persistentBridge: PersistentBridge | null = null;

/**
 * The bridge address for a Python this window starts in a terminal: this window's own
 * (a second window open at the same time listens somewhere else than the first), else
 * the per-user one.
 */
function thisWindowsBridge(): string {
    return persistentBridge?.boundPath ?? persistentBridgePath();
}

export function activate(context: vscode.ExtensionContext): void {
    const participant = vscode.chat.createChatParticipant(
        "frontier-insight.fi",
        async (request, _ctx, stream, token) => {
            await handleRequest(request, stream, token);
        },
    );
    participant.iconPath = new vscode.ThemeIcon("beaker");
    context.subscriptions.push(
        vscode.commands.registerCommand("frontierInsight.questMap", (questId?: string) =>
            openQuestMap(context, typeof questId === "string" ? questId : undefined)),
    );
    context.subscriptions.push(participant);
    context.subscriptions.push(vscode.workspace.onDidChangeConfiguration((e) => {
        if (e.affectsConfiguration("frontierInsight.pythonPath") || e.affectsConfiguration("frontierInsight.repoPath")) {
            forgetFoundFi();
        }
    }));

    // Probe the Axon sidecar on activation. We don't auto-launch from
    // the extension — VSCode users are expected to keep an Axon
    // process running in their workspace — but we surface a one-time
    // notification if it's down so the first /start of the session
    // doesn't pay the cold-init cost silently.
    // Created before the Axon probe so the probe's diagnostics land
    // somewhere the user can actually open, rather than in the extension
    // host's Developer Tools console.
    const outputChannel = vscode.window.createOutputChannel("Frontier Insight");
    context.subscriptions.push(outputChannel);

    void probeAxonOnActivate(context, outputChannel);

    // Start the session-long IPC bridge so a `python launch.py --serve`
    // (or any --tool subprocess) can route LLM calls through
    // ``vscode.lm.*`` without the user wiring a port. The bridge
    // listens on a per-user OS-managed socket / named pipe; see
    // ./bridge-path.ts for the address.
    persistentBridge = new PersistentBridge(outputChannel);
    persistentBridge.listen().catch((err) => {
        outputChannel.appendLine(`[fi] persistent bridge failed to start: ${err}`);
    });
}

export function deactivate(): void {
    if (persistentBridge) {
        void persistentBridge.close();
        persistentBridge = null;
    }
}

async function handleRequest(
    request: vscode.ChatRequest,
    stream: vscode.ChatResponseStream,
    token: vscode.CancellationToken,
): Promise<void> {
    const cmd = request.command;
    const prompt = request.prompt.trim();
    // `request.model` is the model the user picked in the VSCode
    // Chat picker. We forward this into the Bridge so every LLM call
    // from the Python engine routes through THE SAME model the user
    // sees, instead of an arbitrary `selectChatModels[0]` from the
    // full Copilot catalog (which would silently route through a
    // potentially-much-more-expensive model — Claude Opus at 5× vs
    // gpt-5.4-mini at 0.33× per premium request).
    const userPickedModel = request.model;

    if (cmd === "start") {
        await runQuest(prompt, /*fleet*/ false, stream, token, userPickedModel);
        return;
    }
    if (cmd === "fleet") {
        await runQuest(prompt, /*fleet*/ true, stream, token, userPickedModel);
        return;
    }
    if (cmd === "new" || (!cmd && !prompt)) {
        // Interview-driven quest creation. When the user types `@fi`
        // alone (no command, no extra prompt text), we assume they're
        // exploring and the easiest thing is to walk them through
        // setup rather than dump a help screen at them.
        await runInterviewAndQuest(stream, token, userPickedModel);
        return;
    }
    if (cmd === "map") {
        await vscode.commands.executeCommand("frontierInsight.questMap", prompt.trim().split(/\s+/)[0] || undefined);
        stream.markdown("Opened the quest map in its own tab.\n");
        return;
    }
    if (cmd === "resume") {
        await runResume(prompt, stream, token, userPickedModel);
        return;
    }
    if (cmd === "watch") {
        await runResume(prompt, stream, token, userPickedModel, /*watch*/ true);
        return;
    }
    if (cmd === "plan") {
        // The quest's plan.md: opens it to read and edit, or (with words after the
        // quest id) has the model rewrite it as asked. Mirrors `launch.py --revise-plan`
        // and the web quest page's Plan panel.
        await runResume(prompt, stream, token, userPickedModel, /*watch*/ false, /*plan*/ true);
        return;
    }
    if (cmd === "generate") {
        await runGenerate(prompt, stream, token, userPickedModel);
        return;
    }
    if (cmd === "summarize") {
        await runSummarize(prompt, stream, token, userPickedModel);
        return;
    }
    if (cmd === "digest") {
        await runDigest(prompt, stream, token, userPickedModel);
        return;
    }
    if (cmd === "portfolio") {
        await runPortfolio(stream, token, userPickedModel);
        return;
    }
    if (cmd === "critique") {
        await runCritique(prompt, stream, token, userPickedModel);
        return;
    }
    if (cmd === "proposal") {
        await runProposal(prompt, stream, token, userPickedModel);
        return;
    }
    if (cmd === "analyze") {
        await runAnalyze(prompt, stream, token, userPickedModel);
        return;
    }
    if (cmd === "update") {
        // Mid-quest re-entry. Runs `launch.py --update <quest_id>` here in
        // the chat, like /resume (never in a terminal): it approves the
        // settings in the quest's config.yaml and resumes. ``prompt``
        // should be the quest_id; the Python side validates it exists and
        // refuses gracefully if not.
        await runUpdate(prompt, stream, token, userPickedModel);
        return;
    }
    if (cmd === "ingest") {
        // Axon literature ingest. The user's prompt is
        // space-separated paths; quote each one before passing to
        // the shell so paths with spaces / metacharacters don't
        // break (or run unintended commands).
        const paths = parsePathsFromPrompt(prompt);
        if (paths.length === 0) {
            stream.markdown("Pass at least one path after `/ingest`. Example: `@fi /ingest ~/papers/foo.pdf`\n");
            return;
        }
        await runCommandInChat(
            ["--ingest", ...paths], stream, token, userPickedModel,
            `📚 Ingesting ${paths.length} file(s) into Axon.`,
        );
        return;
    }
    if (cmd === "drafts") {
        // List proposal-draft YAMLs in outputs/_drafts/ so the user
        // can pick one to /start without remembering paths. Mirrors
        // the CLI `--list-drafts` flag and the Web /interview
        // drafts picker. Three-interface parity.
        await runListDrafts(stream, token);
        return;
    }
    if (cmd === "skills") {
        await runListSkills(prompt, stream, token);
        return;
    }
    if (cmd === "scan-skill") {
        await runScanSkill(prompt, stream, token);
        return;
    }
    if (cmd === "approve-all-skills") {
        await runApproveAllSkills(prompt, stream, token);
        return;
    }
    if (cmd === "approve-skill") {
        await runApproveSkill(prompt, stream, token);
        return;
    }
    if (cmd === "approve-amendment") {
        // A quest stopped to ask for a change to its frozen protocol: shows the request, asks who approves,
        // and writes the approval (`launch.py --approve-amendment`). Resuming without it keeps the protocol.
        await runApproveAmendment(prompt, stream, token);
        return;
    }
    if (cmd === "accept-checks") {
        // A quest stopped because some checks do not say where their expected value comes from: go on as it is,
        // with the person's name (`launch.py --accept-checks`), then resume it.
        const questId = await runAcceptChecks(prompt, stream, token);
        if (questId && !token.isCancellationRequested) await runResume(questId, stream, token, userPickedModel);
        return;
    }
    if (cmd === "revoke-skill") {
        await runRevokeSkill(prompt, stream, token);
        return;
    }
    if (cmd === "remove-skill") {
        await runRemoveSkill(prompt, stream, token);
        return;
    }
    if (cmd === "import-skill") {
        await runImportSkill(prompt, stream, token);
        return;
    }
    if (cmd === "teach-skill") {
        await runTeachSkill(prompt, stream, token);
        return;
    }
    if (cmd === "axon-status" || cmd === "axon") {
        await runAxonStatus(stream);
        return;
    }
    if (cmd === "probe") {
        // Behavioural check of the model in the Chat picker (or, with
        // `all`, every model VS Code lists) for text FI did not send.
        // Reads no settings and needs no folder: it only talks to the
        // models through vscode.lm.
        await runProbe(prompt, userPickedModel, stream, token);
        return;
    }
    if (cmd === "trace") {
        // The quest's audit trace (core/audit_log.py), shown exactly as `launch.py --trace` prints it.
        await runTrace(prompt, stream, token);
        return;
    }
    if (cmd === "why") {
        // Why it stopped / why the review asked for a revision / why the evidence is at its level / why a step decided
        // what it did (core/why.py), exactly as `launch.py --why` prints it.
        await runWhy(prompt, stream, token);
        return;
    }
    if (cmd === "rename") {
        // Change a finished or paused quest's title (`launch.py --rename`), the same change as the web quest page.
        await runRename(prompt, stream, token);
        return;
    }
    if (cmd === "follow") {
        // Each step of a running quest as it happens (`launch.py --trace <id> --follow`).
        await runFollow(prompt, stream, token);
        return;
    }
    if (cmd === "install-tectonic" || cmd === "tectonic") {
        // No-admin LaTeX install for paper_pdf support.
        await runCommandInChat(
            ["--install-tectonic"], stream, token, userPickedModel,
            "📦 Installing tectonic (~70 MB) into tools/. No admin rights needed.",
        );
        return;
    }
    if (cmd === "help" || prompt === "help") {
        stream.markdown(helpText());
        return;
    }

    // Anything else falls through to help.
    stream.markdown(helpText());
}

function helpText(): string {
    return [
        "**Frontier Insight** — run a research quest end-to-end inside VSCode.",
        "",
        "Commands:",
        "- `@fi` or `@fi /new` — interactive setup (recommended for first-time users).",
        "- `@fi /start <path-to-config.yaml>` — run one quest from an existing YAML.",
        "- `@fi /fleet <yaml-a> <yaml-b> …` — run several in parallel.",
        "- `@fi /resume` — pick a crashed quest and pick up where it died.",
        "- `@fi /resume <quest_id>` — resume that specific quest directly. The short id (the six characters after the last dash) is enough, and the quest may be one started in another folder: FI remembers every quest it has run on this computer. When this folder has no quests, `@fi /resume` offers those.",
        "- `@fi /map <quest_id>` — open the quest map: every step in a few big blocks, what a restart from each would keep and redo, and a Restart button (also **FI: Quest map** in the command palette).",
        "- `@fi /update [<quest_id>]` — approve the settings in a quest's config.yaml after you changed them (a quest stopped for a changed setting says so), then resume it, here in the chat. A different model needs no approval: `/resume` takes it and says so.",
        "- `@fi /watch [<quest_id>]` — for a quest waiting on a background job (HPC): re-check it on a timer and resume it when the job is done.",
        "- `@fi /generate [<quest_id>] [<format>]` — produce one more output format (PDF / slides / poster / talk) for a finished quest WITHOUT re-running it. Picks quest + format if omitted.",
        "- `@fi /rename <quest_id> <new title>` — change a finished or paused quest's title: the paper's title line, its config and summary. Results, data and code are not touched; a PDF, slides or poster already made keep the old title until you `/generate` them again.",
        "- `@fi /summarize <folder>` — walk a folder of papers/code/notes/logs and produce a structured markdown summary; input files + summary land in Axon.",
        "- `@fi /digest [days]` — weekly project-manager digest: completed quests, in-progress, themes, diff vs prior digest, suggested next quests. Default window: 7 days. Lands at `<outputDir>/_digests/<YYYY-Www>.md` (where `<outputDir>` is the `frontierInsight.outputDir` setting, defaulting to `outputs/`).",
        "- `@fi /portfolio` — all-time cross-quest synthesis: topic clusters, near-duplicate detection, meta-paper candidates, coverage gaps, prioritized next-quest suggestions. Lands at `<outputDir>/_portfolio/<YYYY-MM-DD>.md`.",
        "- `@fi /critique <quest_id>` — adversarial second-pass review of a completed quest: methodology challenges, statistical issues, reproducibility gaps, alternative explanations. Lands at `<outputDir>/<quest_id>/critique.md`. For strongest effect, pick a Copilot model different from the one that wrote the paper.",
        "- `@fi /proposal <topic>` — pre-quest planning doc: background, hypothesis, plan, success criteria, risks, recommended next step. Writes both a markdown proposal and a companion YAML ready for `/start`. Lands at `<outputDir>/_drafts/<id>-proposal.md` + `<outputDir>/_drafts/<id>.yaml`.",
        "- `@fi /drafts` — list proposal drafts you've made (most-recent first) with a one-click `/start` command for each. Mirrors `python launch.py --list-drafts` and the web `/interview` drafts picker.",
        "- `@fi /skills [--config <quest.yaml>]` — list the skill library with each entry's promotion status, scan findings, and domain tags. Mirrors `python launch.py --skills`; `--config` also looks in the skill folders that quest names (`engine.skills_dirs`).",
        "- `@fi /scan-skill <name> [--config <quest.yaml>]` — statically review a skill before approving it: injection phrasing, hidden characters, network access, `eval`. Nothing is imported or run.",
        "- `@fi /approve-skill <name> [--config <quest.yaml>]` — approve a skill for use. Shows the review first, then asks who is approving; a high-severity finding needs an extra confirmation. Approval binds to that exact content.",
        "- `@fi /approve-amendment <quest_id>` — approve the change to a quest's frozen protocol that it stopped to ask about. Shows what changes and why, asks who is approving, and records it; resuming the quest without approving keeps the frozen protocol. If the results had already been seen, the run is archived and the paper says the change was post-hoc.",
        "- `@fi /accept-checks <quest_id>` — go on as it is when a quest stopped because some of its checks do not say where their expected value comes from. Shows the checks, asks your name, records the choice and resumes. The checks still run; each is marked *source not confirmed*, and the result and the paper say so. To fill them in instead: `@fi /plan <quest_id> fill in where each check's expected value comes from`.",
        "- `@fi /approve-all-skills` — approve every skill that passes its gates at once, optionally pip-installing what quarantined skills are missing first. Still asks who is approving; a failing self-test is still refused.",
        "- `@fi /revoke-skill <name> [--config <quest.yaml>]` — withdraw approval, returning the skill to proposed.",
        "- `@fi /remove-skill <name> [--config <quest.yaml>]` — remove a skill from FI: deleted if it is in FI's own folder, hidden if another tool installed it.",
        "- `@fi /import-skill [path]` — import a skill written for another agent (Agent Skills layout). Opens a picker with no path, then asks for optional domain tags.",
        "- `@fi /teach-skill <name> <module>` — draft a skill from an installed library, reading its real signatures by introspection.",
        "- `@fi /axon-status` — check whether the Axon sidecar (`python -m axon.api`) is reachable. Its port is discovered automatically; override it with the `frontierInsight.axonUrl` setting. CLI / web launches auto-start it; VSCode users keep their own. Use this to confirm the sidecar is hot before kicking off a quest.",
        "- `@fi /probe [all]` — ask the model selected in the Chat picker (or, with `all`, every model VS Code lists, after a confirmation that states the request count) whether text FI did not send appears to be in its context: a hidden system prompt, tool definitions, a persona. Two small requests per model. It is behavioural evidence only: `vscode.lm` reports no usage figures, so real overhead is not measured. `@fi /probe image` sends the selected model one small test image, as FI sends the figures it reads, and says whether it read it.",
        "- `@fi /analyze <data-path> <topic>` — run a no-simulation quest on pre-staged data. Files under `<data-path>` are copied into the new quest's `data/` directory and the engine routes through `auto_collect_data → wait_for_data → data_load → analyze → write → review`. The inverse of `/proposal`: when you already have the dataset and just want a paper analyzing it.",
        "- `@fi /trace <quest_id> [--node <name>] [--detail summary|checks|debug]` — what the quest did, why, and a check that its record was not edited. Mirrors `python launch.py --trace` and the web quest page's Trace panel.",
        "",
        "All LLM calls go through your Copilot subscription via the",
        "`vscode.lm` Language Model API. Each quest's `provider.node_models`",
        "is honored, so different nodes (and different reviewer-panel",
        "personas) can use different Copilot models within one run.",
    ].join("\n");
}


/**
 * Implementation of `@fi /resume`. Two modes:
 *
 * 1. `@fi /resume` (no args) — scan `<repoPath>/outputs/` for quest
 *    dirs that have a `.fi/state.sqlite` (i.e., at least one node
 *    completed and was checkpointed) and show a QuickPick. The most
 *    recently-modified quest sits at the top.
 *
 * 2. `@fi /resume <quest_id>` — resume that specific quest.
 *
 * For each resume we auto-discover the YAML by title-slug match
 * against `outputs/_drafts/`. The interview writer names YAMLs as
 * `<timestamp>-<slug>.yaml` where the slug also appears in the
 * quest_id (`<unix>-<slug>-<nonce>`). If no YAML matches, we fall
 * back to a file picker.
 *
 * The actual graph state lives in the per-quest `state.sqlite`; the
 * YAML only contributes provider/execution/output settings — so a
 * slug-match miss isn't fatal, the user can pick any YAML with a
 * compatible provider block.
 */
async function runResume(
    promptArgs: string,
    stream: vscode.ChatResponseStream,
    token: vscode.CancellationToken,
    userPickedModel: vscode.LanguageModelChat,
    watch = false,
    plan = false,
): Promise<void> {
    if (token.isCancellationRequested) return;

    // `/resume <quest_id> --revise-plan "<what to change>"` is the CLI's spelling of `/plan <quest_id> <what to change>`.
    // It used to be read as a plain resume (only the first word, the id, was kept), which ran the quest into the same
    // stop again with the plan unchanged: it now rewrites the plan as asked, and runs nothing else, as the CLI does.
    const revise = !plan && !watch ? splitRevisePlan(promptArgs) : null;
    if (revise) {
        if (!revise.questId || !revise.request) {
            stream.markdown("Name the quest and say what to change after `--revise-plan`, for example: " +
                "`@fi /resume <quest_id> --revise-plan \"fill in where each check's expected value comes from\"`.\n");
            return;
        }
        await runResume(`${revise.questId} ${revise.request}`.trim(), stream, token, userPickedModel, false, true);
        return;
    }
    // Any other flag is said, not dropped: a resume that silently ignored what was asked ran into the same stop again.
    const unknown = !plan && !watch ? unknownResumeFlags(promptArgs) : [];
    if (unknown.length) {
        stream.markdown(
            `❌ \`/resume\` does not understand ${unknown.map((f) => `\`${f}\``).join(", ")}, so nothing was run. ` +
            "It takes a quest id, `--from <step>` (redo from a step) and `--revise-plan \"<what to change>\"` " +
            "(rewrite the plan). To go on with checks that do not say where their expected value comes from: " +
            "`@fi /accept-checks <quest_id>`.\n",
        );
        return;
    }

    const cfg = vscode.workspace.getConfiguration("frontierInsight");
    const roots = await rootsForCommand();
    if ("error" in roots) {
        stream.markdown(roots.error);
        return;
    }
    const { repoPath, workDir } = roots;

    // Resolve the outputs dir from settings. The `frontierInsight.outputDir`
    // setting may be a relative path (joined with repoPath) or absolute.
    // Defaults to "outputs". This must match where quests actually land —
    // otherwise the picker shows nothing for users who customized it.
    const outputDirSetting = cfg.get<string>("outputDir") || "outputs";
    const outputsDir = path.isAbsolute(outputDirSetting)
        ? outputDirSetting
        : path.join(workDir, outputDirSetting);
    const outputsExist = await fsExists(outputsDir);

    // Find all quest dirs with a checkpoint. Async I/O so the extension
    // host event loop stays responsive on slow filesystems / large dirs.
    type Candidate = { questId: string; questDir: string; mtimeMs: number };
    const entries = outputsExist ? await fsPromises.readdir(outputsDir, { withFileTypes: true }) : [];
    const candidates: Candidate[] = [];
    await Promise.all(entries.map(async (entry) => {
        if (!entry.isDirectory() || entry.name.startsWith("_")) return;
        const questDir = path.join(outputsDir, entry.name);
        const checkpoint = path.join(questDir, ".fi", "state.sqlite");
        try {
            const stat = await fsPromises.stat(checkpoint);
            if (!stat.isFile()) return;
            // /watch is for quests waiting on a background job only.
            if (watch) await fsPromises.stat(path.join(questDir, ".fi", "pending.json"));
            // /plan is for quests that have written their plan (plan.md).
            if (plan) await fsPromises.stat(path.join(questDir, "plan.md"));
            candidates.push({
                questId: entry.name, questDir, mtimeMs: stat.mtimeMs,
            });
        } catch {
            // Missing checkpoint or unreadable file — skip silently.
        }
    }));
    // Every quest FI has run on this computer, in any folder (core/quest_index.py): offered when this folder has none,
    // and looked up when an id is not one of this folder's. Those this /watch or /plan cannot act on are left out.
    const here = path.resolve(outputsDir).toLowerCase();
    const elsewhere = loadIndex().filter((e) => {
        if (path.resolve(path.dirname(e.questRoot)).toLowerCase() === here) return false;
        if (watch && !fs.existsSync(path.join(e.questRoot, ".fi", "pending.json"))) return false;
        if (plan && !fs.existsSync(path.join(e.questRoot, "plan.md"))) return false;
        return true;
    });
    if (candidates.length === 0 && elsewhere.length === 0) {
        if (!outputsExist) {
            stream.markdown(
                `❌ No outputs directory at \`${outputsDir}\` — nothing to resume. ` +
                `(Override via the \`frontierInsight.outputDir\` setting.)`,
            );
            return;
        }
        stream.markdown(
            plan
                ? `❌ No quest under \`${outputsDir}\` has a plan yet (none has \`plan.md\`). The plan is written after the literature and before the experiment is designed.`
                : watch
                ? `❌ No quest under \`${outputsDir}\` is waiting on a background job (none has \`.fi/pending.json\`).`
                : `❌ No quests with a \`.fi/state.sqlite\` checkpoint under \`${outputsDir}\`. Run \`@fi /new\` to start one.`,
        );
        return;
    }
    candidates.sort((a, b) => b.mtimeMs - a.mtimeMs);

    // Handle multi-token / typo-quoted args. The chat surface can pass
    // through copy/paste artifacts like `/resume "1778…-x" extra` —
    // pick just the first whitespace-separated token so the lookup is
    // deterministic instead of silently failing with a confusing
    // "no quest with id '"178…-x" extra'" message.
    // /resume <quest_id> --from with no step: the steps that quest reached, which are the ones it can be run again from
    // (the newest quest when no id is given; the list names the quest it is for).
    // A `--from` followed by nothing or by another flag is a bare one.
    const bareFrom = /(?:^|\s)--from(?=\s*$|\s+-)/;
    if (!plan && !watch && bareFrom.test(promptArgs)) {
        const named = (promptArgs.replace(bareFrom, " ").trim().split(/\s+/)[0] || "").replace(/^["']+|["']+$/g, "");
        const quest = named && !named.startsWith("-") ? named : candidates[0]?.questId;
        if (!quest) {
            // This folder has no quests: name one rather than have one from another folder picked for you.
            stream.markdown(
                "Which quest? This folder has none; FI has run these elsewhere: " +
                elsewhere.slice(0, 5).map((e) => `\`${shortId(e.questId)}\` ${e.title || e.questId}`).join(", ") +
                ". Then: `@fi /resume <short id> --from`.\n",
            );
            return;
        }
        await runRerunSteps(quest, stream, token);
        return;
    }
    // /resume <quest_id> --from <step> (or --from=<step>): the step is taken out first, so a quest id is never read
    // from it and `/resume --from code` still offers the picker.
    const fromMatch = !plan && !watch ? /(?:^|\s)--from(?:\s+|=)(\S+)/.exec(promptArgs) : null;
    const fromStep = fromMatch ? fromMatch[1].replace(/^["']+|["']+$/g, "").toLowerCase() : undefined;
    const rawArg = (fromMatch ? promptArgs.replace(fromMatch[0], " ") : promptArgs).trim();
    const firstToken = rawArg.split(/\s+/)[0] || "";
    // Also strip surrounding quotes a user might paste from a log line.
    const sanitized = firstToken.replace(/^["']+|["']+$/g, "");
    // /plan <quest_id> <what to change>: the words after the id are the request.
    const planRequest = plan ? rawArg.slice(firstToken.length).trim() : "";
    if (fromStep && RERUN_STEPS_NEEDING_NAME.includes(fromStep)) {
        stream.markdown(
            `❌ Running again from \`${fromStep}\` replaces the plan and the experiment plan (frozen), so it needs ` +
            "your name. In a terminal, run: `python launch.py --config <quest>/config.yaml --resume <quest_id> " +
            `--from ${fromStep} --approve-as <your name>\`.
`,
        );
        return;
    }
    if (fromStep && !RERUN_STEPS.includes(fromStep)) {
        stream.markdown(
            `❌ \`${fromStep}\` is not a step a quest can be run again from. Choose one of: ` +
            RERUN_STEPS.map((step) => `\`${step}\``).join(", ") +
            "; `@fi /resume <quest_id> --from` with no step lists the ones that quest reached. " +
            "To change the plan, use `@fi /plan <quest_id> <what to change>`.\n",
        );
        return;
    }
    let chosenId = sanitized;
    // Set when the quest is not under this folder's outputs: found among every quest FI has run on this computer.
    let foreign: IndexEntry | undefined;
    if (!chosenId) {
        // This folder's quests; when it has none, every quest FI has run on this computer.
        const picks = candidates.length > 0
            ? candidates.map((c) => ({
                label: `$(beaker) ${c.questId}`,
                description: `${shortId(c.questId)} · ${new Date(c.mtimeMs).toLocaleString()}`,
                questId: c.questId, entry: undefined as IndexEntry | undefined,
            }))
            : elsewhere.map((e) => ({
                label: `$(beaker) ${shortId(e.questId)}  ${e.title || e.questId}`,
                description: e.questRoot,
                questId: e.questId, entry: e as IndexEntry | undefined,
            }));
        const picked = await vscode.window.showQuickPick(picks, {
            placeHolder: candidates.length === 0
                ? "This folder has no quests. Pick one FI has run in another folder (most recent first)"
                : plan
                ? "Pick a quest whose plan to open (most recent first)"
                : "Pick a quest to resume (most recent first)",
            matchOnDescription: true,
        });
        if (!picked) return;   // user hit Esc
        chosenId = picked.questId;
        foreign = picked.entry;
    } else if (!candidates.find((c) => c.questId === chosenId)) {
        // Not one of this folder's quests by its full id: a unique start or end of one of their ids (e.g. the six
        // characters after the last dash), else a quest FI has run in another folder.
        const local = matchIds(chosenId, candidates.map((c) => c.questId));
        const inIndex = local.length === 0 ? findInIndex(chosenId, elsewhere) : { kind: "none" as const };
        if (local.length === 1) {
            chosenId = local[0];
        } else if (local.length > 1) {
            stream.markdown("❌ " + describeAmbiguous(chosenId, local.map((q) => ({
                questId: q, title: "", questRoot: path.join(outputsDir, q),
            }))));
            return;
        } else if (inIndex.kind === "found") {
            foreign = inIndex.entry;
            chosenId = inIndex.entry.questId;
        } else if (inIndex.kind === "ambiguous") {
            stream.markdown("❌ " + describeAmbiguous(chosenId, inIndex.entries));
            return;
        } else {
            stream.markdown(
                `❌ No quest \`${chosenId}\` under \`${outputsDir}\` (with a \`.fi/state.sqlite\` checkpoint), ` +
                "nor among the quests FI has run in other folders on this computer. " +
                "`python launch.py tools quests` lists every one with its short id.",
            );
            return;
        }
    }
    // The quest's own folder: under this folder's outputs, or where it was run.
    const questDir = foreign ? foreign.questRoot : path.join(outputsDir, chosenId);
    if (foreign) {
        stream.markdown(`📁 \`${chosenId}\` is in \`${path.dirname(questDir)}\`; it runs there.\n\n`);
    }

    // YAML discovery has three tiers:
    //   1. `<quest_dir>/config.yaml` — the canonical location. launch.py
    //      copies the source YAML here at quest startup so resume is
    //      a one-step lookup. This is the new default path.
    //   2. `<outputs>/_drafts/<ts>-<slug>.yaml` — legacy fallback for
    //      quests that ran before the config-copy feature shipped. The
    //      quest_id shape is `<unix>-<slug>-<6hex>` so we strip the
    //      leading timestamp and trailing nonce and anchor on
    //      `-${slug}.yaml` to avoid substring collisions
    //      (e.g. "cat" matching "caterpillar...").
    //   3. Manual file picker — only used when neither (1) nor (2) hits.
    let yamlPath: string | undefined;
    const inQuestYaml = path.join(questDir, "config.yaml");
    if (await fsExists(inQuestYaml)) {
        yamlPath = inQuestYaml;
    }
    if (!yamlPath && foreign) {
        stream.markdown(`❌ Quest \`${chosenId}\` has no \`config.yaml\` in \`${questDir}\`, so it cannot be resumed from here.`);
        return;
    }
    if (!yamlPath) {
        const slug = chosenId.replace(/^\d+-/, "").replace(/-[0-9a-f]{6}$/i, "");
        const draftsDir = path.join(outputsDir, "_drafts");
        if (await fsExists(draftsDir)) {
            const exactSuffix = `-${slug}.yaml`;
            const draftNames = await fsPromises.readdir(draftsDir);
            const matched = await Promise.all(
                draftNames
                    .filter((f) => f.endsWith(exactSuffix))
                    .map(async (f) => {
                        const fp = path.join(draftsDir, f);
                        const st = await fsPromises.stat(fp);
                        return { f, mtime: st.mtimeMs };
                    }),
            );
            matched.sort((a, b) => b.mtime - a.mtime);
            if (matched.length > 0) {
                yamlPath = path.join(draftsDir, matched[0].f);
            }
        }
    }
    if (!yamlPath) {
        // Fall back to a file picker.
        const picked = await vscode.window.showOpenDialog({
            canSelectFiles: true, canSelectFolders: false, canSelectMany: false,
            filters: { YAML: ["yaml", "yml"] },
            defaultUri: vscode.Uri.file(outputsDir),
            openLabel: `Pick a YAML for ${chosenId}`,
            title: `No config.yaml in quest dir and no draft match. Pick one manually.`,
        });
        if (!picked || picked.length === 0) return;
        yamlPath = picked[0].fsPath;
    }

    // A quest in another folder runs there: from the folder it was started in (its relative paths), writing to its own
    // outputs folder, with its own config.yaml named by its full path.
    const where: RunWhere | undefined = foreign
        ? {
            configPath: yamlPath,
            cwd: foreign.workingFolder && fs.existsSync(foreign.workingFolder) ? foreign.workingFolder : workDir,
            outputDir: path.dirname(questDir),
        }
        : undefined;
    const relYaml = foreign ? yamlPath : path.relative(workDir, yamlPath).split(path.sep).join("/");
    if (plan) {
        const planPath = path.join(questDir, "plan.md");
        if (!planRequest) {
            // No request: open the file. Editing it is editing the design that will run.
            const shownPlan = foreign ? planPath : path.relative(workDir, planPath).split(path.sep).join("/");
            stream.markdown(
                `📋 The plan of \`${chosenId}\` is \`${shownPlan}\`. ` +
                `I opened it in the editor, next to your other tabs.\n\n` +
                `- **Edit it** and save: the block under *The design (used as written)* is what runs, exactly.\n` +
                `- **Or ask for a change**: \`@fi /plan ${chosenId} <what to change>\` rewrites it.\n` +
                `- **When it says what you want**: \`@fi /resume ${chosenId}\`.\n`,
            );
            try {
                const doc = await vscode.workspace.openTextDocument(vscode.Uri.file(planPath));
                await vscode.window.showTextDocument(doc, { preview: false, viewColumn: tabAreaColumn(doc.uri) });
            } catch {
                stream.markdown(`(Could not open it in the editor; open \`${planPath}\` yourself.)\n`);
            }
            return;
        }
        stream.markdown(
            `✏️ Rewriting the plan of \`${chosenId}\` as you asked: “${planRequest}”\n\n` +
            `📝 Using config: \`${relYaml}\`\n\n` +
            `🤖 Model: \`${userPickedModel.family}\` (vendor: ${userPickedModel.vendor})\n\n`,
        );
        await runQuest(
            relYaml, /*fleet*/ false, stream, token, userPickedModel,
            /*resumeQuestId*/ chosenId, /*watch*/ false, /*revisePlan*/ planRequest,
            /*fromStep*/ undefined, roots, where,
        );
        return;
    }
    stream.markdown(
        watch
            ? `👁 Watching quest \`${chosenId}\`: its experiment script is re-run on a timer and the quest resumes when the job is done.\n\n` +
              `📝 Using config: \`${relYaml}\`\n\n`
            : fromStep
            ? `🔁 Running quest \`${chosenId}\` again from the ${fromStep} step: what that step and the later ones made is first moved to \`.fi/previous/\`.\n\n` +
              `📝 Using config: \`${relYaml}\`\n\n`
            : `🔁 Resuming quest \`${chosenId}\`\n\n` +
              `📝 Using config: \`${relYaml}\`\n\n` +
              `🤖 Model: \`${userPickedModel.family}\` (vendor: ${userPickedModel.vendor})\n\n` +
              `▶️ Re-entering the LangGraph from the last checkpointed node…\n\n`,
    );
    await runQuest(
        relYaml,
        /*fleet*/ false,
        stream,
        token,
        userPickedModel,
        /*resumeQuestId*/ chosenId,
        /*watch*/ watch,
        /*revisePlan*/ undefined,
        /*fromStep*/ fromStep,
        roots,
        where,
    );
}

/** Where a quest from another folder runs (see runResume): its config by full path, its working folder, its outputs. */
interface RunWhere {
    configPath: string;
    cwd: string;
    outputDir: string;
}

// The steps `/resume <quest_id> --from <step>` accepts (core/rerun_from.py STEPS).
const RERUN_STEPS = ["skills", "code", "run", "figures", "analysis", "crosscheck", "evidence", "writing", "claims", "review"];
// These replace the plan and need `--approve-as <name>`, which is given on the command line.
const RERUN_STEPS_NEEDING_NAME = ["ideas", "literature", "plan", "design"];

function parsePathsFromPrompt(prompt: string): string[] {
    // Split on whitespace OUTSIDE of double-quoted spans so users
    // can pass paths with spaces by wrapping them in quotes:
    //   /ingest "C:\My Papers\a.pdf" /home/me/b.pdf
    const out: string[] = [];
    const trimmed = (prompt || "").trim();
    if (!trimmed) return out;
    const re = /"([^"]+)"|(\S+)/g;
    let m;
    while ((m = re.exec(trimmed)) !== null) {
        const v = m[1] !== undefined ? m[1] : m[2];
        if (v) out.push(v);
    }
    return out;
}

/**
 * How to spell a command line for the shell VSCode will start in a
 * new terminal. PowerShell parses a line beginning with a quoted
 * string as an expression and needs the call operator; cmd.exe
 * rejects a leading `&`. See ./terminal-command.
 */
function currentShell(): ShellKind {
    return detectShell(vscode.env.shell, process.platform);
}

async function runCommandInChat(
    args: string[],
    stream: vscode.ChatResponseStream,
    token: vscode.CancellationToken,
    userPickedModel: vscode.LanguageModelChat,
    hint: string,
): Promise<void> {
    // /ingest and /install-tectonic: a launch.py command whose whole
    // output is the answer. It runs here, with every line it prints shown
    // in the chat as it comes, as every other chat command runs; no
    // terminal is opened.
    const cfg = vscode.workspace.getConfiguration("frontierInsight");
    const pythonPath = cfg.get<string>("pythonPath") || "python";
    const roots = await rootsForCommand();
    if ("error" in roots) {
        stream.markdown(roots.error);
        return;
    }
    const { repoPath, workDir } = roots;
    stream.markdown(`${hint}\n\n`);
    const ran = await runLaunchInChat(args, stream, token, userPickedModel, {
        pythonPath, launchScript: path.join(repoPath, "launch.py"), cwd: workDir, showAllOutput: true,
    });
    if (!ran) return;
    if (ran.code === 0) {
        stream.markdown("\n✅ Done.\n");
    } else {
        const tail = ran.stderrTail.join("\n");
        stream.markdown(
            `\n❌ **Python exited with code ${ran.code}.**\n\n` +
            (tail.trim() ? "```\n" + tail + "\n```\n" : "stderr was empty.\n"),
        );
    }
}


async function runUpdate(
    promptArgs: string,
    stream: vscode.ChatResponseStream,
    token: vscode.CancellationToken,
    userPickedModel: vscode.LanguageModelChat,
): Promise<void> {
    if (token.isCancellationRequested) return;

    const cfg = vscode.workspace.getConfiguration("frontierInsight");
    const pythonPath = cfg.get<string>("pythonPath") || "python";
    const roots = await rootsForCommand();
    if ("error" in roots) {
        stream.markdown(roots.error);
        return;
    }
    const { repoPath, workDir } = roots;

    const outputDirSetting = cfg.get<string>("outputDir") || "outputs";
    const outputsDir = path.isAbsolute(outputDirSetting)
        ? outputDirSetting
        : path.join(workDir, outputDirSetting);

    let questId = (promptArgs.trim().split(/\s+/)[0] || "").replace(/^["']+|["']+$/g, "");
    if (!questId) {
        if (!(await fsExists(outputsDir))) {
            stream.markdown(
                `❌ No outputs directory at \`${outputsDir}\` — nothing to update. Pass a quest_id explicitly: \`@fi /update <quest_id>\`.`,
            );
            return;
        }
        type Candidate = { questId: string; mtimeMs: number };
        const entries = await fsPromises.readdir(outputsDir, { withFileTypes: true });
        const candidates: Candidate[] = [];
        await Promise.all(entries.map(async (entry) => {
            if (!entry.isDirectory() || entry.name.startsWith("_")) return;
            const configYaml = path.join(outputsDir, entry.name, "config.yaml");
            try {
                const stat = await fsPromises.stat(configYaml);
                if (!stat.isFile()) return;
                candidates.push({ questId: entry.name, mtimeMs: stat.mtimeMs });
            } catch {
                // Quest has no config.yaml — can't be updated. Skip.
            }
        }));
        if (candidates.length === 0) {
            stream.markdown(
                `❌ No quests with a \`config.yaml\` under \`${outputsDir}\`. ` +
                `Only quests started after the \`--new\` interview support \`--update\`.`,
            );
            return;
        }
        candidates.sort((a, b) => b.mtimeMs - a.mtimeMs);
        const picked = await vscode.window.showQuickPick(
            candidates.map((c) => ({
                label: `$(beaker) ${c.questId}`,
                description: new Date(c.mtimeMs).toLocaleString(),
                questId: c.questId,
            })),
            {
                placeHolder: "Pick a quest to update (most recent config.yaml first)",
                matchOnDescription: true,
            },
        );
        if (!picked) return;
        questId = picked.questId;
    }

    // The id as typed may be a short id (the six characters after the last dash) or a quest FI ran in another folder,
    // which `launch.py --update` finds too: resolved here the same way, so the card shown, the config.yaml opened and
    // the end of the run reported are that quest's.
    let questDir = path.join(outputsDir, questId);
    let questsDir = outputsDir;
    if (!(await fsExists(path.join(questDir, "config.yaml")))) {
        let local: string[] = [];
        try {
            local = (await fsPromises.readdir(outputsDir, { withFileTypes: true }))
                .filter((e) => e.isDirectory() && !e.name.startsWith("_")).map((e) => e.name);
        } catch { /* no outputs folder here */ }
        const matched = matchIds(questId, local);
        const here = path.resolve(outputsDir).toLowerCase();
        const elsewhere = loadIndex().filter((e) => path.resolve(path.dirname(e.questRoot)).toLowerCase() !== here);
        const found = matched.length === 0 ? findInIndex(questId, elsewhere) : { kind: "none" as const };
        if (matched.length === 1) {
            questId = matched[0];
            questDir = path.join(outputsDir, questId);
        } else if (matched.length > 1) {
            stream.markdown("❌ " + describeAmbiguous(questId, matched.map((q) => ({
                questId: q, title: "", questRoot: path.join(outputsDir, q),
            }))));
            return;
        } else if (found.kind === "found") {
            questId = found.entry.questId;
            questDir = found.entry.questRoot;
            questsDir = path.dirname(questDir);
            stream.markdown(`📁 \`${questId}\` is in \`${questsDir}\`.\n\n`);
        } else if (found.kind === "ambiguous") {
            stream.markdown("❌ " + describeAmbiguous(questId, found.entries));
            return;
        } else {
            stream.markdown(
                `❌ No quest \`${questId}\` with a \`config.yaml\` under \`${outputsDir}\`, nor among the quests FI has ` +
                "run in other folders on this computer. `python launch.py tools quests` lists every one with its short id.",
            );
            return;
        }
    }

    // `/update` runs here in the chat, like `/resume`: never in a terminal. The settings it approves are the ones in
    // the quest's config.yaml as it is now; the person sees what stopped the quest, then says whether to approve them
    // as they are, or opens config.yaml to change it first.
    const configPath = path.join(questDir, "config.yaml");
    try {
        const card = await fsPromises.readFile(path.join(questDir, "NEXT_STEP.md"), "utf-8");
        stream.markdown(`**Why \`${questId}\` is waiting:**\n\n${card}\n\n---\n\n`);
    } catch { /* not stopped for anything: the update approves config.yaml as it is */ }
    const choice = await vscode.window.showQuickPick(
        [
            { label: "$(check) Approve the settings in config.yaml as they are, and resume", value: "approve" },
            { label: "$(edit) Open config.yaml to change it first", value: "edit" },
            { label: "$(close) Cancel", value: "cancel" },
        ],
        { placeHolder: `Update ${questId}: its settings are the ones in its config.yaml` },
    );
    if (!choice || choice.value === "cancel") {
        stream.markdown("Nothing was changed.\n");
        return;
    }
    // The setup questions one by one need a terminal to answer in; FI does not open one. This is the line to run in
    // one yourself (it reaches this window's models through its session-long bridge).
    const byQuestion = updateTerminalCommand({
        pythonPath,
        questId,
        // The quest was found under `questsDir` (the resolved
        // `frontierInsight.outputDir`, or the folder another quest is in),
        // so --update has to be told to look there too; launch.py would
        // otherwise default to ./outputs and reject a quest just listed.
        outputRoot: questsDir,
        bridgeSocket: thisWindowsBridge(),
        shell: currentShell(),
    });
    if (choice.value === "edit") {
        stream.markdown(
            `📝 Opened \`${configPath}\`. Change it and save, then \`@fi /update ${questId}\` again to approve it and ` +
            `resume. (A different model needs no approval: \`@fi /resume ${questId}\` takes it and says so.)\n\n` +
            `To answer the setup questions one by one instead, run this in a terminal yourself:\n\n` +
            "```\n" + byQuestion + "\n```\n",
        );
        try {
            const doc = await vscode.workspace.openTextDocument(vscode.Uri.file(configPath));
            await vscode.window.showTextDocument(doc, { preview: false, viewColumn: tabAreaColumn(doc.uri) });
        } catch {
            stream.markdown(`(Could not open it in the editor; open \`${configPath}\` yourself.)\n`);
        }
        return;
    }
    stream.markdown(`🔧 Approving the settings in \`${configPath}\` and resuming \`${questId}\`\n\n`);
    const startedAt = Date.now();
    // FI_UPDATE_APPROVE_AS_IS: the person chose to approve config.yaml as it is, so launch.py asks no question
    // (without it, --update asks the setup questions, and with no terminal to answer in it approves nothing).
    const ran = await runLaunchInChat(
        ["--update", questId, "--output-root", questsDir], stream, token, userPickedModel,
        {
            pythonPath, launchScript: path.join(repoPath, "launch.py"), cwd: workDir,
            env: { FI_UPDATE_APPROVE_AS_IS: "1" },
        },
    );
    if (!ran) return;
    await reportQuestEnd(ran, stream, { outputsDir: questsDir, questId, startedAt });
}


/**
 * `@fi /generate [<quest_id>] [<kind>]` — produce ONE more output format
 * (paper_pdf / slides / poster / speech) for an ALREADY-finished quest
 * without re-running the research. Mirrors `python launch.py --emit` and
 * the web Outputs panel. Quest + kind are quick-picked when omitted; the
 * generation runs here in the chat, its model calls through this chat's
 * bridge, as `/resume`'s do.
 */
async function runGenerate(
    promptArgs: string,
    stream: vscode.ChatResponseStream,
    token: vscode.CancellationToken,
    userPickedModel: vscode.LanguageModelChat,
): Promise<void> {
    if (token.isCancellationRequested) return;
    const cfg = vscode.workspace.getConfiguration("frontierInsight");
    const pythonPath = cfg.get<string>("pythonPath") || "python";
    const roots = await rootsForCommand();
    if ("error" in roots) {
        stream.markdown(roots.error);
        return;
    }
    const { repoPath, workDir } = roots;
    const outputDirSetting = cfg.get<string>("outputDir") || "outputs";
    const outputsDir = path.isAbsolute(outputDirSetting)
        ? outputDirSetting
        : path.join(workDir, outputDirSetting);

    const KINDS = ["paper_pdf", "slides", "poster", "speech"];
    const tokens = promptArgs.trim().split(/\s+/).filter(Boolean);
    let questId = (tokens[0] || "").replace(/^["']+|["']+$/g, "");
    let kind = tokens.find((t) => KINDS.includes(t)) || "";
    if (kind) questId = questId === kind ? "" : questId;

    // Pick the quest if not supplied: only quests with a config.yaml AND a
    // paper/paper.md (something to render from).
    if (!questId) {
        if (!(await fsExists(outputsDir))) {
            stream.markdown(
                `❌ No outputs directory at \`${outputsDir}\` — nothing to generate from.`,
            );
            return;
        }
        const entries = await fsPromises.readdir(outputsDir, { withFileTypes: true });
        const candidates: { questId: string; mtimeMs: number }[] = [];
        await Promise.all(entries.map(async (entry) => {
            if (!entry.isDirectory() || entry.name.startsWith("_")) return;
            const yaml = path.join(outputsDir, entry.name, "config.yaml");
            const paper = path.join(outputsDir, entry.name, "paper", "paper.md");
            try {
                if (!(await fsPromises.stat(yaml)).isFile()) return;
                if (!(await fsPromises.stat(paper)).isFile()) return;
                const st = await fsPromises.stat(paper);
                candidates.push({ questId: entry.name, mtimeMs: st.mtimeMs });
            } catch {
                // No config / no paper — can't emit. Skip.
            }
        }));
        if (candidates.length === 0) {
            stream.markdown(
                `❌ No finished quests with a \`config.yaml\` + \`paper/paper.md\` under \`${outputsDir}\`.`,
            );
            return;
        }
        candidates.sort((a, b) => b.mtimeMs - a.mtimeMs);
        const picked = await vscode.window.showQuickPick(
            candidates.map((c) => ({
                label: `$(file-pdf) ${c.questId}`,
                description: new Date(c.mtimeMs).toLocaleString(),
                questId: c.questId,
            })),
            { placeHolder: "Pick a finished quest to generate an output for", matchOnDescription: true },
        );
        if (!picked) return;
        questId = picked.questId;
    }
    if (!kind) {
        const pickedKind = await vscode.window.showQuickPick(
            [
                { label: "$(file-pdf) PDF", value: "paper_pdf" },
                { label: "$(device-camera-video) Slides", value: "slides" },
                { label: "$(layout) Poster", value: "poster" },
                { label: "$(comment-discussion) Talk script", value: "speech" },
            ],
            { placeHolder: "Which output format to generate?" },
        );
        if (!pickedKind) return;
        kind = pickedKind.value;
    }

    const yamlPath = path.join(outputsDir, questId, "config.yaml");
    stream.markdown(`📄 Generating \`${kind}\` for \`${questId}\` from its existing paper — no re-run.\n\n`);
    // slides / poster / speech each make a model call: they come back
    // through this command's bridge, as a /resume's do. paper_pdf renders
    // without one; the bridge client connects lazily on the first call, so
    // the port is inert for that kind.
    const ran = await runLaunchInChat(
        ["--config", yamlPath, "--resume", questId, "--emit", kind], stream, token, userPickedModel,
        { pythonPath, launchScript: path.join(repoPath, "launch.py"), cwd: workDir },
    );
    if (!ran) return;
    if (ran.code === 0) {
        stream.markdown(`\n✅ \`${kind}\` generated for \`${questId}\`.\n`);
    } else {
        const tail = ran.stderrTail.join("\n");
        stream.markdown(
            `\n❌ **Python exited with code ${ran.code}.**\n\n` +
            (tail.trim() ? "```\n" + tail + "\n```\n" : "stderr was empty.\n"),
        );
    }
}


async function runInterviewAndQuest(
    stream: vscode.ChatResponseStream,
    token: vscode.CancellationToken,
    userPickedModel: vscode.LanguageModelChat,
): Promise<void> {
    if (token.isCancellationRequested) return;

    // The picked model is left out of the second reviewer's list (research needs a reviewer on another model).
    const answers = await runInterview(stream, userPickedModel);
    if (!answers) return;  // user hit Esc somewhere

    // Snapshot the active Copilot model into provider.model so the
    // quest stays on a consistent LLM even if the user changes
    // Copilot model later. The interview produces ``provider_model: ""``
    // as a placeholder; we overwrite here. ``family`` (e.g.
    // ``claude-opus-4-7`` / ``gpt-4o``) is the stable identifier
    // the bridge keys on.
    answers.provider_model = userPickedModel.family || "";

    // Resolve repo path the same way runQuest does.
    const cfg = vscode.workspace.getConfiguration("frontierInsight");
    const roots = await rootsForCommand();
    if ("error" in roots) {
        stream.markdown(roots.error);
        return;
    }
    const { repoPath, workDir } = roots;

    // `writeInterviewYaml` does sync mkdir + writeFile; both can throw
    // on EACCES, ENOSPC, or a OneDrive sync lock. Without this catch,
    // the chat handler unwinds and the user sees the interview "finish"
    // with no quest and no error.
    let yamlPath: string;
    try {
        yamlPath = writeInterviewYaml(answers, workDir);
    } catch (err) {
        const draftDir = path.join(workDir, "outputs", "_drafts");
        const rawMsg = err instanceof Error ? err.message : String(err);
        // Node FS errors often already end with punctuation
        // (e.g. ``EACCES: permission denied, open '...'``); strip a
        // trailing period to avoid the awkward double-dot.
        const msg = rawMsg.replace(/\.\s*$/, "");
        stream.markdown(
            `❌ Failed to write quest YAML to \`${draftDir}\`: ${msg}. ` +
            `Common causes: disk full, read-only mount, OneDrive sync conflict, missing parent dir permissions.`,
        );
        return;
    }
    const rel = path.relative(workDir, yamlPath).split(path.sep).join("/");
    stream.markdown(`📝 Wrote config: \`${rel}\`\n\n`);
    // The author line is kept for later quests now that the config is written (the profile the CLI and web read).
    keepAuthorLine(answers);
    // Surface which model the user's calls will route through so the
    // budget impact is visible upfront.
    stream.markdown(
        `🤖 Model: \`${userPickedModel.family}\` (vendor: ${userPickedModel.vendor})\n\n`,
    );
    stream.markdown(`▶️ Starting quest…\n\n`);

    // Hand off to the existing runQuest path. We pass the workspace-
    // relative path so the spawned Python's cwd resolves correctly.
    await runQuest(
        rel, /*fleet*/ false, stream, token, userPickedModel,
        undefined, false, undefined, undefined, roots,
    );
}

/** How a `launch.py` run started from the chat ended (see runLaunchInChat). */
interface ChatRun {
    /** The exit code, or null when the process was killed by a signal. */
    code: number | null;
    /** The quest this run is, from the `[FI] <quest_id> -> <quest folder>` line launch.py prints (not for a fleet). */
    questIdSeen?: string;
    /** The last lines Python wrote to stderr: where an unhandled error lands. */
    stderrTail: string[];
}

/**
 * Run `python launch.py <args>` as a child of the extension with its output in this chat, the way `/start` and
 * `/resume` always have: never in a terminal. Every chat command that runs launch.py goes through here, so none of
 * them opens a terminal the person did not ask for, and every model call the run makes comes back through a
 * per-command bridge to the model picked in the Chat panel (`--vscode-bridge-port`). A command that makes no model
 * call (`/ingest`, `/install-tectonic`) gets the port too; it is never used.
 *
 * Only the lines a person needs are shown: what a quest wrote, source failures, the evidence line, each check of a
 * watched job, a model change (`[FI] model:`) and what `--update` approved (`[FI] update:`). With `showAllOutput`
 * (a command whose whole output is the answer) every line is shown as it is printed.
 *
 * Returns undefined when Python could not be started (the reason is already in the chat).
 */
async function runLaunchInChat(
    args: string[],
    stream: vscode.ChatResponseStream,
    token: vscode.CancellationToken,
    userPickedModel: vscode.LanguageModelChat,
    opts: {
        pythonPath: string; launchScript: string; cwd: string; fleet?: boolean; showAllOutput?: boolean;
        /** Extra environment for this run only (`/update` says the person approved config.yaml as it is). */
        env?: Record<string, string>;
    },
): Promise<ChatRun | undefined> {
    // Bind the bridge to a free port. We thread `userPickedModel`
    // (= request.model from the chat handler) into the bridge so
    // every LLM call routes through THAT model — the one the user
    // chose in the VSCode Chat picker — instead of an arbitrary
    // `selectChatModels[0]` from the full Copilot catalog. Without
    // this, a user who picked gpt-5.4-mini (0.33× per request) could
    // see their calls silently route through Claude Opus (5×) or
    // similar, causing 10–15× the premium-request burn they expected.
    const bridge = new Bridge({
        progress: stream,
        cancellationToken: token,
        defaultModel: userPickedModel,
    });
    const port = await bridge.listen();
    const argv: string[] = ["-u", opts.launchScript, "--vscode-bridge-port", String(port), ...args];

    // FI_SKIP_BOOTSTRAP: every spawn in this file sets it, so a missing dependency on
    // `frontierInsight.pythonPath` surfaces as this file's own diagnostic (which names that setting) rather
    // than launch.py's CLI-only self-bootstrap silently creating and switching to a different `.venv/` the
    // user never configured here.
    const child = spawn(opts.pythonPath, argv, {
        cwd: opts.cwd,
        env: {
            ...process.env, PYTHONUNBUFFERED: "1", PYTHONIOENCODING: "utf-8", FI_SKIP_BOOTSTRAP: "1",
            ...(opts.env ?? {}),
        },
        stdio: ["ignore", "pipe", "pipe"],
    });
    // With showAllOutput every line, stdout's and stderr's (warnings, a skipped file), is shown in one code block,
    // opened at the first line so a Python that never started leaves no empty block around its error.
    let fenceOpen = false;
    const showRaw = (line: string): void => {
        if (!line.trim()) return;
        if (!fenceOpen) {
            stream.markdown("```\n");
            fenceOpen = true;
        }
        stream.markdown(line.replace(/```/g, "` ` `") + "\n");
    };
    // Keep a rolling tail of stderr so we can surface the actual
    // traceback in the chat if Python exits non-zero. Without this,
    // the user only sees "exited with code 1, check run.log" — but
    // unhandled exceptions don't reach the run.log (it only carries
    // what `logging.info(...)` etc. emitted before the crash).
    const stderrTail: string[] = [];
    const STDERR_TAIL_LINES = 80;
    bridge.attachChild(child, (line) => {
        if (opts.showAllOutput) showRaw(line);
        stderrTail.push(line);
        if (stderrTail.length > STDERR_TAIL_LINES) {
            stderrTail.splice(0, stderrTail.length - STDERR_TAIL_LINES);
        }
    });

    // Make sure cancellation kills the child + closes the bridge.
    token.onCancellationRequested(() => {
        try { child.kill("SIGTERM"); } catch { /* noop */ }
    });

    // Surface only the user-meaningful lines from Python's stdout. The
    // `[FI] start/resume quest_id=...` echo is already shown via the
    // extension's own header lines, so dropping it avoids the
    // duplicate-print problem the user reported. Generator-output lines
    // like `[FI] wrote paper_md -> <abs path>` ARE useful (they're the
    // final artifact pointers) so we keep those but re-render with just a
    // basename + a checkmark instead of the raw `[FI]` prefix.
    child.stdout.setEncoding("utf-8");
    let stdoutBuf = "";
    let questIdSeen: string | undefined;
    const onLine = (line: string): void => {
        if (opts.showAllOutput) {
            showRaw(line);
            return;
        }
        const ran = line.match(/^\[FI\] (\d+-[\w-]+) -> /);
        if (ran && !opts.fleet) questIdSeen = ran[1];
        const wrote = line.match(/^\[FI\] wrote (\w+) -> (.+)$/);
        if (wrote) {
            stream.markdown(`  ✅ wrote ${wrote[1]} → \`${path.basename(wrote[2])}\`\n\n`);
            return;
        }
        const summary = line.match(/^\[FI\] summary -> (.+)$/);
        if (summary) {
            stream.markdown(`  📋 summary → \`${path.basename(summary[1])}\`\n\n`);
            return;
        }
        // Literature / full-text sources that failed during the run
        // (rate limits, blocks, timeouts) — otherwise only in run.log.
        const failures = line.match(/^\[FI\] source failures: (.+)$/);
        if (failures) {
            stream.markdown(`  ⚠️ source failures: \`${failures[1]}\`\n\n`);
            return;
        }
        // How much of the result has been checked against something other than itself.
        const evidenceLine = line.match(/^\[FI\] evidence: (.+)$/);
        if (evidenceLine) {
            stream.markdown(`  🔎 evidence: \`${evidenceLine[1]}\`\n\n`);
            return;
        }
        // The quest's model changed since it last ran (core/engine.py _take_model_change): said, never a stop.
        const model = line.match(/^\[FI\] model: (.+)$/);
        if (model) {
            stream.markdown(`  🤖 ${model[1]}\n\n`);
            return;
        }
        // What `--update` approved and did, run here with no terminal (core/interview_update.py).
        const update = line.match(/^\[FI\] update: (.*)$/);
        if (update) {
            if (update[1].trim()) stream.markdown(`  ${update[1]}\n\n`);
            return;
        }
        // The quest's own config.yaml was brought up to date with the config it resumed with.
        if (/^\[FI\] the quest's config\.yaml now matches /.test(line)) {
            stream.markdown(`  📝 ${line.slice("[FI] ".length)}\n\n`);
            return;
        }
        // Each check of a watched background job: the only monitor there is.
        const checked = line.match(/^\[watch\] (.+)$/);
        if (checked) {
            stream.markdown(`  👁 ${checked[1]}\n\n`);
        }
        // Drop other [FI] lines (start/resume quest_id=, paths the
        // user already saw in our header, etc.) — they're noise here.
    };
    child.stdout.on("data", (chunk: string) => {
        stdoutBuf += chunk;
        const lines = stdoutBuf.split(/\r?\n/);
        stdoutBuf = lines.pop() || "";
        for (const line of lines) onLine(line);
    });

    const result = await waitForChildExit(child, bridge, opts.pythonPath, stream);
    if (stdoutBuf) onLine(stdoutBuf);
    if (fenceOpen) stream.markdown("```\n");
    if (result.kind === "spawn-error") return undefined;
    return { code: result.code, questIdSeen, stderrTail };
}


async function runQuest(
    promptArgs: string,
    fleet: boolean,
    stream: vscode.ChatResponseStream,
    token: vscode.CancellationToken,
    userPickedModel: vscode.LanguageModelChat,
    resumeQuestId?: string,
    watch = false,
    revisePlan?: string,
    fromStep?: string,
    knownRoots?: Roots,
    where?: RunWhere,
): Promise<void> {
    // A quest from another folder names its config by full path, which may hold spaces: taken whole.
    const paths = where ? [where.configPath] : promptArgs.split(/\s+/).filter((s) => s.length > 0);
    if (paths.length === 0) {
        stream.markdown(
            "Need at least one YAML path. Example: `@fi /start examples/integrator_bakeoff/config.yaml`",
        );
        return;
    }
    if (!fleet && paths.length > 1) {
        stream.markdown(
            "`/start` takes exactly one YAML. Use `/fleet` for multiple.",
        );
        return;
    }

    const cfg = vscode.workspace.getConfiguration("frontierInsight");
    const pythonPath = cfg.get<string>("pythonPath") || "python";
    // With several folders open, the study's folder is the one holding its YAML.
    const roots = knownRoots ?? await rootsForCommand(paths[0]);
    if ("error" in roots) {
        stream.markdown(roots.error);
        return;
    }
    const { repoPath, workDir } = roots;
    const launchScript = path.join(repoPath, "launch.py");

    stream.markdown(`🧪 Starting ${fleet ? "fleet" : "quest"}: \`${paths.join(", ")}\`\n\n`);

    // 1. What launch.py is asked to do. The per-command bridge (and its port) is added by runLaunchInChat.
    // --vscode-bridge-port selects the vscode_extension provider only
    // when the YAML has not already named a different one: a config
    // that explicitly picked claude_cli / openai / etc. keeps its own
    // transport (launch.py:_apply_vscode_bridge_override). Every YAML
    // the /new interview writes pins vscode_extension, so on the
    // common path every LLM call does come back through this bridge.
    const args: string[] = [];
    if (fleet) {
        args.push("--fleet", ...paths);
    } else {
        args.push("--config", paths[0]);
        if (resumeQuestId) {
            // --watch re-checks the background job on a timer and resumes the
            // quest itself when it is done; --resume resumes right away.
            args.push(watch ? "--watch" : "--resume", resumeQuestId);
            // --revise-plan rewrites plan.md and runs nothing else; the words are one argument.
            if (revisePlan) args.push("--revise-plan", revisePlan);
            if (fromStep) args.push("--from", fromStep);
            // Its own outputs folder, whatever a relative output_dir in its YAML means from here.
            if (where) args.push("--output", where.outputDir);
        }
    }

    // 2. Run it with its output in this chat, and wait for it to end: the
    // chat handler resolves when the child terminates, which is when the
    // quest (or fleet) is done or stops for you.
    const startedAt = Date.now();
    const ran = await runLaunchInChat(args, stream, token, userPickedModel, {
        pythonPath, launchScript, cwd: where?.cwd ?? workDir, fleet,
    });
    if (!ran) return;

    if (ran.code === 0 && revisePlan && resumeQuestId) {
        const outDirSetting = cfg.get<string>("outputDir") || "outputs";
        const planPath = path.join(
            where?.outputDir ?? (path.isAbsolute(outDirSetting) ? outDirSetting : path.join(workDir, outDirSetting)),
            resumeQuestId, "plan.md",
        );
        stream.markdown(
            `\n✅ The plan was rewritten (the old version is kept in \`.fi/plan_versions/\`). ` +
            `Read it: \`@fi /plan ${resumeQuestId}\`. Ask again, edit it yourself, or run it: \`@fi /resume ${resumeQuestId}\`.\n`,
        );
        try {
            const doc = await vscode.workspace.openTextDocument(vscode.Uri.file(planPath));
            await vscode.window.showTextDocument(doc, { preview: false, viewColumn: tabAreaColumn(doc.uri) });
        } catch { /* the message above names the command that opens it */ }
    } else {
        const outDirSetting = cfg.get<string>("outputDir") || "outputs";
        const outputsDir = where?.outputDir ?? (path.isAbsolute(outDirSetting)
            ? outDirSetting
            : path.join(workDir, outDirSetting));
        await reportQuestEnd(ran, stream, { outputsDir, questId: resumeQuestId, fleet, startedAt });
    }
}


/**
 * What a quest run started from the chat ended with (`/start`, `/resume`, `/update`): the to-do card when it stopped
 * for you (a quest that stops also exits 0), "finished" with what is worth a look when it finished, the papers it
 * wants, or the end of stderr when Python failed.
 */
async function reportQuestEnd(
    ran: ChatRun,
    stream: vscode.ChatResponseStream,
    opts: { outputsDir: string; questId?: string; fleet?: boolean; startedAt: number },
): Promise<void> {
    const { outputsDir, fleet, startedAt } = opts;
    // The quest launch.py says it ran (`[FI] <quest_id> -> <folder>`), else the one asked for.
    const questId = ran.questIdSeen ?? opts.questId;
    if (ran.code === 0) {
        // A quest that stopped for you also exits 0: say it is waiting, not that it finished.
        const card = await readNextStep(outputsDir, questId, startedAt);
        if (card) {
            stream.markdown(`\n\n---\n\n⏸ **The quest is waiting for you.**\n\n${card.markdown}\n`);
        } else {
            stream.markdown(`\n✅ ${fleet ? "Fleet" : "Quest"} finished cleanly.`);
            if (questId) await surfaceWorthALook(outputsDir, stream, questId);
        }
        // This run's quest, never another one running beside it in the same folder.
        await surfaceWantedPapers(outputsDir, stream, questId ?? card?.questId, !!card, startedAt);
    } else {
        const tail = ran.stderrTail.join("\n");
        stream.markdown(
            `\n❌ **Python exited with code ${ran.code}.**\n\n` +
            (tail.trim()
                ? "Last lines of stderr (the actual error usually lives here, **not** in `run.log` — unhandled exceptions skip the logger):\n\n" +
                  "```\n" + tail + "\n```\n"
                : "stderr was empty. Check `outputs/<quest_id>/.fi/run.log` for whatever made it to the logger before the crash.\n"),
        );
    }
}


/**
 * Best-effort: when a quest stops for you, the engine writes the to-do card
 * `NEXT_STEP.md` at the quest root (why it stopped, what to decide, the
 * recommendation, the alternatives, everything else waiting; core/todo.py) and
 * deletes it on completion. So its presence means "waiting for you". Returns
 * the card, or null. Mirrors the Web quest page's Action-needed banner.
 */
// A file written by this run can carry a time a few seconds before the run started, by the extension's clock, when the
// quest runs on another machine (Remote SSH, WSL, a container).
const CLOCK_SLACK_MS = 10_000;

async function readNextStep(
    outputsDir: string,
    knownQuestId: string | undefined,
    since: number,
): Promise<{ questId: string; markdown: string } | null> {
    try {
        let questId: string | null = null;
        if (knownQuestId) {
            try {
                await fsPromises.stat(path.join(outputsDir, knownQuestId, "NEXT_STEP.md"));
                questId = knownQuestId;
            } catch { return null; }
        } else {
            // /new: the quest whose NEXT_STEP.md this run wrote (never an older quest's card).
            const entries = await fsPromises.readdir(outputsDir, { withFileTypes: true });
            let best: { id: string; mtime: number } | null = null;
            for (const e of entries) {
                if (!e.isDirectory() || e.name.startsWith("_")) { continue; }
                try {
                    const st = await fsPromises.stat(path.join(outputsDir, e.name, "NEXT_STEP.md"));
                    if (st.mtimeMs >= since - CLOCK_SLACK_MS && (!best || st.mtimeMs > best.mtime)) { best = { id: e.name, mtime: st.mtimeMs }; }
                } catch { /* no NEXT_STEP in this quest */ }
            }
            if (!best) { return null; }
            questId = best.id;
        }
        const markdown = await fsPromises.readFile(path.join(outputsDir, questId, "NEXT_STEP.md"), "utf-8");
        return { questId, markdown };
    } catch { return null; }
}

/**
 * A finished quest's to-do card (`.fi/todo.json`, core/todo.py): what did not stop it but is worth a look.
 */
async function surfaceWorthALook(
    outputsDir: string,
    stream: vscode.ChatResponseStream,
    questId: string,
): Promise<void> {
    try {
        const raw = await fsPromises.readFile(path.join(outputsDir, questId, ".fi", "todo.json"), "utf-8");
        const items = (JSON.parse(raw).items || []) as Array<{ why?: string; recommended?: string; blocking?: boolean }>;
        const rest = items.filter((i) => !i.blocking);
        if (!rest.length) { return; }
        stream.markdown(
            "\n\n**Worth a look:**\n\n" +
            rest.map((i) => `- **${i.why || ""}** ${i.recommended || ""}`).join("\n") + "\n",
        );
    } catch { /* no card: nothing waiting */ }
}


/**
 * Best-effort: after a quest finishes, if it paused for paywalled papers
 * (wrote `needs/WANTED_PAPERS.md`) and the user hasn't supplied any PDFs
 * yet, surface the ranked wanted-papers list + drop-and-resume hint in the
 * chat. Mirrors the Web quest page's papers panel.
 */
async function surfaceWantedPapers(
    outputsDir: string,
    stream: vscode.ChatResponseStream,
    knownQuestId?: string,
    waiting = true,
    since = 0,
): Promise<void> {
    try {
        let questId: string | null = null;
        if (knownQuestId) {
            // /resume <id>: check that quest directly — no mtime guessing.
            try {
                await fsPromises.stat(path.join(outputsDir, knownQuestId, "needs", "WANTED_PAPERS.md"));
                questId = knownQuestId;
            } catch { return; }
        } else {
            // /fleet: no one id — pick the quest whose WANTED_PAPERS.md was
            // written most recently, and only by this run (not an older quest's).
            const entries = await fsPromises.readdir(outputsDir, { withFileTypes: true });
            let best: { id: string; mtime: number } | null = null;
            for (const e of entries) {
                if (!e.isDirectory() || e.name.startsWith("_")) { continue; }
                const wanted = path.join(outputsDir, e.name, "needs", "WANTED_PAPERS.md");
                try {
                    const st = await fsPromises.stat(wanted);
                    if (st.mtimeMs < since - CLOCK_SLACK_MS) { continue; }
                    if (!best || st.mtimeMs > best.mtime) { best = { id: e.name, mtime: st.mtimeMs }; }
                } catch { /* no needs file in this quest */ }
            }
            if (!best) { return; }
            questId = best.id;
        }
        const best = { id: questId };
        const papersDir = path.join(outputsDir, best.id, "inputs", "papers");
        try {
            const dropped = (await fsPromises.readdir(papersDir)).filter(
                (n) => /\.(pdf|md|txt)$/i.test(n) && n !== "README.md",
            );
            if (dropped.length) { return; }  // user already supplied PDFs
        } catch { /* no papers dir yet */ }
        const md = await fsPromises.readFile(
            path.join(outputsDir, best.id, "needs", "WANTED_PAPERS.md"), "utf-8",
        );
        const next = waiting
            ? `, then \`@fi /resume ${best.id}\` — or resume without adding any: those papers ` +
              `are used from their abstracts, and FI does not ask for them again.`
            : `. The quest did not stop for them, so there is nothing to resume: it went on with their abstracts.`;
        stream.markdown(
            "\n\n---\n\n📄 **Some relevant papers are not confirmed free to read** — FI downloads only free " +
            `papers, so it has only their abstracts. Download the ones that matter and drop the PDFs into ` +
            `\`${papersDir}\`${next}\n\n${md}\n`,
        );
    } catch { /* best-effort */ }
}


/**
 * Implementation of `@fi /summarize <folder> [kind]`. Spawns
 * `python launch.py --summarize <abs-folder> [--summarize-kind <kind>]
 * --vscode-bridge-port <N>` and streams progress / errors back to the
 * chat panel, mirroring `runQuest`'s shape.
 *
 * Argument parsing: split on whitespace; the first token is the folder
 * path (relative paths resolve against the workspace folder), the
 * optional second token is the kind override (one of: auto, literature,
 * code, study, execution, mixed).
 */
async function runSummarize(
    promptArgs: string,
    stream: vscode.ChatResponseStream,
    token: vscode.CancellationToken,
    userPickedModel: vscode.LanguageModelChat,
): Promise<void> {
    // Parse "<folder> [kind]" without breaking on Windows paths that
    // contain spaces (`C:\My Papers\thesis`). Strategy:
    //   1. If the user wrapped the path in quotes, honor them.
    //   2. Otherwise, the LAST whitespace-separated token MIGHT be the
    //      optional kind enum. If it matches a known kind, peel it off
    //      and treat the prefix as the path. Otherwise the entire
    //      prompt is the path.
    const validKinds = new Set([
        "auto", "literature", "code", "study", "execution", "mixed",
    ]);
    const raw = promptArgs.trim();
    if (!raw) {
        stream.markdown(
            "Need a folder path. Example: `@fi /summarize ./papers` or " +
            "`@fi /summarize \"C:/My Papers\" literature`.\n",
        );
        return;
    }

    let folderArg: string;
    let kindArg: string = "auto";

    // 1. Quoted path: `"C:/My Papers"` or `"C:/My Papers" literature`.
    const quoted = raw.match(/^"([^"]+)"\s*(\S*)\s*$/);
    if (quoted) {
        folderArg = quoted[1];
        if (quoted[2]) {
            if (!validKinds.has(quoted[2])) {
                stream.markdown(
                    `Invalid kind \`${quoted[2]}\`. Valid: ${
                        Array.from(validKinds).join(", ")}.\n`,
                );
                return;
            }
            kindArg = quoted[2];
        }
    } else {
        // 2. Unquoted: check last token for a kind enum.
        const lastSpace = raw.lastIndexOf(" ");
        if (lastSpace > 0) {
            const tail = raw.slice(lastSpace + 1);
            if (validKinds.has(tail)) {
                folderArg = raw.slice(0, lastSpace).trim();
                kindArg = tail;
            } else {
                // Last token isn't a known kind — treat the whole
                // input as the folder path (preserves spaces in
                // unquoted Windows-style paths).
                folderArg = raw;
            }
        } else {
            folderArg = raw;
        }
    }
    if (!folderArg) {
        stream.markdown("Empty folder path after parsing. Use quotes for paths with spaces.\n");
        return;
    }

    const cfg = vscode.workspace.getConfiguration("frontierInsight");
    const pythonPath = cfg.get<string>("pythonPath") || "python";
    const roots = await rootsForCommand(folderArg);
    if ("error" in roots) {
        stream.markdown(roots.error);
        return;
    }
    const { repoPath, workDir } = roots;
    const launchScript = path.join(repoPath, "launch.py");

    // Resolve the folder argument: absolute paths are honored as-is;
    // relative paths resolve against the workspace folder. This lets
    // a user type `@fi /summarize ./papers` from the chat panel.
    const folderAbs = path.isAbsolute(folderArg)
        ? folderArg
        : path.resolve(workDir, folderArg);
    let folderStat: import("fs").Stats;
    try {
        folderStat = await fsPromises.stat(folderAbs);
    } catch {
        stream.markdown(
            `❌ Path not found: \`${folderAbs}\`. ` +
            `(Resolved from \`${folderArg}\` against \`${workDir}\`.)\n`,
        );
        return;
    }
    if (!folderStat.isDirectory()) {
        stream.markdown(
            `❌ Path exists but is a file, not a directory: \`${folderAbs}\`. ` +
            `\`/summarize\` expects a folder.\n`,
        );
        return;
    }

    stream.markdown(
        `🗂️ Summarizing folder \`${folderAbs}\` (kind: \`${kindArg}\`)\n\n` +
        `🤖 Model: \`${userPickedModel.family}\` (vendor: ${userPickedModel.vendor})\n\n` +
        `▶️ Walking files…\n\n`,
    );

    const bridge = new Bridge({
        progress: stream,
        cancellationToken: token,
        defaultModel: userPickedModel,
    });
    const port = await bridge.listen();

    const argv: string[] = [
        "-u", launchScript,
        "--vscode-bridge-port", String(port),
        "--summarize", folderAbs,
        "--summarize-kind", kindArg,
    ];

    const child = spawn(pythonPath, argv, {
        cwd: workDir,
        env: { ...process.env, PYTHONUNBUFFERED: "1", FI_SKIP_BOOTSTRAP: "1" },
        stdio: ["ignore", "pipe", "pipe"],
    });
    const stderrTail: string[] = [];
    const STDERR_TAIL_LINES = 80;
    bridge.attachChild(child, (line) => {
        stderrTail.push(line);
        if (stderrTail.length > STDERR_TAIL_LINES) {
            stderrTail.splice(0, stderrTail.length - STDERR_TAIL_LINES);
        }
    });
    token.onCancellationRequested(() => {
        try { child.kill("SIGTERM"); } catch { /* noop */ }
    });

    // Surface the [FI] summary -> <path> line so the user knows where
    // to open the result.
    child.stdout.setEncoding("utf-8");
    let stdoutBuf = "";
    child.stdout.on("data", (chunk: string) => {
        stdoutBuf += chunk;
        const lines = stdoutBuf.split(/\r?\n/);
        stdoutBuf = lines.pop() || "";
        for (const line of lines) {
            const m = line.match(/^\[FI\] summary -> (.+)$/);
            if (m) {
                stream.markdown(`  ✅ summary written → \`${m[1]}\`\n\n`);
                continue;
            }
            const sf = line.match(/^\[FI\] source failures: (.+)$/);
            if (sf) {
                stream.markdown(`  ⚠️ source failures: \`${sf[1]}\`\n\n`);
                continue;
            }
            const km = line.match(/^\[FI\] (\d+) files; detected_kind=(\S+); ingested_to_axon=(\S+)$/);
            if (km) {
                stream.markdown(
                    `  📊 ${km[1]} files; detected kind: \`${km[2]}\`; Axon ingest: \`${km[3]}\`\n\n`,
                );
            }
        }
    });

    const result = await waitForChildExit(child, bridge, pythonPath, stream);
    if (result.kind === "spawn-error") return;
    const exitCode = result.code;

    if (exitCode === 0) {
        stream.markdown("\n✅ Summary done.\n");
    } else {
        const tail = stderrTail.join("\n");
        stream.markdown(
            `\n❌ **Python exited with code ${exitCode}.**\n\n` +
            (tail.trim()
                ? "Stderr tail:\n\n```\n" + tail + "\n```\n"
                : "stderr was empty.\n"),
        );
    }
}


/**
 * Implementation of `@fi /digest [days]`. Spawns
 * `python launch.py --digest --days <N> --vscode-bridge-port <P>` and
 * streams progress / errors back to chat.
 *
 * The optional `days` argument is the only parameter we accept from
 * chat — defaults to 7 (rolling week). Pass `@fi /digest 14` for the
 * last fortnight or `@fi /digest 30` for a monthly view.
 *
 * Output lands at `<outputDir>/_digests/<YYYY-Www>.md` (or the dated
 * form for non-7-day windows). The function surfaces the
 * `[FI] digest -> <path>` line so the user can click straight to it.
 */
async function runDigest(
    promptArgs: string,
    stream: vscode.ChatResponseStream,
    token: vscode.CancellationToken,
    userPickedModel: vscode.LanguageModelChat,
): Promise<void> {
    if (token.isCancellationRequested) return;

    // Parse the optional days argument. Accept bare integers only —
    // anything fancier (date ranges, weeks) goes through the launch.py
    // flags directly, not through the chat command.
    const raw = promptArgs.trim();
    let days = 7;
    if (raw) {
        const n = parseInt(raw, 10);
        if (Number.isFinite(n) && n > 0 && String(n) === raw) {
            days = n;
        } else {
            stream.markdown(
                `\`/digest\` expected an optional positive integer (days) or nothing. Got \`${raw}\`.\n\n` +
                `Examples: \`@fi /digest\` (rolling 7-day digest), \`@fi /digest 14\`, \`@fi /digest 30\`.\n`,
            );
            return;
        }
    }

    const cfg = vscode.workspace.getConfiguration("frontierInsight");
    const pythonPath = cfg.get<string>("pythonPath") || "python";
    const roots = await rootsForCommand();
    if ("error" in roots) {
        stream.markdown(roots.error);
        return;
    }
    const { repoPath, workDir } = roots;
    const launchScript = path.join(repoPath, "launch.py");
    const outputDirSetting = cfg.get<string>("outputDir") || "outputs";
    const outputsDir = path.isAbsolute(outputDirSetting)
        ? outputDirSetting
        : path.join(workDir, outputDirSetting);

    stream.markdown(
        `📅 Generating digest for the last **${days} days** of quests under ` +
        `\`${outputsDir}\`.\n\n` +
        `🤖 Model: \`${userPickedModel.family}\` (vendor: ${userPickedModel.vendor})\n\n` +
        `▶️ Walking quests…\n\n`,
    );

    const bridge = new Bridge({
        progress: stream,
        cancellationToken: token,
        defaultModel: userPickedModel,
    });
    const port = await bridge.listen();

    const argv: string[] = [
        "-u", launchScript,
        "--vscode-bridge-port", String(port),
        "--digest",
        "--days", String(days),
        "--output-root", outputsDir,
    ];

    const child = spawn(pythonPath, argv, {
        cwd: workDir,
        env: { ...process.env, PYTHONUNBUFFERED: "1", FI_SKIP_BOOTSTRAP: "1" },
        stdio: ["ignore", "pipe", "pipe"],
    });
    const stderrTail: string[] = [];
    const STDERR_TAIL_LINES = 80;
    bridge.attachChild(child, (line) => {
        stderrTail.push(line);
        if (stderrTail.length > STDERR_TAIL_LINES) {
            stderrTail.splice(0, stderrTail.length - STDERR_TAIL_LINES);
        }
    });
    token.onCancellationRequested(() => {
        try { child.kill("SIGTERM"); } catch { /* noop */ }
    });

    child.stdout.setEncoding("utf-8");
    let stdoutBuf = "";
    child.stdout.on("data", (chunk: string) => {
        stdoutBuf += chunk;
        const lines = stdoutBuf.split(/\r?\n/);
        stdoutBuf = lines.pop() || "";
        for (const line of lines) {
            const m = line.match(/^\[FI\] digest -> (.+)$/);
            if (m) {
                stream.markdown(`  ✅ digest written → \`${m[1]}\`\n\n`);
                continue;
            }
            if (line.startsWith("[FI] vs ")) {
                stream.markdown(`  📊 ${line.slice(5)}\n\n`);
                continue;
            }
            if (line.match(/^\[FI\] \d+ quests touched/)) {
                stream.markdown(`  📈 ${line.slice(5)}\n\n`);
            }
        }
    });

    const result = await waitForChildExit(child, bridge, pythonPath, stream);
    if (result.kind === "spawn-error") return;
    const exitCode = result.code;

    if (exitCode === 0) {
        stream.markdown("\n✅ Digest done.\n");
    } else {
        const tail = stderrTail.join("\n");
        stream.markdown(
            `\n❌ **Python exited with code ${exitCode}.**\n\n` +
            (tail.trim()
                ? "Stderr tail:\n\n```\n" + tail + "\n```\n"
                : "stderr was empty.\n"),
        );
    }
}


/**
 * Implementation of `@fi /portfolio`. Spawns
 * `python launch.py --portfolio --vscode-bridge-port <P>` and streams
 * the result back. Takes no arguments — portfolio is unbounded by
 * design (covers every quest under `outputs/`).
 */
async function runPortfolio(
    stream: vscode.ChatResponseStream,
    token: vscode.CancellationToken,
    userPickedModel: vscode.LanguageModelChat,
): Promise<void> {
    if (token.isCancellationRequested) return;

    const cfg = vscode.workspace.getConfiguration("frontierInsight");
    const pythonPath = cfg.get<string>("pythonPath") || "python";
    const roots = await rootsForCommand();
    if ("error" in roots) {
        stream.markdown(roots.error);
        return;
    }
    const { repoPath, workDir } = roots;
    const launchScript = path.join(repoPath, "launch.py");
    const outputDirSetting = cfg.get<string>("outputDir") || "outputs";
    const outputsDir = path.isAbsolute(outputDirSetting)
        ? outputDirSetting
        : path.join(workDir, outputDirSetting);

    stream.markdown(
        `📚 Synthesizing portfolio across every quest under \`${outputsDir}\`.\n\n` +
        `🤖 Model: \`${userPickedModel.family}\` (vendor: ${userPickedModel.vendor})\n\n` +
        `▶️ Walking quests…\n\n`,
    );

    const bridge = new Bridge({
        progress: stream,
        cancellationToken: token,
        defaultModel: userPickedModel,
    });
    const port = await bridge.listen();

    const argv: string[] = [
        "-u", launchScript,
        "--vscode-bridge-port", String(port),
        "--portfolio",
        "--output-root", outputsDir,
    ];

    const child = spawn(pythonPath, argv, {
        cwd: workDir,
        env: { ...process.env, PYTHONUNBUFFERED: "1", FI_SKIP_BOOTSTRAP: "1" },
        stdio: ["ignore", "pipe", "pipe"],
    });
    const stderrTail: string[] = [];
    const STDERR_TAIL_LINES = 80;
    bridge.attachChild(child, (line) => {
        stderrTail.push(line);
        if (stderrTail.length > STDERR_TAIL_LINES) {
            stderrTail.splice(0, stderrTail.length - STDERR_TAIL_LINES);
        }
    });
    token.onCancellationRequested(() => {
        try { child.kill("SIGTERM"); } catch { /* noop */ }
    });

    child.stdout.setEncoding("utf-8");
    let stdoutBuf = "";
    child.stdout.on("data", (chunk: string) => {
        stdoutBuf += chunk;
        const lines = stdoutBuf.split(/\r?\n/);
        stdoutBuf = lines.pop() || "";
        for (const line of lines) {
            const m = line.match(/^\[FI\] portfolio -> (.+)$/);
            if (m) {
                stream.markdown(`  ✅ portfolio written → \`${m[1]}\`\n\n`);
                continue;
            }
            if (line.match(/^\[FI\] \d+ quests;/)) {
                stream.markdown(`  📈 ${line.slice(5)}\n\n`);
            }
        }
    });

    const result = await waitForChildExit(child, bridge, pythonPath, stream);
    if (result.kind === "spawn-error") return;
    const exitCode = result.code;

    if (exitCode === 0) {
        stream.markdown("\n✅ Portfolio done.\n");
    } else {
        const tail = stderrTail.join("\n");
        stream.markdown(
            `\n❌ **Python exited with code ${exitCode}.**\n\n` +
            (tail.trim()
                ? "Stderr tail:\n\n```\n" + tail + "\n```\n"
                : "stderr was empty.\n"),
        );
    }
}


/**
 * Implementation of `@fi /critique <quest_id>`. Spawns
 * `python launch.py --critique <quest_id> --vscode-bridge-port <P>`
 * and streams the result back to the chat panel.
 *
 * For the strongest adversarial effect the user should pick a Copilot
 * model in the chat picker DIFFERENT from the one that wrote the
 * paper. The extension can't enforce that — the picker is the user's
 * choice — but the produced critique.md records both providers so
 * the user can see post-hoc which was which.
 */
async function runCritique(
    promptArgs: string,
    stream: vscode.ChatResponseStream,
    token: vscode.CancellationToken,
    userPickedModel: vscode.LanguageModelChat,
): Promise<void> {
    if (token.isCancellationRequested) return;

    const questId = promptArgs.trim();
    // Validate the quest_id shape client-side so we can give a clear
    // chat error rather than letting the Python side reject it.
    // <10-digit-epoch>-<lowercase-slug>-<6-hex-nonce>.
    const questIdPattern = /^\d{10}-[a-z0-9-]+-[0-9a-f]{6}$/;
    if (!questId) {
        stream.markdown(
            "Need a quest_id. Example: `@fi /critique 1778452404-euv-mor-photon-shot-noise-ler-e6bfe5`.\n\n" +
            "To find a quest_id, look under your `outputs/` directory or run `@fi /resume` to see the picker.\n",
        );
        return;
    }
    if (!questIdPattern.test(questId)) {
        stream.markdown(
            `\`${questId}\` doesn't look like a valid quest_id. Expected the \`<epoch>-<slug>-<6hex>\` form ` +
            "(e.g. `1778452404-euv-mor-photon-shot-noise-ler-e6bfe5`).\n",
        );
        return;
    }

    const cfg = vscode.workspace.getConfiguration("frontierInsight");
    const pythonPath = cfg.get<string>("pythonPath") || "python";
    const roots = await rootsForCommand();
    if ("error" in roots) {
        stream.markdown(roots.error);
        return;
    }
    const { repoPath, workDir } = roots;
    const launchScript = path.join(repoPath, "launch.py");
    const outputDirSetting = cfg.get<string>("outputDir") || "outputs";
    const outputsDir = path.isAbsolute(outputDirSetting)
        ? outputDirSetting
        : path.join(workDir, outputDirSetting);

    stream.markdown(
        `🔍 Running adversarial critique of quest \`${questId}\`.\n\n` +
        `🤖 Critique model: \`${userPickedModel.family}\` (vendor: ${userPickedModel.vendor})\n\n` +
        `💡 _For the strongest second-opinion effect, pick a different model family in the chat picker from the one that wrote the paper. The critique.md will record both._\n\n` +
        `▶️ Loading paper / code / prior review…\n\n`,
    );

    const bridge = new Bridge({
        progress: stream,
        cancellationToken: token,
        defaultModel: userPickedModel,
    });
    const port = await bridge.listen();

    const argv: string[] = [
        "-u", launchScript,
        "--vscode-bridge-port", String(port),
        "--critique", questId,
        "--output-root", outputsDir,
    ];

    const child = spawn(pythonPath, argv, {
        cwd: workDir,
        env: { ...process.env, PYTHONUNBUFFERED: "1", FI_SKIP_BOOTSTRAP: "1" },
        stdio: ["ignore", "pipe", "pipe"],
    });
    const stderrTail: string[] = [];
    const STDERR_TAIL_LINES = 80;
    bridge.attachChild(child, (line) => {
        stderrTail.push(line);
        if (stderrTail.length > STDERR_TAIL_LINES) {
            stderrTail.splice(0, stderrTail.length - STDERR_TAIL_LINES);
        }
    });
    token.onCancellationRequested(() => {
        try { child.kill("SIGTERM"); } catch { /* noop */ }
    });

    child.stdout.setEncoding("utf-8");
    let stdoutBuf = "";
    child.stdout.on("data", (chunk: string) => {
        stdoutBuf += chunk;
        const lines = stdoutBuf.split(/\r?\n/);
        stdoutBuf = lines.pop() || "";
        for (const line of lines) {
            const m = line.match(/^\[FI\] critique -> (.+)$/);
            if (m) {
                stream.markdown(`  ✅ critique written → \`${m[1]}\`\n\n`);
                continue;
            }
            if (line.startsWith("[FI] critique_provider=")) {
                stream.markdown(`  📊 ${line.slice(5)}\n\n`);
            }
        }
    });

    const result = await waitForChildExit(child, bridge, pythonPath, stream);
    if (result.kind === "spawn-error") return;
    const exitCode = result.code;

    if (exitCode === 0) {
        stream.markdown("\n✅ Critique done.\n");
    } else {
        const tail = stderrTail.join("\n");
        stream.markdown(
            `\n❌ **Python exited with code ${exitCode}.**\n\n` +
            (tail.trim()
                ? "Stderr tail:\n\n```\n" + tail + "\n```\n"
                : "stderr was empty.\n"),
        );
    }
}


/**
 * Implementation of `@fi /proposal <topic>`. Spawns
 * `python launch.py --proposal "<topic>"` and streams the result
 * back. The Python side writes BOTH a planning markdown and a
 * companion YAML under outputs/_drafts/; the chat panel surfaces
 * both paths so the user can read the .md and (if they like the plan)
 * launch the quest via `/start <yaml>`.
 *
 * Argument parsing: the entire prompt after `/proposal ` is the topic
 * — we don't split on whitespace. This matches `/start` semantics
 * for path-like arguments and accepts free-form topic strings with
 * spaces, punctuation, newlines pasted from chat.
 */
/**
 * Look for the Axon sidecar (``python -m axon.api``) when the extension
 * activates. If nothing is reachable, show an info notification with a
 * one-click "Start in terminal" action. We deliberately do NOT
 * auto-launch the sidecar from the extension — users running FI from
 * VSCode are expected to keep an Axon process around themselves (or
 * just live with the cold-init cost on the first quest). The check is
 * informational only.
 *
 * Mirrors `core.axon_sidecar.axon_status` (CLI/web auto-launch path).
 * Three-interface parity for Axon health visibility.
 */
async function probeAxonOnActivate(
    context: vscode.ExtensionContext,
    output: vscode.OutputChannel,
): Promise<void> {
    const waitSec = vscode.workspace
        .getConfiguration("frontierInsight")
        .get<number>("axonStartupWaitSec", DEFAULT_AXON_WAIT_SEC);
    // 0 disables the notice outright — the escape hatch behind the
    // "Don't show again" button below.
    if (!waitSec || waitSec <= 0) return;

    // Activation runs on `onStartupFinished`, which fires long before a
    // sidecar started around the same time is serving: Axon has to load
    // the embedding model and open its vector indexes, and it only
    // writes its lock file once the server is actually up. Giving up
    // after half a minute produces a "not detected" notification for a
    // sidecar that is simply still booting — the user then dismisses a
    // warning that was wrong by the time they read it.
    //
    // So we wait, and we wait quietly. Two phases: a responsive one that
    // catches the common case fast, then a patient one that keeps
    // watching in the background for the rest of the budget. If Axon
    // shows up at any point we return silently and no notification is
    // ever shown. Only a sidecar that never appears produces one.
    //
    // Each sweep re-runs discovery rather than reusing the first
    // sweep's answer: the lock file appears partway through Axon's own
    // boot, so a later sweep can find an endpoint the first one had no
    // way to know about.
    //
    // The polling itself is nearly free — nothing listening on loopback
    // means an instant ECONNREFUSED, not a timeout — so the long tail
    // costs a rounding error of CPU.
    const DEADLINE_MS = waitSec * 1000;
    const PER_PROBE_TIMEOUT_MS = 1000;
    const FAST_PHASE_MS = 60000;
    const FAST_INTERVAL_MS = 2000;
    const SLOW_INTERVAL_MS = 15000;

    // Stop polling if the window closes mid-wait; the budget can now
    // outlive a short session.
    let cancelled = false;
    context.subscriptions.push({ dispose: () => { cancelled = true; } });

    const startedAt = Date.now();
    let found: AxonDiscovery | undefined;
    let attempts = 0;
    while (!cancelled && Date.now() - startedAt < DEADLINE_MS) {
        attempts++;
        // Re-read the override every sweep. The wait can run for
        // minutes, and someone whose Axon isn't being found is exactly
        // the person likely to set `axonUrl` during it — snapshotting
        // it once would ignore them until the next reload.
        found = await discoverAxon({
            overrideUrl: axonUrlSetting(),
            timeoutMs: PER_PROBE_TIMEOUT_MS,
        });
        if (found.live) {
            if (attempts > 1) {
                const waited = ((Date.now() - startedAt) / 1000).toFixed(0);
                output.appendLine(
                    `[fi] Axon came up at ${found.baseUrl} after ${waited}s ` +
                    `(found via ${found.source}).`,
                );
            }
            return;
        }
        const elapsed = Date.now() - startedAt;
        const remaining = DEADLINE_MS - elapsed;
        if (remaining <= 0) break;
        const interval = elapsed < FAST_PHASE_MS ? FAST_INTERVAL_MS : SLOW_INTERVAL_MS;
        await new Promise(r => setTimeout(r, Math.min(interval, remaining)));
    }
    if (cancelled) return;

    // Deadline reached — no candidate answered. Write the per-endpoint
    // detail to the output channel (a panel the user can open) rather
    // than console.warn, which only surfaces in the extension host's
    // Developer Tools.
    const totalElapsedSec = ((Date.now() - startedAt) / 1000).toFixed(1);
    output.appendLine(
        `[fi] no Axon sidecar found after ${attempts} discovery sweeps over ${totalElapsedSec}s.`,
    );
    for (const line of found?.attempts ?? []) {
        output.appendLine(`[fi]   tried ${line}`);
    }
    const waitedMin = Math.round(waitSec / 60);
    const waited = waitedMin >= 1 ? `${waitedMin} min` : `${waitSec}s`;
    const tried = found?.attempts.length
        ? ` Tried ${found.attempts.length} endpoint(s) — details in the "Frontier Insight" output panel.`
        : "";
    const action = await vscode.window.showInformationMessage(
        `Frontier Insight: no Axon sidecar detected after waiting ${waited}.${tried} ` +
        `If Axon IS running somewhere else, point the extension at it with the ` +
        `\`frontierInsight.axonUrl\` setting.`,
        "Start in terminal",
        "Show output",
        "Show /axon-status",
        "Don't show again",
        "Dismiss",
    );
    if (action === "Don't show again") {
        // Persist the opt-out as a setting rather than hidden state, so
        // it's visible in the Settings UI and reversible without a
        // reinstall.
        await vscode.workspace
            .getConfiguration("frontierInsight")
            .update("axonStartupWaitSec", 0, vscode.ConfigurationTarget.Global);
    } else if (action === "Show output") {
        output.show(true);
    } else if (action === "Start in terminal") {
        // The one terminal this extension opens, and only when the person
        // clicks this button: Axon is a server that keeps running after the
        // chat turn ends (a chat command's child would be stopped with it),
        // and the person may want to watch or stop it there.
        const term = vscode.window.createTerminal({ name: "Axon" });
        term.show();
        term.sendText("python -m axon.api");
    } else if (action === "Show /axon-status") {
        await vscode.commands.executeCommand(
            "workbench.action.chat.open",
            { query: "@fi /axon-status" },
        );
    }
    void context;
}

/**
 * Chat-command handler for `@fi /axon-status`. Runs the same discovery
 * sweep as the Python ``axon_status()`` helper — every candidate
 * endpoint, first live one wins — and prints a short markdown verdict.
 *
 * On failure it lists the endpoints it tried. A wrong-port setup is the
 * overwhelmingly common cause of "not running", and the user can only
 * fix what they can see.
 */
async function runAxonStatus(stream: vscode.ChatResponseStream): Promise<void> {
    const found = await discoverAxon({ overrideUrl: axonUrlSetting() });
    if (!found.live) {
        const tried = found.attempts.length
            ? "Tried:\n\n" + found.attempts.map(a => `- \`${a}\``).join("\n") + "\n\n"
            : "";
        stream.markdown(
            `**Axon sidecar:** _not running_\n\n` + tried +
            `Start one, or point the extension at an existing one with the ` +
            `\`frontierInsight.axonUrl\` setting:\n\n` +
            `\`\`\`\npython -m axon.api\n\`\`\`\n`,
        );
        return;
    }
    const via = `_found via ${found.source}_`;
    if (!found.ready) {
        stream.markdown(
            `**Axon sidecar:** \`${found.baseUrl}\` — _running, brain still initializing_ (${via})\n\n` +
            `Give it a few seconds; the embedding model + vector index load on first request.\n`,
        );
        return;
    }
    stream.markdown(
        `**Axon sidecar:** \`${found.baseUrl}\` — _up + ready_ (${via})\n\n` +
        `The next \`@fi /start\` will reuse this warm process.\n`,
    );
}

/**
 * Implementation of `@fi /drafts`. Walks ``<workspaceRoot>/outputs/_drafts/``
 * for ``*.yaml`` files (proposal companions) and renders them as a
 * markdown table with a clickable ``/start <path>`` hint for each.
 * Mirrors `python launch.py --list-drafts` (CLI) and the
 * "Continue a draft" picker on /interview (Web). Three-interface
 * parity for the drafts-discovery feature.
 */
async function runListDrafts(
    stream: vscode.ChatResponseStream,
    token: vscode.CancellationToken,
): Promise<void> {
    if (token.isCancellationRequested) return;
    const fs = require("node:fs") as typeof import("node:fs");
    const pathMod = require("node:path") as typeof import("node:path");
    const ws = workFolderForCommand();
    if (!ws) {
        stream.markdown("Open a folder in VSCode first (the workspace root is where `outputs/_drafts/` lives).\n");
        return;
    }
    const draftsDir = pathMod.join(ws, "outputs", "_drafts");
    if (!fs.existsSync(draftsDir)) {
        stream.markdown(`No \`outputs/_drafts/\` directory yet. Run \`@fi /proposal "<topic>"\` to create one.\n`);
        return;
    }
    let entries: string[];
    try {
        entries = fs.readdirSync(draftsDir).filter((f) => f.endsWith(".yaml"));
    } catch (e: any) {
        stream.markdown(`Failed to read \`${draftsDir}\`: ${e?.message ?? String(e)}\n`);
        return;
    }
    if (entries.length === 0) {
        stream.markdown("No proposal drafts in `outputs/_drafts/` yet. Run `@fi /proposal \"<topic>\"` to create one.\n");
        return;
    }
    // Sort by mtime descending. Parse each YAML's `topic:` + `title:`
    // for preview — we keep it dependency-free (no js-yaml) by line
    // scanning, which is enough for the top-level scalar fields the
    // proposal generator writes.
    const items = entries
        .map((name) => {
            const full = pathMod.join(draftsDir, name);
            const stat = fs.statSync(full);
            let title = "";
            let topic = "";
            try {
                const text = fs.readFileSync(full, "utf-8");
                const titleMatch = text.match(/^title:\s*(.+)$/m);
                if (titleMatch) title = titleMatch[1].trim();
                // ``topic:`` is often a block scalar (``|``); grab the
                // next non-blank line of the body when so.
                const topicMatch = text.match(/^topic:\s*(?:\|[+-]?\s*)?\n((?:  .*\n)+)/m);
                if (topicMatch) {
                    topic = topicMatch[1].split("\n").map(l => l.trim()).filter(Boolean).join(" ").trim();
                } else {
                    const inline = text.match(/^topic:\s*(.+)$/m);
                    if (inline) topic = inline[1].trim();
                }
            } catch (_) {}
            return { name, full, mtime: stat.mtimeMs, title, topic };
        })
        .sort((a, b) => b.mtime - a.mtime)
        .slice(0, 40);

    stream.markdown(`📂 **${items.length} proposal draft(s) in \`outputs/_drafts/\`:**\n\n`);
    for (const it of items) {
        const when = new Date(it.mtime).toLocaleString();
        const titleShown = it.title || it.name.replace(/\.yaml$/, "");
        const topicShown = it.topic ? (it.topic.length > 120 ? it.topic.slice(0, 120) + "…" : it.topic) : "";
        stream.markdown(`### ${titleShown}\n`);
        stream.markdown(`*${when}*\n\n`);
        if (topicShown) stream.markdown(`> ${topicShown}\n\n`);
        stream.markdown(`Run: \`@fi /start ${it.full}\`\n\n`);
        stream.markdown(`---\n\n`);
    }
}

async function runProposal(
    promptArgs: string,
    stream: vscode.ChatResponseStream,
    token: vscode.CancellationToken,
    userPickedModel: vscode.LanguageModelChat,
): Promise<void> {
    if (token.isCancellationRequested) return;

    const topic = promptArgs.trim();
    if (!topic) {
        stream.markdown(
            "Need a topic. Example: `@fi /proposal Bell inequality violations under quantum-classical coupling`.\n\n" +
            "The proposal will be saved to `outputs/_drafts/` along with a companion YAML you can feed to `/start` once the plan looks good.\n",
        );
        return;
    }

    const cfg = vscode.workspace.getConfiguration("frontierInsight");
    const pythonPath = cfg.get<string>("pythonPath") || "python";
    const roots = await rootsForCommand();
    if ("error" in roots) {
        stream.markdown(roots.error);
        return;
    }
    const { repoPath, workDir } = roots;
    const launchScript = path.join(repoPath, "launch.py");
    const outputDirSetting = cfg.get<string>("outputDir") || "outputs";
    const outputsDir = path.isAbsolute(outputDirSetting)
        ? outputDirSetting
        : path.join(workDir, outputDirSetting);

    stream.markdown(
        `📝 Drafting proposal for topic:\n> ${topic.split("\n").slice(0, 3).join("\n> ")}\n\n` +
        `🤖 Model: \`${userPickedModel.family}\` (vendor: ${userPickedModel.vendor})\n\n` +
        `▶️ Asking the LLM for a planning doc…\n\n`,
    );

    const bridge = new Bridge({
        progress: stream,
        cancellationToken: token,
        defaultModel: userPickedModel,
    });
    const port = await bridge.listen();

    const argv: string[] = [
        "-u", launchScript,
        "--vscode-bridge-port", String(port),
        "--proposal", topic,
        "--output-root", outputsDir,
    ];

    const child = spawn(pythonPath, argv, {
        cwd: workDir,
        env: { ...process.env, PYTHONUNBUFFERED: "1", FI_SKIP_BOOTSTRAP: "1" },
        stdio: ["ignore", "pipe", "pipe"],
    });
    const stderrTail: string[] = [];
    const STDERR_TAIL_LINES = 80;
    bridge.attachChild(child, (line) => {
        stderrTail.push(line);
        if (stderrTail.length > STDERR_TAIL_LINES) {
            stderrTail.splice(0, stderrTail.length - STDERR_TAIL_LINES);
        }
    });
    token.onCancellationRequested(() => {
        try { child.kill("SIGTERM"); } catch { /* noop */ }
    });

    child.stdout.setEncoding("utf-8");
    let stdoutBuf = "";
    child.stdout.on("data", (chunk: string) => {
        stdoutBuf += chunk;
        const lines = stdoutBuf.split(/\r?\n/);
        stdoutBuf = lines.pop() || "";
        for (const line of lines) {
            const m = line.match(/^\[FI\] proposal -> (.+)$/);
            if (m) {
                stream.markdown(`  ✅ proposal written → \`${m[1]}\`\n\n`);
                continue;
            }
            const ym = line.match(/^\[FI\] companion YAML -> (.+)$/);
            if (ym) {
                stream.markdown(`  📄 companion YAML → \`${ym[1]}\`\n\n`);
                continue;
            }
            if (line.startsWith("[FI] To run the quest: ")) {
                stream.markdown(`  ▶️ ${line.slice(5)}\n\n`);
            }
        }
    });

    const result = await waitForChildExit(child, bridge, pythonPath, stream);
    if (result.kind === "spawn-error") return;
    const exitCode = result.code;

    if (exitCode === 0) {
        stream.markdown("\n✅ Proposal done. Read the markdown, edit either file if needed, then `/start` the YAML when ready.\n");
    } else {
        const tail = stderrTail.join("\n");
        stream.markdown(
            `\n❌ **Python exited with code ${exitCode}.**\n\n` +
            (tail.trim()
                ? "Stderr tail:\n\n```\n" + tail + "\n```\n"
                : "stderr was empty.\n"),
        );
    }
}


/**
 * Implementation of `@fi /analyze <path> <topic>`. The inverse of
 * `/proposal`: the user already has data in a directory and wants
 * FI to write a paper analyzing it. Spawns `python launch.py
 * --analyze <abs-data-path> --analyze-topic "<topic>"`. Argument
 * parsing: the first whitespace-delimited token is the data
 * directory path (single-quoted if it contains spaces), everything
 * after that is the topic.
 */
async function runAnalyze(
    promptArgs: string,
    stream: vscode.ChatResponseStream,
    token: vscode.CancellationToken,
    userPickedModel: vscode.LanguageModelChat,
): Promise<void> {
    if (token.isCancellationRequested) return;

    const trimmed = promptArgs.trim();
    if (!trimmed) {
        stream.markdown(
            "Need a data directory + topic. Examples:\n" +
            "- `@fi /analyze ./my_data Compare ridership trends across regions`\n" +
            "- `@fi /analyze \"C:/My Data\" Find common failure modes in these logs`\n\n" +
            "The directory's files are copied into the new quest's `data/` dir, " +
            "then the engine runs the no-simulation path: " +
            "`auto_collect_data → wait_for_data → data_load → analyze → write → review`.\n",
        );
        return;
    }

    // Split off the first token (path) — supports double-quoted paths
    // with spaces. Everything after that token is the topic.
    let dataPath: string;
    let topic: string;
    const quoted = trimmed.match(/^"([^"]+)"\s+(.+)$/s);
    if (quoted) {
        dataPath = quoted[1];
        topic = quoted[2].trim();
    } else {
        const idx = trimmed.search(/\s/);
        if (idx < 0) {
            stream.markdown(
                "Need BOTH a data directory AND a topic. " +
                "Example: `@fi /analyze ./my_data Compare ridership trends`.\n",
            );
            return;
        }
        dataPath = trimmed.slice(0, idx);
        topic = trimmed.slice(idx).trim();
    }
    if (!topic) {
        stream.markdown(
            "Need a topic after the data path. " +
            "Example: `@fi /analyze ./my_data Compare ridership trends`.\n",
        );
        return;
    }

    const cfg = vscode.workspace.getConfiguration("frontierInsight");
    const pythonPath = cfg.get<string>("pythonPath") || "python";
    const roots = await rootsForCommand(dataPath);
    if ("error" in roots) {
        stream.markdown(roots.error);
        return;
    }
    const { repoPath, workDir } = roots;
    const launchScript = path.join(repoPath, "launch.py");
    const outputDirSetting = cfg.get<string>("outputDir") || "outputs";
    const outputsDir = path.isAbsolute(outputDirSetting)
        ? outputDirSetting
        : path.join(workDir, outputDirSetting);
    // Resolve dataPath relative to the workspace if it's relative.
    const dataAbs = path.isAbsolute(dataPath)
        ? dataPath
        : path.join(workDir, dataPath);

    stream.markdown(
        `📊 Analyzing pre-staged data:\n` +
        `  - data dir: \`${dataAbs}\`\n` +
        `  - topic: ${topic.split("\n").slice(0, 3).join("\n    ")}\n\n` +
        `🤖 Model: \`${userPickedModel.family}\` (vendor: ${userPickedModel.vendor})\n\n` +
        `▶️ Staging files into the quest's data/ directory, then running the no-simulation graph…\n\n`,
    );

    const bridge = new Bridge({
        progress: stream,
        cancellationToken: token,
        defaultModel: userPickedModel,
    });
    const port = await bridge.listen();

    const argv: string[] = [
        "-u", launchScript,
        "--vscode-bridge-port", String(port),
        "--analyze", dataAbs,
        "--analyze-topic", topic,
        "--output-root", outputsDir,
    ];

    const child = spawn(pythonPath, argv, {
        cwd: workDir,
        env: { ...process.env, PYTHONUNBUFFERED: "1", FI_SKIP_BOOTSTRAP: "1" },
        stdio: ["ignore", "pipe", "pipe"],
    });
    const stderrTail: string[] = [];
    const STDERR_TAIL_LINES = 80;
    bridge.attachChild(child, (line) => {
        stderrTail.push(line);
        if (stderrTail.length > STDERR_TAIL_LINES) {
            stderrTail.splice(0, stderrTail.length - STDERR_TAIL_LINES);
        }
    });
    token.onCancellationRequested(() => {
        try { child.kill("SIGTERM"); } catch { /* noop */ }
    });

    child.stdout.setEncoding("utf-8");
    let stdoutBuf = "";
    child.stdout.on("data", (chunk: string) => {
        stdoutBuf += chunk;
        const lines = stdoutBuf.split(/\r?\n/);
        stdoutBuf = lines.pop() || "";
        for (const line of lines) {
            const m = line.match(/^\[FI\] --analyze: quest_id=(\S+)\s+files_staged=(\d+)/);
            if (m) {
                stream.markdown(
                    `  ✅ quest \`${m[1]}\` (staged ${m[2]} file${m[2] === "1" ? "" : "s"})\n\n`,
                );
                continue;
            }
            if (line.startsWith("[FI] start ")) {
                stream.markdown(`  ▶️ engine started\n`);
            }
        }
    });

    const result = await waitForChildExit(child, bridge, pythonPath, stream);
    if (result.kind === "spawn-error") return;
    const exitCode = result.code;

    if (exitCode === 0) {
        stream.markdown(
            "\n✅ Analyze quest finished. Paper + figures landed in `outputs/<quest_id>/`.\n",
        );
    } else {
        const tail = stderrTail.join("\n");
        stream.markdown(
            `\n❌ **Python exited with code ${exitCode}.**\n\n` +
            (tail.trim()
                ? "Stderr tail:\n\n```\n" + tail + "\n```\n"
                : "stderr was empty.\n"),
        );
    }
}
