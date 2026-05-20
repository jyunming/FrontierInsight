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
import * as fsPromises from "fs/promises";
import * as http from "http";
import * as net from "net";
import * as path from "path";
import { spawn } from "child_process";
import { Bridge } from "./bridge";
import { PersistentBridge } from "./persistent-bridge";
import { runInterview, writeInterviewYaml } from "./interview";


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

export function activate(context: vscode.ExtensionContext): void {
    const participant = vscode.chat.createChatParticipant(
        "frontier-insight.fi",
        async (request, _ctx, stream, token) => {
            await handleRequest(request, stream, token);
        },
    );
    participant.iconPath = new vscode.ThemeIcon("beaker");
    context.subscriptions.push(participant);

    // Probe the Axon sidecar on activation. We don't auto-launch from
    // the extension — VSCode users are expected to keep an Axon
    // process running in their workspace — but we surface a one-time
    // notification if it's down so the first /start of the session
    // doesn't pay the cold-init cost silently.
    void probeAxonOnActivate(context);

    // Start the session-long IPC bridge so a `python launch.py --serve`
    // (or any --tool subprocess) can route LLM calls through
    // ``vscode.lm.*`` without the user wiring a port. The bridge
    // listens on a per-user OS-managed socket / named pipe; see
    // ./bridge-path.ts for the address.
    const outputChannel = vscode.window.createOutputChannel("Frontier Insight");
    context.subscriptions.push(outputChannel);
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
    if (cmd === "resume") {
        await runResume(prompt, stream, token, userPickedModel);
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
        // Mid-quest re-entry. Routes to the same
        // `launch.py --update <quest_id>` flow the CLI uses.
        // ``prompt`` should be the quest_id; the Python side
        // validates it exists and refuses gracefully if not.
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
        const quotedArgs = paths.map(shellQuote).join(" ");
        await runTerminalCommand(
            "ingest", `--ingest ${quotedArgs}`,
            stream, "Opening a terminal to ingest into Axon.",
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
    if (cmd === "axon-status" || cmd === "axon") {
        await runAxonStatus(stream);
        return;
    }
    if (cmd === "install-tectonic" || cmd === "tectonic") {
        // No-admin LaTeX install for paper_pdf support.
        await runTerminalCommand(
            "tectonic", "--install-tectonic",
            stream, "Opening a terminal to install tectonic (~70 MB) " +
            "into tools/. Self-bootstrapping; no admin needed.",
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
        "- `@fi /resume <quest_id>` — resume that specific quest directly.",
        "- `@fi /summarize <folder>` — walk a folder of papers/code/notes/logs and produce a structured markdown summary; input files + summary land in Axon.",
        "- `@fi /digest [days]` — weekly project-manager digest: completed quests, in-progress, themes, diff vs prior digest, suggested next quests. Default window: 7 days. Lands at `<outputDir>/_digests/<YYYY-Www>.md` (where `<outputDir>` is the `frontierInsight.outputDir` setting, defaulting to `outputs/`).",
        "- `@fi /portfolio` — all-time cross-quest synthesis: topic clusters, near-duplicate detection, meta-paper candidates, coverage gaps, prioritized next-quest suggestions. Lands at `<outputDir>/_portfolio/<YYYY-MM-DD>.md`.",
        "- `@fi /critique <quest_id>` — adversarial second-pass review of a completed quest: methodology challenges, statistical issues, reproducibility gaps, alternative explanations. Lands at `<outputDir>/<quest_id>/critique.md`. For strongest effect, pick a Copilot model different from the one that wrote the paper.",
        "- `@fi /proposal <topic>` — pre-quest planning doc: background, hypothesis, plan, success criteria, risks, recommended next step. Writes both a markdown proposal and a companion YAML ready for `/start`. Lands at `<outputDir>/_drafts/<id>-proposal.md` + `<outputDir>/_drafts/<id>.yaml`.",
        "- `@fi /drafts` — list proposal drafts you've made (most-recent first) with a one-click `/start` command for each. Mirrors `python launch.py --list-drafts` and the web `/interview` drafts picker.",
        "- `@fi /axon-status` — check whether the Axon sidecar (`python -m axon.api` on `127.0.0.1:8000`) is reachable. CLI / web launches auto-start it; VSCode users keep their own. Use this to confirm the sidecar is hot before kicking off a quest.",
        "- `@fi /analyze <data-path> <topic>` — run a no-simulation quest on pre-staged data. Files under `<data-path>` are copied into the new quest's `data/` directory and the engine routes through `auto_collect_data → wait_for_data → data_load → analyze → write → review`. The inverse of `/proposal`: when you already have the dataset and just want a paper analyzing it.",
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
): Promise<void> {
    if (token.isCancellationRequested) return;

    const cfg = vscode.workspace.getConfiguration("frontierInsight");
    let repoPath = cfg.get<string>("repoPath") || "";
    if (!repoPath) {
        const ws = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
        if (!ws) {
            stream.markdown(
                "❌ No workspace open. Open the FrontierInsight folder, or set `frontierInsight.repoPath` in settings, then try again.",
            );
            return;
        }
        repoPath = ws;
    }

    // Resolve the outputs dir from settings. The `frontierInsight.outputDir`
    // setting may be a relative path (joined with repoPath) or absolute.
    // Defaults to "outputs". This must match where quests actually land —
    // otherwise the picker shows nothing for users who customized it.
    const outputDirSetting = cfg.get<string>("outputDir") || "outputs";
    const outputsDir = path.isAbsolute(outputDirSetting)
        ? outputDirSetting
        : path.join(repoPath, outputDirSetting);
    if (!(await fsExists(outputsDir))) {
        stream.markdown(
            `❌ No outputs directory at \`${outputsDir}\` — nothing to resume. ` +
            `(Override via the \`frontierInsight.outputDir\` setting.)`,
        );
        return;
    }

    // Find all quest dirs with a checkpoint. Async I/O so the extension
    // host event loop stays responsive on slow filesystems / large dirs.
    type Candidate = { questId: string; questDir: string; mtimeMs: number };
    const entries = await fsPromises.readdir(outputsDir, { withFileTypes: true });
    const candidates: Candidate[] = [];
    await Promise.all(entries.map(async (entry) => {
        if (!entry.isDirectory() || entry.name.startsWith("_")) return;
        const questDir = path.join(outputsDir, entry.name);
        const checkpoint = path.join(questDir, ".fi", "state.sqlite");
        try {
            const stat = await fsPromises.stat(checkpoint);
            if (!stat.isFile()) return;
            candidates.push({
                questId: entry.name, questDir, mtimeMs: stat.mtimeMs,
            });
        } catch {
            // Missing checkpoint or unreadable file — skip silently.
        }
    }));
    if (candidates.length === 0) {
        stream.markdown(
            `❌ No quests with a \`.fi/state.sqlite\` checkpoint under \`${outputsDir}\`. Run \`@fi /new\` to start one.`,
        );
        return;
    }
    candidates.sort((a, b) => b.mtimeMs - a.mtimeMs);

    // Handle multi-token / typo-quoted args. The chat surface can pass
    // through copy/paste artifacts like `/resume "1778…-x" extra` —
    // pick just the first whitespace-separated token so the lookup is
    // deterministic instead of silently failing with a confusing
    // "no quest with id '"178…-x" extra'" message.
    const rawArg = promptArgs.trim();
    const firstToken = rawArg.split(/\s+/)[0] || "";
    // Also strip surrounding quotes a user might paste from a log line.
    const sanitized = firstToken.replace(/^["']+|["']+$/g, "");
    let chosenId = sanitized;
    if (!chosenId) {
        const picks = candidates.map((c) => ({
            label: `$(beaker) ${c.questId}`,
            description: new Date(c.mtimeMs).toLocaleString(),
            questId: c.questId,
        }));
        const picked = await vscode.window.showQuickPick(picks, {
            placeHolder: "Pick a quest to resume (most recent first)",
            matchOnDescription: true,
        });
        if (!picked) return;   // user hit Esc
        chosenId = picked.questId;
    } else {
        // User passed an id directly — validate it has a checkpoint.
        if (!candidates.find((c) => c.questId === chosenId)) {
            stream.markdown(
                `❌ No quest dir with id \`${chosenId}\` under \`${outputsDir}\`, ` +
                `or it has no \`.fi/state.sqlite\` checkpoint.`,
            );
            return;
        }
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
    const inQuestYaml = path.join(outputsDir, chosenId, "config.yaml");
    if (await fsExists(inQuestYaml)) {
        yamlPath = inQuestYaml;
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

    const relYaml = path.relative(repoPath, yamlPath).split(path.sep).join("/");
    stream.markdown(
        `🔁 Resuming quest \`${chosenId}\`\n\n` +
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
    );
}

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

function shellQuote(arg: string): string {
    // Cross-shell-safe quoting. On Windows the integrated terminal
    // is usually PowerShell or cmd.exe; on POSIX it's bash/zsh.
    // Wrapping in double quotes + escaping internal `"`, `$`, `` ` ``,
    // and `\` works for all three. This is conservative but correct.
    if (process.platform === "win32") {
        // PowerShell: backtick + ` for embedded quotes; cmd: ""
        // We pick PowerShell-compatible quoting which is also tolerated
        // by cmd. Escape internal " as `".
        return `"${arg.replace(/`/g, "``").replace(/"/g, "`\"").replace(/\$/g, "`$")}"`;
    }
    // POSIX: single quotes if there's no ' in arg; else fall back to
    // double-quoted with backslash escapes.
    if (!arg.includes("'")) return `'${arg}'`;
    return `"${arg.replace(/\\/g, "\\\\").replace(/"/g, "\\\"").replace(/\$/g, "\\$").replace(/`/g, "\\`")}"`;
}

async function runTerminalCommand(
    label: string,
    pythonArgs: string,
    stream: vscode.ChatResponseStream,
    hint: string,
): Promise<void> {
    // Shared helper for /ingest, /install-tectonic, and any other
    // future CLI command that's easier to expose as a terminal
    // pass-through than as a fully-rendered chat UI. Opens an
    // integrated terminal in the FI repo root and runs the
    // command; the user sees the live output there.
    const cfg = vscode.workspace.getConfiguration("frontierInsight");
    const pythonPath = cfg.get<string>("pythonPath") || "python";
    let repoPath = cfg.get<string>("repoPath") || "";
    if (!repoPath) {
        const ws = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
        if (!ws) {
            stream.markdown(
                "❌ No workspace open. Open the FrontierInsight folder, " +
                "or set `frontierInsight.repoPath` in settings, then try again.",
            );
            return;
        }
        repoPath = ws;
    }
    stream.markdown(`${hint}\n\n`);
    const term = vscode.window.createTerminal({
        name: `FI ${label}`,
        cwd: repoPath,
    });
    term.show();
    // Shell-quote the Python path too — the configured value can
    // include spaces (typical Windows install path:
    // "C:/Program Files/Python311/python.exe").
    term.sendText(`${shellQuote(pythonPath)} launch.py ${pythonArgs}`);
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
    let repoPath = cfg.get<string>("repoPath") || "";
    if (!repoPath) {
        const ws = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
        if (!ws) {
            stream.markdown(
                "❌ No workspace open. Open the FrontierInsight folder, or set `frontierInsight.repoPath` in settings, then try again.",
            );
            return;
        }
        repoPath = ws;
    }

    const outputDirSetting = cfg.get<string>("outputDir") || "outputs";
    const outputsDir = path.isAbsolute(outputDirSetting)
        ? outputDirSetting
        : path.join(repoPath, outputDirSetting);

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

    // The Python --update flow is stdin-interactive. The chat
    // surface can't render Q&A modals back into the Python process,
    // so we open a VSCode integrated terminal and run the command
    // there — the user types answers in the terminal, then resume
    // happens automatically. Same questions, same Python schema,
    // just a different transport for the prompts.
    stream.markdown(
        `🔧 Opening an integrated terminal to run the update interview for \`${questId}\`. ` +
        `Answer in the terminal; the quest auto-resumes when the interview completes.\n\n`,
    );
    const term = vscode.window.createTerminal({
        name: `FI update: ${questId}`,
        cwd: repoPath,
    });
    term.show();
    // Quote arguments defensively in case the shell mangles the
    // quest_id (it shouldn't — quest_ids are slugified — but be
    // safe). The user's shell determines the quoting style; both
    // PowerShell and bash accept double-quotes.
    term.sendText(`${pythonPath} launch.py --update "${questId}"`);
}


async function runInterviewAndQuest(
    stream: vscode.ChatResponseStream,
    token: vscode.CancellationToken,
    userPickedModel: vscode.LanguageModelChat,
): Promise<void> {
    if (token.isCancellationRequested) return;

    const answers = await runInterview(stream);
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
    let repoPath = cfg.get<string>("repoPath") || "";
    if (!repoPath) {
        const ws = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
        if (!ws) {
            stream.markdown(
                "❌ No workspace open. Open the FrontierInsight folder, or set `frontierInsight.repoPath` in settings, then try again.",
            );
            return;
        }
        repoPath = ws;
    }

    // `writeInterviewYaml` does sync mkdir + writeFile; both can throw
    // on EACCES, ENOSPC, or a OneDrive sync lock. Without this catch,
    // the chat handler unwinds and the user sees the interview "finish"
    // with no quest and no error.
    let yamlPath: string;
    try {
        yamlPath = writeInterviewYaml(answers, repoPath);
    } catch (err) {
        const draftDir = path.join(repoPath, "outputs", "_drafts");
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
    const rel = path.relative(repoPath, yamlPath).split(path.sep).join("/");
    stream.markdown(`📝 Wrote config: \`${rel}\`\n\n`);
    // Surface which model the user's calls will route through so the
    // budget impact is visible upfront.
    stream.markdown(
        `🤖 Model: \`${userPickedModel.family}\` (vendor: ${userPickedModel.vendor})\n\n`,
    );
    stream.markdown(`▶️ Starting quest…\n\n`);

    // Hand off to the existing runQuest path. We pass the workspace-
    // relative path so the spawned Python's cwd resolves correctly.
    await runQuest(rel, /*fleet*/ false, stream, token, userPickedModel);
}

async function runQuest(
    promptArgs: string,
    fleet: boolean,
    stream: vscode.ChatResponseStream,
    token: vscode.CancellationToken,
    userPickedModel: vscode.LanguageModelChat,
    resumeQuestId?: string,
): Promise<void> {
    const paths = promptArgs.split(/\s+/).filter((s) => s.length > 0);
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
    let repoPath = cfg.get<string>("repoPath") || "";
    if (!repoPath) {
        const ws = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
        if (!ws) {
            stream.markdown(
                "No workspace open. Open the FrontierInsight repo folder, or set `frontierInsight.repoPath` in settings.",
            );
            return;
        }
        repoPath = ws;
    }
    const launchScript = path.join(repoPath, "launch.py");

    stream.markdown(`🧪 Starting ${fleet ? "fleet" : "quest"}: \`${paths.join(", ")}\`\n\n`);

    // 1. Bind the bridge to a free port. We thread `userPickedModel`
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

    // 2. Build argv. The --vscode-bridge-port flag forces FI to use
    // the vscode_extension provider regardless of what each YAML's
    // `provider` block says — we route every LLM call back through
    // this bridge.
    const argv: string[] = ["-u", launchScript, "--vscode-bridge-port", String(port)];
    if (fleet) {
        argv.push("--fleet", ...paths);
    } else {
        argv.push("--config", paths[0]);
        if (resumeQuestId) {
            argv.push("--resume", resumeQuestId);
        }
    }

    // 3. Spawn Python.
    const child = spawn(pythonPath, argv, {
        cwd: repoPath,
        env: { ...process.env, PYTHONUNBUFFERED: "1" },
        stdio: ["ignore", "pipe", "pipe"],
    });
    // Keep a rolling tail of stderr so we can surface the actual
    // traceback in the chat if Python exits non-zero. Without this,
    // the user only sees "exited with code 1, check run.log" — but
    // unhandled exceptions don't reach the run.log (it only carries
    // what `logging.info(...)` etc. emitted before the crash).
    const stderrTail: string[] = [];
    const STDERR_TAIL_LINES = 80;
    bridge.attachChild(child, (line) => {
        stderrTail.push(line);
        if (stderrTail.length > STDERR_TAIL_LINES) {
            stderrTail.splice(0, stderrTail.length - STDERR_TAIL_LINES);
        }
    });

    // Make sure cancellation kills the child + closes the bridge.
    token.onCancellationRequested(() => {
        try { child.kill("SIGTERM"); } catch { /* noop */ }
    });

    // Surface only the user-meaningful end-of-run lines from Python's
    // stdout. The `[FI] start/resume quest_id=...` echo is already
    // shown via the extension's own header lines, so dropping it
    // avoids the duplicate-print problem the user reported. Generator-
    // output lines like `[FI] wrote paper_md -> <abs path>` ARE useful
    // (they're the final artifact pointers) so we keep those but
    // re-render with just a basename + a checkmark instead of the
    // raw `[FI]` prefix.
    child.stdout.setEncoding("utf-8");
    let stdoutBuf = "";
    child.stdout.on("data", (chunk: string) => {
        stdoutBuf += chunk;
        const lines = stdoutBuf.split(/\r?\n/);
        stdoutBuf = lines.pop() || "";
        for (const line of lines) {
            const wrote = line.match(/^\[FI\] wrote (\w+) -> (.+)$/);
            if (wrote) {
                stream.markdown(`  ✅ wrote ${wrote[1]} → \`${path.basename(wrote[2])}\`\n\n`);
                continue;
            }
            const summary = line.match(/^\[FI\] summary -> (.+)$/);
            if (summary) {
                stream.markdown(`  📋 summary → \`${path.basename(summary[1])}\`\n\n`);
                continue;
            }
            // Drop other [FI] lines (start/resume quest_id=, paths the
            // user already saw in our header, etc.) — they're noise here.
        }
    });

    // 4. Wait for Python to exit. Bridge messages flow through the
    // socket on its own; the chat handler resolves when the child
    // terminates, which is when the quest (or fleet) is done.
    const result = await waitForChildExit(child, bridge, pythonPath, stream);
    if (result.kind === "spawn-error") return;
    const exitCode = result.code;

    if (exitCode === 0) {
        stream.markdown(`\n✅ ${fleet ? "Fleet" : "Quest"} finished cleanly.`);
    } else {
        const tail = stderrTail.join("\n");
        stream.markdown(
            `\n❌ **Python exited with code ${exitCode}.**\n\n` +
            (tail.trim()
                ? "Last lines of stderr (the actual error usually lives here, **not** in `run.log` — unhandled exceptions skip the logger):\n\n" +
                  "```\n" + tail + "\n```\n"
                : "stderr was empty. Check `outputs/<quest_id>/.fi/run.log` for whatever made it to the logger before the crash.\n"),
        );
    }
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
    let repoPath = cfg.get<string>("repoPath") || "";
    if (!repoPath) {
        const ws = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
        if (!ws) {
            stream.markdown(
                "❌ No workspace open. Open the FrontierInsight folder, " +
                "or set `frontierInsight.repoPath` in settings.",
            );
            return;
        }
        repoPath = ws;
    }
    const launchScript = path.join(repoPath, "launch.py");

    // Resolve the folder argument: absolute paths are honored as-is;
    // relative paths resolve against the workspace folder. This lets
    // a user type `@fi /summarize ./papers` from the chat panel.
    const folderAbs = path.isAbsolute(folderArg)
        ? folderArg
        : path.resolve(repoPath, folderArg);
    let folderStat: import("fs").Stats;
    try {
        folderStat = await fsPromises.stat(folderAbs);
    } catch {
        stream.markdown(
            `❌ Path not found: \`${folderAbs}\`. ` +
            `(Resolved from \`${folderArg}\` against \`${repoPath}\`.)\n`,
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
        cwd: repoPath,
        env: { ...process.env, PYTHONUNBUFFERED: "1" },
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
    let repoPath = cfg.get<string>("repoPath") || "";
    if (!repoPath) {
        const ws = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
        if (!ws) {
            stream.markdown(
                "❌ No workspace open. Open the FrontierInsight folder, " +
                "or set `frontierInsight.repoPath` in settings.",
            );
            return;
        }
        repoPath = ws;
    }
    const launchScript = path.join(repoPath, "launch.py");
    const outputDirSetting = cfg.get<string>("outputDir") || "outputs";
    const outputsDir = path.isAbsolute(outputDirSetting)
        ? outputDirSetting
        : path.join(repoPath, outputDirSetting);

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
        cwd: repoPath,
        env: { ...process.env, PYTHONUNBUFFERED: "1" },
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
    let repoPath = cfg.get<string>("repoPath") || "";
    if (!repoPath) {
        const ws = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
        if (!ws) {
            stream.markdown(
                "❌ No workspace open. Open the FrontierInsight folder, " +
                "or set `frontierInsight.repoPath` in settings.",
            );
            return;
        }
        repoPath = ws;
    }
    const launchScript = path.join(repoPath, "launch.py");
    const outputDirSetting = cfg.get<string>("outputDir") || "outputs";
    const outputsDir = path.isAbsolute(outputDirSetting)
        ? outputDirSetting
        : path.join(repoPath, outputDirSetting);

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
        cwd: repoPath,
        env: { ...process.env, PYTHONUNBUFFERED: "1" },
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
    let repoPath = cfg.get<string>("repoPath") || "";
    if (!repoPath) {
        const ws = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
        if (!ws) {
            stream.markdown(
                "❌ No workspace open. Open the FrontierInsight folder, " +
                "or set `frontierInsight.repoPath` in settings.",
            );
            return;
        }
        repoPath = ws;
    }
    const launchScript = path.join(repoPath, "launch.py");
    const outputDirSetting = cfg.get<string>("outputDir") || "outputs";
    const outputsDir = path.isAbsolute(outputDirSetting)
        ? outputDirSetting
        : path.join(repoPath, outputDirSetting);

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
        cwd: repoPath,
        env: { ...process.env, PYTHONUNBUFFERED: "1" },
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
 * Probe the Axon sidecar (``python -m axon.api`` on ``127.0.0.1:8000``)
 * once when the extension activates. If it's not reachable, show an
 * info notification with a one-click "Open instructions" action. We
 * deliberately do NOT auto-launch the sidecar from the extension —
 * users running FI from VSCode are expected to keep an Axon process
 * around themselves (or just live with the cold-init cost on the
 * first quest). The check is informational only.
 *
 * Mirrors `core.axon_sidecar.axon_status` (CLI/web auto-launch path).
 * Three-interface parity for Axon health visibility.
 */
async function probeAxonOnActivate(
    context: vscode.ExtensionContext,
): Promise<void> {
    const host = process.env.AXON_HOST || "127.0.0.1";
    const port = process.env.AXON_PORT || "8000";

    // Activation runs on `onStartupFinished`, which often fires
    // BEFORE the user's Axon sidecar finishes its cold-init (the
    // sentence-transformers model load is the slow step — typically
    // 5-15 s on a fresh launch). A one-shot probe right at activate
    // racing the sidecar produces false-negative "Axon not detected"
    // notifications. Poll until either Axon answers or a wall-clock
    // deadline elapses, so a slow sidecar isn't mistaken for an
    // absent one.
    //
    // Per-probe timeout is 1s (vs. 5s default) so each failed attempt
    // returns fast and the deadline is honored. Worst-case-per-attempt
    // is now ~2s (HTTP timeout + sibling TCP timeout) instead of ~10s,
    // and the loop checks the elapsed deadline before sleeping or
    // probing again — so the ~30s budget is real wall-clock, not just
    // attempt count.
    const DEADLINE_MS = 30000;
    const PER_PROBE_TIMEOUT_MS = 1000;
    const INTERVAL_MS = 2000;
    const startedAt = Date.now();
    let status: AxonHealthResult = { live: false, ready: false };
    let attempts = 0;
    while (Date.now() - startedAt < DEADLINE_MS) {
        attempts++;
        status = await probeAxonHealth(host, port, PER_PROBE_TIMEOUT_MS);
        if (status.live) return;
        const elapsed = Date.now() - startedAt;
        const remaining = DEADLINE_MS - elapsed;
        if (remaining <= 0) break;
        await new Promise(r => setTimeout(r, Math.min(INTERVAL_MS, remaining)));
    }

    // Deadline reached — sidecar genuinely isn't reachable.
    const totalElapsedSec = ((Date.now() - startedAt) / 1000).toFixed(1);
    console.warn(
        `[fi] axon probe failed at http://${host}:${port}/health/live ` +
        `after ${attempts} attempts over ${totalElapsedSec}s: ${status.error}`,
    );
    const detail = status.error ? ` (probe error: ${status.error})` : "";
    const action = await vscode.window.showInformationMessage(
        `Frontier Insight: Axon sidecar not detected at ${host}:${port}${detail}. ` +
        `If Axon IS running, it may be on a different host/port — set the ` +
        `AXON_HOST / AXON_PORT environment variables before launching VS Code.`,
        "Start in terminal",
        "Show /axon-status",
        "Dismiss",
    );
    if (action === "Start in terminal") {
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
 * Single-source probe of the Axon sidecar's two health endpoints.
 * Returns `{live, ready, error?}`. ``error`` is exposed so callers
 * can show the user WHY the probe failed instead of "we couldn't
 * tell."
 *
 * Uses Node's built-in ``http.request`` rather than the global
 * ``fetch`` (undici). VSCode users repeatedly hit
 * ``TypeError: fetch failed`` against a localhost Axon that curl
 * could talk to fine — the opaque undici error hides the real cause
 * (corporate proxy env vars, EDR injection, stale agent connection
 * pool, IPv6 misroute). ``http.request`` has no global agent
 * keepalive pool, surfaces ECONNREFUSED / ETIMEDOUT / etc. directly,
 * and accepts ``family: 4`` to force IPv4 so a misconfigured ``::1``
 * resolution can't quietly break the probe.
 *
 * Each HTTP check is paired with a raw TCP connect against the same
 * host:port. If TCP succeeds but HTTP fails, the bug is above the
 * socket layer (proxy / EDR / HTTP-protocol mismatch) — surface
 * that in the error string so the user knows where to look.
 *
 * 5 s timeout per probe.
 */
interface AxonHealthResult {
    live: boolean;
    ready: boolean;
    error?: string;
}

function tcpConnectCheck(host: string, port: number, timeoutMs: number): Promise<{ ok: boolean; error?: string }> {
    return new Promise(resolve => {
        const sock = net.connect({ host, port, family: 4 });
        const timer = setTimeout(() => {
            sock.destroy();
            resolve({ ok: false, error: `TCP timeout after ${timeoutMs}ms` });
        }, timeoutMs);
        sock.once("connect", () => {
            clearTimeout(timer);
            sock.end();
            resolve({ ok: true });
        });
        sock.once("error", err => {
            clearTimeout(timer);
            const code = (err as NodeJS.ErrnoException).code;
            resolve({ ok: false, error: code ? `TCP ${code}` : `TCP error: ${err.message}` });
        });
    });
}

function httpProbe(host: string, port: number, urlPath: string, timeoutMs: number): Promise<{ ok: boolean; error?: string; status?: number }> {
    return new Promise(resolve => {
        const req = http.request(
            {
                host, port, path: urlPath, method: "GET",
                family: 4,
                // Use a one-shot agent so we don't share undici's
                // poisoned global keepalive pool from any earlier
                // failure window.
                agent: new http.Agent({ keepAlive: false }),
                timeout: timeoutMs,
            },
            res => {
                // Drain so the socket can close.
                res.resume();
                const ok = (res.statusCode ?? 0) >= 200 && (res.statusCode ?? 0) < 300;
                resolve({ ok, status: res.statusCode, error: ok ? undefined : `HTTP ${res.statusCode}` });
            },
        );
        req.on("timeout", () => {
            req.destroy();
            resolve({ ok: false, error: `HTTP timeout after ${timeoutMs}ms` });
        });
        req.on("error", err => {
            const code = (err as NodeJS.ErrnoException).code;
            resolve({ ok: false, error: code ? `HTTP ${code}` : `HTTP error: ${err.message}` });
        });
        req.end();
    });
}

async function probeAxonHealth(
    host: string, port: string, timeoutMs: number = 5000,
): Promise<AxonHealthResult> {
    const portN = Number(port);
    const TIMEOUT = timeoutMs;

    const probe = async (urlPath: string): Promise<{ ok: boolean; error?: string }> => {
        const httpRes = await httpProbe(host, portN, urlPath, TIMEOUT);
        if (httpRes.ok) return { ok: true };

        // HTTP failed. Do a sibling TCP probe so the user can tell
        // socket-level failure (axon not listening) apart from
        // above-socket failure (proxy / EDR / HTTP mismatch).
        const tcpRes = await tcpConnectCheck(host, portN, TIMEOUT);
        if (tcpRes.ok) {
            return {
                ok: false,
                error: `${httpRes.error} (TCP connect succeeded — HTTP request blocked; check HTTP_PROXY / antivirus / EDR)`,
            };
        }
        return { ok: false, error: `${httpRes.error}; ${tcpRes.error}` };
    };

    const live = await probe("/health/live");
    if (!live.ok) {
        return { live: false, ready: false, error: live.error };
    }
    const ready = await probe("/health/ready");
    return { live: true, ready: ready.ok };
}

/**
 * Chat-command handler for `@fi /axon-status`. Fetches the same
 * ``/health/live`` + ``/health/ready`` endpoints as the Python
 * ``axon_status()`` helper and prints a short markdown verdict.
 */
async function runAxonStatus(stream: vscode.ChatResponseStream): Promise<void> {
    const host = process.env.AXON_HOST || "127.0.0.1";
    const port = process.env.AXON_PORT || "8000";
    const base = `http://${host}:${port}`;
    const status = await probeAxonHealth(host, port);
    if (!status.live) {
        stream.markdown(
            `**Axon sidecar:** \`${base}\` — _not running_\n\n` +
            (status.error ? `_Probe error:_ \`${status.error}\`\n\n` : "") +
            `If Axon IS running, the probe may be hitting the wrong host/port. ` +
            `Set \`AXON_HOST\` and \`AXON_PORT\` env vars before launching VS Code, ` +
            `or start it on the default host:port:\n\n` +
            `\`\`\`\npython -m axon.api\n\`\`\`\n`,
        );
        return;
    }
    if (!status.ready) {
        stream.markdown(
            `**Axon sidecar:** \`${base}\` — _running, brain still initializing_\n\n` +
            `Give it a few seconds; the embedding model + vector index load on first request.\n`,
        );
        return;
    }
    stream.markdown(
        `**Axon sidecar:** \`${base}\` — _up + ready_\n\n` +
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
    const ws = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
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
    let repoPath = cfg.get<string>("repoPath") || "";
    if (!repoPath) {
        const ws = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
        if (!ws) {
            stream.markdown(
                "❌ No workspace open. Open the FrontierInsight folder, " +
                "or set `frontierInsight.repoPath` in settings.",
            );
            return;
        }
        repoPath = ws;
    }
    const launchScript = path.join(repoPath, "launch.py");
    const outputDirSetting = cfg.get<string>("outputDir") || "outputs";
    const outputsDir = path.isAbsolute(outputDirSetting)
        ? outputDirSetting
        : path.join(repoPath, outputDirSetting);

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
        cwd: repoPath,
        env: { ...process.env, PYTHONUNBUFFERED: "1" },
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
    let repoPath = cfg.get<string>("repoPath") || "";
    if (!repoPath) {
        const ws = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
        if (!ws) {
            stream.markdown(
                "❌ No workspace open. Open the FrontierInsight folder, " +
                "or set `frontierInsight.repoPath` in settings.",
            );
            return;
        }
        repoPath = ws;
    }
    const launchScript = path.join(repoPath, "launch.py");
    const outputDirSetting = cfg.get<string>("outputDir") || "outputs";
    const outputsDir = path.isAbsolute(outputDirSetting)
        ? outputDirSetting
        : path.join(repoPath, outputDirSetting);
    // Resolve dataPath relative to the workspace if it's relative.
    const dataAbs = path.isAbsolute(dataPath)
        ? dataPath
        : path.join(repoPath, dataPath);

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
        cwd: repoPath,
        env: { ...process.env, PYTHONUNBUFFERED: "1" },
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
