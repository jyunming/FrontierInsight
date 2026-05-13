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
import * as path from "path";
import { spawn } from "child_process";
import { Bridge } from "./bridge";
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

export function activate(context: vscode.ExtensionContext): void {
    const participant = vscode.chat.createChatParticipant(
        "frontier-insight.fi",
        async (request, _ctx, stream, token) => {
            await handleRequest(request, stream, token);
        },
    );
    participant.iconPath = new vscode.ThemeIcon("beaker");
    context.subscriptions.push(participant);
}

export function deactivate(): void {
    // Nothing to clean up — each chat turn manages its own resources.
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

async function runInterviewAndQuest(
    stream: vscode.ChatResponseStream,
    token: vscode.CancellationToken,
    userPickedModel: vscode.LanguageModelChat,
): Promise<void> {
    if (token.isCancellationRequested) return;

    const answers = await runInterview(stream);
    if (!answers) return;  // user hit Esc somewhere

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

    const yamlPath = writeInterviewYaml(answers, repoPath);
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
    const exitCode: number | null = await new Promise((resolve) => {
        child.on("close", (code) => resolve(code));
    });
    await bridge.close();

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

    const exitCode: number | null = await new Promise((resolve) => {
        child.on("close", (code) => resolve(code));
    });
    await bridge.close();

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
