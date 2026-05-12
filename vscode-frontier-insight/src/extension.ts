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
import * as path from "path";
import { spawn } from "child_process";
import { Bridge } from "./bridge";
import { runInterview, writeInterviewYaml } from "./interview";

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
        "",
        "All LLM calls go through your Copilot subscription via the",
        "`vscode.lm` Language Model API. Each quest's `provider.node_models`",
        "is honored, so different nodes (and different reviewer-panel",
        "personas) can use different Copilot models within one run.",
    ].join("\n");
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

    // Surface any Python crash (couldn't import, syntax error, etc.)
    // by reading stdout. The engine itself logs to stderr; stdout is
    // typically empty during a healthy run.
    child.stdout.setEncoding("utf-8");
    let stdoutBuf = "";
    child.stdout.on("data", (chunk: string) => {
        stdoutBuf += chunk;
        const lines = stdoutBuf.split(/\r?\n/);
        stdoutBuf = lines.pop() || "";
        for (const line of lines) {
            if (line.startsWith("[FI]")) {
                stream.markdown(`  ${line}\n\n`);
            }
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
