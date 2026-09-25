/**
 * `@fi /trace <quest_id> [--node <name>] [--detail summary|checks|debug]` — the quest's
 * audit trace (`core/audit_log.py`) in chat.
 *
 * This shells out to the same `launch.py --trace` the CLI uses and shows its output
 * verbatim; it does not re-parse or re-render the hash-chained log in TypeScript. All
 * three surfaces (CLI, web, VSCode) read the one thing `audit_log.py` decides a trace
 * line says, so they cannot drift from each other.
 */
import { spawn } from "child_process";
import * as path from "path";
import * as vscode from "vscode";
import { rootsFromConfig } from "./roots-config";
import { parseTraceArgs } from "./trace-args";

interface RunResult {
    code: number;
    stdout: string;
    stderr: string;
}

/** How much of the trace's own text is shown before it is cut, from the end (the newest events). */
const OUTPUT_CHARS = 6000;

function runLaunch(python: string, repo: string, workDir: string, args: string[]): Promise<RunResult> {
    return new Promise((resolve) => {
        const child = spawn(python, [path.join(repo, "launch.py"), ...args], {
            cwd: workDir,
            env: { ...process.env, PYTHONIOENCODING: "utf-8", FI_SKIP_BOOTSTRAP: "1" },
        });
        // utf8 encoding on the stream itself (not per-chunk d.toString()) so a multibyte
        // character split across two chunks is buffered and decoded whole, not mangled.
        child.stdout?.setEncoding("utf8");
        child.stderr?.setEncoding("utf8");
        let stdout = "";
        let stderr = "";
        child.stdout?.on("data", (d) => (stdout += d));
        child.stderr?.on("data", (d) => (stderr += d));
        child.on("error", (e) => resolve({ code: -1, stdout, stderr: String(e) }));
        child.on("close", (code) => resolve({ code: code ?? -1, stdout, stderr }));
    });
}

export async function runTrace(
    promptArgs: string,
    stream: vscode.ChatResponseStream,
    token: vscode.CancellationToken,
): Promise<void> {
    const parsed = parseTraceArgs(promptArgs.trim());
    if (parsed.error) {
        stream.markdown(`${parsed.error}\n`);
        return;
    }
    if (!parsed.questId) {
        stream.markdown(
            "Which quest? Example: `@fi /trace 1790003131-my-quest`, " +
                "add `--node design` for one step or `--detail summary` for less.\n",
        );
        return;
    }
    const cfg = vscode.workspace.getConfiguration("frontierInsight");
    const roots = rootsFromConfig(cfg);
    if ("error" in roots) {
        stream.markdown(roots.error + "\n");
        return;
    }
    const python = cfg.get<string>("pythonPath") || "python";
    const outputDirSetting = cfg.get<string>("outputDir") || "outputs";
    const outputRoot = path.isAbsolute(outputDirSetting)
        ? outputDirSetting
        : path.join(roots.workDir, outputDirSetting);

    stream.progress(`Reading the trace of ${parsed.questId}…`);
    const args = ["--trace", parsed.questId, "--trace-detail", parsed.detail, "--output-root", outputRoot];
    if (parsed.node) args.push("--trace-node", parsed.node);
    if (token.isCancellationRequested) return;

    const res = await runLaunch(python, roots.repoPath, roots.workDir, args);
    // _show_trace prints everything (including "CHAIN BROKEN") to stdout, so a clean run's
    // noise-free stdout is all that's shown. On a nonzero exit, an uncaught crash elsewhere in
    // launch.py could still land its traceback on stderr instead — show both then, rather than
    // silently dropping whichever one stdout happened to be non-empty over.
    const text = (res.code === 0 ? res.stdout : [res.stdout, res.stderr].filter((s) => s.trim()).join("\n")).trim();
    if (!text) {
        stream.markdown(`No output (exit ${res.code}).\n`);
        return;
    }
    const shown = text.length > OUTPUT_CHARS ? text.slice(-OUTPUT_CHARS) : text;
    stream.markdown(
        "```\n" + (shown === text ? shown : `… (${text.length - shown.length} earlier characters cut)\n${shown}`) + "\n```\n",
    );
    // Exit 1 also covers "no such quest" and "no trace file" (launch.py's own text says which);
    // only a broken hash chain gets this extra line, so it isn't shown for the other two.
    if (res.code !== 0 && text.includes("CHAIN BROKEN")) {
        stream.markdown("\n⚠️ The hash chain check failed — see the line above for where.\n");
    }
}


interface LaunchContext {
    python: string;
    repo: string;
    workDir: string;
    outputRoot: string;
}

function launchContext(stream: vscode.ChatResponseStream): LaunchContext | undefined {
    const cfg = vscode.workspace.getConfiguration("frontierInsight");
    const roots = rootsFromConfig(cfg);
    if ("error" in roots) {
        stream.markdown(roots.error + "\n");
        return undefined;
    }
    const outputDirSetting = cfg.get<string>("outputDir") || "outputs";
    return {
        python: cfg.get<string>("pythonPath") || "python",
        repo: roots.repoPath,
        workDir: roots.workDir,
        outputRoot: path.isAbsolute(outputDirSetting) ? outputDirSetting : path.join(roots.workDir, outputDirSetting),
    };
}

/**
 * `@fi /why <quest_id> [stop|review|evidence|<step>]` (or `--node <step>`) — why the quest did what it did, from what it
 * recorded (core/why.py). The same answer as `python launch.py --why` and the web quest page's Why?.
 */
export async function runWhy(
    promptArgs: string,
    stream: vscode.ChatResponseStream,
    token: vscode.CancellationToken,
): Promise<void> {
    const parsed = parseTraceArgs(promptArgs.trim());
    const words = promptArgs.replace(/(?:^|\s)--node(?:=|\s+)\S+/, " ").trim().split(/\s+/).filter((w) => w);
    const about = parsed.node || words.slice(1).join(" ");
    if (!parsed.questId) {
        stream.markdown(
            "Which quest? Example: `@fi /why 1790003131-my-quest` (why it stopped, why the review asked for a revision, " +
                "why the evidence is at its level), or add a step: `@fi /why 1790003131-my-quest execute`.\n",
        );
        return;
    }
    const ctx = launchContext(stream);
    if (!ctx || token.isCancellationRequested) return;
    stream.progress(`Reading why ${parsed.questId} did what it did…`);
    const args = ["--why", parsed.questId, ...(about ? [about] : []), "--output-root", ctx.outputRoot];
    const res = await runLaunch(ctx.python, ctx.repo, ctx.workDir, args);
    const text = (res.code === 0 ? res.stdout : [res.stdout, res.stderr].filter((t) => t.trim()).join("\n")).trim();
    const shown = text.length > OUTPUT_CHARS ? text.slice(0, OUTPUT_CHARS) + "\n…" : text;
    stream.markdown(shown ? "```\n" + shown + "\n```\n" : `No output (exit ${res.code}).\n`);
}

/**
 * `@fi /follow <quest_id>` — each step of a running quest as it happens (`python launch.py --trace <id> --follow`),
 * until it stops for you, finishes or fails. Stopping the chat stops following; the quest goes on.
 */
export async function runFollow(
    promptArgs: string,
    stream: vscode.ChatResponseStream,
    token: vscode.CancellationToken,
): Promise<void> {
    const parsed = parseTraceArgs(promptArgs.trim());
    if (parsed.error) {
        stream.markdown(`${parsed.error}\n`);
        return;
    }
    if (!parsed.questId) {
        stream.markdown("Which quest? Example: `@fi /follow 1790003131-my-quest` (add `--detail checks` for more).\n");
        return;
    }
    const ctx = launchContext(stream);
    if (!ctx) return;
    // The plain step-by-step view unless more was asked for.
    const detail = /--detail/.test(promptArgs) ? parsed.detail : "summary";
    const args = ["--trace", parsed.questId, "--follow", "--trace-detail", detail, "--output-root", ctx.outputRoot];
    if (parsed.node) args.push("--trace-node", parsed.node);
    await new Promise<void>((resolve) => {
        const child = spawn(ctx.python, [path.join(ctx.repo, "launch.py"), ...args], {
            cwd: ctx.workDir,
            env: { ...process.env, PYTHONIOENCODING: "utf-8", PYTHONUNBUFFERED: "1", FI_SKIP_BOOTSTRAP: "1" },
        });
        child.stdout?.setEncoding("utf8");
        child.stderr?.setEncoding("utf8");
        let pending = "";
        let stderr = "";
        child.stdout?.on("data", (d: string) => {
            pending += d;
            const lines = pending.split(/\r?\n/);
            pending = lines.pop() ?? "";
            const shown = lines.filter((l) => l.trim());
            // One code block per batch keeps each line exactly as launch.py printed it, backticks included.
            if (shown.length) stream.markdown("```\n" + shown.join("\n").replace(/```/g, "` ` `") + "\n```\n");
        });
        child.stderr?.on("data", (d: string) => (stderr += d));
        const stop = token.onCancellationRequested(() => child.kill());
        child.on("error", (e) => {
            stream.markdown(`Could not follow: ${String(e)}\n`);
            stop.dispose();
            resolve();
        });
        child.on("close", (code) => {
            if (pending.trim()) stream.markdown("```\n" + pending.trim().replace(/```/g, "` ` `") + "\n```\n");
            if (token.isCancellationRequested) stream.markdown("Stopped following; the quest goes on.\n");
            if (code !== 0 && !token.isCancellationRequested && stderr.trim()) {
                stream.markdown("```\n" + stderr.trim().slice(-2000) + "\n```\n");
            }
            stop.dispose();
            resolve();
        });
    });
}
