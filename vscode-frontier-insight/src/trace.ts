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
