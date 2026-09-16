/**
 * Command lines for the FI terminal pass-throughs (`/update`,
 * `/generate`, `/ingest`, `/install-tectonic`).
 *
 * Two problems are solved here, both of which bit real users.
 *
 * **1. The shell is not one shell.** `Terminal.sendText` types a line
 * into whatever shell the user's default profile starts, and the three
 * we meet disagree about how a command line may begin:
 *
 * | shell      | `"python" --version`    | `& "python" --version`     |
 * |------------|-------------------------|----------------------------|
 * | PowerShell | ParserError             | runs                       |
 * | cmd.exe    | runs                    | `& was unexpected...`      |
 * | bash/zsh   | runs                    | n/a                        |
 *
 * PowerShell parses a line that *starts* with a quoted string as an
 * expression, so it needs the call operator `&`; cmd.exe treats `&` as
 * a command separator and chokes on a leading one. A single quoting
 * style therefore cannot work everywhere — the shell has to be
 * detected (`vscode.env.shell`) and the line spelled for it.
 *
 * **2. The bridge address has to travel.** Terminal-spawned FI runs are
 * not children of the extension, so they cannot inherit a per-command
 * TCP bridge the way `/start` does. They reach `vscode.lm.*` through the
 * session-long {@link PersistentBridge} instead, whose per-user address
 * comes from `./bridge-path`. That address is passed explicitly as
 * `--vscode-bridge-socket` rather than injected through
 * `ExtensionContext.environmentVariableCollection`, because VS Code
 * persists an environment collection across restarts: a stale pipe name
 * from a dead session would be re-applied to every terminal the user
 * opens, including ones that have nothing to do with FI.
 *
 * **Git Bash on Windows — the quoting is what matters, not MSYS.**
 * Re-measured against a real `C:\Program Files\Git\bin\bash.exe`
 * (GNU bash 5.2.37, MINGW64) by printing the argv a native
 * `python.exe` actually receives, and by opening a live
 * PersistentBridge connection through it:
 *
 * | spelling of `\\.\pipe\fi-bridge-me`   | argv the child receives |
 * |---------------------------------------|-------------------------|
 * | unquoted                              | `\.pipefi-bridge-me`    |
 * | `"..."` double-quoted                 | `\.\pipe\fi-bridge-me`  |
 * | `'...'` single-quoted (what we emit)  | `\\.\pipe\fi-bridge-me` |
 *
 * MSYS rewrites nothing here — this is plain bash backslash handling.
 * Unquoted, `\\`, `\p` and `\f` are escape sequences and the name is
 * destroyed; double quotes eat exactly one level, which is the
 * `\.\pipe\...` symptom previously blamed on MSYS (and an env var
 * behaves the same way: quoted survives, unquoted does not). The
 * single-quoted form {@link shellQuote} emits for `posix` arrives
 * byte-for-byte intact, and a real connect through Git Bash to a
 * listening bridge succeeds — so `/update` and `/generate` do work
 * under a Git Bash terminal profile.
 *
 * The quoting is therefore load-bearing rather than cosmetic: dropping
 * it for the socket would break Git Bash users only, silently. A
 * regression test feeds the generated line through a real Git Bash and
 * asserts the pipe name survives.
 *
 * One measurement trap, and the likely source of the retracted claim:
 * running `bash.exe -c "<line>"` *from a native Windows process* puts a
 * Win32 command-line layer in front of bash, and that layer eats a
 * backslash level on its own — `\\.\pipe\x` reaches bash already
 * `\.\pipe\x`. `Terminal.sendText` never goes through it: it types the
 * line into a shell that is already running. So the honest measurement
 * feeds bash over stdin, and that is what the test does.
 *
 * No `vscode` import — this module is pure so it can be exercised under
 * plain node (see tests/test_vscode_terminal_command.py).
 */

export type ShellKind = "powershell" | "cmd" | "posix";

/**
 * Classify `vscode.env.shell` (an absolute path to the shell binary,
 * or `""` where the host has no shell).
 *
 * Falls back on the platform when the path is empty: Windows hosts
 * default to PowerShell, which is the one shell that *errors* on an
 * unprefixed quoted head, so guessing it is the safe direction — a
 * `&` we did not need would be caught by cmd immediately and loudly,
 * whereas a missing one silently breaks nothing until a user with a
 * spaced interpreter path tries it.
 */
export function detectShell(shellPath: string, platform: string): ShellKind {
    const base = (shellPath || "")
        .split(/[\\/]/)
        .pop()!
        .toLowerCase()
        .replace(/\.exe$/, "");
    if (base === "pwsh" || base === "powershell") return "powershell";
    if (base === "cmd") return "cmd";
    if (base) return "posix";
    return platform === "win32" ? "powershell" : "posix";
}

/** Quote one argument for `shell`. Always returns a quoted token. */
export function shellQuote(arg: string, shell: ShellKind): string {
    if (shell === "powershell") {
        // PowerShell escapes with a backtick, not a backslash. `$` must
        // go too or a path containing `$` would be expanded as a variable.
        return `"${arg.replace(/`/g, "``").replace(/"/g, '`"').replace(/\$/g, "`$")}"`;
    }
    if (shell === "cmd") {
        // cmd.exe has no escape for `"` inside a quoted string, and a
        // Windows path cannot contain one, so dropping it is both safe
        // and the only option that cannot produce a broken line.
        return `"${arg.replace(/"/g, "")}"`;
    }
    return `'${arg.replace(/'/g, "'\\''")}'`;
}

// Tokens made only of these characters need no quoting in any of the
// three shells, so flags (`--emit`), quest ids and `paper_pdf` stay
// readable in the terminal the user is watching. Anything else — every
// Windows path, anything with a space — gets quoted.
const BARE_SAFE = /^[A-Za-z0-9._\/-]+$/;

function quoteArg(arg: string, shell: ShellKind): string {
    return BARE_SAFE.test(arg) ? arg : shellQuote(arg, shell);
}

export interface LaunchCommandOptions {
    pythonPath: string;
    /** Arguments after `launch.py`. */
    args: string[];
    /**
     * Persistent-bridge address. When set, appended as
     * `--vscode-bridge-socket <path>` so the spawned Python can route
     * LLM calls back through `vscode.lm.*`. Omit for commands that make
     * no model call (`--ingest`, `--install-tectonic`).
     */
    bridgeSocket?: string;
    shell: ShellKind;
}

/** Build a full `python launch.py ...` line for `shell`. */
export function buildLaunchCommand(opts: LaunchCommandOptions): string {
    const { pythonPath, args, bridgeSocket, shell } = opts;
    const parts: string[] = [];
    // PowerShell needs the call operator because the line begins with a
    // quoted string; cmd.exe rejects a leading `&`.
    if (shell === "powershell") parts.push("&");
    // The interpreter path is ALWAYS quoted — `"python"` / `& "python"` /
    // `'python'` are each valid in their shell, so quoting unconditionally
    // costs nothing and means a `C:\Program Files\...` path can never
    // split into two tokens.
    parts.push(shellQuote(pythonPath, shell));
    parts.push("launch.py");
    for (const arg of args) parts.push(quoteArg(arg, shell));
    if (bridgeSocket) {
        parts.push("--vscode-bridge-socket", shellQuote(bridgeSocket, shell));
    }
    return parts.join(" ");
}

/** The exact line `@fi /update` sends. */
export function updateTerminalCommand(opts: {
    pythonPath: string;
    questId: string;
    /**
     * Quest output root, i.e. the resolved `frontierInsight.outputDir`.
     * Omitted only by callers that genuinely want launch.py's default.
     */
    outputRoot?: string;
    bridgeSocket?: string;
    shell: ShellKind;
}): string {
    const args = ["--update", opts.questId];
    // `--update <id>` looks the quest up under `--output-root`, whose
    // argparse default is `./outputs`. The picker that produced this id
    // listed `frontierInsight.outputDir` instead, so without the flag a
    // user with a custom output directory gets "no quest directory at
    // <repo>\outputs\<id>" for a quest FI itself just offered them.
    // Every other spawn (--digest / --portfolio / --critique /
    // --proposal / --analyze) already passes the root; this one didn't.
    if (opts.outputRoot) args.push("--output-root", opts.outputRoot);
    return buildLaunchCommand({
        pythonPath: opts.pythonPath,
        args,
        bridgeSocket: opts.bridgeSocket,
        shell: opts.shell,
    });
}

/** The exact line `@fi /generate` sends. */
export function generateTerminalCommand(opts: {
    pythonPath: string;
    yamlPath: string;
    questId: string;
    kind: string;
    bridgeSocket?: string;
    shell: ShellKind;
}): string {
    return buildLaunchCommand({
        pythonPath: opts.pythonPath,
        args: ["--config", opts.yamlPath, "--resume", opts.questId, "--emit", opts.kind],
        bridgeSocket: opts.bridgeSocket,
        shell: opts.shell,
    });
}
