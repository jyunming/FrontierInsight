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
 * **Known limitation — Git Bash as the default profile on Windows.**
 * Measured, not assumed: MSYS rewrites the arguments it hands a native
 * Windows process, collapsing the leading `\\` of
 * `\\.\pipe\fi-bridge-<user>` down to a single `\`, so the pipe name
 * arrives malformed and the connect fails. It does this to an
 * environment variable as well as to an argument, so neither transport
 * dodges it. Both shells VS Code actually defaults to on Windows
 * (PowerShell, cmd.exe) pass the name through byte-for-byte, so this
 * only affects a user who has deliberately set Git Bash as their
 * terminal profile; running `/update` / `/generate` from a PowerShell
 * terminal is the workaround. Not worked around here because every
 * candidate fix (forward slashes, pre-doubling) would need a live
 * bridge to verify, and an unverified transformation is worse than a
 * documented limit.
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
    bridgeSocket?: string;
    shell: ShellKind;
}): string {
    return buildLaunchCommand({
        pythonPath: opts.pythonPath,
        args: ["--update", opts.questId],
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
