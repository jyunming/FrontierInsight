/**
 * Where FrontierInsight is installed, and where the user works.
 *
 * They are one folder only when the workspace IS the FI checkout. Otherwise FI
 * is found (see `locateFi`) and supplies `launch.py`, while the quests run in
 * the workspace folder: the YAML, the example files, the relative `outputs/`
 * all mean the user's project, not FI's checkout. One VS Code window per study
 * folder is the way to run several studies at once.
 *
 * Pure logic with no `vscode` import so plain Node can test it.
 */
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import { execFile } from "child_process";

export interface Roots {
    /** The folder containing FI's `launch.py`. */
    repoPath: string;
    /** The folder quests run in: cwd, relative paths, `outputs/`. */
    workDir: string;
}

export interface RootsError {
    error: string;
}

export interface RootsInput {
    repoPathSetting: string;
    workingDirSetting: string;
    workspaceRoot: string | undefined;
    fileExists?: (p: string) => boolean;
}

export function resolveRoots(input: RootsInput): Roots | RootsError {
    const exists = input.fileExists ?? ((p: string) => fs.existsSync(p));
    const workspace = input.workspaceRoot;

    // Where FI is: what the setting names, else the workspace when it is the FI checkout.
    let repoPath = input.repoPathSetting.trim();
    if (!repoPath && workspace && exists(path.join(workspace, "launch.py"))) {
        repoPath = workspace;
    }
    if (!repoPath) {
        return {
            error: workspace
                ? "❌ FrontierInsight was not found: this workspace has no `launch.py`. " +
                  "Set `frontierInsight.repoPath` to the FrontierInsight folder (the one containing `launch.py`); " +
                  "your quests will run in this workspace folder, not there."
                : "❌ No workspace open. Open your project folder, or set `frontierInsight.repoPath` " +
                  "to the FrontierInsight folder, then try again.",
        };
    }
    if (!exists(path.join(repoPath, "launch.py"))) {
        return {
            error: `❌ \`frontierInsight.repoPath\` is \`${repoPath}\`, which has no \`launch.py\`. ` +
                "Point it at the FrontierInsight folder.",
        };
    }

    // Where the user works: the setting, else the workspace folder, else FI itself.
    const configured = input.workingDirSetting.trim();
    const workDir = configured
        ? path.resolve(workspace ?? repoPath, configured)
        : workspace ?? repoPath;
    return { repoPath, workDir };
}


// ---------------------------------------------------------------------------
// Finding FrontierInsight without any setting.
// ---------------------------------------------------------------------------

/** How FI was found, in the order it is looked for. */
export type FoundHow = "setting" | "workspace" | "python" | "saved" | "picked";

export type Located = { repoPath: string; how: FoundHow; remembered?: boolean } | RootsError;

export interface LocateInput {
    /** `frontierInsight.repoPath`; empty for nearly everyone. */
    setting: string;
    /** Every folder open in this window. */
    workspaceFolders: string[];
    /** Where a folder the person picked is remembered (`~/.frontier-insight/fi_location.json`). */
    savedFile: string;
    /** Ask the configured Python where FI is installed (after `pip install -e <FI>`). */
    askPython: () => Promise<string | undefined>;
    /** The one question, asked only when nothing else found FI: a folder picker. */
    askUser: () => Promise<string | undefined>;
    fileExists?: (p: string) => boolean;
    readText?: (p: string) => string | undefined;
    writeText?: (p: string, text: string) => void;
}

/** Where a picked FI folder is remembered, beside FI's other per-person files (profile.json, skills/). */
export function savedLocationFile(home: string = os.homedir()): string {
    return path.join(home, ".frontier-insight", "fi_location.json");
}

/** A folder holds `launch.py` (all a `frontierInsight.repoPath` someone typed is checked for). */
function hasLaunch(dir: string, exists: (p: string) => boolean): boolean {
    return !!dir && exists(path.join(dir, "launch.py"));
}

/**
 * A folder is FrontierInsight when it holds `launch.py` beside `core/engine.py`: a study's own
 * `launch.py` (a common name for a start script) is never taken for FI.
 */
export function isFiFolder(dir: string, exists: (p: string) => boolean = fs.existsSync): boolean {
    return hasLaunch(dir, exists) && exists(path.join(dir, "core", "engine.py"));
}

function readSaved(file: string, read: (p: string) => string | undefined): string {
    try {
        const raw = read(file);
        if (!raw) return "";
        const data = JSON.parse(raw) as { path?: unknown };
        return typeof data.path === "string" ? data.path : "";
    } catch {
        return "";
    }
}

/**
 * Find FrontierInsight: the `frontierInsight.repoPath` setting when someone set it,
 * else an open folder that is the FI checkout, else the configured Python's install
 * of FI, else the folder a person picked before, else ask once with a folder picker
 * and remember the answer in `~/.frontier-insight/fi_location.json` (never in a
 * VS Code setting), so every window, and every later session, finds it.
 */
export async function locateFi(input: LocateInput): Promise<Located> {
    const exists = input.fileExists ?? ((p: string) => fs.existsSync(p));
    const read = input.readText ?? ((p: string) => {
        try { return fs.readFileSync(p, "utf-8"); } catch { return undefined; }
    });
    const write = input.writeText ?? ((p: string, text: string) => {
        fs.mkdirSync(path.dirname(p), { recursive: true });
        fs.writeFileSync(p, text, "utf-8");
    });

    const setting = input.setting.trim();
    if (setting) {
        // Someone chose this on purpose: a wrong value is reported, not skipped over.
        return hasLaunch(setting, exists)
            ? { repoPath: setting, how: "setting" }
            : {
                error: `❌ \`frontierInsight.repoPath\` is \`${setting}\`, which has no \`launch.py\`. ` +
                    "Point it at the FrontierInsight folder, or clear it and FI is found by itself.",
            };
    }
    const own = input.workspaceFolders.find((f) => isFiFolder(f, exists));
    if (own) return { repoPath: own, how: "workspace" };

    let fromPython: string | undefined;
    try { fromPython = await input.askPython(); } catch { fromPython = undefined; }
    if (fromPython && isFiFolder(fromPython, exists)) return { repoPath: fromPython, how: "python" };

    const saved = readSaved(input.savedFile, read);
    if (saved && isFiFolder(saved, exists)) return { repoPath: saved, how: "saved" };

    let picked: string | undefined;
    try { picked = await input.askUser(); } catch { picked = undefined; }
    if (!picked) {
        return {
            error: "❌ FrontierInsight was not found. Run the command again and pick the FrontierInsight " +
                "folder (the one that contains `launch.py`), or install it into your Python once with " +
                "`pip install -e <FrontierInsight folder>`.",
        };
    }
    if (!isFiFolder(picked, exists)) {
        return {
            error: `❌ \`${picked}\` is not the FrontierInsight folder (that one has \`launch.py\` and \`core/engine.py\`). ` +
                "Run the command again and pick the folder that contains `launch.py`.",
        };
    }
    let remembered = true;
    try {
        write(input.savedFile, JSON.stringify({ path: picked }, null, 2) + "\n");
    } catch {
        // Not remembered (read-only home, say): it still works now, and is asked again next session.
        remembered = false;
    }
    return { repoPath: picked, how: "picked", remembered };
}

/**
 * The Python asked where FI is. It never imports `launch.py` (that would run it): it only
 * looks the module up, with the current folder taken off the search path so a `launch.py`
 * lying in some project can't answer. The answer counts only when it is a `launch.py` beside
 * FI's `core/engine.py` (another project can ship a module called `launch`).
 */
export const PYTHON_WHERE_IS_FI =
    "import sys, importlib.util as u; " +
    "sys.path[:] = [p for p in sys.path if p not in ('', '.')]; " +
    "s = u.find_spec('launch'); " +
    "print(s.origin if s is not None and s.origin else '')";

export function fiFolderFromPythonAnswer(stdout: string, exists: (p: string) => boolean = fs.existsSync): string | undefined {
    const origin = stdout.trim().split(/\r?\n/).pop()?.trim() || "";
    if (!origin || path.basename(origin) !== "launch.py") return undefined;
    const dir = path.dirname(origin);
    return isFiFolder(dir, exists) ? dir : undefined;
}

export function askPythonWhereFiIs(pythonPath: string, timeoutMs = 8000): Promise<string | undefined> {
    return new Promise((resolve) => {
        try {
            execFile(
                pythonPath, ["-c", PYTHON_WHERE_IS_FI],
                {
                    // The extension's own folder: Windows looks for a bare `python` in the current
                    // folder first, so not one anyone writes to. UTF-8 out so C:\Users\張三 survives.
                    cwd: __dirname, timeout: timeoutMs, windowsHide: true, encoding: "utf-8",
                    env: { ...process.env, PYTHONIOENCODING: "utf-8" },
                },
                (err, stdout) => resolve(err ? undefined : fiFolderFromPythonAnswer(String(stdout))),
            );
        } catch {
            resolve(undefined);
        }
    });
}


// ---------------------------------------------------------------------------
// Which open folder a command works in, when a window has several.
// ---------------------------------------------------------------------------

export interface WorkFolderInput {
    /** Every folder open in this window, in order. */
    folders: string[];
    /** A file or folder the command names (the YAML of `/start`), absolute or relative. */
    namedPath?: string;
    /** The file in the active editor. */
    activeFile?: string;
    fileExists?: (p: string) => boolean;
}

function inside(folder: string, p: string): boolean {
    const rel = path.relative(folder, p);
    return rel === "" || (rel !== ".." && !rel.startsWith(".." + path.sep) && !path.isAbsolute(rel));
}

/** The open folder holding `p` (the deepest one when folders nest), or undefined. */
function folderHolding(folders: string[], p: string): string | undefined {
    return folders
        .filter((f) => inside(f, p))
        .sort((a, b) => b.length - a.length)[0];
}

/**
 * With one folder open, that folder. With several: the folder of the file the command
 * names (the YAML of `/start`), else the folder of the file in the active editor, else
 * the first folder that is not FrontierInsight's own checkout (so nothing lands in it by accident),
 * else the first. Never a question: the folder open in the window is where you work.
 * (Finding a quest by its id alone, across folders, is left to the quest index.)
 */
export function chooseWorkFolder(input: WorkFolderInput): string | undefined {
    const { folders } = input;
    if (folders.length <= 1) return folders[0];
    const exists = input.fileExists ?? ((p: string) => fs.existsSync(p));
    const named = (input.namedPath || "").trim().replace(/^["']+|["']+$/g, "");
    if (named) {
        if (path.isAbsolute(named)) {
            const holder = folderHolding(folders, named);
            if (holder) return holder;
        } else {
            const holder = folders.find((f) => exists(path.join(f, named)));
            if (holder) return holder;
        }
    }
    if (input.activeFile) {
        const holder = folderHolding(folders, input.activeFile);
        if (holder) return holder;
    }
    return folders.find((f) => !isFiFolder(f, exists)) ?? folders[0];
}
