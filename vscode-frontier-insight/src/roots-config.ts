/** The VSCode side of `roots.ts`: reads the settings and the open workspace, and finds FI. */
import * as vscode from "vscode";
import {
    Located,
    Roots,
    RootsError,
    askPythonWhereFiIs,
    chooseWorkFolder,
    isFiFolder,
    locateFi,
    resolveRoots,
    savedLocationFile,
} from "./roots";

// Found once per window and kept for the session (re-checked on every use, so a moved
// folder is looked for again). One search at a time, so two commands started together
// never both open the folder picker.
let found: string | undefined;
let searching: Promise<Located> | undefined;

/** Look again on the next command (the Python or FI's folder setting changed). */
export function forgetFoundFi(): void {
    found = undefined;
}

async function findFi(cfg: vscode.WorkspaceConfiguration, folders: string[]): Promise<Located> {
    const setting = (cfg.get<string>("repoPath") || "").trim();
    // An open FI checkout always wins (cheap to check, so before the session's answer).
    const own = setting ? undefined : folders.find((f) => isFiFolder(f));
    if (own) return { repoPath: own, how: "workspace" };
    if (!setting && found && isFiFolder(found)) {
        return { repoPath: found, how: "saved" };
    }
    if (!setting && searching) return searching;
    const search = locateFi({
        setting,
        workspaceFolders: folders,
        savedFile: savedLocationFile(),
        askPython: () => askPythonWhereFiIs(cfg.get<string>("pythonPath") || "python"),
        askUser: async () => {
            const picked = await vscode.window.showOpenDialog({
                canSelectFiles: false,
                canSelectFolders: true,
                canSelectMany: false,
                openLabel: "This is FrontierInsight",
                title: "Where is FrontierInsight? Pick the folder that contains launch.py",
            });
            return picked?.[0]?.fsPath;
        },
    });
    if (setting) return search;
    searching = search;
    try {
        const got = await search;
        if (!("error" in got)) {
            if (got.how !== "workspace") found = got.repoPath;
            if (got.how === "picked") {
                void vscode.window.showInformationMessage(
                    got.remembered
                        ? `FrontierInsight: using ${got.repoPath}. Remembered for every window, so you are not asked again.`
                        : `FrontierInsight: using ${got.repoPath} for now. It could not be saved in ` +
                          `${savedLocationFile()}, so you will be asked again next time.`,
                );
            }
        }
        return got;
    } finally {
        searching = undefined;
    }
}

function openFolders(): string[] {
    return (vscode.workspace.workspaceFolders || []).map((f) => f.uri.fsPath);
}

/** The open folder a command works in (see `chooseWorkFolder`); undefined with no folder open. */
export function workFolderForCommand(namedPath?: string): string | undefined {
    const active = vscode.window.activeTextEditor?.document.uri;
    return chooseWorkFolder({
        folders: openFolders(),
        namedPath,
        activeFile: active && active.scheme === "file" ? active.fsPath : undefined,
    });
}

/**
 * Where FI is and which folder this command works in. `namedPath` is a file the command
 * names (the YAML of `/start`): with several folders open it picks the folder holding it.
 */
export async function rootsForCommand(namedPath?: string): Promise<Roots | RootsError> {
    const cfg = vscode.workspace.getConfiguration("frontierInsight");
    const folders = openFolders();
    const workspaceRoot = workFolderForCommand(namedPath);
    if (!workspaceRoot && !(cfg.get<string>("workingDir") || "").trim() && !(cfg.get<string>("repoPath") || "").trim()) {
        // Nowhere to put the study: never FI's own checkout by default.
        return { error: "❌ No folder open. Open your study's folder in VS Code (File → Open Folder…), then try again." };
    }
    const located = await findFi(cfg, folders);
    if ("error" in located) return located;
    return resolveRoots({
        repoPathSetting: located.repoPath,
        workingDirSetting: cfg.get<string>("workingDir") || "",
        workspaceRoot,
    });
}
