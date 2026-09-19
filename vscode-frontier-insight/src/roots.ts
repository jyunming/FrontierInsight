/**
 * Where FrontierInsight is installed, and where the user works.
 *
 * They are one folder only when the workspace IS the FI checkout. Otherwise FI
 * lives where `frontierInsight.repoPath` says (it supplies `launch.py`), and the
 * quests run in the workspace folder: the YAML, the example files, the relative
 * `outputs/` all mean the user's project, not FI's checkout.
 *
 * Pure logic with no `vscode` import so plain Node can test it.
 */
import * as fs from "fs";
import * as path from "path";

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
