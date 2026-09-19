/** The VSCode side of `roots.ts`: reads the settings and the open workspace. */
import * as vscode from "vscode";
import { Roots, RootsError, resolveRoots } from "./roots";

export function rootsFromConfig(cfg: vscode.WorkspaceConfiguration): Roots | RootsError {
    return resolveRoots({
        repoPathSetting: cfg.get<string>("repoPath") || "",
        workingDirSetting: cfg.get<string>("workingDir") || "",
        workspaceRoot: vscode.workspace.workspaceFolders?.[0]?.uri.fsPath,
    });
}
