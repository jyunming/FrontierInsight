/**
 * `FI: Quest map` (and `@fi /map <quest_id>`) — the quest map in its own tab: every step of a quest in a few big
 * blocks, what a restart from each would keep and redo, and a Restart button.
 *
 * The map is drawn by the same script and stylesheet the web quest page uses (`web/static/quest_map.js` and
 * `quest_map.css`, copied to `media/` at compile time). Its data is `python launch.py --resume <id> --from --json`,
 * read from the quest's checkpoint. Restart sends the same `@fi /resume <id> --from <step>` to chat that a person would
 * type, so the resume runs exactly as it does there (with the model picker and progress in chat).
 */
import * as crypto from "crypto";
import * as fs from "fs";
import * as path from "path";
import * as vscode from "vscode";
import { rootsFromConfig } from "./roots-config";
import { runLaunch } from "./trace";

async function pickQuest(outputRoot: string): Promise<string | undefined> {
    let entries: fs.Dirent[] = [];
    try {
        entries = await fs.promises.readdir(outputRoot, { withFileTypes: true });
    } catch {
        entries = [];
    }
    const quests: { id: string; mtime: number }[] = [];
    for (const e of entries) {
        if (!e.isDirectory()) continue;
        try {
            const st = await fs.promises.stat(path.join(outputRoot, e.name, ".fi", "state.sqlite"));
            quests.push({ id: e.name, mtime: st.mtimeMs });
        } catch {
            // No checkpoint: not a quest that can be mapped.
        }
    }
    if (quests.length === 0) {
        void vscode.window.showInformationMessage(`No quest with a checkpoint under ${outputRoot}.`);
        return undefined;
    }
    quests.sort((a, b) => b.mtime - a.mtime);
    return vscode.window.showQuickPick(quests.map((q) => q.id), { placeHolder: "Which quest? (newest first)" });
}

function nonce(): string {
    return crypto.randomBytes(24).toString("base64").replace(/[^A-Za-z0-9]/g, "A");
}

function pageHtml(webview: vscode.Webview, media: vscode.Uri, questId: string): string {
    const css = webview.asWebviewUri(vscode.Uri.joinPath(media, "quest_map.css"));
    const js = webview.asWebviewUri(vscode.Uri.joinPath(media, "quest_map.js"));
    const n = nonce();
    const csp = `default-src 'none'; style-src ${webview.cspSource} 'unsafe-inline'; script-src 'nonce-${n}' ${webview.cspSource};`;
    return `<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" content="${csp}">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="${css}">
<style>body{margin:0;padding:12px;background:var(--vscode-editor-background)}</style>
</head><body><div id="root"></div>
<script nonce="${n}" src="${js}"></script>
<script nonce="${n}">
(function () {
  var vscode = acquireVsCodeApi();
  var questId = ${JSON.stringify(questId).replace(/</g, "\\u003c")};
  var configPath = '';
  var root = document.getElementById('root');
  var map = window.FIQuestMap.mount(root, {
    questId: questId,
    command: function (n) {
      var cmd = 'python launch.py --config ' + (/\s/.test(configPath) ? '"' + configPath + '"' : configPath) + ' --resume ' + questId + ' --from ' + (n.step || n.node);
      return n.needs_approval ? cmd + ' --approve-as "<your name>"' : cmd;
    },
    restart: function (n) {
      vscode.postMessage({ type: 'restart', step: n.step });
      return Promise.resolve('Sent to chat: @fi /resume ' + questId + ' --from ' + n.step + '. It continues there.');
    },
  });
  function theme() {
    var c = document.body.classList;
    map.setTheme(c.contains('vscode-light') || c.contains('vscode-high-contrast-light') ? 'light' : 'dark');
  }
  theme();
  new MutationObserver(theme).observe(document.body, { attributes: true, attributeFilter: ['class'] });
  window.addEventListener('message', function (e) {
    var m = e.data || {};
    if (m.type === 'data') { configPath = m.config_path || ''; map.update(m.data); }
    if (m.type === 'error') { root.textContent = m.text; }
  });
  vscode.postMessage({ type: 'ready' });
})();
</script></body></html>`;
}

/**
 * Where FI opens a tab of its own (the quest map, `plan.md`): among the tabs already open, never in a new split.
 * - A file already open in some group (`file`) is shown where it is, not opened a second time.
 * - Otherwise the active tab group: it covers every kind of tab (a text file, a preview, a webview), and stays the last
 *   editor group used while focus is on the chat in the side bar.
 * - When the active tab is the chat opened as an editor (or another built-in editor VS Code does not describe to
 *   extensions), a group of documents beside it is used instead, so the chat is not covered; with none, its own group.
 */
export function tabAreaColumn(file?: vscode.Uri): vscode.ViewColumn {
    const groups = vscode.window.tabGroups;
    if (file) {
        for (const g of groups.all) {
            if (g.tabs.some((t) => t.input instanceof vscode.TabInputText && t.input.uri.toString() === file.toString())) {
                return g.viewColumn;
            }
        }
    }
    const active = groups.activeTabGroup;
    const input = active.activeTab?.input;
    const described = [vscode.TabInputText, vscode.TabInputTextDiff, vscode.TabInputCustom, vscode.TabInputWebview,
        vscode.TabInputNotebook, vscode.TabInputNotebookDiff, vscode.TabInputTerminal];
    if (active.activeTab && !described.some((k) => input instanceof k)) {
        const other = groups.all.find((g) => g !== active && g.tabs.length > 0);
        if (other) return other.viewColumn;
    }
    return active.viewColumn;
}

/** One map per quest: opening it again shows the open one (and reads the checkpoint again) instead of a second tab. */
const openMaps = new Map<string, { panel: vscode.WebviewPanel; refresh: () => Promise<void> }>();

export async function openQuestMap(context: vscode.ExtensionContext, questIdArg?: string): Promise<void> {
    const cfg = vscode.workspace.getConfiguration("frontierInsight");
    const roots = rootsFromConfig(cfg);
    if ("error" in roots) {
        void vscode.window.showErrorMessage(roots.error);
        return;
    }
    const python = cfg.get<string>("pythonPath") || "python";
    const outputDirSetting = cfg.get<string>("outputDir") || "outputs";
    const outputRoot = path.isAbsolute(outputDirSetting) ? outputDirSetting : path.join(roots.workDir, outputDirSetting);
    const questId = (questIdArg || "").replace(/^["']+|["']+$/g, "") || (await pickQuest(outputRoot));
    if (!questId) return;
    if (!/^[A-Za-z0-9_\-.]+$/.test(questId)) {
        void vscode.window.showErrorMessage(`"${questId}" is not a quest id (letters, digits, - _ . only).`);
        return;
    }

    const key = `${outputRoot}\n${questId}`;
    const open = openMaps.get(key);
    if (open) {
        // A tab in the background reloads its page when shown, and the page asks for the data itself; only a tab
        // already in view needs to be told the checkpoint may have moved.
        const inView = open.panel.visible;
        open.panel.reveal(open.panel.viewColumn);
        if (inView) void open.refresh();
        return;
    }

    const media = vscode.Uri.joinPath(context.extensionUri, "media");
    const panel = vscode.window.createWebviewPanel("frontierInsight.questMap", `Quest map: ${questId}`,
        { viewColumn: tabAreaColumn(), preserveFocus: true }, { enableScripts: true, localResourceRoots: [media] });
    panel.webview.html = pageHtml(panel.webview, media, questId);
    const configPath = path.join(outputRoot, questId, "config.yaml");

    const send = async (): Promise<void> => {
        const res = await runLaunch(python, roots.repoPath, roots.workDir,
            ["--resume", questId, "--from", "--json", "--output-root", outputRoot]);
        try {
            const line = res.stdout.split(/\r?\n/).filter((l) => l.startsWith("{")).pop() || "";
            void panel.webview.postMessage({ type: "data", data: JSON.parse(line), config_path: configPath });
        } catch {
            const why = [res.stdout, res.stderr].filter((t) => t.trim()).join("\n").trim().slice(0, 1500);
            void panel.webview.postMessage({ type: "error", text: `Could not read the map of ${questId} (exit ${res.code}). ${why}` });
        }
    };
    openMaps.set(key, { panel, refresh: send });
    panel.onDidDispose(() => {
        if (openMaps.get(key)?.panel === panel) openMaps.delete(key);
    });
    panel.webview.onDidReceiveMessage(async (m: { type?: string; step?: string }) => {
        if (m.type === "ready") {
            await send();
        } else if (m.type === "restart" && typeof m.step === "string" && /^[a-z_]+$/.test(m.step)) {
            await vscode.commands.executeCommand("workbench.action.chat.open", {
                query: `@fi /resume ${questId} --from ${m.step}`,
            });
        }
    }); // the panel drops its own listeners when it is closed
}
