/**
 * The one list of quests the extension offers: this folder's quests first, then every other quest FI has run on this
 * computer, each with whether it is running now. It is read from FI's own `launch.py --quests --json --output-root <dir>`
 * (`core/quest_index.py:listed` and `status`), the list and the running test the web page and `fi tools quests` use, so
 * the record and the process test are not written a second time here.
 *
 * Only when Python cannot be run is the index file read directly (`loadIndex`), with no status: a quest of unknown status
 * is never refused, so a broken Python never blocks a resume that the person asked for.
 *
 * Pure logic with no `vscode` import so plain Node can test it.
 */
import { IndexEntry, loadIndex } from "./quest-index";

export type QuestStatus = "running" | "waiting for you" | "failed" | "finished" | "stopped" | "";

export interface ListedQuest extends IndexEntry {
    /** FI's own word for where the quest is; "" when Python could not be asked. */
    status: QuestStatus;
}

export interface LaunchResult {
    code: number;
    stdout: string;
}

/** Runs `launch.py` with these arguments (the extension passes its own spawn; a test passes a fake). */
export type LaunchRunner = (args: string[]) => Promise<LaunchResult>;

function str(v: unknown): string {
    return typeof v === "string" ? v : "";
}

/** The quests in `fi tools quests --json` output (its last JSON line); undefined when there is none to read. */
export function parseQuestList(stdout: string): ListedQuest[] | undefined {
    const line = stdout.split(/\r?\n/).filter((l) => l.startsWith("{")).pop();
    if (!line) return undefined;
    let raw: unknown;
    try {
        raw = JSON.parse(line);
    } catch {
        return undefined;
    }
    const quests = (raw as { quests?: unknown } | null)?.quests;
    if (!Array.isArray(quests)) return undefined;
    const out: ListedQuest[] = [];
    for (const q of quests as Record<string, unknown>[]) {
        const questId = str(q?.quest_id);
        const questRoot = str(q?.quest_root);
        if (!/^[A-Za-z0-9_\-.]+$/.test(questId) || !questRoot) continue;
        out.push({
            questId, questRoot,
            config: str(q.config), workingFolder: str(q.working_folder), title: str(q.title),
            createdAt: str(q.created_at), lastSeen: str(q.last_seen),
            status: str(q.status) as QuestStatus,
        });
    }
    return out;
}

/** The list for `outputsDir`: from Python, else (Python could not run) the index file with no status. */
export async function loadQuestList(
    run: LaunchRunner, outputsDir: string, fallback: () => IndexEntry[] = loadIndex,
): Promise<ListedQuest[]> {
    try {
        const res = await run(["--quests", "--json", "--output-root", outputsDir]);
        const parsed = res.code === 0 ? parseQuestList(res.stdout) : undefined;
        if (parsed) return parsed;
    } catch {
        // fall through to the index file
    }
    return fallback().map((e) => ({ ...e, status: "" as QuestStatus }));
}

/** Words for a quest that is running now (shown beside it in a list), "" for any other. */
export function runningMark(q: { status: string }): string {
    return q.status === "running" ? "running now" : "";
}

/** Whether `questRoot` is running now by FI's own test; false when the list does not hold it or could not say. */
export function isRunning(list: ListedQuest[], questRoot: string): boolean {
    const norm = (p: string): string => p.replace(/[\\/]+$/, "").replace(/\\/g, "/").toLowerCase();
    const want = norm(questRoot);
    return list.some((q) => norm(q.questRoot) === want && q.status === "running");
}

export function stillRunningMessage(questId: string): string {
    return `Quest \`${questId}\` is still running, so nothing was started: a second run of it would share its files. ` +
        "Wait until it stops, or stop it first.";
}
