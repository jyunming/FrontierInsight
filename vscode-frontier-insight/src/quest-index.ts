/**
 * Every quest FI has run on this computer (`~/.frontier-insight/quests.json`, written by `core/quest_index.py`), so a
 * quest started in another folder can be picked or named by its id from this window. Read only: Python is the one
 * writer (it records a quest when it starts, resumes or is renamed, under a file lock). Entries whose folder is gone
 * are skipped here; Python drops them from the file when it next looks one up (or `fi tools quests --prune`).
 *
 * The matching rule is `core/quest_index.py:matching`: the full id, else every id that starts or ends with the text
 * (at least MIN_SHORT characters, letter case ignored). The short id shown in lists is the six characters after the
 * last dash.
 *
 * Pure logic with no `vscode` import so plain Node can test it.
 */
import * as fs from "fs";
import * as path from "path";
import { fiHome } from "./fi-home";

export const MIN_SHORT = 4;

export interface IndexEntry {
    questId: string;
    questRoot: string;
    config: string;
    workingFolder: string;
    title: string;
    createdAt: string;
    lastSeen: string;
}

export function indexPath(env: NodeJS.ProcessEnv = process.env): string {
    return path.join(fiHome(env), "quests.json");
}

export function shortId(questId: string): string {
    const m = /-([0-9a-f]{6})$/.exec(questId);
    return m ? m[1] : questId;
}

/** The ids `text` names: itself when it is one of them, else every id starting or ending with it. */
export function matchIds(text: string, ids: string[]): string[] {
    if (ids.includes(text)) return [text];
    const t = text.toLowerCase();
    if (t.length < MIN_SHORT) return [];
    return ids.filter((i) => i.toLowerCase().startsWith(t) || i.toLowerCase().endsWith(t)).sort();
}

function str(v: unknown): string {
    return typeof v === "string" ? v : "";
}

/** The entries whose quest folder still exists, most recently seen first ([] when there is no readable index). */
export function loadIndex(file: string = indexPath(), exists: (p: string) => boolean = fs.existsSync): IndexEntry[] {
    let raw: unknown;
    try {
        raw = JSON.parse(fs.readFileSync(file, "utf-8"));
    } catch {
        return [];
    }
    const quests = (raw as { quests?: unknown } | null)?.quests;
    if (!quests || typeof quests !== "object") return [];
    const out: IndexEntry[] = [];
    for (const [questId, e] of Object.entries(quests as Record<string, Record<string, unknown>>)) {
        if (!/^[A-Za-z0-9_\-.]+$/.test(questId) || !e || typeof e !== "object") continue;
        const questRoot = str(e.quest_root);
        if (!questRoot || !exists(path.join(questRoot, ".fi"))) continue;
        out.push({
            questId, questRoot,
            config: str(e.config) || path.join(questRoot, "config.yaml"),
            workingFolder: str(e.working_folder),
            title: str(e.title),
            createdAt: str(e.created_at),
            lastSeen: str(e.last_seen),
        });
    }
    out.sort((a, b) => (b.lastSeen || b.createdAt).localeCompare(a.lastSeen || a.createdAt));
    return out;
}

export type IndexLookup =
    | { kind: "found"; entry: IndexEntry }
    | { kind: "ambiguous"; entries: IndexEntry[] }
    | { kind: "none" };

export function findInIndex(text: string, entries: IndexEntry[]): IndexLookup {
    const byId = new Map(entries.map((e) => [e.questId, e]));
    const hits = matchIds(text, [...byId.keys()]);
    if (hits.length === 1) return { kind: "found", entry: byId.get(hits[0])! };
    if (hits.length > 1) return { kind: "ambiguous", entries: hits.map((h) => byId.get(h)!) };
    return { kind: "none" };
}

/** The markdown lines that list quests a shortened id matched, each with its title and folder. */
export function describeAmbiguous(text: string, entries: { questId: string; title: string; questRoot: string }[]): string {
    return `\`${text}\` matches more than one quest; give more of its id:\n\n` +
        entries.map((e) => `- \`${e.questId}\` ${e.title || "(no title yet)"} — \`${e.questRoot}\``).join("\n") + "\n";
}
