/**
 * `--config <quest.yaml>` beside the skill commands in the chat panel.
 *
 * A quest can name folders of skills other agents installed
 * (`engine.skills_dirs`). The commands a person uses to list, review, approve and
 * revoke a skill run without a quest, so on their own they see the usual places
 * only: a skill living just in a quest's own folder could not be reached from the
 * panel. `python launch.py --skills --config quest.yaml` already reaches it, and
 * this reads the same option out of the chat text (`@fi /skills --config
 * "C:\my quests\a.yaml"`).
 *
 * Plain string handling with no `vscode` import, so a test can load the compiled
 * file with Node.
 */

export interface SkillArgs {
    /** What is left of the text once `--config` and its path are taken out. */
    rest: string;
    /** The quest YAML the extra skill folders come from ("" when none was given). */
    config: string;
    /** Set when `--config` was written without a path. */
    error?: string;
}

const CONFIG_WITH_PATH = /(?:^|\s)--config(?:=|\s+)("[^"]*"|'[^']*'|\S+)/;
const CONFIG_ALONE = /(?:^|\s)--config(?:=?\s*$)/;

export function splitSkillArgs(text: string): SkillArgs {
    const found = CONFIG_WITH_PATH.exec(text);
    if (!found) {
        if (CONFIG_ALONE.test(text)) {
            return {
                rest: text.replace(/--config=?/, " ").replace(/\s+/g, " ").trim(),
                config: "",
                error: "`--config` needs the path to a quest YAML, e.g. `--config C:\\quests\\a.yaml`.",
            };
        }
        return { rest: text.trim(), config: "" };
    }
    let value = found[1];
    const quoted =
        (value.startsWith('"') && value.endsWith('"')) ||
        (value.startsWith("'") && value.endsWith("'"));
    if (quoted && value.length >= 2) value = value.slice(1, -1);
    const rest = (text.slice(0, found.index) + " " + text.slice(found.index + found[0].length))
        .replace(/\s+/g, " ")
        .trim();
    return { rest, config: value.trim() };
}

/** The `launch.py` arguments that carry the config (none without one). */
export function configArgs(config: string): string[] {
    return config ? ["--config", config] : [];
}

/** ` --config <path>`, for the follow-up commands printed to a person: run
 *  without it, they would say the skill is not there. */
export function configHint(config: string): string {
    if (!config) return "";
    return /\s/.test(config) ? ` --config "${config}"` : ` --config ${config}`;
}
