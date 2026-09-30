/**
 * The words after `@fi /resume`, read without VS Code (so a test can run them with node).
 *
 * `/resume <quest_id> --revise-plan "<what to change>"` is the CLI's spelling of `/plan <quest_id> <what to change>`.
 * It used to be read as a plain resume (only the first word, the id, was kept), which ran the quest into the same stop
 * again with the plan unchanged. A flag `/resume` does not understand is now said, not dropped.
 */

/**
 * `<quest_id> --revise-plan "<what to change>"` split into the id and the request (the words may be in straight or curly
 * quotes, or bare to the end of the line); `null` when there is no `--revise-plan`.
 */
export function splitRevisePlan(promptArgs: string): { questId: string; request: string } | null {
    const found = /(?:^|\s)--revise-plan(?:=|\s+|$)/.exec(promptArgs);
    if (!found) return null;
    const questId = (promptArgs.slice(0, found.index).trim().split(/\s+/)[0] || "").replace(/^["']+|["']+$/g, "");
    let request = promptArgs.slice(found.index + found[0].length).trim();
    const quoted = /^(["'“”‘’])([\s\S]*?)(["'“”‘’])\s*$/.exec(request);
    if (quoted) request = quoted[2];
    return { questId, request: request.trim() };
}

/**
 * The flags `/resume` reads itself; any other `--word` in its arguments is not used, and is said to be. (`--approve-as`
 * goes with `--from <an early step>`, which `/resume` answers by saying to run it in a terminal.)
 */
const RESUME_FLAGS = ["--from", "--revise-plan", "--approve-as"];

/** The `--flags` in `promptArgs` that `/resume` does not understand (outside quoted text). */
export function unknownResumeFlags(promptArgs: string): string[] {
    const unquoted = promptArgs.replace(/(["“”])[\s\S]*?(["“”])/g, " ");
    const out: string[] = [];
    for (const m of unquoted.matchAll(/(?:^|\s)(--[A-Za-z][\w-]*)/g)) {
        const flag = m[1];
        if (!RESUME_FLAGS.includes(flag) && !out.includes(flag)) out.push(flag);
    }
    return out;
}
