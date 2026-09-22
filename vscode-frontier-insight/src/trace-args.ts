/**
 * `--node <name>` / `--detail <level>` beside `@fi /trace <quest_id>` in the chat panel.
 *
 * Names match the web API's own query parameters (`/api/quests/{id}/trace?node=&detail=`),
 * not the CLI's `--trace-node` / `--trace-detail` flags — those are global `launch.py` flags
 * that need the `--trace-` prefix to stay unambiguous next to `--trace` itself; here, under
 * `/trace`, that prefix is already implied.
 *
 * Plain string handling with no `vscode` import, so a test can load the compiled file with Node.
 */

export interface TraceArgs {
    /** The quest id (first bare word once `--node`/`--detail` are taken out; "" when none was given). */
    questId: string;
    /** "" means every node. */
    node: string;
    /** One of DETAILS; defaults to "checks" (core/audit_log.py's own default). */
    detail: string;
    /** Set when a flag was written without a valid value. */
    error?: string;
}

export const DETAILS = ["summary", "checks", "debug"];

// The bare-word branch excludes a value starting with `--` so a missing value before the next
// flag (`--node --detail checks`) falls through to the ALONE case and is reported, instead of
// silently swallowing that next flag as this one's value.
const NODE_WITH_VALUE = /(?:^|\s)--node(?:=|\s+)("[^"]*"|'[^']*'|(?!--)\S+)/;
const NODE_ALONE = /(?:^|\s)--node(?:=\s*$|\s*$|\s+(?=--))/;
const DETAIL_WITH_VALUE = /(?:^|\s)--detail(?:=|\s+)("[^"]*"|'[^']*'|(?!--)\S+)/;
const DETAIL_ALONE = /(?:^|\s)--detail(?:=\s*$|\s*$|\s+(?=--))/;

function unquote(value: string): string {
    const quoted =
        (value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"));
    return quoted && value.length >= 2 ? value.slice(1, -1) : value;
}

function strip(text: string, match: RegExpExecArray): string {
    return text.slice(0, match.index) + " " + text.slice(match.index + match[0].length);
}

export function parseTraceArgs(text: string): TraceArgs {
    let rest = text;
    let node = "";
    let detail = "checks";
    let error: string | undefined;

    const nodeMatch = NODE_WITH_VALUE.exec(rest);
    if (nodeMatch) {
        node = unquote(nodeMatch[1]);
        rest = strip(rest, nodeMatch);
    } else if (NODE_ALONE.test(rest)) {
        error = "`--node` needs a node name, e.g. `--node design`.";
        rest = rest.replace(/--node=?/, " ");
    }

    const detailMatch = DETAIL_WITH_VALUE.exec(rest);
    if (detailMatch) {
        const value = unquote(detailMatch[1]);
        if (!DETAILS.includes(value)) {
            error = error ?? `\`--detail\` must be one of ${DETAILS.join(", ")}; got "${value}".`;
        } else {
            detail = value;
        }
        rest = strip(rest, detailMatch);
    } else if (DETAIL_ALONE.test(rest)) {
        error = error ?? "`--detail` needs one of summary, checks, debug.";
        rest = rest.replace(/--detail=?/, " ");
    }

    rest = rest.replace(/\s+/g, " ").trim();
    const questId = rest.split(/\s+/)[0] || "";
    return { questId, node, detail, error };
}
