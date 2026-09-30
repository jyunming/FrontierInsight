/**
 * The folder FI keeps its per-person state in: `~/.frontier-insight` (`%USERPROFILE%\.frontier-insight` on Windows);
 * `FI_HOME` overrides it. The same rule as `core/fi_home.py` — keep the two in step.
 *
 * Pure logic with no `vscode` import so plain Node can test it.
 */
import * as os from "os";
import * as path from "path";

export function fiHome(env: NodeJS.ProcessEnv = process.env, home: string = os.homedir()): string {
    const override = (env.FI_HOME || "").trim();
    return override ? path.resolve(override.replace(/^~(?=$|[\\/])/, home)) : path.join(home, ".frontier-insight");
}
