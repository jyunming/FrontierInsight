/**
 * Skills in the chat panel — `@fi /skills`, `/scan-skill`, `/approve-skill`,
 * `/import-skill`, `/teach-skill`.
 *
 * Every handler here shells out to `python launch.py … --json` and renders
 * the result. None of the gate's logic is reimplemented in TypeScript, on
 * purpose: discovery, self-test execution, content hashing, the approval
 * ledger and the static scan all live in Python, and a second copy would
 * drift from it the first time a rule changed. The CLI, the web API and this
 * panel therefore read one serialiser (`SkillState.to_dict`).
 *
 * The one thing this module *does* own is the approval ceremony. The
 * ledger's rule is that a person decides and is named; in a chat panel that
 * means an input box the user actually types into, never an identity derived
 * silently from a setting.
 */
import { spawn } from "child_process";
import * as vscode from "vscode";

export interface SkillRow {
    name: string;
    status: string;
    loadable: boolean;
    maturity: string;
    kind: string;
    reason: string;
    findings: string[];
    selftest_generated: boolean;
    domains: string[];
    path: string;
}

interface RunResult {
    code: number;
    stdout: string;
    stderr: string;
}

/** Resolve (python, repo) or explain to the user why we cannot. */
function resolveRepo(
    stream: vscode.ChatResponseStream,
): { python: string; repo: string } | null {
    const cfg = vscode.workspace.getConfiguration("frontierInsight");
    const python = cfg.get<string>("pythonPath") || "python";
    let repo = cfg.get<string>("repoPath") || "";
    if (!repo) {
        const ws = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
        if (!ws) {
            stream.markdown(
                "No workspace open. Open the FrontierInsight repo folder, or set `frontierInsight.repoPath` in settings.\n",
            );
            return null;
        }
        repo = ws;
    }
    return { python, repo };
}

function runLaunch(
    python: string,
    repo: string,
    args: string[],
): Promise<RunResult> {
    return new Promise((resolve) => {
        const child = spawn(python, ["launch.py", ...args], {
            cwd: repo,
            env: { ...process.env, PYTHONIOENCODING: "utf-8" },
        });
        let stdout = "";
        let stderr = "";
        child.stdout?.on("data", (d) => (stdout += d.toString()));
        child.stderr?.on("data", (d) => (stderr += d.toString()));
        child.on("error", (e) => resolve({ code: -1, stdout, stderr: String(e) }));
        child.on("close", (code) => resolve({ code: code ?? -1, stdout, stderr }));
    });
}

/**
 * Parse JSON starting at the first `{`.
 *
 * `--json` emits exactly one document, but Axon and langgraph print import
 * banners and not all of them reach stderr on every platform. Trusting the
 * whole buffer would let an unrelated warning break the panel.
 */
function parseJsonLoose(stdout: string): any | null {
    const start = stdout.indexOf("{");
    if (start < 0) return null;
    try {
        return JSON.parse(stdout.slice(start));
    } catch (_) {
        return null;
    }
}

/**
 * `@fi /skills` — the library, and why each entry is or is not usable.
 *
 * Self-tests are deliberately **not** run: each may take up to 120 seconds,
 * so a real library would leave the panel silent for minutes. The output says
 * so and names the command that does verify, because a listing that quietly
 * skipped the checks while looking like the CLI's verified one would be worse
 * than a slow listing.
 */
export async function runListSkills(
    stream: vscode.ChatResponseStream,
    token: vscode.CancellationToken,
): Promise<void> {
    const env = resolveRepo(stream);
    if (!env || token.isCancellationRequested) return;

    stream.progress("Reading the skill library…");
    const res = await runLaunch(env.python, env.repo, [
        "--skills",
        "--json",
        "--no-run-selftests",
    ]);
    const data = parseJsonLoose(res.stdout);
    if (!data) {
        stream.markdown(
            `Could not read the skill library (exit ${res.code}).\n\n\`\`\`\n${(
                res.stderr || res.stdout
            ).slice(-800)}\n\`\`\`\n`,
        );
        return;
    }

    const rows: SkillRow[] = data.skills ?? [];
    if (rows.length === 0) {
        stream.markdown(
            "**No skills installed.**\n\nFI ships none — a skill is what it picks up working with you, on this machine. Teach it one:\n\n" +
                "- `@fi /teach-skill <name> <module>` — draft one from a library you already have\n" +
                "- `@fi /import-skill` — adapt one written for another agent\n",
        );
        return;
    }

    stream.markdown(`🧠 **${rows.length} skill(s)**\n\n`);
    stream.markdown("| | Skill | Status | Kind | Why |\n|---|---|---|---|---|\n");
    for (const s of rows) {
        const mark = s.loadable ? "✅" : "⛔";
        const dom = s.domains?.length ? ` <sub>${s.domains.join(", ")}</sub>` : "";
        stream.markdown(
            `| ${mark} | \`${s.name}\`${dom} | ${s.status} | ${s.kind} | ${s.reason} |\n`,
        );
    }
    stream.markdown("\n");
    stream.markdown(
        "_Self-tests were **not** run (each can take up to 120s), so these statuses are as last recorded rather than verified. `python launch.py --skills` runs them._\n\n",
    );

    const flagged = rows.filter((s) =>
        (s.findings ?? []).some((f) => f.startsWith("[high")),
    );
    if (flagged.length) {
        stream.markdown(
            `⚠️ **${flagged.length} skill(s) with high-severity scan findings:** ` +
                flagged.map((s) => `\`${s.name}\``).join(", ") +
                " — review with `@fi /scan-skill <name>` before approving.\n\n",
        );
    }
    const generated = rows.filter((s) => s.selftest_generated);
    if (generated.length) {
        stream.markdown(
            `ℹ️ ${generated.length} skill(s) carry an **auto-generated** self-test: ` +
                generated.map((s) => `\`${s.name}\``).join(", ") +
                " — that proves the bundled tooling runs, not that it behaves the way its instructions describe.\n\n",
        );
    }
    stream.markdown(
        "Next: `@fi /scan-skill <name>` to review one, `@fi /approve-skill <name>` to approve it.\n",
    );
}

/** `@fi /scan-skill <name>` — the static review, rendered. */
export async function runScanSkill(
    promptArgs: string,
    stream: vscode.ChatResponseStream,
    token: vscode.CancellationToken,
): Promise<void> {
    const name = promptArgs.trim().split(/\s+/)[0] || "";
    if (!name) {
        stream.markdown(
            "Which skill? Example: `@fi /scan-skill uncertainty-and-units`\n",
        );
        return;
    }
    const env = resolveRepo(stream);
    if (!env || token.isCancellationRequested) return;

    stream.progress(`Reviewing ${name}…`);
    const res = await runLaunch(env.python, env.repo, [
        "--scan-skill",
        name,
        "--json",
    ]);
    const data = parseJsonLoose(res.stdout);
    if (!data || data.error) {
        stream.markdown(
            `${data?.error ?? "Scan failed"} — run \`@fi /skills\` to see what is installed.\n`,
        );
        return;
    }

    stream.markdown(`🔍 **Static review of \`${data.name}\`**\n\n`);
    stream.markdown(
        `${data.files_reviewed} file(s) reviewed against ${data.rule_count} rules. Nothing was imported or executed — the files were parsed.\n\n`,
    );
    const findings = data.findings ?? [];
    if (findings.length === 0) {
        stream.markdown(
            "**No findings.** That is not a clean bill of health: these rules are heuristics, and a payload none of them describes would not appear here. Read the `SKILL.md` yourself before approving.\n",
        );
        return;
    }
    stream.markdown(`**${data.summary}**\n\n`);
    for (const f of findings) {
        const icon =
            f.severity === "high" ? "🔴" : f.severity === "medium" ? "🟠" : "🔵";
        stream.markdown(
            `${icon} **${f.rule}** \`${f.path}:${f.line}\`\n\n> ${f.detail}\n\n`,
        );
    }
    stream.markdown(
        "_These are heuristics, not verdicts. A skill that legitimately drives a network tool will flag `NET001`; the point is that you saw it._\n",
    );
}

/**
 * `@fi /approve-skill <name>` — the human gate, in chat.
 *
 * The review is shown *before* the prompt, because approval is a decision
 * about content. The approver's name is asked every time — prefilled from
 * `frontierInsight.approveAs`, never auto-submitted. That typing gesture is
 * the gate: an identity filled in silently would turn "a person decided"
 * into "a machine populated a field", which is exactly what the ledger's
 * no-anonymous-approver rule exists to prevent.
 */
export async function runApproveSkill(
    promptArgs: string,
    stream: vscode.ChatResponseStream,
    token: vscode.CancellationToken,
): Promise<void> {
    const name = promptArgs.trim().split(/\s+/)[0] || "";
    if (!name) {
        stream.markdown(
            "Which skill? Example: `@fi /approve-skill uncertainty-and-units`\n",
        );
        return;
    }
    const env = resolveRepo(stream);
    if (!env || token.isCancellationRequested) return;

    const scanRes = await runLaunch(env.python, env.repo, [
        "--scan-skill",
        name,
        "--json",
    ]);
    const scan = parseJsonLoose(scanRes.stdout);
    if (!scan || scan.error) {
        stream.markdown(`${scan?.error ?? "No such skill"} — run \`@fi /skills\`.\n`);
        return;
    }
    const highs = (scan.findings ?? []).filter((f: any) => f.severity === "high");
    if (highs.length) {
        stream.markdown(
            `⚠️ **${highs.length} high-severity finding(s) in \`${name}\`:**\n\n`,
        );
        for (const f of highs) {
            stream.markdown(
                `🔴 **${f.rule}** \`${f.path}:${f.line}\`\n\n> ${f.detail}\n\n`,
            );
        }
    }

    const cfg = vscode.workspace.getConfiguration("frontierInsight");
    const who = await vscode.window.showInputBox({
        title: `Approve skill: ${name}`,
        prompt: highs.length
            ? `${highs.length} high-severity finding(s) shown above. Approving anyway is a legitimate choice — enter your name to record that you decided.`
            : "Enter your name. Approval is recorded against it and binds to this exact content.",
        value: cfg.get<string>("approveAs") || "",
        ignoreFocusOut: true,
        validateInput: (v) =>
            v.trim()
                ? null
                : "The gate exists so a person decides — there is no anonymous approver.",
    });
    if (!who || !who.trim()) {
        stream.markdown("Not approved — no approver given.\n");
        return;
    }

    if (highs.length) {
        const go = await vscode.window.showWarningMessage(
            `Approve "${name}" despite ${highs.length} high-severity scan finding(s)?`,
            { modal: true },
            "Approve anyway",
        );
        if (go !== "Approve anyway") {
            stream.markdown("Not approved.\n");
            return;
        }
    }

    const args = ["--approve-skill", name, "--approve-as", who.trim()];
    if (highs.length) args.push("--despite-findings");
    const res = await runLaunch(env.python, env.repo, args);
    if (res.code === 0) {
        stream.markdown(
            `✅ **${name}** approved as \`${who.trim()}\`${
                highs.length ? " (despite findings — recorded in the ledger)" : ""
            }.\n\nEditing the skill lapses this approval and requires approving again.\n`,
        );
    } else {
        stream.markdown(
            `Approval failed (exit ${res.code}):\n\n\`\`\`\n${(
                res.stdout || res.stderr
            ).slice(-800)}\n\`\`\`\n`,
        );
    }
}

/** `@fi /revoke-skill <name>` — withdraw approval, returning it to proposed. */
export async function runRevokeSkill(
    promptArgs: string,
    stream: vscode.ChatResponseStream,
    token: vscode.CancellationToken,
): Promise<void> {
    const name = promptArgs.trim().split(/\s+/)[0] || "";
    if (!name) {
        stream.markdown("Which skill? Example: `@fi /revoke-skill <name>`\n");
        return;
    }
    const env = resolveRepo(stream);
    if (!env || token.isCancellationRequested) return;

    const res = await runLaunch(env.python, env.repo, ["--revoke-skill", name]);
    stream.markdown(
        `\`\`\`\n${(res.stdout || res.stderr).trim().slice(-600)}\n\`\`\`\n`,
    );
}

/** `@fi /import-skill` — file picker into the same importer the CLI uses. */
export async function runImportSkill(
    promptArgs: string,
    stream: vscode.ChatResponseStream,
    token: vscode.CancellationToken,
): Promise<void> {
    const env = resolveRepo(stream);
    if (!env || token.isCancellationRequested) return;

    let source = promptArgs.trim();
    if (!source) {
        const picked = await vscode.window.showOpenDialog({
            title: "Pick a skill directory (or its SKILL.md)",
            canSelectFiles: true,
            canSelectFolders: true,
            canSelectMany: false,
            openLabel: "Import skill",
        });
        if (!picked?.length) {
            stream.markdown("Import cancelled.\n");
            return;
        }
        source = picked[0].fsPath;
    }

    const domains = await vscode.window.showInputBox({
        title: "Domain tags (optional)",
        prompt:
            "Comma-separated, e.g. `genomics,biomedical`. A tagged skill joins a quest's catalogue only when the topic looks related; leave EMPTY to make it general and always offered.",
        placeHolder: "leave empty for a general skill",
        ignoreFocusOut: true,
    });
    if (domains === undefined) {
        stream.markdown("Import cancelled.\n");
        return;
    }

    stream.progress("Importing…");
    const args = ["--import-skill", source];
    if (domains.trim()) args.push("--domains", domains.trim());
    const res = await runLaunch(env.python, env.repo, args);
    const body = (res.stdout || res.stderr).trim();
    stream.markdown(
        res.code === 0
            ? `\`\`\`\n${body}\n\`\`\`\n\nNext: \`@fi /scan-skill ${""}\` to review it, then \`@fi /approve-skill\`.\n`
            : `Import failed (exit ${res.code}):\n\n\`\`\`\n${body.slice(-800)}\n\`\`\`\n`,
    );
}

/** `@fi /teach-skill <name> <module>` — scaffold from an installed library. */
export async function runTeachSkill(
    promptArgs: string,
    stream: vscode.ChatResponseStream,
    token: vscode.CancellationToken,
): Promise<void> {
    const env = resolveRepo(stream);
    if (!env || token.isCancellationRequested) return;

    const parts = promptArgs.trim().split(/\s+/).filter(Boolean);
    let name = parts[0] || "";
    let moduleName = parts[1] || "";
    if (!name) {
        name =
            (await vscode.window.showInputBox({
                title: "Teach a skill",
                prompt: "Name for the skill, e.g. `ambit`",
                ignoreFocusOut: true,
            })) || "";
        if (!name.trim()) {
            stream.markdown("Cancelled.\n");
            return;
        }
    }
    if (!moduleName) {
        moduleName =
            (await vscode.window.showInputBox({
                title: `Teach a skill: ${name}`,
                prompt:
                    "Importable module it wraps. Its real signatures are read by introspection, never generated — a confidently wrong signature is worse than none.",
                ignoreFocusOut: true,
            })) || "";
        if (!moduleName.trim()) {
            stream.markdown("Cancelled.\n");
            return;
        }
    }

    stream.progress(`Scaffolding ${name} from ${moduleName}…`);
    const res = await runLaunch(env.python, env.repo, [
        "--teach-skill",
        name.trim(),
        "--from",
        moduleName.trim(),
    ]);
    stream.markdown(
        `\`\`\`\n${(res.stdout || res.stderr).trim().slice(-2000)}\n\`\`\`\n`,
    );
}
