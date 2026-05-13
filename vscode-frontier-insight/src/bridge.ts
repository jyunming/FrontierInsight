/**
 * Bridge — TCP server that translates the FI Python engine's bridge
 * protocol (newline-delimited JSON over `127.0.0.1:<port>`) into
 * `vscode.lm` Language Model API calls.
 *
 * Wire protocol (mirrors core/vscode_bridge.py on the Python side):
 *
 *   Python → extension:
 *     {type:"lm_request", id, node, messages, model_hint, temperature}
 *     {type:"quest_event", event, ...}
 *
 *   Extension → Python:
 *     {type:"lm_chunk",  id, delta}
 *     {type:"lm_done",   id, content, total_tokens}
 *     {type:"lm_error",  id, error}
 *
 * One bridge instance is created per `@fi /start` invocation. It
 * binds to a free port, spawns the Python engine with
 * `--vscode-bridge-port <N>`, and shuttles messages until the engine
 * exits or the chat command is cancelled.
 */
import * as vscode from "vscode";
import * as net from "net";
import { ChildProcess } from "child_process";

interface LmRequest {
    type: "lm_request";
    id: number;
    node: string;
    messages: Array<{ role: string; content: string }>;
    model_hint: string;
    temperature: number;
}

interface ClarifyRequest {
    type: "clarify_request";
    id: number;
    questions: Record<string, { question: string; default: unknown }>;
}

interface QuestEvent {
    type: "quest_event";
    event: string;
    [key: string]: unknown;
}

type IncomingMessage = LmRequest | ClarifyRequest | QuestEvent;

export interface BridgeOptions {
    progress: vscode.ChatResponseStream;
    cancellationToken: vscode.CancellationToken;
    /**
     * The model the user picked in VSCode's Chat picker, exposed
     * as `request.model` on the `ChatRequest` passed to the chat
     * handler. We use this as the default when the engine sends an
     * empty `model_hint` over the wire — that way the user's chosen
     * model (e.g. cheap gpt-5.4-mini at 0.33×) is honored instead
     * of `selectChatModels` returning an arbitrary `[0]` from the
     * full Copilot catalog (which silently routes calls through a
     * potentially-much-more-expensive model).
     */
    defaultModel?: vscode.LanguageModelChat;
}

export class Bridge {
    private server: net.Server | null = null;
    private socket: net.Socket | null = null;
    private buffer = "";
    private port = 0;
    private opts: BridgeOptions;

    constructor(opts: BridgeOptions) {
        this.opts = opts;
    }

    /**
     * Bind to a free port on 127.0.0.1 and resolve once the OS has
     * assigned a port number. Returns the port the spawned Python
     * process should connect back to via --vscode-bridge-port.
     */
    async listen(): Promise<number> {
        return new Promise((resolve, reject) => {
            this.server = net.createServer((socket) => {
                this.socket = socket;
                // Don't call socket.setEncoding — newer Node typings then
                // disagree about whether `data` emits `string` or Buffer.
                // We decode each chunk explicitly via .toString("utf-8").
                socket.on("data", (chunk: Buffer) =>
                    this.onData(chunk.toString("utf-8")),
                );
                socket.on("error", (err) => {
                    this.opts.progress.markdown(
                        `\n⚠️ bridge socket error: ${err.message}\n`,
                    );
                });
            });
            this.server.on("error", reject);
            // Port 0 = OS picks a free port.
            this.server.listen(0, "127.0.0.1", () => {
                const addr = this.server!.address();
                if (addr && typeof addr === "object") {
                    this.port = addr.port;
                    resolve(this.port);
                } else {
                    reject(new Error("server.address() returned null"));
                }
            });
        });
    }

    /** Watch the Python child's stderr — surface its logs in the chat
     *  panel so users can see what FI is doing under the hood.
     *
     *  Typed against `ChildProcess` (the loose interface) rather than
     *  `ChildProcessWithoutNullStreams` because we spawn with
     *  `stdio: ["ignore", "pipe", "pipe"]` — stdin is null on purpose
     *  (Python reads its config from disk, not stdin).
     *
     *  `rawLineSink` (optional) receives every stderr line, regardless
     *  of whether it matches a node tag. The caller uses it to keep a
     *  rolling tail buffer so a Python crash can be diagnosed from
     *  the actual traceback (which won't appear in run.log). */
    attachChild(child: ChildProcess, rawLineSink?: (line: string) => void): void {
        if (!child.stderr) {
            // Spawn() with stdio=["ignore","pipe","pipe"] always gives
            // us stderr; this guard is for the type-system only.
            return;
        }
        // Stream Python stderr (where FI's logging goes) into the chat
        // pane. We don't render every line — that would drown the user.
        // Instead, surface lines starting with a recognized node tag
        // (e.g. "[ideate]"), which are the user-meaningful progress
        // markers from the engine.
        child.stderr.setEncoding("utf-8");
        let pending = "";
        child.stderr.on("data", (chunk: string) => {
            pending += chunk;
            const lines = pending.split(/\r?\n/);
            pending = lines.pop() || "";
            for (const line of lines) {
                if (rawLineSink) rawLineSink(line);
                if (/\[(clarify|ideate|literature|design|implement|execute|execute_reflect|analyze|cross_check|write|review)\]/.test(line)) {
                    // Strip the timestamp + log-level prefix to keep
                    // the chat clean.
                    const m = line.match(/(\[[a-z_]+\].*)$/);
                    if (m) this.opts.progress.markdown(`  ${m[1]}\n\n`);
                }
            }
        });
    }

    async close(): Promise<void> {
        if (this.socket && !this.socket.destroyed) {
            this.socket.destroy();
        }
        if (this.server) {
            await new Promise<void>((resolve) =>
                this.server!.close(() => resolve()),
            );
        }
    }

    private onData(chunk: string): void {
        this.buffer += chunk;
        let idx: number;
        while ((idx = this.buffer.indexOf("\n")) >= 0) {
            const line = this.buffer.slice(0, idx);
            this.buffer = this.buffer.slice(idx + 1);
            if (!line.trim()) continue;
            let msg: IncomingMessage;
            try {
                msg = JSON.parse(line);
            } catch {
                continue;
            }
            this.handleMessage(msg).catch((e) => {
                this.opts.progress.markdown(
                    `\n⚠️ bridge handler error: ${String(e)}\n`,
                );
            });
        }
    }

    private async handleMessage(msg: IncomingMessage): Promise<void> {
        if (msg.type === "lm_request") {
            await this.handleLmRequest(msg);
        } else if (msg.type === "clarify_request") {
            await this.handleClarifyRequest(msg);
        } else if (msg.type === "quest_event") {
            this.handleQuestEvent(msg);
        }
    }

    /**
     * Render the engine's pre-flight clarify questions as a sequence
     * of `showInputBox` modals. The user can press Esc on any modal
     * to cancel; we then send `clarify_cancelled` back and the Python
     * side raises BridgeError (which the engine surfaces as a quest
     * failure rather than hanging).
     */
    private async handleClarifyRequest(req: ClarifyRequest): Promise<void> {
        this.opts.progress.markdown(
            "\n🟡 **Pre-flight clarification** — the agent has questions to sharpen the run.\n\n",
        );
        const answers: Record<string, unknown> = {};
        const keys = Object.keys(req.questions);
        for (const key of keys) {
            const entry = req.questions[key];
            const defaultIsList = Array.isArray(entry.default);
            const defaultStr = defaultIsList
                ? (entry.default as unknown[]).join(", ")
                : String(entry.default ?? "");
            const value = await vscode.window.showInputBox({
                title: `Clarify (${key})`,
                prompt: entry.question || key,
                value: defaultStr,
                ignoreFocusOut: true,
            });
            if (value === undefined) {
                // User pressed Esc.
                this.send({
                    type: "clarify_cancelled",
                    id: req.id,
                });
                this.opts.progress.markdown(
                    "\n— clarify cancelled by user; quest will fail.\n",
                );
                return;
            }
            // Re-encode list-typed defaults from the comma-separated
            // string the modal returns.
            if (defaultIsList) {
                answers[key] = value
                    .split(",")
                    .map((s) => s.trim())
                    .filter((s) => s.length > 0);
            } else {
                answers[key] = value;
            }
            this.opts.progress.markdown(
                `  **${key}:** ${typeof answers[key] === "string" ? answers[key] : JSON.stringify(answers[key])}\n\n`,
            );
        }
        this.send({
            type: "clarify_response",
            id: req.id,
            answers,
        });
        this.opts.progress.markdown("\n— clarify answers submitted; resuming…\n\n");
    }

    /**
     * Translate one `lm_request` into a real `vscode.lm` call. The
     * model is selected by `model_hint` (Phase O per-node routing) —
     * we map common Copilot family names to selectChatModels filters.
     * Empty hint = use the user's currently-selected chat model.
     *
     * Fallback ladder when the hint produces zero matches:
     *   1. Try `selectChatModels({vendor: "copilot", family: hint})`.
     *   2. If empty, retry `selectChatModels({vendor: "copilot"})`.
     *   3. If still empty, retry `selectChatModels({})` (any vendor).
     *   4. Only then `lm_error` back to Python.
     *
     * Without this ladder, a stale or mis-spelled model name in a
     * YAML's `provider.node_models` would crash the whole quest on
     * one node's first call. The fallback keeps the quest running
     * with a model that exists; the user sees which one was picked
     * via a chat-panel note so they can correct the YAML.
     */
    private async handleLmRequest(req: LmRequest): Promise<void> {
        try {
            const model = await this.pickModel(req.model_hint);
            if (!model) {
                this.send({
                    type: "lm_error",
                    id: req.id,
                    error:
                        `no language model is available in this VSCode window. ` +
                        `Sign into Copilot Chat (or another LM provider) and ` +
                        `verify the Chat picker has at least one model selected.`,
                });
                return;
            }

            // Translate FI's OpenAI-shaped messages into vscode.LanguageModelChatMessage.
            const chatMessages = req.messages.map((m) =>
                m.role === "assistant"
                    ? vscode.LanguageModelChatMessage.Assistant(m.content)
                    : vscode.LanguageModelChatMessage.User(m.content),
            );

            // Send with bounded retry. Copilot's backend occasionally
            // returns transient HTTP/2 protocol errors mid-stream
            // ("net::ERR_HTTP2_PROTOCOL_ERROR", connection resets, 5xx
            // responses). Without retry here, one network blip on the
            // implement-node call kills a 5-minute quest. The HTTP and
            // CLI transports on the Python side already retry; the
            // bridge needs the same.
            const response = await this.sendWithRetry(
                model,
                chatMessages,
                req.node,
            );

            // Stream the response text back as `lm_chunk` events,
            // then a final `lm_done`.
            //
            // Two reasons we don't just `for await (... of response.text)`:
            //
            //  1. **Per-chunk visibility.** Without surfacing fragments to
            //     the chat panel, the user sees only VSCode's generic
            //     "Reasoning" indicator and can't tell whether the model
            //     is mid-stream, stalled, or wedged. We render a single
            //     heartbeat line that updates as chunks land.
            //
            //  2. **Inactivity timeout.** `model.sendRequest` has no
            //     built-in deadline; if Copilot's HTTP/2 silently stalls
            //     mid-stream the iteration hangs forever. We race each
            //     `next()` against a 180 s timer and treat a no-chunk
            //     gap that long as a stall — surface as `lm_error` with
            //     a marker Python's retry classifier recognizes so the
            //     Python side will try the request again.
            const INACTIVITY_MS = 180_000;
            const HEARTBEAT_MS = 10_000;
            let accumulated = "";
            let chunkCount = 0;
            let chars = 0;
            const startMs = Date.now();
            const iter = response.text[Symbol.asyncIterator]();
            // Use a non-empty fallback so an upstream caller that didn't
            // set `node` (e.g. older Python that hasn't been updated to
            // thread node through `LLMClient.chat`) doesn't render as
            // empty backticks like `` ``.
            const nodeLabel = req.node || "(unnamed-node)";

            // The bot review on PR #30 caught a real flaw in v1: a
            // heartbeat that only fires on chunk arrival is silent
            // during pre-first-token reasoning, which is exactly the
            // window the user is most anxious about ("is anything
            // happening?"). Run a wall-clock interval that fires
            // every HEARTBEAT_MS regardless of stream state.
            const heartbeatTimer = setInterval(() => {
                const elapsed = Math.round((Date.now() - startMs) / 1000);
                const label = chunkCount === 0 ? "reasoning (no chunks yet)" : "streaming";
                this.opts.progress.markdown(
                    `  📥 \`${nodeLabel}\` ${label} — ` +
                    `${chunkCount} chunks, ${chars} chars, ${elapsed} s\n\n`,
                );
            }, HEARTBEAT_MS);
            this.opts.progress.markdown(
                `  ⏳ \`${nodeLabel}\` streaming…\n\n`,
            );
            try {
                while (true) {
                    let timer: ReturnType<typeof setTimeout> | undefined;
                    const stallSignal: Promise<{ stalled: true }> = new Promise((resolve) => {
                        timer = setTimeout(
                            () => resolve({ stalled: true }),
                            INACTIVITY_MS,
                        );
                    });
                    let result: IteratorResult<string> | { stalled: true };
                    try {
                        result = await Promise.race([iter.next(), stallSignal]);
                    } finally {
                        if (timer) clearTimeout(timer);
                    }
                    if ("stalled" in result) {
                        // Phrasing matters: Python's _is_bridge_error_transient
                        // pattern-matches strings; "bridge stalled" must be
                        // in its transient-marker list so the Python side
                        // retries instead of dying on the first stall.
                        throw new Error(
                            `bridge stalled: no chunk for ${INACTIVITY_MS / 1000} s ` +
                            `(received ${chunkCount} chunks / ${chars} chars before stall)`,
                        );
                    }
                    if (result.done) break;
                    const fragment = result.value;
                    accumulated += fragment;
                    chars += fragment.length;
                    chunkCount++;
                    this.send({
                        type: "lm_chunk",
                        id: req.id,
                        delta: fragment,
                    });
                }
            } finally {
                clearInterval(heartbeatTimer);
            }
            const totalElapsed = Math.round((Date.now() - startMs) / 1000);
            this.opts.progress.markdown(
                `  ✅ \`${nodeLabel}\` done — ${chunkCount} chunks, ${chars} chars, ${totalElapsed} s\n\n`,
            );
            this.send({
                type: "lm_done",
                id: req.id,
                content: accumulated,
            });
        } catch (e) {
            const msg = e instanceof Error ? e.message : String(e);
            this.send({
                type: "lm_error",
                id: req.id,
                error: msg,
            });
        }
    }

    private handleQuestEvent(ev: QuestEvent): void {
        // Render quest-level milestones in the chat panel.
        if (ev.event === "node_started") {
            this.opts.progress.markdown(`  → **${ev.node}**\n\n`);
        } else if (ev.event === "quest_done") {
            this.opts.progress.markdown(
                `\n✅ Quest finished. Paper: \`${ev.paper_path}\`\n`,
            );
        } else if (ev.event === "quest_failed") {
            this.opts.progress.markdown(
                `\n❌ Quest failed: ${ev.reason}\n`,
            );
        }
    }

    private send(msg: object): void {
        if (this.socket && !this.socket.destroyed) {
            this.socket.write(JSON.stringify(msg) + "\n");
        }
    }

    /**
     * Map FI's per-node `model_hint` (Phase O strings like "gpt-5",
     * "claude-opus-4-7", "gemini-2.5-pro") to a `selectChatModels`
     * family filter. Conservative: when in doubt, fall through to
     * vendor=copilot with no family filter and let VSCode pick.
     */
    private familyFromHint(hint: string): string | undefined {
        if (!hint) return undefined;
        const h = hint.toLowerCase();
        // Heuristic prefix match — VSCode's family strings change
        // (gpt-4o, gpt-5, claude-sonnet-4-6, ...) so we don't hardcode
        // every one; we pass the hint through as a family filter and
        // selectChatModels does fuzzy matching.
        return h;
    }

    /**
     * Send a chat request with bounded exponential-backoff retry on
     * transient errors. Mirrors the tenacity-based retry that the
     * Python HTTP and CLI transports already do. Without this, a
     * single mid-stream `net::ERR_HTTP2_PROTOCOL_ERROR` (which we've
     * observed in real Copilot calls) kills the whole quest.
     *
     * Up to 4 attempts. Backoff: 2 s, 4 s, 8 s (capped at 20 s).
     * Only retry on error messages that look transient — we don't
     * want to retry, say, an auth error 4 times. Non-transient errors
     * propagate immediately.
     */
    private async sendWithRetry(
        model: vscode.LanguageModelChat,
        messages: vscode.LanguageModelChatMessage[],
        nodeName: string,
    ): Promise<vscode.LanguageModelChatResponse> {
        const MAX_ATTEMPTS = 4;
        let lastErr: Error | undefined;
        for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
            try {
                return await model.sendRequest(messages, {}, this.opts.cancellationToken);
            } catch (e) {
                const err = e instanceof Error ? e : new Error(String(e));
                lastErr = err;
                if (this.opts.cancellationToken.isCancellationRequested) {
                    throw err;  // user clicked cancel — don't retry
                }
                if (!this.isTransient(err.message)) {
                    throw err;  // non-transient (auth, quota, etc.) — surface immediately
                }
                if (attempt === MAX_ATTEMPTS) {
                    throw err;
                }
                // Exponential backoff: 2, 4, 8 (capped at 20) seconds.
                const wait_s = Math.min(2 ** attempt, 20);
                this.opts.progress.markdown(
                    `\n⏳ transient bridge error on \`${nodeName}\` (attempt ${attempt}/${MAX_ATTEMPTS}); ` +
                    `retrying in ${wait_s} s. Cause: \`${err.message.slice(0, 120)}\`\n\n`,
                );
                await new Promise((r) => setTimeout(r, wait_s * 1000));
            }
        }
        // Unreachable — the loop either returns or throws.
        throw lastErr ?? new Error("sendWithRetry exhausted without raising");
    }

    /**
     * Classify whether a `sendRequest` error message is worth retrying.
     * We pattern-match against well-known transient markers Copilot's
     * backend emits — HTTP/2 protocol errors, connection resets, 5xx
     * responses, generic "network connection" prompts. Auth errors and
     * "model not found" failures are NOT retried.
     */
    private isTransient(msg: string): boolean {
        const m = msg.toLowerCase();
        const transientMarkers = [
            "net::err_http2",
            "net::err_connection",
            "net::err_network",
            "err_http2_protocol_error",
            "econnreset",
            "etimedout",
            "socket hang up",
            "503",
            "504",
            "502",
            "network connection",
            "firewall rules and network",
            "temporarily unavailable",
            "rate limit",   // rate-limit usually clears with backoff
            "request failed",
        ];
        return transientMarkers.some((marker) => m.includes(marker));
    }

    /**
     * Resolve which `LanguageModelChat` to call for this request.
     *
     * **Critical for cost discipline**: when `model_hint` is empty,
     * we use `opts.defaultModel` (the model the user picked in the
     * VSCode Chat picker) — NOT `selectChatModels({vendor: "copilot"})[0]`,
     * which would return an arbitrary entry from the full catalog
     * and could silently route calls through a much-more-expensive
     * model (Claude Opus at 5× vs gpt-5.4-mini at 0.33× per request).
     *
     * When `model_hint` is set, we try `selectChatModels({family: hint})`
     * to honor per-node routing (Phase O). If nothing matches, fall
     * back to the user's chat-picker selection so the quest keeps
     * running — and tell them which model was used.
     */
    private async pickModel(hint: string): Promise<vscode.LanguageModelChat | undefined> {
        const family = this.familyFromHint(hint);
        if (!family) {
            // No hint → use the model the user picked in the Chat
            // picker. This is the case for every interview-generated
            // quest that didn't set `provider.node_models`.
            if (this.opts.defaultModel) {
                return this.opts.defaultModel;
            }
            // Defensive: chat handlers always receive request.model,
            // but if for some reason we got here without one, pick
            // SOMETHING.
            const any = await vscode.lm.selectChatModels({ vendor: "copilot" });
            return any[0];
        }
        // Hint set → try the family filter first.
        const matched = await vscode.lm.selectChatModels({
            vendor: "copilot", family,
        });
        if (matched.length > 0) {
            return matched[0];
        }
        // Hint didn't match anything in the user's subscription.
        // Fall back to their chat-picker selection rather than
        // silently grabbing whatever Copilot exposes.
        this.opts.progress.markdown(
            `\n⚠️ no Copilot model matches hint \`${hint}\` — using your Chat-picker selection instead\n\n`,
        );
        if (this.opts.defaultModel) {
            return this.opts.defaultModel;
        }
        // Last-ditch: anything we can find.
        const fallback = await vscode.lm.selectChatModels({ vendor: "copilot" });
        return fallback[0];
    }
}
