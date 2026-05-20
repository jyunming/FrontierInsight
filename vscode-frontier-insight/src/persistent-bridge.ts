/**
 * PersistentBridge — extension-session-long FI bridge over a
 * per-user OS-managed IPC channel.
 *
 * Distinct from :class:`Bridge` (the per-chat-command class), which
 * is created and torn down inside one ``@fi /start`` invocation and
 * speaks newline-JSON over a TCP socket on ``127.0.0.1:<random>``.
 * The PersistentBridge is started when the extension activates and
 * stays up for the whole VSCode session, so a long-running
 * ``python launch.py --serve`` (or any ``--tool`` subprocess) can
 * connect to it on demand without the user wiring a port.
 *
 * Address resolution is centralised in
 * :func:`./bridge-path.persistentBridgePath`:
 *   POSIX → ``$XDG_RUNTIME_DIR/fi-bridge.sock`` (per-user)
 *   Windows → ``\\\\.\\pipe\\fi-bridge-{USERNAME}`` (per-user)
 *
 * Scope is deliberately narrow: it handles ``lm_request`` only.
 * The dashboard has its own clarify panel, log stream, and progress
 * UI, so ``clarify_request`` / ``quest_event`` from a --serve quest
 * never flow through this bridge.
 */
import * as vscode from "vscode";
import * as fs from "fs";
import * as net from "net";
import { persistentBridgePath } from "./bridge-path";

interface LmRequest {
    type: "lm_request";
    id: number;
    node: string;
    messages: Array<{ role: string; content: string }>;
    model_hint: string;
    temperature: number;
}

// Inactivity budget for streaming chunks from ``model.sendRequest``.
// Mirrors ``bridge.ts`` (the per-chat-command transport) so both bridges
// behave identically when Copilot's HTTP/2 stream silently stalls
// mid-response. The phrasing ``bridge stalled`` is load-bearing —
// Python's ``_TRANSIENT_BRIDGE_MARKERS`` (core/provider.py) matches on
// it so tenacity retries the request instead of crashing a long quest.
const INACTIVITY_MS = 180_000;

export class PersistentBridge {
    private server: net.Server | null = null;
    private clients = new Set<net.Socket>();
    private buffers = new WeakMap<net.Socket, string>();
    private path: string;
    private outputChannel: vscode.OutputChannel;
    // True iff THIS instance successfully bound the socket. ``close()``
    // only unlinks when this is true so a window whose ``listen()``
    // refused (because another window already owned the path) won't
    // delete the live owner's socket on deactivation.
    private ownsSocket = false;

    constructor(outputChannel: vscode.OutputChannel) {
        this.path = persistentBridgePath();
        this.outputChannel = outputChannel;
    }

    /**
     * Start listening. Returns the address the Python side should
     * connect to (caller writes it to a debug log or env var the
     * spawned subprocess inherits). On POSIX this is the socket
     * path; on Windows it's the pipe name.
     *
     * If a stale socket from a crashed previous session exists, we
     * unlink it first. Windows named pipes self-clean on process
     * exit so the unlink dance is POSIX-only.
     */
    async listen(): Promise<string> {
        if (process.platform !== "win32") {
            // POSIX: clear stale socket from a previous crash, but
            // ONLY if it's stale. ``stat.isSocket()`` returns true for
            // ANY AF_UNIX file — including an active listener owned by
            // another VSCode window. Blindly unlinking would silently
            // steal the per-user bridge path: the second window's
            // ``listen()`` deletes the first window's socket, binds in
            // its place, and any new ``--serve`` / ``--tools`` Python
            // subprocess (which resolves the same per-user path) would
            // connect to the wrong window's Copilot session — silent
            // cross-window LLM routing.
            //
            // Probe order:
            //   1. ``connect()`` probe. Live → refuse to bind, throw.
            //      EACCES/EPERM/ENOTSOCK → something foreign at the
            //      path (regular file, dir, other-user socket); refuse
            //      to touch it. ENOENT/ECONNREFUSED → stale or absent.
            //   2. If reached, ``lstat`` to confirm the path is
            //      actually a socket before unlink. A regular file at
            //      the per-user path (created accidentally or
            //      maliciously) must NOT be deleted.
            const liveness = await this.probeSocket(this.path);
            if (liveness === "live") {
                const msg =
                    `Another VSCode window already owns the FI bridge socket at ${this.path}. ` +
                    `Close the other window or use a different VSCODE per-user identity.`;
                this.outputChannel.appendLine(`[fi] ${msg}`);
                throw new Error(msg);
            }
            if (liveness === "foreign") {
                const msg =
                    `Refusing to bind FI bridge socket: path ${this.path} exists but is not a usable socket ` +
                    `(permission denied or not a socket file). Inspect and remove it manually if you trust it.`;
                this.outputChannel.appendLine(`[fi] ${msg}`);
                throw new Error(msg);
            }
            // ``stale`` — probe says no live listener AND no
            // permission/non-socket signal. Confirm via lstat before
            // unlinking; only delete genuine sockets.
            try {
                const st = fs.lstatSync(this.path);
                if (st.isSocket()) {
                    fs.unlinkSync(this.path);
                }
                // Non-socket here is unexpected after probeSocket said
                // "stale" — most plausibly a TOCTOU (something replaced
                // the path between probe and lstat). Leave it alone;
                // the subsequent ``server.listen()`` will fail with
                // EADDRINUSE and the user sees a clear error.
            } catch (e) {
                const code = (e as NodeJS.ErrnoException).code;
                if (code !== "ENOENT") {
                    this.outputChannel.appendLine(
                        `[fi] lstat ${this.path} failed (${code}); leaving any existing file in place`,
                    );
                }
            }
        }
        return new Promise((resolve, reject) => {
            this.server = net.createServer((socket) => this.onClient(socket));
            this.server.on("error", reject);
            this.server.listen(this.path, () => {
                this.ownsSocket = true;
                this.outputChannel.appendLine(
                    `[fi] persistent bridge listening at ${this.path}`,
                );
                resolve(this.path);
            });
        });
    }

    /**
     * Probe what's at ``socketPath`` by attempting an IPC connect.
     * Returns one of three states the caller uses to decide whether
     * to bind, refuse, or clean up:
     *
     *   - ``"live"``    — connect succeeded; a listener is active and
     *                     this instance must NOT touch the path.
     *   - ``"foreign"`` — connect failed with EACCES / EPERM / ENOTSOCK
     *                     (something exists but we can't reach it or
     *                     it's not a socket); refuse to unlink.
     *   - ``"stale"``   — connect failed with ENOENT / ECONNREFUSED /
     *                     ECONNRESET / EADDRNOTAVAIL or no recognised
     *                     error code; path is either absent or a
     *                     crash-leftover socket safe to clean up.
     */
    private probeSocket(socketPath: string): Promise<"live" | "foreign" | "stale"> {
        return new Promise((resolve) => {
            const probe = net.createConnection(socketPath);
            const done = (result: "live" | "foreign" | "stale") => {
                probe.removeAllListeners();
                if (!probe.destroyed) probe.destroy();
                resolve(result);
            };
            probe.once("connect", () => done("live"));
            probe.once("error", (err: NodeJS.ErrnoException) => {
                // Codes that mean "path exists but we can't probe it
                // and definitely can't delete it" → foreign.
                if (err.code === "EACCES" || err.code === "EPERM" || err.code === "ENOTSOCK") {
                    done("foreign");
                } else {
                    // ENOENT / ECONNREFUSED / ECONNRESET /
                    // EADDRNOTAVAIL / anything else → safe to treat as
                    // stale (absent or crash-leftover socket).
                    done("stale");
                }
            });
        });
    }

    async close(): Promise<void> {
        for (const sock of this.clients) {
            if (!sock.destroyed) sock.destroy();
        }
        this.clients.clear();
        if (this.server) {
            await new Promise<void>((resolve) =>
                this.server!.close(() => resolve()),
            );
            this.server = null;
        }
        if (process.platform !== "win32" && this.ownsSocket) {
            // Only unlink the socket this instance bound — a window
            // whose ``listen()`` refused (because another window owned
            // the path) must not delete the live owner's socket on
            // its own deactivation.
            try { fs.unlinkSync(this.path); } catch { /* already gone */ }
            this.ownsSocket = false;
        }
    }

    private onClient(socket: net.Socket): void {
        this.clients.add(socket);
        this.buffers.set(socket, "");
        socket.on("data", (chunk: Buffer) => {
            const prev = this.buffers.get(socket) || "";
            const next = prev + chunk.toString("utf-8");
            const lines = next.split(/\r?\n/);
            this.buffers.set(socket, lines.pop() || "");
            for (const line of lines) {
                if (!line.trim()) continue;
                let msg: unknown;
                try { msg = JSON.parse(line); } catch { continue; }
                this.dispatch(socket, msg).catch((e) => {
                    this.outputChannel.appendLine(
                        `[fi] bridge handler error: ${String(e)}`,
                    );
                });
            }
        });
        socket.on("close", () => {
            this.clients.delete(socket);
            this.buffers.delete(socket);
        });
        socket.on("error", (err) => {
            this.outputChannel.appendLine(
                `[fi] persistent bridge client error: ${err.message}`,
            );
        });
    }

    private async dispatch(socket: net.Socket, msg: unknown): Promise<void> {
        if (!msg || typeof msg !== "object") return;
        const m = msg as { type?: string };
        if (m.type === "lm_request") {
            await this.handleLmRequest(socket, msg as LmRequest);
        } else if (m.type === "list_models") {
            await this.handleListModels(socket, msg as { id: number });
        }
    }

    /**
     * Answer a ``{type:"list_models", id}`` probe with the live
     * Copilot model catalog. Used by Python's
     * ``core.provider_models_discover.discover_vscode_extension`` so
     * the picker shows what the user's Copilot session actually
     * federates today (gpt-5, claude-opus-4-7, etc. plus whatever
     * else Copilot has added since), not a static trio.
     *
     * Each model contributes ``{value, label, vendor, family, version}``;
     * the Python side renders ``value`` as the picker option and
     * ``family`` / ``version`` go into the tooltip.
     */
    private async handleListModels(
        socket: net.Socket, req: { id: number },
    ): Promise<void> {
        try {
            const models = await vscode.lm.selectChatModels({ vendor: "copilot" });
            const payload = models.map((m) => ({
                value: m.id,
                label: m.id,
                vendor: m.vendor,
                family: m.family,
                version: m.version,
            }));
            this.send(socket, {
                type: "models", id: req.id, models: payload,
            });
        } catch (e) {
            const msg = e instanceof Error ? `${e.name}: ${e.message}` : String(e);
            this.send(socket, {
                type: "models_error", id: req.id, error: msg,
            });
        }
    }

    private send(socket: net.Socket, obj: unknown): void {
        if (socket.destroyed) return;
        socket.write(JSON.stringify(obj) + "\n");
    }

    private async handleLmRequest(socket: net.Socket, req: LmRequest): Promise<void> {
        try {
            // Resolve the model by *id* first, then by *family*. The
            // model picker (handleListModels) emits ``value: m.id`` so
            // the user-selected string is an id (e.g.
            // ``gemini-3-flash-preview``) — selecting by family with
            // that string misses any model whose family differs from
            // its id (every versioned/preview Copilot model). Falling
            // through to ``family`` keeps fuzzy hints like ``gpt-5``
            // working when the YAML pre-dates the picker.
            let models: vscode.LanguageModelChat[] = [];
            if (req.model_hint) {
                // Constrain to vendor=copilot on both legs — this bridge
                // exists to route FI's Copilot session, and the diagnostic
                // dump below also pins vendor=copilot, so matching against
                // any other vendor would produce inconsistent behavior
                // (and could silently pull a non-Copilot model into a
                // quest the user expects to bill against Copilot).
                models = await vscode.lm.selectChatModels({ vendor: "copilot", id: req.model_hint });
                if (!models.length) {
                    models = await vscode.lm.selectChatModels({ vendor: "copilot", family: req.model_hint });
                }
            } else {
                models = await vscode.lm.selectChatModels({ vendor: "copilot" });
            }
            if (!models.length) {
                // Self-diagnosing error: dump every available id|family
                // pair so the next user who hits this knows exactly
                // what they could have picked, without having to
                // re-instrument the extension.
                const all = await vscode.lm.selectChatModels({ vendor: "copilot" });
                const summary = all.length
                    ? all.map((m) => `${m.id}|family=${m.family}`).join(", ")
                    : "(no Copilot models exposed to this extension)";
                this.send(socket, {
                    type: "lm_error", id: req.id,
                    error: `no model matches hint=${JSON.stringify(req.model_hint)}; available=[${summary}]`,
                });
                return;
            }
            const model = models[0];
            // Align with ``bridge.ts:294-298`` exactly so chat-spawn
            // and --serve transports give the model identical context
            // for the same conversation. ``vscode.lm.*`` has only
            // User / Assistant roles (no System) — both transports
            // therefore collapse ``system`` to ``User``. If we ever
            // want to preserve system-message intent, the fix needs
            // to land in BOTH bridges (e.g., a shared helper that
            // prefixes ``[SYSTEM]`` consistently); diverging here
            // would silently change quest behaviour depending on
            // which transport the user happened to be on.
            const messages = req.messages.map((m) =>
                m.role === "assistant"
                    ? vscode.LanguageModelChatMessage.Assistant(m.content)
                    : vscode.LanguageModelChatMessage.User(m.content),
            );
            const cts = new vscode.CancellationTokenSource();
            const res = await model.sendRequest(messages, {}, cts.token);
            let content = "";
            let chunkCount = 0;
            // Race each ``iter.next()`` against an inactivity timer.
            // Without this, a silent Copilot HTTP/2 stall mid-stream
            // hangs ``for await`` forever — and combined with Python's
            // unbounded ``await fut`` in VSCodeBridgeClient.chat, the
            // entire --serve / --tools engine wedges with no recovery.
            // On stall we cancel the underlying request and emit
            // ``lm_error: bridge stalled``; the Python side's
            // ``_TRANSIENT_BRIDGE_MARKERS`` recognises that string and
            // tenacity retries the call.
            const iter = (res.text as AsyncIterable<string>)[Symbol.asyncIterator]();
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
                        // Cancel the underlying HTTP/2 request so it
                        // doesn't keep streaming into the void after we
                        // surface the error.
                        cts.cancel();
                        // Phrasing matches bridge.ts so Python's
                        // ``_is_bridge_error_transient`` classifier hits
                        // on "bridge stalled" in either transport.
                        throw new Error(
                            `bridge stalled: no chunk for ${INACTIVITY_MS / 1000} s ` +
                            `(received ${chunkCount} chunks / ${content.length} chars before stall)`,
                        );
                    }
                    if (result.done) break;
                    const fragment = result.value;
                    content += fragment;
                    chunkCount++;
                    this.send(socket, { type: "lm_chunk", id: req.id, delta: fragment });
                }
            } finally {
                cts.dispose();
            }
            this.send(socket, {
                type: "lm_done", id: req.id,
                content, total_tokens: 0,
            });
        } catch (e) {
            const msg = e instanceof Error ? `${e.name}: ${e.message}` : String(e);
            this.send(socket, { type: "lm_error", id: req.id, error: msg });
        }
    }
}
