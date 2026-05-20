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

export class PersistentBridge {
    private server: net.Server | null = null;
    private clients = new Set<net.Socket>();
    private buffers = new WeakMap<net.Socket, string>();
    private path: string;
    private outputChannel: vscode.OutputChannel;

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
            // Probe by trying to ``connect()`` first:
            //   - connect succeeds → another listener is live → refuse
            //   - connect fails (ENOENT, ECONNREFUSED, ...) → path is
            //     either stale or never existed → safe to unlink + bind.
            if (await this.isSocketLive(this.path)) {
                const msg =
                    `Another VSCode window already owns the FI bridge socket at ${this.path}. ` +
                    `Close the other window or use a different VSCODE per-user identity.`;
                this.outputChannel.appendLine(`[fi] ${msg}`);
                throw new Error(msg);
            }
            try { fs.unlinkSync(this.path); } catch { /* nothing to clean */ }
        }
        return new Promise((resolve, reject) => {
            this.server = net.createServer((socket) => this.onClient(socket));
            this.server.on("error", reject);
            this.server.listen(this.path, () => {
                this.outputChannel.appendLine(
                    `[fi] persistent bridge listening at ${this.path}`,
                );
                resolve(this.path);
            });
        });
    }

    /**
     * Probe whether ``socketPath`` has an active listener. Resolves
     * ``true`` iff a TCP-level connect succeeds (someone is accepting);
     * ``false`` on ENOENT, ECONNREFUSED, or any other connect failure
     * (path stale or never bound).
     *
     * Used only by ``listen()`` to distinguish a crash-leftover socket
     * from a live cross-window owner before unlinking.
     */
    private isSocketLive(socketPath: string): Promise<boolean> {
        return new Promise((resolve) => {
            const probe = net.createConnection(socketPath);
            const done = (alive: boolean) => {
                probe.removeAllListeners();
                if (!probe.destroyed) probe.destroy();
                resolve(alive);
            };
            probe.once("connect", () => done(true));
            probe.once("error", () => done(false));
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
        if (process.platform !== "win32") {
            try { fs.unlinkSync(this.path); } catch { /* already gone */ }
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
            const messages = req.messages.map((m) =>
                m.role === "system"
                    ? vscode.LanguageModelChatMessage.User(`SYSTEM: ${m.content}`)
                    : vscode.LanguageModelChatMessage.User(m.content),
            );
            const cts = new vscode.CancellationTokenSource();
            const res = await model.sendRequest(messages, {}, cts.token);
            let content = "";
            for await (const fragment of res.text) {
                content += fragment;
                this.send(socket, { type: "lm_chunk", id: req.id, delta: fragment });
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
