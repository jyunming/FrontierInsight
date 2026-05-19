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
            // POSIX: clear stale socket from a previous crash. The
            // OS won't bind() to an existing path; we test for the
            // socket-or-nothing rather than blindly unlinking so we
            // never delete an active listener someone else owns.
            try {
                const stat = fs.statSync(this.path);
                if (stat.isSocket()) fs.unlinkSync(this.path);
            } catch {
                // path doesn't exist; nothing to clean
            }
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
        if (m.type !== "lm_request") return; // narrow scope on purpose
        await this.handleLmRequest(socket, msg as LmRequest);
    }

    private send(socket: net.Socket, obj: unknown): void {
        if (socket.destroyed) return;
        socket.write(JSON.stringify(obj) + "\n");
    }

    private async handleLmRequest(socket: net.Socket, req: LmRequest): Promise<void> {
        try {
            const filter: vscode.LanguageModelChatSelector = req.model_hint
                ? { family: req.model_hint }
                : { vendor: "copilot" };
            const models = await vscode.lm.selectChatModels(filter);
            if (!models.length) {
                this.send(socket, {
                    type: "lm_error", id: req.id,
                    error: `no model matches hint=${JSON.stringify(req.model_hint)}`,
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
