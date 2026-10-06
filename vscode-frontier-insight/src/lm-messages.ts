/**
 * Chat messages from FI, turned into `vscode.LanguageModelChatMessage`s.
 *
 * FI sends OpenAI-shaped messages over the bridge. A message's content is a
 * string, or a list of parts when it carries screenshots:
 *
 *   {type: "text", text: "..."}
 *   {type: "image_url", image_url: {url: "data:image/png;base64,..."}}
 *
 * Images need `LanguageModelDataPart.image`, which older VS Code builds do
 * not have, so it is looked up at run time and a message with images fails
 * with a clear error instead of silently losing them. The vscode API is
 * passed in rather than imported so this can be tested under plain node.
 */

export type BridgeContentPart =
    | { type: "text"; text: string }
    | { type: "image_url"; image_url: { url: string } | string };

export interface BridgeMessage {
    role: string;
    content: string | BridgeContentPart[];
}

export interface ChatMessageApi {
    LanguageModelChatMessage: {
        User(content: string | any[]): unknown;
        Assistant(content: string | any[]): unknown;
    };
    LanguageModelTextPart: new (value: string) => unknown;
    LanguageModelDataPart?: { image?: (data: Uint8Array, mime: string) => unknown };
}

export const IMAGES_UNSUPPORTED =
    "this VS Code version cannot send images to a language model " +
    "(LanguageModelDataPart.image is missing); update VS Code to use the visual check";

const DATA_URL = /^data:(image\/[\w.+-]+);base64,(.+)$/s;

export function toChatMessages<T>(messages: BridgeMessage[], api: ChatMessageApi): T[] {
    return messages.map((m) => {
        const factory = api.LanguageModelChatMessage;
        const make = (content: string | unknown[]) =>
            (m.role === "assistant" ? factory.Assistant(content) : factory.User(content)) as T;
        if (typeof m.content === "string") {
            return make(m.content);
        }
        const parts: unknown[] = [];
        for (const part of m.content) {
            if (part.type === "text") {
                parts.push(new api.LanguageModelTextPart(part.text));
                continue;
            }
            const url = typeof part.image_url === "string" ? part.image_url : part.image_url?.url ?? "";
            const match = DATA_URL.exec(url);
            if (!match) {
                continue;
            }
            const image = api.LanguageModelDataPart?.image;
            if (typeof image !== "function") {
                throw new Error(IMAGES_UNSUPPORTED);
            }
            parts.push(image.call(api.LanguageModelDataPart, new Uint8Array(Buffer.from(match[2], "base64")), match[1]));
        }
        return make(parts);
    });
}

/** The identity of the chat model the extension selected and sent the request to, as its `vscode.lm` model object
 * states it. Sent on `lm_done` so FI can record it (the selected model as VS Code reports it; not a proof beyond
 * that). Fields VS Code leaves empty are left out, and a router alias ("auto") names no model, so nothing is sent. */
export interface ServedModel {
    id?: string;
    vendor?: string;
    family?: string;
    version?: string;
    name?: string;
}

export function servedModel(model: ServedModel | undefined | null): ServedModel | undefined {
    if (!model) {
        return undefined;
    }
    const out: ServedModel = {};
    for (const key of ["id", "vendor", "family", "version", "name"] as const) {
        const value = model[key];
        if (typeof value === "string" && value.trim()) {
            out[key] = value;
        }
    }
    const alias = (s?: string) => ["auto", "default", "copilot-auto", "auto-mode"].includes((s ?? "").trim().toLowerCase());
    if (!out.id || alias(out.id) || alias(out.family)) {
        return undefined;
    }
    return out;
}

/** The launch arguments that hand FI the model picked in the chat panel, which wins over the config's `provider.model`
 * for a quest started or resumed from the chat (`launch.py --vscode-chat-model`). A router alias ("auto") names no
 * model, so nothing is passed and the config's model stays. */
export function chatModelArgs(model: ServedModel | undefined | null): string[] {
    const named = servedModel(model);
    if (!named?.id) return [];
    // The family too, so a config that names the same model by its family is not taken for a change of model.
    const family = named.family && named.family !== named.id ? ["--vscode-chat-model-family", named.family] : [];
    return ["--vscode-chat-model", named.id, ...family];
}

/** What one part of a streamed response is. The vscode API is passed in so this runs under plain node. */
export function partKind(
    api: any,
    p: unknown,
    unknownStringAs: "text" | "unknown" = "unknown",
): "text" | "thinking" | "tool" | "unknown" {
    const TextPart = api?.LanguageModelTextPart;
    const ThinkingPart = api?.LanguageModelThinkingPart;
    const ToolCallPart = api?.LanguageModelToolCallPart;
    if (TextPart && p instanceof TextPart) return "text";
    if (ThinkingPart && p instanceof ThinkingPart) return "thinking";
    if (ToolCallPart && p instanceof ToolCallPart) return "tool";
    // Versions that do not export the classes still give the parts a usable shape.
    const obj = p as any;
    if (obj && Array.isArray(obj.value)) {
        // Only a thinking part's value may be a list of strings (a summary in several parts).
        return obj.constructor?.name === "LanguageModelThinkingPart" ? "thinking" : "unknown";
    }
    if (obj && typeof obj.value === "string") {
        if (obj.constructor?.name === "LanguageModelThinkingPart") return "thinking";
        if (obj.constructor?.name === "LanguageModelTextPart") return "text";
        // A part that names itself as neither is not taken for the answer unless the caller says to.
        return unknownStringAs;
    }
    return "unknown";
}

/** A thinking part's text. The (proposed) API types its `value` as a string or a list of strings. A list is a GPT
 * reasoning summary sent whole, one paragraph per item, so it is joined with a blank line, as Copilot's own chat does.
 * Anything else is not text (`undefined`). */
export function thinkingText(value: unknown): string | undefined {
    if (typeof value === "string") return value;
    if (Array.isArray(value) && value.every((v) => typeof v === "string")) return value.join("\n\n");
    return undefined;
}

/**
 * Asking the chat model for its reasoning.
 *
 * Copilot turns on a Claude model's extended thinking only when the request carries the model option
 * `_enableThinking: true`, and then streams a summary of the reasoning (not the full text) as thinking parts. The option
 * is Copilot's own, undocumented and internal: it may be renamed or stop working in any Copilot release. FI asks by
 * default (Python sends `ask_thinking: false` when the quest keeps no reasoning, `output.save_thinking: false`); a model
 * that refuses the option is asked again once without it, and not asked again for as long as the bridge runs.
 *
 * GPT models (OpenAI's Responses API) keep their reasoning in an encrypted reasoning item; Copilot passes it to another
 * extension only when the request carries `includeEncryptedThinking: true` (a request option VS Code hands to the model
 * provider as it is: extHostLanguageModels.ts, vscode.proposed.chatProvider.d.ts), and then as a thinking part whose
 * metadata holds the encrypted state and whose text is the readable summary when there is one. Measured on Copilot
 * Chat 0.68 (VS Code 1.140): Copilot's own request to OpenAI never asks for `reasoning.summary`, so that text is empty
 * and FI gets no GPT reasoning text, whatever FI sends. The option stays: it is what delivers the text if Copilot starts
 * asking for the summary. FI keeps the readable text only, never the encrypted state, and counts the parts that came
 * with no text (`ThinkingCollector.emptyParts`) so it can say so (`thinking_parts_empty` on `lm_done`).
 */
export const THINKING_REQUEST_OPTIONS = { modelOptions: { _enableThinking: true }, includeEncryptedThinking: true };
/** The version of what this extension tells FI over the bridge. Raise it by one when FI starts to rely on something an
 * older extension does not do (asking for the model's reasoning was the first). It rides on every `lm_done`, on the
 * `models` reply and in the Output line "persistent bridge listening"; an older extension sends none, which FI reads as
 * 0 and says once that the extension is older than this FI (core/vscode_bridge.py REQUIRED_BRIDGE_PROTOCOL, which a
 * test keeps equal to this). It is a number kept by hand, not a build stamp: a stamp that changes on every package would
 * make the committed .vsix manifest differ on every rebuild. */
export const BRIDGE_PROTOCOL = 1;

/** The error for a model that sent no part for `seconds`. It names who went silent, and a model whose VS Code
 * extension forwards only the answer (the Ollama extension reports no part while a model is still thinking) is said to
 * look the same whether it is slow or stuck. "bridge stalled" is the phrase FI retries on. */
export function stallMessage(
    model: { vendor?: string; id?: string } | undefined, seconds: number, detail: string,
): string {
    const who = model?.id ? ` from ${model.vendor ? model.vendor + "/" : ""}${model.id}` : "";
    return `bridge stalled: no part${who} for ${seconds} s (${detail})`;
}

/** Words FI's Python side retries on (core/provider.py `_TRANSIENT_BRIDGE_MARKERS`): the stream failed after the model
 * was asked for its reasoning, so the call is made again without asking. */
export const THINKING_DECLINED_MARKER = "the model did not accept the request for its reasoning";

const TRANSIENT_MARKERS = [
    "net::err_http2", "net::err_connection", "net::err_network", "err_http2_protocol_error", "econnreset", "etimedout",
    "socket hang up", "503", "504", "502", "network connection", "firewall rules and network",
    "temporarily unavailable", "rate limit", "request failed", "stalled", "cancel",
];

/** An error that is about the connection (worth the same request again), not about what was asked. */
export function looksTransient(message: string): boolean {
    const m = message.toLowerCase();
    return TRANSIENT_MARKERS.some((marker) => m.includes(marker));
}

function errorText(e: unknown): string {
    return e instanceof Error ? e.message : String(e);
}

// A `vscode.LanguageModelError` with one of these codes is about access (consent, a blocked request, a missing model),
// never about the option, so it is not read as a refusal of it.
const ACCESS_ERROR_CODES = ["NoPermissions", "Blocked", "NotFound"];

export class ThinkingRequests {
    // Models that refused the option: asked without it from then on.
    private readonly declined = new Map<string, string>();
    // Models whose answer failed before its first part after being asked: asked without it next time, and taken as a
    // refusal only if that request then goes through.
    private readonly tentative = new Map<string, string>();

    constructor(private readonly isTransient: (message: string) => boolean = looksTransient) {}

    /** Why `modelKey` refused the request for its reasoning, when it did. */
    declinedReason(modelKey: string): string | undefined {
        return this.declined.get(modelKey);
    }

    private notARefusal(e: unknown, message: string, isCancelled: () => boolean): boolean {
        const code = (e as { code?: unknown } | null)?.code;
        return isCancelled() || this.isTransient(message)
            || (typeof code === "string" && ACCESS_ERROR_CODES.includes(code));
    }

    /** Send one request, asking for the model's reasoning when `ask` and the model has not refused it before. When the
     * request with the option fails (anything but a connection error, a cancel or an access error), it is sent once
     * more without it; only if that one goes through is the failure taken as the model refusing the option (then it
     * is not asked again). If it fails too, the first error is raised and nothing is remembered: the failure was not
     * about the option (a quota, a prompt too long). */
    async send<R>(
        modelKey: string,
        ask: boolean,
        send: (options: object) => PromiseLike<R>,
        isCancelled: () => boolean = () => false,
    ): Promise<{ response: R; asked: boolean }> {
        if (!ask || this.declined.has(modelKey)) {
            return { response: await send({}), asked: false };
        }
        if (this.tentative.has(modelKey)) {
            // Asked without the option after an answer failed before its first part: a refusal only once this answer's
            // first part arrives (`firstPart`); failing without the option too, it was not about the option.
            try {
                return { response: await send({}), asked: false };
            } catch (e) {
                this.tentative.delete(modelKey);
                throw e;
            }
        }
        try {
            return { response: await send(THINKING_REQUEST_OPTIONS), asked: true };
        } catch (e) {
            const message = errorText(e);
            if (this.notARefusal(e, message, isCancelled)) throw e;
            let response: R;
            try {
                response = await send({});
            } catch {
                throw e;
            }
            this.declined.set(modelKey, message.slice(0, 300));
            return { response, asked: false };
        }
    }

    /** A stream that failed before any part arrived, on a request that asked for the reasoning: the model may have
     * refused the option only once it began to answer. The next request to it goes without the option (a refusal is
     * recorded only when that answer's first part arrives, {@link firstPart}), and the error returned names
     * {@link THINKING_DECLINED_MARKER}, so FI makes the call again. Any other failure is returned as it was. */
    streamFailed(modelKey: string, asked: boolean, partsSeen: number, message: string, cancelled = false): string {
        if (!asked) {
            // The same failure without the option: it was not the option (a request that did not ask for the
            // reasoning at all, `ask` false, leaves a pending mark as it is).
            if (partsSeen === 0 && this.tentative.has(modelKey) && !cancelled && !this.isTransient(message)) {
                this.tentative.delete(modelKey);
            }
            return message;
        }
        if (partsSeen > 0 || cancelled || this.isTransient(message)) return message;
        this.tentative.set(modelKey, message.slice(0, 300));
        return `${THINKING_DECLINED_MARKER} (${message.slice(0, 300)}); asking again without it`;
    }

    /** The first part of an answer arrived. On a request made without the option after an answer that asked for the
     * reasoning failed before its first part, that failure was the model refusing the option: it is not asked again. */
    firstPart(modelKey: string, asked: boolean): void {
        const pending = this.tentative.get(modelKey);
        if (!asked && pending !== undefined) {
            this.declined.set(modelKey, pending);
            this.tentative.delete(modelKey);
        }
    }
}

// One message is one line, and an FI that has not raised its reader's 64 KiB line limit drops the connection on a
// longer one, so the whole `lm_done` line stays under this.
export const LM_DONE_MAX_BYTES = 48 * 1024;
// The one-line "not sent" marker (about 40 bytes) may go past the limit above, up to here, so an answer that fills
// the limit still says its reasoning was left out; this stays clear of the 64 KiB reader limit.
export const LM_DONE_HARD_BYTES = 60 * 1024;

/** The reasoning of one answer as it streams in: only what could ever be sent is kept, the rest is only counted. */
export class ThinkingCollector {
    text = "";
    total = 0;
    /** Reasoning parts that arrived with no text in them (a GPT model's reasoning comes encrypted only). */
    emptyParts = 0;
    add(fragment: string, keepChars: number = LM_DONE_MAX_BYTES): void {
        if (!fragment) this.emptyParts++;
        this.total += fragment.length;
        if (this.text.length < keepChars) this.text += fragment.slice(0, keepChars - this.text.length);
    }
}

/**
 * The `lm_done` message for an answer, carrying the model's reasoning (`thinking`) when there is room for it: the
 * reasoning is cut to what fits and says how much was left out; when the answer alone leaves no room, a short marker
 * that says so is sent in its place. The answer itself is never cut here.
 * `total` is how many characters the model produced when `thinking` holds only the start of them.
 */
export function lmDoneMessage<T extends object>(
    base: T,
    thinking: string,
    maxBytes: number = LM_DONE_MAX_BYTES,
    total: number = thinking.length,
): T {
    if (!thinking) return base;
    const size = (t: string) => Buffer.byteLength(JSON.stringify({ ...base, thinking: t }), "utf8") + 1;
    if (total === thinking.length && size(thinking) <= maxBytes) return { ...base, thinking };
    const withNote = (keep: number) =>
        thinking.slice(0, keep) + `\n[${total - keep} more characters not sent]`;
    if (size(withNote(0)) > maxBytes) {
        // The marker is a fixed few dozen bytes, so it may use the room between the limit and the hard limit; an
        // answer that fills even that sends no reasoning at all.
        const marker = `[${total} chars not sent: no room]`;
        return size(marker) <= Math.max(maxBytes, LM_DONE_HARD_BYTES) ? { ...base, thinking: marker } : base;
    }
    let lo = 0;
    let hi = thinking.length;
    while (lo < hi) {
        const mid = Math.ceil((lo + hi) / 2);
        if (size(withNote(mid)) <= maxBytes) lo = mid; else hi = mid - 1;
    }
    return { ...base, thinking: withNote(lo) };
}
