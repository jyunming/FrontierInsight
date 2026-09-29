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
    if (obj && typeof obj.value === "string") {
        if (obj.constructor?.name === "LanguageModelThinkingPart") return "thinking";
        if (obj.constructor?.name === "LanguageModelTextPart") return "text";
        // A part that names itself as neither is not taken for the answer unless the caller says to.
        return unknownStringAs;
    }
    return "unknown";
}

// One message is one line, and an FI that has not raised its reader's 64 KiB line limit drops the connection on a
// longer one, so the whole `lm_done` line stays under this.
export const LM_DONE_MAX_BYTES = 48 * 1024;

/** The reasoning of one answer as it streams in: only what could ever be sent is kept, the rest is only counted. */
export class ThinkingCollector {
    text = "";
    total = 0;
    add(fragment: string, keepChars: number = LM_DONE_MAX_BYTES): void {
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
        // The answer leaves no room for any of it: say that the reasoning existed rather than send nothing. The marker
        // is always a little shorter than the note, so it fits where the note just did not.
        const marker = `[${total} chars not sent: no room]`;
        return size(marker) <= maxBytes ? { ...base, thinking: marker } : base;
    }
    let lo = 0;
    let hi = thinking.length;
    while (lo < hi) {
        const mid = Math.ceil((lo + hi) / 2);
        if (size(withNote(mid)) <= maxBytes) lo = mid; else hi = mid - 1;
    }
    return { ...base, thinking: withNote(lo) };
}
