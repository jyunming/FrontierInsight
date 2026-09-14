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
