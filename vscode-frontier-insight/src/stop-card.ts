/**
 * The buttons under a "why it stopped" card in the chat.
 *
 * When the known-answer checks stop a quest, FI writes one structured card (core/oracle_card.py) into
 * `.fi/pause.json`, and the same card as text into `NEXT_STEP.md`, which the chat already shows. This module turns the
 * card's `actions` into chat buttons, so each way on is one click, as it is on the web page: "Let FI try again" sends
 * `@fi /resume <id>`; a change to a check puts `@fi /plan <id> <the card's words>` in the chat box to read before
 * sending; going on with a setting changed puts `@fi /update <id>` there. An action this module does not know gets no
 * button (its command is still in the card's text), so the card can grow without the extension breaking.
 *
 * Pure (no `vscode` import), so it can be run under plain node in the tests.
 */

export interface CardAction {
    id?: string;
    label?: string;
    detail?: string;
    cli?: string;
    web?: string;
    vscode?: string;
    prefill?: string;
    file?: string;
    line?: number;
}

export interface StopCard {
    type?: string;
    summary?: string;
    actions?: CardAction[];
}

/** One chat button: the text on it, and the chat query it puts in the chat box (`send`: sent at once). */
export interface CardButton {
    title: string;
    query: string;
    send: boolean;
    tooltip: string;
}

/** The card in a parsed `.fi/pause.json`, or null when it has none. */
export function cardOf(pause: unknown): StopCard | null {
    if (!pause || typeof pause !== "object") { return null; }
    const card = (pause as { card?: unknown }).card;
    if (!card || typeof card !== "object") { return null; }
    const actions = (card as StopCard).actions;
    return Array.isArray(actions) ? (card as StopCard) : null;
}

/** The chat buttons for a card's actions, in the card's order. */
export function cardButtons(card: StopCard | null, questId: string): CardButton[] {
    const out: CardButton[] = [];
    if (!card || !Array.isArray(card.actions) || !/^[A-Za-z0-9._-]+$/.test(questId)) { return out; }
    for (const a of card.actions) {
        if (!a || typeof a !== "object") { continue; }
        const tooltip = String(a.detail || "");
        const title = String(a.label || a.id || "");
        if (a.id === "resume") {
            out.push({ title, query: `@fi /resume ${questId}`, send: true, tooltip });
        } else if ((a.id === "revise_check" || a.id === "accept_proposal") && a.prefill) {
            // Put in the box, not sent: the person reads the request (and may change it) first.
            out.push({ title, query: `@fi /plan ${questId} ${oneLine(a.prefill)}`, send: false, tooltip });
        } else if (a.id === "go_on_recorded" && typeof a.vscode === "string" && a.vscode.startsWith("@fi /update ")) {
            out.push({ title, query: `@fi /update ${questId}`, send: false, tooltip });
        }
    }
    return out;
}

function oneLine(text: string): string {
    return String(text).replace(/\s+/g, " ").trim();
}
