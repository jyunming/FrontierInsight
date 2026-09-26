// Loads the compiled VSCode extension against a stand-in `vscode` module whose language
// models are scripted, activates it, and sends `@fi /probe` to its chat participant. Nothing
// here needs VSCode, a network or Python: it checks what the shipped bundle asks each model,
// what it prints, and what it does when a model errors, the user declines, or the request is
// cancelled.
//
// argv: <path to out/extension.js> <json>
//   command          the chat command to send (default "probe")
//   prompt           its argument: "", "all", ...
//   picked           model spec for `request.model` (the Chat-picker model), or null
//   models           model specs `vscode.lm.selectChatModels()` returns
//   listThrows       make selectChatModels throw
//   modal            what the modal confirmation answers: "Run", or null (the user cancels)
//   cancelAfterSends flip the chat token to "cancelled" as the Nth chat request finishes
//
// A model spec:
//   { id, vendor, family, version, maxInputTokens,
//     replies: [ "text" | { error: "message", code?: "Blocked" } ... ]   one entry per request
//     count:   "chars" (default: the text's length) | "missing" | "throw" | "nan" }
//
// Prints { reply, progress, log, sends, exit } as JSON. `log` is every send, token count,
// model listing and modal, in the order they happened.
const Module = require("module");
const [extJs, argJson] = process.argv.slice(2);
const arg = JSON.parse(argJson);

const noop = () => {};
const channel = () => ({ appendLine: noop, append: noop, show: noop, dispose: noop });
const deep = () =>
  new Proxy(function () {}, {
    get: (t, p) => (p in t ? t[p] : deep()),
    apply: () => undefined,
    construct: () => ({}),
  });

const log = [];
let sends = 0;
const token = {
  isCancellationRequested: false,
  onCancellationRequested: () => ({ dispose: noop }),
};

function makeModel(spec) {
  let mine = 0;
  const model = {
    id: spec.id,
    vendor: spec.vendor,
    family: spec.family,
    version: spec.version,
    maxInputTokens: spec.maxInputTokens,
    sendRequest: async (messages, options, tok) => {
      const i = mine++;
      const sendNo = ++sends;
      log.push({
        kind: "send",
        model: spec.id,
        n: i + 1,
        roles: messages.map((m) => m.role),
        texts: messages.map((m) => m.content),
        options,
        usedChatToken: tok === token,
      });
      const step = (spec.replies || [])[i];
      if (step === undefined) throw new Error(`fixture: no scripted reply #${i + 1} for ${spec.id}`);
      if (typeof step === "object") {
        const e = new Error(step.error);
        if (step.code) e.code = step.code;
        throw e;
      }
      // Two chunks, so a reply is reassembled rather than taken whole.
      const cut = Math.ceil(step.length / 2);
      return {
        text: (async function* () {
          yield step.slice(0, cut);
          yield step.slice(cut);
          if (arg.cancelAfterSends === sendNo) token.isCancellationRequested = true;
        })(),
      };
    },
  };
  if (spec.count !== "missing") {
    model.countTokens = async (text) => {
      log.push({ kind: "count", model: spec.id, type: typeof text, text });
      if (spec.count === "throw") throw new Error("tokenizer offline");
      if (spec.count === "nan") return "many";
      return typeof text === "string" ? text.length : -1;
    };
  }
  return model;
}

const participants = [];
const vscodeMock = new Proxy(
  {
    __esModule: true,
    version: "1.99.0-fixture",
    chat: {
      createChatParticipant: (id, fn) => {
        participants.push({ id, fn });
        return { dispose: noop };
      },
    },
    window: {
      createOutputChannel: channel,
      showInformationMessage: async () => undefined,
      showErrorMessage: async () => undefined,
      showWarningMessage: async (message, options, ...items) => {
        log.push({ kind: "modal", message, options, items, sendsSoFar: sends });
        return arg.modal === undefined ? undefined : arg.modal;
      },
    },
    workspace: {
      // The Axon watcher would poll for minutes; 0 turns it off.
      getConfiguration: () => ({ get: (k, d) => (k === "axonStartupWaitSec" ? 0 : d) }),
      workspaceFolders: undefined,
      onDidChangeConfiguration: () => ({ dispose: noop }),
    },
    ThemeIcon: class {
      constructor(id) {
        this.id = id;
      }
    },
    LanguageModelChatMessage: {
      User: (content) => ({ role: "user", content }),
      Assistant: (content) => ({ role: "assistant", content }),
    },
    LanguageModelTextPart: class {
      constructor(value) {
        this.kind = "text";
        this.value = value;
      }
    },
    // Images for `@fi /probe image`; `noDataPart` plays a VS Code too old to send them.
    LanguageModelDataPart: arg.noDataPart ? undefined : {
      image: (data, mime) => ({ kind: "image", mime, bytes: data.length }),
    },
    lm: {
      selectChatModels: async (selector) => {
        log.push({ kind: "select", selector: selector === undefined ? null : selector });
        if (arg.listThrows) throw new Error("the model service is not ready");
        return (arg.models || []).map(makeModel);
      },
    },
    commands: { registerCommand: () => ({ dispose: noop }), executeCommand: async () => undefined },
    env: { openExternal: async () => true },
    Uri: { file: (p) => ({ fsPath: p }) },
  },
  { get: (t, p) => (p in t ? t[p] : deep()) },
);

const realLoad = Module._load;
Module._load = function (request, ...rest) {
  return request === "vscode" ? vscodeMock : realLoad.call(this, request, ...rest);
};

(async () => {
  const ext = require(extJs);
  ext.activate({ subscriptions: [] });
  const said = [];
  const progress = [];
  const stream = new Proxy(
    { markdown: (m) => said.push(String(m)), progress: (m) => progress.push(String(m)) },
    { get: (t, p) => (p in t ? t[p] : noop) },
  );
  const request = {
    command: arg.command || "probe",
    prompt: arg.prompt || "",
    model: arg.picked ? makeModel(arg.picked) : undefined,
  };
  await participants[0].fn(request, {}, stream, token);
  ext.deactivate();
  process.stdout.write(JSON.stringify({ reply: said.join(""), progress, log, sends }));
  process.exit(0);
})().catch((e) => {
  process.stderr.write(String(e && e.stack ? e.stack : e));
  process.exit(1);
});
