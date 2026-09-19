// Loads the compiled VSCode extension against a stand-in `vscode` module, activates
// it, and sends chat commands to its participant. Nothing here needs VSCode, a
// network or Python: it checks that the shipped bundle loads, registers, and routes.
//
// argv: <path to out/extension.js> <json: {commands, workspace, settings}>
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

const participants = [];
const vscodeMock = new Proxy(
  {
    __esModule: true,
    chat: {
      createChatParticipant: (id, fn) => {
        const p = { dispose: noop };
        participants.push({ id, fn });
        return p;
      },
    },
    window: {
      createOutputChannel: channel,
      showInformationMessage: async () => undefined,
      showWarningMessage: async () => undefined,
      showErrorMessage: async () => undefined,
    },
    workspace: {
      getConfiguration: () => ({ get: (k, d) => (k in arg.settings ? arg.settings[k] : d) }),
      workspaceFolders: arg.workspace ? [{ uri: { fsPath: arg.workspace } }] : undefined,
      onDidChangeConfiguration: () => ({ dispose: noop }),
    },
    ThemeIcon: class {
      constructor(id) {
        this.id = id;
      }
    },
    lm: { selectChatModels: async () => [] },
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
  const context = { subscriptions: [] };
  ext.activate(context);
  const out = { participant: participants[0] && participants[0].id, subscriptions: context.subscriptions.length, replies: {} };
  for (const cmd of arg.commands) {
    const said = [];
    const stream = new Proxy({ markdown: (m) => said.push(String(m)) }, { get: (t, p) => (p in t ? t[p] : noop) });
    await participants[0].fn(
      { command: cmd, prompt: "", model: { family: "x", vendor: "y", id: "z" } },
      {},
      stream,
      { isCancellationRequested: false, onCancellationRequested: noop },
    );
    out.replies[cmd] = said.join(" ");
  }
  ext.deactivate();
  process.stdout.write(JSON.stringify(out));
  process.exit(0);
})().catch((e) => {
  process.stderr.write(String(e && e.stack ? e.stack : e));
  process.exit(1);
});
