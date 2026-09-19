// Drives the compiled "Per-node model overrides" editor (out/interview.js) against a stand-in
// `vscode` module whose menus answer from a script, and prints the resulting answer as JSON.
// Nothing here needs VSCode: it checks the picker's logic, not VSCode's rendering of it.
//
// argv: <path to out/interview.js> <json: {node_models, script, models}>
//   script: the labels to choose, in order; "ESC" presses Esc; a { input: "..." } entry answers
//           an input box. Every QuickPick this editor shows is recorded in `menus`.
const Module = require("module");
const [interviewJs, argJson] = process.argv.slice(2);
const arg = JSON.parse(argJson);

const menus = [];
const script = [...arg.script];
const next = () => (script.length ? script.shift() : "ESC");

const vscodeMock = {
  __esModule: true,
  window: {
    showQuickPick: async (items, opts) => {
      const list = Array.isArray(items) ? items : await items;
      const want = next();
      menus.push({ title: opts && opts.title, labels: list.map((i) => i.label), asked: want });
      if (want === "ESC") return undefined;
      const hit = list.find((i) => i.label === want || i.label.startsWith(want));
      if (!hit) throw new Error(`no menu entry "${want}" among ${JSON.stringify(list.map((i) => i.label))}`);
      return hit;
    },
    showInputBox: async () => {
      const want = next();
      return want && typeof want === "object" ? want.input : undefined;
    },
    showWarningMessage: async (m) => {
      menus.push({ warning: m });
    },
  },
  lm: {
    selectChatModels: async () =>
      arg.models.map((m) => ({ id: m.id, name: m.name, vendor: m.vendor, family: m.family })),
  },
};

const realLoad = Module._load;
Module._load = function (request, ...rest) {
  return request === "vscode" ? vscodeMock : realLoad.call(this, request, ...rest);
};

(async () => {
  const iv = require(interviewJs);
  const answers = { node_models: arg.node_models };
  await iv.editNodeModels(answers);
  process.stdout.write(JSON.stringify({ node_models: answers.node_models, menus, unused: script.length }));
  process.exit(0);
})().catch((e) => {
  process.stderr.write(String(e && e.stack ? e.stack : e));
  process.exit(1);
});
