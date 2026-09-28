#!/usr/bin/env node
// Compares the per-file sha256 of a freshly packaged .vsix manifest with the committed one. The .vsix bytes may differ
// between machines (zlib builds differ); the files packed inside must not.
"use strict";

const fs = require("fs");

const [fresh, committed] = process.argv.slice(2);
if (!fresh || !committed) {
  console.error("usage: node scripts/check-vsix-manifest.js <fresh.manifest.json> <committed.manifest.json>");
  process.exit(2);
}
const a = JSON.parse(fs.readFileSync(fresh, "utf8")).files || {};
const b = JSON.parse(fs.readFileSync(committed, "utf8")).files || {};
const names = [...new Set([...Object.keys(a), ...Object.keys(b)])].sort();
const differ = names.filter((n) => (a[n] || {}).sha256 !== (b[n] || {}).sha256);
if (differ.length) {
  console.error(
    "The committed vscode-frontier-insight.vsix does not hold the files these sources build:\n  " +
      differ.join("\n  ") +
      "\nRebuild it (`npm run package` in vscode-frontier-insight/) and commit the .vsix with its manifest; " +
      "on main the rebuild workflow does this after merge.",
  );
  process.exit(1);
}
console.log(`the packed files match the committed manifest (${names.length} files)`);
