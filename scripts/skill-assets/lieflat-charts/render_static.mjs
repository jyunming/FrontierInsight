#!/usr/bin/env node
// FI addition (not part of the upstream lieflat-charts skill): rasterize a
// finished chart/report HTML file to a static PNG so it can go into a
// paper_pdf / poster / slides deliverable, which cannot embed live
// JS-rendered content (Chart.js / ECharts canvases, animated SVG).
//
// Pure-SVG templates (Lupi Editorial, Basics) need none of this -- they are
// already static vector graphics and can be embedded/converted directly
// (e.g. via rsvg-convert or a PDF toolchain's native SVG support). This
// script exists for the templates that render via Chart.js/ECharts canvas
// or otherwise need a real layout+paint pass to produce pixels: Glance,
// the interactive "big-*" templates, and any report template with an
// embedded canvas chart.
//
// Requires Playwright's Chromium, installed once, globally, on demand --
// deliberately NOT a project dependency (this skill's own dev scripts use
// the same global-install pattern; see scripts/smoke-new-charts.mjs):
//   npm install -g playwright && npx playwright install chromium
//
// Usage:
//   node render_static.mjs <input.html> <output.png> [selector] [width] [height]
//
// <selector> (optional) scopes the screenshot to one element (e.g. a single
// chart card's `.card` container) instead of the full rendered page --
// useful when a gallery/report file holds more than one chart and only one
// is the actual deliverable.

import { execFileSync } from "node:child_process";
import path from "node:path";
import { pathToFileURL } from "node:url";

async function loadChromium() {
  let globalRoot;
  try {
    // `execFileSync("npm", ...)` fails with ENOENT on Windows -- npm is a
    // `.cmd` shim there, not a real executable, and execFileSync won't
    // resolve or run one without a shell (passing "npm.cmd" instead still
    // fails, with EINVAL: a .cmd needs cmd.exe to interpret it, it isn't
    // directly spawnable either). `shell: true` with a single command
    // string (not a file+args array, which triggers a Node deprecation
    // warning about unescaped args) is the portable fix -- safe here since
    // the command is a fixed literal, nothing from user input reaches it.
    globalRoot = execFileSync("npm root -g", { encoding: "utf8", shell: true }).trim();
  } catch (e) {
    throw new Error(
      "could not resolve the global npm root -- is npm on PATH? " +
      `(${e.message})`,
    );
  }
  try {
    const mod = await import(pathToFileURL(path.join(globalRoot, "playwright", "index.js")));
    return mod.chromium ? mod.chromium : mod.default.chromium;
  } catch {
    throw new Error(
      "Playwright is not installed globally. Run once:\n" +
      "  npm install -g playwright && npx playwright install chromium",
    );
  }
}

async function main() {
  const [, , inputPath, outputPath, selector, widthArg, heightArg] = process.argv;
  if (!inputPath || !outputPath) {
    console.error("usage: node render_static.mjs <input.html> <output.png> [selector] [width] [height]");
    process.exit(2);
  }
  const width = Number(widthArg) || 1280;
  const height = Number(heightArg) || 900;

  const chromium = await loadChromium();
  const browser = await chromium.launch();
  try {
    const page = await browser.newPage({ viewport: { width, height } });
    const consoleErrors = [];
    page.on("console", (m) => { if (m.type() === "error") consoleErrors.push(m.text()); });
    page.on("pageerror", (e) => consoleErrors.push(String(e)));

    const url = pathToFileURL(path.resolve(inputPath)).href;
    await page.goto(url, { waitUntil: "networkidle" });
    // Chart.js/ECharts finish their paint pass a beat after networkidle
    // (post-load layout + canvas draw calls, not network activity) --
    // this mirrors the settle wait the skill's own smoke test uses.
    await page.waitForTimeout(500);

    const target = selector ? page.locator(selector).first() : page;
    await target.screenshot({ path: path.resolve(outputPath) });

    if (consoleErrors.length) {
      console.error(`rendered with ${consoleErrors.length} console error(s):`);
      for (const e of consoleErrors) console.error(`  ${e}`);
    }
    console.log(`wrote ${outputPath}`);
  } finally {
    await browser.close();
  }
}

main().catch((e) => {
  console.error(String(e.message || e));
  process.exit(1);
});
