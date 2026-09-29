// The quest map's script and stylesheet live once, in web/static/ (the web quest page uses them there). The VS Code
// panel loads the same two files from media/, copied here at compile time so the two views cannot drift.
const fs = require("fs");
const path = require("path");
const from = path.join(__dirname, "..", "..", "web", "static");
const to = path.join(__dirname, "..", "media");
fs.mkdirSync(to, { recursive: true });
for (const f of ["quest_map.js", "quest_map.css"]) {
    const src = path.join(from, f);
    if (fs.existsSync(src)) {
        fs.copyFileSync(src, path.join(to, f));
    } else if (!fs.existsSync(path.join(to, f))) {
        console.error(`missing ${src}`);
        process.exit(1);
    }
}
