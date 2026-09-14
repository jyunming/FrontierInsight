"""The VSCode bridge turns FI's chat messages into language-model messages.

A message with screenshots arrives as OpenAI content parts. The compiled
``out/lm-messages.js`` must hand the images to ``LanguageModelDataPart.image``
and, on a VS Code build without that API, fail with a clear error instead of
sending the text alone.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

EXT_DIR = Path(__file__).resolve().parent.parent / "vscode-frontier-insight"
COMPILED = EXT_DIR / "out" / "lm-messages.js"

NODE_SCRIPT = """
const { toChatMessages, IMAGES_UNSUPPORTED } = require(%s);
class TextPart { constructor(value) { this.kind = "text"; this.value = value; } }
const api = {
  LanguageModelChatMessage: {
    User: (content) => ({ role: "user", content }),
    Assistant: (content) => ({ role: "assistant", content }),
  },
  LanguageModelTextPart: TextPart,
  LanguageModelDataPart: { image: (data, mime) => ({ kind: "image", mime, bytes: Array.from(data) }) },
};
const png = Buffer.from([137, 80, 78, 71]).toString("base64");
const withImage = {
  role: "user",
  content: [
    { type: "text", text: "Check this page." },
    { type: "image_url", image_url: { url: "data:image/png;base64," + png } },
  ],
};
const out = {
  messages: toChatMessages([{ role: "user", content: "plain" }, withImage, { role: "assistant", content: "ok" }], api),
};
try {
  toChatMessages([withImage], { ...api, LanguageModelDataPart: undefined });
  out.old = "no error";
} catch (e) {
  out.old = e.message === IMAGES_UNSUPPORTED ? "unsupported" : e.message;
}
process.stdout.write(JSON.stringify(out));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH")
def test_images_become_data_parts_and_an_old_vscode_says_it_cannot_send_them() -> None:
    if not COMPILED.exists():
        npm = shutil.which("npm")
        if npm is None or not (EXT_DIR / "node_modules").is_dir():
            pytest.skip("run `npm install && npm run compile` in vscode-frontier-insight/ first")
        subprocess.run([npm, "run", "compile"], cwd=str(EXT_DIR), capture_output=True, text=True, timeout=180, check=True)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False, encoding="utf-8") as handle:
        handle.write(NODE_SCRIPT % json.dumps(str(COMPILED).replace("\\", "/")))
        driver = handle.name
    try:
        run = subprocess.run([shutil.which("node"), driver], capture_output=True, text=True, timeout=30)
    finally:
        Path(driver).unlink(missing_ok=True)
    assert run.returncode == 0, run.stderr
    out = json.loads(run.stdout)
    assert out["messages"][0] == {"role": "user", "content": "plain"}
    assert out["messages"][1] == {
        "role": "user",
        "content": [{"kind": "text", "value": "Check this page."}, {"kind": "image", "mime": "image/png", "bytes": [137, 80, 78, 71]}],
    }
    assert out["messages"][2] == {"role": "assistant", "content": "ok"}
    assert out["old"] == "unsupported"
