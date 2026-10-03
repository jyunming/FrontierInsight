"""Fetch the web UI's fonts (latin subsets) and the Material Symbols static font from Google Fonts into
web/static/vendor/fonts/, and write web/static/vendor/fonts.css pointing at them."""
import hashlib
import re
import sys
import urllib.request
from pathlib import Path

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
dest = Path(sys.argv[1])
fonts = dest / "fonts"
fonts.mkdir(parents=True, exist_ok=True)

FAMILIES = [
    ("Fraunces", "Fraunces:opsz,wght@9..144,400..900"),
    ("Geist", "Geist:wght@400;500;600;700"),
    ("JetBrains Mono", "JetBrains+Mono:wght@400;500;600"),
    ("IBM Plex Sans", "IBM+Plex+Sans:wght@400;500;600"),
    ("IBM Plex Mono", "IBM+Plex+Mono:wght@400;500"),
    ("Material Symbols Outlined", "Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0"),
]


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


out_css = []
total = 0
for name, spec in FAMILIES:
    css = get(f"https://fonts.googleapis.com/css2?family={spec}&display=block").decode()
    blocks = re.findall(r"(/\* ([\w-]+) \*/\s*)?(@font-face \{.*?\})", css, flags=re.S)
    for _c, subset, block in blocks:
        subset = subset or "all"
        if name != "Material Symbols Outlined" and subset not in ("latin", "latin-ext"):
            continue
        url = re.search(r"url\((https://[^)]+)\)", block).group(1)
        data = get(url)
        # A variable font is one file for every weight: named by its content, it is stored once.
        fname = f"{name.replace(' ', '')}-{subset}-{hashlib.sha256(data).hexdigest()[:8]}.woff2"
        if not (fonts / fname).exists():
            (fonts / fname).write_bytes(data)
            total += len(data)
        local = block.replace(url, f"fonts/{fname}")
        if name == "Material Symbols Outlined":
            # Blank, not the icon's name as a word, until the font is there (it is local, so that is at once).
            local = re.sub(r"font-display: \w+;", "font-display: block;", local)
            if "font-display" not in local:
                local = local.replace("src:", "font-display: block;\n  src:", 1)
        else:
            local = re.sub(r"font-display: \w+;", "font-display: swap;", local)
        out_css.append(f"/* {name} — {subset} */\n{local}\n")
        print(fname, len(data))

header = """/* The fonts the web UI uses, served by FI itself (no request leaves the machine): the latin and latin-ext subsets
   of Fraunces, Geist, JetBrains Mono, IBM Plex Sans / Mono (SIL Open Font License 1.1) and the Material Symbols
   Outlined icon font, default instance (Apache License 2.0). Other scripts fall back to the system fonts named in
   style.css. Regenerate with scripts/vendor_web_fonts.py web/static/vendor. */

"""
ICON_CLASS = """
/* The icon class (as Google's stylesheet defines it). */
.material-symbols-outlined {
  font-family: 'Material Symbols Outlined';
  font-weight: normal;
  font-style: normal;
  font-size: 24px;
  line-height: 1;
  letter-spacing: normal;
  text-transform: none;
  display: inline-block;
  white-space: nowrap;
  word-wrap: normal;
  direction: ltr;
  -webkit-font-feature-settings: 'liga';
  font-feature-settings: 'liga';
  -webkit-font-smoothing: antialiased;
}
"""
(dest / "fonts.css").write_text(header + "\n".join(out_css) + ICON_CLASS, encoding="utf-8")
print("total", total)
