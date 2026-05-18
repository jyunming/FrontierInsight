# Installing Frontier Insight

Three install paths depending on your environment:

1. **Standard** — you have admin (or any) install rights. Use this.
2. **No admin** — corporate laptop without admin. All components install
   per-user with no admin rights.
3. **Fully locked-down** — IT blocks per-user installs of LaTeX. Use the
   built-in tectonic fallback.

All three end at the same place: `pip install` complete, optional system
tools available, `python launch.py --config my.yaml` produces a paper.

## System requirements

| Component | Requirement | Notes |
|---|---|---|
| Python | 3.11+ | Type hints + LangGraph need it. |
| pip | recent | `python -m pip --version` |
| Git | optional | Only needed if you clone the repo. `pip install frontier-insight` will skip this. |

Operating systems supported: Windows 10+, macOS 12+ (Intel and Apple
Silicon), Linux (modern distros with glibc). All paths use
`pathlib.Path` and switch on `sys.platform` where needed.

## Path 1 — Standard install

```bash
pip install frontier-insight
```

That's it. The `fi` command is now on your PATH. Try:

```bash
fi --help
fi --config examples/integrator_bakeoff/config.yaml
```

For richer output formats install the optional system tools listed
under [System tools](#system-tools) below.

## Path 2 — No-admin install (Windows User-PATH)

Everything installs into `%LOCALAPPDATA%` and your user PATH. No admin
prompts.

```powershell
# Python (if not already): use the "user installer" from python.org.
# After install, "py -3.11" or "python" works.

# Frontier Insight itself:
pip install --user frontier-insight

# Optional system tools (per-user installs):
winget install JohnMacFarlane.Pandoc    # for paper.pdf
winget install MiKTeX.MiKTeX             # for LaTeX engine (used by pandoc)
npm install -g @marp-team/marp-cli       # for slides.html / slides.pdf

# One-time MiKTeX config: silence missing-package prompts so quests
# don't pop GUI dialogs. Substitute the actual path if different.
& "$env:LOCALAPPDATA\Programs\MiKTeX\miktex\bin\x64\initexmf.exe" `
  --set-config-value=[MPM]AutoInstall=1
```

After install, every quest's PDF compile step will auto-download
missing CTAN packages silently (~30 s on the first quest, instant
thereafter).

## Path 3 — Locked-down environment (tectonic fallback)

When IT blocks per-user MiKTeX installs or when corporate antivirus
quarantines `pdflatex.exe` on first run, use **tectonic** instead.
Tectonic is a single Rust binary (~70 MB) that needs no install step
and self-bootstraps CTAN packages.

```bash
pip install --user frontier-insight
fi --install-tectonic       # downloads tectonic into ./tools/, verifies SHA-256
```

After `--install-tectonic` finishes, Frontier Insight automatically
detects the tectonic binary at `<repo>/tools/tectonic` (or
`tools/tectonic.exe` on Windows) and uses it as the LaTeX engine
whenever `pdflatex` isn't on PATH. The very first quest takes ~30 s
extra while tectonic downloads required CTAN packages into
`%LOCALAPPDATA%/TectonicProject/Tectonic/` (or
`~/.cache/TectonicProject/Tectonic/` on POSIX). Subsequent quests
reuse the cache and complete in seconds.

### Prerequisites for `--install-tectonic`

- HTTPS access to `github.com` (release archive download)
- HTTPS access to `ctan.org` (first-run package fetch)
- ~150 MB free disk in the user profile (binary + first-fetch cache)

If your corporate proxy needs explicit configuration:

```powershell
$env:https_proxy = "http://proxy.company.com:8080"
fi --install-tectonic
```

`urllib` honors these standard env vars.

### Airgapped / strict-proxy / no GitHub access

When the target host can't reach `github.com` at all (fully airgapped
lab, deny-by-default proxy, security policy blocks the
`tectonic-typesetting/tectonic` release page), download the binary
elsewhere and hand-carry it over with `--install-tectonic-from`:

```bash
# On a host with internet, fetch the right asset for your target's
# OS+arch from https://github.com/tectonic-typesetting/tectonic/releases
# (e.g. tectonic-0.16.9-x86_64-unknown-linux-musl.tar.gz for Linux).
# Copy the archive to the target host (USB, scp, S3, internal mirror).

# On the airgapped target, pointed at the archive OR the extracted
# binary OR the directory the archive lives in:
fi --install-tectonic-from ~/downloads/tectonic-0.16.9-x86_64-unknown-linux-musl.tar.gz
fi --install-tectonic-from ~/downloads/tectonic         # already-extracted
fi --install-tectonic-from ~/downloads/                 # scans the dir
```

The same atomic-replace flow drops the binary into `tools/`, with no
network call. Sanity-checks the executable header (ELF / Mach-O / PE)
so a wrong-arch tarball can't silently land.

> First-compile CTAN fetch still happens on the airgapped target.
> Either pre-warm it on a connected host (run one PDF compile so
> `~/.cache/TectonicProject/Tectonic/` populates, then rsync that dir
> across) or pre-stage a local CTAN mirror and set
> `TECTONIC_BUNDLE` to point at it.

## System tools (optional)

Each tool is OPTIONAL — Frontier Insight degrades gracefully without
it (you'll get `paper.md` instead of `paper.pdf`, `slides.md` instead
of `slides.html`, etc.):

| Tool | Used for | Install |
|---|---|---|
| pandoc | `paper.pdf`, `slides.pptx` | `winget install JohnMacFarlane.Pandoc` / `brew install pandoc` / `apt install pandoc` |
| pdflatex (MiKTeX or TeX Live) | `paper.pdf`, `poster.pdf` | `winget install MiKTeX.MiKTeX` / `brew install --cask mactex` |
| tectonic | LaTeX engine fallback (no-admin) | `fi --install-tectonic` |
| Marp CLI | `slides.html`, `slides.pdf` | `npm install -g @marp-team/marp-cli` |
| Node.js | The VSCode extension build | `winget install OpenJS.NodeJS` / nvm |
| Docker Desktop | `execution.sandbox: docker` | docker.com/products/docker-desktop |
| Axon | Knowledge layer (literature search + cross-quest memory) | `pip install axon-rag` |

## VSCode extension

The companion VSCode chat extension is published separately. Once
installed in VSCode, typing `@fi` in Copilot Chat gives you the
guided interview, status streaming, and `/resume` / `/summarize` /
`/fleet` chat commands.

```bash
# From the cloned repo, build the .vsix:
cd vscode-frontier-insight
npm install
npm run package      # produces vscode-frontier-insight.vsix

# Install:
code --install-extension vscode-frontier-insight/vscode-frontier-insight.vsix
```

Reload your VSCode window and `@fi` will appear in the Copilot Chat
participant picker.

## Verifying the install

```bash
fi --help                                                  # console script
fi --config examples/integrator_bakeoff/config.yaml        # ~3 min end-to-end
# Output lands at outputs/<quest_id>/paper/paper.md
```

If `fi` isn't on PATH after `pip install --user`, your user-scripts
directory isn't on PATH. On Windows that's typically
`%APPDATA%\Python\Python311\Scripts`; on POSIX `~/.local/bin`. Add
it to PATH and re-open your shell.

## Troubleshooting

**"`pandoc` not on PATH"** — install pandoc (see [System tools](#system-tools)). Frontier Insight skips PDF generation gracefully when pandoc is missing; the `.md` is still produced.

**"`pdflatex not found`"** during pandoc PDF compile — install MiKTeX or run `fi --install-tectonic`.

**MiKTeX pops a GUI prompt every quest** — set `AutoInstall=1` in MiKTeX config (see [Path 2](#path-2--no-admin-install-windows-user-path) above).

**Corporate antivirus quarantines `tectonic.exe`** — ask IT to whitelist the file. The binary is signed by the maintainers; this is usually a one-time approval.

**`fi --install-tectonic` fails with a SHA-256 mismatch** — the release was retagged, or you hit a network MITM. Confirm by retrying. If it persists, file an issue.

**The VSCode extension doesn't show in Copilot Chat** — VSCode caches participants. Run `Developer: Reload Window` from the Command Palette.
