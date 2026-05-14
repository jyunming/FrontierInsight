"""Paper output generator.

For each quest, the engine has already written `paper/paper.md` plus any
`figures/` produced by the experiment. The generator (1) hoists the
markdown to the top of the quest output, (2) optionally compiles it to
PDF via pandoc + LaTeX using a format-specific template, and (3) copies
figures and bundle manifest if they exist.

If pandoc or a LaTeX engine is unavailable, PDF generation is skipped
with a warning — the markdown is still produced. The generator also
writes a ``paper/paper_pdf_skipped.md`` file when a PDF was requested
(via ``output.kinds``) but couldn't be produced, so the user discovers
the failure without grepping run logs. Closes the silent-skip gap
that hit ``outputs/1778751621-belgium-culture-vs-taiwan-cultur-32f2ff/``
in 2026-05-14 when the host had no pandoc/pdflatex/tectonic installed.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from core.config import Config
from core.engine import QuestArtifacts

_log = logging.getLogger("frontier_insight.paper")

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = REPO_ROOT / "templates" / "paper"


@dataclass
class _PdfSkipReason:
    """Why ``_compile_pdf`` returned ``None``. Surfaced to the user via
    ``paper_pdf_skipped.md`` so they don't have to grep run.log."""

    code: str          # short stable identifier, e.g. "no_pandoc"
    summary: str       # one-line WARNING text already logged
    how_to_fix: str    # 1-3 sentences pointing at the concrete action


class PaperGenerator:
    def __init__(self, config: Config) -> None:
        self.config = config

    def generate(self, art: QuestArtifacts, out_dir: Path) -> dict[str, Path]:
        kinds = set(self.config.output.kinds)
        result: dict[str, Path] = {}
        out_dir.mkdir(parents=True, exist_ok=True)

        if "paper_md" in kinds and art.paper_md is not None:
            dst = out_dir / "paper.md"
            shutil.copy2(art.paper_md, dst)
            result["paper_md"] = dst

        if art.figures_dir is not None and art.figures_dir.exists():
            dst = out_dir / "figures"
            # resolve() so symlinks / alternate spellings of the same directory
            # don't trigger rmtree-then-copytree on the source.
            same = dst.resolve() == art.figures_dir.resolve()
            if not same:
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(art.figures_dir, dst)
            result["figures_dir"] = dst

        if art.bundle_manifest is not None and art.bundle_manifest.exists():
            dst = out_dir / "paper_bundle_manifest.json"
            if dst.resolve() != art.bundle_manifest.resolve():
                shutil.copy2(art.bundle_manifest, dst)
            result["bundle_manifest"] = dst

        if "paper_pdf" in kinds and art.paper_md is not None:
            # If paper_md wasn't requested, the markdown wasn't copied above.
            # Compile from art.paper_md directly so the PDF doesn't depend on
            # paper_md being in output.kinds.
            pdf_src = result.get("paper_md") or art.paper_md
            pdf, skip_reason = self._compile_pdf(pdf_src, out_dir)
            diag_path = out_dir / "paper_pdf_skipped.md"
            if pdf is not None:
                result["paper_pdf"] = pdf
                # Success on this run — remove any stale skip
                # diagnostic from a PREVIOUS failed run. Otherwise the
                # quest dir would report both paper.pdf AND
                # paper_pdf_skipped.md, which is confusing.
                if diag_path.is_file():
                    try:
                        diag_path.unlink()
                    except OSError:
                        pass
            elif skip_reason is not None:
                # User asked for a PDF, none was produced, AND we know why.
                # Write a single-paragraph diagnostic file in the same
                # directory so the failure is discoverable without grepping
                # run.log. Stale files from a prior failed run get
                # overwritten — the LATEST attempt's reason is what
                # matters to the user.
                diag_path.write_text(
                    _render_pdf_skip_md(skip_reason, self.config),
                    encoding="utf-8",
                )
                result["paper_pdf_skipped"] = diag_path
        elif "paper_pdf" not in kinds:
            # ``paper_pdf`` was not requested — clean up any stale
            # diagnostic from a previous run that did request it. Keeps
            # the quest dir tidy after a config edit.
            stale = out_dir / "paper_pdf_skipped.md"
            if stale.is_file():
                try:
                    stale.unlink()
                except OSError:
                    pass

        return result

    def _find_pdf_engine(self) -> tuple[str, str] | None:
        """Locate a LaTeX engine for pandoc's ``--pdf-engine`` flag.

        Search order (stop at first match):
          1. ``shutil.which("pdflatex")`` — the canonical MiKTeX / TeX Live
             install. Preferred when present because a warm cache compiles
             in ~1-3 s.
          2. ``shutil.which("tectonic")`` — a single-binary self-bootstrapping
             LaTeX. Works on corporate laptops without admin (downloads
             packages on first run, no install step, no GUI prompts).
          3. ``<repo_root>/tools/tectonic.exe`` (or ``./tools/tectonic``
             on POSIX) — the opt-in install location populated by
             ``python launch.py --install-tectonic``. Most-isolated path,
             used when neither system install is present.

        Returns ``(engine_name, full_path)`` or ``None`` when no engine
        is reachable. Tectonic is intentionally the fallback — flipping
        the order would penalize the dominant case (existing MiKTeX
        users with a warm cache) to help the corporate-laptop minority.
        """
        pdflatex = shutil.which("pdflatex")
        if pdflatex:
            return ("pdflatex", pdflatex)
        tectonic = shutil.which("tectonic")
        if tectonic:
            return ("tectonic", tectonic)
        # Repo-local opt-in install. Check ONLY the platform-appropriate
        # filename — checking the wrong extension first (e.g.
        # `tectonic.exe` on POSIX) risks picking up a stray Windows
        # binary in a shared / WSL-mounted repo and crashing the pandoc
        # call with an exec-format error. `--install-tectonic` writes
        # under the same platform-appropriate name.
        repo_tectonic = REPO_ROOT / "tools" / (
            "tectonic.exe" if sys.platform == "win32" else "tectonic"
        )
        if repo_tectonic.is_file():
            return ("tectonic", str(repo_tectonic))
        return None

    def _compile_pdf(
        self, paper_md: Path, out_dir: Path,
    ) -> tuple[Path | None, _PdfSkipReason | None]:
        """Run pandoc + a LaTeX engine over ``paper_md`` to produce
        ``out_dir/paper.pdf``. Returns ``(path, None)`` on success,
        ``(None, reason)`` on any kind of skip — the reason carries
        enough info for ``_render_pdf_skip_md`` to write a useful
        ``paper_pdf_skipped.md`` for the user.
        """
        # Pandoc itself: `subprocess.run` on Windows auto-appends `.exe`
        # to bare executable names via CreateProcess, so resolving via
        # `shutil.which` is mostly defensive (would matter if pandoc
        # ever shipped as a .cmd shim like marp does). We do it for
        # symmetry + so the absolute path lands in any stderr logs.
        pandoc_exe = shutil.which("pandoc")
        if pandoc_exe is None:
            msg = "pandoc not on PATH; paper.pdf skipped (paper.md only)"
            _log.warning(msg)
            return None, _PdfSkipReason(
                code="no_pandoc",
                summary=msg,
                how_to_fix=(
                    "Install pandoc and ensure it's on your PATH. Windows: "
                    "winget install --id JohnMacFarlane.Pandoc, or download "
                    "from https://pandoc.org/installing.html. macOS: "
                    "`brew install pandoc`. Linux: your package manager."
                ),
            )

        # CRITICAL: pandoc does its OWN PATH lookup for the
        # `--pdf-engine` binary. On corporate Windows boxes the
        # MiKTeX bin dir (`~/AppData/Local/Programs/MiKTeX/miktex/bin/x64/`)
        # often isn't on the Python child's PATH even though MiKTeX
        # is installed. Resolving the engine up-front and passing the
        # full path bypasses pandoc's lookup. Falls back to tectonic
        # (no-admin LaTeX) when pdflatex isn't reachable.
        engine = self._find_pdf_engine()
        if engine is None:
            msg = (
                "no LaTeX engine found (pdflatex or tectonic); paper.pdf "
                "skipped. Run `python launch.py --install-tectonic` for "
                "a no-admin LaTeX install."
            )
            _log.warning(msg)
            return None, _PdfSkipReason(
                code="no_latex_engine",
                summary=msg,
                how_to_fix=(
                    "Easiest no-admin path: run `python launch.py "
                    "--install-tectonic` from the repo root. That drops a "
                    "single self-bootstrapping LaTeX binary (~70 MB) into "
                    "`tools/`; FI auto-detects it on the next quest. "
                    "Standard alternative: install MiKTeX (Windows) or "
                    "TeX Live (macOS/Linux) so `pdflatex` lands on PATH."
                ),
            )
        engine_name, engine_path = engine
        _log.info("paper.pdf: using pdf-engine=%s at %s", engine_name, engine_path)

        fmt = self.config.output.paper_format
        template = TEMPLATES_DIR / fmt / "template.tex"
        out_pdf = out_dir / "paper.pdf"

        cmd: list[str] = [
            pandoc_exe,
            str(paper_md),
            "-o", str(out_pdf),
            f"--pdf-engine={engine_path}",
            "--standalone",
        ]
        if template.exists():
            cmd.extend(["--template", str(template)])
        else:
            _log.info(
                "no template at %s; using pandoc default (paper_format=%s)",
                template, fmt,
            )

        # Headroom for the first-run package-fetch warm-up. Tectonic's
        # first invocation downloads CTAN packages into
        # `%LOCALAPPDATA%/TectonicProject/Tectonic/` (~30 s); MiKTeX's
        # first compile after a fresh install does the same. 300 s
        # comfortably covers both; bumped to 360 s when tectonic is
        # the picked engine in case the user is on a slow corporate
        # network.
        timeout_s = 360 if engine_name == "tectonic" else 300
        try:
            r = subprocess.run(
                cmd, capture_output=True, text=True, cwd=str(out_dir),
                timeout=timeout_s,
            )
        except FileNotFoundError:
            msg = "pandoc invocation failed; paper.pdf skipped"
            _log.warning(msg)
            return None, _PdfSkipReason(
                code="pandoc_invocation_failed",
                summary=msg,
                how_to_fix=(
                    "`shutil.which('pandoc')` found pandoc but `subprocess.run` "
                    "could not invoke it. Likely a PATH disagreement between "
                    "the shell that resolved the binary and the Python child "
                    "process that tried to spawn it. Reopen the terminal "
                    "after installing pandoc so the Python session inherits "
                    "the updated PATH. (FI does not consult a separate "
                    "`PANDOC_PATH` env var; only `shutil.which('pandoc')` is "
                    "used to locate the binary.)"
                ),
            )
        except subprocess.TimeoutExpired:
            msg = f"{engine_name} timeout (>{timeout_s}s); paper.pdf skipped"
            _log.warning(msg)
            return None, _PdfSkipReason(
                code=f"{engine_name}_timeout",
                summary=msg,
                how_to_fix=(
                    f"The {engine_name} compile took longer than "
                    f"{timeout_s}s. The first-ever compile on a fresh "
                    f"install can download CTAN packages (~30 s); a slow "
                    f"corporate VPN amplifies this. Try again — the second "
                    f"compile is usually fast. If it consistently times "
                    f"out, raise the timeout in `generation/paper.py:_compile_pdf`."
                ),
            )

        if r.returncode != 0:
            stderr_tail = r.stderr[-500:] if r.stderr else "(empty stderr)"
            msg = (
                f"{engine_name} rc={r.returncode} format={fmt}; "
                f"paper.pdf skipped. stderr_tail={stderr_tail}"
            )
            _log.warning(msg)
            return None, _PdfSkipReason(
                code=f"{engine_name}_rc_{r.returncode}",
                summary=(
                    f"{engine_name} exited with rc={r.returncode} when "
                    f"compiling paper.md with paper_format={fmt!r}."
                ),
                how_to_fix=(
                    f"The LaTeX engine errored. Common causes: a malformed "
                    f"LaTeX template, an unsupported character in paper.md, "
                    f"or a missing CTAN package on the first-ever compile. "
                    f"stderr tail:\n\n```\n{stderr_tail}\n```\n\n"
                    f"Inspect the template at "
                    f"`templates/paper/{fmt}/template.tex` and verify "
                    f"paper.md doesn't contain raw LaTeX that conflicts."
                ),
            )
        if not out_pdf.exists():
            msg = (
                f"{engine_name} rc=0 but {out_pdf.name} not on disk; "
                f"paper.pdf skipped."
            )
            _log.warning(msg)
            return None, _PdfSkipReason(
                code="output_missing_after_success",
                summary=msg,
                how_to_fix=(
                    f"{engine_name} reported success but didn't write the "
                    f"output file. This is rare — usually a permissions "
                    f"issue on the output directory. Check `{out_dir}` is "
                    f"writable and re-run."
                ),
            )
        return out_pdf, None


def _render_pdf_skip_md(reason: _PdfSkipReason, config: Config) -> str:
    """Render a ``paper_pdf_skipped.md`` diagnostic file. The user sees
    this file next to ``paper.md`` when the PDF couldn't be produced;
    it tells them what failed and how to fix it without having to grep
    ``run.log`` or read the engine source."""
    return (
        f"# paper.pdf was requested but not produced\n\n"
        f"Your `output.kinds` included `paper_pdf` "
        f"(`paper_format: {config.output.paper_format}`), "
        f"but the paper generator couldn't compile a PDF.\n\n"
        f"## What happened\n\n"
        f"**Reason code:** `{reason.code}`\n\n"
        f"{reason.summary}\n\n"
        f"## How to fix it\n\n"
        f"{reason.how_to_fix}\n\n"
        f"## After fixing\n\n"
        f"To compile just the PDF for this quest (without re-running the "
        f"whole LLM loop), run the following from the repo root. It "
        f"mirrors how `launch.py` invokes the generator on a successful "
        f"run (see `launch.py:_run_generators`).\n\n"
        f"```python\n"
        f"from pathlib import Path\n"
        f"from core.config import Config\n"
        f"from core.engine import QuestArtifacts\n"
        f"from generation.paper import PaperGenerator\n"
        f"\n"
        f"# Replace with the actual quest dir path:\n"
        f"quest_root = Path('outputs/<your_quest_id>').resolve()\n"
        f"cfg = Config.from_yaml(quest_root / 'config.yaml')\n"
        f"art = QuestArtifacts(\n"
        f"    quest_id=quest_root.name,\n"
        f"    quest_root=quest_root,\n"
        f"    paper_md=quest_root / 'paper' / 'paper.md',\n"
        f"    figures_dir=quest_root / 'figures',\n"
        f"    bundle_manifest=None,\n"
        f")\n"
        f"# `art.quest_root` is the same target launch.py passes — the\n"
        f"# generator drops `paper/paper.pdf` (and any other output\n"
        f"# kinds you've configured) under this directory.\n"
        f"PaperGenerator(cfg).generate(art, art.quest_root)\n"
        f"```\n\n"
        f"This file is auto-deleted on the next successful PDF compile.\n"
    )
