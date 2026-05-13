"""Paper output generator.

For each quest, the engine has already written `paper/paper.md` plus any
`figures/` produced by the experiment. The generator (1) hoists the
markdown to the top of the quest output, (2) optionally compiles it to
PDF via pandoc + LaTeX using a format-specific template, and (3) copies
figures and bundle manifest if they exist.

If pandoc or a LaTeX engine is unavailable, PDF generation is skipped
with a warning — the markdown is still produced.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path

from core.config import Config
from core.engine import QuestArtifacts

_log = logging.getLogger("frontier_insight.paper")

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = REPO_ROOT / "templates" / "paper"


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
            pdf = self._compile_pdf(pdf_src, out_dir)
            if pdf is not None:
                result["paper_pdf"] = pdf

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

    def _compile_pdf(self, paper_md: Path, out_dir: Path) -> Path | None:
        # Pandoc itself: `subprocess.run` on Windows auto-appends `.exe`
        # to bare executable names via CreateProcess, so resolving via
        # `shutil.which` is mostly defensive (would matter if pandoc
        # ever shipped as a .cmd shim like marp does). We do it for
        # symmetry + so the absolute path lands in any stderr logs.
        pandoc_exe = shutil.which("pandoc")
        if pandoc_exe is None:
            _log.warning("pandoc not on PATH; paper.pdf skipped (paper.md only)")
            return None

        # CRITICAL: pandoc does its OWN PATH lookup for the
        # `--pdf-engine` binary. On corporate Windows boxes the
        # MiKTeX bin dir (`~/AppData/Local/Programs/MiKTeX/miktex/bin/x64/`)
        # often isn't on the Python child's PATH even though MiKTeX
        # is installed. Resolving the engine up-front and passing the
        # full path bypasses pandoc's lookup. Falls back to tectonic
        # (no-admin LaTeX) when pdflatex isn't reachable.
        engine = self._find_pdf_engine()
        if engine is None:
            _log.warning(
                "no LaTeX engine found (pdflatex or tectonic); paper.pdf "
                "skipped. Run `python launch.py --install-tectonic` for "
                "a no-admin LaTeX install."
            )
            return None
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
            _log.warning("pandoc invocation failed; paper.pdf skipped")
            return None
        except subprocess.TimeoutExpired:
            _log.warning(
                "%s timeout (>%ds); paper.pdf skipped",
                engine_name, timeout_s,
            )
            return None

        if r.returncode != 0:
            _log.warning(
                "%s rc=%d format=%s; paper.pdf skipped. stderr_tail=%s",
                engine_name, r.returncode, fmt, r.stderr[-500:],
            )
            return None
        if not out_pdf.exists():
            return None
        return out_pdf
