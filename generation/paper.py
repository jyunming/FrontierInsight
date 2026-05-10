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

    def _compile_pdf(self, paper_md: Path, out_dir: Path) -> Path | None:
        if shutil.which("pandoc") is None:
            _log.warning("pandoc not on PATH; paper.pdf skipped (paper.md only)")
            return None

        fmt = self.config.output.paper_format
        template = TEMPLATES_DIR / fmt / "template.tex"
        out_pdf = out_dir / "paper.pdf"

        cmd: list[str] = [
            "pandoc",
            str(paper_md),
            "-o", str(out_pdf),
            "--pdf-engine=pdflatex",
            "--standalone",
        ]
        if template.exists():
            cmd.extend(["--template", str(template)])
        else:
            _log.info(
                "no template at %s; using pandoc default (paper_format=%s)",
                template, fmt,
            )

        try:
            r = subprocess.run(
                cmd, capture_output=True, text=True, cwd=str(out_dir), timeout=120
            )
        except FileNotFoundError:
            _log.warning("pandoc invocation failed; paper.pdf skipped")
            return None
        except subprocess.TimeoutExpired:
            _log.warning("pandoc timeout (>120s); paper.pdf skipped")
            return None

        if r.returncode != 0:
            _log.warning(
                "pandoc rc=%d format=%s; paper.pdf skipped. stderr_tail=%s",
                r.returncode, fmt, r.stderr[-500:],
            )
            return None
        if not out_pdf.exists():
            return None
        return out_pdf
