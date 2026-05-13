# `generation/`

Post-engine artifact generators. Each runs after `Engine.run()` returns
a `QuestArtifacts` and is gated on its kind being listed in
`config.output.kinds`. Failures are isolated — one generator crashing
does not abort the others (see `launch.py::_run_generators`).

| File | What it produces | External tools |
|---|---|---|
| `paper.py` | `paper/paper.md` is the engine's output; this generator adds `paper/paper.pdf` via pandoc + LaTeX. Templates in `templates/paper/` (`generic`, `neurips`; others stub). Falls back to md-only if pandoc is missing. | pandoc, pdflatex |
| `slides.py` | `slides.md` always (Marp markdown); `slides.html` and `slides.pdf` via the Marp CLI when on PATH; `slides.pptx` via pandoc when on PATH. Each render target is independent — one failure does not stop the others. | marp, pandoc |
| `poster.py` | `poster.tex` from `templates/poster/poster.tex` substitution; optional `poster.pdf` via pdflatex. | pdflatex |
| `speech.py` | `talk.md` — a spoken-form script for the paper. Single LLM call, no rendering. | none |

Each generator gets a `Config` plus the `QuestArtifacts` and writes
into `quest_root`. The `_strip_outer_fence` and `_run_cli` helpers
defined in `slides.py` are module-local — the other generators
implement their own equivalents where needed.
