# Output Generators Audit (Unit 05)

**Date:** 2026-05-15
**Scope:** `generation/` (~770 LOC: paper 384 + slides 183 + poster 130 + speech 67 + `__init__.py` 2; `wc -l`), the five paper-venue
templates under `templates/paper/`, the beamerposter under
`templates/poster/`, and the `_run_generators` wiring in `launch.py`.
**Baseline:** PR #55 (`paper_pdf_skipped.md` diagnostic) and PR #58
(pre-flight + `output.require_pdf` strict mode) just hardened the paper
pipeline. The other three generators (slides, poster, speech) have
**not** received the same treatment. That asymmetry is the central
finding of this audit.

---

## Findings

### 1. Template story — three of five venues are stubs that BREAK pandoc

`core/config.py:38` declares `PaperFormat = Literal["generic",
"neurips", "iclr", "ieee_access", "nature_mi"]` — five accepted venues.
`templates/paper/` ships a directory per venue (`generation/paper.py:33`
points at `TEMPLATES_DIR / fmt / "template.tex"`). Inventory:

| Venue        | File                                              | Lines | Status                                |
|--------------|---------------------------------------------------|-------|---------------------------------------|
| generic      | `templates/paper/generic/template.tex`            | 19    | Real template — has `$title$`, `$body$`. |
| neurips      | `templates/paper/neurips/template.tex`            | 30    | Real template, "minimal NeurIPS-flavored stand-in." |
| iclr         | `templates/paper/iclr/template.tex`               | 1     | **STUB** — single LaTeX comment, no placeholders. |
| ieee_access  | `templates/paper/ieee_access/template.tex`        | 1     | **STUB** — single LaTeX comment, no placeholders. |
| nature_mi    | `templates/paper/nature_mi/template.tex`          | 1     | **STUB** — single LaTeX comment, no placeholders. |

`generation/paper.py:246-252` decides whether to pass `--template`:

```python
if template.exists():
    cmd.extend(["--template", str(template)])
else:
    _log.info("no template at %s; using pandoc default ...", template, fmt)
```

The "else" branch is unreachable for iclr/ieee_access/nature_mi because
their `template.tex` **does** exist — it just contains nothing but a
LaTeX comment. Pandoc invokes the engine with that file as the
`--template` argument, and because a one-line `%` comment is NOT a
valid pandoc template (no `$body$`, no document scaffolding, no
`\begin{document}`), the LaTeX engine exits non-zero. PR #58's
strict-mode and PR #55's `paper_pdf_skipped.md` diagnostic catch the
failure, so this manifests as a **silent PDF skip**, not as a
successful-looking empty PDF (that earlier framing was wrong).

The user-visible regression is still real: someone sets
`paper_format: iclr` and gets `paper.md` + a `paper_pdf_skipped.md`
with an opaque `iclr_rc_<n>` reason, where the actual cause is
"FI shipped an empty template." A clean fall-through to pandoc's
default would have produced a working PDF.

Worth noting the log line at `generation/paper.py:248-252`
("no template at %s; using pandoc default") describes the
correct fallback path that ONLY triggers when the file is absent.
There is no test in `tests/test_paper_gen.py` for the stub-template
case. `test_missing_template_falls_back_to_pandoc_default` at line
131 covers the file-absent case, not the file-is-a-comment case.

### 2. Failure UX — slides/poster/speech have no diagnostic, by design

PR #55 added `paper_pdf_skipped.md` (`generation/paper.py:83-106`) so
that a user who configured `paper_pdf` but didn't get one sees a
markdown file next to `paper.md` explaining exactly which prerequisite
was missing and how to install it. The diagnostic carries a stable
`reason.code` (e.g. `no_pandoc`, `no_latex_engine`, `pdflatex_timeout`,
`pdflatex_rc_1`, `output_missing_after_success`), a one-line summary,
and 1-3 sentences of how-to-fix copy. It even auto-deletes when a
subsequent successful compile lands (line 90-94).

The other three generators have **zero** equivalent. The failure modes:

- **`generation/slides.py:68-78`** — Marp not on PATH: logs `marp CLI
  not on PATH; slides.html/.pdf skipped` and continues. `slides.md`
  may still be produced (since the LLM call ran first), but
  `slides.html` and `slides.pdf` silently disappear with no on-disk
  trace.
- **`generation/slides.py:82-97`** — pandoc not on PATH: logs `pandoc
  not on PATH; slides.pptx skipped` and continues. No diagnostic.
- **`generation/slides.py:166-172`** — non-zero rc from marp/pandoc:
  logs warning with last 400 bytes of stderr, returns False, no file.
- **`generation/poster.py:88-90`** — pdflatex not on PATH: logs
  warning, returns `poster.tex` only. `poster.pdf` silently absent.
- **`generation/poster.py:100-106`** — pdflatex nonzero rc: warning
  logged, returns `poster.tex` only. The user gets a `.tex` file
  they may not know how to compile.
- **`generation/speech.py`** — single LLM call, no external tools. Only
  failure mode is the LLM call itself, which propagates as an
  exception caught by `launch.py:574` (logs `[FI] speech generator
  failed: ...`). No file artifact at all.

The cost to the user: they ask for `output.kinds: [paper_pdf, slides,
poster, speech]`, the quest runs, they look at the quest dir, and find
`paper.md` + `slides.md` + `poster.tex` + ... nothing telling them why
the renderables are missing. Note that `_run_generators` in `launch.py`
catches per-generator exceptions with `print(..., file=sys.stderr)`,
not the per-quest logger — so unless the caller captures stderr the
failure may not even appear in `.fi/run.log`. PR #55 fixed exactly this
discoverability gap for the paper pipeline by writing a sibling
diagnostic file; the other three generators still leak.

### 3. Strict-mode contract — `require_pdf` is paper-only

`core/config.py:323` adds `require_pdf: bool = False`. The wire-up:

- Pre-flight check at `core/engine.py:1611-1689` (`_preflight_paper_pdf`)
  runs at `Engine.run` start, BEFORE any LLM calls. Raises if
  `paper_pdf` is in `kinds`, prereqs missing, and `require_pdf=True`.
- Post-LLM compile-time at `generation/paper.py:117-123` (PR #58
  bot-fix): if the pre-flight passed but pandoc/LaTeX failed later
  (timeout, rc≠0, output file missing), raise so the quest fails
  rather than completing with a `paper_pdf_skipped.md`.
- Launch-layer escape hatch at `launch.py:553-554`: if PaperGenerator
  raises and `require_pdf=True`, re-raise instead of swallowing.

There is no equivalent for slides, poster, or speech. A user can't
say "this quest is for my conference deadline, fail loudly if the
slide deck or poster doesn't render." They get whatever survived.
The economic argument that motivated `require_pdf` — don't burn 15
minutes of LLM cost on a quest doomed to skip its primary
deliverable — applies equally well to slides (Marp render is part of
the contract) and poster (pdflatex render is part of the contract).
Speech has no external dependency, so a `require_speech` flag would
only catch LLM-call failures, which is lower value.

### 4. PaperGenerator engine selection — no YAML override

`_find_pdf_engine` (`generation/paper.py:137-174`) walks a fixed
order: `pdflatex on PATH` → `tectonic on PATH` → `<repo>/tools/tectonic[.exe]`.
The docstring is excellent — it documents the order, the rationale
(warm pdflatex cache beats tectonic's network round-trip), and the
platform-aware `tectonic.exe` vs `tectonic` choice.

There is no way to **override** the order via YAML. A user who has
both pdflatex and tectonic installed but prefers tectonic (e.g.
because their local MiKTeX is stale and they don't want to refresh
its CTAN mirror) has no recourse short of editing `paper.py`. The
`OutputConfig` model at `core/config.py:309-323` lacks any
`preferred_engine` field. The "how to fix" copy in
`paper_pdf_skipped.md` at `generation/paper.py:296-297` even
suggests "raise the timeout in `generation/paper.py:_compile_pdf`"
— a code change, not a config change. That is fine for a niche knob
but it should be configurable for the engine choice itself.

### 5. Concurrency — strict sequential, but the dependency graph allows parallelism

`launch.py:530-578` runs the four generators sequentially. Each
catches its own exceptions so one failure doesn't kill the rest, but
the next generator only starts after the previous one's
`await/return`. The shared-resource graph:

- All four read `art.paper_md` (read-only).
- `SlideGenerator`, `PosterGenerator`, and `PaperGenerator` read
  `art.figures_dir` (read-only). `SpeechGenerator` does NOT — only
  paper/slides/poster surface figures.
- `SlideGenerator` writes `slides.md` (and `slides.html/pdf/pptx`).
- `PosterGenerator` writes `poster.tex` (and `poster.pdf`).
- `SpeechGenerator` writes `talk.md` AND **reads `slides.md`**
  (`generation/speech.py:43-45`). This is the one true dependency.
- `PaperGenerator` writes `paper.md`, `paper.pdf`,
  `paper_pdf_skipped.md`, `figures/`, `paper_bundle_manifest.json`.

So the topological order is: `paper` and `slides` and `poster` can run
in parallel; `speech` must wait for `slides`. PaperGenerator's PDF
compile is sync (`subprocess.run`). `SlideGenerator` is async for its
LLM call AND uses `asyncio.create_subprocess_exec` for the Marp
render. `PosterGenerator` is async for its LLM call but invokes
pdflatex via synchronous `subprocess.run(..., timeout=180)`.
`SpeechGenerator` is async for its single LLM call and has no
subprocess at all. **Practical implication for R8 below:** to truly
parallelize, the poster render must move to `asyncio.to_thread` or
`asyncio.create_subprocess_exec` — otherwise it blocks the event loop
even when wrapped in a task.

Today's sequential ordering means a quest with all four kinds takes
roughly `paper_pdf_time + slides_LLM_time + slides_render_time +
poster_LLM_time + poster_render_time + speech_LLM_time`. Parallelism
would cut this to `max(paper_pdf, slides_LLM + slides_render, poster_LLM
+ poster_render) + speech_LLM`. For a 4-output quest the savings is
~30-60 s; for the fleet path running 8 quests in parallel under
`--max-concurrent=2`, sequential generators are arguably the right
choice (don't add a second axis of fan-out that competes for
LLM-proxy capacity). But for single-quest interactive runs the
parallel structure is leaving wall-clock on the table.

There's also a subtle correctness lever: `_run_generators` shares
**one** `ProxySupervisor` across all four (`launch.py:557, 565, 572`),
which respects the proxy session lifecycle. A parallel rewrite must
preserve this — but the concurrent-startup race is already handled
by `ProxySupervisor.acquire()`'s internal `asyncio.Lock`, so the
real requirement is just to keep sharing the same supervisor and
preserve balanced `acquire`/`release` semantics, not to serialize
startup explicitly.

### 6. TTS / speech.py — there is no TTS

The audit prompt asks about "TTS / speech.py — likely an underused
feature. What provider does it use? How well does it work? Is there a
default voice / language config?"

There is **no text-to-speech anywhere in the codebase.** `speech.py`
is a one-LLM-call generator that produces `talk.md` — a written script
intended to be **read aloud** by a human at ~10 minutes pace
(`agents/speech.md:10`). The output is plain markdown beginning with
`# Talk: <title>`. No `.wav`, no `.mp3`, no Azure/ElevenLabs/Edge-TTS
integration. The module docstring (`generation/speech.py:1-5`) is
explicit: "One LLM call from `paper.md` (and optionally a slides
outline) to a ~10-minute spoken script. No external tools needed."

This is fine and probably correct (TTS audio output is a huge addition
in disk footprint and licensing scope), but the **name** `speech`
oversells. A user asked "does it generate audio?" — the answer is no,
not even optionally. `agents/speech.md:13` confirms: output is plain
markdown. The README and `docs/capabilities.md` should label this
"talk script" rather than "speech" to avoid the implication.

### 7. Templates folder — auditability and missing packages

The two **real** templates:

- `templates/paper/generic/template.tex` (19 lines): minimal article
  class, sane package set: `inputenc`, `geometry`, `graphicx`,
  `hyperref`, `amsmath`, `amssymb`, `booktabs`, `listings`. All of
  these ship in MiKTeX/TeX Live default install. Pandoc placeholders
  `$title$`, `$date$`, `$body$` are present. Author hardcoded to
  "Frontier Insight" — there is no `$author$` plumbing, which is fine
  for a research-pipeline byline.
- `templates/paper/neurips/template.tex` (30 lines): adds `microtype`
  and `authblk`, declares an empty abstract with marketing copy, has
  the same hardcoded "Frontier Insight" author. The header comment
  is honest: "Replace with the official `neurips_2024.sty` when you
  actually submit." It's a stand-in, not a submission-ready file.
- `templates/poster/poster.tex` (29 lines): beamerposter, fixed 48"x36" landscape (121.92x91.44 cm in the source),
  three columns. Uses `string.Template`-style `$title`, `$left`,
  `$middle`, `$right` placeholders (note: bare `$` without trailing
  `$`, matching Python's `string.Template` not pandoc's `$x$`). The
  generator at `generation/poster.py:78-83` uses `safe_substitute`
  precisely because the LLM-supplied LaTeX may contain unrelated
  `$math$` that would break strict `substitute()`.

The three stub templates (iclr, ieee_access, nature_mi) are
single-line LaTeX comments. They claim "Stub: pandoc default is used
until a real template lands here" but in practice pandoc IS handed
the stub as a template (see Finding #1) and ignores `$body$`. The
stub strategy is broken — these files should either (a) be deleted
so the `template.exists()` check returns False and pandoc falls
back, or (b) be filled in with real placeholders.

### 8. Output bundle — `frontier_insight_summary.json` partial coverage

`launch.py:484-494` builds the summary:

```python
summary = {
    "quest_id": art.quest_id,
    "quest_root": str(art.quest_root),
    "provider": cfg.provider.name,
    "outputs": {k: str(v) for k, v in written.items()},
    "paper_md": str(art.paper_md) if art.paper_md else None,
    "paper_pdf": str(written.get("paper_pdf")) if written.get("paper_pdf") else None,
}
```

`written` aggregates the return dicts from all four generators:
`paper_md`, `paper_pdf`, `paper_pdf_skipped`, `figures_dir`,
`bundle_manifest`, `slides_md`, `slides_html`, `slides_pdf`,
`slides_pptx`, `poster_tex`, `poster_pdf`, `speech_md`. So the
JSON's `outputs` field IS a comprehensive partial-success manifest —
the user can read `summary["outputs"]` and see exactly which
artifacts landed. Good.

However: the top-level convenience keys `paper_md` and `paper_pdf`
are paper-only. There are no top-level `slides_pdf`, `poster_pdf`,
or `speech_md` shortcuts. Worse, `paper_pdf_skipped` (the diagnostic
file from PR #55) IS in `written` but only as an opaque entry — a
caller programmatically inspecting the JSON has no flag telling them
"this quest had a paper-pdf failure." They have to look for the
presence of the `paper_pdf_skipped` key, which is a string-matching
contract that isn't documented. A proper status field
(`paper_pdf_status: "ok" | "skipped" | "not_requested"`) would be
better.

The summary also lacks the `paper_format` value, the `require_pdf`
flag, and any of the engine identifiers (which pdf-engine produced
the PDF, which slide renderer succeeded). That telemetry is in the
run.log but not in the structured artifact. For automation that
inspects quest outputs (e.g. the VSCode extension, the GUI in
`gui/`), this is missing.

### 9. Marp YAML frontmatter — claimed but not generated

`generation/slides.py:18-21` documents that "The Marp YAML
frontmatter at the top of slides.md is harmless to pandoc..." and
the agent prompt at `agents/slides.md:13` instructs the LLM to
"Begin with the Marp front-matter block (`---\nmarp: true\n...\n---`)."
This is **prompt-dependent**, not enforced. If the LLM returns a
deck without frontmatter, the Marp render will fail (or default to
the wrong theme). There is no post-LLM validation that the
frontmatter is present. The `_strip_outer_fence` helper at line 176
only handles outer ```` ``` ```` fences, not missing frontmatter.

Compare with `generation/poster.py:113-130` which has a
`_lenient_json` helper that recovers from a fenced JSON response.
Slides has the same risk surface but no recovery.

### 10. Timeout discipline is inconsistent

- `generation/paper.py:261` — 360 s for tectonic, 300 s for pdflatex.
- `generation/slides.py:136` — `_run_cli` default 120 s for marp AND pandoc.
- `generation/poster.py:95` — 180 s for pdflatex (poster).
- `generation/speech.py` — no timeout (LLM call has its own).

The slides timeout of 120 s is the most concerning. A Marp PDF
render on a corporate VPN with a cold puppeteer install (Marp
downloads Chromium on first run, ~150 MB) is plausibly slower. The
poster timeout of 180 s is half the paper timeout (360 s for tectonic)
even though a beamerposter compile loads similar package weight. No
ceiling is configurable; all are hardcoded.

---

## Recommendations

Numbered, ordered by impact-per-effort.

### R1. Delete the three stub templates [impact: high] [effort: trivial]

`templates/paper/iclr/template.tex`, `templates/paper/ieee_access/template.tex`,
and `templates/paper/nature_mi/template.tex` are actively harmful —
they trip `template.exists()` at `generation/paper.py:246` and feed
pandoc a template that is just a `%` comment, which the LaTeX engine
rejects. Strict-mode raises; default mode writes `paper_pdf_skipped.md`
with an opaque `<engine>_rc_<n>` reason — neither result is what the
user expected from picking `paper_format: iclr`.
**Action:** `git rm` all three stub files (keep the directories and
`__init__.py` so `paper_format: iclr` still validates against the
Literal). The existing `else: _log.info("no template at ..., using
pandoc default")` branch then correctly handles these three venues
with pandoc's default LaTeX template, which contains `$body$`.

Add a regression test: `test_iclr_falls_back_to_pandoc_default` mocks
`subprocess.run` and asserts that the constructed `cmd` does **not**
contain `--template`.

### R2. Add `paper_pdf_status` / generic `outputs_status` to summary [impact: high] [effort: 1-2 h]

`frontier_insight_summary.json` should carry, for every kind the user
requested, an explicit status. Schema:

```json
"outputs_status": {
  "paper_md": "ok",
  "paper_pdf": "skipped",
  "paper_pdf_skip_reason": "no_latex_engine",
  "slides_md": "ok",
  "slides_pdf": "ok",
  "slides_pptx": "skipped",
  "poster_tex": "ok",
  "poster_pdf": "skipped",
  "speech_md": "ok"
}
```

The `_PdfSkipReason.code` already exists for paper; propagate it
through the PaperGenerator return value (e.g.
`result["paper_pdf_skip_reason"] = skip_reason.code`) and surface in
`launch.py:484-494`. For the other three generators, introduce a
parallel skip-reason object — see R3.

### R3. Add `*_skipped.md` diagnostics for slides/poster/speech [impact: high] [effort: 4-6 h]

Mirror the `_PdfSkipReason` + `_render_pdf_skip_md` pattern in
`generation/slides.py` and `generation/poster.py`. The diagnostic
files would be `slides_skipped.md` (for the failed render targets —
html, pdf, pptx) and `poster_pdf_skipped.md`. Speech is excluded
because its only failure is an LLM exception already surfaced at
`launch.py:574`.

The diagnostic should call out:
- which render target was requested (html, pdf, pptx, or all);
- which CLI was looked up and where (`shutil.which("marp")`);
- the install command for the missing tool (`npm install -g @marp-team/marp-cli`);
- the same "how to re-render from existing slides.md" recipe that
  `paper_pdf_skipped.md` provides for PDF re-compile.

This brings the slides/poster failure UX to parity with paper.pdf.

### R4. Add `require_slides`, `require_poster` strict-mode flags [impact: medium] [effort: 2-3 h]

Mirror `require_pdf` at `core/config.py:323`. Pre-flight check at
engine startup verifies marp/pandoc/pdflatex are reachable for the
requested kinds; raise if `require_X=True` and prereq missing. The
launch-layer escape hatch at `launch.py:553-554` extends to slide
and poster generators. No new code paths — pure copy-paste of the
paper pattern.

Skip `require_speech` — its only failure mode (LLM exception) is
already strict by default (no swallowed exception in `speech.py`
beyond the `launch.py:574` outer catch).

### R5. Expose `output.preferred_engine` YAML knob [impact: low] [effort: 1 h]

Add `preferred_engine: Literal["auto", "pdflatex", "tectonic"] = "auto"`
to `OutputConfig`. `_find_pdf_engine` at `generation/paper.py:137`
respects it: `"pdflatex"` forces pdflatex-or-skip,
`"tectonic"` forces tectonic-or-skip, `"auto"` preserves today's
behavior. This unblocks the user who has both engines installed but
wants the tectonic-only deterministic path (e.g. for CI
reproducibility).

### R6. Rename `speech` to `talk_script` [impact: low] [effort: 30 min]

The current naming implies audio output. Rename:
- `generation/speech.py` → `generation/talk_script.py`
- `OutputKind` Literal in `core/config.py`: `"speech"` → `"talk_script"`
- `agents/speech.md` → `agents/talk_script.md`
- The output file is already `talk.md`, which is honest. Keep that.

Backward compat: accept both `"speech"` and `"talk_script"` in the
config parser for one minor release with a deprecation warning. PR
should update `docs/capabilities.md` and the VSCode extension
in the same commit per the repo's "always update docs with features"
convention.

### R7. Parallelize paper/slides/poster, gate speech on slides [impact: medium] [effort: 3-4 h]

Restructure `launch.py:_run_generators`:

```python
paper_task = asyncio.to_thread(PaperGenerator(cfg).generate, art, art.quest_root)
slides_task = SlideGenerator(cfg).generate(art, art.quest_root, supervisor=supervisor)
poster_task = PosterGenerator(cfg).generate(art, art.quest_root, supervisor=supervisor)
paper_r, slides_r, poster_r = await asyncio.gather(
    paper_task, slides_task, poster_task, return_exceptions=True,
)
# Speech reads slides.md; run after slides completes.
speech_r = await SpeechGenerator(cfg).generate(art, art.quest_root, supervisor=supervisor)
```

Save 30-60 s of wall-clock per quest. Gate this on a config flag
`output.parallel_generators: bool = True` so the fleet path can opt
out (fleet already has its own concurrency axis). Verify the
`ProxySupervisor` correctly serializes three concurrent
`resolve_endpoint_async` calls — proxy startup is the one shared
resource.

### R8. Validate Marp frontmatter post-LLM [impact: low] [effort: 1 h]

`generation/slides.py:130` writes the LLM output to `slides.md`
unchanged after `_strip_outer_fence`. Add a check: if the file does
not start with `---\nmarp:`, prepend a default Marp frontmatter
block. Equivalent to `poster.py:113`'s `_lenient_json` recovery
pattern — protect against a single-call hiccup.

### R9. Standardize timeouts via a `OutputConfig.render_timeout_s` map [impact: low] [effort: 2 h]

Today: 360/300 (paper), 120 (slides), 180 (poster) are hardcoded.
Expose as a single nested config object:

```yaml
output:
  render_timeout_s:
    pdflatex: 300
    tectonic: 360
    marp: 240
    pandoc_pptx: 120
    poster_pdflatex: 240
```

Default values mirror current behavior; corporate-VPN users can bump
without code edits.

### R10. Add a stub-template guard test [impact: low] [effort: 30 min]

In `tests/test_paper_gen.py`, add a parametrized test that confirms
every `PaperFormat` value either (a) maps to a file that contains
`$body$` OR (b) has no `template.tex` (falls through to pandoc
default). This regression-proofs R1 and any future template
additions.

---

## References

- `generation/paper.py` — pandoc + LaTeX compile pipeline, 384 LOC.
  Engine search at `:137-174`; `_compile_pdf` at `:176-339`;
  diagnostic renderer at `:342-384`; strict-mode raise at `:117-123`.
- `generation/slides.py` — Marp + pandoc, 183 LOC. `_run_cli`
  swallowing pattern at `:135-173`; LLM author at `:101-132`.
- `generation/poster.py` — beamerposter, 130 LOC. JSON parse at
  `:113-130`; `safe_substitute` reasoning at `:54-58, :78-83`.
- `generation/speech.py` — talk script, 67 LOC. No external tool
  invocation.
- `launch.py:_run_generators` (`:530-578`) — sequential dispatch +
  exception isolation; strict-mode escape hatch at `:553-554`.
- `core/config.py:38, :309-323` — `PaperFormat` Literal +
  `OutputConfig` (`kinds`, `paper_format`, `require_pdf`).
- `core/engine.py:1611-1689` — `_preflight_paper_pdf`, the strict-mode
  pre-flight added in PR #58.
- `templates/paper/generic/template.tex` (19 lines, real),
  `templates/paper/neurips/template.tex` (30 lines, real
  stand-in), `templates/paper/iclr/template.tex` /
  `templates/paper/ieee_access/template.tex` /
  `templates/paper/nature_mi/template.tex` (1 line stubs).
- `templates/poster/poster.tex` — 29 lines, beamerposter 3-column.
- `agents/slides.md`, `agents/poster.md`, `agents/speech.md` — 13, 25,
  13 lines respectively.
- `tests/test_paper_gen.py` — 26 tests, 798 LOC. Strong coverage of
  paper.py state machine; no coverage of stub-template hazard
  (Finding #1) or `require_pdf` interaction with stub templates.
- `tests/test_slides_speech.py` (428 LOC) and `tests/test_poster.py`
  (213 LOC) — predate the diagnostic/strict-mode pattern and would
  need additions for R3/R4.
- PR #55 (`paper_pdf_skipped.md` diagnostic), PR #58 (pre-flight +
  `require_pdf` strict mode), PR #58 bot-fix
  (`generation/paper.py:117-123`).
