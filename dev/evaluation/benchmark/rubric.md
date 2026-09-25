# Benchmark rubric

How a delivered quest is graded by hand. The scores find FI's own faults (a check that let a wrong number through, a
figure that disagrees with its text, a citation nothing supports); they are **debugging scores, not a ranking of
models**. See [README.md](README.md) for what they can and cannot tell you.

Grade what the quest delivered to the person: `paper.pdf` (with `paper.md` beside it), its figures, the retrieved
sources under `literature/`, the run's results and `run.log`.

## The paper: six dimensions, each a share from 0 to 100

- **D1 Citation usable.** Share of in-text scholarly citations whose number is in the paper's References AND whose
  retrieved text supports the sentence.
- **D2 References on topic.** Share of printed References that are scholarly and rated at least 2 of 3 for relevance
  to the topic.
- **D3 Canonical works.** Share of the topic's canonical works the paper cites. For the SIR benchmark topic: Kermack &
  McKendrick 1927, Gillespie 1976/1977, Whittle 1955 or Bartlett (a share of 3). Record separately whether a miss was
  never retrieved or retrieved and not cited: only the second is a writing fault.
- **D4 Figure matches text.** Share of figures whose plot matches both its caption and what the text says about it.
- **D5 Topic goals met.** Share of the goals the topic sets (six for the SIR topic).
- **D6 Format and layout.** Share of six items: an abstract; keywords; no doubled figure numbers ("Figure 2: Figure
  2."); no nearly empty last page; no page more than about 40% blank mid-paper; no figure printed after the
  References heading (judged by position in the reading order, not by page number).

**Mean** = Σ round(Dᵢ) / 6, shown to one decimal. Each dimension is rounded to a whole number first.

## Three gates, pass or fail

A mean hides faults a journal would reject a paper for outright.

- **G1 Numbers correct.** The paper's quantitative results come from an experiment that computed them correctly. A
  fail is a wrong or impossible value; results missing because the final experiment crashed or never ran; or a
  textbook value presented as the paper's own result. An error fixed before the paper was written passes.
- **G2 Every citation supported.** D1 = 100.
- **G3 Figures match the text.** D4 = 100.

**Target:** G1–G3 all pass and the mean is at least 90. That makes a draft worth handing to a human reviewer. It is
not a publication bar: novelty is not measured.

**Page count** is recorded, not a gate: the delivered PDF's pages against what the topic allows (4 for SIR).

## Slides (`slides.pdf`, checked against `slides.pptx`)

Each a share from 0 to 100.

- **S1 Numbers and conclusions correct.** Content slides whose numbers and claims match the results, not only the
  paper. A slide scores 0 if any number or claim contradicts the results or the figure it shows.
- **S2 Fits.** Slides with no text or figure running off the page, clipped or overlapping.
- **S3 Legible.** Slides whose body text is at least 18 pt at a 7.5-inch slide height (size × 540 / page height in pt),
  with readable axis labels.
- **S4 One point per slide.** Content slides with at most about 6 lines (the bold lead sentence plus each bullet) and
  no prose paragraph of 3 or more sentences.
- **S5 Structure.** Title, research question, method, results with a figure, conclusion, references (6 items).
- **S6 Citations resolve.** Citations on the slides that appear in the slides' or the paper's reference list. N/A (left
  out of the mean, and noted) when the slides cite nothing.

Content slides (S1, S4) exclude the title and closing slides and the references slide.

## Poster (`poster.pdf`)

- **P0 Delivered.** No `poster.pdf` scores every P item 0.
- **P1 Numbers and conclusions correct** (neutral).
- **P2 Headline states the main finding** (guidance): 100 when it is the finding, 0 when it is only the title.
- **P3 Legible** (guidance): body at least 24 pt, headings about 1.5× body, numbered figure captions (3 items).
- **P4 Layout** (neutral): columns end at about the same height, no overflow or overlap, no large empty area (3 items).
- **P5 Structure** (neutral): authors, question, method, results, conclusion (5 items; authors count only when an
  author line is printed).
- **P6 References** (guidance): only cited sources, at most 8, no raw URLs (3 items).

Report two poster means side by side: all of P1–P6, and neutral only (P1, P4, P5).

Layout items (S2, P4) judge page elements against each other. A defect inside a figure image (a figure title over a
panel title) is noted, not scored.

## Grading a run

1. Count what needs no judgement first — citations that resolve, canonical works present, pages, keywords, doubled
   figure numbers, figure order — so the judged items are graded against the same facts every time.
2. Judge D1 against each source's retrieved text (`literature/`), never against what the model remembers of it.
3. Judge D4 and G1 against the run's results (`raw/`, `RESULT_JSON`, `run.log`), not against the paper's own tables.
4. Record every item that decided a score, and every place where a reasonable grader could decide it the other way
   ("alternative: D4 = 100, G3 pass, mean 94.5"). A score whose basis is not written down cannot be re-scored when the
   rubric changes.
5. List each FI fault the run shows (a check that passed a wrong claim, or flagged a right one), separately from the
   model's own mistakes. That list is what the scores are for.

## When the rubric changes

Re-score every run in [scores.md](scores.md) from its item-level record, not by rescaling its old percentage, and
add a dated note to that file saying what changed and why. Changes so far:

- **2026-09-15:** the page count is recorded, not a gate ("≤ 4 pages" is this topic's instruction, not a journal
  rule).
- **2026-09-16:** "within 4 pages" left D6 for the same reason; Anderson & May 1991 left D3 (a general textbook no
  question of this topic depends on, cited in 1 run of 38).
- **2026-09-17:** "no figure printed after the References heading" joined D6, after a top-scoring paper was found by
  eye with a figure after its bibliography.
