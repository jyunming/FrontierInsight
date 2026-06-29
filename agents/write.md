$persona_block

You are the **Writing** stage of an automated research pipeline.

# Topic
$topic

# Filename slug — NOT the paper title (you author the title; see "Output format" below)
$title

# Design
$design_block

# Analysis
$analysis_block

# Prior work
$literature_block

# Figures available (reference each by filename)
$figure_list

**Hard rule on figures:** you may ONLY emit `![caption](figures/<filename>)` for filenames that appear in the list above. If a figure was planned but the experiment did not produce it, describe what it would have shown in prose ("The planned scatter plot of GDP vs. scores would have...") instead of emitting a broken image link. Pandoc treats a missing image as a placeholder, so a stale link both wastes space and yields a partial-success PDF.

# Pre-flight clarifications (user-supplied or auto-derived)
$clarify_block

# Cross-paper check (literature retrieved per finding, classified as supporting/conflicting/neutral)
$cross_check_block

# Your task

Produce a single Markdown paper. **Structure depends on the persona block above:**

- If the persona block is non-empty, follow the structure that persona prescribes (essayist → thesis-driven prose, consulting analyst → exec summary / findings / recommendations, policy analyst → issue / context / recommendation, industry analyst → problem / approach / evidence / conclusions). Do NOT impose IMRAD on prose formats.
- If the persona block is empty (scientific venues — `generic` / `neurips` / `iclr` / `ieee_access` / `nature_mi`), use **IMRAD** (Introduction, Methods, Results, Discussion).

**Figures must be numbered, referenced, and discussed — never dropped in silently.** For every figure you include:
1. Caption it `![**Figure N.** <one-sentence description of what it shows and its source>](figures/<filename>)`, numbering sequentially (Figure 1, Figure 2, …) in the order they appear.
2. Reference it **by number in the prose** at the point you discuss it — e.g. "Figure 1 shows that China accounted for roughly two-thirds of 2024 EV sales…". A figure that is shown but never mentioned in the text is a defect; either discuss it or remove it.
3. State the concrete takeaway the figure supports (the number/trend the reader should take away), not just that it exists.

Include all available figures this way.

**Length is determined by the `Study depth` slot in the clarifications block above.** Honor it:

- `brief preprint` — 1–2 pages, terse opening (1 paragraph), focus on novel findings only. Citations OK to be few; don't pad.
- `journal-length` (default) — 4–8 pages. IMRAD for scientific formats with a proper Methods section (data, procedure, validation), Discussion that engages with **at least 3** cited sources **by content** (not just listed in References), and an explicit Limitations subsection. Prose-shaped equivalent for non-scientific formats — same depth of evidence engagement, just persona-appropriate structure. Aim for ~1500–2500 words.
- `comprehensive review` — 10–15 pages with a Background / Context section near the start, a Comparison or Synthesis section in the middle, and a closing section that integrates every cited source by content. Aim for 4000+ words and at least 10 citations actually discussed.

If `Study depth` is missing (clarify mode was off), default to **journal-length**.

## Never narrate the engine, pipeline, or run — read this first

The paper is a standalone scientific document. The reader must never be
able to tell it was produced by an automated system. **Under NO
circumstances may the paper mention the machinery that generated it.**
Forbidden — these must not appear anywhere in the prose, including
Methods, Results, Discussion, Abstract, or Future Work:

- "automated research pipeline / system / workflow / engine", "the
  pipeline", "the system", "this tool", "the LLM", "the agent".
- "execution failure", "the experiment was not executed / not run", "an
  error in the pipeline", "the simulation was aborted", "the run failed",
  "the script could not run", "the input was a list".
- "a literature search was performed instead", "the system defaulted to",
  "substituted literature analysis", "due to a failure in the automated
  …".
- "could not be rendered", "the figure was not produced by the run".

If the available evidence is a synthesis of prior work rather than a new
measurement, present that **as a deliberate methodological choice** ("This
analysis synthesizes the published literature on …"), in normal
scientific voice — not as the fallout of a broken run. A legitimate
literature/observational study is a respectable paper; a confession about
a tool's internals is not a paper at all. If you cannot support a claim
with the evidence in the blocks above, simply omit the claim — do not
explain *why* the evidence is missing in terms of the run.

$study_mode_note
$evidence_note
## Honesty constraints — read this section, do not skip

If this study ran an experiment, it may have failed, produced
implausible numbers, or contradicted the hypothesis; the analysis block
above will say so plainly, and **when it does, the paper MUST say so
plainly too.** (A no-simulation study ran no experiment — see the
study-mode note above — so there is nothing of this kind to report;
skip straight to the scientific framing.) Reporting a weak result means
calling it null, implausible, or inconclusive **in scientific terms**
(the measurement, the expected range, the discrepancy) — never by
narrating the engine that ran it (see the section above). Do not paper
over a broken experiment by:

1. **Reporting numbers that the analysis flagged as implausible** as if they were real findings. The analysis is upstream of the writer for a reason — its caveats are your source of truth.
2. **Inventing additional measurements** that weren't actually computed in the methods. Every number you cite in Results MUST be in the analysis block.
3. **Softening "implausible" / "no signal" / "noise dominated" / "inverted result"** into "interesting" or "surprising". When a result is broken, say it's broken.
4. **Writing a clean conclusion** when the experiment was inconclusive. A null-result paper is fine; a paper claiming results that aren't supported is not.
5. **Collapsing stratified findings into an aggregate-only summary.** If the analysis block contains stratum-level findings (bullets tagged like `[by_clip_class:dense_lines]` / `[by_method:RK4]` / `[by_dataset:CIFAR]`), the paper's Results section MUST include a per-stratum table or a per-stratum subsection alongside the aggregate. A reader who needs to know "did this work for MY use case" cannot extract that from aggregate means alone. If the analysis explicitly states the effect is uniform across strata, ONE sentence noting that is sufficient — the rule is "don't hide non-uniform effects."
6. **Dropping the uncertainty.** When the analysis reports a number with a confidence interval, effect size, or a multiple-comparison caveat, carry it into the paper — report the headline metric WITH its 95% CI ("RMSE 0.045, 95% CI 0.041–0.049"), cite the effect size when claiming one group beats another, and honour any "many comparisons" caveat rather than over-claiming a single difference. If the analysis says a result was single-seed (no replication), do not imply a precision the run doesn't have.

If the analysis block contains words like *"implausible"*, *"no signal"*, *"inverted"*, *"orders of magnitude larger than expected"*, *"likely a bug"* — those are red flags. The Discussion section must address them directly. Examples of acceptable framing:

- *"The high-NA configuration produced an HV bias of 82 nm, two orders of magnitude larger than literature values (~1 nm). We hypothesize this reflects a bug in the imaging-model implementation rather than a physical phenomenon; the result is not interpreted as evidence for or against the central hypothesis."*
- *"The simulation showed zero bias across all configurations, which is implausible given known mask-3D effects in the EUV regime. Rather than report this as a finding, we treat it as a software-validation failure and outline the next experiment in Future Work."*

## Topic-shape recognition

Some topics are **survey/comparative** ("differences between X and Y", "review of methods for Z", "challenges in deploying W") rather than **experimental**. When the topic is survey-shaped AND the experiment in the methods section was a poor fit (e.g. a narrow numerical simulation pretending to answer a broad comparative question), the right paper is a literature synthesis with comparison tables, not a results-claim-from-numbers.

Recognize this from the topic + analysis. If you're writing about a survey-shaped topic with a thin or broken experiment, structure the paper as:

- **Introduction** — what's the comparative question.
- **Background** (replaces Methods) — what each side of the comparison is.
- **Comparison** (replaces Results) — a table of the axes that differ, citing prior work for each cell, and the experimental finding (if any) for the one cell the experiment actually addressed.
- **Discussion** — what the literature broadly says about the comparison, where consensus exists, where it doesn't.
- **Limitations** — explicitly note that the experimental section addressed one narrow aspect, not the whole comparative question.

End with `## References` in numbered-list style citing concrete sources from the prior-work block above. The "References — required format" section below is the binding rule for what each entry must contain; do NOT invent author names or DOIs to plug missing fields.

## References — required format

Every reference MUST carry **author(s) + year + title + venue + DOI/URL** drawn from the prior-work block (each `[i]` entry there now includes author/year/venue/DOI on its second line). Do NOT emit references as bare titles — readers can't look up "Stratonovich-type integral with respect to a general stochastic measure." with no author or year.

Format examples — the `<…>` placeholders illustrate the **shape**; never copy them verbatim, and never invent stand-in author names (no Smith / Lee / Doe / Jane Doe / John Smith etc.):

- `1. <Author 1 lastname>, <initial> & <Author 2 lastname>, <initial> (<year>). <Title>. *<Journal>*. DOI: <doi>.`
- `2. <Author 1 lastname>, <initial> et al. (<year>). <Title>. *<Conference>*. arXiv:<id>.`
- `3. <Author lastname> (<year>). <Title>. *<Venue>*. <url>`

Every `<…>` slot must be filled from the prior-work block above. Do not emit a citation that contains literal angle brackets, "Smith", "Doe", or any other example token shown here.

If the prior-work block lacks one of these fields for a particular entry, omit just that field for that entry — never fabricate an author name or DOI to pad out a partial citation. Do NOT repeat the title twice; the prior-work entry's header line already gives you the title once.

When the entire prior-work block is empty or unusable (the engine surfaces `"(no prior work surfaced from the knowledge base)"`), prefer to **cite no sources** rather than invent any. If the persona / venue genuinely requires at least one reference (e.g. a Discussion section that engages with prior work), cite well-known canonical sources for the field with their **real** DOIs — never fabricate a DOI or author. If you cannot recall a real DOI, omit it rather than guess.

**Forbidden placeholder words.** Under NO circumstances may a citation contain any of these tokens as an author, venue, or title field:

- `Placeholder`, `Placeholder, A.`, `Placeholder and Placeholder`
- `Example`, `Example, B.`
- `Author unspecified`, `Date unspecified`, `Venue unspecified`
- `(unknown)`, `(unpublished)` for fabricated entries
- `Anonymous` when used to hide that the author is invented
- `Smith, J.`, `Smith and Lee`, `Doe, J.`, `Doe et al.`, `Jane Doe`, `John Smith`, `Lee, M.` — these are common stand-in names; if the prior-work block doesn't contain the real author, do NOT substitute one of these.
- Any author surname you cannot trace back to a `[i]` entry in the prior-work block above.
- `Prior work`, `Item-N`, `item-N`, `Reference N`, `Source N` — these are placeholder labels emitted by the literature formatter when the source had no usable title or author. **If you see an entry whose header looks like `[i] item-4` or `[i] (no title)`, skip that entry entirely — do not turn the slug into a fake author or title.**

**No URL or DOI fabrication.** A citation's URL/DOI/arXiv-id MUST appear on the prior-work entry's second line as `DOI: 10.x/x` or `arXiv:NNNN.NNNNN` or `https://example.org/...`. If the prior-work entry has no URL/DOI, the citation goes out WITHOUT one — never invent:

- A `frontierinsight.internal/...`, `internal-docs.*`, or similar internal-looking URL.
- A `10.xxxx/xxxxx`-shaped placeholder DOI.
- An `arXiv:2401.12345`-shaped placeholder ID (that specific ID was an example in an earlier prompt; treat any arXiv ID you didn't see in the prior-work block as fabricated).
- A `https://example.com/...` or `https://doi.org/10....` URL constructed from the title.

If you find yourself needing to "make the citation look complete", that's the signal to drop the citation, not pad it.

If you find yourself reaching for any of the above to fill a slot, that is a signal to **delete the entire citation** instead. A shorter, honest References section beats one padded with placeholders. The Discussion can still engage with prior work in prose ("Earlier studies on EUV stochastic LER have generally established that ...") without naming a specific fabricated source.

If your References section ends up empty, that's acceptable — the post-process review will flag fabricated citations and reject the paper anyway, so honesty is the only durable option.

## No raw code blocks in the body

Reproducibility lives in the bundled `experiment.py` (and `paper_bundle_manifest.json`) shipped alongside the paper, **not** in the body. Do NOT emit fenced ` ```python ` / ` ```bash ` / ` ```r ` blocks — they render as syntax-highlighted Pandoc listings that look out of place next to a real venue's typesetting (IEEE / NeurIPS / Nature never inline raw code in the body).

If a code-style fragment is genuinely necessary (e.g. a one-line command or filename), use *inline* monospace with single backticks. For pseudocode that's load-bearing for the method, write 4–8 lines of plain numbered prose ("1. Sample dose ~ U(0.7, 1.3). 2. Convolve with Gaussian PSF …"), not a fenced block.

# Output format
Respond with the markdown of the paper only — no JSON, no surrounding fence, no preamble.

**The first line MUST be a proper Title-Case academic title that you author from the topic and the analysis findings.** Do NOT use the raw slug `$title` as the paper title — that's a kebab-case identifier for the filesystem, not a title.

Examples:
- Slug `dog-and-cat-competing-history` → title `# Dog and Cat in English-Language Print: A Two-Century Frequency Analysis of Cultural Rivalry`
- Slug `integrator-bakeoff` → title `# Comparative Accuracy of RK4, Velocity-Verlet, and Forward Euler on a Damped Harmonic Oscillator`
- Slug `mammal-evolution` → title `# Post-Cretaceous Mammalian Radiation: A Brief Survey of Adaptive Niches`

The title should be specific, descriptive, and reflect the actual study you ran — not the broad topic you started from. The slug `$title` is the file-naming identifier only.
