# Skill import plan — decided by interview, 2026-09-07

The outcome of a structured interview covering the 163 K-Dense + 98 Orchestra
published skills. The survey itself is [`docs/audits/14_skill_library_shortlist.md`](../docs/audits/14_skill_library_shortlist.md);
this is the **policy** that came out of it, plus the filtered lists, so the
decisions survive the session that produced them.

Supporting data in `dev/skill-import/`:

| File | Contents |
|---|---|
| `kdense_digest.json` | Structured digest of all 163 skills — description, section headings, bundled scripts, declared limits |
| `eligible.json` | The 113 surviving the credential and Axon filters, plus what each excluded skill was excluded *for* |
| `after_deps.json` | The 105 surviving the hard-dependency filter |

---

## The correction that reframed everything

The first pass of audit 14 scored every skill against an assumed
semiconductor-physics scope, inferred from the quest YAMLs in the working tree
(`demo_low_k1_duv*`, `interview_euv_ler_shot_noise`, `interview_mosfet_sce`,
`litho_paper_workflow_proof`). That put ~77 life-science skills in a
"wrong domain" tier.

**FI is domain-general.** The user's correction: 「我不想限縮我們是半導體物理研究」.
There is no domain filter. Those YAMLs are current work, not scope.

---

## Policy

### Hard exclusions

| Rule | Count | Basis |
|---|---|---|
| Requires an API key or account | **48** | Each match recorded in `eligible.json` with the literal token found (`MP_API_KEY`, `NCBI_API_KEY`, "API key", "authenticate"…) |
| Literature retrieval | **2** | Axon owns this path; a second competing retriever is a regression |
| Hard GPU or conda requirement | **8** | Optional GPU/conda does not disqualify |
| **Remaining eligible** | **105** | |

### Allowed

- **Every scientific domain.** No filter.
- **Network without credentials** (OLS4, ARAX, GenSpectrum…) — the scanner must surface it.
- **Research process skills** — ideation, hypothesis generation, critique, peer review. Not only software operation.
- **Skills overlapping non-Axon engine paths** (document output, PDF reading) **only if they can carry the existing workflow**, not merely overlap it. This is a per-skill judgement, not a blanket allowance.

### A caveat that must travel with these lists

The dependency and credential classifications are derived **from documentation
prose**, and the first two attempts were badly wrong — the credential filter
falsely excluded 20 skills by matching the word "account" inside
"accountability", and the hardware filter falsely excluded 27 by matching
*optional* GPU mentions. Both were corrected, but the method remains
heuristic.

**The reliable filter is the self-test.** A skill whose dependencies are
absent goes QUARANTINED on its own, on the actual machine. Treat these lists
as a starting shortlist, never as verified facts about what will install.

---

## Catalogue architecture

The domain filter's removal made the catalogue the new cost centre: 105
eligible skills at the measured 1,073 characters per entry is ~28,000 tokens
**per quest**, against the ~93,000 **per pass** that blanket injection cost.
Better, but no longer cheap, and it grows with the library.

The chosen shape:

```
                     ┌─ general layer (39)  ── always present, ~10.5k tokens
topic ─→ [embedding] ┤
                     └─ domain layer(s) (66) ── selected by similarity to the topic
                                                 ↓
                                        [LLM selection] ─→ chosen skills
```

**General layer (39)** — applies regardless of research field:

```
aeon                          analytical-method-validation  arbor
dask                          experimental-design           exploratory-data-analysis
get-available-resources       hypothesis-generation         iso-standards-readiness
markdown-mermaid-writing      markitdown                    matlab
matplotlib                    networkx                      pdf
peer-review                   polars                        pptx-posters
pymoo                         scholar-evaluation            scientific-brainstorming
scientific-visualization      scientific-writing            scikit-learn
scikit-survival               seaborn                       shap
simpy                         statistical-analysis          statistical-power
statsmodels                   sympy                         umap-learn
uncertainty-and-units         vaex                          venue-templates
what-if-oracle                xlsx                          zarr-python
```

**Open boundary cases** — assigned to the domain layer, arguably general:
- `pymc` — Bayesian inference is domain-neutral
- `qutip` — quantum-specific, but its `convergence_sweep.py` / `result_audit.py` are a general numerical-verification idiom

**Domain layer (66)** — roughly: biomedical/genomics (~30), chemistry/drug (~10), physics/fluids/quantum (~8), lab automation and clinical (~10), geospatial (3), other ML (~5).

**Degradation:** if the embedder is unavailable, fall back to the **full**
catalogue with a log line — never to general-only. Selection can only narrow,
so more candidates is fail-safe and visible in the selection report, whereas
silently dropping domain skills changes results invisibly.

**Domain tags** are assigned at import and stored in `provenance.json`. Tags
route the catalogue; they grant nothing, so unhashed storage is safe for them.

---

## Gate changes

### High-severity findings require explicit confirmation

`--approve-skill` refuses when the scanner reports a high-severity finding,
unless `--despite-findings` is also passed. The ledger records that the
approval was made despite findings.

Findings still never quarantine — they are heuristics, and a legitimate skill
driving a network tool will flag one. The flag makes "I read them" an explicit
act rather than an assumed one.

### Auto-generated self-tests must mark themselves

Self-tests are generated rather than hand-written. A generated test proves
**importability and version match — never behaviour**, and the promotion
gate's semantics depend on the difference.

So the generated file carries a sentinel line saying exactly that, and
`--skills` and the approval path detect it and say so. The marker lives in the
generated file — hashed content — so it **self-erases the moment a person
rewrites the test**, which a `provenance.json` flag would not: that would go
stale at exactly the moment it mattered.

Generation uses structured facts only — frontmatter, file listing, import name
— never parsed `compatibility:` prose. For skills shipping `scripts/`, it adds
a `--help` probe per script expecting exit 0, which K-Dense's own contract
already requires. That raises the floor above "it installs" with no per-skill
knowledge.

### A skill's validator may block a quest

A skill can declare that its checker gates the quest, sending a failing run
back through `execute_reflect` the way the plausibility gate does.

**The grant lives in `SKILL.md` frontmatter, not `provenance.json`.** The user
specified provenance; that would reopen the hole closed earlier the same day.
`provenance.json` is deliberately excluded from `content_hash()` so that usage
write-back does not lapse approval — which means a grant stored there could be
repointed at a different script after approval, with the approval intact. That
is the same allowlist-shaped failure as the `scripts/` tamper, proved with a
clean-vs-tampered hash comparison.

Frontmatter is hashed, already parsed, and is what a person actually reads
while approving — so it satisfies the requirement ("the skill must declare it,
visibly, at approval") strictly better.

Implementation extends `_skill_assertions()`, which already merges
skill-declared assertions into the plausibility gate. A blocking validator is
that channel plus a command runner, not a parallel system.

---

## Build order

The user's directive: **do not import yet — build the layering first.**

1. **Layering** — general/domain split, embedding-based layer selection, fail-safe degradation. Verify Axon exposes a usable embedding endpoint before designing around it; do not add a new vector dependency ([[project_axon_knowledge_layer]]).
2. **`--despite-findings`** — qualifies `--approve-skill`, so it belongs on the parser, not the mode group.
3. **Auto-generated self-tests** — with the self-marking sentinel.
4. **Blocking validators** — last: the largest item, and genuinely unexercisable until something real is imported.

Then, and only then: re-scan the real `uncertainty-and-units` by hand, import
it alone, and **a person approves it** — not the agent.
