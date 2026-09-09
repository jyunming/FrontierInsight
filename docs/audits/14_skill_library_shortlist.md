# 14 — Complete Review of the Published Scientific Skill Libraries

**Audit date:** 2026-09-07
**Scope:** Every skill in the two published libraries, read and classified — what FI should import, what it should not, and what the exercise revealed about FI's own selection machinery.
**Follows:** [`13_agent_skills_ecosystem.md`](13_agent_skills_ecosystem.md), which established the standard and the verification gap.

---

## What was read

Both repositories were cloned and **every `SKILL.md` was read** — not sampled, not inferred from names.

| | Skills | Total `SKILL.md` |
|---|---|---|
| K-Dense-AI/scientific-agent-skills | **163** | 2.18 MB |
| Orchestra-Research/AI-Research-SKILLs | **98** | 1.32 MB |

A correction to audit 13: it cited Orchestra as "87 skills across 23 categories" from the README. The repository actually contains **98** `SKILL.md` files across 23 directories. The README undercounts.

Structural facts across K-Dense's 163:

| Property | Count |
|---|---|
| Ship a `scripts/` directory | 105 |
| Ship a `references/` directory | 154 |
| Ship a `validate_` / `check_` / `audit_` / `lint_` CLI | **37** |
| Declare their limits in prose ("does not cover…") | 25 |
| Standard-library only | 24 |
| Explicitly network-free | 25 |
| Require an API key or account | 30 |
| Require network access | 34 |
| **Fully offline *and* credential-free** | **15** |

---

## The headline finding

Audit 13 concluded, from one skill, that the large libraries are "strong where FI is weak (domain breadth) and weak exactly where FI has been investing (verification)."

**Reading all 163 shows the second half of that is wrong.**

K-Dense's house style is **deterministic local validators**. Thirty-seven skills ship a `validate_*` / `check_*` / `audit_*` / `lint_*` CLI, most of them standard-library-only, network-free, and driven by exit codes. This is not a library of API wrappers with a verification skill bolted on; verification is the idiom throughout. A representative sample of what those CLIs actually do:

- `uncertainty-and-units` — `check_plausibility.py`, `audit_units.py`
- `qutip` — `qobj_model_validator.py`, `convergence_sweep.py`, `result_audit.py`
- `hypothesis-generation` — `lint_causal_claims.py`, `check_falsification_controls.py`, `check_operationalization.py`, `audit_evidence_ledger.py`
- `scientific-writing` — `audit_claims.py`, `check_references.py`, `check_consistency.py`
- `scientific-visualization` — `palette_audit.py`, `image_metadata.py`, `export_plan.py`
- `pufferlib` — `env_contract_validator.py`, `repro_plan.py`
- `matlab` — `scan_m_code.py`, `reproducibility_report.py`
- `simpy` — `validate_simulation_config.py`, `replication_runner.py`
- `geopandas` — `geometry_validity_report.py`, `crs_reprojection_plan.py`, `spatial_join_audit.py`

That is the same architectural bet FI made with its numeric oracle and plausibility gate — arrived at independently, and applied across a hundred-plus domains. The convergence is much stronger evidence for the approach than one skill was.

It also sharpens what FI is uniquely doing. These validators check an *artifact* a person hands them: this code, this figure, this config. FI's oracle checks the **link between a generated paper and the run that produced it** — a check that only exists inside a closed research loop. The two are complementary, not competing, and audit 13's revised recommendation 3 stands: reject LLM-judged rigor, adopt deterministic checkers.

---

## Tier 1 — Import these (6)

Chosen for verification value and design-stage leverage, not domain coverage.

### `uncertainty-and-units` — take this one first

Six offline CLIs. The two that matter to FI:

**`check_plausibility.py`** — tests quantities against 14 dimensionless groups (Reynolds, Knudsen, Péclet, Damköhler, Biot…), characteristic scales (Debye length, diffusion time, Stokes settling), and curated magnitude bands. Verifies each formula's dimensionality *before* computing, and refuses a kinematic viscosity where a dynamic one is required. Exit status 1 on an implausible verdict.

**`audit_units.py`** — static analysis of generated code, parsing without importing. Nine rules with severities and suppression directives:

| Rule | Severity | Detects |
|---|---|---|
| `UNIT003` | high | `.magnitude` without a preceding `.to()` / `.m_as()` |
| `UNC001` | high | `curve_fit` without `absolute_sigma` |
| `UNC004` | high | a `ufloat` rebuilt from `.nominal_value` and `.std_dev` |
| `UNIT004` | medium | logarithmic units, where `+` multiplies |
| `CONST001` | low | a literal within 0.1% of a CODATA constant |

Every high-severity rule describes a defect that **runs without error and produces a plausible number** — the class FI's plausibility gate exists to catch and its numeric oracle cannot see. It adds the axis neither covers: whether the number is possible *at all*, against physics the quest never declared.

*Requires pint, uncertainties, NumPy ≥ 2.5, SciPy ≥ 1.18, Python 3.12+. The static auditor is standard-library only.*

### `experimental-design`

Far more general than the name suggests. The decision tree is Fisher's principles plus DOE, and `scripts/doe_designs.py` supplies `full_factorial`, `fractional_factorial`, `plackett_burman`, `central_composite`, `box_behnken`, and **`latin_hypercube`** — taking factors as real-unit `(low, high)` ranges and returning designs with run order randomised.

One branch is FI's own use case verbatim: *"Explore a simulation/computer model over a continuous space → SPACE-FILLING design: Latin hypercube."* Its listed failure mode "optimizing without curvature — a two-level factorial can't detect a curved response, you'll miss an interior optimum" applies directly to lithography parameter sweeps. Examples are bio-flavoured (mice, plates); the machinery is not.

### `statistical-power` and `statistical-analysis`

Sample size, minimum detectable effect, and power curves (closed-form plus Monte Carlo for designs with no formula); then test selection, assumption checking, and effect sizes. These pair with `experimental-design` as one unit — the library cross-references them explicitly — and they cover the ground the methodologist persona currently reasons about unaided.

### `scientific-visualization`

*"Create and audit truthful, accessible, publication-ready figures."* Network-free CLIs for palette audit, image-metadata validation, and journal export planning. FI already produces figures; this encodes the conventions and, more usefully, checks them.

### `sympy`

Exact symbolic algebra, calculus, and `lambdify`/LaTeX code generation. General to any physics derivation, minimal dependencies, low integration risk.

---

## Tier 2 — Evaluate against a real quest (47)

Plausible but unproven for FI's work. **Highlights**, with the offline-and-credential-free ones marked ✓:

| Skill | Why it might earn a place |
|---|---|
| `qutip` ✓ | Quantum dynamics *"where physical assumptions, dimensions, and numerical convergence must be explicit"* — ships `convergence_sweep.py` and `result_audit.py`. The closest thing in either library to FI's own philosophy. |
| `pymoo` | NSGA-II/III, Pareto fronts, constraint handling — plausibly source-mask optimisation. |
| `pymatgen` ✓ | Crystal structure, phase diagrams, symmetry sensitivity; bounded Materials Project queries. |
| `simpy` ✓ | Discrete-event simulation with replication, warm-up, and reproducible output analysis. |
| `fluidsim` ✓ | CFD with *explicit numerical-validity checks*. **Carries its own security warning**: `--modify-params` executes supplied Python. |
| `get-available-resources` ✓ | Host CPU/memory/disk/accelerator limits for resource-aware planning before a run. |
| `optimize-for-gpu` | GPU-accelerates scientific Python **and verifies the result is correct and faster**. |
| `arbor` | Hypothesis Tree Refinement — iteratively improve an artifact against an objective and evaluator *without overfitting*. Conceptually close to FI's ideate/repair loops. |
| `hypothesis-generation` ✓ | Evidence-bounded questions, rival explanations, discriminating predictions, preregistration scaffolds; `lint_causal_claims.py`. |
| `exploratory-data-analysis` ✓ | Missingness/leakage audits; unknown formats **fail closed**. |
| `matlab` | Genuine tool-operation skill with `scan_m_code.py` and `reproducibility_report.py`. |
| `astropy` | Matches the exoplanet-figure work already in this tree. |

Remainder of tier 2: `openpiv`, `polars`, `dask`, `vaex`, `zarr-python`, `networkx`, `statsmodels`, `pymc`, `scikit-learn`, `shap`, `scikit-survival` ✓, `umap-learn`, `aeon`, `timesfm-forecasting`, `matplotlib`, `seaborn`, `lab-hardware-cad`, `markdown-mermaid-writing`, `venue-templates`, `latex-posters`, `pptx-posters` ✓, `scientific-slides`, `research-grants`, `database-lookup`, `iso-standards-readiness`, `analytical-method-validation`, `scientific-schematics`, `scientific-brainstorming`, `modal`, `nextflow`, `pytorch-lightning`, `torch-geometric`, `transformers`, `stable-baselines3`, `pufferlib` ✓.

---

## Tier 3 — Do not import (110)

### Off-domain life and health sciences, plus vendor platforms (77)

`adaptyv`, `anndata`, `arboreto`, `benchling-integration`, `bids`, `biopython`, `bioservices`, `bulk-rnaseq`, `cellxgene-census`, `cirq`, `clinical-decision-support`, `clinical-reports`, `cobrapy`, `datamol`, `deepchem`, `deepspot-m`, `deeptools`, `depmap`, `diffdock`, `dnanexus-integration`, `esm`, `etetoolkit`, `flowio`, `geniml`, `genomic-coordinates`, `genomic-intelligence`, `gget`, `ginkgo-cloud-lab`, `glycoengineering`, `gtars`, `histolab`, `hugging-science`, `imaging-data-commons`, `labarchive-integration`, `lamindb`, `latchbio-integration`, `matchms`, `medchem`, `molecular-dynamics`, `molfeat`, `ncats-arax`, `neurokit2`, `neuropixels-analysis`, `omero-integration`, `onekgpd`, `ontology-term-resolution`, `opentrons-integration`, `pacsomatic`, `pathml`, `pathogen-variant-surveillance`, `pathway-enrichment`, `pennylane`, `phylogenetics`, `pkpd-modeling`, `polars-bio`, `primekg`, `protocolsio-integration`, `pydeseq2`, `pydicom`, `pyhealth`, `pylabrobot`, `pyopenms`, `pysam`, `pytdc`, `qiskit`, `rdkit`, `relsa-severity-assessment`, `rowan`, `scanpy`, `scikit-bio`, `scvelo`, `scvi-tools`, `tamarind`, `tiledbvcf`, `torchdrug`, `treatment-plans`, `waypoint-bio`.

Excellent work, wrong domain. Importing them inflates the catalogue every selection call reads while never being selected. Note `qiskit` / `cirq` / `pennylane` sit here as adjacent physics rather than off-domain — promote them if quantum work ever starts.

### Duplicates engine infrastructure (20)

`literature-review`, `paper-lookup`, `research-lookup`, `bgpt-paper-search`, `exa-search`, `parallel-web`, `paperclip`, `paperzilla`, `citation-management`, `pyzotero`, `pdf`, `docx`, `xlsx`, `pptx`, `markitdown`, `liteparse`, `open-notebook`, `scientific-writing`, `infographics`, `generate-image`.

FI's literature path is Axon; a second retrieval path competing with it is a regression, not a capability. Reading documents is `data_load`; producing them is pandoc, already engine infrastructure.

**The generalised rule: a skill feeds `design` and `implement`, so anything the engine already does on its own is not a skill candidate.** The catalogue is the wrong place to solve a problem the DAG already solves. (This is the pandoc lesson, restated.)

`scientific-writing` is the painful one to exclude — its `audit_claims.py` and `check_references.py` are genuinely good — but FI's `write.md`, `claim_check` and evidence gate already own that stage, and two claim-auditors disagreeing inside one pipeline is worse than one.

### LLM-judges-quality (3)

`peer-review`, `scholar-evaluation`, `scientific-critical-thinking`. FI has the persona set and the evidence gate; a second, weaker reviewer dilutes rather than strengthens. This is the pattern the verification survey calls insufficient.

### Speculative, or a different acquisition model (10)

`consciousness-council`, `what-if-oracle`, `dhdna-profiler`, `hypogenic`, `autoskill`, `pi-agent`, `usfiscaldata`, `market-research-reports`, `geomaster`, `geopandas`.

---

## Orchestra: nothing to import, one idea worth stealing

All 98 read. The 23 categories are `model-architecture`, `tokenization`, `fine-tuning`, `mechanistic-interpretability`, `data-processing`, `post-training`, `safety-alignment`, `distributed-training`, `infrastructure`, `optimization`, `evaluation`, `inference-serving`, `mlops`, `agents`, `rag`, `prompt-engineering`, `observability`, `multimodal`, `emerging-techniques`, `ml-paper-writing`, `research-ideation`, `agent-native-research-artifact`.

This is **ML/LLM engineering**, not science. For semiconductor physics essentially nothing is applicable, and audit 13's finding stands: no skill anywhere in it addresses experiment design, numerical verification, or result reproducibility.

Three items are worth knowing about anyway, as design references rather than imports:

**`prompt-guard`** — Meta's 86M-parameter prompt-injection and jailbreak detector, claimed 99%+ TPR at <1% FPR, <2 ms on GPU. Relevant to the gap FI's new `core/skills/scan.py` fills. FI's scanner is deliberately deterministic — it must not itself be a prompt-injection target, and it must run with no model or network — so this is not a replacement. It is a plausible *second* layer if the static rules prove too coarse in practice.

**`ara-research-manager`** — records research provenance as a post-task epilogue, scanning the session's history to extract decisions and experiments. That is the shape of FI's unbuilt acquisition path 2 (distil a skill from completed quests).

**`autoskill`** (K-Dense, listed above under a different acquisition model) — its pipeline is worth copying even though the skill itself is not importable, since it needs a screen-recording daemon:

```
cluster repeated workflows
  → match against existing skills with local embeddings (top-k)
  → LLM judge classifies: reuse / compose / novel
  → stage proposals for review
  → a person promotes
```

**FI has no `reuse / compose / novel` trichotomy**, and "compose" — deciding that chaining existing skills already covers a workflow, and emitting a thin recipe rather than a new skill — is a good idea we lack. The staging-then-human-promotion step matches FI's approval model exactly.

Worth noting: `autoskill` would trip FI's new scanner hard (network access, reads `ANTHROPIC_API_KEY` and `SCREENPIPE_TOKEN`) — and it is entirely legitimate. That is the clearest available argument that findings must inform rather than block.

---

## What the full read revealed about FI's own machinery

Two defects, measured against `core/skills/selection.py` rather than predicted. Both are now fixed; recorded here because the *cause* generalises.

**Descriptions were truncated past the point of usefulness.** `MAX_DESCRIPTION` was 110 characters, sized for FI's own prose descriptions. K-Dense descriptions are keyword-dense and long *by design*, because the standard's discovery stage matches on them: `uncertainty-and-units` runs to 845 characters, in three sentences of 115 / 568 / 160. The substance is in the second, the trigger phrases in the third. At 110 the catalogue showed a mid-word fragment naming neither.

Raising the cap to 600 **did not fix it** — the sentence-selection helper stopped at the first boundary under the cap, still yielding one 115-character sentence. The fix needed both a cap sized from the data (900) and a helper that fills to the budget. Measured cost: 1,073 characters per entry, ~13k tokens for 50 such skills, once per quest, against ~93,000 per pass for blanket injection. The old "under a thousand tokens" claim no longer holds and has been corrected in `docs/capabilities.md`.

**`NOT for:` would have been empty for most imports.** `scope_limit()` looked only for a `## When NOT to use` heading. Only 25 of 163 skills state limits that way; the rest put them in prose inside a Scope section. A prose fallback now recovers them — `uncertainty-and-units` yields *"It does not cover statistical inference, model selection, or study design."*

The generalisable lesson: **FI's conventions were calibrated on FI's own skills, and both broke on contact with real external ones.** Anything else assuming FI-shaped input should be suspected before the next import, not after.

---

## The blocker is now cleared, with one caveat

Audit 13's recommendation 2 — scan imported skills — is **built**: `core/skills/scan.py`, 26 rules, parsing without importing or executing, wired into `evaluate()` so hand-authored and entry-point skills are covered too, surfaced by `--scan-skill`, at import, and at approval.

Verified against the real `uncertainty-and-units` (19 KB `SKILL.md` + 6 scripts, ~118 KB): **one INFO finding, nothing higher** — it surfaces `allowed-tools: Read Write Edit Bash`, which is exactly what an approver should know. A seeded hostile skill produced 13 findings (8 high). Both controls were re-run after two rules were widened.

**The caveat:** the committed regression test uses a *reduced replica* of that skill, not the real files — the genuine control needs the network. Re-run the scanner against the downloaded skill by hand before importing anything for real.

And what the scanner does not cover, stated plainly: a poisoned `references/` file read by generated code **at runtime** is a channel it cannot clear (naming rather than inlining reduces this, but does not remove it), and obfuscated Python (`getattr(__import__('o'+'s'), …)`) passes the AST import checks. It raises the cost of a lazy payload, not a determined one — which is why it never reports "safe".

---

## Suggested order

1. **Re-scan the real `uncertainty-and-units` files by hand.** One command, and it is the control the test suite cannot run.
2. **Import it alone.** Write its `selftest.py` — probe `pint` and `uncertainties` at the pinned versions, then one known-answer round trip through `check_plausibility.py` and `audit_units.py`.
3. **A person approves it.** Not the agent: the no-anonymous-approver rule exists so that a human decides, and an agent approving its own import turns the gate into a rubber stamp.
4. **Run one real lithography quest with and without it** — `engine.skills_exclude` makes the A/B direct — and judge the rest of tier 1 on what that shows.

Importing one skill and measuring it is worth more than importing six. The measurement is the same seeded-defect experiment (roadmap D3) that would turn FI's in-loop oracle from a design claim into evidence.

---

## References

- K-Dense-AI/scientific-agent-skills — https://github.com/K-Dense-AI/scientific-agent-skills
  - Paper: Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents.* arXiv:2609.00065 — https://arxiv.org/abs/2609.00065
  - `scan_skills.py` / `scan_pr_skills.py` — their scanner: an orchestration wrapper over a `skill_scanner` package and a Claude API call, i.e. LLM-based. FI's is deterministic by deliberate contrast.
- Orchestra-Research/AI-Research-SKILLs — https://github.com/orchestra-research/AI-research-SKILLs
- JCGM 100:2008 (GUM) and JCGM 101:2008, which `uncertainty-and-units` implements
- FI: `core/skills/scan.py`, `core/skills/selection.py`, `core/numeric_oracle.py`, `core/plausibility.py`
