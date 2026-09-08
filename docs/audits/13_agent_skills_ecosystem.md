# 13 — Agent Skills Ecosystem and the Verification Gap

**Audit date:** 2026-09-07
**Scope:** Is there a skills ecosystem FI should join rather than reinvent, and what does the literature say about verifying an AI scientist's results?
**Follows:** [`09_ai_scientist_landscape.md`](09_ai_scientist_landscape.md) (2026-05-15), which compared workflows but predates the skills concept entirely.

---

## Executive summary

Three findings, in order of how much they should change what FI does.

**1. Agent Skills is an open standard with 40+ implementers, and FI's home-grown envelope is close to it but not conformant.** Conforming would let FI consume two large existing libraries — 163 scientific skills and 98 research skills (counts from the repositories; both READMEs differ) — instead of authoring every skill by hand. This is the highest-leverage item in this audit: it changes the cost of everything else in the skills roadmap.

**2. FI's selection design independently reproduced the standard's own loading model.** The spec describes three-stage *progressive disclosure*: discovery loads only name + description, activation loads the full `SKILL.md`, execution runs bundled code. FI arrived at the same architecture from a token-cost argument (blanket injection of 50 skills = ~93,000 tokens/pass; a catalogue of the same 50 = ~13,000 tokens once, measured on real published skills in audit 14). Convergence from an independent premise is a good sign the shape is right, and it means FI can now describe its behaviour in the standard's vocabulary rather than its own.

**3. FI has no defence against a hostile skill, and recent work has made that worse.** The largest scientific skill library runs a dedicated scanner for prompt injection, data exfiltration and malicious code. FI runs nothing — while supporting `--import-skill` from arbitrary paths and entry-point discovery from any installed package. A skill's `SKILL.md` goes directly into a prompt and its `selftest.py` is executed. Human approval is the only barrier, and reading a prompt-injection payload out of prose is not something a person reliably does.

On verification, the literature's finding is directly relevant to what FI just built: a 2026 survey of 24 runnable systems reports that **no LLM-era system in its corpus demonstrates an externally validated in-loop oracle**. FI's numeric oracle and plausibility gate are in-loop and non-LLM, which is the shape identified as missing — though *internally* rather than externally validated, and that distinction should not be blurred.

---

## Findings

### 1. The Agent Skills standard

[agentskills.io](https://agentskills.io/) — originally developed by Anthropic, released as an open standard, now implemented by 40+ agent products including Cursor, GitHub Copilot, VS Code, Codex, Gemini CLI, OpenHands, Goose, Roo Code, Letta, Snowflake Cortex Code, Databricks Genie Code, Kiro and Amp.

**Format.** A folder containing `SKILL.md` with frontmatter carrying at minimum `name` and `description`, plus optional directories:

```
my-skill/
├── SKILL.md          # Required: metadata + instructions
├── scripts/          # Optional: executable code
├── references/       # Optional: documentation
├── assets/           # Optional: templates, resources
```

**Loading model — the part that matters most.** The spec defines three-stage progressive disclosure:

1. **Discovery** — at startup, agents load only the name and description of each skill, "just enough to know when it might be relevant"
2. **Activation** — when a task matches a description, the full `SKILL.md` is read into context
3. **Execution** — the agent follows the instructions, optionally running bundled code

> "Full instructions load only when a task calls for them, so agents can keep many skills on hand with only a small context footprint."

That is `select_skills` (`core/skills/selection.py`), arrived at independently. FI's catalogue carries name, kind, description, scope limit and track record — a superset of the standard's discovery payload.

### 2. Where FI's envelope diverges

| Concern | Agent Skills standard | FI today |
|---|---|---|
| Instructions | `SKILL.md` + frontmatter (`name`, `description`) | same ✓ |
| Executable code | `scripts/` | `skill.py` |
| Documentation | `references/` | `api_surface.md` |
| Templates / data | `assets/` | — |
| Validation | not in the core spec | `selftest.py` (required for promotion) |
| Provenance | not in the core spec | `provenance.json` |
| Discovery root | host-defined | `FI_SKILLS_DIR`, else `~/.frontier-insight/skills` |

FI's additions (`selftest.py`, `provenance.json`, `kind`) are not in conflict with the standard — the spec permits "any additional files or directories". The divergence is that FI puts code and docs in single files where the standard uses directories, so a standard skill would not be read correctly today, and an FI skill is not portable.

### 3. Existing libraries FI could consume

**[K-Dense-AI/scientific-agent-skills](https://github.com/K-Dense-AI/scientific-agent-skills)** — 163 skills (the README says 165) plus 100+ scientific databases across biology, chemistry, medicine and drug discovery. Categorised by domain (27 bioinformatics, 10 cheminformatics, 22 data-analysis, …) rather than by type.

Two of its rules independently reach the same conclusions FI did:

- **"Every skill that ships `scripts/` must have a test suite under `tests/<skill-name>/`"**, and CI blocks a pull request that adds tooling without one. That is FI's self-test promotion requirement, arrived at separately.
- Each skill's `SKILL.md` **declares appropriate and inappropriate uses** — clinical skills state "research only" and explicitly disclaim diagnosis or treatment recommendation. That is FI's "When NOT to use", which FI's prompts treat as the load-bearing section.

It also enforces a structural contract FI does not have: frontmatter conformance, `SKILL.md` length, local links resolving, scripts parsing, no shipped bytecode, no hardcoded local paths, `--help` behaviour.

**[Orchestra-Research/AI-Research-SKILLs](https://github.com/orchestra-research/AI-research-SKILLs)** — 98 skills across 23 categories (the README says 87; the repository holds 98 `SKILL.md` files — counted in [audit 14](14_skill_library_shortlist.md)), describing a two-loop research architecture (inner optimisation loop, outer synthesis loop) over literature survey → ideation → experiments → synthesis → writing.

Worth noting what it lacks: **no dedicated skills for experiment design, numerical verification, or result reproducibility.** It covers MLOps tracking (W&B, MLflow, TensorBoard) but not validation. Its closest analogue is an "ARA Rigor Reviewer" doing semantic epistemic review — which is a language model reading text, the very thing the verification literature says is insufficient.

So the large libraries are strong where FI is weak (breadth of domain knowledge) and weak exactly where FI has been investing (verification). That is a good trade rather than a redundancy.

> **Half of this is wrong, corrected by [audit 14](14_skill_library_shortlist.md).** It holds for Orchestra. It does not hold for K-Dense: reading all 163 of its skills found **37 shipping deterministic `validate_` / `check_` / `audit_` / `lint_` CLIs**, mostly standard-library-only and exit-code-driven. Verification is that library's house style, not an absence in it. The claim above was generalised from reading one skill.

### 4. The verification gap

[**Autonomous Research Agents: A Survey of AI Scientists and the Verification Gap**](https://arxiv.org/abs/2608.05179) surveys 24 runnable systems across seven audit dimensions (lifecycle stage, autonomy level, evaluation method, released artifacts, human-in-the-loop points, novelty verification, result-selection disclosure).

| Artifact released | Share of systems |
|---|---|
| Code | 83% |
| Seeds or execution traces | 38% |
| Any novelty-verification method | 38% |

Its central claim:

> "no LLM-era system in the corpus demonstrates an externally validated in-loop oracle"

Of nine closed-loop highest-autonomy systems, seven are "mechanical reruns" and one relies on author claims with no external validation.

The framing has shifted: the bottleneck is no longer whether an agent can complete a research task, but **whether a reviewer can verify the claims it produces**. A manuscript is not a discovery — the claim may rest on a weak baseline, an unreproducible run, a hallucinated citation, or a result selected post hoc from many attempts.

**Named failure modes** in the literature, with FI's current position on each:

| Failure mode | FI's position |
|---|---|
| Implementation bugs | `execute_reflect` repair loop; plausibility gate catches wrong-but-plausible output |
| Hallucinated results | numeric oracle compares `paper.md` to `result_json` arithmetically |
| Shortcut reliance | methodologist persona's circular-evaluation must-flag |
| **Bug-as-insight reframing** | `write.md` forbids it explicitly — "when a result is broken, say it's broken" |
| Methodology fabrication | `implement_outline` requires a `source` field per physical constant |
| Frame-lock | ideate tournament / devil's-advocate persona, partially |
| Citation hallucination | `claim_check` + `write.md`'s forbidden-placeholder list |

A related framework cited in that literature, **ABE-Ralph**, proposes "pre-execution YAML contracts and automated Triple-Verification at numerical, logical, and code-structure levels". FI's `result_assertions` are a pre-execution contract and its numeric oracle is the numerical layer; the logical and code-structure layers have no FI analogue.

### 5. The security gap

K-Dense scans every skill with the Cisco AI Defense Skill Scanner for **prompt injection, data exfiltration and malicious code patterns**.

FI has no equivalent, and the recent work widened the exposure:

- `--import-skill` accepts any filesystem path
- entry-point discovery loads skills advertised by any installed package
- a skill's `SKILL.md` is injected verbatim into `design` and the `implement` prompts
- a skill's `selftest.py` is executed as a subprocess, with the skill directory on `PYTHONPATH`

The promotion gate answers "does this still work?" and "did a person approve it?". It does not answer "is this trying to do something to me". Human approval is the only barrier, and spotting an injection payload inside plausible prose is not a reliable human task.

---

## Recommendations

### 1. Conform to the Agent Skills standard `[impact: H][effort: M]`

Read and write `scripts/` and `references/` alongside FI's existing single-file layout, and keep `selftest.py` / `provenance.json` as FI extensions the spec already permits.

This is first because it changes the economics of everything else in the skills roadmap. The most expensive acquisition path — a person authoring each skill — is partly replaced by importing from the published libraries. It also makes FI skills portable to the 40+ other agents, which matters for the cross-project sharing already in place.

> Audit 14, having read all 261 published skills, sizes the actual harvest: **6 worth importing, 47 worth evaluating, 110 to skip.** The economics still favour conforming, but "import 250 skills" was never the shape of the win — most of that library is life sciences.

Doing this before the Web/VSCode approval surfaces is deliberate: building UI for a format about to change is wasted work.

### 2. Scan imported skills `[impact: H][effort: M]`

At minimum, at the single entry point where foreign content arrives (`--import-skill`) and at entry-point discovery: flag instruction-injection patterns in `SKILL.md`, and network or filesystem access in `selftest.py`, for a person to see before approving.

This is FI's own gap, widened by FI's own recent work, and it is the one item here where the risk is not hypothetical.

### 3. Do not adopt *LLM-judges-rigor* verification `[impact: H — by avoiding it][effort: 0]`

Orchestra's "ARA Rigor Reviewer" performs semantic epistemic review — a language model judging rigor. That is precisely the pattern the verification survey identifies as insufficient, and precisely what FI's numeric oracle exists to complement.

> **Revised by [`14_skill_library_shortlist.md`](14_skill_library_shortlist.md).** As first written this recommendation said "do not adopt the large libraries' verification approach", which reads as covering K-Dense too. It does not. K-Dense's `uncertainty-and-units` ships *deterministic*, exit-code-driven tooling — a dimensional-plausibility checker and a nine-rule static auditor for unit and uncertainty defects — which is the same architectural shape as FI's own oracle and covers ground FI's does not. The rule is about LLM-judged rigor, not about external verification tooling in general.

### 4. Position the oracle honestly `[impact: M][effort: L]`

The survey's "no externally validated in-loop oracle" is a real gap that FI's A2/A3 work partially addresses — *in-loop* and *non-LLM*, but internally rather than externally validated. The seeded-defect measurement already planned (roadmap item D3) is what would turn that from a design claim into evidence. Until it runs, the claim should be stated as a design property, not a demonstrated result.

---

## References

### Standard and libraries
- Agent Skills open standard — https://agentskills.io/
- K-Dense-AI/scientific-agent-skills — https://github.com/K-Dense-AI/scientific-agent-skills
- Orchestra-Research/AI-Research-SKILLs — https://github.com/orchestra-research/AI-research-SKILLs
- anthropics/skills — https://github.com/anthropics/skills

### Papers
- Autonomous Research Agents: A Survey of AI Scientists and the Verification Gap — https://arxiv.org/abs/2608.05179
- Beyond Execution: Auditing Experimental Fidelity in LLM-Driven Scientific Research — https://arxiv.org/html/2608.26753
- From Fluent to Verifiable: Claim-Level Auditability for Deep Research Agents — https://arxiv.org/html/2602.13855v1
- Position: Correct Answer, Wrong Mechanism — When AI Scientists Defend General Claims Their Own Data Contradicts — https://arxiv.org/html/2606.23175
- Can AI Validate Science? Benchmarking LLMs for Scientific Claim → Evidence Reasoning — https://arxiv.org/pdf/2506.08235

### FI source references
- `core/skills/selection.py` — catalogue and selection, the progressive-disclosure analogue
- `core/skills/base.py` — envelope, `Kind`, content-hash approval
- `core/skills/importer.py` — the foreign-skill entry point that needs scanning
- `core/numeric_oracle.py`, `core/plausibility.py` — the in-loop non-LLM checks
- `dev/skill-selection-spec.md` — the decisions behind selection
