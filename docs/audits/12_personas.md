# 12 — Personas Across Nodes and Research Domains

**Audit date:** 2026-05-15
**Scope:** What persona framing (if any) is used at each FI engine node and companion command today, what peer systems do instead, and which persona changes are worth the prompt churn.
**Companion audits:** 09 (AI Scientist landscape), 10 (Agent framework patterns), 11 (Workflow / IO comparison).

---

## Executive summary

FI uses persona framing in exactly **one** engine-node place — the four-member `review_panel` (`methodologist`, `statistician`, `devil_advocate`, `reproducibility`). Engine nodes use functional framing ("You are the Ideation stage…"); all **four** of the four prose-output companion commands (`proposal`, `critique`, `portfolio`, `digest`) already declare a named senior-role persona. (`summarize`, the fifth companion command, lacks one.)

The 2024–2026 empirical literature converges on one result that pins the recommendations: **persona prompting is a "double-edged sword" — helps alignment-shaped tasks (writing, role-play, safety), hurts pretraining-shaped tasks (math, factual recall, coding).** Wharton's Prompting Science Report 4 (Mollick et al., Dec 2025) and a USC ACL 2024 long paper both report no-or-negative effect of expert-persona prompts on MMLU-style benchmarks; the same persona prompts measurably help writing tone, safety adherence, and review-panel verdict diversity.

Recommendation in one sentence: **add named personas to the prose-output surfaces that lack them today (the `write` engine node, the `summarize` companion command, and the `poster` / `slides` / `speech` output-generator prompts), leave the JSON-extraction engine nodes (`clarify`, `ideate`, `analyze`, `cross_check`, `data_load`) functional, and add ONE domain-conditional override keyed on existing clarify slots so auto-mode quests stop writing biomed papers in an ML voice.**

---

## Findings

### Part A — Internal audit: current persona framing per FI node

The table below was produced by reading each prompt in `agents/*.md`. The "Persona explicit?" column means the prompt opens with an identifiable named role beyond the generic "you are the X stage." The "Output kind" column is the empirical-vs-rhetorical split that, per the 2024–2026 literature, governs whether a persona will help or hurt.

| Subsystem           | Prompt / surface       | Persona explicit? | Output kind         | Implicit voice                 |
| ------------------- | ---------------------- | ----------------- | ------------------- | ------------------------------ |
| **engine node**     | `clarify.md`           | No                | JSON extraction     | Asks-not-tells                 |
| **engine node**     | `ideate.md`            | No                | JSON enumeration    | Brainstormer                   |
| **engine node**     | `ideate_reflect.md`    | No                | JSON judgement      | Self-critical reader           |
| **engine node**     | `ideate_tournament.md` (PR #77) | No       | JSON judgement      | A-or-B referee                 |
| **engine node**     | literature (retrieval) | n/a               | n/a                 | n/a                            |
| **engine node**     | `design.md`            | No                | JSON spec           | Experimentalist                |
| **engine node**     | `implement.md`         | No                | Python code         | Engineer                       |
| **engine node**     | `execute_reflect.md`   | No                | Python patch        | Debugger                       |
| **engine node**     | `analyze.md`           | No                | JSON judgement      | Senior reviewer                |
| **engine node**     | `cross_check.md`       | No                | JSON classification | Librarian                      |
| **engine node**     | `write.md`             | No                | Markdown prose      | Author                         |
| **engine node**     | `review.md`            | Half (trait only) | JSON judgement      | Demanding peer reviewer        |
| **engine node**     | `review_persona_*.md`  | YES — 4 named     | JSON judgement      | (Per persona, see below)       |
| **engine node**     | `review_moderate.md`   | Half              | JSON aggregation    | Chair                          |
| **engine node**     | `auto_collect_data`    | n/a               | n/a                 | n/a                            |
| **engine node**     | `wait_for_data`        | n/a               | n/a                 | n/a                            |
| **engine node**     | `data_load.md`         | No                | JSON synthesis      | Archivist                      |
| **companion CLI**   | `proposal.md`          | YES               | Markdown plan       | Senior researcher              |
| **companion CLI**   | `critique.md`          | YES               | Markdown critique   | Adversarial reviewer           |
| **companion CLI**   | `portfolio.md`         | YES               | Markdown synthesis  | Lab director                   |
| **companion CLI**   | `digest.md`            | YES               | Markdown digest     | Project manager                |
| **companion CLI**   | `summarize.md`         | No                | Markdown synthesis  | Cataloguer                     |
| **output generator**| `poster.md`            | No                | LaTeX columns       | Designer                       |
| **output generator**| `slides.md`            | No                | Marp markdown       | Lecturer                       |
| **output generator**| `speech.md`            | No                | Spoken-word prose   | Talk-giver                     |

Three patterns jump out:

1. **The review-panel personas earn their keep.** Each of the four prefixes instructs the model to default `revise` on a different failure mode (design-level flaws, missing uncertainties, alternative explanations, missing repro info). `_aggregate_panel_reviews` (`core/engine.py:~2112`) takes median score and union of weaknesses — only useful because the verdicts genuinely disagree across personas.
2. **Companion CLI commands skew rhetorical and mostly already use personas.** `proposal`, `critique`, `portfolio`, `digest` all emit markdown prose and **all four** declare a named senior-role persona — pattern is internally consistent. The fifth companion (`summarize`) is the outlier that doesn't.
3. **Engine nodes skew JSON-extraction and lean functional.** `clarify`, `ideate`, `analyze`, `cross_check`, `data_load` emit fixed-schema JSON. The persona on these would be window dressing — the schema already pins the output. Per the Wharton finding, an expert persona on MMLU-shaped tasks tends to *hurt*.

The outliers across all three subsystems — prose surfaces lacking a named persona: `write.md` (engine node), `summarize.md` (companion CLI), `poster.md` / `slides.md` / `speech.md` (output generators). All five emit free-form prose or layout markup. `write.md` already does extensive constraint-loading (honesty section, topic-shape recognition, study-depth) — a persona prefix sits alongside, not in conflict. High-impact, low-effort win (see Rec 1). Note that any prompt edits to `summarize.md` change the folder-summarizer subsystem, and edits to `poster`/`slides`/`speech` change the `generation/` output-generator subsystem — implementers should edit those modules rather than `core/engine.py`.

### Part B — Competitive research: how peers handle personas

I read the documentation, README, and where available the paper / source for each peer. Persona granularity is the most useful axis to compare on — does the system commit to one persona per stage, or share a persona across stages, or condition on domain?

#### Sakana AI Scientist v2 (April 2025, arXiv:2504.08066)

Persona granularity is **stage-named-functional, not expert-named**, with one exception. Ideation, Experiment Manager, and Manuscript Writer use stage-name prompts ("model_writeup", "model_citation", "model_review" CLI flags rather than a persona string). The only explicit expert persona in v2 is the **Automated Reviewer / Area Chair** — prompted to act as a NeurIPS Area Chair, ensembling five reviews into one decision using official NeurIPS guidelines. The Area Chair achieved 69% balanced accuracy and F1 exceeding inter-human agreement (NeurIPS 2021 consistency experiment). Sakana does **not** adapt personas by domain — the system is locked to ML-on-ML so domain priors live in templates, not in a swappable persona.

#### Google AI co-scientist (Feb 2025, arXiv:2502.18864)

Richest persona schema in the peer landscape. Six specialized agents named for their *role in the scientific method*, not for a discipline:

- **Generation** — explores literature and runs *simulated scientific debates among expert personas* to seed hypotheses.
- **Reflection** — peer-reviewer-style assessment (plausible, novel, testable). Analog to FI's `review`.
- **Ranking** — Elo tournament across hypotheses; closest analog to the pairwise-tournament pattern FI added in PR #77's `ideate_tournament` node, but Co-Scientist uses continuous Elo while FI uses simple win-count plus decisive-margin tiebreak.
- **Evolution** — iteratively improves top-ranked hypotheses. No FI analog.
- **Proximity** — clusters hypotheses by similarity.
- **Meta-review** — synthesizes panel feedback. Analog to FI's `review_moderate`.

Critically, **Co-Scientist personas are domain-conditional in one specific way**: the Generation agent's self-play debate spawns expert personas drawn from the research goal's domain (e.g. "you are an oncologist", "you are a pharmacologist" for the AML drug-repurposing study). The six top-level personas are domain-agnostic; the *sub-personas inside Generation* are domain-conditional. This is the most interesting peer design pattern — and the one Rec 7 explicitly declines to copy. No public ablation isolates the persona effect.

#### Microsoft AutoGen Magentic-One (Nov 2024, arXiv:2411.04468)

**Role-named, not expert-named**. Five agents — `Orchestrator`, `WebSurfer`, `FileSurfer`, `Coder`, `ComputerTerminal` — each with a custom system prompt describing capabilities. The paper notes "identical configuration across all three benchmarks; only additional set up code and unique final prompts for benchmark-specific formatting." So Magentic-One **does not adapt personas by task domain** and doesn't use expert-role personas — `Coder` is "specialized through its system prompt for writing code," not "you are a senior Python engineer." This mirrors FI's current pattern (engine node = capability handle) and is the cleanest evidence that functional naming alone gets you a long way.

#### CrewAI (active 2024–2026)

**Most opinionated persona schema** of any peer: `Agent(role=, goal=, backstory=, tools=[...])` is first-class. The "Crafting Effective Agents" docs advocate `role = "senior financial analyst with 15 years at Goldman Sachs"`-style framings; `backstory` is pitched as a "prompt-engineering shortcut that scales." JetThoughts 2025 benchmark: CrewAI runs QA 5.76x faster than LangGraph with higher eval scores, but on complex multi-step reasoning LangGraph hits 62% vs CrewAI's 54%. This is exactly the Wharton/USC pattern: persona helps rhetorical/interactive, hurts deep reasoning. CrewAI's schema is **net positive on rhetoric, net negative on deep reasoning** — the trade-off FI should respect.

#### The persona-prompting empirical literature (2024–2026)

The literature is now thick enough to draw firm conclusions:

- **Wharton Prompting Science Report 4** (Mollick, Meincke, et al., Dec 2025): assigning an expert persona ("you are a physics expert") matched to the problem type had no significant impact on factual-accuracy performance, except for one specific model (Gemini 2.0 Flash). Headline finding: *expert personas don't improve factual accuracy.*
- **USC ACL 2024 long paper** ("Quantifying the Persona Effect"): persona prompting improves alignment-dependent tasks (writing, role-play, safety) but degrades performance on pretraining-dependent tasks (math, coding, MMLU). Persona is a "double-edged sword."
- **PRISM paper** (arXiv:2603.18507): expert personas help in 5/8 BIG-bench categories (Writing, Roleplay, Reasoning, Extraction, STEM) — strongest gains in Extraction (+0.65) and STEM (+0.60) — but on MMLU all expert variants damage accuracy.
- **Synthetic-persona forecasting** (arXiv:2511.02458): a controlled ablation with >2000 synthetic personas shows no measurable forecasting advantage from any persona description.

The integrated takeaway: persona framing is a *stylistic* lever, not a knowledge lever. Stages whose output is judged on style benefit; stages judged on factual accuracy don't. This maps directly onto FI's prompt portfolio.

---

## Recommendations

Ranked by impact-per-effort. Each recommendation cites the empirical justification.

### 1. Add named personas to the five prose-output prompts that lack them (HIGH impact, LOW effort)

Five prose-output prompts currently have no named persona:
`write.md` (engine node), `summarize.md` (companion CLI), and
`poster.md` / `slides.md` / `speech.md` (output generators). The
other four prose-output companions (`proposal.md`, `critique.md`,
`portfolio.md`, `digest.md`) already declare senior-role personas
and serve as parity references. The Wharton/USC result says
rhetorical tasks gain measurably from persona. Suggested defaults:

- `write.md` → "You are a **senior researcher** writing an IMRAD paper for publication."
- `summarize.md` → "You are a **research librarian** cataloguing a folder of mixed content."
- `poster.md` → "You are a **conference-poster designer** with 10 years of academic poster experience."
- `slides.md` → "You are a **research scientist preparing a 10-minute conference talk**."
- `speech.md` → "You are a **scientist writing the spoken script** for a conference talk."

Edit cost: ~5 single-line edits to existing prompts. No engine changes.

### 2. Build a small reusable persona library and reference it by name (MEDIUM impact, MEDIUM effort)

Today's duplication: `proposal.md`, `portfolio.md` both inline a senior-role sentence. As Rec 1 spreads, this drifts. Fix is the CrewAI pattern, scoped down: a single `agents/personas.md` declaring six named entries — `senior_researcher`, `experimentalist`, `theorist`, `librarian`, `editor`, `adversarial_reviewer` — each a 2–3 sentence prefix. Engine prompts declare their persona by name and `core/engine.py` resolves the prefix the same way it already resolves `review_persona_*.md` (`_load_persona_prefix`, line ~2093). Six entries cover ~90% of FI's prompt surface. Edit cost: new `personas.md` (~50 lines), small loader, prompt edits.

### 3. One domain-conditional persona override, driven by existing clarify slots (HIGH impact, LOW effort)

The clarify node already collects two slots that uniquely identify the appropriate persona:

- `empirical_vs_theoretical` ∈ {empirical, theoretical, mixed}
- `paper_venue` ∈ {generic, neurips, iclr, ieee_access, nature_mi}

Plus the new `simulatability` slot determines no-simulation routing.

Concrete swap table (acts on `write` only — keep `design` functional per Rec 5):

| Slot combination                                    | `write` persona       |
| --------------------------------------------------- | --------------------- |
| empirical + neurips/iclr                            | senior_researcher (ML-aware variant) |
| empirical + ieee_access                             | senior_researcher (engineering-aware) |
| empirical + nature_mi                               | senior_researcher (physical-science-aware) |
| theoretical + any                                   | senior_researcher (math-aware) |
| simulatability=no (humanities, social, archival)    | senior_researcher (essay voice) |

If PR #1 lands an `essay`/`policy_brief` format Literal on `OutputConfig.paper_format`, the override table widens to gate on `state["clarify_answers"]["paper_venue"]` (and on the future format slot if one is added to the clarify questionnaire — today `paper_venue` is the only writing-style hint clarify collects). Add the override table to `_load_persona_prefix`-style resolution and read from `state["clarify_answers"]` (NOT `state["clarify"]` — the answers live under the `_answers` suffix).

The reason this is high-impact-low-effort: the slots are *already collected*, so this is a routing decision, not a new prompt. The two regimes that today produce the most-jarring style mismatches — humanities quests written in ML-paper voice, theoretical quests written as if they ran an experiment — both go away.

### 4. Expose `engine.personas` as zero-config opt-in YAML (LOW impact, LOW effort)

Add `engine.personas: dict[str, str]` to `EngineConfig` (`core/config.py`). Resolution order, lowest-to-highest: (1) current functional default, (2) Rec 1 rhetorical default, (3) Rec 3 domain override, (4) YAML override (`engine.personas: {write: theorist}`). Zero-config users get Recs 1+3 automatically; power users pin per-node via YAML.

### 5. Do NOT add personas to the seven JSON-extraction nodes (DEFENSIVE)

`clarify`, `ideate`, `ideate_reflect`, `ideate_tournament`, `analyze`, `cross_check`, `data_load` all emit a single JSON object with a fixed schema. Per the Wharton MMLU result and the USC ACL 2024 paper, adding an expert persona to schema-shaped tasks tends to be neutral-or-harmful — the model burns attention on the role rather than the extraction. Keep these stages functional ("You are the **Ideation** stage…"). The current prompts already do this, and the recommendation is to *not change them* despite the persona-everywhere temptation.

### 6. Add a `domain_expert` review panelist for no-simulation quests (MEDIUM impact, LOW effort)

Today's four panelists are domain-agnostic. For social/cultural quests (`simulatability=no`), `statistician` is the least-useful seat — formal statistics are usually inappropriate for small-N or qualitative evidence. Add a fifth persona, `domain_expert` ("You are a senior researcher in the topic's field; critique terminology, canonical sources, alignment with what the field already knows"), and have `_node_review` swap `statistician → domain_expert` when `simulatability=no`. Edit cost: one new `review_persona_domain_expert.md`, ~20 LOC of routing. Backward-compatible.

### 7. Do NOT inline Co-Scientist's "self-play debate" inside `ideate` (DEFENSIVE)

Co-Scientist's Generation agent spawns expert-persona sub-debates per topic — the flashiest peer pattern. **Decline** because: (a) `ideate` already emits 3–5 ideas, `ideate_tournament` adds C(N,2) comparisons — debate sub-personas multiply the call count again with no peer-confirmed quality lift outside biomed; (b) the empirical literature shows no general domain-sub-persona win on ideation; (c) Rec 6's `domain_expert` panelist captures the same expertise at `review` where it actually steers `revise`, at a fraction of the cost. Re-evaluate if Axon ever ships a biomed corpus large enough to seed credible sub-personas AND a benchmark justifies the cost.

---

## References

1. Mollick, E. et al. — *Prompting Science Report 4: Playing Pretend: Expert Personas Don't Improve Factual Accuracy*, SSRN (Dec 2025). https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5879722
2. Chen, J., Wang, T. et al. — *Quantifying the Persona Effect in LLM Simulations*, ACL 2024 long paper. https://aclanthology.org/2024.acl-long.554.pdf
3. *Expert Personas Improve LLM Alignment but Damage Accuracy: Bootstrapping Intent-Based Persona Routing with PRISM*, arXiv:2603.18507. https://arxiv.org/html/2603.18507v1
4. *The Prompt Makes the Person(a): A Systematic Evaluation of Sociodemographic Persona Prompting*, arXiv:2507.16076. https://arxiv.org/abs/2507.16076
5. *Prompting for Policy: Forecasting Macroeconomic Scenarios with Synthetic LLM Personas*, arXiv:2511.02458. https://arxiv.org/html/2511.02458
6. Yuan, C. et al. — *The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search*, arXiv:2504.08066 (April 2025). https://arxiv.org/abs/2504.08066
7. Sakana AI — *AI-Scientist-v2* (GitHub repository). https://github.com/SakanaAI/AI-Scientist-v2
8. Gottweis, J. et al. — *Towards an AI Co-Scientist*, arXiv:2502.18864 (Feb 2025). https://arxiv.org/abs/2502.18864
9. Google Research — *Accelerating scientific breakthroughs with an AI co-scientist* (blog). https://research.google/blog/accelerating-scientific-breakthroughs-with-an-ai-co-scientist/
10. Fourney, A. et al. — *Magentic-One: A Generalist Multi-Agent System for Solving Complex Tasks*, arXiv:2411.04468 (Nov 2024). https://arxiv.org/html/2411.04468v1
11. Microsoft AutoGen documentation — *Magentic-One*. https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/magentic-one.html
12. CrewAI — *Crafting Effective Agents*. https://docs.crewai.com/en/guides/agents/crafting-effective-agents
13. Search Engine Journal — *Research Shows Where Persona Prompting Works And When It Backfires* (Dec 2025). https://www.searchenginejournal.com/research-you-are-an-expert-prompts-can-damage-factual-accuracy/570397/
14. The Register — *Telling an AI model that it's an expert makes it worse* (March 2026). https://www.theregister.com/2026/03/24/ai_models_persona_prompting/
15. JetThoughts benchmark (2025) — referenced via AgentRank CrewAI review. https://www.agentrank.tech/blog/crewai-review-multi-agent-framework-2026
16. FI audit 09 — *AI Scientist Competitive Landscape*. `docs/audits/09_ai_scientist_landscape.md`
17. FI audit 10 — *Agent Framework Patterns*. `docs/audits/10_agent_framework_patterns.md`
18. FI source — `agents/review_persona_methodologist.md`, `agents/review_persona_statistician.md`, `agents/review_persona_devil_advocate.md`, `agents/review_persona_reproducibility.md`, `agents/review_persona_generic.md`.
19. FI source — `core/engine.py:_load_persona_prefix` (line ~2093), `_aggregate_panel_reviews` (line ~2112), `EngineConfig.review_panel` (`core/config.py:128–141`).
