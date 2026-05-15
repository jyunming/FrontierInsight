# 09 — AI Scientist Competitive Landscape

**Audit date:** 2026-05-15
**Scope:** Where does FrontierInsight sit among end-to-end LLM research agents in May 2026, and what is actually worth borrowing?
**Author:** parallel-worker 09

---

## Executive summary (read this first)

FrontierInsight (FI) is one of a dozen-plus systems that, in mid-2026, attempt to drive a research project end-to-end with LLM agents. The interesting thing is **not** that FI exists — it is that FI's design decisions are *quantitatively different* from its peers on three axes that matter:

1. **LLM-call budget** — FI spends **8–18 calls per quest** ([`docs/USAGE.md:28`](../../docs/USAGE.md)). Sakana's *AI Scientist v2* spends an estimated **40–45** per paper, and Google's *PaperOrchestra* (April 2026) spends **60–70**. Agent Laboratory (Schmidgall & Moor, EMNLP 2025) is closest to FI at roughly **20–30** per paper. **FI is 3–8x cheaper per quest** than the frontier in token spend, but is also doing less ambitious work (no agentic tree search, no multi-reviewer Elo tournament).
2. **Domain breadth** — Sakana is locked to ML-on-ML research, Google AI Co-Scientist is locked to biomed hypothesis generation, FutureHouse's Phoenix/Crow/Falcon is locked to chemistry+literature. FI is the only one designed for **arbitrary `topic: <string>` + Python kernel**, which is both its market position and its weakness (no domain priors).
3. **Output format breadth** — Most peers produce a single LaTeX→PDF artifact. FI produces markdown paper + figures + reproducible code bundle + slides + speech script, with Axon serving retrieved literature back into future quests ([`docs/capabilities.md:50–64`](../../docs/capabilities.md)). This is genuinely differentiated.

**The three things FI should adopt from peers, in priority order:**
1. **Sakana's agentic tree-search experiment loop** — but bounded. FI's current `execute_reflect` is a flat retry counter (≤3); a *small* tree (3 wide × 2 deep) would catch failure modes a linear retry never explores. **High impact, medium effort.**
2. **Google Co-Scientist's tournament + Elo ranking** for `ideate` — FI currently picks one of 3 ideas via a single LLM critique call ([`core/engine.py:_node_ideate`](../../core/engine.py)). A 3-way pairwise tournament would add one extra call but produces a measurably better-ranked chosen_idea. **Medium impact, low effort.**
3. **Agent Laboratory / AgentRxiv shared preprint server** — partially already implemented. `Engine._write_back_knowledge` (core/engine.py:1713) calls `Knowledge.add_quest_artifacts` when `knowledge.write_back_quests` is enabled and the review verdict is `accept`, indexing five document kinds: `fi_paper_spine`, `fi_quest_paper`, `fi_quest_summary`, `fi_topic_event`, and `fi_external_ref_spine` (one per consumed external reference). Analyzer key-findings are folded INTO the `fi_paper_spine` and `fi_quest_summary` content and metadata — they are not a separate kind. The remaining delta vs. AgentRxiv is mainly *what* gets indexed (chosen_idea + cross_check classifications are not currently stored) and whether the default should be ON. **Medium impact, low effort.**

The five things FI should **NOT** adopt: Aider-based file editing (Sakana's source of half its failure modes), Sakana's 40+ LLM-call writeup pipeline (FI's 1–2 write calls produce comparable markdown), PaperOrchestra's PaperBanana VLM figure refinement (over-engineered for FI's domain breadth), Google's drug-repurposing-specific tooling (not generalizable), and the Sakana "Aider edits its own runtime" architecture (security disaster).

---

## Findings

For each competitor below we report the 8 dimensions specified in the audit recipe. Sources are footnoted; see the References section.

### Competitor 1 — Sakana AI Scientist v2 (April 2025, ICLR 2025 workshop)

Sakana's AI Scientist made headlines in 2024 with v1 (Nature publication of the system itself, August 2024); v2 (arXiv:2504.08066, April 2025) is the relevant comparison for FI today. The v2 paper is notable because it produced the **first fully-AI-generated paper accepted at a peer-reviewed ML workshop** (ICLR 2025 "I Can't Believe It's Not Better"), with reviewer scores 6/7/6 (top 45% of submissions). Sakana voluntarily withdrew the paper post-acceptance citing missing ethical norms.

1. **Workflow stages** — `perform_ideation_temp_free.py` → `launch_scientist_bfts.py` (agentic tree search managed by an *experiment manager agent*) → analysis & tree visualization → writeup → automated review. The key v2 innovation over v1 is removing v1's reliance on human-authored code templates: v2 grows code via **best-first tree search (BFTS)** with a configurable `max_debug_depth`, `debug_prob`, `num_workers`, `steps`. Each tree node is a candidate experiment; failing nodes spawn debug children. **No clarify step**, no separate `auto_collect_data` analog — Sakana assumes ML-on-ML, so data is whatever PyTorch can synthesize.
2. **LLM calls per paper** — Not published explicitly, but PaperOrchestra (which compared itself directly to v2) reports **~40–45 calls per paper** for AI Scientist v2. Includes ideation reflections (3–5), tree-search expansions (15–25 across debug nodes), writeup (5–8), citation insertion (3–5 via separate GPT-4o pass), automated review (5 independent reviews ensemble).
3. **Output formats** — LaTeX → PDF only. A separate "review" markdown is produced by the automated reviewer. No slides, no reproducible bundle (just the working directory).
4. **Retry / iteration** — `max_debug_depth` (default ~3) bounds how deep BFTS goes per failing experiment. `debug_prob` controls whether to debug vs. branch sideways. Effectively a `~3 deep × N wide` tree with VLM-driven figure refinement. **No multi-reviewer panel** — single automated reviewer prompted as Area Chair, ensembling 5 reviews from itself.
5. **Domains** — Machine learning research on machine learning topics (transformers, diffusion, RL). Critics note 5/12 attempted experiments fail at the code stage (42% code-error rate per the SIGIR Forum evaluation). External evaluations report hallucinated results and fake numbers.
6. **Cost (USD/paper)** — **~$15–$20 per run** using Claude 3.5 Sonnet for experimentation + ~$5 for writeup ≈ **~$20–$25 total**. DeepSeek Coder V2 brings it under $10.
7. **Code availability** — Open source under the "AI Scientist Source Code License" (a derivative of the Responsible AI License — *not* OSI-approved, has use-case restrictions). Repo: github.com/SakanaAI/AI-Scientist-v2.
8. **Architecture pattern** — Agentic tree search (BFTS) + dedicated experiment-manager agent + VLM feedback loop for figure refinement. Closest analog in FI is the `execute_reflect` loop, but FI is *linear retry* whereas Sakana is *branching*.

**FI gap vs. Sakana:** Sakana's BFTS catches "the bug is in a region linear retry won't find" cases. FI's `execute_reflect` does 3 sequential attempts; if attempt 1's diagnosis is wrong, attempts 2–3 inherit the bad theory.

**FI advantage over Sakana:** Domain agnosticism (Sakana hardwires the ML domain via template seeding) + reproducibility bundle + 5x lower cost.

### Competitor 2 — Google AI Co-Scientist (Feb 2025, arXiv:2502.18864)

Google DeepMind's AI Co-Scientist is the most architecturally interesting peer. It is **not** a paper-writer — it is a hypothesis-generation system for biomedical research, validated on three concrete domains: drug repurposing (acute myeloid leukemia), novel treatment-target discovery, and antimicrobial resistance mechanisms. The system was peer-reviewed alongside wet-lab validation of its proposals.

1. **Workflow stages** — Six specialized agents in a coalition: **Generation** (literature exploration + simulated scientific debates to seed hypotheses), **Reflection** (peer-reviewer-style assessment), **Ranking** (Elo tournament across hypotheses), **Evolution** (improves top-ranked hypotheses, can combine or branch), **Proximity** (clusters by similarity for navigation), **Meta-review** (synthesizes feedback). No `implement`/`execute` stages — the system stops at hypothesis + proposal; humans run the wet lab.
2. **LLM calls per hypothesis** — Not published. The paper reports the system uses "asynchronous task execution for flexible compute scaling" — i.e., it deliberately scales test-time compute, with Elo improving monotonically as more compute is spent. For a 203-research-goal evaluation, total compute is undisclosed.
3. **Output formats** — Research proposals (text), ranked by Elo with provenance. No paper. No code. Outputs are human-consumed.
4. **Retry / iteration** — Elo-driven tournament: pairwise multi-turn scientific debates for top-ranked hypotheses, single-turn pairwise for lower-ranked. Evolution agent iteratively improves top entries. The system *does not converge*; it runs until compute budget is exhausted, with later hypotheses preferred only if they win Elo matches against earlier ones.
5. **Domains** — Biomed specifically. Tested on drug repurposing, target discovery, AMR. Internal Google tests on math and physics goals, but biomed is the canonical use case.
6. **Cost (USD)** — Not disclosed. Gemini 2.0 internal pricing is below public Gemini Pro; estimates from third parties put end-to-end runs at $50–$200+ per research goal given the test-time-compute scaling story.
7. **Code availability** — **Closed.** Google offered a "trusted tester program." Third-party Swarms re-implementation exists at github.com/The-Swarm-Corporation/AI-CoScientist (untested at scale).
8. **Architecture pattern** — Multi-agent coalition (6 specialized agents) + Elo tournament + generate/debate/evolve cycle. The agent topology is a *pool*, not a DAG — agents are invoked asynchronously by a scheduler.

**FI gap vs. Co-Scientist:** FI has no tournament. FI's `ideate` proposes 3 candidates, runs a single critique LLM call to pick one. Co-Scientist would do `C(3,2)=3` pairwise debates with Elo updates, then expand the winner. This is a small change with large impact on idea quality.

**FI advantage over Co-Scientist:** End-to-end — FI runs experiments and writes papers; Co-Scientist stops at the hypothesis. Also, FI ships as open source.

### Competitor 3 — Agent Laboratory + AgentRxiv (Schmidgall & Moor, Jan/March 2025)

Agent Laboratory (arXiv:2501.04227) and its follow-up AgentRxiv (arXiv:2503.18102) form the closest *architectural* peer to FI. Agent Laboratory uses 5 named LLM-driven roles (PhD agent for literature, Postdoc for planning, ML Engineer + SW Engineer for code, Professor for synthesis) and produces a LaTeX paper. AgentRxiv adds a shared preprint server where multiple Agent Laboratory instances *upload finished papers and retrieve each other's work*, demonstrating that this raises MATH-500 accuracy by 11–14%.

1. **Workflow stages** — Three phases (Literature Review → Experimentation → Report Writing) executed by 5 named agents. Tools: `arxiv` API for literature, `mle-solver` for ML code (with iterative scoring + self-reflection + auto-repair), `paper-solver` for LaTeX. Co-pilot mode lets humans check in at phase boundaries; autonomous mode runs the lot. **Closest match to FI's pipeline** of any peer.
2. **LLM calls per paper** — Not stated as a single number, but the cost-per-paper data implies it. With gpt-4o-mini at $3.11/paper at typical 2025 pricing (input $0.15/M, output $0.60/M, average mix), **~20–30 LLM calls per paper** is the most plausible bracket. With o1-mini the cost rises to $7.51/paper (more reasoning tokens, not more calls).
3. **Output formats** — LaTeX (with `--compile-latex` to PDF via pdflatex) + code repository. No slides, no markdown, no reproducible bundle metadata.
4. **Retry / iteration** — `mle-solver` self-reflects and auto-repairs code; retry limits not explicit in the paper. The `paper-solver` similarly iterates on writeups but without published bounds.
5. **Domains** — Generic ML research. Tested on ML benchmarks; less domain-locked than Sakana but no biomed or chemistry-specific tooling.
6. **Cost (USD/paper)** — **$3.11 with gpt-4o-mini, $7.51 with o1-mini.** Sakana measured 84% lower than v1 baseline.
7. **Code availability** — **MIT license.** github.com/SamuelSchmidgall/AgentLaboratory.
8. **Architecture pattern** — Specialized-agent pool (5 named personas) + tool-using subagents (mle-solver, paper-solver). Not a DAG — orchestration is procedural.

**FI status vs. Agent Laboratory:** AgentRxiv's shared-preprint idea is the closest analog to FI's existing `Engine._write_back_knowledge` + `Knowledge.add_quest_artifacts` path. The gap is narrower than first looks: FI already writes back 5 document kinds for accepted quests, but `chosen_idea` and per-finding `cross_check` classifications are not currently among them. Agent Laboratory's 11–14% empirical lift is the strongest evidence in 2026 that cross-run memory pays off — FI's task is enriching the existing write-back, not adding it.

**FI advantage over Agent Laboratory:** Reviewer panel (FI's multi-persona panel is more rigorous than Agent Laboratory's single Professor synthesis), reproducibility bundle, slides/speech outputs.

### Bonus competitor — PaperOrchestra (Google Research, April 2026)

Surfaced in research after the audit started — relevant because it is the **most recent** entry and benchmarks itself directly against Sakana.

- **Workflow** — 5-agent pipeline (Outline → parallel{Plotting via PaperBanana VLM, Literature via web search + Semantic Scholar verification} → Section Writing → Content Refinement via AgentReview scoring with monotonic-improvement constraint).
- **LLM calls per paper** — **60–70** (vs. Sakana v2's 40–45 by their measurement).
- **Cost / time** — Not published; runtime 39.6 min/paper.
- **Output** — LaTeX. PaperWritingBench (200 papers from CVPR 2025 + ICLR 2025) is the eval set.
- **Iteration** — `Content Refinement Agent` reverts immediately if score drops, enforcing monotonic quality. Refined drafts beat unrefined 79–81% of the time across 180 paired human comparisons.

**Worth borrowing:** The monotonic-improvement constraint. FI's `review` loop revises on a `verdict == "revise"` flag but doesn't compare the revised paper to the prior version. Adding a "if revised review < prior review, revert" gate would prevent regression-on-revise, a known failure of LLM iteration. Low effort, medium impact.

### Bonus competitor — FutureHouse (Crow / Falcon / Phoenix / Owl, Aviary)

FutureHouse's positioning: not end-to-end paper generation, but **domain-specialized literature and reasoning agents** trained with their `aviary` RL framework (arXiv:2412.21154). Crow does scientific Q&A with citations; Falcon does literature reviews; Phoenix does chemistry informatics + safety; Owl was added in 2025. Aviary's contribution: training open-source LLMs to surpass humans on two LAB-Bench tasks (literature search + DNA-construct reasoning) using Expert Iteration and majority voting.

This is **not** a direct competitor to FI's end-to-end loop — but their literature subagents would outperform FI's current `_node_literature` (which is a single synthesis LLM call over Axon retrievals). FutureHouse Crow's precision on cited literature is higher than PhD baselines per their MIT News piece.

**Worth borrowing:** Conceptually, the "specialist literature agent" idea — but the *implementation* is heavy (RL-trained model, expert iteration). FI should keep using Axon embedding retrieval + LLM synthesis but consider a *cited-answer verification pass* (one extra LLM call that checks each cited claim against the retrieved doc).

### Comparison table

| Dimension | FI (May 2026) | Sakana v2 | Google Co-Scientist | Agent Lab + AgentRxiv | PaperOrchestra |
|---|---|---|---|---|---|
| End-to-end paper? | Yes (md + pdf + code) | Yes (LaTeX/PDF) | No (hypothesis only) | Yes (LaTeX) | Yes (LaTeX) |
| LLM calls / paper | **8–18** | ~40–45 | undisclosed | ~20–30 | 60–70 |
| Cost (USD/paper) | ~$0.50–$3 (Copilot premium reqs) | $15–$25 | undisclosed (high) | $3.11 (gpt-4o-mini) | undisclosed |
| Clarify / scope step | Yes (interactive or auto) | No | No | No | No |
| Literature step | Axon retrieval + 1 synthesis call | Semantic Scholar + ideation reflections | Generation agent reads lit | PhD agent + arXiv API | Web search + Semantic Scholar verify |
| Auto-collect data | Yes (Axon → quest data dir) | No | N/A | No | No |
| Experiment loop | Linear retry (≤3) | BFTS tree search (~3 deep) | N/A | mle-solver auto-repair | Plotting VLM loop |
| Analysis / cross-check | `analyze` + `cross_check` per finding | Tree-evaluator scoring | Reflection agent | Professor synthesis | Section writing agent |
| Reviewer panel | 1 or N (configurable) | 1 reviewer ensembling 5 | Reflection + Meta-review | 1 (Professor) | AgentReview with score gate |
| Revision / iteration bound | `max_iterations: 2` | `max_debug_depth: 3` | Until compute exhausted | implicit | Score-gated, monotonic |
| Tournament / Elo | **No** | No | Yes (Elo on hypotheses) | No | No |
| Tree search | **No** | Yes (BFTS) | No | No | No |
| Multi-quest memory | Axon retrieval + write-back of accepted quests (`fi_quest_*` docs) | No | No | **Yes (AgentRxiv preprint server)** | No |
| Output: slides | Yes | No | No | No | No |
| Output: speech script | Yes | No | No | No | No |
| Output: reproducibility bundle | Yes (`bundle_manifest`) | No | N/A | No | No |
| Domains | Generic (any topic + Python kernel) | ML-on-ML | Biomedical | Generic ML | ML research |
| Code license | Open source (planned MIT per CONTRIBUTING) | Custom "Responsible AI" derivative | Closed | MIT | Unreleased |
| Architecture | LangGraph DAG + SQLite checkpoint | Tree-search + experiment manager | Multi-agent pool + scheduler | 5-persona procedural | 5-agent procedural |
| Notable strength | Output breadth + domain breadth + concurrency | First peer-reviewed AI paper | Wet-lab-validated hypotheses | Lowest cost-per-paper | Highest PaperWritingBench score |
| Notable weakness | No tree search, no Elo, no own-output memory | 42% code-error rate, ML-only | Hypothesis-only, closed | Single reviewer | Most expensive in calls |

---

## Recommendations

Each recommendation is tagged `[impact: H/M/L][effort: H/M/L]`. Sorted by impact-to-effort ratio.

### 1. Borrow Sakana's bounded tree-search experiment loop — modify `execute_reflect` in `core/engine.py` `[impact: H][effort: M]`

FI's `_node_execute_reflect` ([`core/engine.py:1207`](../../core/engine.py)) is a flat linear retry: attempt 1 fails → attempt 2 fails → attempt 3 fails → give up. Each attempt inherits the prior attempt's diagnosis. If diagnosis 1 is wrong (e.g., "the bug is in the integrator" when it's actually in the data loader), attempts 2 and 3 are wasted.

**Proposal:** Replace the linear counter with a 2-wide × 2-deep search. On first failure, generate *two* different diagnoses (one LLM call producing a dict of two repair theories), apply each to a separate code branch, execute both in parallel via the existing `ExecutionResult` plumbing. If either succeeds, take that branch; if both fail, do one more `(2,2)` step from the better of the two failures. Bound at 4 total executions per failed `execute` (vs. current 3 linear).

**Cost impact:** +1 to +3 LLM calls per `execute` failure case; pushes FI's worst-case ceiling from 18 to ~21 calls. Still half of Sakana.

**File pointer:** `core/engine.py:1207–1310` (`_node_execute_reflect`). Add a `engine.execute_tree_search: bool` config flag (default `false` to preserve the cheap path).

### 2. Add ideate tournament with pairwise Elo — modify `_node_ideate` `[impact: M][effort: L]`

FI's `_node_ideate` ([`core/engine.py:634`](../../core/engine.py)) generates 3 ideas and picks one via a single critique LLM call. Google Co-Scientist demonstrates this is leaving signal on the table: pairwise multi-turn debate produces measurably better rankings than single-shot rating.

**Proposal:** When `engine.ideate_tournament: true` (new flag, default `false`), after generating the 3 ideas, run `C(3,2)=3` pairwise comparison LLM calls. Each call sees two ideas and outputs winner + reasoning. Update simple Elo (start everyone at 1500, K=32). Pick the highest-Elo idea. Optionally, with `ideate_reflect: true` already enabled, fold the reflection into the tournament rather than running it as a separate step.

**Cost impact:** +3 LLM calls when enabled (one more than `ideate_reflect`'s +1). Worth it for high-stakes runs.

**File pointer:** `core/engine.py:634–697`. New `engine.ideate_tournament` flag in `core/config.py:Engine`.

### 3. Enrich existing Axon write-back with `chosen_idea` + `cross_check` documents `[impact: M][effort: L]`

The original framing of this recommendation was wrong — FI already ingests accepted quests via `Engine._write_back_knowledge` (core/engine.py:1713) and `Knowledge.add_quest_artifacts` (core/knowledge.py), gated on `knowledge.write_back_quests` and a passing `review` verdict. Five document kinds are indexed today: `fi_paper_spine`, `fi_quest_paper`, `fi_quest_summary`, `fi_topic_event`, and `fi_external_ref_spine` (one per consumed external reference; the analyzer key-findings are folded into the spine + summary content, not a separate kind).

**What's actually missing vs. AgentRxiv:** `chosen_idea` (the rationale for picking one of the brainstormed directions over the others) and the per-finding `cross_check` classifications (supports/conflicts/neutral). Both would meaningfully sharpen what a future `_node_literature` retrieval surfaces back about prior FI work — "we already tried this hypothesis and it didn't beat the baseline".

**Proposal:** Extend `Knowledge.add_quest_artifacts` to accept two additional payloads (`chosen_idea`, `cross_check_findings`) and emit two new document kinds (`fi_quest_idea`, `fi_quest_finding`) alongside the existing five. Update the caller in `_write_back_knowledge` to pass them.

**Beware:** The existing accept-only gate is correct policy — don't change it. AgentRxiv's paper acknowledged that letting failed runs into the corpus hurt; FI gets this right today.

### 4. Add monotonic-improvement gate on `review → revise → write` `[impact: M][effort: L]`

Borrowed from PaperOrchestra's `Content Refinement Agent`. FI's `review` loop currently fires `verdict == "revise"` → re-runs `design` (per [`docs/capabilities.md:35`](../../docs/capabilities.md)) but never compares the revised review against the prior review. If the revision makes things worse, FI still ships the revision.

**Proposal:** Store `prior_review_score` in `QuestState`. On the second `review` call, compare; if the revised review has a lower aggregate score (e.g., review JSON's `overall_quality`), keep the prior paper and write a "regression on revise" note to the quest log.

**Cost impact:** 0 extra LLM calls — just keep the previous artifact.

**File pointer:** `core/engine.py:_node_review` and the `revise → design` edge.

### 5. Add `novelty_check` literature subnode — modify `_node_literature` or insert post-`ideate` `[impact: M][effort: M]`

Both Sakana (Semantic Scholar query before designing) and Google Co-Scientist (Reflection agent's novelty check) explicitly verify that the chosen hypothesis hasn't already been done. FI does not. The current `_node_literature` does retrieval + 1 synthesis but doesn't ask "is the chosen idea novel relative to retrievals?"

**Proposal:** Insert a small node `_node_novelty_check` between `ideate` and `literature` (or fold into `literature`). One LLM call: "Given chosen idea X and these K retrieved abstracts, is X (a) genuinely novel, (b) incrementally novel, (c) substantially overlapping with prior work?" If (c), route back to `ideate` (consuming `max_iterations` budget).

**Cost impact:** +1 call per quest (always), +1 quest cycle in the (c) case (rare). Pushes typical to 9–19, ceiling to 19.

**File pointer:** new node before `core/engine.py:_node_literature`, edges adjusted in `_build_graph()`.

### 6. Add citation-verification pass on `paper_md` — borrow from FutureHouse Crow `[impact: M][effort: M]`

FutureHouse's research is that LLMs hallucinate citations at PhD-comparable rates *unless* every cited claim is verified against its source. FI currently retrieves docs into `state.literature` and the `write` node uses them — but there's no after-the-fact check that the generated paper's citations are grounded.

**Proposal:** New `_node_verify_citations` after `write`, before `review`. One LLM call: "For each [cite-N] tag in this paper, does the cited doc's abstract actually support the claim?" Mark unsupported claims for the reviewer to flag.

**Cost impact:** +1 call per paper. Falls into the existing 18-call ceiling.

**File pointer:** new node between `write` and `review`.

### 7. Make the review panel optionally adversarial-tournament-style `[impact: L][effort: M]`

Sakana's automated reviewer ensembles 5 *independent* reviews from one LLM; FI's review panel does N personas. The cheap win: when `review_panel.adversarial: true`, generate pairwise debates between personas (e.g., methodologist vs. devil_advocate) before the moderator synthesizes. This raises panel cost from N+1 calls to N + C(N,2)/2 + 1, but mirrors Google Co-Scientist's debate machinery for the reviewer side.

**Cost impact:** With N=3 personas, +3 debate calls → panel cost rises from 4 to 7. Significant; gate behind a flag.

**File pointer:** `core/engine.py:_node_review` panel branch.

### 8. **Reject** Sakana's Aider integration for code edits `[impact: H — by avoiding it][effort: 0]`

Sakana's biggest reliability problem is that Aider edits FI's own runtime code. There is documented "AI Scientist edited its launcher to extend the time limit" misbehavior. FI's current `_node_implement` writes a fresh `experiment.py` per attempt and never edits the engine — **keep it that way**. Don't borrow this.

### 9. **Reject** PaperOrchestra's PaperBanana VLM figure refinement `[impact: 0][effort: 0]`

Useful for benchmark-style ML papers with standard plot types, over-engineered for FI's domain breadth (physics, biology, software studies — wildly different figure conventions). FI's existing matplotlib-based plotting in the executed Python is sufficient. **Don't borrow.**

---

## Prioritized adoption roadmap

If FI commits to one feature per release:

- **v0.next:** Recommendation 3 (own-output Axon ingestion) — low effort, highest impact. Affects every future quest.
- **v0.next+1:** Recommendation 1 (bounded execute tree search) — Phase K2 in `docs/plan.md` is a natural slot.
- **v0.next+2:** Recommendation 2 (ideate tournament) + Recommendation 4 (review monotonicity gate) — both cheap, ship together.
- **v0.next+3:** Recommendation 5 (novelty check) + Recommendation 6 (citation verification) — these go together as a "literature integrity" milestone.

Total LLM-call ceiling impact after all six adoptions: FI's worst case rises from ~18 to ~25 — still well under Sakana (40+), Agent Laboratory (20-30, comparable), and PaperOrchestra (60-70).

---

## References

### Papers
- Yamada et al. (April 2025), *The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search*. arXiv:2504.08066. https://arxiv.org/abs/2504.08066
- Lu et al. (August 2024), *The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery* (Nature, 2025). https://sakana.ai/ai-scientist-nature/
- Gottweis, Weng, Daryin et al. (Feb 2025), *Towards an AI Co-Scientist*. arXiv:2502.18864. https://arxiv.org/abs/2502.18864
- Schmidgall & Moor (Jan 2025), *Agent Laboratory: Using LLM Agents as Research Assistants*. arXiv:2501.04227. https://arxiv.org/abs/2501.04227
- Schmidgall & Moor (March 2025), *AgentRxiv: Towards Collaborative Autonomous Research*. arXiv:2503.18102. https://arxiv.org/abs/2503.18102
- Narayan et al. (Dec 2024), *Aviary: training language agents on challenging scientific tasks*. arXiv:2412.21154. https://arxiv.org/abs/2412.21154
- *Evaluating Sakana's AI Scientist: Bold Claims, Mixed Results, and a Promising Future?* ACM SIGIR Forum (Sept 2025). arXiv:2502.14297.
- *PaperOrchestra: A Multi-Agent Framework for Automated AI Research Paper Writing.* MarkTechPost report, April 2026. https://www.marktechpost.com/2026/04/08/google-ai-research-introduces-paperorchestra-a-multi-agent-framework-for-automated-ai-research-paper-writing/

### Repositories
- github.com/SakanaAI/AI-Scientist-v2
- github.com/SamuelSchmidgall/AgentLaboratory
- github.com/Future-House/aviary
- github.com/The-Swarm-Corporation/AI-CoScientist (community Co-Scientist re-implementation)
- github.com/FoundationAgents/OpenManus

### Blog posts & analyses
- Sakana AI Scientist Nature publication announcement. https://sakana.ai/ai-scientist-nature/
- Sakana AI Scientist first peer-reviewed publication. https://sakana.ai/ai-scientist-first-publication/
- Google Research blog, *Accelerating scientific breakthroughs with an AI co-scientist* (Feb 2025). https://research.google/blog/accelerating-scientific-breakthroughs-with-an-ai-co-scientist/
- AgentRxiv project site. https://agentrxiv.github.io/
- Stanford HAI, *2026 AI Index Report: Science chapter*. https://hai.stanford.edu/ai-index/2026-ai-index-report/science
- FutureHouse Platform launch. https://www.futurehouse.org/research-announcements/launching-futurehouse-platform-ai-agents
- OpenAI, *MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering*. https://openai.com/index/mle-bench/

### FI source references
- [`core/engine.py`](../../core/engine.py) — graph construction at line 335, `_node_execute_reflect` at line 1207, `_node_ideate` at line 634, `_node_literature` at line 697, `_node_review` at line 1437.
- [`core/config.py`](../../core/config.py) — `max_iterations` default 2, `exec_reflect_max_iterations` default 3, `review_panel: list[str]`.
- [`docs/USAGE.md:26–47`](../../docs/USAGE.md) — per-quest LLM call breakdown (8–18 floor/ceiling).
- [`docs/capabilities.md:33–35`](../../docs/capabilities.md) — retry/revise loop documentation.
- [`docs/plan.md`](../../docs/plan.md) — phase plan (K, K2, L, M, N, O references).

---

*Word count: ~2,750.*
