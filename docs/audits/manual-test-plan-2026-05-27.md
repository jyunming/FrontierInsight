# Manual test plan — PR #158 + #159 + #160 + #161 + #162 + #163 + #164

After this PR series merges, exercise the new behaviour end-to-end in both `--serve` (web dashboard) and VSCode (`@fi /...` chat). The Python unit + e2e suites already pin the wiring; this list covers the surfaces tests can't cover (browser rendering, VSCode QuickPick / input-box, real LLM streaming, file-system pause/resume).

Each scenario lists the recipe, what to look for, and which PR's behaviour it exercises.

---

## Topic for every scenario

```
Compare whether SGD or Adam converges faster on a small MLP fitting
y = sin(x) on a single training run with one random seed; report the
iteration count to reach loss < 0.01.
```

Deliberately under-specified so the methodologist persona should flag at least `single_point_eval` and possibly `pseudo_units`. This is what makes the must-flag chip light up in the UI.

---

## Cheap-flow YAML (paste into web `/interview` "Power user" or save as `quests/manual-test.yaml`)

```yaml
topic: |
  Compare whether SGD or Adam converges faster on a small MLP fitting y = sin(x)
  on a single training run with one random seed; report the iteration count
  to reach loss < 0.01.

title: "manual-test"

provider:
  name: "openai"          # for web run — swap to "vscode_extension" for VSCode
  model: "gpt-4.1-mini"

engine:
  framework: "langgraph"
  clarify_mode: "off"
  max_iterations: 2
  review_loop: true
  review_panel: ["methodologist", "statistician", "devil_advocate"]
  human_feedback_gate: "after_review"   # default after PR #158, included for clarity
  # human_feedback_gate is the always-on review gate. To exercise pause-drop:
  # pause_for_user_input: "after_design"      # PR #164 only
  # cross_check_verify: true                  # PR #162 only — extra LLM cost per finding
  # execute_replicates: 3                     # PR #163 only — runs experiment 3x
  # auto_accept_on_pass: false                # PR #158 — true skips clean reviews

execution:
  sandbox: "venv"
  timeout_s: 120

knowledge:
  enabled: false

output:
  kinds: ["paper_md"]
```

---

## Web `--serve` scenarios

Launch: `python launch.py --serve`. Browse http://127.0.0.1:8765.

### W-1 — Human-review banner: refine path (PR #158)

1. Submit the topic via `/interview` (Power user → paste the YAML above).
2. Watch `/quest/<id>` until `[review]` fires. The status badge should flip to **"Paused (human review)"** (orange dot, warning colour).
3. Confirm the **"Paused for human review"** orange banner shows:
   - Verdict + score chips.
   - Red **"Must-flag hits (non-bypassable)"** section with at least one chip (likely `[methodologist] single_point_eval`).
   - **"Suggestions"** bulleted list with revision asks.
   - The "Feedback (required for refine)" textarea.
4. Type one line of feedback (e.g. `"add 5 seeds and report mean ± std"`), click **Refine**.
5. Engine loops back to design with the feedback injected. `/quest/<id>/run.log` should show `[human_feedback] refine → iteration 2 (feedback len=…, total entries=1)` and `[design] iteration=1`.

**Watch for**:
- Pressing **Enter** in the textarea must NOT submit the form (PR-A copilot review fix #5). It should just insert a newline.
- After Refine, the banner disappears; status returns to **Running**.

### W-2 — Human-review banner: accept path (PR #158)

1. New quest with same YAML, wait for the human-review banner.
2. Click **Accept**.
3. The banner clears immediately. Status flips to the verdict ("accept" / "complete"). `paper.md` is the final artifact.

**Watch for**:
- After accept, `<quest_root>/.fi/human_review.json` and `<quest_root>/.fi/human_review_answer.json` are BOTH gone. (PR-A copilot review fix #1 + #7 — snapshot cleanup on resolve. If you see either file lingering, the cleanup regressed.)

### W-3 — Human-review banner: reject path (PR #158)

1. New quest, wait for the banner.
2. Click **Reject** without typing feedback.
3. The summary's verdict should be **`rejected`** (not `accept`). The Axon write-back should NOT fire (existing `write_back_only_on_accept` gate).

### W-4 — Feedback history accumulates across iterations (PR #158)

1. New quest, hit the banner.
2. Refine with feedback **A** (e.g. "add seeds").
3. Wait for next review banner (will fire again because methodologist flag won't disappear in 1 pass).
4. The "Prior refinement asks (this quest)" section should now show **"round 0: add seeds"**.
5. Refine with feedback **B** (e.g. "include validation curve").
6. Wait for the third review banner. The history should now show BOTH:
   - round 0: add seeds
   - round 1: include validation curve

**Watch for**: the design prompt log (`/quest/<id>/run.log`) should reference both asks, not just the most recent.

### W-5 — Refine button validates feedback (PR #158)

1. Trigger the human-review banner.
2. Click **Refine** with the textarea empty.
3. A browser `alert("Refine requires feedback text.")` should fire. No POST should hit the server.

### W-6 — Pause-drop after-design (PR #164)

Edit the YAML to add `pause_for_user_input: "after_design"` under `engine:`.

1. New quest. Watch the log for `[design] iteration=0` then `[after_design] paused for user input: drop files into …`.
2. Quest exits clean (rc=0). The status badge stays at "Running" briefly then the subprocess terminates.
3. Open `<quest_root>/inputs/`. Confirm `README.md` is present with drop-zone hints AND `papers/` + `data/` subdirs exist.
4. Drop a small CSV into `<quest_root>/inputs/data/` (one with a numeric column).
5. Click **Resume** on the dashboard.
6. Watch `[analyze]` fire. Its `run.log` line should mention `picked up X user-supplied dataset(s) from inputs/data/`. The analyze prompt in the log should include the relative file path under `_user_supplied_datasets`.

**Watch for**:
- On resume, the engine MUST NOT pause again at after-design. (`user_pauses_fired` state field; if you hit a second pause, that's a regression.)

### W-7 — Pause-drop after-paper (PR #164)

Edit YAML: `pause_for_user_input: "after_paper"`.

1. New quest. Watch `[write] wrote …/paper.md` then `[after_paper] paused for user input`.
2. Drop a reference PDF into `<quest_root>/inputs/papers/`. (Any short PDF works; the literature node ingests it.)
3. Resume. Watch `[literature] picked up 1 user-supplied paper(s) from inputs/papers/`.
4. The paper draft on the next iteration should cite the dropped paper if the methodology revise loop re-fires.

### W-8 — Auto-accept-on-pass (PR #158)

Edit YAML: `engine.auto_accept_on_pass: true`. Use a topic that's likely to get a CLEAN verdict (e.g. drop the panel down to `[]` and pick a topic where methodologist won't flag anything).

1. New quest. Watch `[review] verdict=accept` then `[run] human_feedback auto-accept (verdict=accept, no must_flag_hits)`.
2. No banner appears. The quest finalises automatically.

### W-9 — Subprocess pause-drop flow without web button (PR #158)

This is the "headless" variant that mirrors what `--fleet` does.

1. Run quest with `python launch.py --config quests/manual-test.yaml`. Stop after the human-review pause logs `paused for human review: write your decision into <path>/.fi/human_review_answer.json`.
2. Manually `echo '{"action":"accept","feedback":""}' > <quest_root>/.fi/human_review_answer.json`.
3. Re-run: `python launch.py --config quests/manual-test.yaml --resume <quest_id>`.
4. Quest should consume the answer file (logged: `consuming pre-staged human-review answer (action=accept)`) and finalise.

---

## VSCode `@fi /...` scenarios

Install the rebuilt `.vsix`: `cd vscode-frontier-insight && npm run package && code --install-extension vscode-frontier-insight.vsix --force`. Reload window.

### V-1 — Icon at 512×512 (PR #159)

1. Open the Extensions sidebar → installed extensions → "Frontier Insight".
2. The icon should render crisply at the marketplace gallery size (not blurry/upscaled).
3. Hi-DPI screens should show the indigo→cyan gradient sharply.

### V-2 — Human-review QuickPick: accept (PR #158)

1. `@fi /new` in Copilot Chat. Walk through the interview using the same topic. Accept the 3-persona panel default (PR #161 should default to it).
2. Watch the chat panel for `→ **review** · panel mode: …`.
3. After review, the chat should render:
   ```
   🟡 **Human review** — the engine paused after review.

   - **Verdict:** `revise`
   - **Score:** `2`
   - **Iteration:** `0`
   - **Paper:** `…/paper.md`
   - **Must-flag hits (non-bypassable):** `[methodologist] single_point_eval`

   **Suggestions:**
     - …
   ```
4. A QuickPick should appear with **Accept / Reject / Refine**. Pick **Accept**.
5. Chat shows `— human review: **accept**; resuming…`. Quest finalises.

### V-3 — Human-review QuickPick: refine (PR #158)

1. Same as V-2 but pick **Refine** from the QuickPick.
2. A second input box appears with placeholder "What should the rewriter change?".
3. Type feedback (one line), press Enter.
4. Chat shows `— human review: **refine** (with feedback); resuming…`. Engine loops back to design.

**Watch for**: Esc on either modal should NOT crash the quest. It cancels and the chat shows `— human-review cancelled by user; engine falls back to accept.`

### V-4 — Markdown injection guard (PR #158)

1. Trigger a review that produces a suggestion containing `#` or `-` at line start (e.g. methodologist writes `# Critical issue: …` or `- circular evaluation`).
2. In the chat panel, those characters should render as literal text, NOT as a markdown heading / bullet.

(This validates the PR-A copilot-review fix to `escapeMd` — the leading `#` / `-` / `+` / `>` / `<digits>.` are now escaped.)

### V-5 — 3-persona default panel from VSCode `@fi /new` (PR #161)

1. `@fi /new`. Walk through to the "Reviewer panel" question.
2. The default option should be **"3-persona panel (default)"** (not "Single reviewer"). The description should explain that single-reviewer loses must-flag enforcement.
3. Accept the default. After submission, open the generated YAML at `outputs/_drafts/…`.
4. The YAML should include:
   ```yaml
   engine:
     review_panel:
       - "methodologist"
       - "statistician"
       - "devil_advocate"
   ```

### V-6 — Pause-drop slot in interview (PR #164)

1. `@fi /new`. Walk to the new **"Pause for user-supplied papers / datasets"** question.
2. The four choices should appear: Never / After design / After paper / Both.
3. Pick **After design**. The generated YAML should contain `pause_for_user_input: "after_design"`.

### V-7 — Pause-drop in VSCode quest (PR #164)

1. Edit a manual YAML to set `pause_for_user_input: "after_design"` and `provider.name: "vscode_extension"`.
2. `@fi /start <path>`.
3. After the design node fires, the quest should exit cleanly. Chat shows the pause message.
4. Drop a CSV into `<quest_root>/inputs/data/`.
5. `@fi /resume <quest_id>`. Confirm analyze picks up the file.

---

## Cross-cutting validations

### X-1 — Output-gate registry surfaces the gate name (PR #160)

This needs a real CLI invocation that returns a rate-limit message. Hard to trigger synthetically; the unit tests already pin the behaviour. The on-disk evidence:

- Hit any quest with `provider.name: claude_cli` on an exhausted session, OR temporarily edit `core/provider._CLI_RATE_LIMIT_MARKERS` to add a string you know your mock CLI will return.
- Inspect run.log for a `_CliTransientError` line. Confirm it includes the gate name: `claude tripped output gate 'rate_limit_message' (matched: …)`.

### X-2 — CoVe verification pass (PR #162)

1. Add `engine.cross_check_verify: true` to a quest YAML that will actually hit non-neutral cross-check classifications (needs `knowledge.enabled: true` + Axon hits, or a topic with rich literature in your local corpus).
2. After the quest finishes, open `frontier_insight_summary.json`.
3. For each finding under `cross_check`, confirm:
   - `verification_notes` is a non-empty list when the first pass produced non-neutral hits.
   - `first_pass` carries the original (pre-verification) classification.
   - The top-level `supporting` / `conflicting` / `neutral` lists reflect the post-verification revision.

### X-3 — Multi-seed replication (PR #163)

1. Edit a manual YAML to set `engine.execute_replicates: 3`.
2. Use a topic that legitimately generates an experiment script (don't use no_simulation).
3. The generated `experiment.py` (PR-F prompt change) should mention reading `os.environ.get("FI_REPLICATE_SEED", ...)` for its RNG seeding.
4. Watch run.log for `[execute] replicating: 2 additional seeds (1..2)`.
5. After analyze fires, `frontier_insight_summary.json` should show:
   - `result_json_replicates` carries 3 entries, each tagged with `_seed`.
   - The analyze prompt log mentions `aggregate_mean_std` with `mean` + `std` + `n=3` for each numeric field.

### X-4 — Jittered retry backoff (PR #161)

Visible only under load. The unit-test patch confirms the call site uses `wait_random_exponential`. To stress-test, launch a fleet of 8 quests with `python launch.py --fleet a.yaml b.yaml … --max-concurrent 8` against a flaky upstream. Watch the retry timestamps in each quest's run.log — they should be staggered, not synchronised.

### X-5 — Resume-from-disk human-review answer (PR #158)

1. Launch a quest via `python launch.py --config quests/manual-test.yaml` (NOT --interactive, NOT --serve in-process).
2. Wait for the `paused for human review` log line; the subprocess exits cleanly (rc=0).
3. Edit `<quest_root>/.fi/human_review_answer.json`:
   ```json
   {"action": "refine", "feedback": "tighten the abstract"}
   ```
4. Re-run with `--resume <quest_id>`.
5. Engine consumes the file and loops back to design. Confirm via the log line `consuming pre-staged human-review answer (action=refine)`.

---

## Quick sanity checks (do these first)

Before any of the above:

```bash
# 1. Confirm the right code is on main.
git pull
git log --oneline -8

# 2. Run the affected unit test clusters.
python -m pytest tests/test_human_feedback_gate.py tests/test_pause_drop_anytime.py \
  tests/test_replicates.py tests/test_cross_check.py tests/test_provider_streaming.py \
  tests/test_interview_tiers.py tests/test_interview.py tests/test_web_server.py \
  tests/test_vscode_extension_icon.py -v

# 3. Rebuild the vsix.
cd vscode-frontier-insight
npm run compile
npm run package
cd ..
```

If any of those fail, that's a regression — don't proceed to manual testing until they're green.
