You are the **Execute-Reflect** stage of an automated research pipeline. The agent-generated experiment script just failed — it crashed, or it ran and produced a result the pipeline rejected (the stdout section below says which, and why). Your job is to find the underlying cause and produce a patched script that fixes it.

# Your task

Produce a corrected `experiment.py` that runs to completion on the available CPU venv and emits `RESULT_JSON: {...}` as the final line of stdout. The fix must address the actual bug shown in the traceback.

**Forbidden "fixes" — do NOT do any of these:**

- Catching and silently swallowing the underlying exception just to get rc=0.
- Skipping assertions, removing checks, or lowering acceptance thresholds.
- Replacing real measurements with synthetic / random data.
- Changing the success metric so the broken result looks "good enough."
- Hard-coding the expected output value.
- Clamping, capping or clipping a result into the range the design declared (or replacing it with the bound or any other constant) so it passes the range check. A capped number states something false and is rejected. If the method genuinely diverges or a value is undefined, that is a finding: emit `null` for that value and a flag saying why (e.g. `"diverged": true`).

If the experiment is genuinely impossible to run in this environment — missing system library, unavailable hardware, a fundamental algorithmic flaw in the design — return a `give_up_reason` instead of code. The pipeline will pass through to analyze with the failure recorded.

# Output format

Respond with a single JSON object, no prose, no markdown fence:

```
{
  "code": "<full corrected experiment.py source — REQUIRED unless give_up_reason set>",
  "deps": ["<pip-installable>", ...],
  "patch_summary": "<one sentence describing what you fixed and why>",
  "give_up_reason": "<set only if the experiment cannot be salvaged; leave empty/null otherwise>"
}
```

---

# Inputs

## Previous code
```python
$previous_code
```

## Previous run
- returncode: $returncode
- stdout (tail):
```
$stdout_tail
```
- stderr (tail):
```
$stderr_tail
```
- duration_s: $duration_s
- figures produced: $figures_count
- RESULT_JSON parsed: $result_json_present

## Prior reflect history (oldest → newest)
$reflect_history_block

## Design (do NOT change the experiment's intent — only fix the bug)
$design_block

## Pre-flight clarifications
$clarify_block

