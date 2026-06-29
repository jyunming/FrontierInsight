You are reading the **collected evidence** for a research quest running in **no-simulation mode**. The files below are whatever this quest gathered into its data dir — these may be **datasets the user supplied**, and/or **sources the engine auto-retrieved** (web pages and academic literature downloaded to `data/literature/`). Treat each file as evidence to be summarized faithfully, attributing claims to the file they came from. Your job is to synthesize it into a structured `result_json` payload that downstream nodes (analyze → cross_check → write → review) can treat as the basis for the analysis.

You are NOT running a simulation. You are NOT inventing data. You are summarizing what is actually in the collected files — do not assert the user personally produced data that came from a retrieved source, and don't claim a measurement the files don't contain.

## Quest topic

${topic}

## Design (what the prior LLM call planned)

```json
${design_block}
```

The design above tells you what variables the user was supposed to measure. Use it to organize the result_json — when the design specifies a metric like "trust_in_institutions" or "average_decision_latency_ms", produce a key with that exact name. If the user's data doesn't cover a planned variable, note that under a `missing_measurements` array — DO NOT invent values.

## File manifest (deterministic — do not edit IDs)

${file_manifest}

## File contents

${content_blocks}

# Instructions

Produce a single JSON object with these top-level keys (omit any key you have no evidence for; never invent values to fill a slot):

```json
{
  "summary": "1-3 sentence description of what the user collected and what overall pattern emerges.",
  "n_files": <integer count of files actually parsed>,
  "key_findings": [
    {
      "finding": "Concrete claim grounded in the dropped data.",
      "evidence": "Quote or specific reference to the file IDs above that support this claim.",
      "confidence": "high | medium | low"
    }
    // 2-6 findings; each MUST cite at least one file from the manifest by ID.
  ],
  "measurements": {
    // Key-value mapping of the design's variables to the user's measured values.
    // Each value is either a number, a string label, OR an object with
    // {value, unit, n_observations, source_file_ids}. Use whatever shape
    // the data naturally takes. NEVER guess — if you can't extract a value
    // from the files, omit the key and add the variable name to
    // missing_measurements below.
  },
  "missing_measurements": [
    // Variable names the design planned for but the user's data doesn't cover.
    // The analyze node will treat these as gaps the user should fill on a
    // future resume.
  ],
  "limitations": [
    // 2-4 honest limitations of the dropped data: sample size, selection
    // bias, missing covariates, format problems, etc. The user is doing
    // empirical work, not running a controlled experiment — surface this.
  ],
  "primary_sources": [
    // Short list of the most authoritative files in the manifest, by ID.
    // The analyze + write nodes will treat these as the citation backbone
    // when authoring the paper. Anchor every key_finding to one of these
    // when possible.
  ]
}
```

## Style constraints

- Markdown / prose responses are NOT acceptable — the result_json field must be a single JSON object.
- Cite file IDs from the manifest above (e.g. `[3]`, `[7]`) when grounding any claim.
- If the user's data is sparse or ambiguous, say so explicitly under `limitations`. The downstream analyze + write nodes will be more useful with an honest "limited evidence" finding than a confidently-wrong invented one.
- Do NOT add a top-level `result_json` wrapper — the engine reads the parsed object directly. The very first character of your response should be `{`.
- If the dropped files genuinely don't answer the research question (e.g., the user dropped a single unrelated PDF), produce a result_json with empty `key_findings` and `measurements`, a clear `summary` saying so, and 1-2 `limitations` explaining what additional data would be needed.
