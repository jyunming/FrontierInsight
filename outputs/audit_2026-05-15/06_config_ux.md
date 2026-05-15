# 06 — Config UX Audit

Scope: `core/config.py` (344 LOC, six PRs in the last fortnight added
six new knobs to `EngineConfig` alone), the YAMLs under `examples/`,
and the schema table in `docs/USAGE.md`. The question is whether the
config surface is still navigable by a first-time user, and whether
the defaults reflect what a first-time user actually wants.

The short answer: the schema is correct, internally consistent, and
mostly typed well — but the UX is now visibly suffering from a year of
additive growth. `EngineConfig` has 14 user-facing fields spanning
five conceptual phases (clarify → execute-repair → cross-check →
review-panel → no-simulation), every YAML in `examples/` still copies
the same six-block boilerplate, and there is no preset story for "I
just want a paper, pick sane defaults." Two newly added knobs
(`OutputConfig.require_pdf`, `EngineConfig.dataset_adapters`) lack
cross-field validation — a misconfigured user gets silent no-ops
instead of a YAML error. Doc parity is good but `fi --help` exposes
zero of the YAML schema.

## Findings

### 1. EngineConfig field census

The class now has 14 user-facing fields (and the engine reads exactly
one more, `provider.node_models[...]`, that I'm not counting here).
Grouped by the phase that added them — phase letters come straight
from the comments in `core/config.py:80-194` and the matching commits:

| Field | Type / default | Validator | Phase / PR | Engine read site | Notes |
|---|---|---|---|---|---|
| `framework` | `Literal["langgraph"] = "langgraph"` | — | original (#0) | none — only asserted in `tests/test_config.py:26` | **Dead.** No engine code branches on this. It was forward-looking for "autogen / crew" but those never landed. |
| `max_iterations` | `int = 2` | none | original | `core/engine.py:453,462,1408` | **Missing `ge=`.** A negative value silently disables the design-revise and analyze-reroute loops. |
| `review_loop` | `bool = True` | — | original | `core/engine.py:458` | OK. |
| `clarify_mode` | `ClarifyMode = "off"` | Literal | Phase I (#22) | `core/engine.py:487` | OK. Default is `off` which is correct for fleet/test ergonomics but arguably wrong for first-time interactive use — see §4. |
| `exec_reflect_max_iterations` | `int = 3` | `ge=0` | Phase K (#22) | `core/engine.py:439,1229` | OK. |
| `cross_check_per_finding_k` | `int = 3` | `ge=0` | Phase L (#22) | `core/engine.py:1341` | OK. `ge=0` is correct here: 0 cleanly disables the cross-check loop. Contrast with `auto_collect_top_k`'s `ge=1` (the engine has no "disabled" path for that knob — auto-collect is gated by the parent `auto_collect_data` bool, not by top_k). |
| `enable_analyze_reroute` | `bool = True` | — | Phase L (#22) | `core/engine.py:446,1404` | OK. |
| `ideate_reflect` | `bool = True` | — | Phase M (#22) | `core/engine.py:657` | OK. One extra LLM call per quest — see §4 on default. |
| `review_panel` | `list[str] = []` | none | Phase N (#22) | `core/engine.py:1470` | **Missing string-Literal.** Built-in personas are documented as `methodologist`, `statistician`, `devil_advocate`, `reproducibility` but the type is `list[str]`. Typos fall through to the "generic persona" fallback at runtime with no warning — a misspelled `methadologist` silently uses a generic prompt. See §3. |
| `no_simulation` | `bool = False` | — | Phase D / #57 | `core/engine.py:496,573` | OK. |
| `auto_collect_data` | `bool = True` | — | Phase D1 / #60 | `core/engine.py:763` | Defaults to True. See §4 — this means every no-simulation quest queries Axon by default even when the corpus is empty. The graceful-passthrough story is good (engine just logs INFO and falls through), but the implicit "Axon was queried for every quest you ran" is surprising. |
| `auto_collect_top_k` | `int = 5`, `ge=1` | `ge=1` | Phase D1 / #60 | `core/engine.py:825` | OK. `ge=1` added after a bot review on #60. |
| `dataset_adapters` | `list[str] = []` | none | Phase D2 / #61 | `core/engine.py:878` | **Missing string-Literal.** Available values today are exactly `["worldbank", "wikipedia"]` (`core/datasets/__init__.py:ADAPTER_REGISTRY`). A typo like `worldbnk` is silently dropped with a runtime WARNING — a YAML-load error would catch it earlier. See §3. |
| `dataset_adapter_top_k` | `int = 3`, `ge=1` | `ge=1` | Phase D2 / #61 | `core/engine.py:883` | OK. |

Outside `EngineConfig`, `OutputConfig.require_pdf` (added in Phase D /
#58) also belongs in this audit. It defaults to False and is enforced
both pre-flight (`core/engine.py:1674`) and post-LLM
(`generation/paper.py:117`). The validation gap is **cross-field**
rather than per-field — see §7.

Density read: **6 of the 14 EngineConfig fields landed in the last 6
weeks**, all on the same class, all flat. The class is no longer "the
engine knobs" but is now better described as "every engine knob across
five subsystems." Nested config objects would clarify intent without
moving any logic — see §2.

### 2. Grouping — the case for nested sub-models

Today's `EngineConfig` flattens five conceptual clusters. A natural
re-grouping:

```python
class SimulationConfig(BaseModel):
    no_simulation: bool = False
    auto_collect_data: bool = True
    auto_collect_top_k: int = Field(5, ge=1)
    dataset_adapters: list[str] = []
    dataset_adapter_top_k: int = Field(3, ge=1)

class ReviewConfig(BaseModel):
    review_loop: bool = True
    review_panel: list[str] = []

class ClarifyConfig(BaseModel):
    mode: ClarifyMode = "off"

class ReflectConfig(BaseModel):
    ideate_reflect: bool = True
    exec_reflect_max_iterations: int = Field(3, ge=0)

class CrossCheckConfig(BaseModel):
    per_finding_k: int = Field(3, ge=0)
    enable_analyze_reroute: bool = True

class EngineConfig(BaseModel):
    framework: EngineFramework = "langgraph"   # remove? see §1
    max_iterations: int = Field(2, ge=0)
    clarify: ClarifyConfig = ...
    reflect: ReflectConfig = ...
    cross_check: CrossCheckConfig = ...
    review: ReviewConfig = ...
    simulation: SimulationConfig = ...
```

YAML before/after:

```yaml
# Before (today's flat shape — see examples/integrator_bakeoff/config.yaml)
engine:
  framework: langgraph
  max_iterations: 2
  review_loop: true
  clarify_mode: auto
  ideate_reflect: true
  exec_reflect_max_iterations: 3
  cross_check_per_finding_k: 3
  enable_analyze_reroute: true
  review_panel: [methodologist, statistician]
  no_simulation: false
  auto_collect_data: true
  auto_collect_top_k: 5
  dataset_adapters: [worldbank]
  dataset_adapter_top_k: 3

# After (nested)
engine:
  max_iterations: 2
  clarify: { mode: auto }
  review:
    loop: true
    panel: [methodologist, statistician]
  simulation:
    no_simulation: false
    auto_collect_data: true
    auto_collect_top_k: 5
    dataset_adapters: [worldbank]
    dataset_adapter_top_k: 3
```

Pros of nested:

- **Discoverability**. A user reading a YAML can read the top-level
  keys and immediately know there are "five things." Today the flat
  list looks like 14 unrelated knobs.
- **Documentation locality**. Each sub-model's docstring lives next to
  its fields; users `yaml.dump(EngineConfig.model_json_schema())` get
  a self-grouping schema.
- **Future extensibility**. When Phase D4 adds an `arxiv` adapter and
  a fourth dataset-shaped knob, it lands inside `simulation`, not
  inflating an already 14-deep namespace.
- **Validation locality**. Cross-field rules (e.g. "auto_collect_top_k
  only meaningful when auto_collect_data is True") become trivial
  `model_validator` on the sub-model.

Cons of nested:

- **Migration burden**. Every existing YAML breaks. Three example
  YAMLs ship in the repo + an unknown number of user YAMLs. A
  shim — accept flat layout in `Config.from_yaml` and migrate to
  nested before validation — adds ~40 LoC and a deprecation warning,
  but is the right path. Bonus: it's a self-documenting upgrade
  signal in `run.log`.
- **Engine-side rename**. Every `self.config.engine.cross_check_per_finding_k`
  becomes `self.config.engine.cross_check.per_finding_k`. ~30 sites
  in `core/engine.py`. Mechanical.
- **Test-side rename**. ~15 sites in `tests/test_engine_helpers.py`
  and friends.

Net: the migration is a real day of work, but ergonomics improve
dramatically and every new phase pays a smaller tax. I'd do it
before the next phase ships, not after.

### 3. Validation gaps

Per-field issues, ordered by likelihood-of-bite:

- **`review_panel: list[str]`** (`core/config.py:130`). No Literal
  union and no `field_validator`. The four built-ins are documented
  (`core/config.py:127-129`) but typos fail silently — the engine
  uses a generic persona prefix at `core/engine.py:1470+`. Better:
  define `ReviewPersona = Literal["methodologist", "statistician",
  "devil_advocate", "reproducibility"]` plus a `validate_assignment`
  knob that accepts custom strings only when an env opt-in is set
  (or just print a WARNING at config load time when an entry isn't
  in the built-in list). The async/runtime warning is too late — by
  then the user has paid the panel coordination cost.

- **`dataset_adapters: list[str]`** (`core/config.py:188`). Same
  shape as `review_panel`. Available values today are exactly
  `["worldbank", "wikipedia"]` (per `core/datasets/__init__.py`).
  Typo handling is documented as "WARNING and skipped" (a soft
  failure mode) but a YAML-load error is more honest — the user
  asked for an adapter that does not exist; nothing about the quest
  can fix that. Use `Literal["worldbank", "wikipedia"]` typed
  against the registry's `Literal` shape so adding a new adapter
  flows through type-checked.

- **`max_iterations: int = 2`** (`core/config.py:82`). No
  constraint. Negative values silently disable the design-revise
  loop (`core/engine.py:453`: `state.get("iteration", 0) <
  self.config.engine.max_iterations` is trivially False when
  `max_iterations < 0`). Add `Field(default=2, ge=0)`. Note that 0
  is a legitimate "disable revise" value, so `ge=0` not `ge=1`.

- **`exec_reflect_max_iterations`, `cross_check_per_finding_k`,
  `auto_collect_top_k`, `dataset_adapter_top_k`** — all four
  already have constraints. The asymmetry between `ge=0`
  (cross-check, exec-reflect: 0 means "disabled") and `ge=1`
  (auto-collect, dataset-adapter: gated by the parent bool, so
  passing 0 is nonsensical) is intentional and well-justified by
  the comments. Leave as-is.

- **`KnowledgeConfig.full_text_fetch_timeout_s: float = 15.0`**
  (`core/config.py:265`). No `ge=` or `gt=`. A negative value would
  pass to `httpx.Timeout(...)` and either silently disable timeouts
  or raise from inside `httpx`. Add `Field(15.0, gt=0)`. Same for
  `full_text_fetch_total_s` (`gt=0`) and `full_text_max_kb`
  (`gt=0`).

- **`ExecutionConfig.timeout_s: int = 60 * 30`** (`core/config.py:199`).
  No `ge=`. Negative or zero timeout is a config bug. Add
  `Field(default=1800, ge=1)`.

- **`KnowledgeConfig.external_fallback`** — the source-name
  Literal is documented in the comment block at
  `core/config.py:222-231` (seven legal names) but the type is
  `list[str] | str`. Same fix as `review_panel`: a Literal would
  catch typos at YAML load. The "user-supplied custom sources via
  Axon catalog" extension story (`seed_source_catalog: bool`) is
  the only reason `list[str]` exists — but a custom-source path can
  still validate through a `model_validator` that allows the union
  of built-ins + names known to the live Axon catalog.

### 4. Defaults audit

Defaults are the single biggest UX surface — most users never
override them. The current set is reasonable for the maintainer
running everything; less so for a new user with no Axon corpus.

- **`auto_collect_data: bool = True`** (`core/config.py:165`). The
  comment justifies it well: if Axon is configured, calling it
  before pausing the user is strictly better. But the cascading
  default `knowledge.enabled: True` means the **default-default** is
  "every no-simulation quest queries Axon." For a user who has not
  built an Axon corpus, this is a wasted call, an INFO log line,
  and a (mild) surprise. Two options:

  - Keep `auto_collect_data = True` and document the
    cascading-default contract more loudly — fine if we trust the
    `knowledge.enabled` gate to fail fast (it does; see
    `core/engine.py:819`).
  - Make `auto_collect_data` default to whatever `knowledge.enabled`
    resolves to (a `model_validator(mode="after")` on `Config` that
    sets the engine field when not explicitly set). The user
    intuition matches: "Axon on" → "use Axon for collection too."

  I'd do the second. It eliminates a class of "why is the engine
  calling Axon when I disabled it elsewhere" confusion. The
  `knowledge.enabled=False` short-circuit at `core/engine.py:819`
  already enforces this at runtime — making the default mirror it
  saves one log line per quest.

- **`clarify_mode: ClarifyMode = "off"`** (`core/config.py:95`).
  Justified ("Default for tests and fleet"). But the bare `fi --new`
  / `@fi /new` path is for first-time interactive users — they
  benefit most from the clarify questionnaire. Consider:
  - `EngineConfig.clarify_mode: "off"` (engine default — what fleet
    sees) AND
  - Wizard-generated YAML default `auto` (what the `@fi` interview
    writes). The wizard is the right place to choose
    user-facing defaults; the engine default should be the
    fleet-safe one. (Check: which side wins today? See
    `vscode-frontier-insight/src/`.)

- **`ideate_reflect: bool = True`** (`core/config.py:116`). One
  extra LLM call per quest. Fine if you're on Copilot Pro; less fine
  if you're paying per token on Codex / Claude API. The phase
  comment says "Cheap (one extra LLM call per quest)" — true in
  absolute terms but it doubles `ideate` cost on the budget node.
  Keep default True; document the cost trade-off near
  `docs/PROVIDERS.md`'s cost section (not in `USAGE.md`'s schema —
  schema docs are for "what's the value", not "should I use it").

- **`enable_analyze_reroute: bool = True`** (`core/config.py:112`).
  Fine — the reroute only fires when analyze says so, and it
  consumes from `max_iterations` so worst-case cost is bounded.

- **`OutputConfig.require_pdf: bool = False`** (`core/config.py:323`).
  Correct for back-compat. Unattended/CI users should flip to True;
  the docstring + USAGE.md cover this clearly. I'd add a
  `paper_pdf_skipped.md`-shaped recommendation in `paper_pdf`'s
  output to nudge users toward `require_pdf=True` after they hit a
  skip — a single sentence at the bottom of the diagnostic file.

### 5. YAML UX — line count and preset story

Counting fields a typical user must set in the existing examples:

| Example | Total YAML lines | Engine block lines | Knowledge block lines | Output block lines | Required-by-user fields |
|---|---|---|---|---|---|
| `bernstein_vazirani_noise/config.yaml` | 97 | 4 | 2 | 9 | `topic`, `title`, `provider.name`, `engine.review_loop`, `output.kinds`, `output.paper_format` |
| `euv_mor_shot_noise/config.yaml` | 88 | 4 | 2 | 9 | same shape — 6 fields plus the long topic block |
| `integrator_bakeoff/config.yaml` | 51 | 4 | 12 | 5 | 6 fields plus Axon inline block |

The 80%-of-the-bytes is the topic prose. Of the **structural**
fields, every example overrides the same handful: `provider.name`,
`engine.max_iterations` or `review_loop`, `execution.timeout_s`,
`output.kinds`, sometimes `knowledge.enabled: false`. This is a
preset shape begging to be named.

Proposal: add `Config.preset: Literal["minimal", "scientific",
"journal", "fleet"] | None = None`, with semantics:

```python
PRESETS = {
    "minimal":     # paper.md only, no review loop, clarify off, no panel
    "scientific":  # current "scientific" defaults (today's defaults)
    "journal":     # review_panel of 3, require_pdf=True, paper_format hint
    "fleet":       # clarify_mode=off, ideate_reflect=False, max_iterations=1
}
```

The preset resolves via a `model_validator(mode="before")` that
merges the preset's dict into the user's dict (user wins, preset
fills in). Result: a `minimal` quest is **two lines**:

```yaml
topic: "Compare RK4 vs Verlet on the Kepler problem."
preset: minimal
```

This is the right answer to the "new user landing page" question —
the README's "first quest" example would become 2 lines instead of
30, and the existing `examples/*/config.yaml` files would survive
as advanced-mode references.

Implementation cost: ~50 LoC + tests + a USAGE.md section. The
hard part is choosing what each preset's defaults are; let an early
PR start by hard-coding three presets and iterating in subsequent
PRs based on user feedback. Don't over-design the preset DSL.

### 6. Discoverability

`fi --help` (rendered from `launch.py:38-277`) lists every CLI flag
but ZERO YAML schema. A new user runs `fi --help`, sees `--config
<yaml>`, has no idea what goes inside the YAML.

Three discoverability fixes, increasing impact:

- **`fi --help` mentions USAGE.md** — one line. "For the YAML
  config schema, see https://github.com/jyunming/FrontierInsight/blob/main/docs/USAGE.md
  or run `fi --print-schema`." Trivial. [effort: 5 min].

- **`fi --print-schema`** — emit Pydantic's
  `Config.model_json_schema()` as YAML-shaped output, or pretty-print
  it as a tree. The Pydantic schema already carries every
  description, default, and constraint — the function is `print(
  yaml.safe_dump(Config.model_json_schema()))`. [effort: 30 min,
  excluding the cosmetic prettier-tree if anyone cares].

- **`fi --init [preset]`** — write a stub YAML to stdout (or a file
  with `--init-out path.yaml`). Pairs with the preset story from §5.
  Saves users typing the boilerplate. [effort: 1-2 hours].

The current discoverability path — `cd examples/ && copy config.yaml
mine.yaml && edit` — works, but reaches the user only after they've
already committed to cloning the repo. PyPI installers (`pip install
frontier-insight`) get nothing from the install side. `fi
--print-schema` fixes this for both audiences.

### 7. Cross-validation

Three cross-field rules are not enforced today and would each catch
a real footgun:

- **`output.require_pdf=True` without `paper_pdf` in
  `output.kinds`**. `core/engine.py:1636` early-returns when
  `paper_pdf` is not in `kinds`, and `generation/paper.py:117`'s
  strict-mode raise sits inside an `if "paper_pdf" in kinds` branch
  at `paper.py:84+` — so `require_pdf=True` with `kinds=[paper_md]`
  is a silent no-op. The user thinks they enabled strict mode but
  the engine is in default-skip mode. Add a
  `model_validator(mode="after")` on `OutputConfig`:

  ```python
  if self.require_pdf and "paper_pdf" not in self.kinds:
      raise ValueError(
          "output.require_pdf=True only makes sense when 'paper_pdf' "
          "is in output.kinds. Either add 'paper_pdf' to kinds or "
          "set require_pdf=false."
      )
  ```

- **`engine.dataset_adapters: [...]` with `engine.auto_collect_data:
  False`**. The adapters only run from inside `_node_auto_collect_data`
  (`core/engine.py:763` short-circuits the entire node when
  `auto_collect_data=False`). A user with `auto_collect_data: false,
  dataset_adapters: [worldbank]` gets nothing — silently. Add a
  `model_validator` warning (not error — could be intentional during
  a `no_simulation` resume from a half-run).

- **`engine.no_simulation: True` with `output.kinds`
  containing `slides`/`poster`/`speech` but no `paper_md`**. The
  no-simulation flow always produces `paper.md`, but it's worth
  documenting the dependency (the slides/poster/speech generators
  read from `paper.md`). Lower priority — most users include
  `paper_md`. Document, don't validate.

- **`knowledge.enabled=False` with `auto_collect_data=True` and
  `dataset_adapters=[]`**. Today this is a fine config — the
  auto_collect_data node short-circuits the Axon branch
  (`engine.py:819`) and runs zero adapters. The result is exactly
  the same as `auto_collect_data=False`. Either is fine; the §4
  recommendation (auto_collect_data inherits from knowledge.enabled)
  collapses this case.

### 8. Doc parity (USAGE.md schema vs config.py)

Cross-checking the YAML schema in `docs/USAGE.md:131-208` against
`core/config.py`:

| Field | In USAGE.md? | In config.py? | Note |
|---|---|---|---|
| `engine.framework` | yes (L154) | yes | matches |
| `engine.review_panel` | yes (L162) | yes | matches; available personas listed |
| `engine.no_simulation` | yes (L167) | yes | matches |
| `engine.auto_collect_data` | yes (L168) | yes | matches |
| `engine.dataset_adapters` | yes (L170) | yes | matches; **mentions `worldbank, wikipedia` — does the registry actually have wikipedia?** Confirmed yes (`core/datasets/__init__.py` exports both; commits c700596 + 00a9943) |
| `output.require_pdf` | yes (L207, plus L210-237 expanded section) | yes | matches |
| `knowledge.full_text_max_kb` | yes (L201) | yes | matches |
| `knowledge.source_routing` | yes (L190) | yes | matches |
| `knowledge.seed_source_catalog` | yes (L191) | yes | matches |
| `engine.exec_reflect_max_iterations` | yes (L159) | yes | matches |
| `engine.cross_check_per_finding_k` | yes (L160) | yes | matches |
| `engine.enable_analyze_reroute` | yes (L161) | yes | matches |
| `engine.ideate_reflect` | yes (L158) | yes | matches |
| `provider.node_models` | yes (L148-151) | yes | matches |
| `knowledge.write_back_only_on_accept` | yes (L187) | yes | matches |
| `knowledge.local_papers` | yes (L194-196) | yes | matches |

Doc parity is good — every EngineConfig field appears in
`USAGE.md`. Two specific notes:

- The schema table in `USAGE.md:170` says the available dataset
  adapters are `"worldbank", "wikipedia"`. The registry exports
  both. No drift.
- `USAGE.md` does not show every `KnowledgeConfig` field (e.g. the
  paywall-fetch trio appears in expanded form but not as a single
  reference table). Acceptable — those are advanced; users who need
  them will read `config.py`. If we ever ship `fi --print-schema`,
  the question becomes moot.

The biggest doc gap is **NOT in USAGE.md**: it's that the README and
`docs/capabilities.md` show only the simulation flow and never
describe what fraction of `EngineConfig` an `@fi /new` interview
actually fills in. A new user reading the README has no mental model
for "the 14 knobs exist, but the wizard sets 13 for me." Worth a
sentence in the README's "first quest" section.

## Recommendations

Tagged `[impact]` (S/M/L) and `[effort]` (S/M/L), roughly ordered by
ratio.

1. **Add `model_validator` on `OutputConfig` for `require_pdf`
   without `paper_pdf` in `kinds`.** [impact: M] [effort: S]
   Catches a real silent-no-op footgun. ~10 LoC + 1 test.

2. **Add `ge=` / `gt=` constraints to the unconstrained ints/floats.**
   [impact: M] [effort: S] `max_iterations: ge=0`,
   `execution.timeout_s: ge=1`, the three `full_text_*` floats:
   `gt=0`. ~5 lines across `config.py` + 1 parametric test. No
   behavior change for any valid config.

3. **Type `dataset_adapters` and `review_panel` as `list[Literal[...]]`
   (with a custom-string fallback path validated at config load,
   not at runtime).** [impact: M] [effort: M] Catches typos before
   any LLM cost is incurred. Custom-persona / custom-adapter users
   can opt out of strict validation via a documented env flag — or
   we can simply WARN at load time and continue (a strict-mode
   variant only flips to error). ~30 LoC + 2 tests each.

4. **Add `fi --print-schema` that emits
   `yaml.safe_dump(Config.model_json_schema())` (or a prettier
   tree).** [impact: M] [effort: S] Pure win for discoverability;
   the schema is already richly described via Pydantic. ~20 LoC +
   1 test. Mention in `fi --help`.

5. **Add a `Config.preset: Literal[...] | None = None` with three
   curated presets (`minimal` / `scientific` / `journal`).**
   [impact: L] [effort: M] Drops the new-user first-quest YAML
   from 30 lines to 2. Pair with an `fi --init [preset]` command.
   Bonus: the preset is logged at run start, which is good
   debuggability — "this run was preset=minimal, here's what that
   meant on commit X." ~60 LoC + 4 tests + a USAGE.md section.

6. **Restructure `EngineConfig` into nested sub-models
   (`simulation`, `review`, `clarify`, `reflect`, `cross_check`).**
   [impact: L] [effort: L] The right shape for the next 6 months
   of phases. Provide a one-version backward-compat shim that
   accepts the flat layout with a deprecation warning. Drives down
   per-field comment density (each sub-model owns ~3-5 fields, not
   14). Migration ~200 LoC across `core/`, `tests/`, `examples/`,
   `docs/USAGE.md`.

7. **Make `engine.auto_collect_data` default to
   `knowledge.enabled`** (via a `model_validator(mode="after")` on
   `Config`). [impact: S] [effort: S] Mirrors the runtime
   short-circuit and removes a confusing log-line for users without
   Axon. ~10 LoC + 1 test. Document the behavior in the field's
   description.

8. **Remove or document `engine.framework`.** [impact: S]
   [effort: S] No engine code reads it. Either delete (one line
   deletion + adjust `tests/test_config.py:26`) or document the
   roadmap if `autogen`/`crew` adapters are still planned. Today
   it confuses readers — a field labeled "the only value supported
   today" begs the question of what's coming.

9. **Cross-field validator for
   `dataset_adapters` + `auto_collect_data=False`.** [impact: S]
   [effort: S] WARN at load if the adapters are configured but the
   parent gate is off. ~10 LoC + 1 test.

10. **Update README to mention the 14-knob reality and the
    interview-fills-defaults story.** [impact: S] [effort: S] One
    paragraph; pairs with the preset PR.

If only one thing ships: do (4) or (5). (4) is the smallest tool
that fixes the discoverability complaint forever; (5) is the biggest
UX bump per line of code. (6) is the right long-term shape but
should wait until at least two of (1)–(5) are merged so the nested
migration absorbs the small fixes rather than scheduling them
separately.

## References

- `core/config.py` (full file) — the audited surface; field
  comments are dense and worth preserving when restructuring.
- `core/config.py:80-194` — `EngineConfig`, 14 fields, five phases.
- `core/config.py:309-329` — `OutputConfig`, including `require_pdf`
  added in #58.
- `core/engine.py:1611-1690` — pre-flight PDF check; demonstrates the
  silent no-op when `require_pdf=True` but `paper_pdf` not in kinds.
- `core/engine.py:763` — `auto_collect_data=False` short-circuit;
  precedent for the recommended cascading default.
- `core/engine.py:819` — `knowledge.enabled=False` short-circuit in
  the auto-collect path; runtime version of the same.
- `examples/bernstein_vazirani_noise/config.yaml` — 97-line example;
  ~6 structural fields outside the topic prose.
- `examples/euv_mor_shot_noise/config.yaml` — 88 lines, same shape.
- `examples/integrator_bakeoff/config.yaml` — 51 lines, only example
  with knowledge enabled + inline Axon block.
- `docs/USAGE.md:131-208` — YAML schema reference; doc parity is good.
- `docs/USAGE.md:296-400` — `no_simulation` mode prose; the right
  level of detail, lives away from the schema table.
- `tests/test_config.py` — coverage is solid for Literal rejection,
  path expansion, and round-trip stability; lacks coverage for the
  cross-field rules recommended above.
- `tests/test_engine_helpers.py:1190-1380` — `require_pdf` test
  block; would extend cleanly with the `require_pdf without paper_pdf`
  case from §7.
- Phase PRs referenced: #57 (no-simulation), #58 (require_pdf), #59
  (clarify simulatability), #60 (auto_collect_data), #61
  (dataset_adapters), #62 (wikipedia adapter).
