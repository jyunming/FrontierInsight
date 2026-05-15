# Vendor parity — `clarify` slot contract

_Generated 2026-05-15T21:38:23+00:00_

**Topic:** `Compare three numerical integrators (RK4, Velocity-Verlet, forward Euler) on a damped harmonic oscillator. Report energy drift over 10^4 periods.`

Excluded by policy: `copilot_cli` (agentic, broken as a chat backend) and `github_copilot_cli` (third-party proxy). VSCode extension parity is covered by `tests/test_vscode_extension_typescript.py`.

## Summary

| Provider | OK | Elapsed | Sim | Venue | Errors | Warnings |
|---|---|---|---|---|---|---|
| `claude_cli` | ✅ | 27.0s | yes | nature_mi | — | — |
| `codex_cli` | ✅ | 84.5s | yes | generic | — | — |
| `gemini_cli` | ✅ | 31.9s | yes | nature_mi | — | — |

## Per-vendor clarify_answers

### `claude_cli`

```json
{
  "comparative_baseline": "Compare against the analytic exponential energy-decay envelope E(t)=E0·exp(-2γt) as ground truth, with the Hairer & Lubich (2006) symplectic-integration drift bounds as the secondary reference for ordering RK4 vs Verlet vs Euler.",
  "empirical_vs_theoretical": "mixed",
  "simulatability": "yes",
  "success_metric": "Relative energy error |ΔE|/E0 at t = 10^4·T, with the expected ordering Euler (largest, secular growth) > RK4 (small, bounded secular decay) > Velocity-Verlet (smallest, bounded oscillatory drift) in the undamped limit, and matching the analytic exp(-2γt) envelope in the damped case.",
  "budget": "A few minutes on a laptop CPU (single-threaded numpy; ~10^6 steps × 3 integrators × a small grid of dt and ζ values).",
  "output_kinds": [
    "paper_md"
  ],
  "study_depth": "journal-length",
  "paper_venue": "nature_mi"
}
```

### `codex_cli`

```json
{
  "comparative_baseline": "Compare RK4, Velocity-Verlet, and forward Euler against the analytic underdamped solution, with a very small-step RK4/adaptive reference used as a numerical sanity check.",
  "empirical_vs_theoretical": "mixed",
  "simulatability": "yes",
  "success_metric": "Relative energy drift |E(t_final)-E_reference(t_final)|/E(0) after 10^4 periods, minimized at matched timestep sizes and matched force evaluations where possible.",
  "budget": "a few minutes on a laptop CPU",
  "output_kinds": [
    "paper_md"
  ],
  "study_depth": "brief preprint",
  "paper_venue": "generic"
}
```

### `gemini_cli`

```json
{
  "comparative_baseline": "the exact analytical solution of the damped harmonic oscillator ODE",
  "empirical_vs_theoretical": "mixed",
  "simulatability": "yes",
  "success_metric": "minimization of cumulative energy drift relative to the analytical decay curve",
  "budget": "a few minutes on a laptop CPU",
  "output_kinds": [
    "paper_md",
    "slides",
    "poster"
  ],
  "study_depth": "journal-length",
  "paper_venue": "nature_mi"
}
```
