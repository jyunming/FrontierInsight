# Vendor parity — `clarify` slot contract

_Generated 2026-05-15T21:01:00+00:00_

**Topic:** Compare three numerical integrators (RK4, Velocity-Verlet, forward Euler) on a damped harmonic oscillator. Report energy drift over 10**4 periods.

Excluded by policy: `copilot_cli` (agentic, broken as a chat backend) and `github_copilot_cli` (third-party proxy). VSCode extension parity is covered by `tests/test_vscode_extension_typescript.py`.

## Summary

| Provider | OK | Elapsed | Sim | Venue | Issues |
|---|---|---|---|---|---|
| `claude_cli` | ✅ | 29.1s | yes | generic | — |
| `codex_cli` | ✅ | 33.5s | yes | generic | — |
| `gemini_cli` | ✅ | 26.0s | yes | nature_mi | — |

## Per-vendor clarify_answers

### `claude_cli`

```json
{
  "comparative_baseline": "analytical closed-form solution of the underdamped harmonic oscillator x(t) = A·exp(-γt)·cos(ωd·t+φ), with total mechanical energy E(t) = ½mv²+½kx² as the drift metric",
  "empirical_vs_theoretical": "mixed",
  "simulatability": "yes",
  "success_metric": "relative energy drift |E(t=10^4 T) − E_analytical(t)|/E₀ vs timestep, expected ranking: forward Euler diverges (drift grows unbounded), RK4 shows slow secular drift ~O(dt^4), Velocity-Verlet bounded oscillatory drift ~O(dt^2) in undamped limit",
  "budget": "a few minutes on a laptop CPU (vectorized numpy, ~10 dt values × 3 integrators × 10^4 periods)",
  "output_kinds": [
    "paper_md"
  ],
  "study_depth": "journal-length",
  "paper_venue": "generic"
}
```

### `codex_cli`

```json
{
  "comparative_baseline": "Compare RK4, Velocity-Verlet, and forward Euler against the exact underdamped analytic solution, with a high-accuracy adaptive SciPy solve_ivp result as a numerical sanity check.",
  "empirical_vs_theoretical": "mixed",
  "simulatability": "yes",
  "success_metric": "Rank methods by lower maximum relative deviation from the analytic damped energy envelope over 10**4 periods, with instability or blow-up counted as worst.",
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
  "comparative_baseline": "exact analytic solution",
  "empirical_vs_theoretical": "mixed",
  "simulatability": "yes",
  "success_metric": "relative energy error compared to the theoretical dissipative energy decay",
  "budget": "a few minutes on a laptop CPU",
  "output_kinds": [
    "paper_md",
    "poster"
  ],
  "study_depth": "journal-length",
  "paper_venue": "nature_mi"
}
```
