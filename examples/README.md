# `examples/`

Each subdirectory ships a `config.yaml` for one self-contained quest, and the
`commission.yaml` it starts from (the question, what it covers, the limits and
what counts as success; `fi check <file>` checks it). Run any of them with:

```bash
python launch.py --config examples/<name>/config.yaml
```

| Example | Topic | Provider in the YAML | Approx. wall time |
|---|---|---|---|
| `integrator_bakeoff/` | The midpoint method, 4th-order Runge-Kutta and Velocity-Verlet on a damped harmonic oscillator | (provider-agnostic — works with any backend) | ~3 minutes |
| `bootstrap_ci_coverage/` | How often bootstrap and t confidence intervals contain the true mean for small, skewed samples | `ollama` (cloud-routed reasoning model) | not timed end to end (its simulation runs in under a minute) |
| `bernstein_vazirani_noise/` | Bernstein-Vazirani algorithm under per-gate depolarizing noise (numpy state-vector simulator) | `claude_cli` (reuses `claude auth login` OAuth) | ~20 minutes |

`integrator_bakeoff` is the recommended first quest — it's the one the
README's quickstart walks through. The other two exercise a
simulation with sampled data (`bootstrap_ci_coverage`) and a sanctioned CLI provider
(`bernstein_vazirani_noise`). `format_validation/` holds nine configs, one per paper format, that share one
`commission.yaml` (a made-up study on synthetic image patches); their titles are `format-check-<format>`.

Outputs land in `outputs/<quest_id>/` not in this directory. See
[`docs/capabilities-reference.md`](../docs/capabilities-reference.md) for the per-quest
artifact layout.
