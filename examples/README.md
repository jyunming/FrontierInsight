# `examples/`

Each subdirectory ships a single `config.yaml` for one self-contained
quest. Run any of them with:

```bash
python launch.py --config examples/<name>/config.yaml
```

| Example | Topic | Provider in the YAML | Approx. wall time |
|---|---|---|---|
| `integrator_bakeoff/` | RK4 vs Velocity-Verlet vs forward Euler on a damped harmonic oscillator | (provider-agnostic — works with any backend) | ~3 minutes |
| `euv_mor_shot_noise/` | Theoretical LER floor in metal-oxide EUV resists imposed by Poisson photon shot noise at production doses | `ollama` (cloud-routed reasoning model) | ~15 minutes |
| `bernstein_vazirani_noise/` | Bernstein-Vazirani algorithm under per-gate depolarizing noise (numpy state-vector simulator) | `claude_cli` (reuses `claude login` OAuth) | ~20 minutes |

`integrator_bakeoff` is the recommended first quest — it's the one the
README's quickstart walks through. The other two exercise the
literature-router (`euv_mor_shot_noise`) and a sanctioned CLI provider
(`bernstein_vazirani_noise`).

Outputs land in `outputs/<quest_id>/` not in this directory. See
[`docs/capabilities.md`](../docs/capabilities.md) for the per-quest
artifact layout.
