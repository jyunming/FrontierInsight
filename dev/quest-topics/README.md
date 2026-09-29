# General-science test topics

Eight small simulation studies from different fields, each with a known answer (a closed form or a textbook
relation), so a finished run can be checked against theory. They exist to test FI on more than one kind of
science; none is tied to an industry.

| File | Field | Question |
|---|---|---|
| `sir_final_size.yaml` | epidemiology | stochastic epidemic vs the final-size relation |
| `pendulum_amplitude.yaml` | physics | small-angle period error vs amplitude |
| `false_discovery.yaml` | statistics | Bonferroni vs Benjamini-Hochberg |
| `cv_bias_small_n.yaml` | machine learning | cross-validation bias with few examples |
| `first_order_kinetics.yaml` | chemistry | recovering a rate constant and activation energy |
| `giant_component.yaml` | network science | when a random network forms a giant cluster |
| `neuron_firing_rate.yaml` | neuroscience | model-neuron firing rate vs closed form |
| `heat_diffusion.yaml` | earth science | finite-difference heat flow vs the exact solution |

Run one with `python launch.py --config dev/quest-topics/<name>.yaml`. The only thing to set first is the model
(`provider:` in the file; see `docs/PROVIDERS.md`). Keep new topics field-neutral, simulation only, and under 60
seconds of CPU; `tests/test_quest_topics.py` checks that each file loads and states a 4-page limit.

The fixed benchmark topic (a stochastic SIR study) and its grades are in `dev/evaluation/benchmark/`.
