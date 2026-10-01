# General-science test topics

Eight small simulation studies from different fields, each with a known answer (a closed form or a textbook
relation), so a finished run can be checked against theory, and one literature review whose conclusion is settled.
They exist to test FI on more than one kind of science; none is tied to an industry.

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
| `mmr_autism.yaml` | medicine (a literature review) | is MMR vaccination associated with autism |

Run one with `python launch.py --config dev/quest-topics/<name>.yaml`. Before the first run, set the model in the
`provider:` block of the file. The files ship with `openai`, which needs an `OPENAI_API_KEY` environment variable;
`docs/PROVIDERS.md` lists the other choices and what each one needs. Each run pauses for nothing (no paper or
review prompts) and does not write its result back to the knowledge store, so repeat runs stay independent.

Checking a run: every topic states its reference answer in the text, and a run should draw it in the figures. The
paper's own citations are whatever the model found; a run is asked to cite only what it has actually read.

Keep new topics field-neutral and, for a simulation, under 60 seconds of CPU. `tests/test_quest_topics.py` checks
that each file loads, states a 4-page limit, has the SCOPE and GOALS sections and a unique title, and stays
unattended.

`answers/` holds the self-benchmark's fixed answers (`<topic>.answer.json`: the metric ids and the setting the
answer is read at, the expected values with their tolerance and source); `dev/evaluation/bench/README.md` explains
them. The fixed benchmark topic (a stochastic SIR study) and its hand grades are in `dev/evaluation/benchmark/`.
