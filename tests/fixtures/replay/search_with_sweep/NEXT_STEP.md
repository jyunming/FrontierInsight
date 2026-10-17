# Action needed — the plan leaves out what the topic asks for

Quest **1791546148-duv-lowk1-source-dof-opc-ambit-0103fc** is paused and waiting for you.

## What to decide
Is the plan what you want to run?

## Recommended
Try another model for the plan step.

## Or
- Or say it in words: `--resume 1791546148-duv-lowk1-source-dof-opc-ambit-0103fc --revise-plan "<what to add, with its values>"`.

## What to do
1. Nothing was run. The topic asks for something the plan could not include, so an experiment run from it would not answer the topic's question: The topic says to vary or compare illumination sources (conventional, annular, x-dipole, quasar, free-form pixelated source), but the plan's settings do not include it. The topic says to vary or compare pitches (isolated), but the plan's settings do not include it. The topic says to vary or compare conventional sigma (0.3 to 0.9), but the plan's settings do not include it. The topic says to vary or compare annular inner/outer sigma pairs, but the plan's settings do not include it. The topic says to vary or compare x-dipole pole centre (0.5 to 0.95), but the plan's settings do not include it. The topic says to vary or compare x-dipole pole radius (0.1 to 0.2), but the plan's settings do not include it. The topic says to vary or compare conventional sources with sigma (0.1 to 0.95), but the plan's settings do not include it. The topic says to vary or compare neighbour (80 to 2000), but the plan's settings do not include it. The topic says to vary or compare defocus z, but the plan's settings do not include it. The topic says to vary or compare mask line width, but the plan's settings do not include it. The topic asks to report optical ambit, but the plan does not measure it. The topic asks to report extent of CD oscillation, but the plan does not measure it.
2. FI asked the plan's model 2 time(s) to add it, and the plan still lacks it.
3. You do not need to write it yourself. What to do:
4. 1. Try another model for the plan step: give `provider.node_providers.plan_revise.model` (or `provider.model`) another model in the quest's config.yaml, then `python launch.py --update 1791546148-duv-lowk1-source-dof-opc-ambit-0103fc`.
5. 2. Or say it in words: `--resume 1791546148-duv-lowk1-source-dof-opc-ambit-0103fc --revise-plan "<what to add, with its values>"`.

## Also waiting for you
- **Some papers could not be downloaded (needs/WANTED_PAPERS.md lists them, most relevant first).** Nothing to do unless one of them matters: put its PDF in inputs/papers/ and go on.

## Then go on
- **CLI:** `python launch.py --resume 1791546148-duv-lowk1-source-dof-opc-ambit-0103fc` (or `fi --resume 1791546148-duv-lowk1-source-dof-opc-ambit-0103fc`)
- **Web / VS Code:** open the quest and click **Resume**, or `@fi /resume 1791546148-duv-lowk1-source-dof-opc-ambit-0103fc`.
