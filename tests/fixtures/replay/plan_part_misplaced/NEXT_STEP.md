# Action needed — the search for the best design cannot start

Quest **1791467091-duv-lowk1-source-dof-opc-ambit-31443b** is paused and waiting for you.

## What to decide
Is the plan what you want to run?

## Recommended
Try another model for the plan step.

## What to do
1. Nothing was run. This quest is a search for the best design, and the search cannot start yet. Running the plan as a plain sweep over fixed settings would answer a different question, so the quest stops here.
2. Still missing from the plan: what the search should make as low or as high as possible; which settings the search may change, and over what range; the design to beat; how many designs the search may try.
3. FI asked the plan's model 2 time(s) to write it from your topic, and it still could not.
4. You do not need to write any of it yourself. What to do:
5. 1. Try another model for the plan step: give `provider.node_models.plan_revise` (or `provider.model`) another model in the quest's config.yaml, then `python launch.py --update 1791467091-duv-lowk1-source-dof-opc-ambit-31443b`. FI asks that model to write the search.

## Also waiting for you
- **Some papers could not be downloaded (needs/WANTED_PAPERS.md lists them, most relevant first).** Nothing to do unless one of them matters: put its PDF in inputs/papers/ and go on.

## Then go on
- **CLI:** `python launch.py --resume 1791467091-duv-lowk1-source-dof-opc-ambit-31443b` (or `fi --resume 1791467091-duv-lowk1-source-dof-opc-ambit-31443b`)
- **Web / VS Code:** open the quest and click **Resume**, or `@fi /resume 1791467091-duv-lowk1-source-dof-opc-ambit-31443b`.
