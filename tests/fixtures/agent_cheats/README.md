# Clean code of two real quests, for tests that judge a quest's numbers

`kinetics/clean/` and `fdr/clean/` are the code of the s1 suite round (quests `1792078228-kinetics-parameter-recovery`
and `1792081309-fdr-bonferroni-vs-bh`), copied without FI's own generated files (`run.py`, `README.md`, `CHANGELOG.md`,
`requirements.txt`, `tests/`); `protocol.json` is each quest's frozen protocol (`needs/FROZEN_PROTOCOL.json`,
`protocol` block). The protocols carry the plan's expected values: that is test data here, not something an agent sees.

Used by `tests/test_equation_verdict_is_fis.py` and `tests/test_fi_judges_numbers.py`.
