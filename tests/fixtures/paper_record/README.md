# Replay fixtures for the paper-record checks

Two real quests whose papers carried FI's own blocks with false facts (read only; copied here):

* `kinetics_s1` (suite round s1, first-order kinetics): the paper said "The experiment ran for 2.3 s"; the run took 223.2 s.
* `fdr_e1` (compare E1, Bonferroni vs Benjamini-Hochberg): the paper said "base seed 0" and "ran for 7.4 s ... within [60 s]";
  `run_trial` never uses its seed (the code seeds from the literal 20261015) and the run took 121.3 s.

`code/`, `raw/ledger.jsonl` (planned lines and the first three trials of each setting), `.fi/topic_*.json`, `config.yaml` and
`paper_old.md` (the whole old paper, blocks included) are copies. `.fi/run_facts.json` is synthesised: the old quests predate
that record, so its `wall_s` is the run.log line of the run that produced the trials (see its `note`).
