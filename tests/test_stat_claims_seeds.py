"""A seed the paper gives is checked against the run's records (replay of suite r8 false_discovery: the paper called
the seed of its bootstrap intervals, ``default_rng(20261013)`` in code/experiment.py, the base seed every trial was
derived from; the study record's base seed was 0 and nothing compared the sentence with the records)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

from core import disclosure
from core.engine import Engine
from core.stat_claims import SeedRecords, check, seed_records

FIXTURE = Path(__file__).parent / "fixtures" / "stat_claims_seed" / "fdr_r8"


def _quest(tmp_path: Path) -> Path:
    shutil.copytree(FIXTURE, tmp_path, dirs_exist_ok=True)
    return tmp_path


def _seed_findings(paper: str, seeds: SeedRecords) -> list:
    return [f for f in check(paper, seeds=seeds).findings if f.kind == "seed_not_recorded"]


def test_the_records_are_read(tmp_path: Path) -> None:
    seeds = seed_records(_quest(tmp_path))
    assert seeds.base_seed == 0
    assert 1078139634 in seeds.trial_seeds and len(seeds.trial_seeds) > 300
    assert ("code/experiment.py:291", 20261013) in seeds.code_seeds


def test_r8_s_base_seed_sentence_is_named(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    paper = (root / "methods.md").read_text(encoding="utf-8")
    found = _seed_findings(paper, seed_records(root))
    assert len(found) == 1
    msg = found[0].message
    assert "base seed 20261013" in msg and "base seed 0" in msg and "code/experiment.py:291" in msg


def test_the_corrected_sentence_passes(tmp_path: Path) -> None:
    seeds = seed_records(_quest(tmp_path))
    paper = ("Each trial's seed was derived from base seed 0, the setting and the trial number. The bootstrap "
             "intervals used seed 20261013. Trial 0 of the first setting used seed 1078139634.")
    assert _seed_findings(paper, seeds) == []


def test_a_seed_no_record_holds_is_named(tmp_path: Path) -> None:
    seeds = seed_records(_quest(tmp_path))
    found = _seed_findings("All results use random seed 42.", seeds)
    assert len(found) == 1 and "seed 42" in found[0].message


def test_counts_lists_and_numbers_after_seed_words_are_not_seeds(tmp_path: Path) -> None:
    seeds = seed_records(_quest(tmp_path))
    paper = ("Each setting used 1,537 seeds. We ran 10 seeds per setting. Seeds 3 to 9 were dropped. "
             "The seed policy gives one seed of 1,537 trials per setting. A seed-dependent result was 0.41.")
    assert _seed_findings(paper, seeds) == []


def test_a_seed_constant_in_the_code_is_read_and_named(tmp_path: Path) -> None:
    """Suite r9k: ``BASE_SEED = 20261014`` fed only simulate.py's own stand-alone run; FI's harness derived every
    trial's seed from base seed 0, and the paper said the trials used 20261014."""
    root = _quest(tmp_path)
    (root / "code" / "simulate.py").write_text("BASE_SEED = 20261014  # seed for the stand-alone run\n"
                                               "rng = np.random.default_rng(seed)\nx = rng.random(5)\n",
                                               encoding="utf-8")
    seeds = seed_records(root)
    assert ("code/simulate.py:1", 20261014) in seeds.code_seeds
    assert not any(value == 5 for _loc, value in seeds.code_seeds)
    found = _seed_findings("A fixed base seed of 20261014 defined independent random streams.", seeds)
    assert len(found) == 1 and "code/simulate.py:1" in found[0].message


def test_silent_without_records(tmp_path: Path) -> None:
    seeds = seed_records(tmp_path)
    assert seeds == SeedRecords()
    assert _seed_findings("The base seed was 99 and the random seed 42.", seeds) == []


def test_seed_n_is_not_checked_without_the_ledger(tmp_path: Path) -> None:
    """Without FI's ledger the trial seeds are unknown, so only a base seed against the study record is checked."""
    seeds = SeedRecords(base_seed=7)
    assert _seed_findings("We used seed 42.", seeds) == []
    assert len(_seed_findings("We used base seed 42.", seeds)) == 1


def test_fi_s_own_run_record_is_not_read(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    seeds = seed_records(root)
    record = disclosure.run_record(root, topic="t", duration_s=10.0)
    assert "base seed 0" in record
    paper = disclosure.mark_run_record("# P\n\n## Limitations\nThe grid is coarse.\n", record)
    assert "fi:run-record:" in paper and _seed_findings(paper, seeds) == []
    found = _seed_findings(paper + "\nThe master seed was 5.\n", seeds)
    assert len(found) == 1 and found[0].message.startswith("the paper gives base seed 5")


class _Recorder:
    def __init__(self, root: Path) -> None:
        self.quest_root = root
        self._log = SimpleNamespace(warning=lambda *a, **k: None, info=lambda *a, **k: None)


def test_the_engine_passes_the_quest_s_records(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    paper = (root / "methods.md").read_text(encoding="utf-8")
    state = {"result_json": {}, "result_json_replicates": [], "figure_records": {}, "design": {}}
    hits = Engine._statistics_claim_hits(_Recorder(root), paper, state)  # type: ignore[arg-type]
    assert len(hits) == 1 and hits[0].startswith("mislabelled_statistic:") and "20261013" in hits[0]
    audit = json.loads((root / "paper" / "statistics_audit.json").read_text(encoding="utf-8"))
    assert "seed_not_recorded" in audit["checks_run"] and audit["findings"][0]["kind"] == "seed_not_recorded"


def test_the_third_of_ten_seeds_is_not_a_seed_value(tmp_path: Path) -> None:
    seeds = seed_records(_quest(tmp_path))
    assert _seed_findings("Results are shown for seed 3 of 10 and seed 4 out of 10.", seeds) == []


def test_only_seed_names_are_seeds_in_the_code(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    (root / "code" / "simulate.py").write_text("use_seed = 1\nn_seed_batches = 4\nrandom_seed = 77\n", encoding="utf-8")
    values = {v for _loc, v in seed_records(root).code_seeds}
    assert 77 in values and 1 not in values and 4 not in values


# ---- the second review (P1-0): FI cannot tell, or a fixed seed drives the trials too ---------------------------------------

_KEPT_ON_OBJECT = ("import numpy as np\nclass Sim:\n    def __init__(self, seed):\n        self.seed = seed\n"
                   "    def go(self):\n        return np.random.default_rng(self.seed).normal()\n"
                   "def run_trial(cell, trial_id, seed):\n    return {'x': Sim(seed).go()}\n")
_PARTLY = ("from model import sim\nimport numpy as np\ndef run_trial(cell, trial_id, seed):\n"
           "    rng = np.random.default_rng(seed)\n    return sim(cell, rng)\n")
_BASE_SEED_CLAIMS = ["Each trial's seed is derived from FI's base seed and the trial number.",
                     "Trials were seeded with 0 plus the trial index.",
                     "Each trial used base seed 0 plus its index."]


def _seeded_quest(tmp_path: Path, simulate: str, extra: dict[str, str] | None = None) -> Path:
    (tmp_path / "code").mkdir(parents=True)
    (tmp_path / "code" / "simulate.py").write_text(simulate, encoding="utf-8")
    for name, text in (extra or {}).items():
        (tmp_path / "code" / name).write_text(text, encoding="utf-8")
    (tmp_path / "code" / "study.json").write_text(json.dumps({"base_seed": 0}), encoding="utf-8")
    return tmp_path


def test_when_fi_cannot_tell_a_claim_that_its_base_seed_drives_the_trials_is_flagged(tmp_path: Path) -> None:
    from core import run_facts

    root = _seeded_quest(tmp_path, _KEPT_ON_OBJECT)
    assert run_facts.seed_verdict(root)["verdict"] == "unknown"
    seeds = seed_records(root)
    for sentence in _BASE_SEED_CLAIMS:
        assert _seed_findings(f"# P\n\n{sentence}\n", seeds), sentence
    assert not _seed_findings("# P\n\nThe study ran 9 settings.\n", seeds)


def test_when_a_fixed_seed_drives_the_trials_too_a_base_seed_claim_is_flagged(tmp_path: Path) -> None:
    from core import run_facts

    root = _seeded_quest(tmp_path, _PARTLY, {"model.py": "import numpy as np\ndef sim(cell, rng):\n"
                                                         "    return np.random.default_rng(7).normal()\n"})
    assert run_facts.seed_verdict(root)["verdict"] == "partly"
    seeds = seed_records(root)
    for sentence in _BASE_SEED_CLAIMS:
        assert _seed_findings(f"# P\n\n{sentence}\n", seeds), sentence


def test_a_seed_in_a_table_is_read_as_its_column_s_words(tmp_path: Path) -> None:
    """Second review (P1-3): a seed in a table row was never checked; the same number in a sentence was."""
    seeds = seed_records(_quest(tmp_path))
    two_cols = "## Methods\n\n| Quantity | Value |\n|---|---|\n| Random seed | 5 |\n| Trials | 400 |\n"
    assert _seed_findings(two_cols, seeds), "a two-column row reads 'Random seed: 5'"
    by_header = "## Methods\n\n| Setting | Seed | Trials |\n|---|---|---|\n| A | 5 | 400 |\n"
    assert _seed_findings(by_header, seeds), "a cell reads with its column's header: 'Seed: 5'"
    true_row = "## Methods\n\n| Quantity | Value |\n|---|---|\n| Trials | 400 |\n| Settings | 27 |\n"
    assert not _seed_findings(true_row, seeds)
