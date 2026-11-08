"""Which findings may end a quest, by where the finding comes from (no model call, no reading of its words here).

A check says, where it makes a finding, what the finding rests on:

- :data:`MEASURED`: FI computed it itself, from its own record of the run or from running code (the trial ledger, a re-run,
  arithmetic on recorded numbers, a reply that does not hold what was asked, a script that does not load);
- :data:`AGREED`: two independent sources agree (a reading of the code and a test FI ran that fails by its assertion);
- :data:`READ`: one reading, of the script's text, of free text, of a name, or of one model's answer;
- :data:`COPY`: FI ran the code on a small copy of the recorded results and saw it fail (the analysis raised an error, a
  figure drew nothing). That is FI's own observation, not a reading; but a small copy can lack the trials a step needs, so
  it is not yet a fact of the full run.

A finding that is measured or agreed may end the quest when its repairs are used up. A finding that rests on one reading, or
on a run of a small copy, may still be sent back for its repairs, but what stands after them does not end the quest: it is
recorded (run.log, QUALITY.md) and the paper says it in its limitations, with what it rests on, not as a fact of the run. A reading can mistake
one list for another, or a name for a quantity; ending a quest on it stopped correct studies, while the faults it was
meant to find are the ones FI measures from its own record of the trials.
"""

from __future__ import annotations

MEASURED = "measured"
AGREED = "agreed"
READ = "read"
COPY = "copy"
SOURCES = (MEASURED, AGREED, READ, COPY)

#: How QUALITY.md and run.log name what a finding rests on.
_WORDS = {
    MEASURED: "measured by FI from the run's own record",
    AGREED: "a reading of the code that a test FI ran confirms",
    READ: "FI's reading of the code (not confirmed by the run's own record)",
    COPY: "measured by FI on a small copy of the run's own record (the full run was not yet seen)",
}


def may_stop(source: object) -> bool:
    """Whether a finding from ``source`` may end the quest. An unknown or missing source may not: a stop must say what
    it rests on."""
    return source in (MEASURED, AGREED)


def source_words(source: object) -> str:
    """What the finding rests on, in plain words."""
    return _WORDS.get(str(source), _WORDS[READ])
