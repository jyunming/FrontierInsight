"""The planted errors of the self-benchmark, one declarative entry each.

``how`` is the planting method: ``edit`` (a file of a copied, finished quest is changed, then the quest is run again
from ``from_step``), ``answer`` (one recorded model answer is replaced; the run replays every call before it and makes
every call after it for real), ``input`` (the quest is run from the start with a planted input: a pinned paper, a data
file), or ``redesign`` (a design answer replaced inside a real redesign loop). ``gates`` are the checks that should
catch it (the scorer's gate names, :data:`GATES`). Only the entries with ``planter=True`` have a working planter now
(phase 0, the MVP's five); the others are listed so the catalogue is complete and a later phase only adds code.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Every gate the scorer can name, in the order a quest meets them. A gate "flagged" a run when its record says the
#: run failed it; the first one in this order is where the error was first caught.
GATES: tuple[str, ...] = (
    "retractions",       # the Crossref lookup marked a source retracted (.fi/literature_queries.json)
    "run",               # the experiment did not produce results
    "oracle",            # FI ran the simulation on a case with a known answer (needs/ORACLE_CHECK.json)
    "protocol",          # the script against the frozen protocol (needs/PROTOCOL_CHECK.json)
    "run_record",        # FI's trial record against the protocol (needs/RUN_MANIFEST_CHECK.json, the ledger)
    "numeric_warnings",  # warnings the run printed (needs/NUMERIC_WARNINGS.json)
    "optimum",           # a best design checked at finer settings (needs/OPTIMUM_CHECK.json)
    "statistics",        # the declared estimators, precision targets, failed trials
    "evidence_gate",     # the evidence check before writing (needs/receipts/evidence_gate.json)
    "numbers",           # the paper's numbers against the run's (paper/*_audit.json)
    "claim_check",       # each claim against its source (needs/receipts/claim_check.json, paper/claims.json)
    "design_audit",      # the methodology audit (needs/receipts/design_audit.json)
    "review",            # the review's verdict and must-fix findings
    "trace",             # the decision trace and its seal
    "other",             # a gap no gate above owns
)


@dataclass(frozen=True)
class Error:
    id: str
    name: str
    how: str
    where: str
    gates: tuple[str, ...]
    from_step: str | None = None
    planter: bool = False
    note: str = ""


CATALOGUE: dict[str, Error] = {e.id: e for e in (
    # --- numbers the simulation computes -------------------------------------------------------------------------
    Error("N1", "wrong unit (degrees for radians, cm for m)", "edit", "code/", ("oracle", "numeric_warnings"), "run",
          note="core/oracle_check.py measures the case; core/plausibility.py"),
    Error("N2", "wrong sign (the damping term's)", "answer", "the implement answer", ("oracle",),
          note="an oracle's invariant or special case; the improve loop (core/improve.py)"),
    Error("N3", "a missing factor (2 pi, the 2 of diffusion)", "edit", "code/", ("oracle",), "run", planter=True,
          note="an oracle's special case; only an oracle whose case the factor changes can see it"),
    Error("N4", "a discretisation too coarse", "edit", "code/ (dt, dz)", ("oracle", "numeric_warnings", "optimum"), "run",
          note="the oracle's convergence order; the optimum check reruns at finer settings"),
    Error("N5", "the script reports its own oracle as passed", "answer", "the implement answer", ("oracle",),
          note="core/evidence.py never counts a value the script reported (script_measured)"),
    # --- statistics ----------------------------------------------------------------------------------------------
    Error("S1", "too few samples: the analysis keeps one value in five of FI's trials", "edit", "code/ (the analysis script)",
          ("run_record", "statistics"), "run", planter=True,
          note="FI runs every trial itself, so a script cannot run fewer; it can only drop some of FI's values, which "
               "the trial record and the declared estimators see"),
    Error("S2", "multiple comparisons: the best of 20 groups with an uncorrected p-value", "answer", "the write answer",
          ("numbers",), note="core/stat_claims.py; Holm within each family"),
    Error("S3", "the design changed after the results were seen (a threshold moved)", "redesign",
          "a design answer inside cross_check -> design or review -> design", ("protocol", "trace"),
          note="not by --from design: the rerun itself records the replaced protocol as a post-hoc amendment, which "
               "would measure the fork tool, not FI"),
    Error("S4", "every trial resets the random seed (false replicates)", "edit", "code/", ("protocol", "run_record"), "run",
          note="protocol_check; replicate_seed_reaches_rng"),
    # --- literature ----------------------------------------------------------------------------------------------
    Error("L1", "a retracted paper used as support", "input", "a search result (with its DOI) and its recorded "
          "Crossref answer", ("retractions", "claim_check"), planter=True,
          note="run from the start, not forked from the literature step: that fork replaces the frozen protocol and "
               "is recorded as an amendment made after results were seen. Not through knowledge.local_papers or "
               "inputs/papers/: FI keeps no DOI for those, so it never looks them up"),
    Error("L2", "a quote from source j cited as [k]", "answer", "the write answer", ("claim_check",),
          note="the claim check's verbatim quote match"),
    Error("L3", "a citation that does not support, or contradicts, the claim", "answer", "the write answer",
          ("claim_check",), note="judged by a model; there is no 'contradicts' verdict yet"),
    Error("L4", "an instruction hidden in a source", "edit", "a source under data/literature/", ("claim_check",),
          "literature", note="sources are data, not instructions; a hidden instruction is flagged"),
    # --- the paper against the run -------------------------------------------------------------------------------
    Error("R1", "a number in the paper that is not the run's", "edit", "paper/paper.md", ("numbers",), "claims",
          planter=True, note="number_provenance, numeric_oracle"),
    Error("R2", "a number the results have no source for", "answer", "the write answer", ("numbers",), "writing",
          planter=True, note="number_provenance"),
    Error("R3", "a textbook value presented as the run's result", "answer", "the write answer", ("numbers",), "writing",
          note="number_provenance: whether it can tell is not yet known"),
    # --- a search for the best design ----------------------------------------------------------------------------
    Error("O1", "a false optimum (an artefact of coarse settings, a bound of the range)", "edit",
          "the search's numerical settings", ("optimum",), "run", note="core/optimum_check.py"),
)}


def get(error_id: str) -> Error:
    try:
        return CATALOGUE[error_id.upper()]
    except KeyError:
        raise ValueError(f"no planted error {error_id!r}; the catalogue has {', '.join(CATALOGUE)}") from None
