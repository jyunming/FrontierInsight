"""Planting an error in a copy of a recorded benchmark run.

Each planter changes one thing and writes ``bench/plant.json`` saying what, so the scorer and a reader know exactly what
the run was given. A planter changes a file of the copied quest (``edit``), or names a recorded model answer to replace
(``answer``: the replacement is served by the replay client, core/replay.py), or prepares the inputs of a fresh run
(``input``). Which step the copy is then run again from is the catalogue's (catalogue.py).

The MVP planters: R1 (a number of the paper changed), R2 (a sentence with a number the run never computed added to the
writer's answer), N3 (a factor taken out of the simulation), S1 (the analysis keeps one value in five of FI's trials) and
L1 (a retracted paper pinned as a source, with its recorded Crossref answer).
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from core.replay import Recording

from . import catalogue

PLANT_FILE = "plant.json"
# A number worth changing: three or more significant digits, not a year, not a reference or figure number.
_NUMBER = re.compile(r"(?<![\w.\[])(\d+\.\d{2,})(?![\w.\]])")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""


def write_record(bench_dir: Path, record: dict[str, Any]) -> Path:
    bench_dir.mkdir(parents=True, exist_ok=True)
    path = bench_dir / PLANT_FILE
    path.write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
    return path


def read_record(bench_dir: Path) -> dict[str, Any] | None:
    path = Path(bench_dir) / PLANT_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def edit_file(quest_root: Path, rel: str, find: str, replace: str) -> dict[str, Any]:
    """Replace the first ``find`` in ``quest_root/rel`` with ``replace``; refuses when ``find`` is not there."""
    path = Path(quest_root) / rel
    text = path.read_text(encoding="utf-8")
    if find not in text:
        raise ValueError(f"{rel} does not contain {find!r}: nothing was planted")
    before = _sha(path)
    path.write_text(text.replace(find, replace, 1), encoding="utf-8")
    return {"file": rel, "find": find, "replace": replace, "sha256_before": before, "sha256_after": _sha(path)}


def _perturbed(token: str) -> str:
    """A different number of the same shape: x 1.37, written with the same number of decimals."""
    decimals = len(token.split(".")[1]) if "." in token else 0
    value = float(token) * 1.37
    return f"{value:.{decimals}f}"


def paper_files(quest_root: Path) -> list[str]:
    """The paper's Markdown (paper/paper.md, and the top-level copy when there is one)."""
    return [rel for rel in ("paper/paper.md", "paper.md") if (Path(quest_root) / rel).is_file()]


def plant_r1(quest_root: Path, *, find: str | None = None, replace: str | None = None) -> dict[str, Any]:
    """R1: one number of the paper changed. Without ``find``, the first number with two or more decimals in the paper's
    text before its references is multiplied by 1.37."""
    rels = paper_files(quest_root)
    if not rels:
        raise ValueError("the quest has no paper to plant a number in")
    text = (Path(quest_root) / rels[0]).read_text(encoding="utf-8")
    if find is None:
        body = re.split(r"(?im)^#+\s*references\b", text)[0]
        m = _NUMBER.search(body)
        if not m:
            raise ValueError("the paper states no number with two or more decimals to change; give --find/--replace")
        find = m.group(1)
        replace = _perturbed(find)
    if replace is None:
        raise ValueError("give the number that replaces it")
    edits = [edit_file(quest_root, rel, find, replace) for rel in rels]
    return {"error": "R1", "how": "edit", "from_step": catalogue.get("R1").from_step, "edits": edits,
            "planted_value": replace, "original_value": find}


#: R2's planted sentence: a precise number nothing in the run computed.
R2_SENTENCE = "In a supplementary sweep the effect reached 7.413 (95% CI 7.208 to 7.618)."


def plant_r2(recording: Recording, *, node: str = "write", index: int = 1,
             sentence: str = R2_SENTENCE) -> tuple[dict[str, Any], dict[tuple[str, int], str]]:
    """R2: the writer's answer with a sentence whose number the results have no source for, added after its first
    paragraph of results (or its first paragraph). Returns the record and the replacement answer by call."""
    recorded = recording.get(node, index)
    if recorded is None:
        raise ValueError(f"the recording has no call {node}#{index} to replace")
    text = str(recorded["response"])
    parts = re.split(r"(\n\s*\n)", text, maxsplit=2)
    if len(parts) >= 3:
        new = parts[0] + parts[1] + sentence + "\n\n" + "".join(parts[2:])
    else:
        new = text.rstrip() + "\n\n" + sentence + "\n"
    record = {"error": "R2", "how": "answer", "from_step": catalogue.get("R2").from_step,
              "call": f"{node}#{index}", "added": sentence,
              "sha256_before": hashlib.sha256(text.encode("utf-8")).hexdigest(),
              "sha256_after": hashlib.sha256(new.encode("utf-8")).hexdigest()}
    return record, {(node, index): new}


def plant_code(error: str, quest_root: Path, *, file: str, find: str, replace: str) -> dict[str, Any]:
    """N1, N3, N4, S1, S4: a change to a file under ``code/`` (the simulation, its model package, or the analysis)."""
    entry = catalogue.get(error)
    if entry.how != "edit" or not entry.where.startswith("code/") or not file.replace("\\", "/").startswith("code/"):
        raise ValueError(f"{error} is planted as {entry.how} in {entry.where}, not as an edit of {file}")
    return {"error": entry.id, "how": "edit", "from_step": entry.from_step,
            "edits": [edit_file(quest_root, file, find, replace)]}


#: The retracted paper L1 plants by default: Wakefield et al. 1998, retracted by The Lancet in 2010. The text is the
#: published abstract's opening and its central sentence.
WAKEFIELD = {
    "doi": "10.1016/S0140-6736(97)11096-0",
    "title": "Ileal-lymphoid-nodular hyperplasia, non-specific colitis, and pervasive developmental disorder in children",
    "year": 1998, "venue": "The Lancet",
    "content": ("We investigated a consecutive series of children with chronic enterocolitis and regressive developmental "
                "disorder. Onset of behavioural symptoms was associated, by the parents, with measles, mumps, and "
                "rubella vaccination in eight of the 12 children. We identified associated gastrointestinal disease "
                "and developmental regression in a group of previously normal children, which was generally "
                "associated in time with possible environmental triggers."),
}


#: L1's control: the same kind of search result, a paper that was never retracted (Madsen et al. 2002).
MADSEN = {
    "doi": "10.1056/NEJMoa021134",
    "title": "A population-based study of measles, mumps, and rubella vaccination and autism",
    "year": 2002, "venue": "The New England Journal of Medicine",
    "content": ("We conducted a retrospective cohort study of all children born in Denmark from January 1991 through "
                "December 1998. The relative risk of autistic disorder in the group of vaccinated children, as compared "
                "with the unvaccinated group, was 0.92. This study provides strong evidence against the hypothesis "
                "that MMR vaccination causes autism."),
}


def plant_l1(*, paper: dict[str, Any] | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """L1: a retracted paper among the literature search's results of a fresh run (the run is not forked: see the
    catalogue). Returns the record and the search hit to add (``BenchEngine(extra_hits=...)``).

    The hit carries its DOI, as a search result does. A paper pinned with ``knowledge.local_papers`` or dropped into
    ``inputs/papers/`` is not a way to plant it: FI keeps no DOI for those (core/knowledge.py::_load_local_paper,
    core/engine.py::_ingest_user_dropped_papers), so no retraction lookup is made for them at all."""
    paper = dict(paper or WAKEFIELD)
    hit = {"content": paper["content"], "metadata": {
        "title": paper["title"], "doi": paper["doi"], "year": paper.get("year"), "venue": paper.get("venue"),
        "source": "crossref"}}
    record = {"error": "L1", "how": "input", "from_step": None, "doi": paper["doi"], "title": paper["title"],
              "sha256": hashlib.sha256(json.dumps(hit, sort_keys=True).encode("utf-8")).hexdigest()}
    return record, [hit]
