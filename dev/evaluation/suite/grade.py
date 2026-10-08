"""The grader's side of a round: the prompt for one delivered paper, and how a finished grade enters ``grades.json``.

The grader (a separate step, by a person or another model) writes its report as a Markdown file with the rubric's table
row (see ``dev/evaluation/benchmark/rubric.md``). :func:`import_grade` reads that row into ``grades.json`` so the
collector and the trend read one small structured file, not prose.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .collect import GRADES, load_grades

RUBRIC = Path(__file__).resolve().parents[1] / "benchmark" / "rubric.md"
SCORES = Path(__file__).resolve().parents[1] / "benchmark" / "scores.md"

PROMPT = """You are grading ONE FrontierInsight (FI) quest by hand with FI's benchmark rubric. Read-only on the quest; write only your report file. Do not modify any repository. Do not run the quest again.

Rubric: read `{rubric}` in full and follow it exactly (paper D1-D6, gates G1-G3, slides S1-S6, poster P0-P6, the grading steps 1-5). Also read the first table rows of `{scores}` to see how a graded row looks.

Quest: `{quest}`
Round: {round}, topic: {topic}.

The topic is the one in the quest's `config.yaml` (`topic:` block), reproduced here:

----- TOPIC (verbatim) -----
{topic_text}
----- END TOPIC -----

Adapt the three topic-specific items of the rubric:
- D3 Canonical works: score it only if the topic above names canonical works; otherwise D3 is N/A, left out of the mean (mean over the 5 others, sum of round(Di) / 5), and you say so. Separately note, for information only, whether the paper cites a standard reference for the topic's central relation.
- D5 Topic goals met: the goals are the topic's own SCOPE, GOALS and CONSTRAINTS above. Number them yourself first (one goal per stated deliverable, figure, comparison, limit and reporting duty), write the numbered list in your report, then count how many the paper meets.
- G1 Numbers correct: compute the reference values yourself from the formulas and reference values the topic states in its CONTEXT, SCOPE and GOALS (write down what you computed and with what), then check the paper's numbers against those AND against the run's own results (`raw/`, `data/`, `run_output/`, `.fi/run.log`, RESULT_JSON in `.fi/state.sqlite` if needed). A textbook value printed as the paper's own result, a value the run never computed, or results missing because the experiment crashed or never ran fail G1. Where the topic states no reference value, say so and judge G1 on the run's results alone.

Sources for D1: the retrieved sources are wherever this quest stored them (look under the quest folder: `literature/`, `data/`, `needs/*.json` with source records, `paper/` claim files such as `claims.json` / `CLAIMS.md`, `.fi/`). Judge each in-text citation against the retrieved text, never against memory. If a source's text was not retrieved, it does not support the sentence.

Read `paper.pdf` (render pages to images if you need to see layout and figures; `pdftoppm` or Python `fitz` may be available; if not, read `paper.md` and the figure PNGs in `figures/`), `slides.pdf`/`slides.pptx`, and `poster.pdf` (if missing, P0 = 0 for every P item).

Write your report to `{report}` with:
1. A one-line table row in the scores.md format: run | date graded | FI build | model / arm | D1 | D2 | D3 | D4 | D5 | D6 | Mean | G1 | G2 | G3 | pages | tokens / calls | note. G1, G2, G3 are the words `pass` or `fail` (you may add a short qualifier after it).
2. Slides S1-S6 and their mean; poster P1-P6 (both means).
3. Item-level record: every item that decided each score, and every place where a reasonable grader could decide otherwise ("alternative: ...").
4. FI faults (a check that passed something wrong, or flagged something right) listed separately from the model's own mistakes.
5. If, and only if, the quest was an honest stop, a line `STOP: correct` or `STOP: wrong` saying whether the cause FI gave for stopping was true of the run, with the evidence.
Be concise and factual; do not inflate. Finish by replying with just the table row and the 3 most important faults.
"""


def prompt_for(*, quest: str, round_name: str, topic: str, topic_text: str, report: str) -> str:
    return PROMPT.format(rubric=RUBRIC, scores=SCORES, quest=quest, round=round_name, topic=topic,
                         topic_text=topic_text.rstrip(), report=report)


def write_templates(round_dir: Path, topic_texts: dict[str, str]) -> list[Path]:
    """One ``GRADER_PROMPT.md`` per delivered paper of a collected round (``<round>/<topic>/GRADER_PROMPT.md``)."""
    rdir = Path(round_dir)
    data = json.loads((rdir / "round.json").read_text(encoding="utf-8"))
    written = []
    for t in data["topics"]:
        if t["outcome"] != "paper" or not t.get("folder"):
            continue
        out = rdir / t["topic"] / "GRADER_PROMPT.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        text = topic_texts.get(t["topic"]) or _topic_from_config(Path(t["folder"]))
        out.write_text(prompt_for(quest=t["folder"], round_name=data["round"], topic=t["topic"], topic_text=text,
                                  report=str(rdir / t["topic"] / "GRADE.md")), encoding="utf-8")
        written.append(out)
    return written


def _topic_from_config(quest: Path) -> str:
    try:
        import yaml
        return str((yaml.safe_load((quest / "config.yaml").read_text(encoding="utf-8")) or {}).get("topic") or "")
    except Exception:  # noqa: BLE001
        return ""


# ------------------------------------------------------------------------------------------------ import

_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def parse_row(text: str) -> dict[str, Any]:
    """The grade in a report's table row: the first table row with three consecutive pass/fail cells (G1, G2, G3)."""
    for line in text.splitlines():
        if not line.lstrip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        for i in range(len(cells) - 2):
            trio = cells[i:i + 3]
            if all(re.match(r"(pass|fail)", c, re.I) for c in trio):
                mean = _NUM.search(cells[i - 1]) if i >= 1 else None
                ds = {f"D{n}": (_NUM.search(cells[i - 8 + n]).group(0) if _NUM.search(cells[i - 8 + n]) else None)
                      for n in range(1, 7)} if i >= 7 else {}
                return {"g1": trio[0], "g2": trio[1], "g3": trio[2], "mean": float(mean.group(0)) if mean else None,
                        "d": ds, "notes": cells[-1] if len(cells) > i + 3 else ""}
    return {}


def import_grade(round_dir: Path, topic: str, report_file: Path, *, stop_correct: bool | None = None,
                 cause_class: str | None = None, stage: str | None = None, note: str | None = None) -> dict[str, Any]:
    """Read a grader's report into ``grades.json`` for one topic. A report that is a stop verdict (``STOP: wrong``) sets
    ``stop_correct``; ``cause_class`` is a taxonomy class the grader sets by hand (the only way to record an FI bug)."""
    text = Path(report_file).read_text(encoding="utf-8")
    entry = parse_row(text)
    m = re.search(r"^\s*STOP:\s*(correct|wrong)", text, re.I | re.M)
    if m and stop_correct is None:
        stop_correct = m.group(1).lower() == "correct"
    if stop_correct is not None:
        entry["stop_correct"] = stop_correct
    if cause_class:
        entry["class"] = cause_class
        entry["stage"] = stage or ""
        entry["class_note"] = note or ""
    if not entry:
        raise ValueError(f"no grade row (three pass/fail cells) and no STOP verdict in {report_file}")
    entry["report"] = str(report_file)
    grades = load_grades(round_dir)
    grades[topic] = {**grades.get(topic, {}), **entry}
    Path(round_dir).mkdir(parents=True, exist_ok=True)
    (Path(round_dir) / GRADES).write_text(json.dumps(grades, indent=1) + "\n", encoding="utf-8")
    return grades[topic]
