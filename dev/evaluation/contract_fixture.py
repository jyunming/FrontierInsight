"""Build a scratch quest folder from a recorded raw result and a contract-form design (made by ``plan_parts_ab``), so FI's
analysis can be run on it without touching the recorded quest.

    python -m dev.evaluation.contract_fixture <recorded quest folder> <plan_parts_ab output folder> <scratch folder>

The scratch folder gets: ``raw/trials.json`` (and ``raw/trials.analysis.json`` when the quest has one), ``config.yaml``,
``design.json`` and ``contract.json`` (copied from the plan_parts_ab output), ``.fi/run_facts.json`` and
``.fi/trial_checks.json`` when the recorded quest has them, and ``fixture.json`` (where each file came from). The recorded
quest is only read.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

RAW_FILES = ("raw/trials.json", "raw/trials.analysis.json")
OPTIONAL_FILES = ("config.yaml", ".fi/run_facts.json", ".fi/trial_checks.json")
DESIGN_FILES = ("design.json", "contract.json")


def find_design_folder(out: Path) -> Path:
    """The folder holding ``design.json``: ``out`` itself, or the one folder below it that has it."""
    if (out / "design.json").is_file():
        return out
    found = sorted(p.parent for p in out.rglob("design.json"))
    if len(found) != 1:
        raise SystemExit(f"{out}: expected one design.json, found {len(found)}")
    return found[0]


def build(quest: Path, design_out: Path, scratch: Path) -> dict[str, str]:
    """Copy the raw, the design and the optional records into ``scratch``; return {scratch file: source file}."""
    quest, scratch = quest.resolve(), scratch.resolve()
    if scratch == quest or quest in scratch.parents:
        raise SystemExit("the scratch folder must be outside the recorded quest")
    if not (quest / "raw" / "trials.json").is_file():
        raise SystemExit(f"{quest}: no raw/trials.json")
    folder = find_design_folder(design_out)
    design = json.loads((folder / "design.json").read_text(encoding="utf-8"))
    protocol = design.get("protocol") if isinstance(design, dict) else None
    if not isinstance(protocol, dict) or protocol.get("form") != "contract":
        print(f"warning: {folder / 'design.json'} is not in the contract form", file=sys.stderr)
    copied: dict[str, str] = {}
    for rel in RAW_FILES + OPTIONAL_FILES:
        src = quest / rel
        if src.is_file():
            dst = scratch / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied[rel] = str(src)
    for rel in DESIGN_FILES:
        src = folder / rel
        if src.is_file():
            scratch.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, scratch / rel)
            copied[rel] = str(src)
    (scratch / "fixture.json").write_text(json.dumps(copied, indent=1), encoding="utf-8")
    return copied


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("quest", type=Path)
    ap.add_argument("design_out", type=Path)
    ap.add_argument("scratch", type=Path)
    a = ap.parse_args()
    print(json.dumps(build(a.quest, a.design_out, a.scratch), indent=1))


if __name__ == "__main__":
    main()
