"""Re-run the behaviour probes on the stored code of a finished quest, without a model and without touching the quest.

    python dev/probe_replay.py <quest_dir> --relations tests/fixtures/probes/<name>.json [--seeds N] [--mutants]
    python dev/probe_replay.py <quest_dir> --saturation-only          # the generic test on the study's own stored numbers

The quest's code, protocol and stored study numbers are copied to a temporary folder first; nothing is written into the
quest. The relations file is ``{"metric_alias": {protocol id: name run_trial returns}, "relations": [<relation>, ...]}``
(the same objects a reader model replies with; see ``agents/probe_relations.md``), hand-checked against the plan.

With ``--mutants`` every faulty copy of ``code/simulate.py`` that :func:`core.probes.mutants` can make (a setting never
read, a metric hard-coded, clipped to a bound, a sign swapped, a refinement knob made the same at every level) is run
through the same probes, and the report says which were caught (a ``broken`` relation or saturation) and which relation
caught it: the catch rate FI needs before it lets a probe stop a quest.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import probes as P  # noqa: E402
from core import probes_run as PR  # noqa: E402
from core.execution import SharedInterpreterExecutor  # noqa: E402

_COPY = ("code", "needs", "raw")


def _protocol(quest: Path, fixture: dict[str, Any] | None = None) -> dict[str, Any]:
    """The quest's frozen protocol. ``fixture["metrics_extra"]``: names ``run_trial`` returns that the protocol does not
    list as metrics but a hand-written relation is about (added so ``parse`` accepts them)."""
    data = json.loads((quest / "needs" / "FROZEN_PROTOCOL.json").read_text(encoding="utf-8"))
    protocol = dict(data.get("protocol", data))
    extra = [{"id": m, "kind": "mean"} for m in (fixture or {}).get("metrics_extra", [])]
    protocol["metrics"] = list(protocol.get("metrics") or []) + extra
    return protocol


def _copy(quest: Path, dest: Path) -> Path:
    for name in _COPY:
        if (quest / name).is_dir():
            shutil.copytree(quest / name, dest / name, ignore=shutil.ignore_patterns("__pycache__", ".git"), dirs_exist_ok=True)
    if (quest / "data" / "results").is_dir():  # what the analysis reported: saturation judges only reported quantities
        shutil.copytree(quest / "data" / "results", dest / "data" / "results", dirs_exist_ok=True)
    (dest / ".fi").mkdir(exist_ok=True)
    return dest


def _line(rid: str, res: dict[str, Any]) -> str:
    return f"  {rid:<28} {res['verdict']:<12} {res['says']}"


async def _replay(quest: Path, fixture: dict[str, Any], seeds: int, budget_s: float,
                  edits: dict[str, str] | None = None, base_seed: int = 0) -> dict[str, Any]:
    """``edits``: path under ``code/`` -> the text to put there (a mutant), in the temporary copy only."""
    protocol = _protocol(quest, fixture)
    relations, dropped = P.parse({"relations": fixture.get("relations", [])}, protocol)
    with tempfile.TemporaryDirectory(prefix="fi_probe_") as tmp:
        root = _copy(quest, Path(tmp))
        for rel, text in (edits or {}).items():
            (root / "code" / rel).write_text(text, encoding="utf-8")
        ex = SharedInterpreterExecutor()
        rec = await PR.run(ex, sys.executable, root, protocol, relations, seeds=seeds, budget_s=budget_s,
                           metric_alias=fixture.get("metric_alias"), dropped=dropped, base_seed=base_seed, write=False)
    rec["_dropped"] = dropped
    return rec


def _only(rec: dict[str, Any]) -> dict[str, Any]:
    (sha, res), = rec["results"].items()
    return res


async def _mutants(quest: Path, fixture: dict[str, Any], seeds: int, budget_s: float) -> list[dict[str, Any]]:
    protocol = _protocol(quest, fixture)
    grid, _ = P._protocol_parts(protocol)
    relations, _d = P.parse({"relations": fixture.get("relations", [])}, protocol)
    alias = fixture.get("metric_alias") or {}
    source = (quest / "code" / "simulate.py").read_text(encoding="utf-8")
    settings = {s: P._base(grid)[s] for s in grid}
    metrics = sorted({alias.get(r.metric, r.metric) for r in relations})
    knobs = {}
    for r in relations:
        if r.kind == "convergence" and r.setting in grid:
            vs = P._sorted_values(grid[r.setting])
            knobs[r.setting] = vs[0] if r.refine == "decrease" else vs[-1]
    todo: list[tuple[str, dict[str, str]]] = [(n, {"simulate.py": t}) for n, t in P.mutants(source, settings, metrics, knobs)]
    for extra in fixture.get("extra_mutants", []):   # hand-written: {"name", "file" (under code/), "find", "replace"}
        path = quest / "code" / extra.get("file", "simulate.py")
        text = path.read_text(encoding="utf-8")
        if extra["find"] not in text:
            print(f"  {extra['name']:<40} skipped: the text to change is not in {path.name}")
            continue
        todo.append((extra["name"], {extra.get("file", "simulate.py"): text.replace(extra["find"], extra["replace"], 1)}))
    out = []
    for name, edits in todo:
        rec = await _replay(quest, fixture, seeds, budget_s, edits=edits)
        res = _only(rec)
        broken = [k for k, v in res.items() if v["verdict"] == "broken" and k != "saturation"]
        unc = [k for k, v in res.items() if v["verdict"] == "unconfirmed" and not k.startswith("saturation")]
        out.append({"mutant": name, "caught_by": broken, "caught": bool(broken), "unconfirmed": unc})
        print(f"  {name:<40} {'CAUGHT by ' + ', '.join(broken) if broken else 'missed'}"
              + (f"   (could not check: {', '.join(unc)})" if unc and not broken else ""), flush=True)
    return out


async def main_async(args: argparse.Namespace) -> int:
    quest = Path(args.quest).resolve()
    if args.saturation_only:
        protocol = _protocol(quest)
        grid, _ = P._protocol_parts(protocol)
        cells = PR.stored_cells(quest)
        print(f"{quest.name}: {len(cells)} stored cells")
        verdicts = PR.saturation_of(cells, quest, protocol, grid, PR.reported_names(quest, protocol))
        for v in verdicts:
            print(_line(v.relation_id, v.to_dict()))
        broken = [v for v in verdicts if v.verdict == "broken"]
        print(f"saturation fired: {len(broken)} broken, {len(verdicts) - len(broken)} unconfirmed" if verdicts
              else "saturation did not fire")
        return 0
    fixture = json.loads(Path(args.relations).read_text(encoding="utf-8"))
    if args.repeat > 1:   # the false-alarm rate on good code: the same relations at other seeds
        total = flagged = 0
        for b in range(args.repeat):
            res = _only(await _replay(quest, fixture, args.seeds, args.budget, base_seed=b))
            bad = [k for k, v in res.items() if v["verdict"] == "broken" and k != "saturation"]
            unc = [k for k, v in res.items() if v["verdict"] == "unconfirmed" and not k.startswith("saturation")]
            total += len([k for k in res if not k.startswith("saturation")])
            flagged += len(bad)
            print(f"  base seed {b}: broken {bad or 'none'}; unconfirmed {unc or 'none'}", flush=True)
        print(f"{flagged} broken of {total} relation checks over {args.repeat} seeds")
        return 1 if flagged else 0
    rec = await _replay(quest, fixture, args.seeds, args.budget)
    res = _only(rec)
    print(f"{quest.name}: {len(rec['relations'])} relations ({len(rec['_dropped'])} dropped: {rec['_dropped']}), "
          f"{rec.get('seconds_spent')} s of cells")
    for rid, r in res.items():
        print(_line(rid, r))
    broken = [k for k, r in res.items() if r["verdict"] == "broken"]
    print(f"broken: {len(broken)}  {broken}")
    if args.mutants:
        print("mutants:")
        got = await _mutants(quest, fixture, args.seeds, args.budget)
        caught = sum(m["caught"] for m in got)
        print(f"caught {caught} of {len(got)} ({100 * caught / max(1, len(got)):.0f}%)")
    return 1 if broken else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("quest")
    ap.add_argument("--relations")
    ap.add_argument("--seeds", type=int, default=P.DEFAULT_SEEDS)
    ap.add_argument("--budget", type=float, default=PR.DEFAULT_BUDGET_S, help="seconds of cell time per run")
    ap.add_argument("--mutants", action="store_true")
    ap.add_argument("--repeat", type=int, default=1, help="run the relations at this many base seeds and count false broken")
    ap.add_argument("--saturation-only", action="store_true")
    args = ap.parse_args()
    if not args.saturation_only and not args.relations:
        ap.error("--relations is required unless --saturation-only")
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
