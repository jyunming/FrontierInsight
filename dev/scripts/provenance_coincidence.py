"""Measure how blind the "unsourced number" check is on stored quests.

For every stored quest that has a paper and recorded results, pair each paper with
each quest's results:

* same quest  -> numbers the check FLAGS (an honest paper should flag ~none),
* other quest -> numbers the check ACCEPTS although they cannot be sourced there
  (pure coincidence; the lower the better).

Usage:  python dev/scripts/provenance_coincidence.py [ROUNDS_DIR] [--cache FILE]
Reads each quest's LangGraph checkpoint (.fi/state.sqlite) read-only.
"""
from __future__ import annotations

import argparse
import glob
import pickle
import sqlite3
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core import number_provenance as npv  # noqa: E402


def load_quest(d: str) -> dict | None:
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    paper = Path(d) / "paper.md"
    db = Path(d) / ".fi" / "state.sqlite"
    if not paper.exists() or not db.exists():
        return None
    c = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    row = c.execute("select type,checkpoint from checkpoints order by rowid desc limit 1").fetchone()
    st = JsonPlusSerializer().loads_typed((row[0], row[1]))["channel_values"]
    if not st.get("result_json"):
        return None
    from core import engine as E

    reps = list(st.get("result_json_replicates") or [])
    kw = dict(result_json=st.get("result_json") or {}, replicates=reps,
              figure_records=st.get("figure_records") or {}, design=st.get("design") or {}, n_seeds=len(reps))
    try:
        kw["intervals"] = E._replicate_result_intervals(st)
        kw["aggregate"] = (E._aggregate_result_json_replicates(
            reps, assertions=E._replicate_assertions(st), kinds=E._replicate_metric_kinds(st)) if reps else {})
    except Exception:  # noqa: BLE001
        pass
    return {"name": Path(d).parent.parent.name + "/" + Path(d).name[:32],
            "paper": paper.read_text(encoding="utf-8"), "kw": kw}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rounds", nargs="?", default="C:/fi_runs/suite/rounds")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--max-chance", type=float, default=None)
    a = ap.parse_args()
    if a.cache and Path(a.cache).exists():
        quests = pickle.loads(Path(a.cache).read_bytes())
    else:
        quests = [q for d in sorted(glob.glob(f"{a.rounds}/*/*/out/*")) if (q := load_quest(d))]
        if a.cache:
            Path(a.cache).write_bytes(pickle.dumps(quests))
    if a.max_chance is not None:
        npv.MAX_CHANCE_UNNAMED = a.max_chance
    built: dict[int, tuple] = {}
    orig = npv.build_traceable
    cur = {"i": 0}

    def cached(**kw):
        i = cur["i"]
        if i not in built:
            t = time.time()
            res = orig(**kw)
            built[i] = (*res, time.time() - t)
        return built[i][:2]

    npv.build_traceable = cached
    self_flag = self_n = cross_ok = cross_n = 0
    for pi, p in enumerate(quests):
        for qi, q in enumerate(quests):
            cur["i"] = qi
            r = npv.check(p["paper"], **q["kw"])
            if r.skipped:
                continue
            if pi == qi:
                self_flag += r.untraceable
                self_n += r.paper_numbers
                print(f"SAME   {p['name']:46s} checked={r.paper_numbers:4d} flagged={r.untraceable}")
            else:
                ok = r.paper_numbers - r.untraceable
                cross_ok += ok
                cross_n += r.paper_numbers
                print(f"CROSS  paper={p['name']:44s} results={q['name']:44s} checked={r.paper_numbers:4d} accounted_for={ok}")
    for i, b in built.items():
        print(f"build {quests[i]['name']}: {len(b[0])} values, {b[2]:.1f}s")
    print(f"\nSAME-QUEST: {self_flag} flagged of {self_n} paper numbers")
    print(f"CROSS-QUEST: {cross_ok} accounted for by coincidence of {cross_n} paper numbers")


if __name__ == "__main__":
    main()
