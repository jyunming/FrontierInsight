"""``python -m dev.evaluation.acceptance --tree TREE --cases CASES.yaml --out DIR``: see runner.py."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import runner


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dev.evaluation.acceptance", description=__doc__)
    parser.add_argument("--tree", required=True, type=Path, help="the FI checkout to score")
    parser.add_argument("--cases", required=True, type=Path, help="the cases file (kept outside the repository)")
    parser.add_argument("--out", required=True, type=Path, help="where score.json, score.md and each case's output go")
    parser.add_argument("--only", nargs="*", default=None, help="case ids or groups to run (default: all)")
    parser.add_argument("--jobs", type=int, default=2, help="cases run at the same time (default 2)")
    parser.add_argument("--against", type=Path, default=None, help="an older score.json to list regressions against")
    args = parser.parse_args(argv)
    report = runner.score(args.tree.resolve(), args.cases.resolve(), args.out.resolve(),
                          only=set(args.only) if args.only else None, jobs=args.jobs, against=args.against)
    print(runner.markdown(report).split("\n| case")[0].strip())
    print(f"written: {args.out / 'score.md'}")
    return 1 if report.get("against", {}).get("regressions") else 0


if __name__ == "__main__":
    sys.exit(main())
