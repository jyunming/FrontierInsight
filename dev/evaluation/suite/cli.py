"""``python -m dev.evaluation.suite``: run a fixed set of topics on an FI change and compare rounds.

    run            start a group's topics as independent processes (at most N at a time)
    status / wait  follow a round
    collect        read every quest of a round into round.json / round.md
    grade-template write the grader's prompt for each delivered paper of a round
    grade-import   read a grader's report (its table row, a STOP verdict) into the round's grades.json
    report         rewrite trend.md and pareto.md from every collected round
    adopt          bring existing runs in as rounds (a seed file)
    make-configs   write a group's topic configs without starting anything (to look at them)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import adopt as _adopt
from . import collect as _collect
from . import grade as _grade
from . import runner as _runner
from . import trend as _trend
from .suite import Suite, SuiteError


def _suite(args: argparse.Namespace) -> Suite:
    return Suite(args.suite)


def _round_dir(args: argparse.Namespace, suite: Suite) -> Path:
    if getattr(args, "round_dir", None):
        return Path(args.round_dir)
    if not getattr(args, "round", None):
        raise SuiteError("give --round <name> (or --round-dir)")
    return _runner.round_dir_of(suite, args.round)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m dev.evaluation.suite", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp: argparse.ArgumentParser, *, rnd: bool = True) -> None:
        sp.add_argument("--suite", required=True, help="the suite.yaml (kept outside the repository)")
        if rnd:
            sp.add_argument("--round", help="the round's name")
            sp.add_argument("--round-dir", help="the round's folder (instead of --round)")

    r = sub.add_parser("run", help="start a group's topics")
    common(r, rnd=False)
    r.add_argument("--group", required=True)
    r.add_argument("--round", required=True)
    r.add_argument("--fi", required=True, help="a frozen FI worktree (its launch.py is run, with it as working directory)")
    r.add_argument("--max-parallel", type=int, default=3)
    r.add_argument("--acceptance", action="store_true", help="allow a group held back for the acceptance run")
    r.add_argument("--python", help="the Python that runs launch.py (default: this one)")
    r.add_argument("--foreground", action="store_true", help="drive the round in this process instead of a detached one")
    r.add_argument("--interval", type=float, default=20.0)

    d = sub.add_parser("drive", help="(internal) keep starting a prepared round's topics until all have ended")
    common(d)
    d.add_argument("--interval", type=float, default=20.0)

    for name in ("status",):
        s = sub.add_parser(name, help="show a round's topics")
        common(s)
    w = sub.add_parser("wait", help="block until every topic of a round has ended")
    common(w)
    w.add_argument("--interval", type=float, default=30.0)
    w.add_argument("--timeout", type=float, default=None, help="seconds; exit code 2 when it passes")

    c = sub.add_parser("collect", help="build round.json and round.md")
    common(c)
    c.add_argument("--no-report", action="store_true", help="do not rewrite trend.md / pareto.md")

    g = sub.add_parser("grade-template", help="write GRADER_PROMPT.md for each delivered paper")
    common(g)

    gi = sub.add_parser("grade-import", help="read a grader's report into grades.json")
    common(gi)
    gi.add_argument("--topic", required=True)
    gi.add_argument("--report", required=True)
    gi.add_argument("--stop", choices=["correct", "wrong"], help="the grader's verdict on an honest stop")
    gi.add_argument("--class", dest="cause_class", help="a taxonomy class set by the grader (the only way to record fi_bug)")
    gi.add_argument("--stage")
    gi.add_argument("--note")

    rp = sub.add_parser("report", help="rewrite trend.md and pareto.md")
    common(rp, rnd=False)

    a = sub.add_parser("adopt", help="bring existing runs in as rounds")
    common(a, rnd=False)
    a.add_argument("--from", dest="seed", required=True, help="the seed file")
    a.add_argument("--collect", action="store_true", help="collect each adopted round and rewrite the reports")

    m = sub.add_parser("make-configs", help="write a group's configs for a round (nothing is started)")
    common(m, rnd=False)
    m.add_argument("--group", required=True)
    m.add_argument("--round", required=True)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _dispatch(args)
    except SuiteError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _dispatch(args: argparse.Namespace) -> int:
    suite = _suite(args)
    if args.cmd == "run":
        rdir = _runner.prepare_round(suite, args.group, args.round, Path(args.fi), max_parallel=args.max_parallel,
                                     acceptance=args.acceptance, python=args.python)
        print(f"round {args.round}: {len(suite.group_topics(args.group))} topic(s) prepared in {rdir}")
        if args.foreground:
            _runner.drive(suite, rdir, interval=args.interval)
            print(_runner.render_status(_runner.status(rdir)))
        else:
            h = _runner.start_detached_driver(suite.path, rdir, interval=args.interval)
            print(f"driver pid {h.pid}; follow with: python -m dev.evaluation.suite status --suite {suite.path} --round {args.round}")
        return 0
    if args.cmd == "make-configs":
        for t, p in suite.write_configs(args.group, _runner.round_dir_of(suite, args.round)).items():
            print(f"{t}: {p}")
        return 0
    if args.cmd == "report":
        for p in _trend.write_reports(suite.rounds_dir):
            print(p)
        return 0
    if args.cmd == "adopt":
        dirs = _adopt.adopt_file(suite, Path(args.seed))
        for rdir in dirs:
            print(f"adopted {rdir.name}")
            if args.collect:
                _collect.collect_round(rdir)
        if args.collect:
            for p in _trend.write_reports(suite.rounds_dir):
                print(p)
        return 0
    rdir = _round_dir(args, suite)
    if args.cmd == "drive":
        _runner.drive(suite, rdir, interval=args.interval)
        return 0
    if args.cmd == "status":
        print(_runner.render_status(_runner.status(rdir)))
        return 0
    if args.cmd == "wait":
        done, m = _runner.wait(rdir, interval=args.interval, timeout=args.timeout)
        print(_runner.render_status(m))
        return 0 if done else 2
    if args.cmd == "collect":
        data = _collect.collect_round(rdir)
        print(f"{rdir / 'round.md'}  ({data['metrics']['papers']} paper(s), {data['metrics']['honest_stops']} honest stop(s), "
              f"{data['metrics']['pauses_needing_a_person']} pause(s), {data['metrics']['crashes']} crash(es))")
        if not args.no_report:
            for p in _trend.write_reports(suite.rounds_dir):
                print(p)
        return 0
    if args.cmd == "grade-template":
        texts = {t: suite.topic_text(t) for t in suite.topic_files}
        paths = _grade.write_templates(rdir, texts)
        print("\n".join(str(p) for p in paths) or "no delivered paper in this round")
        return 0
    if args.cmd == "grade-import":
        stop = None if args.stop is None else args.stop == "correct"
        entry = _grade.import_grade(rdir, args.topic, Path(args.report), stop_correct=stop, cause_class=args.cause_class,
                                    stage=args.stage, note=args.note)
        print(entry)
        return 0
    return 1
