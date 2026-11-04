# Acceptance bench

One command scores an FI checkout on a fixed list of recorded cases, so a change is judged against a baseline score
instead of by reading it.

```bash
python -m dev.evaluation.acceptance --tree <checkout> --cases <cases.yaml> --out <dir> [--against <old>/score.json]
```

**Cases file.** Kept outside the repository, because its cases are real quests. Each case is one recorded input with a
known right outcome:

```yaml
root: <folder the quest paths are relative to>
probes: <folder the probe paths are relative to>
cases:
  - {id: a-false-stop, group: A, quest: <quest folder>, expect: go_on, probe: replay_gate.py}
  - {id: a-planted-fault, group: C, quest: <quest folder>, expect: catch, probe: repair_accept.py, args: [<commit>]}
  - {id: not-yet, group: E, quest: <quest folder>, expect: go_on, note: "no probe written yet"}
```

`expect` is the right outcome: `go_on` (must not stop), `stop` (must still stop), `catch` (a fault that must be found),
`clean` (must not be flagged), or any other word the probe prints (for example `keep`, `fill`).

**Probe.** A script run as `python <probe> <checkout> <quest folder> [args...]` in a scratch folder of its own. It
replays the case on the checkout and prints, as its last line, `ACCEPT_JSON: {"got": "<outcome>", "detail": "..."}`.
It must not write into the quest folder. A probe is tied to today's code; when a change removes what a probe calls, the
probe is rewritten and the case and its `expect` stay.

**Score.** `score.md` and `score.json` hold each case's expected and measured outcome and the totals: false stops,
missed stops, faults caught of all faults, false flags, other failures, not measured (the probe failed or printed no
outcome; never counted as a pass) and cases with no probe yet, plus the line count of `core/`. With `--against`, the
cases that passed in the older score and do not now are listed, and the command exits 1 when there is any.
