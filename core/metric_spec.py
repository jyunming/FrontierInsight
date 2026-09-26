"""What each headline number is an estimate of, declared before the run, and the estimator that goes with it.

FI used to guess the statistics from the names in ``RESULT_JSON``: ``<name>_count`` and ``<name>_total`` meant independent
Bernoulli trials, ``<name>_values`` meant independent observations, and every comparison was a Cohen's d over the per-seed
summaries. Those methods need conditions the engine cannot see: whether trials are independent or come in clusters (trials of
one household, steps of one trajectory), whether two settings shared their random numbers and so are paired, whether a value is
a proportion or a mean. A wrong guess gives an interval that looks fine and is too narrow.

The plan's protocol therefore declares ``metrics``. Each is a **MetricSpec**::

    {"id": "outbreak_probability", "estimand": "P(outbreak | R0, N)", "kind": "proportion",
     "unit": "trajectory", "cluster": null, "paired": false, "family": "R0 contrasts"}

``id`` is the name the script uses for the number in ``RESULT_JSON``, at any depth. ``estimand`` and ``unit`` are required text:
what the number actually estimates, and what one independent observation is -- a spec that only names an estimator (``kind``)
without saying what it is an estimator *of* is refused, the same as a protocol missing any other thing a gate needs. ``kind`` is
``proportion`` (a probability from trials that succeed or fail) or ``mean``. ``cluster`` names a list in ``RESULT_JSON`` aligned
with the values that gives each observation's cluster (``true`` means ``<id>_clusters``). ``paired`` says trial *i* of every
setting used the same random numbers, so the settings are compared trial by trial. ``family`` is the set of comparisons a
multiplicity correction covers.

The engine dispatches only to estimators it has, and states which one it used:

============  ==================================  ==============================================================
kind          one setting                         a contrast between two settings
============  ==================================  ==============================================================
proportion    Wilson, pooled counts               two-proportion z test with Newcombe's interval
proportion    cluster bootstrap over clusters     cluster bootstrap of the difference
proportion    (paired)                            paired sign-flip permutation of the per-pair differences
mean          percentile bootstrap, pooled        bootstrap interval with a permutation p-value
mean          cluster bootstrap over clusters     cluster bootstrap of the difference
mean          (paired)                            paired sign-flip permutation of the per-pair differences
============  ==================================  ==============================================================

A paired design over clusters is not supported and is reported as such; so is any design whose data the script did not give (a
cluster design without ``<id>_clusters``). The p-values of each ``family`` are adjusted together (Holm). A metric that has
counts or values in ``RESULT_JSON`` and no spec keeps the old guess and is listed as ``undeclared``: the evidence level does not
call the statistics adequate while there is one.
"""

from __future__ import annotations

import math
from typing import Any

from . import stats as _stats

KINDS = ("proportion", "mean")
DEFAULT_FAMILY = "all"


def normalize(metrics: Any) -> tuple[list[dict[str, Any]] | None, str | None]:
    """Check the shape of a protocol's ``metrics``: ``(clean list, None)`` or ``(None, why)``. Strict, because what is checked
    decides which estimator runs."""
    if isinstance(metrics, dict):
        metrics = [metrics]
    if not isinstance(metrics, list):
        return None, "`protocol.metrics` must be a list of metric specs (each with an `id` and a `kind`)"
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(metrics, start=1):
        if not isinstance(item, dict) or not str(item.get("id") or "").strip():
            return None, f"`protocol.metrics` entry {index} has no `id` (the name the script uses for the number in RESULT_JSON)"
        ident = str(item["id"]).strip()
        if ident in seen:
            return None, f"`protocol.metrics` names `{ident}` twice"
        seen.add(ident)
        kind = str(item.get("kind") or "").strip().lower()
        if kind not in KINDS:
            return None, f"`protocol.metrics` entry `{ident}`: `kind` must be `proportion` or `mean`"
        # `estimand` and `unit` are what make a spec say what the number actually is (a probability of what,
        # over what population) rather than just which estimator function to call -- required, the same as
        # every other gate here (a config missing what protocol_check/oracle_check need is refused, not
        # silently guessed at).
        for key in ("estimand", "unit"):
            value = item.get(key)
            if not isinstance(value, str) or not value.strip():
                what = "what this number estimates" if key == "estimand" else "what one independent observation is"
                return None, f"`protocol.metrics` entry `{ident}` needs `{key}` (text saying {what})"
        if item.get("family") is not None and not isinstance(item["family"], str):
            return None, f"`protocol.metrics` entry `{ident}`: `family` must be text"
        cluster = item.get("cluster")
        if cluster is not None and not isinstance(cluster, (bool, str)):
            return None, f"`protocol.metrics` entry `{ident}`: `cluster` must be true, false, or the name of the list that holds each observation's cluster"
        paired = item.get("paired")
        if paired is not None and not isinstance(paired, bool):
            return None, f"`protocol.metrics` entry `{ident}`: `paired` must be true or false"
        # No `given` (left out, null or empty) is a mean over every trial. A live plan wrote `"given": null` on every
        # metric, the key survived into the spec, and every entry then read as declaring a subset: the list was dropped.
        given = item.get("given")
        if isinstance(given, str):
            given = given.strip() or None
        if given is not None and not isinstance(given, str):
            return None, f"`protocol.metrics` entry `{ident}`: `given` must be the id of a proportion metric"
        entry = {k: v for k, v in item.items() if k != "given"} | {"id": ident, "kind": kind}
        if given is not None:
            entry["given"] = given
        out.append(entry)
    # A mean over a subset of the trials names the proportion whose successes are that subset (see ``given_counts``).
    kinds = {m["id"]: m["kind"] for m in out}
    for m in out:
        if "given" not in m:
            continue
        if m["kind"] != "mean":
            return None, f"`protocol.metrics` entry `{m['id']}`: only a `mean` can be `given` a subset of the trials"
        if kinds.get(m["given"]) != "proportion":
            return None, (
                f"`protocol.metrics` entry `{m['id']}`: `given` must name a declared `proportion` metric, "
                f"not `{m['given']}`"
            )
    return out, None


def repair(metrics: Any) -> tuple[list[dict[str, Any]], list[str]]:
    """``(the specs that pass, a sentence per one left out)``: :func:`normalize` entry by entry, for a draft. A live
    plan declared one metric of a kind FI has no estimator for (`count`) beside valid ones, and the strict check threw
    the whole list away with it."""
    if isinstance(metrics, dict):
        metrics = [metrics]
    if not isinstance(metrics, list):
        return [], [normalize(metrics)[1] or "`protocol.metrics` is not a list"]
    kept: list[dict[str, Any]] = []
    notes: list[str] = []
    # A `given` names another metric, so the metrics it can name are checked first, whatever the order they came in.
    ordered = [m for m in metrics if not (isinstance(m, dict) and m.get("given"))] + [
        m for m in metrics if isinstance(m, dict) and m.get("given")
    ]
    for item in ordered:
        clean, why = normalize(kept + [item])
        if clean is None:
            notes.append(why or "a `protocol.metrics` entry could not be checked")
        else:
            kept = clean
    return kept, notes


def declared(protocol: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The valid metric specs of a protocol (an entry that is not a mapping with an id and a kind is left out)."""
    items = protocol.get("metrics") if isinstance(protocol, dict) else None
    clean, _why = normalize(items) if items is not None else ([], None)
    return clean or []


def _cluster_key(spec: dict[str, Any]) -> str | None:
    cluster = spec.get("cluster")
    if cluster is True:
        return f"{spec['id']}_clusters"
    return cluster if isinstance(cluster, str) and cluster.strip() else None


def _walk(obj: Any, prefix: tuple[str, ...] = ()) -> dict[tuple[str, ...], Any]:
    """Every scalar number and list in a nested mapping, by path (``_seed`` at the top level is skipped)."""
    found: dict[tuple[str, ...], Any] = {}
    if not isinstance(obj, dict):
        return found
    for k, v in obj.items():
        if not prefix and k == "_seed":
            continue
        path = prefix + (str(k),)
        if isinstance(v, dict):
            found.update(_walk(v, path))
        elif isinstance(v, bool):
            continue
        elif isinstance(v, (int, float)) or isinstance(v, list):
            found[path] = v
    return found


def _numbers(value: Any) -> list[float] | None:
    if isinstance(value, list) and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in value):
        return [float(x) for x in value]
    return None


def undeclared(replicates: list[dict[str, Any]], specs: list[dict[str, Any]]) -> list[str]:
    """The metrics whose estimator matters (a proportion reported with counts, or numbers reported as ``<name>_values``) that
    no spec covers."""
    ids = {s["id"] for s in specs}
    names: set[str] = set()
    for replicate in replicates[:1]:
        for path in _walk(replicate):
            leaf = path[-1]
            for suffix in ("_count", "_total", "_values"):
                if leaf.endswith(suffix) and len(leaf) > len(suffix):
                    names.add(leaf[: -len(suffix)])
    return sorted(names - ids)


def _settings(replicates: list[dict[str, Any]], spec: dict[str, Any]) -> dict[tuple[str, ...], dict[str, Any]]:
    """Per setting (the mapping that holds the metric), what every seed reported for it: counts, values, clusters."""
    ident = spec["id"]
    cluster_key = _cluster_key(spec)
    walked = [_walk(r) for r in replicates]
    parents: set[tuple[str, ...]] = set()
    for path in walked[0]:
        if path[-1] in (ident, f"{ident}_count", f"{ident}_total", f"{ident}_values"):
            parents.add(path[:-1])
    out: dict[tuple[str, ...], dict[str, Any]] = {}
    for parent in sorted(parents):
        per_seed = []
        for w in walked:
            per_seed.append({
                "count": w.get(parent + (f"{ident}_count",)), "total": w.get(parent + (f"{ident}_total",)),
                "values": _numbers(w.get(parent + (f"{ident}_values",))),
                "clusters": w.get(parent + (cluster_key,)) if cluster_key else None,
                "pair_id": w.get(parent + (f"{ident}_pair_id",)),
            })
        out[parent] = {"per_seed": per_seed}
    return out


def _pooled(setting: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    """Counts, values and cluster ids of one setting pooled over the seeds; ``problem`` says what the design needs and lacks."""
    rows = setting["per_seed"]
    proportion = spec["kind"] == "proportion"
    out: dict[str, Any] = {"values": None, "clusters": None, "k": None, "n": None, "problem": None}
    if all(isinstance(r["values"], list) for r in rows):
        out["values"] = [x for r in rows for x in r["values"]]
        out["by_seed"] = [r["values"] for r in rows]
    if proportion:
        if all(isinstance(r["count"], (int, float)) and isinstance(r["total"], (int, float)) for r in rows):
            k, n = sum(r["count"] for r in rows), sum(r["total"] for r in rows)
            if float(k).is_integer() and float(n).is_integer() and 0 <= k <= n and n > 0:
                out["k"], out["n"] = int(k), int(n)
        elif out["values"] is not None and all(v in (0.0, 1.0) for v in out["values"]) and out["values"]:
            out["k"], out["n"] = int(sum(out["values"])), len(out["values"])
        if out["n"] is None:
            out["problem"] = f"no `{spec['id']}_count`/`{spec['id']}_total` (or 0/1 `{spec['id']}_values`) in every seed"
    elif out["values"] is None:
        out["problem"] = f"no `{spec['id']}_values` in every seed"
    if _cluster_key(spec):
        if all(isinstance(r["clusters"], list) for r in rows) and out["values"] is not None:
            out["clusters"] = [f"s{i}:{c}" for i, r in enumerate(rows) for c in r["clusters"]]
            if len(out["clusters"]) != len(out["values"]):
                out["clusters"], out["problem"] = None, f"`{_cluster_key(spec)}` is not the same length as `{spec['id']}_values`"
        else:
            out["problem"] = out["problem"] or f"a cluster design needs `{spec['id']}_values` and `{_cluster_key(spec)}` in every seed"
    if spec.get("paired") and out["values"] is None:
        out["problem"] = out["problem"] or f"a paired design needs `{spec['id']}_values` (one value per trial, the same trials in every setting) in every seed"
    if spec.get("paired") and out["values"] is not None:
        # An explicit `<id>_pair_id` per seed lets the contrast JOIN trial i of one setting to trial i of the
        # other by the id the script gave them, instead of trusting that both settings' lists happen to be in
        # the same order (a script that iterates a dict, or reorders for its own reasons, would otherwise
        # silently mismatch pairs and still produce a p-value). Three states, not two: no pair_id anywhere
        # (None -- legitimate, falls back to position); a well-formed pair_id in every seed with no id
        # repeated within a single seed's own list (the list itself, usable for the join); or pair_id given
        # but malformed somewhere -- wrong length, or an id that appears twice in the SAME seed's list (which
        # would let `dict(zip(ids, values))` silently drop a real observation) -- "invalid", a sentinel
        # distinct from None so a partially/incorrectly supplied pair_id is refused, not silently ignored.
        lists = [r["pair_id"] for r in rows]
        if all(x is None for x in lists):
            out["by_seed_pair_id"] = None
        elif all(
            isinstance(x, list) and len(x) == len(r["values"]) and len(set(x)) == len(x)
            for x, r in zip(lists, rows)
        ):
            out["by_seed_pair_id"] = lists
        else:
            out["by_seed_pair_id"] = "invalid"
    return out


def _estimate(pooled: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any] | None:
    if pooled["problem"]:
        return None
    proportion = spec["kind"] == "proportion"
    if _cluster_key(spec):
        ci = _stats.cluster_bootstrap_interval(pooled["values"], pooled["clusters"])
        if ci is None:
            return None
        return {"estimate": ci["mean"], "ci_lower": ci["ci_lower"], "ci_upper": ci["ci_upper"], "n_clusters": ci["n_clusters"],
                "n_observations": ci["n_observations"], "estimator": "cluster_bootstrap"}
    if proportion:
        wilson = _stats.wilson_interval(pooled["k"], pooled["n"])
        if wilson is None:
            return None
        return {"estimate": pooled["k"] / pooled["n"], "ci_lower": wilson[0], "ci_upper": wilson[1], "n_trials": pooled["n"], "estimator": "wilson_pooled_counts"}
    values = pooled["values"]
    ci = _stats.bootstrap_mean_interval(values, cap=20000)
    if ci is None:
        return None
    return {"estimate": sum(values) / len(values), "ci_lower": ci[0], "ci_upper": ci[1], "n_values": len(values), "estimator": "bootstrap_pooled_values"}


def _contrast(a: dict[str, Any], b: dict[str, Any], spec: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """``(result, None)`` or ``(None, why it is not supported)``."""
    for pooled in (a, b):
        if pooled["problem"]:
            return None, pooled["problem"]
    paired, clustered, proportion = bool(spec.get("paired")), bool(_cluster_key(spec)), spec["kind"] == "proportion"
    if paired and clustered:
        return None, "a paired design over clusters is not supported"
    if paired:
        ids_a, ids_b = a.get("by_seed_pair_id"), b.get("by_seed_pair_id")
        if ids_a == "invalid" or ids_b == "invalid":
            return None, "a paired design's `<id>_pair_id` was given but is malformed (wrong length, or repeats an id within one setting's own seed)"
        if (ids_a is None) != (ids_b is None):
            return None, "a paired design's `<id>_pair_id` was given by only one setting; both settings must give it, or neither"
        diffs: list[float] = []
        id_verified = ids_a is not None  # both None or both valid lists, by the checks above
        if id_verified:
            for sa, pa, sb, pb in zip(a["by_seed"], ids_a, b["by_seed"], ids_b):
                if set(pa) != set(pb):
                    return None, "a paired design's `<id>_pair_id` names different trials in the two settings in the same seed"
                by_id_a, by_id_b = dict(zip(pa, sa)), dict(zip(pb, sb))
                diffs.extend(by_id_a[pid] - by_id_b[pid] for pid in pa)
        else:
            for va, vb in zip(a["by_seed"], b["by_seed"]):
                if len(va) != len(vb):
                    return None, "a paired design needs the same number of trials in both settings in every seed"
                diffs.extend(x - y for x, y in zip(va, vb))
        result = _stats.paired_permutation_test(diffs)
        if result is not None:
            result["paired_id_verified"] = id_verified
        return result, None
    if clustered:
        return _stats.cluster_bootstrap_difference(a["values"], a["clusters"], b["values"], b["clusters"]), None
    if proportion:
        return _stats.two_proportion_test(a["k"], a["n"], b["k"], b["n"]), None
    return _stats.two_sample_mean_test(a["values"], b["values"]), None


def statistics(replicates: list[dict[str, Any]], protocol: dict[str, Any] | None) -> dict[str, Any]:
    """What the declared metrics estimate, and every pairwise contrast between the settings of a factor, with the p-values of
    each family adjusted together. ``{}`` when the protocol declares no metric."""
    specs = declared(protocol)
    if not specs or not replicates or not all(isinstance(r, dict) for r in replicates):
        return {}
    estimates: dict[str, dict[str, Any]] = {}
    contrasts: list[dict[str, Any]] = []
    unsupported: list[dict[str, str]] = []
    declared_contrasts = protocol.get("contrasts") if isinstance(protocol, dict) and isinstance(protocol.get("contrasts"), list) else []
    any_prespecified = False
    for spec in specs:
        settings = _settings(replicates, spec)
        if not settings:
            unsupported.append({"id": spec["id"], "reason": f"`{spec['id']}` does not appear in the results"})
            continue
        pooled = {path: _pooled(s, spec) for path, s in settings.items()}
        for path, p in pooled.items():
            label = ".".join(path + (spec["id"],))
            est = _estimate(p, spec)
            if est is None:
                unsupported.append({"id": label, "reason": p["problem"] or "too few observations for an interval"})
            else:
                estimates[label] = {**est, "kind": spec["kind"], "cluster": _cluster_key(spec), "paired": bool(spec.get("paired"))}
        # A protocol may prespecify exactly which pairs to compare (protocol.contrasts:
        # [{"metric": <id>, "a": <setting>, "b": <setting>}, ...]) instead of leaving every
        # pairwise comparison of a factor's settings to be generated after the fact -- the audit's
        # point that an unplanned contrast set inflates the family (and lets a post-hoc "significant"
        # pair be picked from many) rather than testing what was decided before the results were
        # seen. Falls back to every pairwise comparison when the protocol names none for this metric.
        wanted = {
            frozenset((str(c.get("a")), str(c.get("b"))))
            for c in declared_contrasts if isinstance(c, dict) and c.get("metric") == spec["id"]
        }
        prespecified = bool(wanted)
        any_prespecified = any_prespecified or prespecified
        matched_wanted: set[frozenset[str]] = set()
        groups: dict[tuple[str, ...], list[tuple[str, ...]]] = {}
        for path in pooled:
            if path:
                groups.setdefault(path[:-1], []).append(path)
        for factor, members in sorted(groups.items()):
            members.sort()
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    a_label, b_label = members[i][-1], members[j][-1]
                    pair = frozenset((a_label, b_label))
                    if prespecified and pair not in wanted:
                        continue
                    matched_wanted.add(pair)
                    result, why = _contrast(pooled[members[i]], pooled[members[j]], spec)
                    tag = f"{'.'.join(factor)}: {a_label} vs {b_label}"
                    if result is None:
                        unsupported.append({"id": f"{spec['id']} {tag}", "reason": why or "too few observations"})
                        continue
                    contrasts.append({
                        "metric": spec["id"], "factor": ".".join(factor), "a": a_label, "b": b_label,
                        "family": str(spec.get("family") or DEFAULT_FAMILY), "prespecified": prespecified, **result,
                    })
        # A prespecified pair that names no real setting at all (a typo in `a`/`b`) would otherwise silently
        # produce zero contrasts with nothing to say why -- named here instead of failing quietly.
        for pair in wanted - matched_wanted:
            unsupported.append({
                "id": f"{spec['id']} {' vs '.join(sorted(pair))}",
                "reason": "protocol.contrasts names this pair, but no setting in the results matches both sides of it",
            })
    # A protocol.contrasts entry naming a metric id that matches none of the declared specs (a typo, e.g.
    # "outbrek_probability") would otherwise be silently ignored -- every real metric would then fall back to
    # all-pairwise with `contrasts_prespecified: false`, indistinguishable from a protocol that simply chose not
    # to prespecify anything for it.
    known_ids = {s["id"] for s in specs}
    for c in declared_contrasts:
        if isinstance(c, dict) and c.get("metric") is not None and c.get("metric") not in known_ids:
            unsupported.append({
                "id": f"protocol.contrasts: {c.get('metric')!r}",
                "reason": "names a metric id that matches none of the protocol's declared metric specs",
            })
    families: dict[str, list[int]] = {}
    for idx, c in enumerate(contrasts):
        families.setdefault(c["family"], []).append(idx)
    for members in families.values():
        adjusted = _stats.holm_adjust([contrasts[i]["p"] for i in members])
        for i, p_adj in zip(members, adjusted):
            contrasts[i]["p_holm"] = p_adj
            contrasts[i]["significant_after_holm"] = bool(p_adj < 0.05)
            contrasts[i]["comparisons_in_family"] = len(members)
    return {
        "specs": [{k: v for k, v in s.items() if k in ("id", "estimand", "kind", "unit", "cluster", "paired", "family")} for s in specs],
        "estimates": estimates,
        "contrasts": contrasts,
        "unsupported": unsupported,
        "undeclared": undeclared(replicates, specs),
        "contrasts_prespecified": any_prespecified,
    }


def coverage_gaps(protocol: dict[str, Any] | None, replicates: list[dict[str, Any]], computed: dict[str, Any] | None) -> list[str]:
    """What keeps the statistics from being called adequate, one sentence each (for the evidence level)."""
    gaps: list[str] = []
    if not declared(protocol):
        if replicates and undeclared(replicates, []):
            gaps.append(
                "the protocol declares no metric specs, so the estimators (independence, pairing, clustering, proportion or mean) were "
                "guessed from the names in RESULT_JSON"
            )
        return gaps
    computed = computed or {}
    if computed.get("undeclared"):
        gaps.append("no metric spec covers " + ", ".join(computed["undeclared"][:5]) + ": its estimator was guessed from the names in RESULT_JSON")
    if computed.get("unsupported"):
        first = computed["unsupported"][0]
        gaps.append(f"{len(computed['unsupported'])} estimate(s) or contrast(s) could not be made (first: {first['id']}: {first['reason']})")
    by_position = [c for c in computed.get("contrasts") or [] if c.get("paired_id_verified") is False]
    if by_position:
        gaps.append(
            f"{len(by_position)} paired contrast(s) joined the two settings' trials by their position in the lists, not by "
            "a pair id (`<id>_pair_id`): a trial missing or out of order pairs the wrong trials"
        )
    if computed.get("contrasts") and not computed.get("contrasts_prespecified"):
        gaps.append(
            "the protocol names no prespecified contrasts, so every pairwise comparison of each factor's settings was "
            "computed after the fact (protocol.contrasts can fix the comparisons before the results are seen)"
        )
    return gaps


def missing_pair_ids(protocol: dict[str, Any] | None, result_json: Any) -> list[str]:
    """The paired metrics whose per-trial values the results print with no ``<id>_pair_id`` beside them: one sentence
    each. Without an id a paired comparison can only join trials by position (``rigor_profile: research`` stops for
    this; otherwise it is a gap in the evidence)."""
    paired = [spec["id"] for spec in declared(protocol) if spec.get("paired")]
    lacking: set[str] = set()

    def walk(node: Any) -> None:
        # Looked for in each mapping on its own: one setting's ids do not stand for another setting's missing ones. A
        # single value (a setting run once) has nothing to pair.
        if isinstance(node, dict):
            for mid in paired:
                values = node.get(f"{mid}_values")
                if isinstance(values, list) and len(values) > 1 and f"{mid}_pair_id" not in node:
                    lacking.add(mid)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node[:200]:
                walk(value)

    walk(result_json)
    out = []
    for mid in paired:
        if mid in lacking:
            out.append(
                f"the paired metric `{mid}` prints `{mid}_values` with no `{mid}_pair_id` beside them: print, beside "
                f"each `{mid}_values`, the trial each value came from as `{mid}_pair_id` (FI_TRIALS gives it: "
                f"`metrics[...][\"trials\"]`), so the two settings' trials are joined by id, not by position"
            )
    return out
