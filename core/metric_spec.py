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
        out.append({**item, "id": ident, "kind": kind})
    return out, None


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
        diffs: list[float] = []
        for va, vb in zip(a["by_seed"], b["by_seed"]):
            if len(va) != len(vb):
                return None, "a paired design needs the same number of trials in both settings in every seed"
            diffs.extend(x - y for x, y in zip(va, vb))
        return _stats.paired_permutation_test(diffs), None
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
        groups: dict[tuple[str, ...], list[tuple[str, ...]]] = {}
        for path in pooled:
            if path:
                groups.setdefault(path[:-1], []).append(path)
        for factor, members in sorted(groups.items()):
            members.sort()
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    result, why = _contrast(pooled[members[i]], pooled[members[j]], spec)
                    tag = f"{'.'.join(factor)}: {members[i][-1]} vs {members[j][-1]}"
                    if result is None:
                        unsupported.append({"id": f"{spec['id']} {tag}", "reason": why or "too few observations"})
                        continue
                    contrasts.append({
                        "metric": spec["id"], "factor": ".".join(factor), "a": members[i][-1], "b": members[j][-1],
                        "family": str(spec.get("family") or DEFAULT_FAMILY), **result,
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
    return gaps
